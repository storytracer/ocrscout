"""ViewerStore: in-memory data layer for the Gradio inspector.

Loads ``results.parquet`` once at startup, derives a per-page disagreement
score, and resolves markdown / DoclingDocument items / source images on
demand. Polars is used for the initial load (fast, easy filtering); plain
Python dicts/lists carry the rows around afterwards.
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
from PIL import Image

from ocrscout.viewer.diff import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBoxItem:
    """One layout box with its label and the markdown anchor index."""

    bbox: tuple[float, float, float, float]  # (left, top, right, bottom), pixel coords
    label: str
    text: str
    item_idx: int


@dataclass(frozen=True)
class TextItem:
    """One labeled text item from a DoclingDocument, in document order."""

    label: str
    text: str
    item_idx: int


@dataclass
class ModelRow:
    """One (page, model) record materialized for the viewer."""

    page_id: str
    model: str
    output_format: str
    document_json: str | None
    error: str | None
    metrics: dict[str, Any]
    markdown: str = ""
    bboxes: list[BBoxItem] = field(default_factory=list)
    items: list[TextItem] = field(default_factory=list)


@dataclass
class PageRow:
    """One page summary, aggregated across all models that touched it."""

    page_id: str
    source_uri: str | None
    models: list[str]
    error_models: list[str]
    disagreement: float  # 0..1, mean pairwise word-diff distance (lower = agreement)
    char_count: int  # max chars across the present models, for sorting by output volume


class ViewerStore:
    """Holds the loaded run and serves rows to the Gradio app."""

    # Color palette for the AnnotatedImage overlay, keyed by DocItemLabel value.
    LABEL_COLORS: ClassVar[dict[str, str]] = {
        "title": "#7c4dff",
        "section_header": "#3d5afe",
        "paragraph": "#26a69a",
        "text": "#26a69a",
        "list_item": "#00897b",
        "table": "#d81b60",
        "picture": "#fb8c00",
        "chart": "#fb8c00",
        "caption": "#8e24aa",
        "footnote": "#9e9e9e",
        "page_header": "#9e9e9e",
        "page_footer": "#9e9e9e",
        "formula": "#5e35b1",
        "code": "#1e88e5",
        "reference": "#558b2f",
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.parquet_path = output_dir / "results.parquet"
        self.text_dir = output_dir / "text"
        if not self.parquet_path.is_file():
            raise FileNotFoundError(
                f"No results.parquet at {self.parquet_path}. Pass an output dir "
                "produced by `ocrscout run`."
            )
        self._rows: list[dict[str, Any]] = self._load_rows(self.parquet_path)
        self._by_page_model: dict[tuple[str, str], dict[str, Any]] = {
            (r["page_id"], r["model"]): r for r in self._rows
        }
        self.all_models: list[str] = sorted({r["model"] for r in self._rows})
        self._pages: list[PageRow] = self._build_page_index()
        log.info(
            "viewer: loaded %d rows across %d pages × %d models from %s",
            len(self._rows), len(self._pages), len(self.all_models), self.parquet_path,
        )

    # ------------------------------------------------------------------ public

    def pages(self, *, sort: str = "page_id") -> list[PageRow]:
        """Return all pages.

        Default ordering is alphabetical by ``page_id``, which is predictable
        and stable across runs. Pass ``sort="parquet"`` to keep the order in
        which rows appeared in ``results.parquet`` (whatever the source
        adapter emitted).
        """
        if sort == "parquet":
            return list(self._pages)
        pages = list(self._pages)
        if sort == "page_id":
            pages.sort(key=lambda p: p.page_id)
        elif sort == "chars":
            pages.sort(key=lambda p: -p.char_count)
        else:
            pages.sort(key=lambda p: (-p.disagreement, p.page_id))
        return pages

    def page_ids(self, *, sort: str = "page_id") -> list[str]:
        return [p.page_id for p in self.pages(sort=sort)]

    def models_for(self, page_id: str) -> list[str]:
        """All models that produced a row for this page (alphabetical)."""
        return sorted(
            m for (pid, m) in self._by_page_model if pid == page_id
        )

    def layout_models_for(self, page_id: str) -> list[str]:
        """Models that produced layout bboxes for this page (alphabetical).

        Used by the viewer's Layout source dropdown so the user can only
        pick a model whose output has something to overlay on the image.
        Models that emit plain markdown (no provenance bboxes) are filtered
        out — picking one of them would just show an empty overlay.
        """
        out: list[str] = []
        for model in self.models_for(page_id):
            row = self._by_page_model.get((page_id, model))
            if row is None:
                continue
            if self._bboxes_for(row):
                out.append(model)
        return out

    def get(self, page_id: str, model: str) -> ModelRow | None:
        raw = self._by_page_model.get((page_id, model))
        if raw is None:
            return None
        markdown = self._markdown_for(raw)
        bboxes = self._bboxes_for(raw)
        items = self._text_items_for(raw)
        return ModelRow(
            page_id=raw["page_id"],
            model=raw["model"],
            output_format=raw["output_format"],
            document_json=raw["document_json"],
            error=raw["error"],
            metrics=raw["metrics"],
            markdown=markdown,
            bboxes=bboxes,
            items=items,
        )

    def image_for(self, page_id: str) -> Image.Image | None:
        """Return the PIL image at ``source_uri`` for ``page_id``, or None.

        Cached via an LRU on the resolved path so the same image isn't reread
        when toggling models. Returns ``None`` on missing/unreadable sources
        — the viewer will render a placeholder.
        """
        # Find any row for this page; they all share the same source_uri.
        raw = next(
            (r for r in self._rows if r["page_id"] == page_id and r.get("source_uri")),
            None,
        )
        if raw is None or not raw.get("source_uri"):
            return None
        return _load_image_cached(str(raw["source_uri"]))

    def annotated_for(
        self, page_id: str, model: str
    ) -> tuple[Image.Image | None, list[tuple[tuple[int, int, int, int], str]]]:
        """Return the source image plus per-item ``((x1,y1,x2,y2), label)`` tuples.

        Coords come from the model's ``ProvenanceItem.bbox`` in pixel space.
        Empty list if the model has no layout data.
        """
        img = self.image_for(page_id)
        row = self.get(page_id, model)
        if row is None or img is None:
            return img, []
        annotations: list[tuple[tuple[int, int, int, int], str]] = []
        for item in row.bboxes:
            left, top, right, bottom = item.bbox
            annotations.append(
                (
                    (
                        int(round(left)),
                        int(round(top)),
                        int(round(right)),
                        int(round(bottom)),
                    ),
                    item.label,
                )
            )
        return img, annotations

    # ------------------------------------------------------------------ internal

    def _load_rows(self, parquet_path: Path) -> list[dict[str, Any]]:
        df = pl.read_parquet(parquet_path)
        out: list[dict[str, Any]] = []
        for raw in df.iter_rows(named=True):
            metrics_raw = raw.get("metrics_json") or ""
            try:
                metrics = json.loads(metrics_raw) if metrics_raw else {}
            except json.JSONDecodeError:
                metrics = {}
            out.append(
                {
                    "page_id": raw["page_id"],
                    "source_uri": raw.get("source_uri"),
                    "output_format": raw.get("output_format"),
                    "document_json": raw.get("document_json"),
                    "error": raw.get("error"),
                    "model": metrics.get("model", "?"),
                    "metrics": metrics,
                }
            )
        return out

    def _build_page_index(self) -> list[PageRow]:
        by_page: dict[str, list[dict[str, Any]]] = {}
        for r in self._rows:
            by_page.setdefault(r["page_id"], []).append(r)
        pages: list[PageRow] = []
        for page_id, rows in by_page.items():
            rows.sort(key=lambda r: r["model"])
            present = [r["model"] for r in rows]
            errors = [r["model"] for r in rows if r["error"]]
            chars = max(
                (int(r["metrics"].get("text_length") or 0) for r in rows),
                default=0,
            )
            disagreement = self._page_disagreement(rows)
            pages.append(
                PageRow(
                    page_id=page_id,
                    source_uri=next(
                        (r["source_uri"] for r in rows if r.get("source_uri")), None
                    ),
                    models=present,
                    error_models=errors,
                    disagreement=disagreement,
                    char_count=chars,
                )
            )
        return pages

    def _page_disagreement(self, rows: list[dict[str, Any]]) -> float:
        """Mean pairwise word-diff distance (1 - SequenceMatcher.ratio).

        0.0 = all models agreed exactly; 1.0 = no shared tokens. Cheap to
        compute even for 5+ models since each ratio is O(n+m) in tokens.
        """
        token_streams = []
        for r in rows:
            md = self._markdown_for(r)
            if md:
                token_streams.append(tokenize(md))
        if len(token_streams) < 2:
            # Single-model pages have no disagreement to measure; sort them last
            # by giving them a 0 score (they fall under the contested ones).
            return 0.0
        total = 0.0
        n = 0
        for i in range(len(token_streams)):
            for j in range(i + 1, len(token_streams)):
                ratio = difflib.SequenceMatcher(
                    None, token_streams[i], token_streams[j], autojunk=False
                ).quick_ratio()
                total += 1.0 - ratio
                n += 1
        return total / n if n else 0.0

    def _markdown_for(self, row: dict[str, Any]) -> str:
        """Prefer the on-disk markdown sidecar; fall back to re-rendering JSON."""
        stem = Path(row["page_id"]).stem.replace("/", "_").replace("\\", "_")
        sidecar = self.text_dir / f"{stem}.{row['model']}.md"
        if sidecar.is_file():
            try:
                return sidecar.read_text(encoding="utf-8")
            except OSError:
                pass
        doc_json = row["document_json"]
        if not doc_json:
            return ""
        try:
            from docling_core.types.doc import DoclingDocument

            doc = DoclingDocument.model_validate_json(doc_json)
            return doc.export_to_markdown()
        except Exception:  # noqa: BLE001
            return ""

    def _iter_doc_items(self, row: dict[str, Any]):
        """Yield ``(item, item_idx, label_str, text_str)`` in body reading order.

        Uses ``DoclingDocument.iterate_items()`` so texts, pictures, and
        tables come back interleaved the way the model laid them out — not
        grouped by collection (which would put every picture at the end of
        the page). Empty pictures/tables get a stub text so they render as
        opaque markers in the colored text pane.
        """
        doc_json = row["document_json"]
        if not doc_json:
            return
        try:
            from docling_core.types.doc import DoclingDocument

            doc = DoclingDocument.model_validate_json(doc_json)
        except Exception:  # noqa: BLE001
            return
        for idx, (item, _level) in enumerate(doc.iterate_items()):
            label = getattr(item, "label", None)
            label_value = getattr(label, "value", None)
            label_str = label_value or (str(label) if label is not None else "item")
            text = (getattr(item, "text", "") or "").strip()
            cls = type(item).__name__
            if not text and cls == "TableItem":
                # TableItems carry data in `.data.table_cells`, not `.text`.
                # docling-core renders the cell grid as a GFM pipe-table.
                try:
                    text = item.export_to_markdown(doc).strip() or "[table]"
                except Exception:  # noqa: BLE001
                    text = "[table]"
            elif not text and cls == "PictureItem":
                text = "[picture]"
            elif not text:
                # Backends sometimes emit a region with a bbox but no
                # recognised text (truncation, mid-block cut-off, OCR
                # failure). Show a placeholder so the user can see that
                # a region was detected without content — mirrors
                # ``[picture]``/``[table]`` and keeps the section pane in
                # one-to-one correspondence with the bbox overlay.
                text = "[empty]"
            yield item, idx, label_str, text

    def _text_items_for(self, row: dict[str, Any]) -> list[TextItem]:
        """Labeled text items in document reading order (used by text panes)."""
        return [
            TextItem(label=label, text=text, item_idx=idx)
            for _item, idx, label, text in self._iter_doc_items(row)
        ]

    def _bboxes_for(self, row: dict[str, Any]) -> list[BBoxItem]:
        """Pull ``ProvenanceItem.bbox`` off every text/picture/table item."""
        out: list[BBoxItem] = []
        for item, idx, label, text in self._iter_doc_items(row):
            provs = getattr(item, "prov", None) or []
            if not provs:
                continue
            bbox = provs[0].bbox
            if bbox is None:
                continue
            # docling BoundingBox carries .l .t .r .b in its native origin.
            # We don't try to re-orient: the layout_json normalizer stores
            # TOPLEFT, which is what AnnotatedImage expects.
            tup = (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
            out.append(BBoxItem(bbox=tup, label=label, text=text, item_idx=idx))
        return out


@lru_cache(maxsize=32)
def _load_image_cached(path: str) -> Image.Image | None:
    try:
        img = _open_fsspec(path) if "://" in path else Image.open(path)
        img.load()
        return img
    except (OSError, FileNotFoundError) as e:
        log.warning("viewer: cannot open source image %s: %s", path, e)
        return None


def _open_fsspec(url: str) -> Image.Image:
    """Open a fsspec URL (``s3://``, ``gs://``, ``https://``, ``hf://``).

    Anonymous read is the default for ``s3://`` so anonymous-access buckets
    (BHL, Common Crawl, etc.) work without credentials. If the user has
    creds configured for the same bucket, fsspec's normal config chain
    still picks them up — we only set ``anon`` when no kwargs are given.
    """
    import io

    import fsspec

    storage_options: dict[str, Any] = {}
    if url.startswith("s3://"):
        storage_options["anon"] = True
    with fsspec.open(url, "rb", **storage_options) as f:
        data = f.read()
    return Image.open(io.BytesIO(data))

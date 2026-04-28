"""ViewerStore: in-memory data layer for the Gradio inspector.

Loads ``data/train-*.parquet`` once at startup via ``datasets.load_dataset``,
derives a per-page disagreement score, and resolves markdown / DoclingDocument
items / source images on demand. Plain Python dicts/lists carry the rows
around afterwards.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from ocrscout.exports.layout import (
    find_parquet_files,
    find_volumes_files,
    parquet_data_files,
    volumes_data_files,
)
from ocrscout.interfaces.comparison import ComparisonResult
from ocrscout.types import ReferenceProvenance, Volume

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
    """One labeled text item from a DoclingDocument, in document order.

    ``html`` is set for items that have a structured HTML rendering
    (currently just ``TableItem``) — section panes embed it raw so browsers
    render real ``<table>`` elements; everything else flows through the
    plain-text escape path.
    """

    label: str
    text: str
    item_idx: int
    html: str | None = None


@dataclass
class ModelRow:
    """One (page, model) record materialized for the viewer."""

    file_id: str
    page_id: str
    model: str
    output_format: str
    document_json: str | None
    error: str | None
    metrics: dict[str, Any]
    markdown: str = ""
    text: str = ""
    bboxes: list[BBoxItem] = field(default_factory=list)
    items: list[TextItem] = field(default_factory=list)
    # Per-comparison-name typed result loaded from the parquet's
    # ``comparisons_json`` envelope. Keys are comparison names; values are
    # ``ComparisonResult`` subclass instances.
    comparisons: dict[str, ComparisonResult] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineRow:
    """A page-level reference artifact materialized for the viewer.

    Distinct from ``ModelRow`` because a baseline is per-page (one
    reference per page), not per-(page, model). Carries provenance so the
    Compare panel can label it correctly ("bhl_ocr (ocr, bhl-legacy)") and
    so consumers can interpret comparison numbers as agreement-vs-OCR
    rather than accuracy-vs-truth.
    """

    file_id: str
    label: str  # "reference" by default; multiple baselines could differentiate later
    text: str
    provenance: ReferenceProvenance | None = None


@dataclass
class PageRow:
    """One page summary, aggregated across all models that touched it.

    ``file_id`` is the user-facing canonical identifier (``volume_id/filename``
    for volume sources, ``parent_dir/filename`` for flat sources). ``page_id``
    is the source-side raw id (BHL's PageID, etc.) — kept around for
    debugging and source-side lookups but not surfaced in the viewer UI.
    """

    file_id: str
    page_id: str
    barcode: str | None
    sequence: int | None
    source_uri: str | None
    models: list[str]
    error_models: list[str]
    has_reference: bool
    disagreement: float  # 0..1, mean pairwise word-diff distance (lower = agreement)
    char_count: int  # max chars across the present models, for sorting by output volume
    comparison_summary: dict[str, float] = field(default_factory=dict)


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
        shards = find_parquet_files(output_dir)
        if not shards:
            raise FileNotFoundError(
                f"No data/train-*.parquet under {output_dir}. Pass an output "
                "dir produced by `ocrscout run` (or a `snapshot_download`'d "
                "ocrscout dataset repo)."
            )
        self._rows: list[dict[str, Any]] = self._load_rows(output_dir)
        self._by_file_model: dict[tuple[str, str], dict[str, Any]] = {
            (r["file_id"], r["model"]): r for r in self._rows
        }
        self.all_models: list[str] = sorted({r["model"] for r in self._rows})
        self._volumes: dict[str, Volume] = self._load_volumes(output_dir)
        self._pages: list[PageRow] = self._build_page_index()
        log.info(
            "viewer: loaded %d rows across %d pages × %d models from %s%s",
            len(self._rows), len(self._pages), len(self.all_models),
            ", ".join(str(p.relative_to(output_dir)) for p in shards),
            f" (+{len(self._volumes)} volume(s))" if self._volumes else "",
        )

    # ------------------------------------------------------------------ public

    def pages(self, *, sort: str = "file_id") -> list[PageRow]:
        """Return all pages.

        Default ordering is alphabetical by ``file_id``, which is predictable
        and stable across runs. Pass ``sort="parquet"`` to keep the order in
        which rows appeared in the parquet shards (whatever the source
        adapter emitted), ``"chars"`` to sort by output volume,
        ``"errors"`` to surface failed pages first, or any other value for
        the disagreement-desc default.
        """
        if sort == "parquet":
            return list(self._pages)
        pages = list(self._pages)
        if sort == "file_id":
            pages.sort(key=lambda p: p.file_id)
        elif sort == "chars":
            pages.sort(key=lambda p: -p.char_count)
        elif sort == "errors":
            pages.sort(key=lambda p: (-len(p.error_models), p.file_id))
        else:
            pages.sort(key=lambda p: (-p.disagreement, p.file_id))
        return pages

    def file_ids(self, *, sort: str = "file_id") -> list[str]:
        return [p.file_id for p in self.pages(sort=sort)]

    # Back-compat alias — older app.py code path.
    def page_ids(self, *, sort: str = "file_id") -> list[str]:
        return self.file_ids(sort=sort)

    def models_for(self, file_id: str) -> list[str]:
        """All models that produced a row for this page (alphabetical)."""
        return sorted(
            m for (fid, m) in self._by_file_model if fid == file_id
        )

    def layout_models_for(self, file_id: str) -> list[str]:
        """Models that produced layout bboxes for this page (alphabetical).

        Used by the viewer's Layout source dropdown so the user can only
        pick a model whose output has something to overlay on the image.
        Models that emit plain markdown (no provenance bboxes) are filtered
        out — picking one of them would just show an empty overlay.
        """
        out: list[str] = []
        for model in self.models_for(file_id):
            row = self._by_file_model.get((file_id, model))
            if row is None:
                continue
            if self._bboxes_for(row):
                out.append(model)
        return out

    def get(self, file_id: str, model: str) -> ModelRow | None:
        raw = self._by_file_model.get((file_id, model))
        if raw is None:
            return None
        markdown = self._markdown_for(raw)
        bboxes = self._bboxes_for(raw)
        items = self._text_items_for(raw)
        return ModelRow(
            file_id=raw["file_id"],
            page_id=raw["page_id"],
            model=raw["model"],
            output_format=raw["output_format"],
            document_json=raw["document_json"],
            error=raw["error"],
            metrics=raw["metrics"],
            markdown=markdown,
            text=raw.get("text") or "",
            bboxes=bboxes,
            items=items,
            comparisons=self._comparisons_for(raw),
        )

    def baselines_for(self, file_id: str) -> list[BaselineRow]:
        """Return all reference-style baselines registered for the page.

        Today the run loop only ever wires up a single reference adapter,
        so this returns at most one row per page. The plural shape is
        forward-looking: future runs may attach multiple baselines (an
        old-OCR layer plus a human transcription, for instance) and the
        Compare panel will show all of them.
        """
        for raw in self._rows:
            if raw["file_id"] != file_id:
                continue
            text = raw.get("reference_text")
            if not text:
                continue
            provenance = _parse_provenance(raw.get("reference_provenance_json"))
            return [
                BaselineRow(
                    file_id=file_id,
                    label="reference",
                    text=text,
                    provenance=provenance,
                )
            ]
        return []

    def has_any_baseline(self) -> bool:
        return any(r.get("reference_text") for r in self._rows)

    def volume_for(self, file_id: str) -> Volume | None:
        """Return the ``Volume`` (loaded from ``volumes-*.parquet``) backing
        this page, or ``None`` if the source has no volume concept or the
        sidecar wasn't written.
        """
        raw = next((r for r in self._rows if r["file_id"] == file_id), None)
        if raw is None:
            return None
        vid = raw.get("barcode")
        if not vid:
            return None
        return self._volumes.get(str(vid))

    def image_for(self, file_id: str) -> Image.Image | str | None:
        """Return the source image for ``file_id`` — PIL image, URL, or None.

        Resolution order:

        1. The ``image`` column (bytes embedded in the parquet by the publisher
           when ``--bundle-images`` was set). Decoded once per call; the LRU
           cache guards repeat calls.
        2. For BHL JP2 ``source_uri`` values, return the pre-converted WebP
           HTTPS URL directly — Gradio fetches it server-side, skipping the
           multi-MB JP2 download and the slow OpenJPEG decode. BHL's "full"
           WebP is published at the same dimensions as the JP2, so bbox
           overlays align without any scaling.
        3. ``source_uri`` — local path resolved relative to ``self.output_dir``
           if not absolute, fsspec URL otherwise. Loaded as PIL via the LRU
           cache.

        Returns ``None`` for pages with neither.
        """
        from ocrscout.sources.bhl import bhl_web_image_url

        raw = next((r for r in self._rows if r["file_id"] == file_id), None)
        if raw is None:
            return None
        img_bytes = raw.get("image_bytes")
        if img_bytes:
            return _decode_image_bytes(img_bytes)
        src = raw.get("source_uri")
        if not src:
            return None
        src = str(src)
        web_src = bhl_web_image_url(src)
        if web_src is not None:
            return web_src
        if "://" not in src:
            p = Path(src)
            if not p.is_absolute():
                p = self.output_dir / p
            src = str(p)
        return _load_image_cached(src)

    def annotated_for(
        self, file_id: str, model: str
    ) -> tuple[
        Image.Image | str | None,
        list[tuple[tuple[int, int, int, int], str]],
    ]:
        """Return the source image plus per-item ``((x1,y1,x2,y2), label)`` tuples.

        Image may be a PIL.Image, a URL string (handed to Gradio for direct
        fetch), or ``None``. Coords come from the model's
        ``ProvenanceItem.bbox`` in pixel space. Empty list if the model has
        no layout data.
        """
        img = self.image_for(file_id)
        row = self.get(file_id, model)
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

    def _load_rows(self, output_dir: Path) -> list[dict[str, Any]]:
        from datasets import Image as HfImage
        from datasets import load_dataset

        ds = load_dataset(
            "parquet",
            data_files=parquet_data_files(output_dir),
            split="train",
        )
        has_image = "image" in ds.column_names
        if has_image:
            # Keep image bytes raw so we don't materialise PIL.Images for every
            # row at load time — image_for() decodes on demand.
            ds = ds.cast_column("image", HfImage(decode=False))

        out: list[dict[str, Any]] = []
        for raw in ds:
            metrics_raw = raw.get("metrics_json") or ""
            try:
                metrics = json.loads(metrics_raw) if metrics_raw else {}
            except json.JSONDecodeError:
                metrics = {}
            image_bytes: bytes | None = None
            if has_image:
                payload = raw.get("image")
                if isinstance(payload, dict):
                    image_bytes = payload.get("bytes")
            # Older parquets predate the file_id column — fall back to
            # page_id so the viewer keeps working without rewrites.
            file_id = raw.get("file_id") or raw["page_id"]
            out.append(
                {
                    "file_id": file_id,
                    "page_id": raw["page_id"],
                    "model": raw["model"],
                    "source_uri": raw.get("source_uri"),
                    "barcode": raw.get("barcode"),
                    "sequence": raw.get("sequence"),
                    "output_format": raw.get("output_format"),
                    "document_json": raw.get("document_json"),
                    "markdown": raw.get("markdown"),
                    "text": raw.get("text"),
                    "reference_text": raw.get("reference_text"),
                    "reference_provenance_json": raw.get("reference_provenance_json"),
                    "comparisons_json": raw.get("comparisons_json"),
                    "error": raw.get("error"),
                    "metrics": metrics,
                    "image_bytes": image_bytes,
                }
            )
        return out

    def _load_volumes(self, output_dir: Path) -> dict[str, Volume]:
        """Load the ``volumes-*.parquet`` sidecar (when present) into a map.

        Sources without a volume concept (hf_dataset over a flat folder)
        don't write this sidecar; we just return an empty map and the
        viewer renders without volume context.
        """
        shards = find_volumes_files(output_dir)
        if not shards:
            return {}
        try:
            from datasets import load_dataset
        except ImportError:
            return {}
        try:
            ds = load_dataset(
                "parquet",
                data_files=volumes_data_files(output_dir),
                split="train",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("viewer: failed to load volumes sidecar: %s", e)
            return {}
        out: dict[str, Volume] = {}
        for raw in ds:
            vid = raw.get("barcode")
            if not vid:
                continue
            try:
                creators = json.loads(raw.get("creators_json") or "[]")
            except json.JSONDecodeError:
                creators = []
            try:
                extra = json.loads(raw.get("extra_json") or "{}")
            except json.JSONDecodeError:
                extra = {}
            out[str(vid)] = Volume(
                barcode=str(vid),
                title=raw.get("title"),
                creators=creators if isinstance(creators, list) else [],
                language=raw.get("language"),
                year=raw.get("year"),
                rights=raw.get("rights"),
                page_count=raw.get("page_count"),
                source_uri=raw.get("source_uri"),
                extra=extra if isinstance(extra, dict) else {},
            )
        return out

    def _build_page_index(self) -> list[PageRow]:
        by_file: dict[str, list[dict[str, Any]]] = {}
        for r in self._rows:
            by_file.setdefault(r["file_id"], []).append(r)
        pages: list[PageRow] = []
        for file_id, rows in by_file.items():
            rows.sort(key=lambda r: r["model"])
            present = [r["model"] for r in rows]
            errors = [r["model"] for r in rows if r["error"]]
            chars = max(
                (int(r["metrics"].get("text_length") or 0) for r in rows),
                default=0,
            )
            disagreement = self._page_disagreement(rows)
            page_id = rows[0].get("page_id") or file_id
            barcode = next(
                (r.get("barcode") for r in rows if r.get("barcode")), None
            )
            sequence = next(
                (r.get("sequence") for r in rows if r.get("sequence") is not None),
                None,
            )
            has_reference = any(r.get("reference_text") for r in rows)
            comparison_summary = _aggregate_comparison_summary(rows)
            pages.append(
                PageRow(
                    file_id=file_id,
                    page_id=str(page_id),
                    barcode=str(barcode) if barcode else None,
                    sequence=int(sequence) if sequence is not None else None,
                    source_uri=next(
                        (r["source_uri"] for r in rows if r.get("source_uri")), None
                    ),
                    models=present,
                    error_models=errors,
                    has_reference=has_reference,
                    disagreement=disagreement,
                    char_count=chars,
                    comparison_summary=comparison_summary,
                )
            )
        return pages

    def _page_disagreement(self, rows: list[dict[str, Any]]) -> float:
        """Mean pairwise word-diff distance (1 - SequenceMatcher.ratio).

        0.0 = all models agreed exactly; 1.0 = no shared tokens. Delegates to
        :func:`ocrscout.publish._stats.compute_page_disagreement` after
        materialising each row's markdown body — the publisher's dataset
        card consumes the same helper so both surfaces compute identical
        scores.
        """
        from ocrscout.publish._stats import compute_page_disagreement

        materialized = [{"markdown": self._markdown_for(r)} for r in rows]
        return compute_page_disagreement(materialized)

    def _markdown_for(self, row: dict[str, Any]) -> str:
        """Return the row's pre-rendered markdown (parquet ``markdown`` column).

        The writer populates this column from
        ``DoclingDocument.export_to_markdown`` at run time and falls back to
        ``""`` on failure, so the column is always present.
        """
        return row.get("markdown") or ""

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
            html: str | None = None
            if not text and cls == "TableItem":
                # TableItems carry data in `.data.table_cells`, not `.text`.
                # We compute BOTH a plain-text rendering (for diff/legend/
                # plaintext-only consumers) and a structured HTML rendering
                # (for the section pane to render a real <table> element).
                try:
                    text = item.export_to_markdown(doc).strip() or "[table]"
                except Exception:  # noqa: BLE001
                    text = "[table]"
                try:
                    html = item.export_to_html(doc, add_caption=False) or None
                except Exception:  # noqa: BLE001
                    html = None
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
            yield item, idx, label_str, text, html

    def _text_items_for(self, row: dict[str, Any]) -> list[TextItem]:
        """Labeled text items in document reading order (used by text panes)."""
        return [
            TextItem(label=label, text=text, item_idx=idx, html=html)
            for _item, idx, label, text, html in self._iter_doc_items(row)
        ]

    def _bboxes_for(self, row: dict[str, Any]) -> list[BBoxItem]:
        """Pull ``ProvenanceItem.bbox`` off every text/picture/table item."""
        out: list[BBoxItem] = []
        for item, idx, label, text, _html in self._iter_doc_items(row):
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

    def _comparisons_for(self, row: dict[str, Any]) -> dict[str, ComparisonResult]:
        """Decode the ``comparisons_json`` envelope into typed results.

        Uses the comparisons registry to round-trip each entry through its
        ``ComparisonResult`` subclass. Unknown comparison names (downstream
        plugins that aren't installed locally) are silently skipped — the
        viewer still renders cleanly without their typed payload.
        """
        raw = row.get("comparisons_json")
        if not raw:
            return {}
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(envelope, dict):
            return {}
        out: dict[str, ComparisonResult] = {}
        for name, payload in envelope.items():
            try:
                # Lazy-import avoids forcing the comparisons package at
                # store init even when no comparisons ran.
                from ocrscout import registry as _reg

                cmp_cls = _reg.registry.get("comparisons", name)
                # The Comparison class itself doesn't carry the result
                # type; introspect by importing per-comparison module
                # convention. Each module exports a Result subclass next
                # to the Comparison; we look it up by ``__module__`` of
                # the comparison and pick the first ComparisonResult
                # subclass. Defensive for plugin authors who may not
                # follow the convention.
                module = __import__(cmp_cls.__module__, fromlist=["*"])
                result_cls = next(
                    (
                        getattr(module, attr)
                        for attr in dir(module)
                        if isinstance(getattr(module, attr, None), type)
                        and issubclass(
                            getattr(module, attr), ComparisonResult
                        )
                        and getattr(module, attr) is not ComparisonResult
                    ),
                    None,
                )
                if result_cls is None:
                    continue
                out[name] = result_cls.model_validate(payload)
            except Exception:  # noqa: BLE001
                continue
        return out


def _aggregate_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Lift ``comparisons[*].summary`` keys onto a flat per-page dict.

    Used by the sidebar to surface comparison metrics (text_similarity,
    layout_iou_mean, …) as row indicators without re-running comparisons
    at view time. When multiple models on the same page each carry their
    own summary, we keep the max similarity / min CER / max IoU — the
    "best agreement with the reference" reading, which matches what the
    user wants to scan for. Pages without comparisons get ``{}``.
    """
    out: dict[str, float] = {}
    for r in rows:
        raw = r.get("comparisons_json")
        if not raw:
            continue
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(envelope, dict):
            continue
        for cmp_name, payload in envelope.items():
            if not isinstance(payload, dict):
                continue
            summary = payload.get("summary") or {}
            if not isinstance(summary, dict):
                continue
            for key, val in summary.items():
                if not isinstance(val, (int, float)):
                    continue
                flat = f"{cmp_name}_{key}"
                # Aggregate: keep max for similarity/IoU-style "higher is
                # better" metrics; min for error-rate metrics.
                if "cer" in flat or "wer" in flat or "delta" in flat:
                    cur = out.get(flat)
                    out[flat] = float(val) if cur is None else min(cur, float(val))
                else:
                    cur = out.get(flat)
                    out[flat] = float(val) if cur is None else max(cur, float(val))
    return out


def _parse_provenance(raw: str | None) -> ReferenceProvenance | None:
    if not raw:
        return None
    try:
        return ReferenceProvenance.model_validate_json(raw)
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=32)
def _load_image_cached(path: str) -> Image.Image | None:
    try:
        img = _open_fsspec(path) if "://" in path else Image.open(path)
        img.load()
        return img
    except (OSError, FileNotFoundError) as e:
        log.warning("viewer: cannot open source image %s: %s", path, e)
        return None


def _decode_image_bytes(payload: bytes) -> Image.Image | None:
    """Decode an inline image payload (the ``image.bytes`` from a Hub-bundled
    parquet) into a PIL.Image, or ``None`` on failure."""
    try:
        with Image.open(io.BytesIO(payload)) as img:
            img.load()
            return img.copy()
    except (OSError, ValueError) as e:
        log.warning("viewer: cannot decode bundled image bytes: %s", e)
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

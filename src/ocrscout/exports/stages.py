"""Writer + row-builders for the decoupled pipeline stages.

``ocrscout sample`` / ``layout`` / ``ocr`` each materialize a lean intermediate
parquet (``pages-*.parquet`` / ``layout-*.parquet`` / ``raw-*.parquet``) under
``<output_dir>/data/``. These artifacts are a nested superset family (see
``schema.py``) so each stage's output is a self-describing input to the next.

The final ``normalize`` stage produces the rich ``train-*.parquet`` via the
existing :class:`~ocrscout.exports.parquet.ParquetExportAdapter`; the lean
intermediates use :class:`StageWriter` here — same shard-naming / ``Dataset``
machinery, but generic over ``(Features, prefix)`` and without the
``ExportRecord`` shape.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from datasets import Dataset, Features

from ocrscout.exports.layout import DATA_DIR, find_stage_files
from ocrscout.types import LayoutRegion, PageImage, RawOutput

_DEFAULT_BATCH_SIZE = 1000

# Stage prefixes in upstream→downstream order. ``resolve_stage_files`` prefers
# the most upstream artifact present, since all of them carry the pages columns
# every consumer needs.
_STAGE_PREFERENCE: tuple[str, ...] = ("pages", "layout", "raw", "train")


class StageWriter:
    """Incrementally append plain ``dict`` rows to ``data/<prefix>-NNNNN.parquet``.

    Buffers in memory and flushes a fresh shard every ``batch_size`` rows; the
    final partial batch is flushed on ``close()``. Shard numbering continues
    after any shards already present so a resumed stage doesn't clobber prior
    output. Thread-safe: detector / OCR pools call ``write`` from many workers.
    """

    def __init__(
        self,
        output_dir: str | Path,
        features: Features,
        prefix: str,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._features = features
        self._prefix = prefix
        self._batch_size = max(1, int(batch_size))
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._batch_idx = _next_shard_index(self._output_dir, prefix)

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(row)
            should_flush = len(self._rows) >= self._batch_size
        if should_flush:
            self._flush()

    def close(self) -> None:
        self._flush()

    def __enter__(self) -> StageWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _flush(self) -> None:
        with self._lock:
            batch = self._rows
            self._rows = []
            idx = self._batch_idx
            self._batch_idx += 1
        if not batch:
            return
        data_dir = self._output_dir / DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{self._prefix}-{idx:05d}.parquet"
        ds = Dataset.from_list(batch, features=self._features)
        ds.to_parquet(str(path))


def resolve_stage_files(path: str | Path) -> list[Path]:
    """Resolve a ``--source`` parquet pointer to a concrete shard list.

    Accepts, in order of interpretation:

    * a single ``*.parquet`` file → ``[that file]``;
    * a directory holding stage parquets (either the dir itself or its
      ``data/`` subdir) → all shards of the most upstream stage present
      (``pages`` → ``layout`` → ``raw`` → ``train``); since each is a superset
      of the pages columns, any one reconstructs ``PageImage``s;
    * anything else → treated as a glob pattern.

    Returns ``[]`` when nothing matches (callers raise a clear error).
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        for base in (p / DATA_DIR, p):
            if not base.is_dir():
                continue
            for prefix in _STAGE_PREFERENCE:
                shards = sorted(base.glob(f"{prefix}-*.parquet"))
                if shards:
                    return shards
        return []
    matches = sorted(Path().glob(str(path)))
    return [m for m in matches if m.is_file()]


def read_stage_rows(files: list[Path]) -> list[dict[str, Any]]:
    """Read the pages-identity columns from stage parquet ``files`` in order.

    Projects only the ``PAGES_FEATURES`` columns that are present (stage
    parquets are supersets, so every file has them) and returns plain row
    dicts with ``extra_json`` decoded back into an ``extra`` dict.
    """
    import pyarrow.parquet as pq

    from ocrscout.exports.schema import PAGES_COLUMNS

    rows: list[dict[str, Any]] = []
    for path in files:
        schema_names = set(pq.read_schema(str(path)).names)
        cols = [c for c in PAGES_COLUMNS if c in schema_names]
        table = pq.read_table(str(path), columns=cols)
        for rec in table.to_pylist():
            extra_json = rec.pop("extra_json", None)
            rec["extra"] = json.loads(extra_json) if extra_json else {}
            rows.append(rec)
    return rows


def read_raw_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read every ``data/raw-*.parquet`` row under ``output_dir``.

    Returns full ``RAW_FEATURES`` row dicts (with ``extra_json`` decoded back
    into ``extra``), in shard then row order. Consumed by ``ocrscout
    normalize`` to rebuild ``(page, RawOutput, cost_ctx)`` per row.
    """
    import pyarrow.parquet as pq

    files = find_stage_files(Path(output_dir), "raw")
    rows: list[dict[str, Any]] = []
    for path in files:
        for rec in pq.read_table(str(path)).to_pylist():
            extra_json = rec.pop("extra_json", None)
            rec["extra"] = json.loads(extra_json) if extra_json else {}
            rows.append(rec)
    return rows


def _next_shard_index(output_dir: Path, prefix: str) -> int:
    """Next free shard number for ``<prefix>-NNNNN.parquet`` under ``data/``."""
    pat = re.compile(rf"^{re.escape(prefix)}-(\d+)(?:-of-\d+)?\.parquet$")
    max_idx = -1
    for p in find_stage_files(output_dir, prefix):
        m = pat.match(p.name)
        if m is not None:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


# --- row builders -------------------------------------------------------------


def _extra_json(extra: dict[str, Any] | None) -> str | None:
    if not extra:
        return None
    # BHL stuffs string/int identifiers into ``extra``; anything not JSON-able
    # is coerced to its string form so a row never fails to serialize.
    return json.dumps(extra, default=str)


def page_to_row(page: PageImage) -> dict[str, Any]:
    """Serializable ``PageImage`` fields → a ``PAGES_FEATURES`` row."""
    return {
        "page_id": page.page_id,
        "file_id": page.file_id,
        "barcode": page.barcode,
        "sequence": page.sequence,
        "source_uri": page.source_uri,
        "width": page.width or None,
        "height": page.height or None,
        "dpi": page.dpi,
        "extra_json": _extra_json(page.extra),
    }


def layout_row(
    page: PageImage,
    regions: list[LayoutRegion],
    *,
    detector: str,
    detect_seconds: float | None,
    detect_error: str | None,
) -> dict[str, Any]:
    """Pages row + detected regions → a ``LAYOUT_FEATURES`` row."""
    regions_json = json.dumps(
        [r.model_dump(mode="json") for r in regions]
    )
    return {
        **page_to_row(page),
        "regions_json": regions_json,
        "detector": detector,
        "detect_seconds": detect_seconds,
        "detect_error": detect_error,
    }


# Cost/autoscaler columns carried verbatim from OCR into the raw parquet, so
# ``normalize`` can re-emit them into ``train-*.parquet`` without recomputing.
RAW_COST_KEYS: tuple[str, ...] = (
    "gpu_type",
    "provider",
    "cost_per_hour",
    "elapsed_seconds",
    "input_tokens",
    "output_tokens",
    "litellm_cost",
    "gpu_time_cost",
    "kv_cache_memory_bytes",
    "concurrent_requests",
    "region_concurrency",
)


def raw_row(
    page: PageImage,
    model: str,
    raw: RawOutput,
    cost_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pages row + ``RawOutput`` + cost context → a ``RAW_FEATURES`` row."""
    ctx = cost_ctx or {}
    row: dict[str, Any] = {
        **page_to_row(page),
        "model": model,
        "output_format": raw.output_format,
        "raw_payload": raw.payload,
        "tokens": raw.tokens,
        "error": raw.error,
    }
    for key in RAW_COST_KEYS:
        row[key] = ctx.get(key)
    return row

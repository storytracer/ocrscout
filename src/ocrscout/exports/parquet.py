"""ParquetExportAdapter: incrementally append ExportRecord rows to parquet shards.

Rows are buffered in memory and flushed to a fresh shard
(``data/train-NNNNN.parquet``) every ``batch_size`` rows; the final partial
batch is flushed on ``close()``. A sibling ``progress.json`` records the
``page_id`` of every flushed row, so ``ocrscout submit --resume`` can skip
already-processed pages after a crash.

Each row stores the full DoclingDocument as serialized JSON in
``document_json`` plus a pre-rendered markdown string in ``markdown`` so the
parquet remains self-contained for both ``ocrscout viewer`` and HF Hub
publishing.

When the source yielded ``Volume``s, a parallel ``volumes-NNNNN.parquet``
sidecar lands next to the per-page shards, joinable on ``barcode``. Volume
rows are flushed once at close (volumes are small and few; multi-shard
volume files would gain nothing).

When per-page comparisons (text/document/layout) ran, the structured
``ComparisonResult`` envelope is stored in ``comparisons_json`` and the
most-queried metrics are also lifted into flat top-level columns for SQL
ergonomics. See ``RESULTS_FEATURES`` for the canonical set.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from ocrscout.errors import ScoutError
from ocrscout.exports.layout import (
    DATA_DIR,
    SPLIT,
    VOLUMES_PREFIX,
    find_parquet_files,
)
from ocrscout.exports.schema import RESULTS_FEATURES, VOLUMES_FEATURES
from ocrscout.interfaces.export import ExportAdapter
from ocrscout.types import ExportRecord, Volume

_DEFAULT_BATCH_SIZE = 1000
_SHARD_NUMBER_RE = re.compile(rf"^{re.escape(SPLIT)}-(\d+)(?:-of-\d+)?\.parquet$")


class ProgressTracker:
    """Records completed ``page_id``s in ``<output_dir>/progress.json``.

    Atomic writes (tmp file in the same dir + ``os.replace``) so an
    interrupted update never leaves a corrupt JSON. ``--resume`` reads
    this file and filters out already-done page ids before dispatching
    work; on first run, the file is created on the first flush.

    Maintaining the list of completed ids in memory plus on disk is
    O(N) per flush for both reads (already loaded) and writes (atomic
    overwrite). At BHL scale (8000 pages × ~16 byte ids = 128 KB
    progress.json) that's negligible.
    """

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / "progress.json"
        self._completed_page_ids: set[str] = set()
        self._completed_batches: int = 0
        self._source: str | None = None
        self._models: list[str] = []
        self._lock = threading.Lock()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def completed_page_ids(self) -> set[str]:
        """Snapshot of currently-recorded completed page ids."""
        with self._lock:
            return set(self._completed_page_ids)

    def next_batch_index(self) -> int:
        with self._lock:
            return self._completed_batches

    def configure(self, *, source: str | None = None, models: list[str] | None = None) -> None:
        """Stamp run-level metadata. Called at exporter open()."""
        with self._lock:
            if source is not None:
                self._source = source
            if models is not None:
                self._models = list(models)
        self._persist()

    def commit(self, page_ids: list[str]) -> None:
        """Mark a batch as flushed: bump the index, record ids, persist."""
        with self._lock:
            for pid in page_ids:
                self._completed_page_ids.add(pid)
            self._completed_batches += 1
        self._persist()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Corrupt progress.json shouldn't be load-bearing on resume —
            # downgrade to "fresh start" rather than crashing the run.
            return
        if not isinstance(data, dict):
            return
        ids = data.get("completed_page_ids")
        if isinstance(ids, list):
            self._completed_page_ids = {str(x) for x in ids}
        self._completed_batches = int(data.get("completed_batches", 0))
        self._source = data.get("source")
        models = data.get("models")
        if isinstance(models, list):
            self._models = [str(x) for x in models]

    def _persist(self) -> None:
        payload = {
            "completed_page_ids": sorted(self._completed_page_ids),
            "completed_batches": self._completed_batches,
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": self._source,
            "models": self._models,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=False)
            os.replace(tmp_name, self._path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def done_pairs_from_parquet(output_dir: Path) -> dict[str, set[str]]:
    """Per-model done-set built by projecting ``page_id`` + ``model`` from
    every ``data/train-*.parquet`` shard.

    Replaces ``ProgressTracker`` as the resume cursor for ``ocrscout run``:
    the parquet shards are the single source of truth for "which (page, model)
    pairs have been written," so resume cannot drift from the actual output.
    Column projection means only those two columns hit disk; for BHL-scale
    runs (8K pages × N models) this completes in well under a second.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    done: dict[str, set[str]] = {}
    for path in find_parquet_files(output_dir):
        try:
            table = pq.read_table(str(path), columns=["page_id", "model"])
        except (OSError, pa.ArrowInvalid):
            # A partially-written shard would have been atomically replaced
            # by Dataset.to_parquet, but tolerate read errors as "not done"
            # rather than crashing the resume.
            continue
        page_ids = table.column("page_id").to_pylist()
        models = table.column("model").to_pylist()
        for pid, model in zip(page_ids, models):
            if pid is None or model is None:
                continue
            done.setdefault(str(model), set()).add(str(pid))
    return done


class ParquetExportAdapter(ExportAdapter):
    """Incrementally append rows to ``<output_dir>/data/train-NNNNN.parquet``.

    Auto-flushes every ``batch_size`` rows; the final partial batch is
    written on ``close()``. ``batch_size`` defaults to 1000 — tune via
    ``--export-arg batch_size=N`` on the CLI or directly in the adapter
    constructor.
    """

    name = "parquet"

    def __init__(
        self,
        dest: str | Path | None = None,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._output_dir: Path | None = (
            _resolve_output_dir(Path(dest)) if dest is not None else None
        )
        self._batch_size = max(1, int(batch_size))
        self._rows: list[dict[str, Any]] = []
        self._volume_rows: list[dict[str, Any]] = []
        self._progress: ProgressTracker | None = None
        self._batch_idx: int = 0
        self._opened = False
        self._lock = threading.Lock()

    def open(self, dest: str) -> None:
        self._output_dir = _resolve_output_dir(Path(dest))
        self._rows = []
        self._volume_rows = []
        self._progress = ProgressTracker(self._output_dir)
        self._batch_idx = max(
            self._progress.next_batch_index(),
            _next_shard_index(self._output_dir),
        )
        self._opened = True

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    @property
    def progress(self) -> ProgressTracker | None:
        return self._progress

    def write(self, record: ExportRecord) -> None:
        self._ensure_open()
        with self._lock:
            self._rows.append(_record_to_row(record))
            should_flush = len(self._rows) >= self._batch_size
        if should_flush:
            self._flush_rows()

    def write_volume(self, volume: Volume) -> None:
        self._ensure_open()
        with self._lock:
            self._volume_rows.append(_volume_to_row(volume))

    def close(self) -> None:
        if not self._opened:
            return
        self._flush_rows()
        self._flush_volumes()
        self._opened = False

    def __enter__(self) -> ParquetExportAdapter:
        if self._output_dir is not None and not self._opened:
            # Bypass open()'s str signature when we already have a path.
            self.open(str(self._output_dir / DATA_DIR))
        return self

    # --- internals ---------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._opened:
            return
        if self._output_dir is None:
            raise ScoutError(
                "ParquetExportAdapter: write() called before open(); pass "
                "`dest=` to the constructor or call open() first."
            )
        self.open(str(self._output_dir / DATA_DIR))

    def _flush_rows(self) -> None:
        with self._lock:
            batch = self._rows
            self._rows = []
            idx = self._batch_idx
            self._batch_idx += 1
        if not batch:
            return
        assert self._output_dir is not None and self._progress is not None
        data_dir = self._output_dir / DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{SPLIT}-{idx:05d}.parquet"
        ds = Dataset.from_list(batch, features=RESULTS_FEATURES)
        ds.to_parquet(str(path))
        self._progress.commit([row["page_id"] for row in batch])

    def _flush_volumes(self) -> None:
        with self._lock:
            volumes = self._volume_rows
            self._volume_rows = []
        if not volumes:
            return
        assert self._output_dir is not None
        data_dir = self._output_dir / DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        # Volume sidecar lives next to the per-page shards. We always write
        # one combined volumes file rather than one-per-batch — volumes are
        # small and few; multi-shard volume files would gain nothing.
        path = data_dir / f"{VOLUMES_PREFIX}-00000.parquet"
        vol_ds = Dataset.from_list(volumes, features=VOLUMES_FEATURES)
        vol_ds.to_parquet(str(path))


def _resolve_output_dir(path: Path) -> Path:
    """Interpret ``dest`` from the AdapterRef as either an output_dir or a
    legacy file path inside one.

    Legacy call sites pass ``output_dir/data/train-00000-of-00001.parquet``
    (a file path); new ones can pass ``output_dir`` or ``output_dir/data``
    directly. We accept all three.
    """
    if path.suffix == ".parquet":
        # Legacy: ``output_dir/data/<file>.parquet`` → output_dir is grandparent.
        return path.parent.parent
    if path.name == DATA_DIR:
        # ``output_dir/data`` → output_dir is parent.
        return path.parent
    return path


def _next_shard_index(output_dir: Path) -> int:
    """Find the next free shard number under ``output_dir/data/``.

    Used on open() to continue numbering after a crash + resume. Both
    ``train-NNNNN.parquet`` and the legacy ``train-NNNNN-of-MMMMM.parquet``
    naming are recognised.
    """
    existing = find_parquet_files(output_dir)
    if not existing:
        return 0
    max_idx = -1
    for p in existing:
        m = _SHARD_NUMBER_RE.match(p.name)
        if m is None:
            continue
        max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _record_to_row(record: ExportRecord) -> dict[str, Any]:
    doc = record.document
    if hasattr(doc, "model_dump_json"):
        document_json = doc.model_dump_json()
    elif hasattr(doc, "export_to_dict"):
        document_json = json.dumps(doc.export_to_dict())
    else:
        document_json = json.dumps(doc) if doc is not None else None

    comparisons_json: str | None = None
    flat_metrics: dict[str, Any] = {
        "text_similarity": None,
        "text_cer": None,
        "text_wer": None,
        "document_heading_count_delta": None,
        "document_table_count_delta": None,
        "document_picture_count_delta": None,
        "layout_iou_mean": None,
    }
    if record.comparisons:
        envelope: dict[str, Any] = {}
        for name, result in record.comparisons.items():
            envelope[name] = (
                result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else result
            )
            summary = getattr(result, "summary", {}) or {}
            for key, val in summary.items():
                flat_key = f"{name}_{key}"
                if flat_key in flat_metrics:
                    flat_metrics[flat_key] = val
        comparisons_json = json.dumps(envelope)

    reference_provenance_json: str | None = None
    if record.reference is not None:
        prov = record.reference.provenance
        if prov is not None:
            reference_provenance_json = (
                prov.model_dump_json() if hasattr(prov, "model_dump_json")
                else json.dumps(prov)
            )

    return {
        "file_id": record.page.file_id,
        "page_id": record.page.page_id,
        "model": record.model,
        "source_uri": record.page.source_uri,
        "barcode": record.page.barcode,
        "sequence": record.page.sequence,
        "output_format": record.raw.output_format,
        "document_json": document_json,
        "markdown": record.markdown,
        "text": record.text,
        "reference_text": record.reference.text if record.reference else None,
        "reference_provenance_json": reference_provenance_json,
        "raw_payload": record.raw.payload,
        "tokens": record.raw.tokens,
        "error": record.raw.error,
        "metrics_json": json.dumps(record.metrics) if record.metrics else None,
        "comparisons_json": comparisons_json,
        **flat_metrics,
        "gpu_type": record.gpu_type,
        "provider": record.provider,
        "cost_per_hour": record.cost_per_hour,
        "elapsed_seconds": record.elapsed_seconds,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "litellm_cost": record.litellm_cost,
        "gpu_time_cost": record.gpu_time_cost,
        "kv_cache_memory_bytes": record.kv_cache_memory_bytes,
        "concurrent_requests": record.concurrent_requests,
        "region_concurrency": record.region_concurrency,
    }


def _volume_to_row(volume: Volume) -> dict[str, Any]:
    return {
        "barcode": volume.barcode,
        "title": volume.title,
        "creators_json": json.dumps(volume.creators) if volume.creators else None,
        "language": volume.language,
        "year": volume.year,
        "rights": volume.rights,
        "page_count": volume.page_count,
        "source_uri": volume.source_uri,
        "extra_json": json.dumps(volume.extra) if volume.extra else None,
    }

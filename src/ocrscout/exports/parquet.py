"""ParquetExportAdapter: append ExportRecord rows to a parquet file.

Buffers in memory and flushes on ``close()`` via ``datasets.Dataset.to_parquet``.
Each row stores the full DoclingDocument as serialized JSON in
``document_json`` plus a pre-rendered markdown string in ``markdown`` so the
parquet remains self-contained for both ``ocrscout viewer`` and HF Hub
publishing.

When the source yielded ``Volume``s, a parallel ``volumes-NNNNN.parquet``
sidecar lands next to the per-page file, joinable on ``volume_id``.

When per-page comparisons (text/document/layout) ran, the structured
``ComparisonResult`` envelope is stored in ``comparisons_json`` and the
most-queried metrics are also lifted into flat top-level columns for SQL
ergonomics. See ``RESULTS_FEATURES`` for the canonical set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from ocrscout.errors import ScoutError
from ocrscout.exports.layout import volumes_dest_for_pages
from ocrscout.exports.schema import RESULTS_FEATURES, VOLUMES_FEATURES
from ocrscout.interfaces.export import ExportAdapter
from ocrscout.types import ExportRecord, Volume


class ParquetExportAdapter(ExportAdapter):
    name = "parquet"

    def __init__(self, dest: str | Path | None = None) -> None:
        self._dest: str | None = str(dest) if dest is not None else None
        self._rows: list[dict[str, Any]] = []
        self._volume_rows: list[dict[str, Any]] = []
        self._opened = False

    def open(self, dest: str) -> None:
        self._dest = dest
        self._rows = []
        self._volume_rows = []
        self._opened = True

    def write(self, record: ExportRecord) -> None:
        if not self._opened and self._dest is None:
            raise ScoutError("ParquetExportAdapter.write called before open()")
        self._opened = True
        self._rows.append(_record_to_row(record))

    def write_volume(self, volume: Volume) -> None:
        if not self._opened and self._dest is None:
            raise ScoutError("ParquetExportAdapter.write_volume called before open()")
        self._opened = True
        self._volume_rows.append(_volume_to_row(volume))

    def close(self) -> None:
        if self._dest is None:
            return
        path = Path(self._dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = Dataset.from_list(self._rows, features=RESULTS_FEATURES)
        ds.to_parquet(str(path))
        if self._volume_rows:
            volumes_path = volumes_dest_for_pages(path)
            volumes_path.parent.mkdir(parents=True, exist_ok=True)
            vol_ds = Dataset.from_list(self._volume_rows, features=VOLUMES_FEATURES)
            vol_ds.to_parquet(str(volumes_path))
        self._rows = []
        self._volume_rows = []
        self._opened = False

    def __enter__(self) -> ParquetExportAdapter:
        if self._dest is not None and not self._opened:
            self.open(self._dest)
        return self


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
        # Each entry's value is a ComparisonResult Pydantic model — dump it
        # via model_dump (it round-trips because each subclass declares a
        # Literal `comparison` discriminator).
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
        "page_id": record.page.page_id,
        "model": record.model,
        "source_uri": record.page.source_uri,
        "volume_id": record.page.volume_id,
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
    }


def _volume_to_row(volume: Volume) -> dict[str, Any]:
    return {
        "volume_id": volume.volume_id,
        "title": volume.title,
        "creators_json": json.dumps(volume.creators) if volume.creators else None,
        "language": volume.language,
        "year": volume.year,
        "rights": volume.rights,
        "page_count": volume.page_count,
        "source_uri": volume.source_uri,
        "extra_json": json.dumps(volume.extra) if volume.extra else None,
    }

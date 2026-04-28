"""ParquetExportAdapter: append ExportRecord rows to a parquet file.

Buffers in memory and flushes on ``close()`` via ``datasets.Dataset.to_parquet``.
Each row stores the full DoclingDocument as serialized JSON in
``document_json`` plus a pre-rendered markdown string in ``markdown`` so the
parquet remains self-contained for both ``ocrscout viewer`` and HF Hub
publishing.

When the source yielded ``Volume``s, a parallel ``volumes-NNNNN.parquet``
sidecar lands next to the per-page file, joinable on ``volume_id``.
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
    return {
        "page_id": record.page.page_id,
        "model": record.model,
        "source_uri": record.page.source_uri,
        "volume_id": record.page.volume_id,
        "sequence": record.page.sequence,
        "output_format": record.raw.output_format,
        "document_json": document_json,
        "markdown": record.markdown,
        "reference_text": record.reference.text if record.reference else None,
        "raw_payload": record.raw.payload,
        "tokens": record.raw.tokens,
        "error": record.raw.error,
        "metrics_json": json.dumps(record.metrics) if record.metrics else None,
        "scores_json": json.dumps(record.scores) if record.scores else None,
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

"""ParquetExportAdapter: append ExportRecord rows to a parquet file.

Buffers in memory and flushes on ``close()`` via ``datasets.Dataset.to_parquet``.
Each row stores the full DoclingDocument as serialized JSON in
``document_json`` plus a pre-rendered markdown string in ``markdown`` so the
parquet remains self-contained for both ``ocrscout viewer`` and HF Hub
publishing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from ocrscout.errors import ScoutError
from ocrscout.exports.schema import RESULTS_FEATURES
from ocrscout.interfaces.export import ExportAdapter
from ocrscout.types import ExportRecord


class ParquetExportAdapter(ExportAdapter):
    name = "parquet"

    def __init__(self, dest: str | Path | None = None) -> None:
        self._dest: str | None = str(dest) if dest is not None else None
        self._rows: list[dict[str, Any]] = []
        self._opened = False

    def open(self, dest: str) -> None:
        self._dest = dest
        self._rows = []
        self._opened = True

    def write(self, record: ExportRecord) -> None:
        if not self._opened and self._dest is None:
            raise ScoutError("ParquetExportAdapter.write called before open()")
        self._opened = True
        self._rows.append(_record_to_row(record))

    def close(self) -> None:
        if self._dest is None:
            return
        path = Path(self._dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = Dataset.from_list(self._rows, features=RESULTS_FEATURES)
        ds.to_parquet(str(path))
        self._rows = []
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
        "output_format": record.raw.output_format,
        "document_json": document_json,
        "markdown": record.markdown,
        "raw_payload": record.raw.payload,
        "tokens": record.raw.tokens,
        "error": record.raw.error,
        "metrics_json": json.dumps(record.metrics) if record.metrics else None,
        "scores_json": json.dumps(record.scores) if record.scores else None,
    }

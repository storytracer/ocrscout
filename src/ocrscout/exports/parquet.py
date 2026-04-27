"""ParquetExportAdapter: append ExportRecord rows to a parquet file.

Buffers in memory and flushes on ``close()``. Each row stores the full
DoclingDocument as serialized JSON in the ``document_json`` column so the
parquet remains self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ocrscout.errors import ScoutError
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
        if not self._rows:
            # Write an empty file with the schema so downstream tooling sees it.
            table = pa.Table.from_pylist([], schema=_SCHEMA)
        else:
            table = pa.Table.from_pylist(self._rows, schema=_SCHEMA)
        pq.write_table(table, path)
        self._rows = []
        self._opened = False

    def __enter__(self) -> ParquetExportAdapter:
        if self._dest is not None and not self._opened:
            self.open(self._dest)
        return self


_SCHEMA = pa.schema(
    [
        ("page_id", pa.string()),
        ("source_uri", pa.string()),
        ("output_format", pa.string()),
        ("document_json", pa.string()),
        ("raw_payload", pa.string()),
        ("tokens", pa.int64()),
        ("error", pa.string()),
        ("metrics_json", pa.string()),
        ("scores_json", pa.string()),
    ]
)


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
        "source_uri": record.page.source_uri,
        "output_format": record.raw.output_format,
        "document_json": document_json,
        "raw_payload": record.raw.payload,
        "tokens": record.raw.tokens,
        "error": record.raw.error,
        "metrics_json": json.dumps(record.metrics) if record.metrics else None,
        "scores_json": json.dumps(record.scores) if record.scores else None,
    }

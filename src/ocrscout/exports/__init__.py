"""Export adapters: write a stream of ExportRecord rows to storage."""

from ocrscout.exports.parquet import ParquetExportAdapter

__all__ = ["ParquetExportAdapter"]

"""Shared schema for ocrscout's results parquet.

Single source of truth for the column shape, used by:

* ``ParquetExportAdapter`` (writer)
* ``ViewerStore`` and ``cli.inspect`` (readers)
* ``cli.publish`` (which adds an extra ``image`` column for HF Hub publishing)

All columns are nullable. ``image`` is omitted from the writer schema and is
appended by the publisher when ``--bundle-images`` is set.
"""

from __future__ import annotations

from datasets import Features, Value

RESULTS_FEATURES: Features = Features(
    {
        "page_id": Value("string"),
        "model": Value("string"),
        "source_uri": Value("string"),
        "output_format": Value("string"),
        "document_json": Value("string"),
        "markdown": Value("string"),
        "raw_payload": Value("string"),
        "tokens": Value("int64"),
        "error": Value("string"),
        "metrics_json": Value("string"),
        "scores_json": Value("string"),
    }
)

RESULTS_COLUMNS: tuple[str, ...] = tuple(RESULTS_FEATURES.keys())

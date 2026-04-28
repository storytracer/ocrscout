"""Shared schema for ocrscout's results parquet.

Single source of truth for the column shape, used by:

* ``ParquetExportAdapter`` (writer)
* ``ViewerStore`` and ``cli.inspect`` (readers)
* ``cli.publish`` (which adds an extra ``image`` column for HF Hub publishing)

All columns are nullable. ``image`` is omitted from the writer schema and is
appended by the publisher when ``--bundle-images`` is set.

``volume_id`` and ``sequence`` are populated by sources that group pages into
bibliographic units (BHL, IA, HathiTrust); flat sources like ``hf_dataset``
leave them ``None``. The companion ``volumes-NNNNN.parquet`` sidecar (one row
per volume) holds the bibliographic metadata, joinable on ``volume_id``.

``reference_text`` carries the ground-truth OCR string when a reference
adapter is configured; null otherwise.
"""

from __future__ import annotations

from datasets import Features, Value

RESULTS_FEATURES: Features = Features(
    {
        "page_id": Value("string"),
        "model": Value("string"),
        "source_uri": Value("string"),
        "volume_id": Value("string"),
        "sequence": Value("int64"),
        "output_format": Value("string"),
        "document_json": Value("string"),
        "markdown": Value("string"),
        "reference_text": Value("string"),
        "raw_payload": Value("string"),
        "tokens": Value("int64"),
        "error": Value("string"),
        "metrics_json": Value("string"),
        "scores_json": Value("string"),
    }
)

RESULTS_COLUMNS: tuple[str, ...] = tuple(RESULTS_FEATURES.keys())

VOLUMES_FEATURES: Features = Features(
    {
        "volume_id": Value("string"),
        "title": Value("string"),
        "creators_json": Value("string"),
        "language": Value("string"),
        "year": Value("int64"),
        "rights": Value("string"),
        "page_count": Value("int64"),
        "source_uri": Value("string"),
        "extra_json": Value("string"),
    }
)

VOLUMES_COLUMNS: tuple[str, ...] = tuple(VOLUMES_FEATURES.keys())

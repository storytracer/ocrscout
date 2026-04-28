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

``markdown`` is the prediction rendered for human reading; ``text`` is the
same DoclingDocument flattened to plain text so it's directly comparable to
``reference_text`` (no markdown escapes / heading syntax / image refs to
strip first). ``reference_text`` carries the reference adapter's text when
configured; null otherwise.

Comparisons are persisted in two complementary forms: ``comparisons_json``
holds the full structured ``ComparisonResult`` envelope (one entry per
comparison name that ran), and a small set of canonical flat columns
(``text_similarity``, ``text_cer``, ``text_wer``, ``document_*_delta``,
``layout_iou_mean``) lift the most-queried metrics into top-level columns
for ergonomic SQL aggregation. ``reference_provenance_json`` carries the
typed provenance of the reference for the row's page (same value for every
model row of a given page) so downstream consumers can interpret the
metrics correctly (oracle vs incumbent OCR).
"""

from __future__ import annotations

from datasets import Features, Value

RESULTS_FEATURES: Features = Features(
    {
        "file_id": Value("string"),
        "volume_id": Value("string"),
        "sequence": Value("int64"),
        "page_id": Value("string"),
        "text": Value("string"),
        "reference_text": Value("string"),
        "reference_provenance_json": Value("string"),
        "markdown": Value("string"),
        "source_uri": Value("string"),
        "output_format": Value("string"),
        "model": Value("string"),
        "document_json": Value("string"),
        "raw_payload": Value("string"),
        "tokens": Value("int64"),
        "error": Value("string"),
        "metrics_json": Value("string"),
        "comparisons_json": Value("string"),
        # Canonical flat metric columns. Populated from
        # `comparisons[<name>].summary[<key>]` when the matching comparison
        # ran; null otherwise. Add more names here as new comparison-summary
        # keys earn their place.
        "text_similarity": Value("float64"),
        "text_cer": Value("float64"),
        "text_wer": Value("float64"),
        "document_heading_count_delta": Value("int64"),
        "document_table_count_delta": Value("int64"),
        "document_picture_count_delta": Value("int64"),
        "layout_iou_mean": Value("float64"),
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

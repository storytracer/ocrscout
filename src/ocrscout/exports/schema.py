"""Shared schema for ocrscout's results parquet.

Single source of truth for the column shape, used by:

* ``ParquetExportAdapter`` (writer)
* ``ViewerStore`` and ``cli.inspect`` (readers)
* ``cli.publish`` (which adds an extra ``image`` column for HF Hub publishing)

All columns are nullable. ``image`` is omitted from the writer schema and is
appended by the publisher when ``--bundle-images`` is set.

``barcode`` and ``sequence`` are populated by sources that group pages into
bibliographic units (BHL, IA, HathiTrust); flat sources like ``hf_dataset``
leave them ``None``. The companion ``volumes-NNNNN.parquet`` sidecar (one row
per volume) holds the bibliographic metadata, joinable on ``barcode``.

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
        "barcode": Value("string"),
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
        # Cost / GPU context. The first three describe the hardware /
        # provider the page ran on (stamped uniformly across a run); the
        # next five come per-page from LiteLLM's success_callback via
        # ``ocrscout.cost.recorder``. ``gpu_time_cost`` is derived
        # (``elapsed_seconds / 3600 × cost_per_hour``) and kept as its own
        # column so ``ocrscout costs`` can sum either token cost or infra
        # cost without re-deriving. All nullable: backends that don't go
        # through LiteLLM (e.g. Tesseract) leave them null.
        "gpu_type": Value("string"),
        "provider": Value("string"),
        "cost_per_hour": Value("float64"),
        "elapsed_seconds": Value("float64"),
        "input_tokens": Value("int64"),
        "output_tokens": Value("int64"),
        "litellm_cost": Value("float64"),
        "gpu_time_cost": Value("float64"),
        # Autoscaler context. Stamped per row from the active profile at
        # the time the page ran, so cross-run / cross-hardware DuckDB
        # queries can correlate throughput / $-per-page with the
        # concurrency the runner chose. ``kv_cache_memory_bytes`` and
        # ``concurrent_requests`` are populated for every ``runtime: vllm``
        # row; ``region_concurrency`` is populated only when the row
        # came from ``backend: layout_chat`` (null for full-page
        # ``backend: litellm`` rows). Hosted-API rows leave all three
        # null since the autoscaler doesn't apply.
        "kv_cache_memory_bytes": Value("int64"),
        "concurrent_requests": Value("int32"),
        "region_concurrency": Value("int32"),
    }
)

RESULTS_COLUMNS: tuple[str, ...] = tuple(RESULTS_FEATURES.keys())

# --- Decoupled-stage intermediate schemas -------------------------------------
#
# The pipeline can run as four independent stages, each reading a parquet and
# writing a parquet (see ``ocrscout sample`` / ``layout`` / ``ocr`` /
# ``normalize``). The three schemas below are a nested family so each stage's
# output is a self-describing superset of the previous one's:
#
#   PAGES_FEATURES  ⊂  LAYOUT_FEATURES   (pages + detected regions)
#   PAGES_FEATURES  ⊂  RAW_FEATURES  ⊂  RESULTS_FEATURES   (pages + OCR + cost)
#
# All carry ``source_uri`` so the consuming stage can re-fetch / re-decode the
# image on demand (images are reference-only — never embedded). ``extra`` (e.g.
# BHL's BarCode/ItemID/PageID) round-trips as ``extra_json``. All nullable.

# The serializable subset of ``PageImage`` (image / image_loader are runtime
# only). This is the ``ocrscout sample`` output and the minimal input every
# downstream stage can reconstruct a lazy ``PageImage`` from.
PAGES_FEATURES: Features = Features(
    {
        "page_id": Value("string"),
        "file_id": Value("string"),
        "barcode": Value("string"),
        "sequence": Value("int64"),
        "source_uri": Value("string"),
        "width": Value("int64"),
        "height": Value("int64"),
        "dpi": Value("int64"),
        "extra_json": Value("string"),
    }
)

PAGES_COLUMNS: tuple[str, ...] = tuple(PAGES_FEATURES.keys())

# ``ocrscout layout`` output: pages columns + detected regions. A superset of
# PAGES_FEATURES so it also serves directly as an ``ocrscout ocr`` input
# (no separate pages parquet needed). ``regions_json`` is a JSON list of
# ``LayoutRegion``; ``detect_error`` is null on success.
LAYOUT_FEATURES: Features = Features(
    {
        **PAGES_FEATURES,
        "regions_json": Value("string"),
        "detector": Value("string"),
        "detect_seconds": Value("float64"),
        "detect_error": Value("string"),
    }
)

LAYOUT_COLUMNS: tuple[str, ...] = tuple(LAYOUT_FEATURES.keys())

# ``ocrscout ocr`` output: pages columns + the ``RawOutput`` fields + the
# cost/autoscaler columns (cost is recorded at OCR time via the LiteLLM
# callback). A strict subset of RESULTS_FEATURES: ``ocrscout normalize`` reads
# this, fills in document/text/markdown/comparisons/reference, and emits the
# full results schema — carrying these cost columns through verbatim.
RAW_FEATURES: Features = Features(
    {
        **PAGES_FEATURES,
        "model": Value("string"),
        "output_format": Value("string"),
        "raw_payload": Value("string"),
        "tokens": Value("int64"),
        "error": Value("string"),
        "gpu_type": Value("string"),
        "provider": Value("string"),
        "cost_per_hour": Value("float64"),
        "elapsed_seconds": Value("float64"),
        "input_tokens": Value("int64"),
        "output_tokens": Value("int64"),
        "litellm_cost": Value("float64"),
        "gpu_time_cost": Value("float64"),
        "kv_cache_memory_bytes": Value("int64"),
        "concurrent_requests": Value("int32"),
        "region_concurrency": Value("int32"),
    }
)

RAW_COLUMNS: tuple[str, ...] = tuple(RAW_FEATURES.keys())

VOLUMES_FEATURES: Features = Features(
    {
        "barcode": Value("string"),
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

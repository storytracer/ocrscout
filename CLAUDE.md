# CLAUDE.md

## Project overview

ocrscout is a toolkit for evaluating SOTA OCR models on user-supplied data and hardware. Inspired by the HuggingFace `uv-scripts/ocr` collection (which we read as reference material but no longer execute), ocrscout drives vLLM and Docling directly via curated per-model profiles, normalizes their varied outputs (markdown / DocTags / layout JSON) into a single `DoclingDocument` model, and adds the measurement layer (timing, throughput, edit distance, side-by-side comparison) those scripts lack. The package itself has **zero GPU dependencies** — all GPU work happens in `VllmBackend` subprocesses or remote vLLM servers.

## How to add a new component

ocrscout is extended through Abstract Base Classes and Python entry points.

1. Subclass the relevant ABC in [src/ocrscout/interfaces/](src/ocrscout/interfaces/):
   - `SourceAdapter` — yields `PageImage` objects from somewhere (a directory, an HF dataset, IIIF, PDF). The shipped `hf_dataset` adapter at [src/ocrscout/sources/hf_dataset.py](src/ocrscout/sources/hf_dataset.py) covers local dirs, fsspec URLs (`s3://`, `gs://`, `https://`, `hf://`), and HF Hub repo IDs (`org/dataset`) through one code path — extend it (or write a new adapter) for IIIF, PDF rasterisation, etc. The [`bhl`](src/ocrscout/sources/bhl.py) adapter shows how to handle a catalog-driven corpus (305K volumes, 67M pages on S3). Sources may optionally implement `iter_volumes()` to yield typed `Volume` metadata that lands as a parallel `volumes-*.parquet` sidecar.
   - `ReferenceAdapter` — returns a `Reference` (text and/or `DoclingDocument`, with typed `provenance`) for a `PageImage`. Note "reference", not "ground truth": most references in the wild are themselves OCR — see "Comparisons subsystem" below.
   - `ModelBackend` — prepares a `BackendInvocation` and runs it, yielding `RawOutput`.
   - `LayoutDetector` — emits typed `LayoutRegion`s (bbox + native category) for a `PageImage`. Consumed by layout-aware backends like `LayoutChatBackend`. Built-in: `pp-doclayout-v3` (transformers + torch via the `[layout]` extra). See "Layout-aware OCR" below.
   - `Normalizer` — converts a `RawOutput` + `PageImage` into a `DoclingDocument`.
   - `ExportAdapter` — writes a stream of `ExportRecord` objects to a destination.
   - `Comparison` — one analytic axis (text / document / layout / future semantic). Takes typed `PredictionView` + `BaselineView`, returns a typed `ComparisonResult` subclass. See "Comparisons subsystem" below.
   - `ComparisonRenderer` — terminal/HTML/Gradio surfaces for one `ComparisonResult` subclass. Same registry pattern as comparisons themselves.
   - `Benchmark` — bundles a source, reference adapter, and a list of comparisons with a canonical summary protocol.
   - `Reporter` — turns a results directory into a report (HTML, markdown, terminal).

2. Set the `name: ClassVar[str]` on your class — this is the lookup key in the registry.

3. Register your component one of two ways:
   - **In a downstream package**: declare an entry point in your `pyproject.toml`:
     ```toml
     [project.entry-points."ocrscout.normalizers"]
     my_normalizer = "my_pkg.normalizers:MyNormalizer"
     ```
     Entry-point groups: `ocrscout.sources`, `ocrscout.references`, `ocrscout.backends`, `ocrscout.normalizers`, `ocrscout.exports`, `ocrscout.comparisons`, `ocrscout.comparison_renderers`, `ocrscout.benchmarks`, `ocrscout.reporters`, `ocrscout.layout_detectors`.
   - **In-process**: `from ocrscout import registry; registry.register("normalizers", "my_normalizer", MyNormalizer)`.

Built-in components are registered in [src/ocrscout/registry.py](src/ocrscout/registry.py) and are protected from being shadowed by third-party entry points.

## Model profiles

Each OCR model has a hand-curated YAML profile shipped at [src/ocrscout/profiles/](src/ocrscout/profiles/). There is no auto-generated tier and no synthesized fallback: if `ocrscout run --models X` can't find `X.yaml`, it errors with a `ProfileNotFound` and points the user at `ocrscout introspect`.

A profile records:

- **Identity**: `name`, `source` (which backend — any registered name; the field is a plain `str` so adding a backend doesn't require touching the schema), `model_id`, `model_size`, `upstream_script` (informational pointer to the HF reference script).
- **Output shape**: `output_format`, `normalizer`, `has_bboxes`, `has_layout`, `category_mapping` (script-output-category → DoclingDocument label).
- **Prompts**: `prompt_templates` (mode → prompt string), `preferred_prompt_mode`, `chat_template_content_format`.
- **vLLM tuning**: `vllm_engine_args` (passed to `vllm.LLM(...)`), `sampling_args` (passed to `vllm.SamplingParams(...)`), `vllm_version`, `server_url` (for HTTP server mode).
- **Layout-aware orchestration** (consumed by `source: layout_chat`): `layout_detector` (registered detector name, e.g. `pp-doclayout-v3`), `layout_detector_args` (constructor kwargs for the detector — `device`, `score_threshold`, `revision`, …), `prompt_mode_per_category` (detector-native category → `prompt_templates` key, for routing per-region prompts; falls back to `preferred_prompt_mode` for anything not listed). A cross-field validator on `ModelProfile` rejects malformed `layout_chat` profiles at YAML load time (requires `layout_detector` set, `output_format == "layout_json"`, `normalizer == "layout_json"`).
- **Free-form**: `backend_args`, `metadata`.

### Profile defaults

Some `vllm_engine_args` keys are constant across every shipped profile and live in `DEFAULT_VLLM_ENGINE_ARGS` in [src/ocrscout/profile.py](src/ocrscout/profile.py) instead of being repeated in each YAML — currently `trust_remote_code: true` and `cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32]`. Both runner and managed modes call `effective_vllm_engine_args(profile)` to merge defaults under the profile's overrides (profile wins). Likewise, the `vllm_version` field defaults to `">=0.15.1"` and `backend_args.batch_size` defaults to `16` (in `_DEFAULT_BATCH_SIZE` at [src/ocrscout/backends/vllm.py](src/ocrscout/backends/vllm.py)). Don't restate any of these in a new profile unless you need to override.

### vLLM backend modes

Profiles with `source: vllm` can run in three modes:

| Mode | Trigger | Lifecycle | Use case |
| --- | --- | --- | --- |
| **runner** | default (no flag) | per ocrscout invocation: `uv run --with vllm runner.py` subprocess; killed when the run exits | one-off scout, single model, no warm server available |
| **external-server** | `--server-url URL` | nothing — user/operator manages the server | a `vllm serve` you started by hand, or any OpenAI-compatible endpoint (incl. a LiteLLM proxy you started) |
| **managed** | `ocrscout serve` (long-lived) **or** `ocrscout run --managed` (inline) | ocrscout owns N `vllm serve` + 1 LiteLLM proxy (when N≥2); torn down on exit/Ctrl-C | comparing multiple models with warm servers; one URL fronting many models |

Examples of each:

```bash
# runner (default)
uv run ocrscout run --source ./images/ --models dots-mocr

# external-server (user starts vllm serve in another terminal)
uv run ocrscout run --source ./images/ --models dots-mocr \
  --server-url http://localhost:8000/v1

# managed, inline (ocrscout spawns + tears down for this single run)
uv run ocrscout run --source ./images/ --models dots-mocr,dots-ocr,glm-ocr \
  --managed

# managed, long-lived (ocrscout serves; another terminal drives runs)
uv run ocrscout serve --models dots-mocr,dots-ocr,glm-ocr
# in another terminal:
uv run ocrscout run --source ./images/ --models dots-mocr \
  --server-url http://localhost:4000/v1
```

`--managed` and `--server-url` are mutually exclusive — you'd be telling ocrscout to both manage its own server and connect to someone else's.

**Server-mode notes.** The base URL must include the OpenAI prefix (`/v1`); we don't auto-append because some deployments live behind a custom proxy path. Concurrency is controlled by `backend_args.concurrent_requests` on the profile (default 8 parallel POSTs); per-request timeout by `backend_args.request_timeout` (default 300s). Precedence for the server URL: `--server-url` flag → `OCRSCOUT_VLLM_URL` env var → `server_url:` in the profile YAML.

**Managed-mode lifecycle.** ocrscout owns the subprocess tree: the vllm-serve PIDs and (when N≥2) the LiteLLM proxy PID live until the run exits or the user hits Ctrl-C. Logs land in `/tmp/ocrscout-managed-<uuid>/<model>.log` and `litellm.log` and survive teardown for debugging. Teardown signals each child's process group with SIGTERM (10s grace), then SIGKILL fallback. Children are spawned with `PR_SET_PDEATHSIG=SIGTERM`, so they die promptly even on abnormal ocrscout exit (SIGKILL, OOM, screen `-X quit`). When only one vllm-source model is requested, the proxy is skipped (no routing benefit) and the single vllm-serve port is the endpoint.

**Per-profile KV cache budgets.** Each managed-mode profile must declare an absolute KV cache budget via `vllm_engine_args.kv_cache_memory_bytes` (vLLM's `--kv-cache-memory-bytes` flag; SI suffixes accepted, e.g. `16G` = 16 GiB, `25.6m` = 25,600,000 bytes). When set, vLLM allocates exactly that much for KV and ignores `--gpu-memory-utilization` — which eliminates the cudaMemGetInfo race that previously plagued parallel sibling spawns and gives every engine a deterministic, predictable footprint. Sizing rule of thumb: `kv_cache_memory_bytes ≈ concurrent_requests × max_model_len × ~30_000` bytes per token (the per-token cost depends on architecture; profile and adjust).

**GPU budget.** `--gpu-budget` (default 0.85) is the collective ceiling fraction of total VRAM the managed stack may claim — it's a sanity check, not a splitter. At spawn time, ocrscout sums each profile's `kv_cache_memory_bytes` plus a per-profile overhead estimate (model weights from `model_size` × bytes-per-param from `dtype`, plus a 25%-of-weights / 1 GiB-min working slack for activations + cudagraphs + allocator), and rejects the run if the total would exceed `min(total_VRAM × gpu_budget, free_VRAM × 0.95)`. The error message names each profile's KV + overhead contribution so you can see which to shrink. If `model_size` is missing or unparseable, ocrscout falls back to a flat 8 GiB conservative estimate. If `kv_cache_memory_bytes` is missing, the run is rejected before any subprocess spawns with a remediation message.

**Sequential model execution by default.** `--parallel-models` defaults to `1` even when multiple models are managed: vLLM servers all spawn in parallel (one startup cost), but page requests run against one server at a time. On a single GPU, concurrent execution just halves throughput per model and yields contended s/page numbers that aren't useful for benchmarking. Sequential gives each model the full GPU, produces honest per-model timings, and total wall-clock is ~equivalent (the GPU is the bottleneck either way). Override with `-P > 1` only if you have separate GPUs per model.

**When to pick which mode.** Runner is right when you're scouting one model on a fresh dataset; the startup cost amortizes across the run. External-server is right when you're iterating on prompts/inputs against a single model and want to avoid paying startup per ocrscout invocation. Managed is right when you're comparing multiple models in one go (the proxy lets a single ocrscout run drive all of them) or when you want the convenience of "one command spins everything up and tears it down."

**High-resolution images are a known soft spot.** ocrscout's preflight is sizing-blind to image dimensions: it sizes KV/overhead for "typical" workloads (~256 vision tokens per MP at default Qwen2-VL-style processor settings). With much larger inputs (>4 MP), four things degrade in this order: (1) `prompt_tokens + max_tokens > max_model_len` → vLLM returns HTTP 400 and the page is recorded as `pages_failed`; (2) per-request KV demand grows linearly, draining the cache and pushing requests into vLLM's `Waiting:` queue; (3) vision-encoder transient activations may spike past the per-engine cap and OOM mid-run (preflight won't catch this — the cap covers steady-state, not the encoder peak); (4) vLLM's encoder cache budget (~14400 tokens by default) overflows and visual features get recomputed per request, slowing prefill. None of this surfaces as a clear up-front error today. If/when this bites, the cleanest defense is `vllm_engine_args.mm_processor_kwargs.max_pixels: <N>` to cap input resolution at the source — bounds prefill, KV use, and encoder memory in one knob with a predictable accuracy trade-off. Other levers: bump `max_model_len`, lower `sampling_args.max_tokens`, raise `kv_cache_memory_bytes`. Worth revisiting when scouting tabloid-size scans, large maps, or anything substantially above 4 MP.

### Layout-aware OCR (`layout_chat` backend)

Single-task-per-call OCR models (GLM-OCR, PaddleOCR-VL, …) take one prompt prefix and do exactly one thing — text, table, formula, or chart extraction — per call. Run them in default `ocr` mode against a mixed-content page and tables come back as flat prose because that's literally what "Text Recognition:" returns. The `layout_chat` backend (registered as `source: layout_chat`) wraps these with an external layout detector: detect typed regions on the page → for each region, crop the source image and POST one chat-completion with the prompt that matches the region's category → assemble the per-region results into one `layout_json` payload that the existing `LayoutJsonNormalizer` turns into a structured `DoclingDocument` with proper `TableItem`s and bbox provenance.

| Knob | Where | What it does |
| --- | --- | --- |
| `layout_detector` | profile YAML | Registered `LayoutDetector` name (built-in: `pp-doclayout-v3`). |
| `layout_detector_args` | profile YAML | Detector kwargs (`device`, `score_threshold`, `revision`, …). |
| `prompt_mode_per_category` | profile YAML | Detector-native category → `prompt_templates` key. Lookup is on the **detector's** raw label (before `category_mapping` is applied). Anything not listed falls through to `preferred_prompt_mode`. |
| `backend_args.region_concurrency` | profile YAML | Parallel POSTs per page (default 16). Pages stay sequential within a profile. |

**Server-mode-only.** `layout_chat.prepare()` fails fast if no OpenAI-compatible endpoint is configured (`--server-url`, `--managed`, or `profile.server_url`). Subprocess-mode vLLM is unsupported because the per-region launch cost would dwarf any benefit. The backend declares `requires_managed_vllm: ClassVar[bool] = True`, which is what `managed.py`'s filter looks at when deciding which profiles get a managed `vllm serve` spawned for them — so adding a new layout-aware backend is one class flag, not a new string in `managed.py`.

**The first detector.** `pp-doclayout-v3` uses the official Hugging Face Transformers integration of PP-DocLayoutV3 (`AutoModelForObjectDetection` + `AutoImageProcessor` against `PaddlePaddle/PP-DocLayoutV3_safetensors`). Lazy-imports `transformers` and `torch` only on first `detect()` call (so `import ocrscout` stays snappy without the `[layout]` extra). The model emits results in its own predicted reading order — `LayoutChatBackend._sort_reading_order` prefers `LayoutRegion.reading_order` when the detector populates it, falling back to a top-then-left bucketed sort otherwise.

**A/B pattern.** Pair a layout-aware profile with its whole-page sibling so users can compare honestly: ship `glm-ocr.yaml` (whole-page, single mode) alongside `glm-ocr-layout.yaml` (PP-DocLayoutV3 + per-region routing); same for `paddleocr-vl` vs `paddleocr-vl-layout`. The layout variant's per-page wall-clock is typically 2–3× higher because each region is one HTTP call, but you get structured tables and bbox provenance for the viewer overlay.

### Table parsing (`_tables.py`)

Normalizer-internal helper at [src/ocrscout/normalizers/_tables.py](src/ocrscout/normalizers/_tables.py) provides three parsers + a smart dispatcher:

- `parse_html_table(text)` — `<table>...</table>` (lighton-ocr2, dots-ocr, glm-ocr in Table Recognition mode).
- `parse_pipe_table(text)` — GitHub-flavoured Markdown pipe tables.
- `parse_otsl_table(text)` — DocTags / OTSL `<fcel>...<nl>` fragments (PaddleOCR-VL Table Recognition mode; delegates to `docling_core.types.doc.document.parse_otsl_table_content`).
- `parse_table_payload(text)` — the dispatcher; picks OTSL → HTML → pipe-table by inspecting the payload markers.

`LayoutJsonNormalizer` calls `parse_table_payload` for every `category == "table"` block, so future models that emit any of these formats inside a layout-json table block work without normalizer changes. All variants degrade to an empty `TableData` on parse failure rather than raising — per-block resilience is the contract.

### Authoring a new profile (Claude-Code-assisted)

1. Run `uv run ocrscout introspect <name>` — fetches the upstream HF script (`uv-scripts/ocr/<name>.py`) and prints a draft YAML with TODO markers (model_id, vLLM engine args, sampling params, prompt template selection).
2. Pipe the draft to `src/ocrscout/profiles/<name>.yaml` and open it alongside the upstream script (cached at `~/.cache/ocrscout/uv-scripts-ocr/<name>.py`).
3. Ask Claude Code (or do it manually) to read the upstream source and resolve the TODOs: copy `PROMPT_TEMPLATES` verbatim, port the `LLM(...)` kwargs to `vllm_engine_args`, port `SamplingParams(...)` to `sampling_args`, decide `chat_template_content_format` based on how the script calls `llm.chat(...)`. Also pick an absolute `kv_cache_memory_bytes` (start from `concurrent_requests × max_model_len × ~30_000` bytes and round to a friendly number like `16G`). Skip `trust_remote_code`, `cudagraph_capture_sizes`, `vllm_version`, and `backend_args.batch_size` — those have defaults (see "Profile defaults" above) and should only be set to override.
4. Validate with `uv run ocrscout run --source <fixture> --models <name>` on a small dataset.

Introspection is purely static (`ast.parse`) — the upstream script is never executed during this workflow.

## Key decisions

- **`DoclingDocument` is THE document model.** Never invent another. If it can't represent something, extend `docling-core`, don't fork.
- **Zero GPU deps in core.** `pyproject.toml` core dependencies must install in seconds on a CPU-only machine. GPU work belongs in `VllmBackend` subprocesses (`uv run --with vllm ...`), remote vLLM servers, or external Docling installs.
- **Pydantic v2** for all config, profiles, metrics, and IO. YAML round-trips through `model_validate` / `model_dump`.
- **`uv`** for environment management throughout. The CLI and the `VllmBackend` runner subprocesses both use it.
- **Entry points** are the extensibility contract. Downstream packages add sources, normalizers, evaluators, etc., without touching ocrscout's source tree.

## Logging

All status output flows through Python's stdlib `logging` module under the `ocrscout` namespace, rendered by a plain `StreamHandler` configured via [src/ocrscout/log.py:setup_logging](src/ocrscout/log.py). No Rich, no markup parsing, no width-based wrapping — one logical line per record so the output is grep-friendly and pastes cleanly into bug reports. CLI commands accept `-v` / `-vv` / `-q` to control verbosity:

| Flag | Level | Format | What appears |
| --- | --- | --- | --- |
| `-q` | WARNING | bare message | Errors, warnings, summary table, ready banners |
| (default) | INFO | bare message | + per-page progress, per-model start/done, GPU allocation summary |
| `-v` | VERBOSE (15) | `HH:MM:SS LEVEL message` | + timestamps and level names, full URLs/paths, GPU per-process telemetry |
| `-vv` | DEBUG | `HH:MM:SS LEVEL name:line  message` | + source paths, subprocess argvs, every nvitop probe |

**When to log vs. rprint.** Use `log.info(...)` (or `log.debug` / `log.log(VERBOSE, ...)`) for events — anything with a level that should be filterable. Reserve `rich.print` for *presentation* artifacts that should always render regardless of verbosity: the ready banner from `ocrscout serve`, the per-model summary table from `ocrscout run`. The rule of thumb: if a user's `--quiet` invocation should still see it, it's presentation and uses `rich.print`; otherwise, it's an event and goes through the logger. Don't put Rich markup tags (`[bold]`, `[red]`, etc.) in log messages — the stdlib formatter won't parse them and they'd appear as literal characters.

**Per-model prefix.** When a backend processes pages for a model, prefix every log line with `[{profile.name}]`. This is the contract that lets parallel-model output (`--parallel-models > 1`) interleave readably. See [src/ocrscout/backends/vllm.py](src/ocrscout/backends/vllm.py) for the pattern.

**Subprocess output.** `VllmBackend` runner mode streams the vllm child's stdout to the parent terminal at INFO level (via `log.isEnabledFor` — the firehose is suppressed at `-q`). The `managed.py` lifecycle writes each child's output to a per-model log file under `/tmp/ocrscout-managed-<uuid>/` regardless of verbosity, so post-mortem debugging always has the full picture.

## Inspecting results

Two read-only post-mortem tools share the same input — a previous run's `output_dir/data/train-*.parquet`. Each row carries its own pre-rendered markdown in the `markdown` column (no on-disk sidecars; the column is populated at write time from `DoclingDocument.export_to_markdown`):

| Command | Where | Deps | Use case |
| --- | --- | --- | --- |
| `ocrscout inspect <out>` | terminal | core only | Rich summary table (with conditional comparison metric columns when present: `text_similarity`, `cer`, `wer`, `iou_mean`, …); `--page <id>` per-model markdown dump plus reference text when present; `--compare a,b [--html] [--comparison-type {text,document,layout}]` runs a typed `Comparison` and serves the result either inline (terminal-rendered) or as a one-shot HTML page over the LAN. Right tool for SSH, CI logs, and quick pokes. Either side of `--compare` may be the literal `reference` to compare against the run's reference adapter. |
| `ocrscout viewer <out>` | browser | `[viewer]` extra (`gradio`) | Long-lived Gradio app. Three view modes: Single, Side-by-side, Compare. Page picker, mode-aware artifact picker (radio for Single, checkbox for Side-by-side — including a `reference` checkbox when the run has baselines, paired Prediction / Baseline dropdowns for Compare), source page with color-coded bbox overlay + deduplicated category legend, and matching color-coded section blocks in the text pane for layout-aware models. Compare mode stacks every registered comparison whose required modality is satisfied by both sides. State (page / models / mode) persists via `gr.BrowserState` and can be supplied via URL query params for shareable views. |

Both surfaces share rendering through the comparison-renderer registry: every `ComparisonResult` subclass has a `ComparisonRenderer` registered under `comparison_renderers`, which exposes `render_html` (used by `inspect --compare --html`), `render_terminal` (used by `inspect --compare`), and `render_gradio` (embedded inside the viewer's Compare mode). Updating styling on a renderer updates both surfaces in lockstep. The legacy `viewer/diff.py` module is now reduced to the `tokenize()` helper used by `TextComparison` and the `publish/_stats.py` cross-model disagreement scorer.

The viewer code lives under [src/ocrscout/viewer/](src/ocrscout/viewer/):

- `store.py` — `ViewerStore`: loads the parquet via Polars, walks each `DoclingDocument` in body order via `iterate_items()` (so picture/table items stay interleaved with their surrounding text), exposes `pages()` / `get(page_id, model)` / `image_for(page_id)` / `annotated_for(page_id, model)`. Source images are resolved from the row's `source_uri` and LRU-cached.
- `app.py` — `build_app(output_dir) -> gr.Blocks`. Three control groups in the top bar (Page navigation / View mode / Layout source / Actions); a sticky image column with custom legend; a text pane re-rendered via `@gr.render` based on view mode and selected models.
- `static/viewer.css` and `static/viewer.js` — shipped via `force-include` in `pyproject.toml`. CSS handles the section-block coloring, sticky positioning, and control-group framing; JS provides keyboard shortcuts (j/k for prev/next page, 1/2/3 for view modes, i for image toggle), synchronized scroll across markdown panes, and URL ↔ state sync.

When changing the viewer: do not write code that auto-hides the image pane on view-mode change — the image stays visible across all modes (Single / Side-by-side / Compare), and only the user's manual Toggle image button hides it.

## Comparisons subsystem

Reference vs prediction agreement is a first-class concept, with three coupled abstractions in [src/ocrscout/interfaces/comparison.py](src/ocrscout/interfaces/comparison.py):

- **`Comparison`** — one analytic axis (text / document / layout / future semantic). Takes `PredictionView` + `BaselineView`, returns a typed `ComparisonResult` subclass or `None` (when required modality isn't available on both sides). Built-ins under [src/ocrscout/comparisons/](src/ocrscout/comparisons/).
- **`ComparisonResult`** — Pydantic discriminated union, ``comparison`` field is the literal name. Subclasses carry rich payload (diff opcodes, item-count breakdowns, per-category IoU); the ``summary: dict[str, float]`` projection lifts the most-queried numbers for cross-cutting consumers.
- **`ComparisonRenderer`** — terminal/HTML/Gradio surface for one result type. Lives in [src/ocrscout/comparisons/renderers/](src/ocrscout/comparisons/renderers/), registered under the `comparison_renderers` registry group. Updating styling in one renderer updates the inspector's `--compare --html` page and the viewer's Compare mode in lockstep.

**Reference is not ground truth.** Most references in the wild (BHL, IA legacy OCR) are themselves OCR output. `Reference.provenance` (`method`, `engine`, `confidence`) is mandatory in spirit and consumed by the inspector and Compare mode so users can interpret comparison numbers as agreement-vs-incumbent rather than accuracy-vs-truth. Reference adapters set provenance in their constructors (see [src/ocrscout/references/bhl_ocr.py](src/ocrscout/references/bhl_ocr.py): `method="ocr", engine="bhl-legacy"`).

**Storage.** Per-page `ExportRecord.comparisons: dict[str, ComparisonResult]` round-trips through the parquet's `comparisons_json` column; canonical metrics also lift into flat top-level columns (`text_similarity`, `text_cer`, `text_wer`, `document_*_delta`, `layout_iou_mean`) for SQL ergonomics. `reference_provenance_json` carries the typed provenance.

**Auto-firing.** When a reference adapter is configured and `--comparisons` is unset, the run loop runs every registered comparison whose `requires` set is satisfied. Pass `--comparisons text,layout` to whitelist or `--comparisons none` to skip. Optional CER/WER gated behind `pip install ocrscout[eval]` (jiwer).

**Entry-point groups**: `ocrscout.comparisons`, `ocrscout.comparison_renderers`. Built-ins are registered in [src/ocrscout/registry.py](src/ocrscout/registry.py); third-party packages add via these groups without touching ocrscout's source tree.

## BHL source adapter quirks

The [`bhl`](src/ocrscout/sources/bhl.py) source adapter samples pages by joining BHL's TSV catalogs in DuckDB, then fetches JP2 images from `s3://bhl-open-data/images/{BarCode}/{SequenceOrder:04d}.jp2` and OCR sidecars from `s3://bhl-open-data/ocr/item-{ItemID:06d}/item-{ItemID:06d}-{PageID:08d}-{SequenceOrder:04d}.txt`. Per BHL's own README, a tiny historical minority of items used `_0000.jp2` instead of `_0001.jp2` for the first leaf — empirically not encountered in random sampling (0/100). The adapter trusts the modern convention universally; if a truly-legacy item slips through, the image fetch 404s and the page is cleanly skipped with a warning. Image and OCR sidecars share the same NNNN suffix per item by construction (the OCR pipeline names its output after the input image), so they cannot drift apart within an item — only relative to BHL's logical SequenceOrder, which would shift both by one in the legacy case. If/when a legacy item is empirically observed and matters, the cleanest fix is consulting IA-style scandata XML (BHL hosts it under `bhl-open-data/scandata/`) for authoritative per-leaf metadata.

## Tests are paused

The project is in rapid-prototyping mode. The previous test suite was deleted (recoverable from git history if needed) and `pytest` is removed from the `dev` extras. Do not write new tests or run pytest until the user explicitly re-enables them.

## Implementation roadmap

1. **Skeleton**: public API, ABCs, three normalizers, working sources/references/exports, CLI stubs.
2. **Single-model run end-to-end** with `VllmBackend` (subprocess mode): manifest-based runner, output capture, normalization, export to parquet. First curated model: `dots-mocr`.
3. **Multi-model scouting**: parallel runs, per-model metrics, side-by-side report.
4. **Reference comparison**: ALTO/hOCR adapters; the `Comparison` subsystem (text/document/layout) is in place; remaining is corpus-level improvement/regression aggregation in a Reporter.
5. **Benchmarks**: MDPBench plugin (source + reference + comparisons + canonical score).
6. **Pipeline-mode YAML**: `ocrscout apply pipeline.yaml` with full DAG of stages.
7. **Additional backends**: extended `DoclingBackend` for non-SmolDocling VLMs, OpenAI-compatible (Ollama, LM Studio, Gemini, Claude), Tesseract.
8. **Ecosystem**: HF Hub publishing of results, VLM judge evaluator with ELO, more reporters (HTML, terminal, web).

## What NOT to do

- **Don't add GPU dependencies to the core `pyproject.toml`.** No `vllm`, no `torch`, no `transformers` in core. Those are concerns of `VllmBackend` subprocesses (`uv run --with vllm ...`), remote vLLM servers, or optional extras (`docling` ships Docling; `layout` ships `transformers` + `torch` for the PP-DocLayoutV3 detector). New extras are fine when they unlock a real workload; growing core is not.
- **Don't invent a custom document model.** Use `DoclingDocument` everywhere. Convert inputs to it as early as possible; convert outputs from it as late as possible.
- **Don't build a production OCR pipeline.** That's Docling's lane. ocrscout measures and compares; it does not productionize.
- **Don't add an auto-generated profile tier.** Profiles are curated only — every model has a hand-tuned YAML in `src/ocrscout/profiles/`. The `ocrscout introspect` command produces a draft to refine, not a profile to install.
- **Don't execute upstream HF scripts.** `introspect_hf_script` uses `ast.parse` only — never `exec`, never `importlib.import_module`. The HF scripts are reference material we read; ocrscout itself never runs them. (Same rule applies to `VllmBackend`: it talks to vLLM directly, never to the upstream script.)

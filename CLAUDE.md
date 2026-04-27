# CLAUDE.md

## Project overview

ocrscout is a toolkit for evaluating SOTA OCR models on user-supplied data and hardware. Inspired by the HuggingFace `uv-scripts/ocr` collection (which we read as reference material but no longer execute), ocrscout drives vLLM and Docling directly via curated per-model profiles, normalizes their varied outputs (markdown / DocTags / layout JSON) into a single `DoclingDocument` model, and adds the measurement layer (timing, throughput, edit distance, side-by-side comparison) those scripts lack. The package itself has **zero GPU dependencies** — all GPU work happens in `VllmBackend` subprocesses or remote vLLM servers.

## How to add a new component

ocrscout is extended through Abstract Base Classes and Python entry points.

1. Subclass the relevant ABC in [src/ocrscout/interfaces/](src/ocrscout/interfaces/):
   - `SourceAdapter` — yields `PageImage` objects from somewhere (a directory, an HF dataset, IIIF, PDF).
   - `ReferenceAdapter` — returns ground-truth text or `DoclingDocument` for a `page_id`.
   - `ModelBackend` — prepares a `BackendInvocation` and runs it, yielding `RawOutput`.
   - `Normalizer` — converts a `RawOutput` + `PageImage` into a `DoclingDocument`.
   - `ExportAdapter` — writes a stream of `ExportRecord` objects to a destination.
   - `Evaluator` — scores a prediction `DoclingDocument` against a `Reference`.
   - `Benchmark` — bundles a source, reference, and evaluator with a canonical scoring protocol.
   - `Reporter` — turns a results directory into a report (HTML, markdown, terminal).

2. Set the `name: ClassVar[str]` on your class — this is the lookup key in the registry.

3. Register your component one of two ways:
   - **In a downstream package**: declare an entry point in your `pyproject.toml`:
     ```toml
     [project.entry-points."ocrscout.normalizers"]
     my_normalizer = "my_pkg.normalizers:MyNormalizer"
     ```
     Entry-point groups: `ocrscout.sources`, `ocrscout.references`, `ocrscout.backends`, `ocrscout.normalizers`, `ocrscout.exports`, `ocrscout.evaluators`, `ocrscout.benchmarks`, `ocrscout.reporters`.
   - **In-process**: `from ocrscout import registry; registry.register("normalizers", "my_normalizer", MyNormalizer)`.

Built-in components are registered in [src/ocrscout/registry.py](src/ocrscout/registry.py) and are protected from being shadowed by third-party entry points.

## Model profiles

Each OCR model has a hand-curated YAML profile shipped at [src/ocrscout/profiles/](src/ocrscout/profiles/). There is no auto-generated tier and no synthesized fallback: if `ocrscout run --models X` can't find `X.yaml`, it errors with a `ProfileNotFound` and points the user at `ocrscout introspect`.

A profile records:

- **Identity**: `name`, `source` (which backend), `model_id`, `model_size`, `upstream_script` (informational pointer to the HF reference script).
- **Output shape**: `output_format`, `normalizer`, `has_bboxes`, `has_layout`, `category_mapping` (script-output-category → DoclingDocument label).
- **Prompts**: `prompt_templates` (mode → prompt string), `preferred_prompt_mode`, `chat_template_content_format`.
- **vLLM tuning**: `vllm_engine_args` (passed to `vllm.LLM(...)`), `sampling_args` (passed to `vllm.SamplingParams(...)`), `vllm_version`, `server_url` (for HTTP server mode).
- **Free-form**: `backend_args`, `metadata`.

### vLLM backend modes

Profiles with `source: vllm` can run in two modes:

- **Subprocess mode (default)** — `VllmBackend` spawns `uv run --with "vllm{vllm_version}" --with pillow runners/vllm_runner.py manifest.json out.jsonl`. The subprocess loads the model, processes the manifest, and writes per-page outputs to JSONL. The parent process never imports vLLM. Best for one-off runs and when the model is small enough that startup cost (~30–60s) is acceptable.
- **Server mode** — set the `--server-url` flag on `ocrscout run` (or the `OCRSCOUT_VLLM_URL` env var, or `server_url:` in the profile YAML; precedence: env > profile). The backend POSTs to the OpenAI-compatible `/chat/completions` endpoint of an externally-running `vllm serve` process. No vLLM dep in the parent or any subprocess. Best when iterating on the same model — pay startup cost once, then drive many runs.

  Example workflow:
  ```bash
  # Terminal 1 — start the server (one-time, leave running)
  vllm serve rednote-hilab/dots.mocr \
    --max-model-len 24000 --gpu-memory-utilization 0.8 \
    --trust-remote-code --port 8000

  # Terminal 2 — drive runs against it
  uv run ocrscout run --source ./images/ --models dots-mocr \
    --server-url http://localhost:8000/v1
  ```

  The base URL must include the OpenAI prefix (`/v1`); we don't auto-append because some deployments live behind a custom proxy path. Concurrency is controlled by `backend_args.concurrent_requests` on the profile (default 8 parallel POSTs); per-request timeout by `backend_args.request_timeout` (default 300s).

### Authoring a new profile (Claude-Code-assisted)

1. Run `uv run ocrscout introspect <name>` — fetches the upstream HF script (`uv-scripts/ocr/<name>.py`) and prints a draft YAML with TODO markers (model_id, vLLM engine args, sampling params, prompt template selection).
2. Pipe the draft to `src/ocrscout/profiles/<name>.yaml` and open it alongside the upstream script (cached at `~/.cache/ocrscout/uv-scripts-ocr/<name>.py`).
3. Ask Claude Code (or do it manually) to read the upstream source and resolve the TODOs: copy `PROMPT_TEMPLATES` verbatim, port the `LLM(...)` kwargs to `vllm_engine_args`, port `SamplingParams(...)` to `sampling_args`, decide `chat_template_content_format` based on how the script calls `llm.chat(...)`.
4. Validate with `uv run ocrscout run --source <fixture> --models <name>` on a small dataset.

Introspection is purely static (`ast.parse`) — the upstream script is never executed during this workflow.

## Key decisions

- **`DoclingDocument` is THE document model.** Never invent another. If it can't represent something, extend `docling-core`, don't fork.
- **Zero GPU deps in core.** `pyproject.toml` core dependencies must install in seconds on a CPU-only machine. GPU work belongs in `VllmBackend` subprocesses (`uv run --with vllm ...`), remote vLLM servers, or external Docling installs.
- **Pydantic v2** for all config, profiles, metrics, and IO. YAML round-trips through `model_validate` / `model_dump`.
- **`uv`** for environment management throughout. The CLI and the `VllmBackend` runner subprocesses both use it.
- **Entry points** are the extensibility contract. Downstream packages add sources, normalizers, evaluators, etc., without touching ocrscout's source tree.

## Tests are paused

The project is in rapid-prototyping mode. The previous test suite was deleted (recoverable from git history if needed) and `pytest` is removed from the `dev` extras. Do not write new tests or run pytest until the user explicitly re-enables them.

## Implementation roadmap

1. **Skeleton**: public API, ABCs, three normalizers, working sources/references/exports, CLI stubs.
2. **Single-model run end-to-end** with `VllmBackend` (subprocess mode): manifest-based runner, output capture, normalization, export to parquet. First curated model: `dots-mocr`.
3. **Multi-model scouting**: parallel runs, per-model metrics, side-by-side report.
4. **Reference comparison**: ALTO/hOCR adapters, `EditDistanceEvaluator`, improvement/regression rates.
5. **Benchmarks**: MDPBench plugin (source + reference + evaluator + canonical score).
6. **Pipeline-mode YAML**: `ocrscout apply pipeline.yaml` with full DAG of stages.
7. **Additional backends**: extended `DoclingBackend` for non-SmolDocling VLMs, OpenAI-compatible (Ollama, LM Studio, Gemini, Claude), Tesseract.
8. **Ecosystem**: HF Hub publishing of results, VLM judge evaluator with ELO, more reporters (HTML, terminal, web).

## What NOT to do

- **Don't add GPU dependencies to the core `pyproject.toml`.** No `vllm`, no `torch`, no `transformers`. Those are concerns of `VllmBackend` subprocesses (`uv run --with vllm ...`) or optional extras like `docling`.
- **Don't invent a custom document model.** Use `DoclingDocument` everywhere. Convert inputs to it as early as possible; convert outputs from it as late as possible.
- **Don't build a production OCR pipeline.** That's Docling's lane. ocrscout measures and compares; it does not productionize.
- **Don't add an auto-generated profile tier.** Profiles are curated only — every model has a hand-tuned YAML in `src/ocrscout/profiles/`. The `ocrscout introspect` command produces a draft to refine, not a profile to install.
- **Don't execute upstream HF scripts.** `introspect_hf_script` uses `ast.parse` only — never `exec`, never `importlib.import_module`. The HF scripts are reference material we read; ocrscout itself never runs them. (Same rule applies to `VllmBackend`: it talks to vLLM directly, never to the upstream script.)

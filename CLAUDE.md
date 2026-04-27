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

**GPU budget.** `--gpu-budget` (default 0.85) is the *ceiling* fraction of total VRAM the managed stack will collectively claim. ocrscout additionally clamps to 95% of currently-free VRAM via NVML, so a partly-busy GPU degrades gracefully rather than rejecting the run. The N managed models split the resulting effective budget equally; per-model `--gpu-memory-utilization` is logged on startup. If the per-model fraction would fall below vLLM's practical floor (0.1), the run is rejected with a clear "free up GPU memory or reduce model count" message.

**When to pick which mode.** Runner is right when you're scouting one model on a fresh dataset; the startup cost amortizes across the run. External-server is right when you're iterating on prompts/inputs against a single model and want to avoid paying startup per ocrscout invocation. Managed is right when you're comparing multiple models in one go (the proxy lets a single ocrscout run drive all of them) or when you want the convenience of "one command spins everything up and tears it down."

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

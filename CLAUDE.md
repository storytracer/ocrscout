# CLAUDE.md

## Project overview

ocrscout is a toolkit for evaluating SOTA OCR models on user-supplied data and hardware. Inspired by the HuggingFace `uv-scripts/ocr` collection (which we read as reference material but no longer execute), ocrscout drives vLLM, Docling, and hosted LLM APIs through a single LiteLLM proxy, normalizes their varied outputs (markdown / DocTags / layout JSON) into a single `DoclingDocument` model, and adds the measurement layer (timing, throughput, per-page cost, edit distance, side-by-side comparison) those scripts lack. The package itself has **zero GPU dependencies in core** — all GPU work happens in daemonised `vllm serve` subprocesses (spawned by `LocalRunner`) or remote workers (`SkyPilotRunner` / `HuggingFaceRunner`).

### Architecture in one diagram

```
ocrscout (CLI / stateless)
    │  (resolves Runner from ~/.ocrscout/state.yaml + --runner flag)
    ▼
Runner ABC ── local / skypilot / hf ──┐
    │                                 │
    │ launch() → daemonise stack      │
    ▼                                 ▼
LiteLLM proxy (localhost:4000) ───────────────────── (remote: SkyPilot pod / HF Job)
    ├── vLLM serve: dots-mocr     (localhost:8000)
    ├── vLLM serve: paddleocr-vl  (localhost:8001)
    ├── vLLM serve: glm-ocr       (localhost:8002)
    └── hosted: gemini-2.5-flash  (no local process; api.google.com)
    │
    │  litellm.completion(model=..., metadata={"page_id": …})
    ▼
LiteLLMBackend → cost.recorder ← litellm.success_callback (in-process)
                       │
                       ▼
            incremental Parquet (data/train-NNNNN.parquet)
              + progress.json checkpoint
```

Every OCR call flows through the LiteLLM proxy. This gives ocrscout a uniform cost-callback hook (`ocrscout.cost`) regardless of whether the model is vLLM-served locally or a hosted API, and the same code path scales from a workstation to a Kubernetes pool to HF-sponsored compute.

## How to add a new component

ocrscout is extended through Abstract Base Classes and Python entry points.

1. Subclass the relevant ABC in [src/ocrscout/interfaces/](src/ocrscout/interfaces/):
   - `SourceAdapter` — yields `PageImage` objects from somewhere (a directory, an HF dataset, IIIF, PDF). Concrete sources are Pydantic v2 classes — `class FooSourceAdapter(SourceAdapter, BaseModel)` — with structural constants on the class as `ClassVar`s, validation via `Field(...)` and `@model_validator(mode="after")`, and source-specific admin verbs declared as inner classes (see "Source admin subsystem" below). The shipped `hf_dataset` adapter at [src/ocrscout/sources/hf_dataset.py](src/ocrscout/sources/hf_dataset.py) covers local dirs, fsspec URLs (`s3://`, `gs://`, `https://`, `hf://`), and HF Hub repo IDs (`org/dataset`) through one code path — extend it (or write a new adapter) for IIIF, PDF rasterisation, etc. The [`bhl`](src/ocrscout/sources/bhl.py) adapter shows how to handle a catalog-driven corpus (305K volumes, 67M pages on S3) with cached metadata, per-source `info.yaml`, and inline `Setup`/`Refresh` actions. Sources may optionally implement `iter_volumes()` to yield typed `Volume` metadata that lands as a parallel `volumes-*.parquet` sidecar. Sources should also accept `start_idx` / `end_idx` constructor kwargs so SkyPilot/HF workers can take non-overlapping windows of a deterministic sample.
   - `SourceAction` — one admin verb for one source (e.g. `setup`, `refresh`, `stats`). Defined as an inner class on its `SourceAdapter`; lists itself on the adapter's `actions: ClassVar[list[type[SourceAction]]]`. CLI auto-generation walks `registry.list("sources")` and surfaces each as `ocrscout source <name> <verb>` via the Pydantic-flags-to-typer bridge at [src/ocrscout/sources/_flags_bridge.py](src/ocrscout/sources/_flags_bridge.py). See "Source admin subsystem" below.
   - `ReferenceAdapter` — returns a `Reference` (text and/or `DoclingDocument`, with typed `provenance`) for a `PageImage`. Note "reference", not "ground truth": most references in the wild are themselves OCR — see "Comparisons subsystem" below.
   - `ModelBackend` — prepares a `BackendInvocation` and runs it, yielding `RawOutput`. Built-ins: `litellm` (every OpenAI-compatible call routed through the LiteLLM proxy — covers both `runtime: vllm` and `runtime: hosted` profiles), `layout_chat` (per-region OCR over the same proxy), `docling`, `tesseract`. Backends that hit the proxy declare `requires_runner: ClassVar[bool] = True`.
   - `Runner` — orchestrates the compute stack (LiteLLM proxy + N vLLM backends) on whatever infrastructure the user picks. Built-ins: `local` (daemonised processes via [src/ocrscout/runners/local.py](src/ocrscout/runners/local.py)), `skypilot` (Kubernetes pool via [src/ocrscout/runners/skypilot.py](src/ocrscout/runners/skypilot.py)), `hf` (HuggingFace Jobs API via [src/ocrscout/runners/hf.py](src/ocrscout/runners/hf.py)). See "Runners" below.
   - `LayoutDetector` — emits typed `LayoutRegion`s (bbox + native category) for a `PageImage`. Consumed by layout-aware backends like `LayoutChatBackend`. Built-in: `pp-doclayout-v3` (transformers + torch via the `[layout]` extra). See "Layout-aware OCR" below.
   - `Normalizer` — converts a `RawOutput` + `PageImage` into a `DoclingDocument`.
   - `ExportAdapter` — writes a stream of `ExportRecord` objects to a destination. The built-in `parquet` adapter does incremental shard writes (one file per `batch_size` rows, default 1000) plus a `progress.json` checkpoint that powers `--resume`.
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
     Entry-point groups: `ocrscout.sources`, `ocrscout.references`, `ocrscout.backends`, `ocrscout.runners`, `ocrscout.normalizers`, `ocrscout.exports`, `ocrscout.comparisons`, `ocrscout.comparison_renderers`, `ocrscout.benchmarks`, `ocrscout.reporters`, `ocrscout.layout_detectors`. Source-specific admin verbs are not a separate entry-point group — they piggy-back on the source by being listed on the adapter's `actions` ClassVar.
   - **In-process**: `from ocrscout import registry; registry.register("normalizers", "my_normalizer", MyNormalizer)`.

Built-in components are registered in [src/ocrscout/registry.py](src/ocrscout/registry.py) and are protected from being shadowed by third-party entry points.

## Model profiles

Each OCR model has a hand-curated YAML profile shipped at [src/ocrscout/profiles/](src/ocrscout/profiles/). There is no auto-generated tier and no synthesized fallback: if `ocrscout run --models X` can't find `X.yaml`, it errors with a `ProfileNotFound` and points the user at `ocrscout introspect`.

A profile records:

- **Identity**: `name`, `model_id`, `model_size`, `upstream_script` (informational pointer to the HF reference script).
- **Routing**: `backend` (which `ModelBackend` class drives the call — any registered name; common values: `litellm`, `layout_chat`, `docling`, `tesseract`) and `runtime` (`vllm` | `hosted` | `cpu`, what infra the active `Runner` provisions for this model). These two fields are orthogonal: `runtime` says *what to launch*, `backend` says *how to call*. A locally-served vLLM VLM and a hosted Gemini API both go through the same `backend: litellm` code path; the Runner uses `runtime` to decide whether to spawn `vllm serve` or just add a hosted entry to the LiteLLM model_list.
- **Output shape**: `output_format`, `normalizer`, `has_bboxes`, `has_layout`, `category_mapping` (script-output-category → DoclingDocument label).
- **Prompts**: `prompt_templates` (mode → prompt string), `preferred_prompt_mode`, `chat_template_content_format`.
- **vLLM tuning** (only meaningful for `runtime: vllm`): `vllm_engine_args` (forwarded to `vllm serve` flags), `sampling_args` (passed via OpenAI top-level / `extra_body`), `vllm_version`, `server_url` (escape hatch for pre-existing external proxies; normally the active Runner sets `OCRSCOUT_VLLM_URL`).
- **Layout-aware orchestration** (consumed by `backend: layout_chat`): `layout_detector` (registered detector name, e.g. `pp-doclayout-v3`), `layout_detector_args` (constructor kwargs for the detector — `device`, `score_threshold`, `revision`, …), `prompt_mode_per_category` (detector-native category → `prompt_templates` key, for routing per-region prompts; falls back to `preferred_prompt_mode` for anything not listed).
- **Free-form**: `backend_args`, `metadata` (the `metadata.pricing` block becomes `model_info` entries in the LiteLLM proxy config — `input_cost_per_token` / `output_cost_per_token` for self-hosted models that aren't in LiteLLM's built-in pricing DB).

**Cross-field validators** on `ModelProfile` reject malformed YAML at load time:

- `backend: layout_chat` requires `layout_detector` set, `output_format == "layout_json"`, `normalizer == "layout_json"`, and `runtime == "vllm"` (the OSS VLMs it wraps need a local serve).
- `runtime: vllm` requires `vllm_engine_args.kv_cache_memory_bytes` so the Runner's KV-budget preflight has what it needs.
- `runtime: hosted` rejects any `vllm_engine_args` — those flags are meaningless for an API and signal a copy-paste from a vllm-runtime template.

### Profile defaults

Some `vllm_engine_args` keys are constant across every shipped profile and live in `DEFAULT_VLLM_ENGINE_ARGS` in [src/ocrscout/profile.py](src/ocrscout/profile.py) instead of being repeated in each YAML — currently `trust_remote_code: true` and `cudagraph_capture_sizes: [1, 2, 4, 8, 16, 24, 32]`. Every Runner that spawns `vllm serve` calls `effective_vllm_engine_args(profile)` to merge defaults under the profile's overrides (profile wins). Likewise, the `vllm_version` field defaults to `">=0.15.1"`. Don't restate any of these in a new profile unless you need to override.

## Runners

The `Runner` ABC ([src/ocrscout/interfaces/runner.py](src/ocrscout/interfaces/runner.py)) wraps all compute orchestration behind a single `launch / submit / status / logs / down` contract. Three concrete runners ship:

| Runner | Where work runs | When to use |
| --- | --- | --- |
| **`local`** | Daemonised processes on this machine (LiteLLM proxy + N `vllm serve` children, PID-tracked under `~/.ocrscout/pids/`, logs under `~/.ocrscout/logs/`). Survives terminal close, SSH disconnect, laptop sleep. | Workstations, dev iteration, single-GPU benchmarking |
| **`skypilot`** | A [SkyPilot](https://skypilot.readthedocs.io/) pool on Kubernetes (OVHcloud Managed K8s, AWS EKS, GCP GKE, on-prem). Each worker installs `ocrscout[vllm]` via `uv` and starts its own LiteLLM + vLLM stack locally. Job execution is managed by SkyPilot's controller (not the ocrscout CLI). | Distributed re-OCR, multi-GPU benchmarking, anything needing >1 GPU |
| **`hf`** | HuggingFace-sponsored compute via the HuggingFace Jobs API. Source / output use `hf://` paths so datasets stay on the Hub. | The BHL / FineBooks pilot |

Active runner state lives in `~/.ocrscout/state.yaml` (atomic write via tmp + rename). Every CLI command (`status`, `submit`, `logs`, `down`) reads this file fresh — there's no in-memory state between invocations, so the orchestrating CLI can crash/restart without losing track of in-flight work.

### CLI lifecycle

```bash
# Default: ephemeral. Ctrl-C reliably kills the entire stack — no state
# is left on disk. Pass --keep-up to use the persistent daemonised
# path so the stack outlives this invocation for follow-up `submit`s.
uv run ocrscout run --source ./images/ --models dots-mocr,glm-ocr-layout

# Stateful lifecycle. The CLI is stateless across commands — closing
# the terminal between them is fine; daemons survive.
uv run ocrscout launch --models dots-mocr,glm-ocr-layout
uv run ocrscout status                          # ready / launching / busy / down
uv run ocrscout submit --source s3://… --output ./out/ --pages 8000 --resume
uv run ocrscout logs <job-id>                   # tails worker.log
uv run ocrscout down                            # SIGTERM each recorded PID
uv run ocrscout down --force                    # …plus port-scan for orphans

# Switch runner with --runner. State file records which one is active.
uv run ocrscout launch --runner skypilot --gpu L4 --workers 3 \
  --models dots-mocr
uv run ocrscout submit --source s3://… --output s3://… --pages 8000 --num-jobs 10
```

### LocalRunner lifecycle: persistent vs ephemeral

[src/ocrscout/runners/local.py](src/ocrscout/runners/local.py) drives the local stack in one of two modes, selected by the `persistent: bool = True` kwarg on `LocalRunner.launch()`:

**Persistent** (`ocrscout launch`, `ocrscout run --keep-up`). Each child is spawned via classic UNIX double-fork in [src/ocrscout/runners/_daemon.py](src/ocrscout/runners/_daemon.py) (parent → first child `setsid`s + forks grandchild → grandchild `execvp`s the target). The grandchild is reparented to init/launchd so it survives terminal close; the OS auto-reaps on exit (no zombies; `pid_alive()` reports accurate state). PIDs land atomically in `~/.ocrscout/pids/<name>.pid`; stdout/stderr go to `~/.ocrscout/logs/<name>.log`. The lifecycle is recorded in `state.yaml` via a `phase` field:

1. **Preflight done.** Write `state.yaml` with `phase: launching` + empty `processes` BEFORE any spawn. The crash breadcrumb — anything short of `phase: ready` on disk means the launcher didn't finish.
2. **Per-daemon spawn + readiness + PID correction.** For each `vllm serve` and the LiteLLM proxy: daemonise → wait for `/v1/models` (or `/health/liveliness` for the proxy) → query the **actual** port-listener PID via [src/ocrscout/runners/_ports.py](src/ocrscout/runners/_ports.py)`:resolve_listener_pid` and rewrite the recorded PID + PID file. The correction step is load-bearing because `uv run --with vllm -- vllm serve` exits its transient resolver subprocess before the real Python server is up; the PID `execvp` recorded points at a dead transient by the time `/v1/models` answers. After correction, the `ManagedProcess` is appended to `state.processes` and the state file is atomically rewritten (incremental per-daemon).
3. **Atomic flip to ready.** After every health check passes and every PID is corrected, `state.yaml` is rewritten with `phase: ready` — the only success indicator.

PR_SET_PDEATHSIG is **not** used on the persistent path — by design the daemons outlive the launching ocrscout process. `ocrscout status` reading `phase: launching` with a `phase_updated_at` older than ~15 min (1.5× the default ready_timeout) flags a crashed launcher and recommends `ocrscout down --force`.

**Ephemeral** (`ocrscout run` default, `ocrscout apply` as the SkyPilot/HF worker entry). Each child is spawned by [src/ocrscout/runners/_ephemeral.py](src/ocrscout/runners/_ephemeral.py)`:EphemeralStack` via `subprocess.Popen`. Four layers of defense in depth tie child fate to this process:

1. **`PR_SET_PDEATHSIG=SIGTERM`** set in each child's preexec_fn via ctypes-bound `prctl`. The kernel SIGTERMs the child the instant its parent dies. Linux only — no-op on macOS.
2. **`start_new_session=True`** on every Popen so the controlling terminal's Ctrl-C doesn't double-deliver. The orchestrator owns the signal and propagates explicit termination.
3. **`atexit` registration** that calls `terminate_all()` (reverse-order SIGTERM → 10s grace → SIGKILL on each process group). Catches clean exits from non-signal paths.
4. **SIGINT/SIGTERM/SIGHUP handlers** that call `terminate_all()` synchronously before re-raising. Handlers are installed lazily on first spawn and restored on teardown.

The ephemeral path writes **nothing** under `~/.ocrscout/state.yaml` or `~/.ocrscout/pids/`. Log files still land in `~/.ocrscout/logs/` so `ocrscout logs` can tail them while the run is live. As a direct consequence, `ocrscout submit` after an ephemeral `run` cleanly errors with "no active stack" — submit is stateful and ephemeral runs opt out.

This bifurcation closes a class of bugs where Ctrl-C during a `ocrscout run` left orphaned vLLM children holding GPU memory: with PDEATHSIG, the leaf vllm-serve dies regardless of which intermediate `uv run` wrapper Python recorded as the immediate child.

**Reuse**: `LocalRunner.launch(models=…)` reads `state.yaml` first. If a persistent stack is already up and matches the requested models + proxy port (and `phase == "ready"`), it returns the existing handle without spawning anything — for both the persistent and ephemeral branches. `ocrscout run` skips its `finally` teardown via `reused_existing` so a `launch` followed by `run` doesn't leave the stack down. A mismatch errors out with a hint to call `ocrscout down`.

**Teardown** (`down()`): the in-memory `EphemeralStack` is terminated first when present (ephemeral path). Otherwise the persistent path flips `phase: tearing_down`, then walks `state.processes` in reverse order (proxy first, then upstreams) SIGTERMing each PID's process group with 10s grace + SIGKILL fallback, clears PID files, and removes `state.yaml`. With `--force`, an additional belt-and-suspenders port scan (`_down_by_port_scan`) enumerates listeners on the LiteLLM (4000) and vLLM (8000-8031) port ranges via [_ports.py](src/ocrscout/runners/_ports.py)`:listeners_on_ports` and SIGTERMs each one. `--force` works whether or not `state.yaml` exists; use it after a Ctrl-C'd run, an OOM-killed orchestrator, or any suspected leak.

### Per-profile KV cache budgets

Each `runtime: vllm` profile must declare an absolute KV cache budget via `vllm_engine_args.kv_cache_memory_bytes` (vLLM's `--kv-cache-memory-bytes` flag; SI suffixes accepted — `16G` = 16 GiB, `25.6m` = 25,600,000 bytes). When set, vLLM allocates exactly that much for KV and ignores `--gpu-memory-utilization` — which eliminates the cudaMemGetInfo race that previously plagued parallel sibling spawns and gives every engine a deterministic, predictable footprint. Sizing rule of thumb: `kv_cache_memory_bytes ≈ concurrent_requests × max_model_len × ~30_000` bytes per token (the per-token cost depends on architecture; profile and adjust).

A `ModelProfile` validator rejects `runtime: vllm` profiles missing this field at load time, before any subprocess spawns.

### GPU budget preflight

`--gpu-budget` (default 0.85) is the collective ceiling fraction of total VRAM the stack may claim — it's a sanity check, not a splitter. At spawn time, `_preflight_kv_budgets` sums each profile's `kv_cache_memory_bytes` plus a per-profile overhead estimate (model weights from `model_size` × bytes-per-param from `dtype`, plus a 25%-of-weights / 1 GiB-min working slack for activations + cudagraphs + allocator), and rejects the run if the total would exceed `min(total_VRAM × gpu_budget, free_VRAM × 0.95)`. The error message names each profile's KV + overhead contribution so you can see which to shrink. If `model_size` is missing or unparseable, ocrscout falls back to a flat 8 GiB conservative estimate.

### Sequential model execution by default

`--parallel-models` defaults to `1` even when multiple models are launched: vLLM servers all spawn in parallel (one startup cost), but page requests run against one server at a time. On a single GPU, concurrent execution just halves throughput per model and yields contended s/page numbers that aren't useful for benchmarking. Sequential gives each model the full GPU, produces honest per-model timings, and total wall-clock is ~equivalent (the GPU is the bottleneck either way). Override with `-P > 1` only if you have separate GPUs per model.

### SkyPilotRunner notes

[src/ocrscout/runners/skypilot.py](src/ocrscout/runners/skypilot.py) is a thin wrapper over the `sky` CLI (lazy-imported on first call so the orchestrating workstation doesn't need a GPU). `launch` generates a pool YAML, `submit` generates a per-job YAML and shells out to `sky jobs launch --pool`. Each worker's setup script does:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --system "ocrscout[vllm]"
```

…and the run block invokes `ocrscout apply config.yaml --start-idx $LO --end-idx $HI` — the `apply` command is the worker-side entry point that loads a serialized PipelineConfig + auto-launches its own worker-local LocalRunner stack. Work is partitioned by deriving `--start-idx` / `--end-idx` from `$SKYPILOT_JOB_RANK` and `$SKYPILOT_NUM_JOBS`.

GPU pricing for the `gpu_type` column lives in [src/ocrscout/runners/_gpu_pricing.py](src/ocrscout/runners/_gpu_pricing.py); the SkyPilot job YAML's `envs:` block stamps `OCRSCOUT_GPU_TYPE` / `OCRSCOUT_COST_PER_HOUR` / `OCRSCOUT_PROVIDER` so each worker writes self-describing Parquet rows.

### HuggingFaceRunner notes

[src/ocrscout/runners/hf.py](src/ocrscout/runners/hf.py) uses `huggingface_hub.HfApi.run_job(...)` to submit each pipeline as an HF Job. Same `apply` command on the worker side; same env-var pattern for cost context. The HF Jobs API is still maturing — keyword arguments may shift between SDK versions, so this runner catches `Exception` from `run_job` and surfaces actionable error messages.

### High-resolution images are a known soft spot

ocrscout's preflight is sizing-blind to image dimensions: it sizes KV/overhead for "typical" workloads (~256 vision tokens per MP at default Qwen2-VL-style processor settings). With much larger inputs (>4 MP), four things degrade in this order: (1) `prompt_tokens + max_tokens > max_model_len` → vLLM returns HTTP 400 and the page is recorded as `pages_failed`; (2) per-request KV demand grows linearly, draining the cache and pushing requests into vLLM's `Waiting:` queue; (3) vision-encoder transient activations may spike past the per-engine cap and OOM mid-run (preflight won't catch this — the cap covers steady-state, not the encoder peak); (4) vLLM's encoder cache budget (~14400 tokens by default) overflows and visual features get recomputed per request, slowing prefill. None of this surfaces as a clear up-front error today. If/when this bites, the cleanest defense is `vllm_engine_args.mm_processor_kwargs.max_pixels: <N>` to cap input resolution at the source — bounds prefill, KV use, and encoder memory in one knob with a predictable accuracy trade-off. Other levers: bump `max_model_len`, lower `sampling_args.max_tokens`, raise `kv_cache_memory_bytes`. Worth revisiting when scouting tabloid-size scans, large maps, or anything substantially above 4 MP.

## Cost tracking

LiteLLM is in the request path for every backend call (whether the model is a local vLLM-served VLM or a hosted API). [src/ocrscout/cost.py](src/ocrscout/cost.py) registers an in-process `success_callback` on first use; backends pass `metadata={"page_id": page_id, "model_name": profile.name}` to `litellm.completion(...)` so the callback can correlate each request to its page. `cost.recorder` is a thread-safe per-page accumulator (a page that issues multiple requests — e.g. `layout_chat`'s per-region fan-out — sums into one record).

After a model's `_execute` loop finishes a page, `cli/run.py` calls `cost.recorder.close_page(page_id)` and stamps the result onto the `ExportRecord` along with the active `GpuConfig` (gpu_type / provider / cost_per_hour from `~/.ocrscout/config.yaml` or env vars). The Parquet schema carries these as flat top-level columns (see [src/ocrscout/exports/schema.py](src/ocrscout/exports/schema.py)):

| Column | Source |
|---|---|
| `gpu_type`, `provider`, `cost_per_hour` | dispatcher startup (env / config) |
| `elapsed_seconds` | LiteLLM callback (`end_time - start_time`) |
| `input_tokens`, `output_tokens` | LiteLLM response `usage` |
| `litellm_cost` | `litellm.completion_cost(completion_response=…)` — uses LiteLLM's built-in pricing DB for commercial APIs, `model_info` overrides in `~/.ocrscout/litellm.yaml` for self-hosted vLLM |
| `gpu_time_cost` | `elapsed_seconds / 3600 × cost_per_hour` |

`ocrscout costs` ([src/ocrscout/cli/costs.py](src/ocrscout/cli/costs.py)) aggregates these via DuckDB over `<output>/data/train-*.parquet`. Works on local paths, `s3://`, `gs://`, and `hf://`.

## Incremental Parquet + checkpoint/resume

[src/ocrscout/exports/parquet.py](src/ocrscout/exports/parquet.py) auto-flushes the row buffer every `batch_size` rows (default 1000) to `data/train-NNNNN.parquet`. A sibling `<output>/progress.json` records every flushed `page_id` atomically (tmp + rename). On `ocrscout submit --resume` (or `ocrscout apply --resume` on a SkyPilot/HF retry), the dispatcher loads `progress.json`, filters out already-done page ids from the source's deterministic sample, and only dispatches what remains. Partial output is always readable; a crash loses at most one un-flushed batch.

Existing readers (inspect, viewer, publish) already glob `data/train-*.parquet` so they see all shards transparently.

### Layout-aware OCR (`layout_chat` backend)

Single-task-per-call OCR models (GLM-OCR, PaddleOCR-VL, …) take one prompt prefix and do exactly one thing — text, table, formula, or chart extraction — per call. Run them in default `ocr` mode against a mixed-content page and tables come back as flat prose because that's literally what "Text Recognition:" returns. The `layout_chat` backend (registered as `backend: layout_chat`) wraps these with an external layout detector: detect typed regions on the page → for each region, crop the source image and call `litellm.completion(...)` with the prompt that matches the region's category → assemble the per-region results into one `layout_json` payload that the existing `LayoutJsonNormalizer` turns into a structured `DoclingDocument` with proper `TableItem`s and bbox provenance.

| Knob | Where | What it does |
| --- | --- | --- |
| `layout_detector` | profile YAML | Registered `LayoutDetector` name (built-in: `pp-doclayout-v3`). |
| `layout_detector_args` | profile YAML | Detector kwargs (`device`, `score_threshold`, `revision`, …). |
| `prompt_mode_per_category` | profile YAML | Detector-native category → `prompt_templates` key. Lookup is on the **detector's** raw label (before `category_mapping` is applied). Anything not listed falls through to `preferred_prompt_mode`. |
| `backend_args.region_concurrency` | profile YAML | Parallel POSTs per page (default 16). Pages stay sequential within a profile. |

**Requires a Runner.** `layout_chat.prepare()` fails fast if `OCRSCOUT_VLLM_URL` isn't set (i.e. no Runner has launched a LiteLLM proxy for it). The backend declares `requires_runner: ClassVar[bool] = True`; backends that touch the proxy use this flag, and `ModelProfile`'s validator pairs the `backend: layout_chat` check with `runtime: vllm` (the OSS VLMs it wraps need a local serve). Subprocess-mode vLLM is unsupported because the per-region launch cost would dwarf any benefit.

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
3. Ask Claude Code (or do it manually) to read the upstream source and resolve the TODOs: set `backend: litellm` + `runtime: vllm` (or `backend: layout_chat` for a layout-aware variant); copy `PROMPT_TEMPLATES` verbatim; port the `LLM(...)` kwargs to `vllm_engine_args`; port `SamplingParams(...)` to `sampling_args`; decide `chat_template_content_format` based on how the script calls `llm.chat(...)`. Pick an absolute `kv_cache_memory_bytes` (start from `concurrent_requests × max_model_len × ~30_000` bytes and round to a friendly number like `16G`) — the `runtime: vllm` validator requires this. Skip `trust_remote_code`, `cudagraph_capture_sizes`, `vllm_version` — those have defaults (see "Profile defaults" above) and should only be set to override.
4. Validate with `uv run ocrscout run --source <fixture> --models <name>` on a small dataset.

Introspection is purely static (`ast.parse`) — the upstream script is never executed during this workflow.

## Key decisions

- **`DoclingDocument` is THE document model.** Never invent another. If it can't represent something, extend `docling-core`, don't fork.
- **Zero GPU deps in core.** `pyproject.toml` core dependencies must install in seconds on a CPU-only machine. GPU work belongs in daemonised `vllm serve` children (spawned by `LocalRunner`) or remote workers (`SkyPilotRunner` / `HuggingFaceRunner` — each worker installs `ocrscout[vllm]` via `uv pip install --system` in its setup script).
- **LiteLLM proxy is always in the request path.** Every OCR call goes through `litellm.completion(...)` against a Runner-managed proxy URL, so the cost callback wires per-page tokens / elapsed / dollar cost into Parquet uniformly — local vLLM and hosted APIs flow through the same code path.
- **Pydantic v2** for all config, profiles, metrics, and IO. YAML round-trips through `model_validate` / `model_dump`.
- **`uv`** for environment management throughout. The CLI uses it locally; remote workers install ocrscout into a system Python via `uv pip install --system` so subsequent invocations don't pay env setup cost.
- **`~/.ocrscout/` for runtime state.** Stateless CLI design — every command re-reads `state.yaml` rather than holding in-memory state between invocations. Daemons survive terminal close; commands can be retried after a crash without losing track.
- **Entry points** are the extensibility contract. Downstream packages add sources, runners, normalizers, evaluators, etc., without touching ocrscout's source tree.

## Logging

All status output flows through Python's stdlib `logging` module under the `ocrscout` namespace, rendered by a plain `StreamHandler` configured via [src/ocrscout/log.py:setup_logging](src/ocrscout/log.py). No Rich, no markup parsing, no width-based wrapping — one logical line per record so the output is grep-friendly and pastes cleanly into bug reports. CLI commands accept `-v` / `-vv` / `-q` to control verbosity:

| Flag | Level | Format | What appears |
| --- | --- | --- | --- |
| `-q` | WARNING | bare message | Errors, warnings, summary table, ready banners |
| (default) | INFO | bare message | + per-page progress, per-model start/done, GPU allocation summary |
| `-v` | VERBOSE (15) | `HH:MM:SS LEVEL message` | + timestamps and level names, full URLs/paths, GPU per-process telemetry |
| `-vv` | DEBUG | `HH:MM:SS LEVEL name:line  message` | + source paths, subprocess argvs, every nvitop probe |

**When to log vs. rprint.** Use `log.info(...)` (or `log.debug` / `log.log(VERBOSE, ...)`) for events — anything with a level that should be filterable. Reserve `rich.print` for *presentation* artifacts that should always render regardless of verbosity: the ready banner from `ocrscout launch`, the per-model summary table from `ocrscout run`, the cost table from `ocrscout costs`. The rule of thumb: if a user's `--quiet` invocation should still see it, it's presentation and uses `rich.print`; otherwise, it's an event and goes through the logger. Don't put Rich markup tags (`[bold]`, `[red]`, etc.) in log messages — the stdlib formatter won't parse them and they'd appear as literal characters.

**Per-model prefix.** When a backend processes pages for a model, prefix every log line with `[{profile.name}]`. This is the contract that lets parallel-model output (`--parallel-models > 1`) interleave readably. See [src/ocrscout/backends/litellm.py](src/ocrscout/backends/litellm.py) for the pattern.

**Daemon output.** Daemonised processes (vLLM serves + LiteLLM proxy) redirect stdout/stderr to per-name log files under `~/.ocrscout/logs/`. `ocrscout logs` tails them (multiplexed with `[name]` prefixes) with optional `--no-follow` for a snapshot. SkyPilot/HF runners delegate to their underlying `sky jobs logs` / `huggingface_hub.stream_job_logs` for remote workers.

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

## Source admin subsystem

Sources have admin concerns that aren't part of the per-page iteration contract — cache provisioning, derived-artifact construction, freshness checks, classifier pipelines. These live as `SourceAction` subclasses **inline on the source adapter** (not in sibling modules) and surface as `ocrscout source <name> <verb>`. The mechanics:

- **Per-source `info.yaml`** at `~/.ocrscout/sources/<name>/info.yaml` — a typed Pydantic record (schema in [src/ocrscout/sources/_info.py](src/ocrscout/sources/_info.py)) of what's been provisioned: cache freshness (ETags + `last_refresh` timestamps), pointers to derived HF datasets, intermediate-parquet metadata. Top-level `extra="ignore"` so an older ocrscout reading a newer `info.yaml` survives unknown sections; per-section `extra="forbid"` catches typos.
- **Per-source cache** at `~/.ocrscout/sources/<name>/{catalog,derived}/`. `catalog/` holds raw upstream data; `derived/` holds intermediate artifacts produced by refresh (e.g. BHL's `volumes.parquet` pre-join). Paths come from [src/ocrscout/sources/_paths.py](src/ocrscout/sources/_paths.py).
- **`SourceAction` ABC** at [src/ocrscout/interfaces/source_action.py](src/ocrscout/interfaces/source_action.py). Each action declares a Pydantic `Flags` model; the CLI driver at [src/ocrscout/cli/source.py](src/ocrscout/cli/source.py) auto-generates a typer command from the model via [_flags_bridge.py](src/ocrscout/sources/_flags_bridge.py). Action lifecycle (driver-managed): load `info.yaml` → `action.fill_defaults(flags, info)` → `action.run(flags, ctx)` → atomic merge of the returned patch dict back into `info.yaml` (locked via `fcntl.flock` on a sibling `.lock` file).
- **No hardcoded HF dataset names** anywhere in `sources/`. The user types the dataset id at `setup` or `refresh`; `info.yaml` stores the choice. The error path when nothing is configured names the relevant CLI commands but no specific dataset.
- **Universal verbs** `info` and `clear` live in the CLI layer and apply to every source automatically.

## BHL source adapter

The [`bhl`](src/ocrscout/sources/bhl.py) adapter samples pages by joining BHL's catalog parquets in DuckDB, then fetches JP2 images from `s3://bhl-open-data/images/{BarCode}/{SequenceOrder:04d}.jp2` and OCR sidecars from `s3://bhl-open-data/ocr/item-{ItemID:06d}/item-{ItemID:06d}-{PageID:08d}-{SequenceOrder:04d}.txt`. Image and OCR sidecars share the same NNNN suffix per item by construction. Per BHL's own README, a tiny historical minority of items used `_0000.jp2` instead of `_0001.jp2` for the first leaf — empirically not encountered in random sampling (0/100). The adapter trusts the modern convention universally; if a truly-legacy item slips through, the image fetch 404s and the page is cleanly skipped with a warning. If/when a legacy item is empirically observed and matters, the cleanest fix is consulting IA-style scandata XML (BHL hosts it under `bhl-open-data/scandata/`) for authoritative per-leaf metadata.

### Lifecycle

`BhlSourceAdapter` is a Pydantic v2 class with all structural constants as `ClassVar`s (`BUCKET`, `DATA_PREFIX`, `IMAGES_PREFIX`, `JP2_PATTERN`, `CATALOG_FILES`, `COPYRIGHT_PARQUET_PATH`, `COPYRIGHT_JOIN_COLUMNS`, plus the `CLASSIFIER_*` invocation contract). Iteration requires the cache to be provisioned by the inline admin verbs:

| Verb | What it does |
|---|---|
| `ocrscout source bhl setup --read-from <dataset>` | Records the rights-classification dataset to read at sample time. Lightweight — no upstream sync. |
| `ocrscout source bhl refresh --only catalog` | Re-fetches the three upstream TSVs (`item.txt.gz`, `page.txt.gz`, `title.txt.gz`) with per-file ETag invalidation and converts each one to `catalog/{item,page,title}.parquet` (ZSTD-compressed, `all_varchar`). The TSVs are dropped after conversion; ETag tracking moves to the parquet's sibling `.etag` file. Pure local. Triggers a `derived/volumes.parquet` rebuild iff a rights source is already configured. |
| `ocrscout source bhl refresh --runner local` | Fully local rights pipeline. Extracts unique 4-field combos from `item.parquet`, runs the bundled [`_bhl_classify_local.py`](src/ocrscout/sources/_bhl_classify_local.py) PEP-723 UV script (parquet → parquet) on this machine, writes `derived/rights_classified.parquet`, and rebuilds `derived/volumes.parquet`. No HuggingFace Hub round-trip — `info.rights.local_parquet` records the path. |
| `ocrscout source bhl refresh --runner hf --source-repo <repo> --output-repo <repo>` | HF-Jobs rights pipeline. Extract combos → push to `--source-repo` → classify via the upstream `uv-scripts/classification` UV script on HF Jobs → publish to `--output-repo` → rebuild `derived/volumes.parquet`. `source_repo`/`output_repo` are sticky in `info.yaml`. Needs `HF_TOKEN` in the env. |
| `ocrscout source bhl info` / `clear` | Universal: render state / wipe the source dir. |

`iter_pages` errors out if `derived/volumes.parquet` is missing or older than `catalog.last_refresh` — no silent rebuilds; the user runs refresh explicitly.

### Sampling query

[BhlSourceAdapter._run_duckdb_sample_from_volumes](src/ocrscout/sources/bhl.py) reads the pre-joined `derived/volumes.parquet` (~300K rows, ~14MB compressed), applies user filters (rights, languages, year_range) against that small table, ranks each surviving volume by `hash(ItemID || seed)`, ranks pages within each volume the same way, then joins `catalog/page.parquet` only for surviving ItemIDs and fills `sample_n` rows volume-by-volume up to `pages_per_volume` per volume. This replaces the previous 4-way TSV+parquet join that ran on every iteration.

### Classifier dispatch

The rights-classification pipeline takes one of two paths depending on `--runner`:

* **`--runner local`** calls [`classify_parquet`](src/ocrscout/sources/_bhl_classify_local.py) **in-process** in the current Python interpreter. Parquet in (`derived/_rights_combos.parquet`), parquet out (`derived/rights_classified.parquet`) — no HuggingFace Hub involvement, no UV-managed throwaway env, no subprocess. The user installs `ocrscout[vllm]` into their main environment (`vllm` + `transformers` + `pyarrow`); `torch` is deliberately left out of the extra because DGX-class ARM64+CUDA hardware needs a wheel from Nvidia's index that pypi can't deliver. On import failure `classify_parquet` raises `ScoutError` with the install hint. Going in-process sidesteps the `vllm._C` → `libtorch_cuda.so` linkage break that bit us when `uv run` installed a stock vLLM wheel into a fresh env.
* **`--runner hf`** shells out to the community-maintained `https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py` via [src/ocrscout/jobs.py](src/ocrscout/jobs.py)'s `run_uv_script(runner="hf")`. Input/output are HF datasets (`--source-repo` / `--output-repo`); needs `HF_TOKEN` to push.

The classifier model (`Qwen/Qwen3-4B`), labels, prompt, max-tokens, and HF Jobs flavor all live on `BhlSourceAdapter` as `CLASSIFIER_*` ClassVars — the invocation contract is the BHL adapter's, the local and HF paths are interchangeable downstream because both write the same `classification` / `parsing_success` / `reasoning` columns.

### Partitioning for SkyPilot / HF workers

The adapter accepts `start_idx` / `end_idx` constructor kwargs that slice the deterministic sample after the DuckDB rank query but before fetching. With every worker receiving the same `(sample, seed, …)` plus a distinct `[start_idx, end_idx)` window, the union is the full deterministic sample with no overlaps. SkyPilot's job YAML computes these from `$SKYPILOT_JOB_RANK` / `$SKYPILOT_NUM_JOBS` in the worker run script (see [src/ocrscout/runners/skypilot.py](src/ocrscout/runners/skypilot.py)).

## Tests are paused

The project is in rapid-prototyping mode. The previous test suite was deleted (recoverable from git history if needed) and `pytest` is removed from the `dev` extras. Do not write new tests or run pytest until the user explicitly re-enables them.

## Implementation roadmap

**Done:**

1. **Skeleton**: public API, ABCs, normalizers, working sources/references/exports, CLI stubs.
2. **Single + multi-model scouting**: `LiteLLMBackend` over a LiteLLM proxy, layout-aware OCR via `layout_chat`, per-page metrics + comparison subsystem.
3. **Runner abstraction**: `Runner` ABC; `LocalRunner` with proper daemonisation; `SkyPilotRunner` for K8s pools; `HuggingFaceRunner` for HF Jobs.
4. **Cost tracking**: in-process LiteLLM `success_callback` → per-page `litellm_cost` / `gpu_time_cost` / tokens / `elapsed_seconds` columns; `ocrscout costs` for DuckDB aggregation.
5. **Incremental Parquet + checkpoint/resume**: `data/train-NNNNN.parquet` shards + `progress.json` checkpoint; `--resume` filters done pages.
6. **Flat CLI**: `launch / submit / status / down / logs / run / benchmark / costs / apply` plus the existing `inspect / viewer / introspect / publish`. **Two-level for source admin**: `source <name> <verb>` (`setup`, `refresh`, `info`, `clear`, …) — see "Source admin subsystem".

**Open:**

7. **Reference comparison ecosystem**: ALTO/hOCR adapters; corpus-level improvement/regression aggregation in a Reporter.
8. **Benchmarks**: MDPBench plugin (source + reference + comparisons + canonical score), an opinionated `ocrscout benchmark` matrix that fans out across cloud GPU pools via `SkyPilotRunner`.
9. **Additional backends**: extended `DoclingBackend` for non-SmolDocling VLMs, Tesseract polishing.
10. **Ecosystem**: HF Hub publishing of results, VLM-judge evaluator with ELO, more reporters (HTML, terminal, web).

## What NOT to do

- **Don't add GPU dependencies to the core `pyproject.toml`.** No `vllm`, no `torch`, no `transformers` in core. Those are concerns of GPU workers (which install `ocrscout[vllm]` via `uv pip install --system` from their launch script), or optional extras (`docling` ships Docling; `layout` ships `transformers` + `torch` for the PP-DocLayoutV3 detector). `litellm` (without `[proxy]`) is in core because every backend hits it for cost tracking; `litellm[proxy]` is in the `[serve]` extra (it ships the proxy binary that `LocalRunner` spawns). New extras are fine when they unlock a real workload; growing core is not.
- **Don't invent a custom document model.** Use `DoclingDocument` everywhere. Convert inputs to it as early as possible; convert outputs from it as late as possible.
- **Don't bypass the LiteLLM proxy.** Every OCR call should flow through `litellm.completion(model=..., api_base=proxy_url, metadata={"page_id": ...})` so the in-process success callback captures cost / tokens / elapsed correctly. Going direct to vLLM HTTP loses the cost callback and breaks the cost columns.
- **Don't build a production OCR pipeline.** That's Docling's lane. ocrscout measures and compares; it does not productionize.
- **Don't add an auto-generated profile tier.** Profiles are curated only — every model has a hand-tuned YAML in `src/ocrscout/profiles/`. The `ocrscout introspect` command produces a draft to refine, not a profile to install.
- **Don't execute upstream HF scripts.** `introspect_hf_script` uses `ast.parse` only — never `exec`, never `importlib.import_module`. The HF scripts are reference material we read; ocrscout itself never runs them.

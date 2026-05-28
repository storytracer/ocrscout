# ocrscout

**Find the right OCR model for your documents**

![ocrscout viewer comparing dots-mocr and glm-ocr-layout side-by-side](screenshot.jpg)

## Why this exists

You have a stack of documents you wish were searchable text. Letters from grandparents. Family recipe cards. Tax receipts you're legally required to keep. Council minutes from 1880. Census records from a genealogy hunt. A bookcase of out-of-copyright novels. Pages from your local newspaper's archive. Doctors' notes. Scientific papers.

For most of the last twenty years OCR meant [Tesseract](https://github.com/tesseract-ocr/tesseract). It's still fine on clean modern print but chokes on anything awkward — smudged pages, multi-column newspapers, mixed languages, tables, old typewriter fonts, handwriting.

In the last two years a wave of vision-language models has reset the bar. They follow reading order across complicated layouts, recover tables, recognise mathematical formulas, and handle handwriting that classical OCR gave up on. Dozens are now public — dots-OCR, GLM-OCR, LightOnOCR, SmolDocling, PaddleOCR-VL — and a new one ships every few weeks.

So which one should you use? *It depends on your documents.* A model that wins on modern PDFs may stumble on a Victorian novel. One that handles English newspapers beautifully may butcher a 1900 birth record in German blackletter. One that aces tables in scientific papers may scramble the columns of your grandmother's recipe cards. Public benchmarks don't tell you what works on **your** corpus.

Finding out is real work. For each candidate you have to:

- Read its docs and figure out how to install it
- Configure vLLM (or whatever runtime it uses) for your hardware
- Write glue code to feed your images through it
- Parse its custom output into something you can compare against the others
- Time it, count failures, eyeball the quality, decide if it's worth using

A few hours per model. A week for five. By then somebody has released a sixth.

## What ocrscout does

ocrscout turns "compare these models on my documents" into one command:

```bash
uv run ocrscout run --source ./my-documents/ --models dots-ocr,glm-ocr-layout,paddleocr-vl-layout
```

Behind the scenes:

- **Runs each model for you.** Loads each one onto the GPU, runs your pages through it, and shuts down cleanly when it's done. Manages GPU memory if you load several at once.
- **Normalises every output.** One model returns Markdown, another HTML, another its own custom token stream. ocrscout converts every result into a common document model so you can compare apples to apples.
- **Writes one HuggingFace-shaped dataset.** One row per `(page, model)` in `data/train-*.parquet` — incrementally flushed during the run, so a partial run is always readable and `--resume` picks up where it left off (per-model: a page already done for model A is still attempted for model B if B hadn't processed it yet — the parquet shards themselves are the resume cursor). Each row carries timings, success/failure, the normalised document, a pre-rendered Markdown rendering, plus per-page cost / token / GPU-context columns (`litellm_cost`, `input_tokens`, `output_tokens`, `elapsed_seconds`, `gpu_type`, `cost_per_hour`, `gpu_time_cost`) populated automatically by the unified LiteLLM proxy that fronts every model call.
- **Has a browser viewer.** Flip through pages, switch between models, see bounding boxes overlaid on the source image, compare two artifacts word-by-word, place the page's reference baseline alongside any model output for direct comparison.
- **Scores predictions against any baseline.** Drop in a reference adapter (plain-text files, BHL legacy OCR, future ALTO/hOCR) and ocrscout runs a configurable suite of comparisons per page — text similarity always, character/word error rates with the `[eval]` extra, structural and layout deltas when both sides carry the relevant data. Reference is *not* assumed to be ground truth: provenance (`method`, `engine`, `confidence`) is captured so you can interpret the numbers as accuracy-vs-truth or agreement-vs-incumbent depending on the source.
- **Scales from one machine to a cloud pool.** One Runner abstraction (`local`, `skypilot`, `hf`) drives every deployment. Start with `ocrscout run` on your workstation, scale to an OVHcloud / AWS / GCP Kubernetes pool via `--runner skypilot --gpu L4 --workers 3`, or hand off to HuggingFace-sponsored compute via `--runner hf`. Same command shape, same Parquet output, same cost columns.

It's a *scout*, not a production system — the goal is to help you decide which model is worth using, or whether re-running your existing corpus through a newer one is worth the GPU-hours. Once you've picked a model, [Docling](https://github.com/docling-project/docling) is the toolchain for running it at scale.

## What it is not

- Not a production OCR pipeline — that's [Docling](https://github.com/docling-project/docling).
- Not a benchmark dataset — that's [MDPBench](https://huggingface.co/datasets/Delores-Lin/MDPBench) (which ocrscout can score against).
- Not a model repository — that's the HuggingFace Hub.
- Not a re-implementation of the [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) collection — those scripts are reference material; ocrscout drives vLLM and Docling directly through curated per-model profiles.

## Install

```bash
uv add 'ocrscout[all]'
```

That gets you everything — every input format, every backend, the browser viewer. If you want a smaller install:

```bash
uv add ocrscout                              # core only
uv add 'ocrscout[viewer,layout]'             # just what you need
```

The core install has **zero GPU or AI dependencies** and finishes in seconds on a CPU-only laptop. All the heavy stuff (PyTorch, vLLM, Transformers) is opt-in via the extras below — and even then, the actual model inference runs in isolated subprocesses or remote servers, never bloating your main Python environment.

| Extra | What it gives you | Add it when… |
| --- | --- | --- |
| `pdf` | Read PDF documents directly | Your source documents are PDFs (otherwise you need to convert them to images first) |
| `alto` | Parse ALTO/hOCR reference files | You have existing OCR transcriptions in ALTO/hOCR format and want to compare against them |
| `iiif` | Read images from a IIIF-compliant server | Your documents live in a digital library / archive that exposes IIIF |
| `cloud` | Read images from cloud storage | Your documents live on S3 (`s3://...`) or Google Cloud Storage (`gs://...`) |
| `bhl` | Sample images and OCR from the [Biodiversity Heritage Library](https://registry.opendata.aws/bhl-open-data/) S3 bucket | You want to scout OCR models against BHL's 305K-volume / 67M-page open-data corpus (DuckDB-driven catalog sampling, anonymous S3, JP2 decode via `imagecodecs`) |
| `eval` | Add character/word error rates to text comparisons | You want CER/WER (industry-standard OCR metrics from `jiwer`) on top of the always-on word-level similarity score |
| `docling` | Add the [Docling](https://github.com/docling-project/docling) backend | You also want to test classical OCR (Tesseract, EasyOCR) alongside the AI models |
| `serve` | The LiteLLM proxy binary that `LocalRunner` spawns to front your vLLMs | You're running ocrscout against vLLM-served models locally (the `LocalRunner` daemon stack) |
| `viewer` | A browser-based inspector for your results | You want to flip through pages and compare models visually instead of reading raw output |
| `layout` | The PP-DocLayoutV3 layout detector | You want to use single-task models (GLM-OCR, PaddleOCR-VL) on pages that mix text, tables, and formulas — see "Layout-aware models" below |
| `skypilot` | Remote orchestration via [SkyPilot](https://skypilot.readthedocs.io/) on Kubernetes (OVHcloud / AWS / GCP / on-prem K8s) | You're driving cloud GPU pools from a laptop, or running benchmarks/re-OCR jobs at scale |
| `hf` | The HuggingFace Jobs API runner | You're using HuggingFace-sponsored compute (the BHL / FineBooks pilot pattern) |
| `all` | All of the above | Recommended unless you really need a slim footprint |

## Quick start

Point ocrscout at a folder of images and a comma-separated list of models you want to test:

```bash
uv run ocrscout run --source ./images/ --models dots-mocr,smoldocling --sample 20
```

Here's what happens, in plain terms:

1. ocrscout looks up the recipe for each model (where to download it from, how to talk to it, what prompt it expects).
2. It downloads each model on first use — they're cached locally afterwards — and starts them up on your GPU.
3. It picks 20 pages from your image folder and sends every page through every model.
4. Whatever the models emit (markdown, HTML tables, custom token streams), ocrscout converts into one common document format so you can compare them apples-to-apples.
5. Everything goes into `./data/results/` (the default `--output-dir`; `./data/` is the conventional output root and is gitignored):
   - `data/train-*.parquet` — one row per (page, model) with timings, token counts, success/failure, the normalized document, and a pre-rendered `markdown` column. The directory is laid out exactly like a HuggingFace Hub dataset, so `datasets.load_dataset("./data/results", split="train")` works without further conversion.
   - `pipeline.yaml` — the resolved configuration ocrscout actually ran (source, models, normalizer overrides, sample/seed), so the run is reproducible.
6. A summary table is printed: how many pages each model handled, how often it failed, mean seconds per page, total output tokens.

**Got reference transcriptions or legacy OCR?** Point ocrscout at a reference adapter — plain-text files matched by page-id, or one of the corpus-specific adapters like `bhl_ocr` — and ocrscout runs every applicable comparison per (page, model). The text comparison's word-level similarity is always present; CER/WER are added if you install the `[eval]` extra; document and layout comparisons fire when both sides supply the relevant data.

```bash
# Plain-text references on disk
uv run ocrscout run --source ./images/ \
                    --reference plain_text --reference-path ./txt/ \
                    --models dots-mocr,smoldocling

# BHL legacy OCR as reference, sampled directly from the open-data S3 bucket
uv run ocrscout run --source-name bhl --sample 20 \
                    --source-arg 'languages=["eng"]' \
                    --reference bhl_ocr \
                    --models lighton-ocr2
```

ocrscout records each per-page comparison result in the parquet (rich JSON for renderer use, plus flat columns like `text_similarity`/`text_cer`/`text_wer`/`layout_iou_mean` for SQL ergonomics) and rolls per-model averages into the run summary table. References are not assumed to be ground truth — provenance (`method`/`engine`/`confidence`) flows through the parquet so you can interpret the numbers correctly.

**Look at the results:**

```bash
# Terminal summary + per-page dumps (good over SSH)
uv run ocrscout inspect ./data/results/

# Browser viewer for visual comparison
uv run ocrscout viewer ./data/results/
```

**Reproducible runs:** every `ocrscout run` writes a `pipeline.yaml` capturing exactly what was run. Re-run the same comparison later — or share the file with a teammate — with `ocrscout apply pipeline.yaml`.

## Bundled models

ocrscout ships ready-to-use recipes ("profiles") for nine models out of the box. Before reading the table, two distinctions matter:

**Layout-aware vs. single-task.** Modern OCR models split into two camps:

- **Layout-aware** models segment the page first — finding the headings, paragraphs, tables, and figures — and OCR each region with the right strategy. They handle mixed content (text + tables + formulas on one page) in a single call. *Examples: dots-ocr, smoldocling.*
- **Single-task** models do exactly one thing per call. You ask them to read text and they return text; you ask them to read tables and they return tables. They tend to be much smaller and faster, but on a mixed page their text mode flattens tables into prose. *Examples: GLM-OCR, PaddleOCR-VL.*

To get the best of both worlds with single-task models, ocrscout pairs them with an **external layout detector** ([PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors)) that finds regions on the page first, then dispatches each region to the right task mode. Profiles ending in `-layout` use this orchestration.

| Profile | Best for | Why |
| --- | --- | --- |
| `dots-ocr` | Mixed pages (text + tables + figures) | Compact (1.7B), layout-aware, structured output with bounding boxes |
| `dots-mocr` | Same as above, more structured output | DocTags-format output (closer to docling's native model) |
| `smoldocling` | Tiny GPUs / CPU-only hosts | 256M parameter model; runs on hardware too small for the others |
| `lighton-ocr2` | Mixed pages, prefer Markdown | Mid-size; emits HTML tables inline in markdown text |
| `glm-ocr` | Plain text pages, fast inference | 0.9B; you pick text/table/formula mode per run |
| `glm-ocr-layout` | Mixed pages on a small GPU | GLM-OCR + layout detector; recovers structured tables that plain `glm-ocr` would flatten |
| `paddleocr-vl` | Plain text pages, fastest in the zoo | 0.9B; ~5 s/page on a single GPU |
| `paddleocr-vl-layout` | Pages with charts, formulas, and tables | Same model + layout detector; the only profile that can recognize charts |
| `rolm-ocr` | Plain text alternative | Reducto's RolmOCR — different training corpus, sometimes better on receipts/invoices |

**Adding a new model.** Drop a YAML file into `src/ocrscout/profiles/` describing the model's identity, prompt template, and tuning settings. `uv run ocrscout introspect <name>` jump-starts this by drafting a YAML from the matching reference script in the [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) collection on Hugging Face — you fill in the gaps. Or ship profiles from your own package via Python entry points; nothing in ocrscout has to be edited.

## How models get loaded and served

OCR models are big — loading one onto a GPU takes 30 seconds for the small ones, a couple of minutes for the larger ones. Every ocrscout deployment runs the same three-tier stack:

```
ocrscout → LiteLLM proxy (localhost:4000) → one vLLM per model (localhost:8000, 8001, …)
```

The LiteLLM proxy is the unifying layer: it gives every model the same OpenAI-compatible URL, attributes per-request cost back to the calling page (via the success callback), and lets you mix locally-hosted vLLM models with hosted APIs (Gemini, Anthropic) in the same run. **You don't manage this stack by hand** — a `Runner` does it for you. Pick by where the compute lives:

Both daemons are bound to `127.0.0.1` (loopback only). vLLM and LiteLLM default to `0.0.0.0`, which on a public-IP cloud GPU would let anyone in the world submit inference requests using your GPU — ocrscout forces loopback so the stack is unreachable from outside the host even when the box has a public IP. Per-call timeouts default to **300 s** with **2 retries** on retryable failures (rate limits / 5xx / timeouts); override either via `backend_args.request_timeout` / `backend_args.num_retries` in a profile YAML if your hardware or workload needs different bounds.

### "Just run it" — `ocrscout run`

You point ocrscout at a folder, it spins up the local stack, runs the pages, and shuts down. By default the dispatch is **model-major**: one vLLM serve at a time — spawn → all pages → tear down → next model. Each model gets the full GPU (so the timing numbers are honest), and the GPU-memory preflight only needs to fit the **largest single model**, not the sum of all of them. That means a benchmark matrix of e.g. `dots-mocr` (3B) + `glm-ocr-layout` (0.9B) runs comfortably on a 16 GB GPU even though their combined working set wouldn't fit.

```bash
uv run ocrscout run --source ./images/ --models dots-mocr,glm-ocr-layout,paddleocr-vl-layout
```

The trade-off is vLLM cold-start (~30–60s) gets paid once per model instead of once per run. For real benchmark workloads (hundreds+ pages per model) it's noise; for tiny test runs it can dominate.

Pass `--parallel-models N` to widen the chunk to N concurrent serves — useful if you have VRAM headroom and want to amortize the cold-start cost. Pass `--keep-up` to switch back to the old behavior: every requested model loaded simultaneously and the stack left running after the pipeline finishes, so subsequent `ocrscout submit` calls against the same stack skip the launch cost entirely.

Pass `--resume` to skip pages already in `data/train-*.parquet` (per-model: a page already done for model A is still attempted for model B if B hadn't processed it yet).

### "Stateful workflow" — `launch` / `submit` / `status` / `down`

Same daemon stack, but split into four commands so you can fire-and-forget batches and close your terminal. Daemonised processes (proper double-fork + PID files in `~/.ocrscout/`) survive SSH disconnects and laptop sleep:

```bash
# Bring the stack up once; survives terminal close
uv run ocrscout launch --models dots-mocr,glm-ocr-layout

# Submit work; returns a job id immediately, runs in the background
uv run ocrscout submit --source ./scans/ --output ./out/ --pages 1000

# Inspect from anywhere — reads ~/.ocrscout/state.yaml
uv run ocrscout status
uv run ocrscout logs <job-id>

# Resume after a crash (per-model: reads (page_id, model) pairs from <output>/data/train-*.parquet)
uv run ocrscout submit --source ./scans/ --output ./out/ --resume

# Tear down when done
uv run ocrscout down
```

### "Scale to a cluster" — `--runner skypilot`

The same Runner ABC drives [SkyPilot](https://skypilot.readthedocs.io/) on any Kubernetes context — OVHcloud Managed Kubernetes, AWS EKS, GCP GKE, on-prem. Workers install `ocrscout[vllm]` via `uv`, start the same LiteLLM + vLLM stack locally on each pod, and write Parquet directly to your S3 / GCS / `hf://` output:

```bash
uv run ocrscout launch --runner skypilot --gpu L4 --workers 3 \
  --models dots-mocr
uv run ocrscout submit --source s3://my-bucket/scans/ --output s3://my-bucket/out/ \
  --pages 8000 --num-jobs 10
uv run ocrscout down
```

SkyPilot's controller manages job execution independently of your CLI process; if your laptop reboots mid-run, jobs keep going.

### "Hosted compute" — `--runner hf`

For HuggingFace-sponsored compute. Source and output use `hf://` paths so datasets stay on the Hub end-to-end:

```bash
uv run ocrscout run --runner hf --gpu L4 --models dots-mocr \
  --source hf://datasets/biodiversitylibrary/sample \
  --output hf://datasets/yourname/ocrscout-results
```

### "I have my own proxy" — env var

If you've started a LiteLLM proxy yourself (or have access to one your team manages), set the URL in the environment and skip the Runner lifecycle entirely:

```bash
OCRSCOUT_VLLM_URL=http://my-proxy:4000/v1 uv run ocrscout run \
  --source ./images/ --models dots-mocr
```

The proxy must serve the requested profile name(s) in its `model_list`. Useful when you're testing changes to the proxy config without re-launching the whole stack.

For the GPU-memory bookkeeping rules, per-profile sizing knobs, daemon lifecycle, and Runner internals, see [CLAUDE.md](CLAUDE.md).

## Performance

ocrscout's autoscaler tunes three knobs at launch time so you don't have to:

- **vLLM KV-cache budget** — sized from detected GPU free VRAM, capped by `--gpu-budget` (default 85 %).
- **Region concurrency** for `backend: layout_chat` profiles — derived from the GPU's memory bandwidth via [dbgpu](https://github.com/orf/dbgpu) (`bandwidth_GB_s / 15`, ~2800 SKUs covered).
- **CPU detector pool size** for the PP-DocLayoutV3 layout detector — derived from `sched_getaffinity` (`(cores − 2) // 2` detector workers × matching PyTorch intra-op threads), so the layout stage doesn't serialise the run on bandwidth-rich GPUs.

The result is layout-aware OCR throughput that scales meaningfully across GPU classes without any per-host tuning:

| GPU | Region concurrency (auto) | Detector pool (auto) | s/page on `glm-ocr-layout` |
|---|---:|---:|---:|
| H100 PCIe 80 GB | 136 | 11 workers | **0.50** |
| A10 PCIe | 40 | 14 workers | **1.43** |
| L4 | 20 | 3 workers | **2.18** |
| GB10 (DGX Spark) | 18 | 9 workers | **2.02** |

Measured on a 240-page BHL sample (mixed Victorian books, journals, postcards, field notebooks; deterministic seed). No flags passed beyond `--source-name bhl --sample 240`; every knob came from the autoscaler. Override anything per-run via `--batch-concurrency N` (region pool) or `--detector-workers N` (CPU pool); see [CLAUDE.md](CLAUDE.md) for the full sizing rules and architectural detail.

## Cost tracking

Every backend call goes through LiteLLM, which fires a success callback that accumulates per-page tokens, elapsed seconds, and dollar cost into the row written for each `(page, model)`. The Parquet output carries flat columns for direct SQL access:

| Column | Source | What it measures |
|---|---|---|
| `gpu_type`, `provider`, `cost_per_hour` | `~/.ocrscout/config.yaml` or env (`OCRSCOUT_GPU_TYPE`, `OCRSCOUT_COST_PER_HOUR`, `OCRSCOUT_PROVIDER`) | GPU context stamped uniformly across a run |
| `elapsed_seconds` | LiteLLM callback | Wall-clock inference time |
| `input_tokens`, `output_tokens` | LiteLLM response usage | Token accounting |
| `litellm_cost` | `litellm.completion_cost()` | Token-based cost (works for commercial APIs out of the box; populate `model_info` in the proxy config for self-hosted models) |
| `gpu_time_cost` | `elapsed_seconds / 3600 × cost_per_hour` | Actual GPU infrastructure cost |

For cross-run aggregation, `ocrscout costs` runs a DuckDB query over any local or remote Parquet glob:

```bash
ocrscout costs --output ./out/ --by model
ocrscout costs --output s3://my-bucket/benchmark/ --by model,gpu_type
ocrscout costs --output hf://datasets/yourname/ocrscout-results --by model
```

## Looking at results

After a run, your output directory contains `data/train-*.parquet` (one row per page-and-model with timings, the normalized document, a pre-rendered `markdown` column, and accuracy scores if you provided ground truth) plus the resolved `pipeline.yaml`. The layout is HF-Hub-compatible, so `datasets.load_dataset(<out>, split="train")` works directly. ocrscout has two ways to browse it:

### `ocrscout inspect <out>` — terminal

A summary table you can read over SSH or paste into a bug report. Zero extra dependencies. Common uses:

```bash
ocrscout inspect ./out/                                    # per-model summary table (with comparison metric columns when present)
ocrscout inspect ./out/ --page <page_id>                   # dump every model's output for one page (plus the reference, if any)
ocrscout inspect ./out/ --page <id> --compare A,B          # word-level comparison between two models for that page
ocrscout inspect ./out/ --page <id> --compare A,reference  # compare a model to the run's configured reference baseline
ocrscout inspect ./out/ --page <id> --compare A,B --comparison-type document  # structural (heading/table/picture-count) deltas
```

Add `--html` to a `--compare` invocation and ocrscout serves a one-shot HTML page over your LAN — handy when you want to share a comparison link without standing up the full viewer.

### `ocrscout viewer <out>` — browser

A long-lived web app for visual comparison (needs the `viewer` extra). Three view modes:

- **Single** — one artifact at a time (any model output, or the page's reference), full text, with the source page on the left and bounding boxes overlaid showing which regions the model detected.
- **Side-by-side** — two or more artifacts in parallel columns. Pick any combination of models — and the run's `reference` baseline if one is configured — to see where they agree and disagree at a glance.
- **Compare** — pick a Prediction (model output) and a Baseline (another model or the reference, with provenance shown), and ocrscout stacks every registered comparison whose required modality is satisfied: word-level text diff (always), structural deltas (when both sides have a `DoclingDocument`), per-category bbox IoU (when both sides have layout).

Click through pages with `j`/`k`, switch modes with `1`/`2`/`3`, toggle the image pane with `i`. The current view is encoded in the URL, so you can share a link to a specific page-and-comparison with a teammate.

## Extending ocrscout

ocrscout is built to be plugged into without touching its source tree. If your situation needs something it doesn't ship — a custom source format, a new OCR model, an evaluator that measures something specific to your domain — you write a small Python class in your own package and register it. ocrscout discovers it via Python's standard "entry points" mechanism the next time you run it.

The extension points (each is a Python Abstract Base Class):

- **`SourceAdapter`** — yields page images from somewhere. Built-in supports local folders, S3/GCS buckets, HuggingFace datasets, and the Biodiversity Heritage Library (`bhl`). Add one for IIIF, PDF rasterization at scale, your in-house DAM, etc. Sources may optionally yield typed `Volume` metadata for catalog-driven corpora.
- **`ReferenceAdapter`** — supplies a reference (text or `DoclingDocument`, with typed provenance) for a page. Built-in supports plain `.txt` files and BHL legacy OCR; add ALTO, hOCR, page XML, human transcription archives, etc.
- **`ModelBackend`** — runs an OCR model and returns raw output. Built-in: `litellm` (every OpenAI-compatible call — vLLM-served VLMs *and* hosted APIs like Gemini), `layout_chat` (per-region OCR over the same proxy), `docling`, `tesseract`. Add new backends for inference runtimes that don't fit the LiteLLM-proxied path.
- **`Runner`** — orchestrates the compute stack (LiteLLM proxy + N vLLM backends) for OCR workloads. Built-in: `local` (daemonised processes on this machine), `skypilot` (Kubernetes pool via SkyPilot), `hf` (HuggingFace Jobs API). Add new runners for AI Factory, Slurm, your in-house scheduler, etc.
- **`LayoutDetector`** — finds typed regions on a page image. Used by layout-aware backends. Built-in: PP-DocLayoutV3.
- **`Normalizer`** — converts a model's raw output into ocrscout's common document format. Add one when a new model emits a custom output dialect.
- **`ExportAdapter`** — writes results somewhere. Built-in: parquet (incremental shards; `--resume` reads `(page_id, model)` pairs from the shards directly). Add a CSV or JSON-Lines exporter, or write to a database.
- **`Comparison`** — one analytic axis between a prediction and a baseline. Built-in: text similarity (with optional CER/WER), document structure deltas, layout IoU. Add a semantic comparison, an LLM-as-judge, table-cell F1, anything you need. Each `Comparison` produces a typed `ComparisonResult`.
- **`ComparisonRenderer`** — renders a `ComparisonResult` for terminal, HTML, and the in-viewer Compare mode. Built-in renderers ship for each built-in comparison; downstream packages can register their own.
- **`Benchmark`** — bundles a fixed source + reference adapter + comparison suite into a named, citable benchmark.
- **`Reporter`** — turns a results directory into a human-readable report (HTML, terminal table, etc.).

Subclass the relevant ABC, set a `name`, and either register in-process (`registry.register(...)`) or via a `pyproject.toml` entry point in your own package. ocrscout's built-ins can't be shadowed by third-party plugins. Full guide in [CLAUDE.md](CLAUDE.md).

## Logging

Every CLI command takes verbosity flags:

| Flag | What you see |
| --- | --- |
| `-q` | Only warnings and errors. The summary table still prints. |
| (none) | Default. Per-page progress, per-model start/done, GPU allocation. |
| `-v` | Above plus timestamps, full URLs/paths, GPU per-process telemetry. |
| `-vv` | Everything, including subprocess command lines and source-file:line markers. Useful for bug reports. |

Output is one logical line per record (no fancy formatting), so it's safe to pipe into `grep`, paste into bug reports, or capture in CI logs.

## License

Apache 2.0 — see [LICENSE](LICENSE).

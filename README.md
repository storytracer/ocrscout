# ocrscout

**Scout frontier OCR models on your data, your hardware, your terms.**

## The problem

Modern VLM-based OCR is fragmented. Every new release — dots-OCR, GLM-OCR, LightOnOCR, RolmOCR, SmolDocling, and the next one shipping next week — comes as a one-off inference script with its own prompt template, sampling settings, output format (markdown? DocTags? layout JSON?), and vLLM tuning. If you want to know which one actually works best on *your* documents, at *your* throughput targets, on *your* GPU, you end up writing glue code per model and eyeballing results.

ocrscout is the missing layer:

- **One command** runs any of N models on the same input.
- **One document model** (`DoclingDocument`) for every output, regardless of what the model emitted.
- **One results table** (parquet) with timing, throughput, output shape, and — when you have ground truth — edit-distance scores.
- **One viewer** to flip through pages, diff models side-by-side, and see bbox overlays.

It's a *scout*, not a production pipeline: the goal is to help you decide whether a new model is worth deploying, or whether a re-OCR of an existing corpus is worth the GPU-hours.

## What it is not

- Not a production OCR pipeline — that's [Docling](https://github.com/docling-project/docling).
- Not a benchmark dataset — that's [MDPBench](https://huggingface.co/datasets/Delores-Lin/MDPBench) (which ocrscout can score against).
- Not a model repository — that's the HuggingFace Hub.
- Not a re-implementation of the [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) collection — those scripts are reference material; ocrscout drives vLLM and Docling directly through curated per-model profiles.

## Install

```bash
uv add ocrscout
# or, with optional extras:
uv add 'ocrscout[pdf,docling,serve,viewer]'
```

The core install has **zero GPU dependencies** and finishes in seconds on a CPU-only host. All GPU work happens in subprocess-isolated `vllm` workers (spawned via `uv run --with vllm …`) or against remote vLLM servers, so the package itself stays light.

Optional extras:

| Extra | Adds | When you need it |
| --- | --- | --- |
| `pdf` | `ocrmypdf`, `pikepdf` | PDF source/reference adapters |
| `alto` | `lxml` | ALTO/hOCR reference adapter |
| `iiif` | `httpx` | IIIF source adapter |
| `docling` | `docling` | non-VLM OCR backend (Tesseract, EasyOCR, etc. via Docling) |
| `serve` | `litellm[proxy]` | managed multi-model server mode |
| `viewer` | `gradio`, `polars` | interactive browser inspector |

## Quick start

Point ocrscout at a folder of images and a comma-separated list of model profiles:

```bash
uv run ocrscout run --source ./images/ --models dots-mocr,smoldocling --sample 20
```

What this does:

1. Loads the YAML profiles for `dots-mocr` and `smoldocling`.
2. Spawns a `uv run --with vllm` subprocess per model, runs all 20 pages through it, captures raw outputs and per-page timings.
3. Normalizes each raw output (DocTags, markdown, layout JSON, …) into a `DoclingDocument`.
4. Writes one row per (page, model) to `./ocrscout-results/results.parquet`, plus a `text/<page>.<model>.md` sidecar for every page.
5. Prints a per-model summary table: pages succeeded/failed, mean s/page, output token counts.

If you have reference OCR (one `.txt` per page, file-stem matched), add it and you also get edit-distance scores per page and overall:

```bash
uv run ocrscout run --source ./images/ \
                    --reference plain_text --reference-path ./txt/ \
                    --models dots-mocr,smoldocling
```

To look at what came out:

```bash
uv run ocrscout inspect ./ocrscout-results/             # terminal table + page dumps
uv run --extra viewer ocrscout viewer ./ocrscout-results/   # browser viewer
```

For repeatable runs, capture the full configuration in a YAML and apply it:

```bash
uv run ocrscout apply pipeline.yaml
```

## Bundled model profiles

Each model is described by a hand-curated YAML in [src/ocrscout/profiles/](src/ocrscout/profiles/). Out of the box:

| Profile | Backend | Output | Notes |
| --- | --- | --- | --- |
| `dots-mocr` | vLLM | DocTags + bboxes | Layout-aware, structured |
| `dots-ocr` | vLLM | Markdown + bboxes | Plain reading-order text |
| `glm-ocr` | vLLM | Markdown | GLM-4V-based |
| `lighton-ocr2` | vLLM | Markdown | LightOnOCR-2 |
| `rolm-ocr` | vLLM | Markdown | Reducto's RolmOCR |
| `smoldocling` | vLLM | DocTags | Layout-aware, low VRAM |

A profile records the model's identity (HF id, size), output shape (format + normalizer + label mapping), prompt templates, and vLLM tuning (`vllm_engine_args`, `sampling_args`, `kv_cache_memory_bytes`). To add a model, drop a YAML into `src/ocrscout/profiles/` — or ship one from a downstream package via the `ocrscout.backends` / `ocrscout.normalizers` entry points. `uv run ocrscout introspect <name>` produces a draft profile from a matching [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) reference script (static `ast.parse` only — the upstream script is never executed).

## Running vLLM-backed models

vLLM-source profiles can run in three modes — pick by what's already running:

| Mode | Trigger | Lifecycle | Use case |
| --- | --- | --- | --- |
| **runner** | _(default)_ | per-invocation `uv run --with vllm …` subprocess; killed on exit | one-off scout, single model, no warm server available |
| **external-server** | `--server-url URL` | not managed by ocrscout | a `vllm serve` you started by hand, or any OpenAI-compatible endpoint (incl. a LiteLLM proxy) |
| **managed** | `ocrscout serve` (long-lived) **or** `ocrscout run --managed` (inline) | ocrscout owns N `vllm serve` + 1 LiteLLM proxy (when N≥2); torn down on exit | comparing multiple models with warm servers; one URL fronting many models |

```bash
# runner (default): spin up vllm just for this run
uv run ocrscout run --source ./images/ --models dots-mocr

# external-server: connect to a vllm-serve / proxy you already started
uv run ocrscout run --source ./images/ --models dots-mocr \
  --server-url http://localhost:8000/v1

# managed inline: ocrscout spawns + tears down for this single run
uv run ocrscout run --source ./images/ --models dots-mocr,dots-ocr,glm-ocr \
  --managed

# managed long-lived: keep warm servers up, drive runs from another terminal
uv run ocrscout serve --models dots-mocr,dots-ocr,glm-ocr
# in another terminal:
uv run ocrscout run --source ./images/ --models dots-mocr \
  --server-url http://localhost:4000/v1
```

`--managed` and `--server-url` are mutually exclusive. By default models execute sequentially even when several are managed (each gets the full GPU); override with `--parallel-models / -P` only if you have separate GPUs per model. See [CLAUDE.md](CLAUDE.md) for the GPU-budget model and per-profile KV-cache sizing rules.

## Inspecting results

Every `ocrscout run` writes a `results.parquet` (one row per page × model, with timings, raw output, and the normalized `DoclingDocument` JSON) plus optional `text/<page>.<model>.md` sidecars. Two read-only tools share that input:

| Command | When to use |
| --- | --- |
| `ocrscout inspect <out>` | Terminal — Rich summary table; `--page <id>` per-model markdown dump; `--diff a,b --html` serves a one-shot side-by-side diff over the LAN. Zero extra deps; works over SSH and pastes cleanly into bug reports. |
| `ocrscout viewer <out>` | Browser — long-lived Gradio app with a page picker, mode-aware model picker (radio for Single, checkbox for Side-by-side, paired Model A / Model B dropdowns for Diff), source page with color-coded bbox overlay + deduplicated category legend, and matching color-coded section blocks in the text pane for layout-aware models. State persists via `BrowserState` and round-trips through URL query params for shareable views. Pulls in `gradio` + `polars` from the `viewer` extra. |

## Extending

ocrscout is built around a small set of ABCs in [src/ocrscout/interfaces/](src/ocrscout/interfaces/):

- `SourceAdapter` — yields `PageImage` objects (directory, HF dataset, IIIF, PDF, …).
- `ReferenceAdapter` — returns ground-truth text or `DoclingDocument` for a `page_id`.
- `ModelBackend` — runs inference and yields `RawOutput`.
- `Normalizer` — converts `RawOutput` + `PageImage` into a `DoclingDocument`.
- `ExportAdapter` — writes a stream of `ExportRecord` objects.
- `Evaluator` — scores a prediction against a reference.
- `Benchmark` — bundles source + reference + evaluator + canonical scoring protocol.
- `Reporter` — turns a results directory into a report.

Subclass the relevant ABC, set `name: ClassVar[str]`, and register either in-process (`registry.register("normalizers", "my_normalizer", MyNormalizer)`) or via a `pyproject.toml` entry point group like `ocrscout.normalizers`. Built-ins are protected from being shadowed by third-party entry points. See [CLAUDE.md](CLAUDE.md) for the full extension guide and design decisions.

## Logging

Status output flows through stdlib `logging` under the `ocrscout` namespace. CLI commands accept `-q` (warnings only), default (info), `-v` (timestamps + verbose events), and `-vv` (debug + module:line). One logical line per record — grep-friendly, no Rich markup in log lines.

## License

Apache 2.0 — see [LICENSE](LICENSE).

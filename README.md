# ocrscout

**Scout frontier OCR models on your data, your hardware, your terms.**

ocrscout is a toolkit for evaluating state-of-the-art OCR models on user-supplied data. It answers a single question:

> Which OCR approach gives the best results on our documents, at what speed, and is it worth re-processing what we already have?

It is **not** a production OCR pipeline (that's [Docling](https://github.com/docling-project/docling)), **not** a benchmark dataset (that's [MDPBench](https://huggingface.co/datasets/Delores-Lin/MDPBench)), and **not** a model repository (that's HuggingFace Hub). It sits between these.

## What it does

- Wraps the ~20 community OCR scripts in [HuggingFace `uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) via lightweight model profiles.
- Normalizes their varied outputs (markdown, DocTags, layout JSON) into a single document model — `DoclingDocument` from [`docling-core`](https://github.com/docling-project/docling-core).
- Adds the measurement layer: timing, throughput (pages/hour), edit distance vs. reference OCR, GPU memory peak, layout capability, side-by-side comparison.
- Runs on local single-GPU machines, HuggingFace Jobs, vLLM API servers, OpenAI-compatible endpoints, Kubernetes, and HPC/SLURM. The core package has **zero GPU dependencies** — GPU work always happens in a separate process or server.

## Install

```bash
uv add ocrscout
# or, with optional extras:
uv add 'ocrscout[pdf,docling,serve,viewer]'
```

Optional extras:

- `pdf` / `alto` / `iiif` — extra source/reference adapters
- `docling` — `DoclingBackend` for non-VLM OCR
- `serve` — `litellm[proxy]` for managed multi-model server mode
- `viewer` — Gradio-based interactive inspector (see below)

## Quick start

```bash
# Run multiple models on a folder of images
uv run ocrscout run --source ./images/ --models dots-mocr,smoldocling --sample 20

# Compare against existing reference OCR
uv run ocrscout run --source ./images/ \
                     --reference plain_text --reference-path ./txt/ \
                     --models dots-mocr

# Inspect a previous run's results in the terminal
uv run ocrscout inspect ./out/

# …or open the same results in an interactive Gradio viewer
uv run --extra viewer ocrscout viewer ./out/

# Apply a full pipeline from a YAML config
uv run ocrscout apply pipeline.yaml
```

## Bundled model profiles

Each model is described by a hand-curated YAML in [src/ocrscout/profiles/](src/ocrscout/profiles/). Out of the box:

| Profile | Backend | Notes |
| --- | --- | --- |
| `dots-mocr` | vLLM | DocTags-style structured output, layout-aware |
| `dots-ocr` | vLLM | Plain markdown + bboxes |
| `glm-ocr` | vLLM | GLM-4V-based OCR |
| `lighton-ocr2` | vLLM | LightOnOCR-2 |
| `rolm-ocr` | vLLM | Reducto's RolmOCR |
| `smoldocling` | vLLM | Layout-aware, low VRAM |

Add a new model by dropping a YAML in `src/ocrscout/profiles/` (or shipping it via the `ocrscout.backends` / `ocrscout.normalizers` entry points from a downstream package). `uv run ocrscout introspect <name>` produces a draft profile from an [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) reference script.

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

`--managed` and `--server-url` are mutually exclusive. By default models execute sequentially even when several are managed (each gets the full GPU); override with `--parallel-models / -P` if you have separate GPUs per model. See [CLAUDE.md](CLAUDE.md) for the GPU-budget model and per-profile KV-cache sizing rules.

## Inspecting results

Two ways to look at a previous run's `output_dir/results.parquet`:

| Command | When to use |
| --- | --- |
| `ocrscout inspect <out>` | Terminal — Rich summary table, `--page <id>` per-model markdown dump, `--diff a,b --html` one-shot side-by-side diff page. Zero extra deps; works over SSH. |
| `ocrscout viewer <out>` | Browser — long-lived Gradio app: page picker, adaptive model picker (radio for Single, checkbox for Side-by-side, paired dropdowns for Diff), source page with color-coded bbox overlay + deduplicated legend, color-matched section blocks in the text pane for layout-aware models. Pulls in `gradio` + `polars` from the `viewer` extra. |

## Logging

Status output uses stdlib `logging` under the `ocrscout` namespace. CLI commands accept `-q` (warnings only), default (info), `-v` (timestamps + verbose events), and `-vv` (debug + module:line). One logical line per record — grep-friendly, no Rich markup.

See [CLAUDE.md](CLAUDE.md) for the implementation roadmap, design decisions, and contributor guide.

## License

Apache 2.0 — see [LICENSE](LICENSE).

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
uv add 'ocrscout[pdf,docling]'
```

## Quick start

```bash
# Run multiple models on a folder of images
uv run ocrscout run --source ./images/ --models dots-mocr,smoldocling --sample 20

# Compare against existing reference OCR
uv run ocrscout run --source ./images/ \
                     --reference plain_text --reference-path ./txt/ \
                     --models dots-mocr

# Refresh auto-generated profiles from uv-scripts/ocr
uv run ocrscout sync

# Apply a full pipeline from a YAML config
uv run ocrscout apply pipeline.yaml
```

See [CLAUDE.md](CLAUDE.md) for the implementation roadmap, design decisions, and contributor guide.

## License

Apache 2.0 — see [LICENSE](LICENSE).

"""`ocrscout ocr` — run OCR model(s) over a source, write raw model output.

Stage 3 of the decoupled pipeline. Reads pages from a stage parquet (``ocrscout
sample`` / ``layout`` output) or any source adapter, runs the requested OCR
model(s) through the full runner stack (LiteLLM proxy + vLLM autoscaler +
model-major chunking, identical to ``ocrscout run``), and writes one
``data/raw-*.parquet`` row per (page, model): the ``RawOutput`` payload plus the
cost/autoscaler columns. Normalization, reference comparison, and the rich
``train-*.parquet`` are deferred to ``ocrscout normalize``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import typer

from ocrscout.cli import app
from ocrscout.cli._resolve import build_source_ref
from ocrscout.cli.run import _parse_source_args, run_pipeline
from ocrscout.exports.layout import parquet_dest
from ocrscout.log import setup_logging
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("ocr")
def ocr_cmd(
    source: str | None = typer.Option(
        None, "--source", "-s",
        help="A stage parquet (sample/layout output) or any source the "
             "default `hf_dataset` adapter understands (dir / URL / Hub id).",
    ),
    source_name: str = typer.Option("hf_dataset", "--source-name"),
    source_arg: list[str] = typer.Option(None, "--source-arg"),
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated profile names."
    ),
    layout: str | None = typer.Option(
        None, "--layout",
        help="A layout-*.parquet (or dir) of precomputed regions. layout_chat "
             "profiles OCR against these instead of running detection; "
             "whole-page profiles ignore it.",
    ),
    sample: int | None = typer.Option(None, "--sample"),
    seed: int = typer.Option(42, "--seed"),
    start_idx: int | None = typer.Option(
        None, "--start-idx", min=0,
        help="Half-open [start, end) window over a stage-parquet source, for "
             "partitioning across workers.",
    ),
    end_idx: int | None = typer.Option(None, "--end-idx", min=0),
    output_dir: Path = typer.Option(
        Path("./data/results"), "--output-dir", "-o",
        help="Where to write data/raw-*.parquet.",
    ),
    gpu_budget: float = typer.Option(0.85, "--gpu-budget"),
    base_port: int = typer.Option(8000, "--base-port"),
    proxy_port: int = typer.Option(4000, "--proxy-port"),
    keep_up: bool = typer.Option(False, "--keep-up"),
    parallel_models: int | None = typer.Option(None, "--parallel-models", "-P"),
    batch_concurrency: int | None = typer.Option(None, "--batch-concurrency", min=1),
    detector_workers: int | None = typer.Option(None, "--detector-workers", min=1),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip (page, model) pairs already in data/raw-*.parquet.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Run OCR model(s) and write raw output to data/raw-*.parquet (no normalize)."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if detector_workers is not None:
        os.environ["OCRSCOUT_DETECTOR_WORKERS"] = str(detector_workers)
    if source_name == "hf_dataset" and source is None:
        raise typer.BadParameter(
            "--source is required with the default `hf_dataset` adapter."
        )

    source_args: dict[str, Any] = _parse_source_args(source_arg or [])
    ref = build_source_ref(
        source=source, source_name=source_name, source_args=source_args,
        sample=sample, seed=seed, start_idx=start_idx, end_idx=end_idx,
    )
    cfg = PipelineConfig(
        name="ocr",
        source=ref,
        comparisons=[],
        models=[m.strip() for m in models.split(",") if m.strip()],
        layout=(
            AdapterRef(name="precomputed", args={"path": layout})
            if layout else None
        ),
        # export dest is unused in ocr_only (the StageWriter targets
        # output_dir/data directly) but PipelineConfig requires it.
        export=AdapterRef(name="parquet", args={"dest": str(parquet_dest(output_dir))}),
        sample=sample,
        output_dir=output_dir,
    )

    log.info(
        "ocrscout ocr: %d model(s) [%s] · source=%s · output=%s",
        len(cfg.models), ",".join(cfg.models), ref.name, output_dir / "data",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if parallel_models is None:
        parallel_models = 1
    parallel_models = max(1, min(parallel_models, len(cfg.models)))

    run_pipeline(
        cfg,
        parallel_models=parallel_models,
        base_port=base_port,
        proxy_port=proxy_port,
        gpu_budget=gpu_budget,
        keep_up=keep_up,
        resume=resume,
        batch_concurrency=batch_concurrency,
        ocr_only=True,
    )

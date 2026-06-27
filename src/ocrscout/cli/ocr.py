"""`ocrscout ocr` — run OCR model(s) over a pages/layout parquet → raw parquet."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from ocrscout.cli import app
from ocrscout.log import setup_logging
from ocrscout.pipeline.engine import PipelineEngine
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("ocr")
def ocr_cmd(
    source: Path | None = typer.Option(None, "--source", "-s",
        help="Dir holding data/pages-*.parquet (or layout-*.parquet for "
             "precomputed regions). Defaults to --output-dir."),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated profiles."),
    output_dir: Path = typer.Option(Path("./data/results"), "--output-dir", "-o"),
    gpu_budget: float = typer.Option(0.85, "--gpu-budget"),
    base_port: int = typer.Option(8000, "--base-port"),
    proxy_port: int = typer.Option(4000, "--proxy-port"),
    keep_up: bool = typer.Option(False, "--keep-up"),
    parallel_models: int | None = typer.Option(None, "--parallel-models", "-P"),
    batch_concurrency: int | None = typer.Option(None, "--batch-concurrency", min=1),
    detector_workers: int | None = typer.Option(None, "--detector-workers", min=1),
    resume: bool = typer.Option(False, "--resume"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Run OCR model(s) → data/raw-*.parquet (no normalize)."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if detector_workers is not None:
        os.environ["OCRSCOUT_DETECTOR_WORKERS"] = str(detector_workers)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    cfg = PipelineConfig(
        name="ocr", source=AdapterRef(name="pages", args={}), models=model_list,
        comparisons=[], export=AdapterRef(name="parquet", args={}), output_dir=output_dir,
    )
    PipelineEngine().execute(
        cfg, stages=["ocr"], resume=resume, input_dir=source,
        parallel_models=_clamp(parallel_models, len(model_list)),
        gpu_budget=gpu_budget, base_port=base_port, proxy_port=proxy_port,
        keep_up=keep_up, batch_concurrency=batch_concurrency,
    )


def _clamp(parallel_models: int | None, n_models: int) -> int:
    return max(1, min(parallel_models or 1, max(1, n_models)))

"""`ocrscout run` — one-shot: sample → ocr → normalize through the engine.

Thin wrapper: build a ``PipelineConfig`` from the flags and hand it to the
``PipelineEngine``, which composes the stages and (via the ``Provisioner``)
stands up a runner only for the OCR stage. The old 1500-line orchestrator is
gone — its concerns now live in ``io/``, ``pipeline/``, and ``orchestration/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.table import Table

from ocrscout.cli import app
from ocrscout.cli._resolve import (
    build_source_ref,
    parse_comparisons_flag,
    parse_source_args,
)
from ocrscout.log import setup_logging
from ocrscout.pipeline.engine import PipelineEngine
from ocrscout.pipeline.pipeline import PipelineResult
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("run")
def run(
    source: str | None = typer.Option(None, "--source", "-s",
        help="Image source: local dir, fsspec URL, HF Hub id, or a stage "
             "parquet dir to resume from."),
    source_name: str = typer.Option("hf_dataset", "--source-name"),
    source_arg: list[str] = typer.Option(None, "--source-arg"),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated profiles."),
    reference: str | None = typer.Option(None, "--reference"),
    reference_path: Path | None = typer.Option(None, "--reference-path"),
    comparisons: str | None = typer.Option(None, "--comparisons"),
    sample: int | None = typer.Option(None, "--sample"),
    seed: int = typer.Option(42, "--seed"),
    layout: str | None = typer.Option(None, "--layout",
        help="A layout-*.parquet dir of precomputed regions for layout_chat models."),
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
    """Run OCR model(s) against a source and emit normalized, compared results."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if detector_workers is not None:
        os.environ["OCRSCOUT_DETECTOR_WORKERS"] = str(detector_workers)
    if source_name == "hf_dataset" and source is None:
        raise typer.BadParameter("--source is required with the `hf_dataset` adapter.")

    args: dict[str, Any] = parse_source_args(source_arg or [])
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    cfg = PipelineConfig(
        name="run",
        source=build_source_ref(
            source=source, source_name=source_name, source_args=args, sample=sample, seed=seed),
        reference=(
            AdapterRef(name=reference, args={"root": str(reference_path)} if reference_path else {})
            if reference else None
        ),
        comparisons=parse_comparisons_flag(comparisons),
        models=model_list,
        layout=AdapterRef(name="precomputed", args={"path": layout}) if layout else None,
        export=AdapterRef(name="parquet", args={}),
        sample=sample,
        output_dir=output_dir,
    )
    log.info("ocrscout run: %d model(s) [%s] → %s",
             len(model_list), ",".join(model_list), output_dir / "data")

    result = PipelineEngine().execute(
        cfg, resume=resume,
        parallel_models=max(1, min(parallel_models or 1, max(1, len(model_list)))),
        gpu_budget=gpu_budget, base_port=base_port, proxy_port=proxy_port,
        keep_up=keep_up, batch_concurrency=batch_concurrency,
    )
    _print_summary(result, dest=str(output_dir / "data"))


def _print_summary(result: PipelineResult, *, dest: str) -> None:
    stages = Table(title="ocrscout run — stages", show_lines=False)
    for col, just in [("stage", "left"), ("written", "right"), ("skipped", "right"),
                      ("failed", "right"), ("seconds", "right")]:
        stages.add_column(col, justify=just)
    for s in result.stages:
        stages.add_row(s.stage, str(s.rows_written), str(s.rows_skipped),
                       str(s.rows_failed), f"{s.seconds:.1f}")
    rprint(stages)

    # Per-model breakdown from the most downstream stage that tallied models.
    by_model = next((s.by_model for s in reversed(result.stages) if s.by_model), {})
    if by_model:
        models = Table(title="per model", show_lines=False)
        for col, just in [("model", "left"), ("ok", "right"), ("failed", "right")]:
            models.add_column(col, justify=just)
        for name, tally in by_model.items():
            models.add_row(name, str(tally.written), str(tally.failed))
        rprint(models)
    rprint(f"[dim]Output: {dest}[/dim]")

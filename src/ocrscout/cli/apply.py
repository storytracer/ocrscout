"""`ocrscout apply` — execute a pipeline.yaml file.

Used by ``ocrscout run`` (which writes a pipeline.yaml for
reproducibility) and by SkyPilot/HF workers, which receive their
``PipelineConfig`` as a file in the workdir and invoke
``ocrscout apply pipeline.yaml`` after the runner's setup script has
installed ocrscout. ``--start-idx`` / ``--end-idx`` are merged into the
source args at load time so a single config plus a rank lets each
worker take a non-overlapping slice of the deterministic sample.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
import yaml
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.errors import ProfileNotFound
from ocrscout.log import setup_logging
from ocrscout.types import PipelineConfig

log = logging.getLogger(__name__)


@app.command("apply")
def apply(
    pipeline: Path = typer.Argument(
        ..., help="Path to a pipeline.yaml (PipelineConfig) file.",
    ),
    start_idx: int | None = typer.Option(
        None, "--start-idx",
        help="Override source_args.start_idx; lets SkyPilot/HF workers "
             "take a non-overlapping slice of the deterministic sample.",
    ),
    end_idx: int | None = typer.Option(
        None, "--end-idx",
        help="Override source_args.end_idx (exclusive).",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o",
        help="Override the config's output_dir.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip page ids already in <output_dir>/progress.json. Used "
             "by SkyPilot/HF workers on auto-retry: the second attempt "
             "picks up where the first left off without recomputing.",
    ),
    parallel_models: int = typer.Option(1, "--parallel-models", "-P"),
    batch_concurrency: int | None = typer.Option(
        None, "--batch-concurrency", min=1,
        help="Override the GPU-aware autoscaler's per-profile concurrency "
             "for this worker. Sets concurrent_requests / region_concurrency "
             "and sizes vLLM's KV cache to fit.",
    ),
    detector_workers: int | None = typer.Option(
        None, "--detector-workers", min=1,
        help="Override the CPU detector pool size for layout_chat profiles "
             "on this worker. Default: auto-derived from sched_getaffinity.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Run a pipeline previously serialized to YAML."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if detector_workers is not None:
        os.environ["OCRSCOUT_DETECTOR_WORKERS"] = str(detector_workers)

    try:
        raw = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise typer.BadParameter(f"cannot read {pipeline}: {e}") from e

    try:
        cfg = PipelineConfig.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(f"invalid pipeline config in {pipeline}: {e}") from e

    # Per-worker overrides: slice the source via start_idx/end_idx, and
    # optionally redirect the output_dir (useful when the worker shares
    # a S3 prefix with the orchestrator but writes to a rank-specific
    # subdir).
    if start_idx is not None:
        cfg.source.args["start_idx"] = start_idx
    if end_idx is not None:
        cfg.source.args["end_idx"] = end_idx
    if output_dir is not None:
        cfg.output_dir = output_dir
        cfg.export.args["dest"] = str(
            output_dir / "data" / "train-00000.parquet"
        )

    rprint(f"[bold]Applying pipeline {cfg.name!r}[/bold]")
    log.info(
        "models=%s source=%s output=%s",
        ",".join(cfg.models),
        cfg.source.args.get("path", "?"),
        cfg.output_dir,
    )

    try:
        from ocrscout.pipeline.engine import PipelineEngine
        PipelineEngine().execute(
            cfg,
            resume=resume,
            parallel_models=parallel_models,
            batch_concurrency=batch_concurrency,
        )
    except ProfileNotFound as e:
        raise typer.BadParameter(str(e)) from e

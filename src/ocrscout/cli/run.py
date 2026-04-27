"""`ocrscout run` — execute a pipeline.yaml file."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.pipeline.engine import PipelineEngine


@app.command("run")
def run(
    pipeline: Path = typer.Argument(..., help="Path to a pipeline.yaml file."),
) -> None:
    """Load a pipeline.yaml and (eventually) execute it."""
    cfg = PipelineEngine().load(pipeline)
    rprint(f"[bold]Loaded pipeline {cfg.name!r}:[/bold]")
    rprint(cfg.model_dump(mode="json"))
    rprint(
        "[yellow](stub) pipeline execution arrives in phase 6 of the roadmap; "
        "see CLAUDE.md.[/yellow]"
    )

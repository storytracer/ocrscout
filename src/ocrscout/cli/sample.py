"""`ocrscout sample` — materialize a source sample to ``pages-*.parquet``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from ocrscout.cli import app
from ocrscout.cli._resolve import build_source_ref, parse_source_args
from ocrscout.log import setup_logging
from ocrscout.pipeline.engine import PipelineEngine
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("sample")
def sample_cmd(
    source: str | None = typer.Option(None, "--source", "-s",
        help="Image source: local dir, fsspec URL, or HF Hub dataset id."),
    source_name: str = typer.Option("hf_dataset", "--source-name"),
    source_arg: list[str] = typer.Option(None, "--source-arg"),
    sample: int | None = typer.Option(None, "--sample"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: Path = typer.Option(Path("./data/results"), "--output-dir", "-o"),
    resume: bool = typer.Option(False, "--resume"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Materialize a source sample to data/pages-*.parquet (no OCR, no GPU)."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if source_name == "hf_dataset" and source is None:
        raise typer.BadParameter("--source is required with the `hf_dataset` adapter.")
    args: dict[str, Any] = parse_source_args(source_arg or [])
    cfg = PipelineConfig(
        name="sample", source=build_source_ref(
            source=source, source_name=source_name, source_args=args, sample=sample, seed=seed),
        models=[], comparisons=[], export=AdapterRef(name="parquet", args={}),
        sample=sample, output_dir=output_dir,
    )
    PipelineEngine().execute(cfg, stages=["sample"], resume=resume)

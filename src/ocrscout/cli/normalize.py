"""`ocrscout normalize` — raw parquet → train parquet (normalize + compare, no GPU)."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ocrscout.cli import app
from ocrscout.cli._resolve import parse_comparisons_flag
from ocrscout.log import setup_logging
from ocrscout.pipeline.engine import PipelineEngine
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("normalize")
def normalize_cmd(
    source: Path | None = typer.Option(None, "--source", "-s",
        help="Dir holding data/raw-*.parquet. Defaults to --output-dir."),
    output_dir: Path = typer.Option(Path("./data/results"), "--output-dir", "-o"),
    reference: str | None = typer.Option(None, "--reference"),
    reference_path: Path | None = typer.Option(None, "--reference-path"),
    comparisons: str | None = typer.Option(None, "--comparisons"),
    resume: bool = typer.Option(False, "--resume"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Normalize + compare raw OCR output → data/train-*.parquet (no GPU)."""
    setup_logging(verbosity=verbose, quiet=quiet)
    cfg = PipelineConfig(
        name="normalize", source=AdapterRef(name="pages", args={}), models=[],
        reference=(
            AdapterRef(name=reference, args={"root": str(reference_path)} if reference_path else {})
            if reference else None
        ),
        comparisons=parse_comparisons_flag(comparisons),
        export=AdapterRef(name="parquet", args={}), output_dir=output_dir,
    )
    PipelineEngine().execute(cfg, stages=["normalize"], resume=resume, input_dir=source)

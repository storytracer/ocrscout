"""`ocrscout layout` — run layout detection over a pages parquet → layout parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ocrscout.cli import app
from ocrscout.cli._resolve import parse_source_args
from ocrscout.errors import ProfileNotFound
from ocrscout.log import setup_logging
from ocrscout.pipeline.engine import PipelineEngine
from ocrscout.profile import resolve
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("layout")
def layout_cmd(
    source: Path | None = typer.Option(None, "--source", "-s",
        help="Dir holding data/pages-*.parquet. Defaults to --output-dir."),
    profile: str | None = typer.Option(None, "--profile", "-p",
        help="Layout-aware profile to source the detector config from."),
    detector: str | None = typer.Option(None, "--detector",
        help="Registered LayoutDetector name (alternative to --profile)."),
    detector_arg: list[str] = typer.Option(None, "--detector-arg"),
    detector_workers: int | None = typer.Option(None, "--detector-workers", min=1),
    output_dir: Path = typer.Option(Path("./data/results"), "--output-dir", "-o"),
    resume: bool = typer.Option(False, "--resume"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Detect layout regions for each page → data/layout-*.parquet (CPU only)."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if profile is None and detector is None:
        raise typer.BadParameter("pass --profile or --detector.")

    detector_ref = _detector_ref(profile, detector, parse_source_args(detector_arg or []))
    cfg = PipelineConfig(
        name="layout", source=AdapterRef(name="pages", args={}), models=[],
        comparisons=[], detector=detector_ref,
        export=AdapterRef(name="parquet", args={}), output_dir=output_dir,
    )
    PipelineEngine().execute(
        cfg, stages=["layout"], resume=resume,
        input_dir=source, detector_workers=detector_workers,
    )


def _detector_ref(profile: str | None, detector: str | None, detector_args: dict) -> AdapterRef:
    if profile is not None:
        try:
            prof = resolve(profile)
        except ProfileNotFound as e:
            raise typer.BadParameter(str(e)) from e
        name = detector or prof.layout_detector
        if not name:
            raise typer.BadParameter(
                f"profile {profile!r} has no layout_detector; use --detector.")
        return AdapterRef(name=name, args=detector_args or dict(prof.layout_detector_args or {}))
    return AdapterRef(name=detector, args=detector_args)

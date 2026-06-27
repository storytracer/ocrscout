"""`ocrscout layout` — run layout detection over a pages parquet.

Stage 2 of the decoupled pipeline. Reads pages from a stage parquet (or any
source), runs a CPU :class:`~ocrscout.backends._detector_pool.DetectorPool`, and
writes ``data/layout-*.parquet`` — a superset of the pages columns plus the
detected regions. That parquet feeds ``ocrscout ocr --layout`` (via the
``precomputed`` detector), so layout detection runs once, on CPU, decoupled
from GPU OCR.

The detector config comes from a layout-aware profile (``--profile
glm-ocr-layout``) or directly from ``--detector pp-doclayout-v3``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from ocrscout.cli import app
from ocrscout.cli._resolve import build_source_ref
from ocrscout.cli.run import _construct_source, _parse_source_args
from ocrscout.errors import ProfileNotFound, ScoutError
from ocrscout.exports.schema import LAYOUT_FEATURES
from ocrscout.exports.stages import StageWriter, layout_row
from ocrscout.log import setup_logging
from ocrscout.profile import ModelProfile, resolve
from ocrscout.types import PipelineConfig

log = logging.getLogger(__name__)


@app.command("layout")
def layout_cmd(
    source: str | None = typer.Option(
        None, "--source", "-s",
        help="A stage parquet (sample output) or any source adapter input.",
    ),
    source_name: str = typer.Option("hf_dataset", "--source-name"),
    source_arg: list[str] = typer.Option(None, "--source-arg"),
    profile: str | None = typer.Option(
        None, "--profile", "-p",
        help="Layout-aware profile to source the detector config from "
             "(e.g. glm-ocr-layout). Uses its layout_detector + "
             "layout_detector_args.",
    ),
    detector: str | None = typer.Option(
        None, "--detector",
        help="Registered LayoutDetector name (e.g. pp-doclayout-v3). Alternative "
             "to --profile when you just want a detector with default args.",
    ),
    detector_arg: list[str] = typer.Option(
        None, "--detector-arg",
        help="Repeatable key=value kwargs for the detector (with --detector).",
    ),
    detector_workers: int | None = typer.Option(None, "--detector-workers", min=1),
    sample: int | None = typer.Option(None, "--sample"),
    seed: int = typer.Option(42, "--seed"),
    start_idx: int | None = typer.Option(None, "--start-idx", min=0),
    end_idx: int | None = typer.Option(None, "--end-idx", min=0),
    output_dir: Path = typer.Option(
        Path("./data/results"), "--output-dir", "-o",
        help="Where to write data/layout-*.parquet.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip page_ids already present in data/layout-*.parquet.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Detect layout regions for each page → data/layout-*.parquet (CPU only)."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if profile is None and detector is None:
        raise typer.BadParameter("pass --profile <layout-aware profile> or --detector <name>.")

    prof = _resolve_detector_profile(profile, detector, _parse_source_args(detector_arg or []))
    if detector_workers is not None:
        prof = prof.model_copy(
            update={"backend_args": {**prof.backend_args, "detector_workers": detector_workers}}
        )

    if source_name == "hf_dataset" and source is None:
        raise typer.BadParameter(
            "--source is required with the default `hf_dataset` adapter."
        )
    ref = build_source_ref(
        source=source, source_name=source_name,
        source_args=_parse_source_args(source_arg or []),
        sample=sample, seed=seed, start_idx=start_idx, end_idx=end_idx,
    )
    cfg = PipelineConfig(
        name="layout", source=ref, comparisons=[], models=[prof.name],
        export={"name": "parquet", "args": {}}, output_dir=output_dir,
    )
    try:
        adapter = _construct_source(cfg)
    except (ScoutError, Exception) as e:  # noqa: BLE001
        raise typer.BadParameter(f"failed to build source: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume:
        from ocrscout.exports.parquet import done_page_ids
        done = done_page_ids(output_dir, "layout")
        if done:
            log.info("Resume: %d page(s) already detected — skipping them", len(done))

    pages = [p for p in adapter.iter_pages() if p.page_id not in done]

    from ocrscout.backends._detector_pool import DetectorPool
    pool = DetectorPool(prof, log_prefix="[layout]")

    writer = StageWriter(output_dir, LAYOUT_FEATURES, "layout")
    ok = failed = 0
    try:
        for page, regions, secs, err in pool.map(pages):
            writer.write(
                layout_row(
                    page, regions,
                    detector=prof.layout_detector or "",
                    detect_seconds=round(secs, 4),
                    detect_error=err,
                )
            )
            if err:
                log.warning("layout: page %s detect error: %s", page.file_id, err)
                failed += 1
            else:
                ok += 1
    finally:
        writer.close()

    log.info(
        "Detected layout for %d page(s) → %s (%d failed)",
        ok, output_dir / "data", failed,
    )


def _resolve_detector_profile(
    profile: str | None, detector: str | None, detector_args: dict[str, Any]
) -> ModelProfile:
    """Get a ModelProfile carrying the detector config.

    ``--profile`` resolves a real (layout-aware) profile. ``--detector``
    synthesizes a minimal profile whose only meaningful fields are
    ``layout_detector`` + ``layout_detector_args`` (backend ``tesseract`` so the
    layout_chat cross-field validator doesn't fire).
    """
    if profile is not None:
        try:
            prof = resolve(profile)
        except ProfileNotFound as e:
            raise typer.BadParameter(str(e)) from e
        if not prof.layout_detector:
            raise typer.BadParameter(
                f"profile {profile!r} has no layout_detector; pass a "
                "layout-aware profile (e.g. glm-ocr-layout) or use --detector."
            )
        if detector is not None and detector != prof.layout_detector:
            prof = prof.model_copy(update={"layout_detector": detector})
        return prof
    return ModelProfile(
        name=f"layout:{detector}",
        model_id="n/a",
        backend="tesseract",
        runtime="cpu",
        output_format="layout_json",
        normalizer="layout_json",
        layout_detector=detector,
        layout_detector_args=detector_args,
    )

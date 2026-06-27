"""`ocrscout normalize` — turn raw OCR output into the final results parquet.

Stage 4 of the decoupled pipeline. Reads ``data/raw-*.parquet`` (the
``ocrscout ocr`` output), reconstructs ``(PageImage, RawOutput, cost columns)``
per row, normalizes each into a ``DoclingDocument``, runs reference comparisons,
and writes the rich ``data/train-*.parquet`` — exactly the artifact a fused
``ocrscout run`` produces. No runner is launched; no GPU is touched, so this can
re-normalize / re-compare existing OCR output for free.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from ocrscout import registry
from ocrscout.cli import app
from ocrscout.cli.run import (
    _parse_comparisons_flag,
    _resolve_comparisons,
    normalize_compare_to_record,
)
from ocrscout.errors import ProfileNotFound
from ocrscout.exports.parquet import done_pairs_from_parquet
from ocrscout.exports.stages import RAW_COST_KEYS, read_raw_rows
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.log import setup_logging
from ocrscout.profile import resolve
from ocrscout.sources.pages import page_from_row
from ocrscout.types import RawOutput

log = logging.getLogger(__name__)


@app.command("normalize")
def normalize_cmd(
    source: Path | None = typer.Option(
        None, "--source", "-s",
        help="Directory (or parquet) holding data/raw-*.parquet. Defaults to "
             "--output-dir.",
    ),
    output_dir: Path = typer.Option(
        Path("./data/results"), "--output-dir", "-o",
        help="Where to write data/train-*.parquet (and default raw source).",
    ),
    reference: str | None = typer.Option(
        None, "--reference", help="Reference adapter name (e.g. bhl_ocr)."
    ),
    reference_path: Path | None = typer.Option(None, "--reference-path"),
    comparisons: str | None = typer.Option(
        None, "--comparisons",
        help="Comparisons to run per (page, model). Default: all whose data is "
             "available when a reference is set. `none` to skip.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip (page, model) pairs already in data/train-*.parquet.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Normalize + compare raw OCR output into data/train-*.parquet (no GPU)."""
    setup_logging(verbosity=verbose, quiet=quiet)

    raw_source = source or output_dir
    rows = read_raw_rows(raw_source)
    if not rows:
        raise typer.BadParameter(
            f"no data/raw-*.parquet rows found under {raw_source}. Run "
            "`ocrscout ocr` first."
        )
    log.info("Normalizing %d raw row(s) from %s", len(rows), raw_source)

    reference_adapter: ReferenceAdapter | None = None
    if reference is not None:
        ref_cls = registry.get("references", reference)
        ref_args = {"root": str(reference_path)} if reference_path else {}
        reference_adapter = ref_cls(**ref_args)
        log.info("Reference adapter: %s", reference)

    active_comparisons = _resolve_comparisons(
        _parse_comparisons_flag(comparisons),
        has_reference=reference_adapter is not None,
    )

    done: dict[str, set[str]] = {}
    if resume:
        done = done_pairs_from_parquet(output_dir)

    # Resolve profile + normalizer once per model (cached).
    profiles: dict[str, Any] = {}
    normalizers: dict[str, Any] = {}

    def _resolve_model(model: str) -> bool:
        if model in profiles:
            return True
        try:
            profile = resolve(model)
            profiles[model] = profile
            normalizers[model] = registry.get("normalizers", profile.normalizer)()
            return True
        except ProfileNotFound as e:
            log.error("skipping model %r: %s", model, e)
            profiles[model] = None
            return False

    exporter_cls = registry.get("exports", "parquet")
    exporter = exporter_cls()
    output_dir.mkdir(parents=True, exist_ok=True)
    from ocrscout.exports.layout import parquet_dest
    exporter.open(str(parquet_dest(output_dir)))

    ok = failed = skipped = 0
    try:
        for row in rows:
            model = row.get("model")
            page_id = str(row.get("page_id"))
            if not model:
                continue
            if page_id in done.get(model, set()):
                skipped += 1
                continue
            if not _resolve_model(model) or profiles[model] is None:
                failed += 1
                continue

            page = page_from_row(row)
            if page is None:
                failed += 1
                continue
            raw = RawOutput(
                page_id=page_id,
                output_format=row.get("output_format") or "markdown",
                payload=row.get("raw_payload") or "",
                tokens=row.get("tokens"),
                error=row.get("error"),
            )
            if raw.error:
                # Error pages produced no usable output — no train row, same
                # as the fused run. Counted as a failure.
                failed += 1
                continue

            cost_ctx = {k: row.get(k) for k in RAW_COST_KEYS}
            record = normalize_compare_to_record(
                page=page,
                raw=raw,
                model_name=model,
                profile=profiles[model],
                normalizer=normalizers[model],
                reference_adapter=reference_adapter,
                comparisons=active_comparisons,
                cost_ctx=cost_ctx,
            )
            if record is None:
                failed += 1
                continue
            exporter.write(record)
            ok += 1
    finally:
        exporter.close()

    log.info(
        "Normalized %d page(s) → %s (%d failed%s)",
        ok, output_dir / "data", failed,
        f", {skipped} already done" if skipped else "",
    )

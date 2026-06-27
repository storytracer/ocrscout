"""`ocrscout sample` — materialize a source sample to a pages parquet.

Stage 1 of the decoupled pipeline: iterate a real source adapter's
``iter_pages()`` and write the page identity + ``source_uri`` (no image bytes)
to ``<output_dir>/data/pages-*.parquet``. Downstream stages (``layout`` /
``ocr`` / ``run``) read that parquet via the ``pages`` source adapter, so a
sample is fixed once and reused without re-listing the corpus.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import typer

from ocrscout import registry
from ocrscout.cli import app
from ocrscout.cli._resolve import build_source_ref
from ocrscout.cli.run import _parse_source_args
from ocrscout.errors import ScoutError
from ocrscout.exports.schema import PAGES_FEATURES
from ocrscout.exports.stages import StageWriter, page_to_row
from ocrscout.log import setup_logging

log = logging.getLogger(__name__)


@app.command("sample")
def sample_cmd(
    source: str | None = typer.Option(
        None, "--source", "-s",
        help="Image source: a local dir, fsspec URL, or HF Hub dataset id "
             "(used by the default `hf_dataset` adapter; ignored by `bhl`).",
    ),
    source_name: str = typer.Option(
        "hf_dataset", "--source-name",
        help="Source adapter name (`hf_dataset`, `bhl`, …).",
    ),
    source_arg: list[str] = typer.Option(
        None, "--source-arg",
        help="Repeatable `key=value` kwargs forwarded to the source adapter.",
    ),
    sample: int | None = typer.Option(
        None, "--sample", help="Take a random subset of N pages."
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed for --sample."),
    output_dir: Path = typer.Option(
        Path("./data/results"), "--output-dir", "-o",
        help="Where to write data/pages-*.parquet.",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip page_ids already present in data/pages-*.parquet.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Materialize a source sample to data/pages-*.parquet (no OCR, no GPU)."""
    setup_logging(verbosity=verbose, quiet=quiet)

    source_args: dict[str, Any] = _parse_source_args(source_arg or [])
    if source_name == "hf_dataset" and source is None:
        raise typer.BadParameter(
            "--source is required with the default `hf_dataset` adapter; pass "
            "--source-name to select another (e.g. `bhl`)."
        )
    ref = build_source_ref(
        source=source, source_name=source_name, source_args=source_args,
        sample=sample, seed=seed,
    )
    try:
        source_cls = registry.get("sources", ref.name)
        adapter = source_cls(**ref.args)
    except (ScoutError, Exception) as e:  # noqa: BLE001 — surface ctor errors
        raise typer.BadParameter(f"failed to build source {ref.name!r}: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume:
        from ocrscout.exports.parquet import done_page_ids
        done = done_page_ids(output_dir, "pages")
        if done:
            log.info("Resume: %d page(s) already sampled — skipping them", len(done))

    pages = adapter.iter_pages()
    if sample is not None:
        pages = itertools.islice(pages, sample)

    writer = StageWriter(output_dir, PAGES_FEATURES, "pages")
    written = skipped = 0
    try:
        for page in pages:
            if page.page_id in done:
                continue
            if not page.source_uri:
                log.warning(
                    "sample: page %r has no source_uri (un-reconstructable "
                    "image, e.g. an HF streaming bytes-only row); skipping.",
                    page.page_id,
                )
                skipped += 1
                continue
            writer.write(page_to_row(page))
            written += 1
            if written % 500 == 0:
                log.info("sampled %d pages…", written)
    finally:
        writer.close()

    log.info(
        "Sampled %d page(s) → %s%s",
        written, output_dir / "data",
        f" ({skipped} skipped: no source_uri)" if skipped else "",
    )
    if written == 0:
        log.warning("No pages written. Check the source filters / --sample.")

"""SampleStage — source → ``pages-*.parquet`` (no runner, no GPU)."""

from __future__ import annotations

import itertools
import logging
import time
from typing import ClassVar

from ocrscout.errors import ScoutError
from ocrscout.io import PageRow, ResumeMode
from ocrscout.pipeline.components import build_source
from ocrscout.pipeline.context import ExecutionContext
from ocrscout.pipeline.stage import Stage, StageIO, StageResult

log = logging.getLogger(__name__)


class SampleStage(Stage):
    name: ClassVar[str] = "sample"
    io: ClassVar[StageIO] = StageIO(reads="source", writes="pages")

    def execute(self, ctx: ExecutionContext) -> StageResult:
        if ctx.source is None:
            raise ScoutError("SampleStage requires a source in the ExecutionContext")
        source = build_source(ctx.source)
        resume = ctx.store.resume("pages", ResumeMode.PAGE if ctx.resume is not ResumeMode.OFF else ResumeMode.OFF)

        # Volume sidecar (cheap; carries no images).
        ctx.store.write_volumes(source.iter_volumes())

        pages = source.iter_pages()
        if ctx.sample is not None:
            pages = itertools.islice(pages, ctx.sample)

        t0 = time.perf_counter()
        written = skipped = 0
        with ctx.store.writer("pages") as writer:
            for page in pages:
                if resume.seen(page.page_id):
                    skipped += 1
                    continue
                if not page.source_uri:
                    log.warning(
                        "sample: page %r has no source_uri (un-reconstructable); skipping",
                        page.page_id,
                    )
                    skipped += 1
                    continue
                writer.write(PageRow.from_page(page))
                written += 1
                if written % 500 == 0:
                    log.info("sampled %d pages…", written)

        log.info("sample: wrote %d page(s)%s", written, f", skipped {skipped}" if skipped else "")
        return StageResult(
            stage=self.name, rows_written=written, rows_skipped=skipped,
            seconds=time.perf_counter() - t0,
        )

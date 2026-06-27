"""LayoutStage — pages → ``layout-*.parquet`` (CPU detector pool, no runner).

Owns the CPU detector pool (lifted out of the ``layout_chat`` backend so it's
reusable standalone). Writes detected regions per page; that parquet then feeds
``OcrStage`` via the ``precomputed`` detector, decoupling detection from OCR.
The detector is specified directly (name + args) on the context — no synthetic
``ModelProfile``.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

from ocrscout.errors import ScoutError
from ocrscout.io import LayoutRow, ResumeMode
from ocrscout.pipeline.context import ExecutionContext
from ocrscout.pipeline.stage import Stage, StageIO, StageResult

log = logging.getLogger(__name__)


class LayoutStage(Stage):
    name: ClassVar[str] = "layout"
    io: ClassVar[StageIO] = StageIO(reads="pages", writes="layout")

    def execute(self, ctx: ExecutionContext) -> StageResult:
        if ctx.detector is None:
            raise ScoutError("LayoutStage requires a detector in the ExecutionContext")
        from ocrscout.backends._detector_pool import (
            DetectorPool,
            default_detector_workers,
            default_torch_threads,
        )

        n_workers = ctx.detector_workers or default_detector_workers()
        pool = DetectorPool(
            detector_name=ctx.detector.name,
            detector_args=ctx.detector.args,
            n_workers=n_workers,
            torch_threads=default_torch_threads(n_workers),
        )

        resume = ctx.store.resume(
            "layout", ResumeMode.PAGE if ctx.resume is not ResumeMode.OFF else ResumeMode.OFF
        )
        pages = [
            page
            for row in ctx.store.reader("pages")
            if not resume.seen(row.page_id)
            and (page := row.to_page(ctx.storage_options)) is not None
        ]

        t0 = time.perf_counter()
        written = failed = 0
        with ctx.store.writer("layout") as writer:
            for page, regions, secs, err in pool.map(pages):
                writer.write(
                    LayoutRow.from_detection(
                        page, regions, detector=ctx.detector.name,
                        detect_seconds=round(secs, 4), detect_error=err,
                    )
                )
                if err:
                    log.warning("layout: page %s detect error: %s", page.file_id, err)
                    failed += 1
                else:
                    written += 1

        log.info("layout: wrote %d page(s), %d failed", written, failed)
        return StageResult(
            stage=self.name, rows_written=written, rows_failed=failed,
            seconds=time.perf_counter() - t0,
        )

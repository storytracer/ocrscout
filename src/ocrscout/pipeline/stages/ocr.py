"""OcrStage — pages → ``raw-*.parquet`` (the only runner-requiring stage).

Reads page rows, reconstructs lazy ``PageImage``s, runs each requested model's
``ModelBackend`` against the runner-provided proxy, and writes raw OCR output
plus the per-page cost/autoscaler columns. Normalization is deferred to the
NormalizeStage — error pages are written (with ``error`` set) so resume skips
them and downstream stages can see the failure.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

from ocrscout.errors import BackendError, ProfileNotFound
from ocrscout.io import RawRow, ResumeMode
from ocrscout.orchestration.autoscale import AutoscaleContext
from ocrscout.pipeline.components import resolve_profile
from ocrscout.pipeline.context import ExecutionContext
from ocrscout.pipeline.cost import finalize_cost_columns
from ocrscout.pipeline.stage import ModelTally, Stage, StageIO, StageResult
from ocrscout.registry import registry

log = logging.getLogger(__name__)


class OcrStage(Stage):
    name: ClassVar[str] = "ocr"
    io: ClassVar[StageIO] = StageIO(reads="pages", writes="raw")
    requires_runner: ClassVar[bool] = True

    def execute(self, ctx: ExecutionContext) -> StageResult:
        if ctx.runner is None:
            raise BackendError("OcrStage requires a runner (no proxy in ExecutionContext)")
        # Read precomputed regions when a layout artifact is present (Phase 5
        # attaches them to PageImage.regions); else plain pages.
        input_store = ctx.input_store()
        input_prefix = "layout" if input_store.has("layout") else "pages"
        autoscale = ctx.runner.autoscale or AutoscaleContext()

        resume = ctx.store.resume(
            "raw", ResumeMode.PAGE_MODEL if ctx.resume is not ResumeMode.OFF else ResumeMode.OFF
        )

        t0 = time.perf_counter()
        tally: dict[str, ModelTally] = {}
        written = skipped = failed = 0

        with ctx.store.writer("raw") as writer:
            for model in ctx.models:
                t = tally.setdefault(model, ModelTally())
                try:
                    profile = resolve_profile(model)
                    backend = registry.get("backends", profile.backend)()
                except ProfileNotFound as e:
                    log.error("[%s] %s", model, e)
                    continue

                pages = []
                for row in input_store.reader(input_prefix):
                    if resume.seen(row.page_id, model):
                        skipped += 1
                        t.skipped += 1
                        continue
                    page = row.to_page(ctx.storage_options)
                    if page is not None:
                        pages.append(page)
                if not pages:
                    continue

                try:
                    inv = backend.prepare(profile)
                except BackendError as e:
                    log.error("[%s] prepare failed: %s", model, e)
                    continue

                decision = autoscale.for_profile(model)
                log.info("[%s] OCR over %d page(s)", model, len(pages))
                try:
                    for page, raw in backend.run(inv, pages):
                        cost = finalize_cost_columns(page.page_id, gpu=ctx.gpu, decision=decision)
                        writer.write(RawRow.from_output(page, model, raw, cost))
                        if raw.error:
                            log.warning("[%s] page %s error: %s", model, page.file_id, raw.error)
                            failed += 1
                            t.failed += 1
                        else:
                            written += 1
                            t.written += 1
                except BackendError as e:
                    log.error("[%s] backend failed: %s", model, e)

        log.info("ocr: wrote %d raw row(s), %d failed%s", written, failed,
                 f", {skipped} already done" if skipped else "")
        return StageResult(
            stage=self.name, rows_written=written, rows_skipped=skipped,
            rows_failed=failed, by_model=tally, seconds=time.perf_counter() - t0,
        )

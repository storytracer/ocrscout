"""NormalizeStage — ``raw-*.parquet`` → ``train-*.parquet`` (no runner, no GPU).

Reads raw OCR output, reconstructs ``(PageImage, RawOutput, cost columns)`` per
row, normalizes + compares via :func:`normalize_one`, and writes the rich
results parquet. Re-runnable for free — re-normalize / re-compare existing OCR
output without paying for inference.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

from ocrscout.errors import ProfileNotFound
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.io import RawRow, ResumeMode
from ocrscout.pipeline.components import (
    build_reference,
    resolve_comparisons,
    resolve_normalizer,
    resolve_profile,
)
from ocrscout.pipeline.context import ExecutionContext
from ocrscout.pipeline.normalize_ops import normalize_one
from ocrscout.pipeline.stage import ModelTally, Stage, StageIO, StageResult
from ocrscout.profile import ModelProfile

log = logging.getLogger(__name__)


class NormalizeStage(Stage):
    name: ClassVar[str] = "normalize"
    io: ClassVar[StageIO] = StageIO(reads="raw", writes="train")

    def execute(self, ctx: ExecutionContext) -> StageResult:
        reference = build_reference(ctx.reference)
        comparisons = resolve_comparisons(
            ctx.comparison_names, has_reference=reference is not None
        )
        resume = ctx.store.resume(
            "train", ResumeMode.PAGE_MODEL if ctx.resume is not ResumeMode.OFF else ResumeMode.OFF
        )

        profiles: dict[str, ModelProfile | None] = {}
        normalizers: dict[str, Normalizer] = {}

        def resolved(model: str) -> bool:
            if model in profiles:
                return profiles[model] is not None
            try:
                prof = resolve_profile(model)
            except ProfileNotFound as e:
                log.error("normalize: skipping model %r: %s", model, e)
                profiles[model] = None
                return False
            profiles[model] = prof
            normalizers[model] = resolve_normalizer(prof)
            return True

        t0 = time.perf_counter()
        tally: dict[str, ModelTally] = {}
        written = skipped = failed = 0

        def bump(model: str, field: str) -> None:
            t = tally.setdefault(model, ModelTally())
            setattr(t, field, getattr(t, field) + 1)

        with ctx.store.writer("train") as writer:
            row: RawRow
            for row in ctx.input_store().reader("raw"):
                model = row.model
                if resume.seen(row.page_id, model):
                    skipped += 1
                    bump(model, "skipped")
                    continue
                if row.error:
                    failed += 1
                    bump(model, "failed")
                    continue
                if not resolved(model):
                    failed += 1
                    bump(model, "failed")
                    continue
                page = row.to_page(ctx.storage_options)
                if page is None:
                    failed += 1
                    bump(model, "failed")
                    continue
                result = normalize_one(
                    page=page, raw=row.to_raw_output(), model_name=model,
                    profile=profiles[model], normalizer=normalizers[model],
                    cost=row.cost_columns(), reference_adapter=reference,
                    comparisons=comparisons,
                )
                if result is None:
                    failed += 1
                    bump(model, "failed")
                    continue
                writer.write(result)
                written += 1
                bump(model, "written")

        log.info("normalize: wrote %d row(s), %d failed%s", written, failed,
                 f", {skipped} already done" if skipped else "")
        return StageResult(
            stage=self.name, rows_written=written, rows_skipped=skipped,
            rows_failed=failed, by_model=tally, seconds=time.perf_counter() - t0,
        )

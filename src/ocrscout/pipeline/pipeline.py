"""Pipeline — an ordered composition of stages.

The fused ``run`` is just ``Pipeline([Sample, (Layout), Ocr, Normalize])``; a
single stage command is a one-stage pipeline. ``requires_runner`` is true iff
any stage needs a proxy, which is how the Provisioner decides whether to stand
one up at all.
"""

from __future__ import annotations

from pydantic import BaseModel

from ocrscout.pipeline.context import ExecutionContext
from ocrscout.pipeline.stage import Stage, StageResult
from ocrscout.pipeline.stages import NormalizeStage, SampleStage
from ocrscout.pipeline.stages.ocr import OcrStage


class PipelineResult(BaseModel):
    stages: list[StageResult]

    @property
    def rows_written(self) -> int:
        return sum(s.rows_written for s in self.stages)


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        if not stages:
            raise ValueError("Pipeline needs at least one stage")
        self.stages = stages

    @property
    def requires_runner(self) -> bool:
        return any(s.requires_runner for s in self.stages)

    def execute(self, ctx: ExecutionContext) -> PipelineResult:
        """Run every stage in order against the same context (no-runner path).

        The runner-aware path (per-chunk launch/teardown) is the Provisioner's
        job; it calls into the individual stages.
        """
        return PipelineResult(stages=[s.execute(ctx) for s in self.stages])


def fused_run(*, with_layout: bool = False, with_normalize: bool = True) -> Pipeline:
    """The default ``ocrscout run`` pipeline."""
    from ocrscout.pipeline.stages.layout import LayoutStage  # local: optional

    stages: list[Stage] = [SampleStage()]
    if with_layout:
        stages.append(LayoutStage())
    stages.append(OcrStage())
    if with_normalize:
        stages.append(NormalizeStage())
    return Pipeline(stages)

"""PipelineEngine — build a Pipeline + ExecutionContext from config and run it.

The single entry point the CLI commands call: turn a ``PipelineConfig`` (plus
runtime flags) into the stage composition and the typed context, then dispatch
through the ``Provisioner`` (which only stands up a runner when a stage needs
one). Replaces the old ``run_pipeline`` god function.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ocrscout import state as state_mod
from ocrscout.errors import PipelineError
from ocrscout.io import ParquetStore, ResumeMode
from ocrscout.pipeline.context import ExecutionContext
from ocrscout.pipeline.pipeline import Pipeline, PipelineResult, fused_run
from ocrscout.pipeline.stage import Stage
from ocrscout.registry import registry
from ocrscout.types import PipelineConfig

log = logging.getLogger(__name__)

_STAGE_CLASSES: dict[str, type[Stage]] = {}


def _stage_classes() -> dict[str, type[Stage]]:
    if not _STAGE_CLASSES:
        from ocrscout.pipeline.stages.layout import LayoutStage
        from ocrscout.pipeline.stages.normalize import NormalizeStage
        from ocrscout.pipeline.stages.ocr import OcrStage
        from ocrscout.pipeline.stages.sample import SampleStage

        _STAGE_CLASSES.update(
            sample=SampleStage, layout=LayoutStage, ocr=OcrStage, normalize=NormalizeStage,
        )
    return _STAGE_CLASSES


class PipelineEngine:
    def load(self, path: str | Path) -> PipelineConfig:
        p = Path(path)
        if not p.is_file():
            raise PipelineError(f"pipeline file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise PipelineError(f"invalid YAML in {p}: {e}") from e
        if not isinstance(data, dict):
            raise PipelineError(f"pipeline {p} must be a YAML mapping")
        try:
            return PipelineConfig.model_validate(data)
        except Exception as e:  # noqa: BLE001
            raise PipelineError(f"invalid pipeline config in {p}: {e}") from e

    def build_pipeline(self, stages: list[str]) -> Pipeline:
        classes = _stage_classes()
        try:
            return Pipeline([classes[name]() for name in stages])
        except KeyError as e:
            raise PipelineError(f"unknown stage {e}; known: {sorted(classes)}") from e

    def build_context(
        self,
        config: PipelineConfig,
        *,
        resume: bool = False,
        input_dir: Path | None = None,
        detector_workers: int | None = None,
        storage_options: dict | None = None,
    ) -> ExecutionContext:
        return ExecutionContext(
            output_dir=config.output_dir,
            store=ParquetStore(config.output_dir),
            input_dir=input_dir,
            resume=ResumeMode.PAGE_MODEL if resume else ResumeMode.OFF,
            models=tuple(config.models),
            source=config.source,
            reference=config.reference,
            comparison_names=config.comparisons,
            detector=config.detector,
            detector_workers=detector_workers,
            layout=config.layout,
            sample=config.sample,
            gpu=state_mod.read_config().gpu,
            storage_options=storage_options,
        )

    def execute(
        self,
        config: PipelineConfig,
        *,
        stages: list[str] | None = None,
        with_layout: bool = False,
        resume: bool = False,
        input_dir: Path | None = None,
        detector_workers: int | None = None,
        parallel_models: int = 1,
        gpu_budget: float = 0.85,
        base_port: int = 8000,
        proxy_port: int = 4000,
        keep_up: bool = False,
        batch_concurrency: int | None = None,
        runner_name: str = "local",
    ) -> PipelineResult:
        pipeline = (
            self.build_pipeline(stages) if stages is not None
            else fused_run(with_layout=with_layout)
        )
        ctx = self.build_context(
            config, resume=resume, input_dir=input_dir, detector_workers=detector_workers,
        )

        if not pipeline.requires_runner:
            return pipeline.execute(ctx)

        from ocrscout.orchestration.provisioner import Provisioner

        runner = registry.get("runners", runner_name)()
        provisioner = Provisioner(
            runner, gpu_budget=gpu_budget, base_port=base_port,
            proxy_port=proxy_port, keep_up=keep_up, batch_concurrency=batch_concurrency,
        )
        return provisioner.run(pipeline, ctx, parallel_models=parallel_models)

    def execute_on_proxy(
        self,
        config: PipelineConfig,
        *,
        proxy_url: str,
        autoscale: object | None = None,
        stages: list[str] | None = None,
        with_layout: bool = False,
        resume: bool = False,
        input_dir: Path | None = None,
        detector_workers: int | None = None,
    ) -> PipelineResult:
        """Run against an already-launched proxy (submit→worker, benchmark).

        No Provisioner launch/teardown — the caller owns the runner lifecycle;
        the stages run once against the supplied proxy URL.
        """
        from ocrscout.pipeline.context import RunnerContext

        pipeline = (
            self.build_pipeline(stages) if stages is not None
            else fused_run(with_layout=with_layout)
        )
        ctx = self.build_context(
            config, resume=resume, input_dir=input_dir, detector_workers=detector_workers,
        ).with_runner(RunnerContext(proxy_url=proxy_url, autoscale=autoscale))
        return pipeline.execute(ctx)

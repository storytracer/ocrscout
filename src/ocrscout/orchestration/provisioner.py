"""Provisioner — owns runner lifecycle so stages don't have to.

Separates *what to run* (the Pipeline of Stages) from *standing up the compute*
(launch a proxy + vLLM stack per model chunk, tear it down after). Non-runner
stages (sample/layout/normalize) run once against the base context; the
runner-requiring stage (ocr) runs per chunk against a context bound to that
chunk's launched proxy.
"""

from __future__ import annotations

import json
import logging
import os

from ocrscout.interfaces.runner import Runner
from ocrscout.orchestration.autoscale import AutoscaleContext
from ocrscout.orchestration.plan import RunnerChunk, RunnerPlan
from ocrscout.pipeline.context import ExecutionContext, RunnerContext
from ocrscout.pipeline.pipeline import Pipeline, PipelineResult
from ocrscout.pipeline.stage import Stage, StageResult

log = logging.getLogger(__name__)

_PROXY_ENV = "OCRSCOUT_VLLM_URL"
_OVERRIDES_ENV = "OCRSCOUT_BACKEND_OVERRIDES"


def _publish_overrides(autoscale: AutoscaleContext) -> None:
    """Bridge: backends still read per-profile concurrency from this env var
    (``litellm._state_override``). Mirror the autoscaler's typed decisions into
    it so they fire the right concurrency. Removed when backends take the
    decision via ``BackendInvocation``."""
    payload: dict[str, dict[str, int]] = {}
    for name, d in autoscale.decisions.items():
        entry = {
            k: v
            for k, v in (
                ("concurrent_requests", d.concurrent_requests),
                ("region_concurrency", d.region_concurrency),
                ("kv_cache_memory_bytes", d.kv_cache_memory_bytes),
            )
            if v
        }
        if entry:
            payload[name] = entry
    if payload:
        os.environ[_OVERRIDES_ENV] = json.dumps(payload)
    else:
        os.environ.pop(_OVERRIDES_ENV, None)


class Provisioner:
    def __init__(
        self,
        runner: Runner,
        *,
        gpu_budget: float = 0.85,
        base_port: int = 8000,
        proxy_port: int = 4000,
        keep_up: bool = False,
        batch_concurrency: int | None = None,
    ) -> None:
        self.runner = runner
        self.gpu_budget = gpu_budget
        self.base_port = base_port
        self.proxy_port = proxy_port
        self.keep_up = keep_up
        self.batch_concurrency = batch_concurrency

    def run(
        self, pipeline: Pipeline, ctx: ExecutionContext, *, parallel_models: int = 1
    ) -> PipelineResult:
        if not pipeline.requires_runner:
            return pipeline.execute(ctx)

        results: list[StageResult] = []
        plan = RunnerPlan.from_models(ctx.models, parallel_models=parallel_models)
        for stage in pipeline.stages:
            if not stage.requires_runner:
                results.append(stage.execute(ctx))
                continue
            for chunk in plan.chunks:
                results.append(self._run_chunk(stage, ctx, chunk))
        return PipelineResult(stages=results)

    def _run_chunk(
        self, stage: Stage, ctx: ExecutionContext, chunk: RunnerChunk
    ) -> StageResult:
        if not chunk.needs_launch:
            # CPU profiles (e.g. tesseract) run in-process, no proxy.
            rctx = ctx.with_runner(RunnerContext(proxy_url=""), models=chunk.models)
            return stage.execute(rctx)

        log.info("launching %s chunk: %s", chunk.runtime, ",".join(chunk.models))
        handle = self.runner.launch(
            models=list(chunk.models),
            base_port=self.base_port,
            proxy_port=self.proxy_port,
            gpu_budget=self.gpu_budget,
            persistent=self.keep_up,
            batch_concurrency=self.batch_concurrency,
        )
        # Bridge: unmodified backends still read the proxy URL from the env;
        # the stage itself reads ctx.runner.proxy_url. (Env bridge removed when
        # backends are reworked to take the proxy via BackendInvocation.)
        os.environ[_PROXY_ENV] = handle.proxy_url
        autoscale = AutoscaleContext.from_handle_extra(handle.extra)
        _publish_overrides(autoscale)
        rctx = ctx.with_runner(
            RunnerContext(proxy_url=handle.proxy_url, autoscale=autoscale),
            models=chunk.models,
        )
        try:
            return stage.execute(rctx)
        finally:
            os.environ.pop(_OVERRIDES_ENV, None)
            if not self.keep_up:
                try:
                    self.runner.down()
                except Exception as e:  # noqa: BLE001
                    log.warning("runner teardown reported error: %s", e)

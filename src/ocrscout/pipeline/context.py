"""ExecutionContext — the single typed object every stage receives.

Replaces the env-var (`OCRSCOUT_VLLM_URL`, `OCRSCOUT_BACKEND_OVERRIDES`) and
`state.yaml` re-resolution side channels in the hot path: a stage reads
everything it needs off this immutable context. Stages that don't need a runner
get ``runner is None`` and physically cannot reach the proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ocrscout.io import ParquetStore, ResumeMode
from ocrscout.state import GpuConfig
from ocrscout.types import AdapterRef


@dataclass(frozen=True)
class RunnerContext:
    """What the Provisioner hands a runner-requiring stage. ``autoscale`` is the
    typed per-profile decision snapshot (fleshed out in the orchestration
    phase); kept loosely typed here so the no-runner stages don't depend on it.
    """

    proxy_url: str
    autoscale: Any | None = None


@dataclass(frozen=True)
class ExecutionContext:
    output_dir: Path
    store: ParquetStore
    resume: ResumeMode = ResumeMode.OFF
    models: tuple[str, ...] = ()
    source: AdapterRef | None = None
    reference: AdapterRef | None = None
    comparison_names: list[str] | None = None
    # Detector to run in LayoutStage (name + args). Distinct from ``layout``,
    # which injects *precomputed* regions into OcrStage.
    detector: AdapterRef | None = None
    detector_workers: int | None = None
    layout: AdapterRef | None = None
    sample: int | None = None
    seed: int = 42
    start_idx: int | None = None
    end_idx: int | None = None
    gpu: GpuConfig | None = None
    storage_options: dict[str, Any] | None = None
    runner: RunnerContext | None = None

    def with_runner(
        self, runner: RunnerContext, *, models: tuple[str, ...] | None = None
    ) -> ExecutionContext:
        """Derive a context bound to a launched runner (and optionally a model
        chunk) — used by the Provisioner per vLLM chunk."""
        return replace(
            self, runner=runner, models=models if models is not None else self.models
        )

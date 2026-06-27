"""RunnerPlan — partition models by runtime and build launch chunks.

Pulls the model-major chunking and runtime partitioning out of the old
``run_pipeline`` god function. A ``RunnerChunk`` is one launch unit: ``vllm``
chunks size up to ``parallel_models`` (model-major dispatch — one serve at a
time by default), ``hosted`` shares a single proxy-only launch, and ``cpu``
runs in-process with no launch at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from ocrscout.profile import resolve


@dataclass(frozen=True)
class RunnerChunk:
    models: tuple[str, ...]
    runtime: str  # "vllm" | "hosted" | "cpu"

    @property
    def needs_launch(self) -> bool:
        return self.runtime in ("vllm", "hosted")


@dataclass(frozen=True)
class RunnerPlan:
    chunks: tuple[RunnerChunk, ...]

    @classmethod
    def from_models(cls, models: tuple[str, ...], *, parallel_models: int = 1) -> RunnerPlan:
        by_runtime: dict[str, list[str]] = {"vllm": [], "hosted": [], "cpu": []}
        for name in models:
            runtime = resolve(name).runtime
            by_runtime.setdefault(runtime, []).append(name)

        chunks: list[RunnerChunk] = []
        size = max(1, parallel_models)
        vllm = by_runtime.get("vllm", [])
        for i in range(0, len(vllm), size):
            chunks.append(RunnerChunk(tuple(vllm[i : i + size]), "vllm"))
        if by_runtime.get("hosted"):
            chunks.append(RunnerChunk(tuple(by_runtime["hosted"]), "hosted"))
        if by_runtime.get("cpu"):
            chunks.append(RunnerChunk(tuple(by_runtime["cpu"]), "cpu"))
        return cls(tuple(chunks))

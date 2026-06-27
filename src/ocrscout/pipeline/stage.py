"""The Stage abstraction — the trunk of the staged pipeline.

A Stage reads one parquet artifact (or a source) and writes the next, owning its
own resume + IO via the ``ParquetStore`` on the ``ExecutionContext``. It reads no
env vars and no ``state.yaml``. ``requires_runner`` is the single source of truth
for whether the Provisioner must stand up a LiteLLM proxy — only ``OcrStage``
sets it. A ``Pipeline`` composes stages; the fused ``run`` is just
``Pipeline([Sample, (Layout), Ocr, Normalize])``.

Lives in ``pipeline/`` rather than ``interfaces/`` because a Stage is defined in
terms of the ``ExecutionContext`` (an L3 type); the layering stays acyclic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ocrscout.pipeline.context import ExecutionContext

Artifact = Literal["source", "pages", "layout", "raw", "train"]


class StageIO(BaseModel):
    reads: Artifact
    writes: Artifact


class ModelTally(BaseModel):
    written: int = 0
    skipped: int = 0
    failed: int = 0


class StageResult(BaseModel):
    stage: str
    rows_written: int = 0
    rows_skipped: int = 0
    rows_failed: int = 0
    seconds: float = 0.0
    by_model: dict[str, ModelTally] = Field(default_factory=dict)


class Stage(ABC):
    name: ClassVar[str]
    io: ClassVar[StageIO]
    requires_runner: ClassVar[bool] = False

    @abstractmethod
    def execute(self, ctx: ExecutionContext) -> StageResult: ...

"""Pydantic data models shared across ocrscout.

ABCs live in ``ocrscout.interfaces``; this module holds the typed payloads that
flow between them.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ocrscout.profile import ModelProfile

if TYPE_CHECKING:
    pass


class PageImage(BaseModel):
    """A single page image flowing through the pipeline.

    ``image`` is the live PIL.Image and is intentionally not serialized — it is
    runtime state. Use ``source_uri`` to recover the source if needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    image: Any  # PIL.Image.Image at runtime
    width: int
    height: int
    dpi: int | None = None
    source_uri: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Reference(BaseModel):
    """Ground-truth content for a single page.

    Either ``text`` or ``document`` (or both) may be set, depending on what the
    reference adapter produces. ``document`` is a ``DoclingDocument`` at runtime
    but typed as ``Any`` here to keep this module import-light.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    text: str | None = None
    document: Any | None = None


class RawOutput(BaseModel):
    """Raw output from a model backend, before normalization."""

    page_id: str
    output_format: Literal["markdown", "doctags", "layout_json"]
    payload: str
    tokens: int | None = None
    error: str | None = None


class BackendInvocation(BaseModel):
    """A resolved command-or-call, ready for the backend to execute."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["subprocess", "in_process", "http"]
    argv: list[str] | None = None
    callable_path: str | None = None
    endpoint: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    profile: ModelProfile
    pages: list[str] = Field(default_factory=list)


class ExportRecord(BaseModel):
    """One row written by an ExportAdapter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page: PageImage
    document: Any  # DoclingDocument at runtime
    raw: RawOutput
    metrics: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """Final metrics envelope for a complete run."""

    pipeline_id: str
    started_at: datetime
    finished_at: datetime | None = None
    pages_total: int = 0
    pages_ok: int = 0
    pages_failed: int = 0
    output_tokens: int = 0
    gpu_memory_peak_mb: float | None = None
    stage_seconds: dict[str, float] = Field(default_factory=dict)

    @property
    def pages_per_hour(self) -> float | None:
        if self.finished_at is None:
            return None
        elapsed = (self.finished_at - self.started_at).total_seconds()
        if elapsed <= 0:
            return None
        return self.pages_ok * 3600.0 / elapsed


class AdapterRef(BaseModel):
    """Reference to a registered adapter, with optional kwargs."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Top-level shape of a ``pipeline.yaml`` (and the dump produced by
    ``ocrscout run`` for reproducibility)."""

    name: str
    source: AdapterRef
    reference: AdapterRef | None = None
    models: list[str]
    normalizer_overrides: dict[str, str] = Field(default_factory=dict)
    export: AdapterRef
    evaluator: AdapterRef | None = None
    reporter: AdapterRef | None = None
    sample: int | None = None
    output_dir: Path

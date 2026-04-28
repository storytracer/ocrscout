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

    ``volume_id`` joins to ``Volume.volume_id`` for sources that group pages
    into bibliographic units (BHL items, IA items, HathiTrust volumes, IIIF
    manifests). It stays ``None`` for flat sources like ``hf_dataset``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    image: Any  # PIL.Image.Image at runtime
    width: int
    height: int
    dpi: int | None = None
    source_uri: str | None = None
    volume_id: str | None = None
    sequence: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Volume(BaseModel):
    """A logical bibliographic unit grouping pages.

    Maps to: BHL Item, IA item, HathiTrust volume, IIIF Manifest, PDF document.
    Sources that have a volume concept yield these from
    ``SourceAdapter.iter_volumes()``; the run loop materializes them into a
    ``volumes-NNNNN.parquet`` sidecar next to the per-page results parquet.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    volume_id: str
    title: str | None = None
    creators: list[str] = Field(default_factory=list)
    language: str | None = None
    year: int | None = None
    rights: str | None = None
    page_count: int | None = None
    source_uri: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class LayoutRegion(BaseModel):
    """A typed region detected on a page by a ``LayoutDetector``.

    The detector emits regions in page-pixel coordinates (top-left origin).
    Cropping happens downstream in the layout-aware backend, which preserves
    these page-coordinate bboxes through into the layout-JSON payload.
    """

    id: int
    category: str
    bbox: tuple[float, float, float, float]
    score: float | None = None
    polygon: list[tuple[float, float]] | None = None
    reading_order: int | None = None
    parent_id: int | None = None


class ReferenceProvenance(BaseModel):
    """Where a ``Reference`` came from. Load-bearing: most references in the
    wild (BHL legacy OCR, IA djvu, ABBYY exports) are themselves OCR output,
    not ground truth. Consumers need ``method`` to know whether to interpret
    a comparison result as accuracy-vs-truth or agreement-vs-incumbent.
    """

    method: Literal["human", "ocr", "llm", "mixed", "unknown"] = "unknown"
    engine: str | None = None
    confidence: float | None = None


class Reference(BaseModel):
    """Pre-existing OCR or transcription content for a single page.

    NOT necessarily an oracle. Use ``provenance`` to capture what kind of
    artifact this is. Either ``text`` or ``document`` (or both) may be set,
    depending on what the reference adapter produces. ``document`` is a
    ``DoclingDocument`` at runtime but typed as ``Any`` here to keep this
    module import-light.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    text: str | None = None
    document: Any | None = None
    provenance: ReferenceProvenance = Field(default_factory=ReferenceProvenance)


class RawOutput(BaseModel):
    """Raw output from a model backend, before normalization."""

    page_id: str
    output_format: Literal["markdown", "doctags", "layout_json", "docling_document"]
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
    # Runtime-only state shared between prepare() and run() — temp HF repo IDs,
    # page-id-to-row-index maps, PIL image lists for in-process backends, etc.
    # Not serialized to pipeline.yaml.
    extra: dict[str, Any] = Field(default_factory=dict)


class ExportRecord(BaseModel):
    """One row written by an ExportAdapter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page: PageImage
    model: str
    document: Any  # DoclingDocument at runtime
    raw: RawOutput
    reference: Reference | None = None
    markdown: str | None = None
    text: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Per-comparison-name results from any Comparison objects that ran on
    # this (page, model). Typed as ``Any`` here so this module stays free of
    # ABC imports; the run loop puts ``ComparisonResult`` subclass instances
    # in, and the parquet writer round-trips them as JSON.
    comparisons: dict[str, Any] = Field(default_factory=dict)


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
    # Comparison names to run per (page, model). ``None`` lets the run loop
    # pick a default based on what data is available; ``[]`` (or the literal
    # ``["none"]``) explicitly disables all comparisons.
    comparisons: list[str] | None = None
    models: list[str]
    normalizer_overrides: dict[str, str] = Field(default_factory=dict)
    export: AdapterRef
    reporter: AdapterRef | None = None
    sample: int | None = None
    output_dir: Path

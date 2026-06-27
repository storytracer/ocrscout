"""Pydantic data models shared across ocrscout.

ABCs live in ``ocrscout.interfaces``; this module holds the typed payloads that
flow between them.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from ocrscout.profile import ModelProfile

if TYPE_CHECKING:
    pass


class PageImage(BaseModel):
    """A single page image flowing through the pipeline.

    Image lifecycle: sources install ``image_loader`` (a zero-arg callable
    that returns a fresh PIL.Image) instead of decoding eagerly; consumers
    enter ``with page.open_image() as img:`` to load + close in a bounded
    scope. ``image`` exists as an optional eager slot (defaults to ``None``)
    that backends can pin for the lifetime of multi-step operations — e.g.
    layout-aware OCR loads once, then fans out concurrent region crops
    against the same decoded buffer before clearing it. Both fields are
    runtime-only and excluded from serialization.

    ``page_id`` is the source-side raw identifier (BHL's PageID, an HF row id,
    a filename stem) used as the join key by the run loop, backends, and
    reference adapters. ``file_id`` is the human-facing canonical identifier
    surfaced by viewer/inspect/publish: ``{volume_id}/{filename.ext}`` for
    volume sources, ``{parent_dir}/{filename.ext}`` for flat folder/URL/Hub
    sources. file_id is globally unique within a run.

    ``barcode`` joins to ``Volume.barcode`` for sources that group pages
    into bibliographic units (BHL items, IA items, HathiTrust volumes, IIIF
    manifests). It stays ``None`` for flat sources like ``hf_dataset``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    file_id: str
    image: Any = Field(default=None, exclude=True, repr=False)
    image_loader: Any = Field(default=None, exclude=True, repr=False)
    width: int = 0
    height: int = 0
    dpi: int | None = None
    source_uri: str | None = None
    barcode: str | None = None
    sequence: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @contextmanager
    def open_image(self) -> Iterator[Any]:
        """Yield a PIL Image scoped to this with-block.

        If ``image`` is already set (a backend pinned it for cross-step
        sharing), yield it untouched — whoever pinned it owns the close.
        Otherwise call ``image_loader()``, yield the fresh image, and
        close it on exit so the decoded RGB buffer is freed immediately.

        Backfills ``width`` / ``height`` on first load so downstream
        consumers (prompt substitution, normalizer page-size hints) see
        real dimensions even when the source didn't decode upfront.

        Raises ``RuntimeError`` if neither ``image`` nor ``image_loader``
        is set — that's a contract violation by the source adapter.
        Loader exceptions propagate; backends are expected to catch and
        record them as per-page failures.
        """
        eager = self.image
        if eager is not None:
            self._backfill_dims(eager)
            yield eager
            return
        loader = self.image_loader
        if loader is None:
            raise RuntimeError(
                f"PageImage {self.page_id!r} has neither image nor image_loader"
            )
        img = loader()
        try:
            self._backfill_dims(img)
            yield img
        finally:
            try:
                img.close()
            except Exception:  # noqa: BLE001
                pass

    def _backfill_dims(self, img: Any) -> None:
        if self.width > 0 and self.height > 0:
            return
        size = getattr(img, "size", None)
        if not size:
            return
        try:
            w, h = size
            self.width = int(w)
            self.height = int(h)
        except (TypeError, ValueError):
            pass


class Volume(BaseModel):
    """A logical bibliographic unit grouping pages.

    Maps to: BHL Item, IA item, HathiTrust volume, IIIF Manifest, PDF document.
    Sources that have a volume concept yield these from
    ``SourceAdapter.iter_volumes()``; the run loop materializes them into a
    ``volumes-NNNNN.parquet`` sidecar next to the per-page results parquet.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    barcode: str
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

    # --- cost / GPU context, stamped onto every row at write time ---------
    # Populated by ``ocrscout.cost.recorder`` (from LiteLLM's
    # success_callback) and the active ``GpuConfig`` (env vars on remote
    # workers, ``~/.ocrscout/config.yaml`` locally). All optional so
    # backends that don't touch LiteLLM (e.g. Tesseract) and runs that
    # don't have a configured GPU still produce a valid row.
    gpu_type: str | None = None
    provider: str | None = None
    cost_per_hour: float | None = None
    elapsed_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    litellm_cost: float | None = None
    gpu_time_cost: float | None = None

    # Autoscaler context. Populated for runtime: vllm rows from the active
    # profile at the time the page ran; null for hosted / cpu rows.
    # region_concurrency is only meaningful for backend: layout_chat.
    kv_cache_memory_bytes: int | None = None
    concurrent_requests: int | None = None
    region_concurrency: int | None = None


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


class RunnerEndpoint(BaseModel):
    """One model endpoint exposed by a launched Runner.

    For LocalRunner this is the vLLM ``/v1`` URL; for remote runners this
    is whatever the worker advertised. Always reachable via the proxy URL
    on ``RunnerHandle`` — these are diagnostic detail, not what callers
    actually post to.
    """

    model: str
    url: str
    pid: int | None = None


class RunnerHandle(BaseModel):
    """Returned by ``Runner.launch()`` when the stack is healthy.

    The ``proxy_url`` is the single OpenAI-compatible endpoint every
    backend posts to. ``endpoints`` is diagnostic — the per-model vLLM
    URLs behind the proxy.
    """

    runner: str
    proxy_url: str
    endpoints: list[RunnerEndpoint] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class JobHandle(BaseModel):
    """Returned by ``Runner.submit()``. Fire-and-forget identifier."""

    job_id: str
    runner: str
    output_dir: str


RunnerState = Literal[
    "down", "launching", "ready", "busy", "tearing_down", "error",
]


class RunnerStatus(BaseModel):
    """Returned by ``Runner.status()``. Snapshot of the runner's current state.

    Cheap to compute (reads state file, checks PIDs / queries remote API).
    Numbers come from the output Parquet glob: ``pages_done`` counts rows,
    ``cumulative_cost`` sums the ``litellm_cost`` + ``gpu_time_cost``
    columns.
    """

    runner: str
    state: RunnerState
    models: list[str] = Field(default_factory=list)
    proxy_url: str | None = None
    uptime_seconds: float | None = None
    pages_done: int = 0
    pages_failed: int = 0
    cumulative_cost: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


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
    # When set, ``backend: layout_chat`` profiles use this detector (typically
    # the ``precomputed`` detector reading a layout-*.parquet) instead of the
    # one named in their profile YAML. Whole-page profiles ignore it.
    layout: AdapterRef | None = None
    export: AdapterRef
    reporter: AdapterRef | None = None
    sample: int | None = None
    output_dir: Path

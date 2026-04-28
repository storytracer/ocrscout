"""Comparison subsystem: typed agreement metrics between two OCR artifacts.

Replaces the older ``Evaluator`` ABC. The framing change matters: most
references in the wild (BHL, IA, ABBYY exports) are themselves OCR output,
not human-transcribed ground truth, so calling these objects "evaluators"
and their results "scores" baked an oracle assumption that's wrong for the
common case. The right framing is **comparison/agreement**, not accuracy.

Each ``Comparison`` is one analytic axis (text content, document structure,
bbox layout, eventually semantic / VLM-judge). It takes a ``PredictionView``
and a ``BaselineView`` (both nullable on document/layout fields) and returns
a typed ``ComparisonResult`` subclass. Results carry both a flat ``summary``
dict (for run-summary tables and parquet flat columns) and rich type-specific
payload fields (opcodes, bbox matches, item counts) consumed by per-type
``ComparisonRenderer`` implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ocrscout.types import ReferenceProvenance

if TYPE_CHECKING:
    from rich.console import Console


class PredictionView(BaseModel):
    """A-side of a comparison — typically a model's output for a page."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    label: str
    text: str | None = None
    document: Any | None = None  # DoclingDocument at runtime


class BaselineView(BaseModel):
    """B-side of a comparison — a reference, another model output, or any
    other artifact you want to compare against. ``provenance`` is what
    distinguishes "compared against incumbent OCR" from "compared against
    a human transcription"; consumers reading downstream parquets should
    look at it before interpreting any agreement metric as accuracy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_id: str
    label: str
    text: str | None = None
    document: Any | None = None
    provenance: ReferenceProvenance | None = None


class ComparisonResult(BaseModel):
    """Typed result of a single ``Comparison``.

    ``comparison`` is the discriminator (matches the ``Comparison.name`` of
    the producer). ``summary`` is a flat ``str -> float`` projection used by
    cross-cutting consumers (run-summary aggregation, parquet flat columns)
    that don't want to dispatch on the subclass type. Subclasses add their
    own typed payload fields (opcodes, bbox lists, item counts).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    comparison: str
    summary: dict[str, float] = Field(default_factory=dict)


class Comparison(ABC):
    """One analytic dimension of agreement between a prediction and a baseline.

    ``requires`` declares which input modalities must be present on BOTH
    sides for the comparison to fire. The run loop and the viewer's
    Compare-mode UI consult this set so layout-only comparisons aren't
    offered when a side is text-only.
    """

    name: ClassVar[str]
    requires: ClassVar[frozenset[str]]

    @abstractmethod
    def compare(
        self,
        prediction: PredictionView,
        baseline: BaselineView,
    ) -> ComparisonResult | None:
        """Run the comparison. Return ``None`` if either side lacks the
        required modality (e.g. the baseline has no document for a layout
        comparison) so the run loop can quietly skip rather than error."""


class ComparisonRenderer(ABC):
    """Renders a ``ComparisonResult`` for a particular surface.

    One renderer instance handles all three surfaces (terminal, HTML,
    Gradio); ``name`` matches the result's ``comparison`` discriminator,
    so the registry can dispatch from a result back to its renderer.
    """

    name: ClassVar[str]

    @abstractmethod
    def render_html(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
    ) -> str:
        """Return a fully-self-contained HTML page (used by ``inspect --html``)."""

    @abstractmethod
    def render_terminal(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
        console: Console,
    ) -> None:
        """Print to a Rich ``Console`` (used by ``inspect`` without ``--html``)."""

    @abstractmethod
    def render_gradio(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
    ) -> str:
        """Return an HTML fragment for embedding in ``gr.HTML``."""


def aggregate_summaries(
    results: Iterable[ComparisonResult],
) -> dict[str, tuple[float, int]]:
    """Aggregate ``summary`` keys across many results.

    Returns ``{key: (mean, n)}`` where ``n`` counts how many results
    contributed (i.e. had that key non-null). Sample-mean averaging — the
    convention users expect from a per-model summary row. Used by the run
    summary table and any other roll-up consumer.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        for k, v in (r.summary or {}).items():
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k], counts[k]) for k in sums}

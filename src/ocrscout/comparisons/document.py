"""DocumentComparison: structural agreement between two ``DoclingDocument``s.

Counts items by ``label`` on each side and reports the deltas. Useful when
you want to know whether two models produced the *same shape* of output
(e.g. one detected the table the other missed) independently of how well
their text matches.

Today's deltas are pure counts — heading_count_delta, table_count_delta,
picture_count_delta. They surface in ``summary`` so they roll up into the
run summary table and the parquet's flat columns. The fuller per-label
breakdown lives on the typed result for the renderer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, ClassVar, Literal

from ocrscout.interfaces.comparison import (
    BaselineView,
    Comparison,
    ComparisonResult,
    PredictionView,
)

# Label-name groupings used to roll item counts up into a few summary metrics.
# DoclingDocument item labels (from docling-core) include "title",
# "section_header", "subtitle_level_1", "paragraph", "list_item", "caption",
# "footnote", "page_header", "page_footer", and so on. Headings are the
# load-bearing structural signal we care about for cross-model agreement;
# paragraphs are too noisy and pictures/tables already have their own count.
_HEADING_LABELS: frozenset[str] = frozenset({
    "title",
    "section_header",
    "subtitle_level_1",
    "subtitle_level_2",
    "subtitle_level_3",
    "subtitle_level_4",
    "subtitle_level_5",
    "page_header",
})


class DocumentComparisonResult(ComparisonResult):
    comparison: Literal["document"] = "document"
    item_counts_pred: dict[str, int]
    item_counts_base: dict[str, int]
    heading_count_delta: int
    table_count_delta: int
    picture_count_delta: int


class DocumentComparison(Comparison):
    name = "document"
    requires: ClassVar[frozenset[str]] = frozenset({"document"})

    def compare(
        self, prediction: PredictionView, baseline: BaselineView
    ) -> DocumentComparisonResult | None:
        if prediction.document is None or baseline.document is None:
            return None
        pc = _count_items(prediction.document)
        bc = _count_items(baseline.document)
        heading_delta = _heading_count(pc) - _heading_count(bc)
        table_delta = pc.get("__tables__", 0) - bc.get("__tables__", 0)
        picture_delta = pc.get("__pictures__", 0) - bc.get("__pictures__", 0)
        return DocumentComparisonResult(
            item_counts_pred=pc,
            item_counts_base=bc,
            heading_count_delta=heading_delta,
            table_count_delta=table_delta,
            picture_count_delta=picture_delta,
            summary={
                "heading_count_delta": float(heading_delta),
                "table_count_delta": float(table_delta),
                "picture_count_delta": float(picture_delta),
            },
        )


def _count_items(doc: Any) -> dict[str, int]:
    """Tally items in a ``DoclingDocument`` by label (text items) plus
    fixed pseudo-keys for tables (``__tables__``) and pictures
    (``__pictures__``). Defensive against attribute absences so a
    docling-core version skew degrades to a partial count rather than
    crashing the comparison.
    """
    counts: Counter[str] = Counter()
    try:
        for t in (doc.texts or []):
            label = getattr(t, "label", None)
            if label is not None:
                counts[str(label)] += 1
    except Exception:  # noqa: BLE001
        pass
    try:
        counts["__tables__"] = len(doc.tables or [])
    except Exception:  # noqa: BLE001
        counts["__tables__"] = 0
    try:
        counts["__pictures__"] = len(doc.pictures or [])
    except Exception:  # noqa: BLE001
        counts["__pictures__"] = 0
    return dict(counts)


def _heading_count(counts: dict[str, int]) -> int:
    return sum(n for label, n in counts.items() if label in _HEADING_LABELS)

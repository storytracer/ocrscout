"""LayoutComparison: bbox-level agreement between two layout-aware OCR outputs.

Walks every text/picture/table item on both DoclingDocuments, pulls each
item's first ``ProvenanceItem.bbox`` (matching the existing viewer store
extraction in ``src/ocrscout/viewer/store.py``), then greedy-matches
regions across sides by category at IoU >= threshold.

Today this comparison fires almost exclusively model-vs-model — most
references in the wild are text-only and carry no layout. The architecture
supports references with layout (a future ALTO adapter, for instance);
``compare`` returns ``None`` if either side has no extractable regions.

Greedy IoU matching is the established quick baseline; Hungarian matching
is more correct but pure-python rectangle IoU + greedy is fast and
adequate for scouting comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Literal

from ocrscout.interfaces.comparison import (
    BaselineView,
    Comparison,
    ComparisonResult,
    PredictionView,
)

# Greedy match threshold. Below this IoU a region is unmatched.
_DEFAULT_IOU_THRESHOLD = 0.5

# (left, top, right, bottom) in the document's native coord space.
Bbox = tuple[float, float, float, float]


class LayoutRegion:
    """One bbox + its category, for matching."""

    __slots__ = ("category", "bbox")

    def __init__(self, category: str, bbox: Bbox) -> None:
        self.category = category
        self.bbox = bbox


class LayoutComparisonResult(ComparisonResult):
    comparison: Literal["layout"] = "layout"
    iou_mean: float
    iou_per_category: dict[str, float]
    matched_regions: int
    unmatched_pred: int
    unmatched_base: int


class LayoutComparison(Comparison):
    name = "layout"
    requires: ClassVar[frozenset[str]] = frozenset({"layout"})

    def __init__(self, *, iou_threshold: float = _DEFAULT_IOU_THRESHOLD) -> None:
        self.iou_threshold = float(iou_threshold)

    def compare(
        self, prediction: PredictionView, baseline: BaselineView
    ) -> LayoutComparisonResult | None:
        regs_a = _extract_regions(prediction.document)
        regs_b = _extract_regions(baseline.document)
        if not regs_a or not regs_b:
            return None
        matched_ious, unmatched_a, unmatched_b = _greedy_match(
            regs_a, regs_b, threshold=self.iou_threshold
        )
        if matched_ious:
            iou_mean = sum(iou for _, _, iou in matched_ious) / len(matched_ious)
        else:
            iou_mean = 0.0
        per_cat: dict[str, list[float]] = defaultdict(list)
        for region_a, _region_b, iou in matched_ious:
            per_cat[region_a.category].append(iou)
        per_cat_mean = {
            cat: (sum(ious) / len(ious) if ious else 0.0)
            for cat, ious in per_cat.items()
        }
        return LayoutComparisonResult(
            iou_mean=iou_mean,
            iou_per_category=per_cat_mean,
            matched_regions=len(matched_ious),
            unmatched_pred=len(unmatched_a),
            unmatched_base=len(unmatched_b),
            summary={"iou_mean": iou_mean},
        )


def _extract_regions(doc: Any) -> list[LayoutRegion]:
    """Collect (category, bbox) for every text/picture/table item that
    carries a ``ProvenanceItem``. Picture and table items are tagged with
    fixed pseudo-categories (``"picture"`` / ``"table"``) since they don't
    have a docling ``label`` field. Defensive against missing attributes
    so docling-core version skew degrades gracefully.
    """
    if doc is None:
        return []
    out: list[LayoutRegion] = []
    for items, default_label in (
        (getattr(doc, "texts", None) or [], None),
        (getattr(doc, "pictures", None) or [], "picture"),
        (getattr(doc, "tables", None) or [], "table"),
    ):
        for item in items:
            bbox = _first_bbox(item)
            if bbox is None:
                continue
            label = default_label or str(getattr(item, "label", "") or "unknown")
            out.append(LayoutRegion(category=label, bbox=bbox))
    return out


def _first_bbox(item: Any) -> Bbox | None:
    provs = getattr(item, "prov", None) or []
    if not provs:
        return None
    bbox = getattr(provs[0], "bbox", None)
    if bbox is None:
        return None
    try:
        return (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
    except (AttributeError, TypeError, ValueError):
        return None


def _iou(a: Bbox, b: Bbox) -> float:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    inter_l = max(al, bl)
    inter_t = max(at, bt)
    inter_r = min(ar, br)
    inter_b = min(ab, bb)
    iw = max(0.0, inter_r - inter_l)
    ih = max(0.0, inter_b - inter_t)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ar - al) * max(0.0, ab - at)
    area_b = max(0.0, br - bl) * max(0.0, bb - bt)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _greedy_match(
    pred: list[LayoutRegion],
    base: list[LayoutRegion],
    *,
    threshold: float,
) -> tuple[list[tuple[LayoutRegion, LayoutRegion, float]], list[LayoutRegion], list[LayoutRegion]]:
    """Within each category, repeatedly take the highest-IoU pair until no
    more pairs exceed ``threshold``. Returns the matched pairs (with their
    IoU) plus the unmatched-on-each-side lists.
    """
    matched: list[tuple[LayoutRegion, LayoutRegion, float]] = []
    by_cat_pred: dict[str, list[LayoutRegion]] = defaultdict(list)
    by_cat_base: dict[str, list[LayoutRegion]] = defaultdict(list)
    for r in pred:
        by_cat_pred[r.category].append(r)
    for r in base:
        by_cat_base[r.category].append(r)
    unmatched_pred: list[LayoutRegion] = []
    unmatched_base: list[LayoutRegion] = []
    categories = set(by_cat_pred) | set(by_cat_base)
    for cat in categories:
        ps = list(by_cat_pred.get(cat, ()))
        bs = list(by_cat_base.get(cat, ()))
        # Score every pair, then greedily take the largest available.
        pairs: list[tuple[float, int, int]] = [
            (_iou(p.bbox, b.bbox), pi, bi)
            for pi, p in enumerate(ps)
            for bi, b in enumerate(bs)
        ]
        pairs.sort(reverse=True)
        used_p: set[int] = set()
        used_b: set[int] = set()
        for iou, pi, bi in pairs:
            if iou < threshold:
                break
            if pi in used_p or bi in used_b:
                continue
            matched.append((ps[pi], bs[bi], iou))
            used_p.add(pi)
            used_b.add(bi)
        unmatched_pred.extend(p for i, p in enumerate(ps) if i not in used_p)
        unmatched_base.extend(b for i, b in enumerate(bs) if i not in used_b)
    return matched, unmatched_pred, unmatched_base

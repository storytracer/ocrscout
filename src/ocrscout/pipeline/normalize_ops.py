"""Pure normalize + compare: ``(page, raw) → ResultRow``.

The one place OCR output becomes a final results row. Shared by the
``normalize`` stage and (via the OcrStage→NormalizeStage composition) the fused
``run``. No IO, no runner, no globals — given the inputs it returns a row or
``None`` (normalizer failure). Cost columns are carried in verbatim, never
recomputed.
"""

from __future__ import annotations

import logging
from typing import Any

from ocrscout.errors import NormalizerError
from ocrscout.interfaces.comparison import (
    BaselineView,
    Comparison,
    PredictionView,
)
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.io import CostColumns, ResultRow
from ocrscout.profile import ModelProfile
from ocrscout.types import ExportRecord, PageImage, RawOutput

log = logging.getLogger(__name__)


def doc_stats(doc: Any) -> tuple[int, int, str, str]:
    """Return ``(item_count, text_length, markdown, text)`` for a
    DoclingDocument, degrading to zero/empty on any missing attribute."""
    try:
        text_length = sum(len(t.text or "") for t in (doc.texts or []))
    except Exception:  # noqa: BLE001
        text_length = 0
    try:
        item_count = len(doc.texts or []) + len(doc.pictures or []) + len(doc.tables or [])
    except Exception:  # noqa: BLE001
        item_count = 0
    try:
        markdown = doc.export_to_markdown()
    except Exception as e:  # noqa: BLE001
        log.warning("export_to_markdown failed: %s", e)
        markdown = ""
    try:
        text = doc.export_to_text()
    except Exception as e:  # noqa: BLE001
        log.warning("export_to_text failed: %s", e)
        text = ""
    return item_count, text_length, markdown, text


def normalize_one(
    *,
    page: PageImage,
    raw: RawOutput,
    model_name: str,
    profile: ModelProfile,
    normalizer: Normalizer,
    cost: CostColumns,
    reference_adapter: ReferenceAdapter | None = None,
    comparisons: list[Comparison] | None = None,
    base_metrics: dict[str, Any] | None = None,
) -> ResultRow | None:
    """Normalize one ``(page, raw)`` and compare against the reference.

    Returns ``None`` when the normalizer raises (caller counts a failure).
    Callers must drop ``raw.error`` rows before calling — error pages produce
    no results row.
    """
    try:
        doc = normalizer.normalize(raw, page, profile)
    except NormalizerError as e:
        log.warning("normalizer failed for %s/%s: %s", model_name, page.file_id, e)
        return None

    item_count, text_length, markdown, text = doc_stats(doc)

    reference = None
    if reference_adapter is not None:
        try:
            reference = reference_adapter.get(page)
        except Exception as e:  # noqa: BLE001
            log.warning("reference adapter failed for %s/%s: %s", model_name, page.file_id, e)

    page_comparisons: dict[str, Any] = {}
    if reference is not None and comparisons:
        pred = PredictionView(page_id=page.page_id, label=model_name, text=text, document=doc)
        base = BaselineView(
            page_id=page.page_id,
            label=reference_adapter.name if reference_adapter else "reference",
            text=reference.text, document=reference.document, provenance=reference.provenance,
        )
        for cmp in comparisons:
            try:
                result = cmp.compare(pred, base)
            except Exception as e:  # noqa: BLE001
                log.warning("comparison %s failed for %s/%s: %s", cmp.name, model_name, page.file_id, e)
                continue
            if result is not None:
                page_comparisons[cmp.name] = result

    metrics = dict(base_metrics or {})
    metrics.update(tokens=raw.tokens, item_count=item_count, text_length=text_length)

    record = ExportRecord(
        page=page, model=model_name, document=doc, raw=raw, reference=reference,
        markdown=markdown, text=text, metrics=metrics, comparisons=page_comparisons,
        **cost.model_dump(),
    )
    return ResultRow.from_export_record(record)

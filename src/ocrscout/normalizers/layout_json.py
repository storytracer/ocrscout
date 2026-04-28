"""Layout-JSON normalizer for dots.mocr-style ``layout-all`` output.

Input: a JSON list of blocks, each with ``category`` (string), ``bbox``
(``[l, t, r, b]`` in pixel coords, top-left origin), and optional ``text``.

Output: a ``DoclingDocument`` with a page and one item per block, each carrying
a ``ProvenanceItem`` so downstream consumers can render bounding boxes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from ocrscout.errors import NormalizerError
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.normalizers._tables import parse_table_payload
from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput

log = logging.getLogger(__name__)


# Profile category_mapping values (lowercased docling-style strings) → handler.
# Anything not listed here, or not in the profile mapping, falls through to
# add_text(DocItemLabel.TEXT).
_HEADING_KINDS = {"heading", "section_header"}
_TEXT_LABELS: dict[str, DocItemLabel] = {
    "text": DocItemLabel.TEXT,
    "paragraph": DocItemLabel.PARAGRAPH,
    "list_item": DocItemLabel.LIST_ITEM,
    "caption": DocItemLabel.CAPTION,
    "footnote": DocItemLabel.FOOTNOTE,
    "page_header": DocItemLabel.PAGE_HEADER,
    "page_footer": DocItemLabel.PAGE_FOOTER,
    "formula": DocItemLabel.FORMULA,
    "code": DocItemLabel.CODE,
    "furniture": DocItemLabel.PAGE_HEADER,  # generic furniture → page_header
    "reference": DocItemLabel.REFERENCE,
}


class LayoutJsonNormalizer(Normalizer):
    name = "layout_json"
    output_format = "layout_json"

    def normalize(
        self, raw: RawOutput, page: PageImage, profile: ModelProfile
    ) -> DoclingDocument:
        if raw.output_format != "layout_json":
            raise NormalizerError(
                f"LayoutJsonNormalizer expects output_format='layout_json', "
                f"got {raw.output_format!r}"
            )
        try:
            blocks = json.loads(raw.payload)
        except json.JSONDecodeError as e:
            raise NormalizerError(
                f"layout JSON for {page.page_id!r} is not valid JSON: {e}"
            ) from e
        if not isinstance(blocks, list):
            raise NormalizerError(
                f"layout JSON for {page.page_id!r} must be a list, got {type(blocks).__name__}"
            )

        doc = DoclingDocument(name=page.page_id)
        page_no = 1
        doc.add_page(page_no=page_no, size=Size(width=page.width, height=page.height))

        for block in blocks:
            try:
                self._add_block(doc, block, page_no=page_no, profile=profile)
            except Exception as e:  # noqa: BLE001
                # Per-block resilience: log and continue. The doc is still
                # returned with all successful items.
                log.warning(
                    "skipping malformed block in %s: %s (block=%r)", page.page_id, e, block
                )
        return doc

    def _add_block(
        self,
        doc: DoclingDocument,
        block: dict[str, Any],
        *,
        page_no: int,
        profile: ModelProfile,
    ) -> None:
        if not isinstance(block, dict):
            raise NormalizerError(f"block must be a mapping, got {type(block).__name__}")

        category_raw = block.get("category", "Text")
        kind = profile.category_mapping.get(category_raw, str(category_raw).lower())
        text = (block.get("text") or "").strip()
        bbox_raw = block.get("bbox")
        prov = _build_prov(bbox_raw, page_no=page_no, charspan=(0, len(text)))

        if kind == "title":
            doc.add_title(text=text or "", prov=prov)
            return
        if kind in _HEADING_KINDS:
            level = int(block.get("level", 1))
            level = max(1, min(level, 6))
            doc.add_heading(text=text or "", level=level, prov=prov)
            return
        if kind == "picture":
            picture_prov = _build_prov(bbox_raw, page_no=page_no, charspan=(0, 0))
            doc.add_picture(prov=picture_prov)
            return
        if kind == "table":
            data = parse_table_payload(text)
            table_prov = _build_prov(bbox_raw, page_no=page_no, charspan=(0, 0))
            doc.add_table(data=data, prov=table_prov)
            return

        label = _TEXT_LABELS.get(kind, DocItemLabel.TEXT)
        doc.add_text(label=label, text=text, prov=prov)


def _build_prov(
    bbox_raw: Any, *, page_no: int, charspan: tuple[int, int]
) -> ProvenanceItem | None:
    if bbox_raw is None:
        return None
    if not (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4):
        raise NormalizerError(
            f"bbox must be a 4-tuple [l,t,r,b], got {bbox_raw!r}"
        )
    bbox = BoundingBox.from_tuple(tuple(float(v) for v in bbox_raw), origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=page_no, bbox=bbox, charspan=charspan)

"""Layout-JSON normalizer for dots.mocr-style ``layout-all`` output.

Input: a JSON list of blocks, each with ``category`` (string), ``bbox``
(``[l, t, r, b]`` in pixel coords, top-left origin), and optional ``text``.

Output: a ``DoclingDocument`` with a page and one item per block, each carrying
a ``ProvenanceItem`` so downstream consumers can render bounding boxes.
"""

from __future__ import annotations

import json
import logging
import math
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

        # Whole-page VLMs (dots.mocr / dots.ocr) emit bboxes in the resized
        # coordinate space their vision processor produced, not original
        # pixels. Map them back so the viewer overlay aligns with the
        # full-resolution source image.
        scale = _smart_resize_scale(page, profile)

        for block in blocks:
            try:
                self._add_block(
                    doc, block, page_no=page_no, profile=profile, scale=scale
                )
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
        scale: tuple[float, float] | None,
    ) -> None:
        if not isinstance(block, dict):
            raise NormalizerError(f"block must be a mapping, got {type(block).__name__}")

        category_raw = block.get("category", "Text")
        kind = profile.category_mapping.get(category_raw, str(category_raw).lower())
        text = (block.get("text") or "").strip()
        bbox_raw = block.get("bbox")
        prov = _build_prov(
            bbox_raw, page_no=page_no, charspan=(0, len(text)), scale=scale
        )

        if kind == "title":
            doc.add_title(text=text or "", prov=prov)
            return
        if kind in _HEADING_KINDS:
            level = int(block.get("level", 1))
            level = max(1, min(level, 6))
            doc.add_heading(text=text or "", level=level, prov=prov)
            return
        if kind == "picture":
            picture_prov = _build_prov(
                bbox_raw, page_no=page_no, charspan=(0, 0), scale=scale
            )
            doc.add_picture(prov=picture_prov)
            return
        if kind == "table":
            data = parse_table_payload(text)
            table_prov = _build_prov(
                bbox_raw, page_no=page_no, charspan=(0, 0), scale=scale
            )
            doc.add_table(data=data, prov=table_prov)
            return

        label = _TEXT_LABELS.get(kind, DocItemLabel.TEXT)
        doc.add_text(label=label, text=text, prov=prov)


def _build_prov(
    bbox_raw: Any,
    *,
    page_no: int,
    charspan: tuple[int, int],
    scale: tuple[float, float] | None = None,
) -> ProvenanceItem | None:
    if bbox_raw is None:
        return None
    if not (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4):
        raise NormalizerError(
            f"bbox must be a 4-tuple [l,t,r,b], got {bbox_raw!r}"
        )
    left, top, right, bottom = (float(v) for v in bbox_raw)
    if scale is not None:
        sx, sy = scale
        left, top, right, bottom = left * sx, top * sy, right * sx, bottom * sy
    bbox = BoundingBox.from_tuple(
        (left, top, right, bottom), origin=CoordOrigin.TOPLEFT
    )
    return ProvenanceItem(page_no=page_no, bbox=bbox, charspan=charspan)


def _smart_resize_scale(
    page: PageImage, profile: ModelProfile
) -> tuple[float, float] | None:
    """Return ``(scale_x, scale_y)`` mapping resized-space bboxes back to
    original pixels, or ``None`` when no rescale applies.

    The model perceived the page at ``smart_resize(height, width)``; its
    bboxes live in that space. Multiplying by ``original / resized`` lifts
    them back to the full-resolution image the viewer displays.
    """
    if profile.bbox_coordinate_space != "smart_resize":
        return None
    if page.width <= 0 or page.height <= 0:
        return None
    args = profile.effective_smart_resize_args()
    resized_h, resized_w = _smart_resize(
        page.height,
        page.width,
        factor=args["factor"],
        min_pixels=args["min_pixels"],
        max_pixels=args["max_pixels"],
    )
    if resized_w <= 0 or resized_h <= 0:
        return None
    scale = (page.width / resized_w, page.height / resized_h)
    # Near-identity (page already within the processor's pixel budget): skip
    # the no-op multiply so coords stay byte-identical to the model output.
    if abs(scale[0] - 1.0) < 1e-6 and abs(scale[1] - 1.0) < 1e-6:
        return None
    return scale


def _smart_resize(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Qwen2-VL / DotsVLProcessor ``smart_resize`` — rescale to a multiple of
    ``factor`` while keeping the pixel count within ``[min_pixels,
    max_pixels]`` and preserving aspect ratio. Mirrors the upstream
    implementation so our bbox rescale matches what the model saw.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

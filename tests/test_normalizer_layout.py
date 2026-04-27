"""LayoutJsonNormalizer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocrscout.errors import NormalizerError
from ocrscout.normalizers.layout_json import LayoutJsonNormalizer
from ocrscout.profile import resolve
from ocrscout.types import PageImage, RawOutput


def _page() -> PageImage:
    return PageImage(page_id="p1", image=object(), width=640, height=1024)


def test_normalizes_dots_mocr_blocks(sample_layout: Path) -> None:
    profile = resolve("dots-mocr")
    raw = RawOutput(
        page_id="p1",
        output_format="layout_json",
        payload=sample_layout.read_text(encoding="utf-8"),
    )
    doc = LayoutJsonNormalizer().normalize(raw, _page(), profile)

    assert len(doc.pages) == 1
    # 1 title + 1 heading + 1 paragraph + 1 page-footer = 4 text items.
    assert len(doc.texts) == 4
    assert len(doc.pictures) == 1

    titles = [t for t in doc.texts if t.label.value == "title"]
    assert len(titles) == 1
    assert titles[0].text == "Annual Report 2024"

    headings = [t for t in doc.texts if t.label.value == "section_header"]
    assert len(headings) == 1
    assert headings[0].text == "Executive Summary"


def test_provenance_carries_bbox(sample_layout: Path) -> None:
    profile = resolve("dots-mocr")
    raw = RawOutput(
        page_id="p1",
        output_format="layout_json",
        payload=sample_layout.read_text(encoding="utf-8"),
    )
    doc = LayoutJsonNormalizer().normalize(raw, _page(), profile)
    title = next(t for t in doc.texts if t.label.value == "title")
    assert title.prov, "title should carry a ProvenanceItem"
    bbox = title.prov[0].bbox
    assert bbox.l == 50.0 and bbox.r == 600.0


def test_rejects_non_layout_json_output_format() -> None:
    profile = resolve("dots-mocr")
    raw = RawOutput(page_id="p1", output_format="markdown", payload="# hi")
    with pytest.raises(NormalizerError):
        LayoutJsonNormalizer().normalize(raw, _page(), profile)


def test_rejects_invalid_json() -> None:
    profile = resolve("dots-mocr")
    raw = RawOutput(page_id="p1", output_format="layout_json", payload="not-json")
    with pytest.raises(NormalizerError):
        LayoutJsonNormalizer().normalize(raw, _page(), profile)


def test_skips_malformed_blocks_but_returns_doc() -> None:
    profile = resolve("dots-mocr")
    blocks = [
        {"category": "Title", "bbox": [10, 10, 100, 50], "text": "OK"},
        {"category": "Text", "bbox": "not-a-bbox", "text": "broken"},
        {"category": "Text", "bbox": [10, 60, 100, 200], "text": "also OK"},
    ]
    raw = RawOutput(page_id="p1", output_format="layout_json", payload=json.dumps(blocks))
    doc = LayoutJsonNormalizer().normalize(raw, _page(), profile)
    # The malformed block should be dropped; the other two survive.
    texts = [t.text for t in doc.texts]
    assert "OK" in texts and "also OK" in texts
    assert "broken" not in texts

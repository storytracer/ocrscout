"""MarkdownNormalizer tests."""

from __future__ import annotations

from ocrscout.normalizers.markdown import MarkdownNormalizer
from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput

PROFILE = ModelProfile(
    name="md-test",
    source="custom",
    model_id="x/y",
    output_format="markdown",
    normalizer="markdown",
)


def _page() -> PageImage:
    return PageImage(page_id="p1", image=object(), width=400, height=600)


def test_h1_h2_paragraphs() -> None:
    md = "# Top Title\n\n## Subhead\n\nFirst paragraph.\n\nSecond paragraph."
    doc = MarkdownNormalizer().normalize(
        RawOutput(page_id="p1", output_format="markdown", payload=md), _page(), PROFILE
    )
    labels = [t.label.value for t in doc.texts]
    assert labels.count("title") == 1
    assert labels.count("section_header") == 1
    # Paragraphs come back as TextItems labeled "paragraph".
    assert labels.count("paragraph") == 2


def test_multiline_paragraph_is_collapsed() -> None:
    md = "A line\nof text\nthat continues."
    doc = MarkdownNormalizer().normalize(
        RawOutput(page_id="p1", output_format="markdown", payload=md), _page(), PROFILE
    )
    paragraphs = [t for t in doc.texts if t.label.value == "paragraph"]
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "A line of text that continues."


def test_empty_input_yields_empty_doc() -> None:
    doc = MarkdownNormalizer().normalize(
        RawOutput(page_id="p1", output_format="markdown", payload=""), _page(), PROFILE
    )
    assert len(doc.texts) == 0

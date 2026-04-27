"""Basic markdown normalizer.

Converts ``# ... ###`` headings (level by leading-hash count) and blank-line-
separated paragraphs into a flat ``DoclingDocument``. No bounding boxes, no
provenance — markdown carries no spatial information.
"""

from __future__ import annotations

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from ocrscout.errors import NormalizerError
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput


class MarkdownNormalizer(Normalizer):
    name = "markdown"
    output_format = "markdown"

    def normalize(
        self, raw: RawOutput, page: PageImage, profile: ModelProfile
    ) -> DoclingDocument:
        if raw.output_format not in ("markdown", "layout_json", "doctags"):
            raise NormalizerError(
                f"MarkdownNormalizer cannot handle output_format={raw.output_format!r}"
            )
        doc = DoclingDocument(name=page.page_id)
        for kind, text, level in _parse_blocks(raw.payload):
            if kind == "heading" and level == 1:
                doc.add_title(text=text)
            elif kind == "heading":
                doc.add_heading(text=text, level=level)
            else:
                doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
        return doc


def _parse_blocks(text: str) -> list[tuple[str, str, int]]:
    """Yield (kind, text, level) tuples. ``kind`` is "heading" or "paragraph"."""
    out: list[tuple[str, str, int]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(s.strip() for s in paragraph if s.strip())
            if joined:
                out.append(("paragraph", joined, 0))
            paragraph.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush_paragraph()
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            flush_paragraph()
            level = 0
            for ch in stripped:
                if ch == "#":
                    level += 1
                else:
                    break
            heading_text = stripped[level:].strip()
            if heading_text:
                level = max(1, min(level, 6))
                out.append(("heading", heading_text, level))
            continue
        paragraph.append(line)
    flush_paragraph()
    return out

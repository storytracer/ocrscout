"""Basic markdown normalizer.

Converts ``# ... ###`` headings (level by leading-hash count) and blank-line-
separated paragraphs into a flat ``DoclingDocument``. No bounding boxes, no
provenance — markdown carries no spatial information.

Tables are recognised in two forms and inserted as proper ``TableItem``s:

- HTML ``<table>...</table>`` blocks anywhere in the payload.
- GitHub-flavoured Markdown pipe tables (header row + ``|---|---|`` delimiter).
"""

from __future__ import annotations

import re

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from ocrscout.errors import NormalizerError
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.normalizers._tables import (
    is_pipe_row,
    looks_like_pipe_table,
    parse_html_table,
    parse_pipe_table,
)
from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput

# Matches a single <table>...</table> span (case-insensitive, greedy on inner content).
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)


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
        for kind, payload, level in _parse_blocks(raw.payload):
            if kind == "heading" and level == 1:
                doc.add_title(text=payload)
            elif kind == "heading":
                doc.add_heading(text=payload, level=level)
            elif kind == "html_table":
                doc.add_table(data=parse_html_table(payload))
            elif kind == "pipe_table":
                doc.add_table(data=parse_pipe_table(payload))
            else:
                doc.add_text(label=DocItemLabel.PARAGRAPH, text=payload)
        return doc


def _parse_blocks(text: str) -> list[tuple[str, str, int]]:
    """Yield (kind, payload, level) tuples.

    ``kind`` is one of: ``heading``, ``paragraph``, ``html_table``, ``pipe_table``.
    For non-heading blocks ``level`` is 0.
    """
    out: list[tuple[str, str, int]] = []

    # First pass: split out HTML <table> blocks so they survive paragraph
    # joining intact. This handles the common case where a model emits the
    # entire <table>...</table> on a single line embedded in prose.
    segments: list[tuple[str, str]] = []  # (kind, content) where kind in {"text", "html_table"}
    pos = 0
    for m in _HTML_TABLE_RE.finditer(text):
        if m.start() > pos:
            segments.append(("text", text[pos : m.start()]))
        segments.append(("html_table", m.group(0)))
        pos = m.end()
    if pos < len(text):
        segments.append(("text", text[pos:]))
    if not segments:
        segments = [("text", text)]

    for seg_kind, content in segments:
        if seg_kind == "html_table":
            out.append(("html_table", content, 0))
            continue
        _parse_text_segment(content, out)
    return out


def _parse_text_segment(text: str, out: list[tuple[str, str, int]]) -> None:
    """Line-based block parser for non-table text: headings, pipe tables, paragraphs."""
    lines = text.splitlines()
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(s.strip() for s in paragraph if s.strip())
            if joined:
                out.append(("paragraph", joined, 0))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.lstrip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if looks_like_pipe_table(lines, i):
            flush_paragraph()
            rows = [lines[i].strip(), lines[i + 1].strip()]
            j = i + 2
            while j < len(lines) and is_pipe_row(lines[j]):
                rows.append(lines[j].strip())
                j += 1
            out.append(("pipe_table", "\n".join(rows), 0))
            i = j
            continue

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
            i += 1
            continue

        paragraph.append(line)
        i += 1
    flush_paragraph()

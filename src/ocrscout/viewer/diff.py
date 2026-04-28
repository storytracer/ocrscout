"""Diff tokenization shared across the comparisons subsystem.

This module used to also house the rendering surfaces (HTML/Gradio/term)
that ``ocrscout inspect --diff`` and the viewer's Diff mode consumed; that
work has moved into :mod:`ocrscout.comparisons` (computation) and
:mod:`ocrscout.comparisons.renderers` (rendering). What remains here is
the word/paragraph tokenizer — used by :class:`TextComparison` and by
:func:`ocrscout.publish._stats.compute_page_disagreement`.
"""

from __future__ import annotations

import re

# Word-level diff tokenizer: ``\S+`` captures a word, ``\n+`` captures a
# paragraph break. Inline whitespace is dropped (renderers re-emit single
# spaces between words), but newline runs survive as their own tokens so
# paragraphs stay visible in the diff output.
DIFF_TOKEN_RE = re.compile(r"\S+|\n+")


def tokenize(text: str) -> list[str]:
    """Word + paragraph-break tokenization for diff alignment."""
    return DIFF_TOKEN_RE.findall(text)

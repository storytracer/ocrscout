"""TextComparison: word-level agreement between two text artifacts.

Produces a ``TextComparisonResult`` carrying:

* ``similarity`` — ``difflib.SequenceMatcher`` word-level ratio scaled to
  0..100. Always present when both sides have text. Same metric the viewer
  and inspect surfaces render in their diff tables, so the number in the
  parquet matches the one users see in Compare mode.
* ``cer`` and ``wer`` — character / word error rates from ``jiwer``, only
  populated when the optional ``[eval]`` extra is installed. Provided
  because they're the lingua franca of the OCR/ASR community even though
  similarity captures the same information shape.
* ``opcodes`` + ``pred_tokens`` + ``base_tokens`` — the full diff payload,
  consumed by ``TextComparisonRenderer`` for HTML/Gradio/terminal output.

Diff direction: ``prediction`` is the left/A side, ``baseline`` is the
right/B side. ``opcodes`` indices reference ``pred_tokens`` for ``i1/i2``
and ``base_tokens`` for ``j1/j2``, matching ``SequenceMatcher.get_opcodes``.
"""

from __future__ import annotations

import difflib
from typing import ClassVar, Literal

from ocrscout.interfaces.comparison import (
    BaselineView,
    Comparison,
    ComparisonResult,
    PredictionView,
)
from ocrscout.viewer.diff import tokenize


class TextComparisonResult(ComparisonResult):
    comparison: Literal["text"] = "text"
    similarity: float
    cer: float | None = None
    wer: float | None = None
    opcodes: list[tuple[str, int, int, int, int]]
    pred_tokens: list[str]
    base_tokens: list[str]
    common: int
    removed: int
    added: int


class TextComparison(Comparison):
    name = "text"
    requires: ClassVar[frozenset[str]] = frozenset({"text"})

    def compare(
        self, prediction: PredictionView, baseline: BaselineView
    ) -> TextComparisonResult | None:
        a = (prediction.text or "").strip()
        b = (baseline.text or "").strip()
        if not a or not b:
            return None
        toks_a = tokenize(a)
        toks_b = tokenize(b)
        matcher = difflib.SequenceMatcher(None, toks_a, toks_b, autojunk=False)
        opcodes: list[tuple[str, int, int, int, int]] = [
            (str(tag), i1, i2, j1, j2)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        ]
        similarity = matcher.ratio() * 100
        common = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal")
        removed = sum(
            i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("delete", "replace")
        )
        added = sum(
            j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("insert", "replace")
        )
        result = TextComparisonResult(
            similarity=similarity,
            opcodes=opcodes,
            pred_tokens=toks_a,
            base_tokens=toks_b,
            common=common,
            removed=removed,
            added=added,
            summary={"similarity": similarity},
        )
        # CER/WER from jiwer when [eval] is installed. Lazy-imported so the
        # core install stays cheap.
        try:
            import jiwer

            cer = float(jiwer.cer(b, a))
            wer = float(jiwer.wer(b, a))
        except ImportError:
            return result
        result.cer = cer
        result.wer = wer
        result.summary["cer"] = cer
        result.summary["wer"] = wer
        return result

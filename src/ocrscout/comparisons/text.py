"""TextComparison: line+word agreement between two text artifacts.

Produces a ``TextComparisonResult`` carrying:

* ``similarity`` — ``difflib.SequenceMatcher`` word-level ratio scaled to
  0..100. Always present when both sides have text.
* ``cer`` and ``wer`` — character / word error rates from ``jiwer``, only
  populated when the optional ``[eval]`` extra is installed.
* ``pred_tokens`` / ``base_tokens`` / ``opcodes`` — the word-level diff
  payload (legacy field; kept for terminal renderers and small-diff fallback).
* ``pred_lines`` / ``base_lines`` / ``line_opcodes`` — the line-level diff,
  consumed by the VSCode-style renderer (split + unified). ``line_opcodes``
  is a list of ``(tag, i1, i2, j1, j2)`` tuples where indices reference
  ``pred_lines`` (i) and ``base_lines`` (j).
* ``inline_word_opcodes`` — for each ``replace`` line opcode, the per-line
  word-level diff used to highlight the actual changed words inside a
  modified line. Keyed by pred-line index (str — JSON dict keys must be
  strings). Each value is a list of ``(tag, i1, i2, j1, j2)`` over the
  *line's* tokens.

Diff direction: ``prediction`` is the left/A side, ``baseline`` is the
right/B side.
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
    # Legacy word-level diff (kept for back-compat & small-diff fallback).
    opcodes: list[tuple[str, int, int, int, int]]
    pred_tokens: list[str]
    base_tokens: list[str]
    common: int
    removed: int
    added: int
    # Line-level diff (used by the VSCode-style renderer).
    pred_lines: list[str] = []
    base_lines: list[str] = []
    line_opcodes: list[tuple[str, int, int, int, int]] = []
    # Per-replace-line word-level diff. Keys are pred_lines indices as
    # strings (JSON requirement); values are the line's word opcodes.
    inline_word_opcodes: dict[str, list[tuple[str, int, int, int, int]]] = {}
    # Line-level summary counts.
    lines_added: int = 0
    lines_removed: int = 0
    lines_unchanged: int = 0


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

        # Word-level diff (legacy payload). Used by the terminal renderer
        # and as a small-diff fallback.
        toks_a = tokenize(a)
        toks_b = tokenize(b)
        word_matcher = difflib.SequenceMatcher(None, toks_a, toks_b, autojunk=False)
        opcodes: list[tuple[str, int, int, int, int]] = [
            (str(tag), i1, i2, j1, j2)
            for tag, i1, i2, j1, j2 in word_matcher.get_opcodes()
        ]
        similarity = word_matcher.ratio() * 100
        common = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal")
        removed = sum(
            i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("delete", "replace")
        )
        added = sum(
            j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("insert", "replace")
        )

        # Line-level diff. Splits on newline so the renderer can render
        # VSCode-style line-aligned panes; for ``replace`` runs we go one
        # level deeper and run a per-line word diff (greedy 1:1 pairing,
        # overflow is flushed as solo delete/insert lines so the line-pair
        # word diff stays cheap).
        pred_lines = a.split("\n")
        base_lines = b.split("\n")
        line_matcher = difflib.SequenceMatcher(
            None, pred_lines, base_lines, autojunk=False
        )
        line_opcodes: list[tuple[str, int, int, int, int]] = [
            (str(tag), i1, i2, j1, j2)
            for tag, i1, i2, j1, j2 in line_matcher.get_opcodes()
        ]
        inline_word_opcodes: dict[
            str, list[tuple[str, int, int, int, int]]
        ] = {}
        for tag, i1, i2, j1, j2 in line_opcodes:
            if tag != "replace":
                continue
            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                pred_idx = i1 + k
                base_idx = j1 + k
                pred_toks = tokenize(pred_lines[pred_idx])
                base_toks = tokenize(base_lines[base_idx])
                if not pred_toks and not base_toks:
                    continue
                m = difflib.SequenceMatcher(
                    None, pred_toks, base_toks, autojunk=False
                )
                inline_word_opcodes[str(pred_idx)] = [
                    (str(t), ii1, ii2, jj1, jj2)
                    for t, ii1, ii2, jj1, jj2 in m.get_opcodes()
                ]
        lines_unchanged = sum(
            i2 - i1 for tag, i1, i2, _, _ in line_opcodes if tag == "equal"
        )
        lines_removed = sum(
            i2 - i1 for tag, i1, i2, _, _ in line_opcodes
            if tag in ("delete", "replace")
        )
        lines_added = sum(
            j2 - j1 for tag, _, _, j1, j2 in line_opcodes
            if tag in ("insert", "replace")
        )

        result = TextComparisonResult(
            similarity=similarity,
            opcodes=opcodes,
            pred_tokens=toks_a,
            base_tokens=toks_b,
            common=common,
            removed=removed,
            added=added,
            pred_lines=pred_lines,
            base_lines=base_lines,
            line_opcodes=line_opcodes,
            inline_word_opcodes=inline_word_opcodes,
            lines_added=lines_added,
            lines_removed=lines_removed,
            lines_unchanged=lines_unchanged,
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

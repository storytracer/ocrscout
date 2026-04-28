"""Renderer for ``TextComparisonResult`` — VSCode-style line+word hybrid diff.

The result already carries:

* word-level opcodes over the full token streams (legacy / fallback),
* line-level opcodes over the lines of each side, plus
* per-replace-line word-level opcodes.

Two render modes are produced for the HTML/Gradio surfaces: ``split``
(two columns, line-aligned, deletions on the left and insertions on the
right) and ``unified`` (single column, ``+``/``−`` prefixes). The mode is
chosen client-side via a radio toggle so the same DOM can flip without a
server round-trip. A "changes only" toggle collapses long runs of
unchanged lines.

The terminal renderer emits a unified line diff with VSCode-style tints,
suitable for ``ocrscout inspect --compare``.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from rich.text import Text

from ocrscout.comparisons.text import TextComparisonResult
from ocrscout.interfaces.comparison import ComparisonRenderer, ComparisonResult

if TYPE_CHECKING:
    from rich.console import Console


class TextComparisonRenderer(ComparisonRenderer):
    name = "text"

    def render_html(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
    ) -> str:
        if not isinstance(result, TextComparisonResult):
            return _wrong_result_html(result)
        body = _render_diff_html(result, prediction_label, baseline_label)
        return _STANDALONE_TEMPLATE.format(
            title=escape(f"{prediction_label} ↔ {baseline_label}"),
            body=body,
            css=_DIFF_CSS,
            js=_DIFF_JS,
        )

    def render_gradio(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
    ) -> str:
        if not isinstance(result, TextComparisonResult):
            return _wrong_result_html(result)
        return _render_diff_html(result, prediction_label, baseline_label)

    def render_terminal(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
        console: Console,
    ) -> None:
        if not isinstance(result, TextComparisonResult):
            console.print(
                f"[yellow]TextRenderer can't render {type(result).__name__}[/]"
            )
            return

        console.print(
            f"\n[bold cyan]=== diff[/bold cyan]   "
            f"[red]── {baseline_label}[/red]   "
            f"[green]── {prediction_label}[/green]"
        )
        line_summary = (
            f"+{result.lines_added} −{result.lines_removed} "
            f"({result.lines_unchanged} unchanged)"
        )
        console.print(
            f"[dim]similarity {result.similarity:.1f}%  ·  {line_summary}"
            + (
                f"  ·  cer {result.cer:.3f}"
                if result.cer is not None else ""
            )
            + (
                f"  ·  wer {result.wer:.3f}"
                if result.wer is not None else ""
            )
            + "[/dim]\n"
        )

        # Prefer the line-level opcodes; fall back to the word-level
        # rendering for older parquet-loaded results that predate them.
        if result.line_opcodes:
            console.print(_render_terminal_unified(result))
        else:
            console.print(_render_terminal_word_level(result))


# --------------------------------------------------------------------- terminal


def _render_terminal_unified(result: TextComparisonResult) -> Text:
    """Unified line diff with `+`/`−` prefixes, VSCode-style tints."""
    out = Text()
    base = result.base_lines
    pred = result.pred_lines
    for tag, i1, i2, j1, j2 in result.line_opcodes:
        if tag == "equal":
            for k in range(i1, i2):
                out.append("  " + base[k] + "\n", style="dim")
        elif tag == "delete":
            for k in range(i1, i2):
                out.append("- " + base[k] + "\n", style="red")
        elif tag == "insert":
            for k in range(j1, j2):
                out.append("+ " + pred[k] + "\n", style="green")
        elif tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                inline = result.inline_word_opcodes.get(str(i1 + k))
                if inline:
                    _emit_replace_pair_terminal(
                        out, base[i1 + k], pred[j1 + k], inline,
                    )
                else:
                    out.append("- " + base[i1 + k] + "\n", style="red")
                    out.append("+ " + pred[j1 + k] + "\n", style="green")
            for k in range(pairs, i2 - i1):
                out.append("- " + base[i1 + k] + "\n", style="red")
            for k in range(pairs, j2 - j1):
                out.append("+ " + pred[j1 + k] + "\n", style="green")
    return out


def _emit_replace_pair_terminal(
    out: Text,
    base_line: str,
    pred_line: str,
    inline_opcodes: list[tuple[str, int, int, int, int]],
) -> None:
    """Render a `replace` line pair with word-level highlighting."""
    from ocrscout.viewer.diff import tokenize

    base_toks = tokenize(base_line)
    pred_toks = tokenize(pred_line)

    out.append("- ", style="red")
    for tag, i1, i2, _j1, _j2 in inline_opcodes:
        if tag == "equal":
            for tok in base_toks[i1:i2]:
                _emit_token(out, tok, style="red")
        elif tag in ("delete", "replace"):
            for tok in base_toks[i1:i2]:
                _emit_token(out, tok, style="red bold underline")
    out.append("\n")

    out.append("+ ", style="green")
    for tag, _i1, _i2, j1, j2 in inline_opcodes:
        if tag == "equal":
            for tok in pred_toks[j1:j2]:
                _emit_token(out, tok, style="green")
        elif tag in ("insert", "replace"):
            for tok in pred_toks[j1:j2]:
                _emit_token(out, tok, style="green bold underline")
    out.append("\n")


def _render_terminal_word_level(result: TextComparisonResult) -> Text:
    """Fallback used when only word-level opcodes are present."""
    rendered = Text()
    for tag, i1, i2, j1, j2 in result.opcodes:
        if tag == "equal":
            for tok in result.base_tokens[i1:i2]:
                _emit_token(rendered, tok, style="")
        elif tag == "delete":
            for tok in result.base_tokens[i1:i2]:
                _emit_token(rendered, tok, style="red strike")
        elif tag == "insert":
            for tok in result.pred_tokens[j1:j2]:
                _emit_token(rendered, tok, style="green")
        elif tag == "replace":
            for tok in result.base_tokens[i1:i2]:
                _emit_token(rendered, tok, style="red strike")
            for tok in result.pred_tokens[j1:j2]:
                _emit_token(rendered, tok, style="green")
    return rendered


def _emit_token(text_obj: Text, token: str, *, style: str) -> None:
    if "\n" in token:
        text_obj.append(token)
    else:
        text_obj.append(token + " ", style=style)


# ----------------------------------------------------------------------- HTML


def _render_diff_html(
    result: TextComparisonResult,
    prediction_label: str,
    baseline_label: str,
) -> str:
    """Build the shared diff DOM (used by both render_html and render_gradio).

    Emits both ``split`` and ``unified`` variants so the JS toggle can
    swap between them without a round-trip — only one is visible at a
    time via CSS. The minimap is rendered alongside the line lists.
    """
    if not result.line_opcodes:
        return _render_word_only_html(result, prediction_label, baseline_label)

    split_rows, unified_rows, minimap_rows = _build_rows(result)
    header = _render_header(result, prediction_label, baseline_label)
    split_table = (
        '<table class="ocrscout-diff-split">'
        f"<thead><tr>"
        f'<th class="ln"></th>'
        f'<th class="side base">{escape(baseline_label)}</th>'
        f'<th class="ln"></th>'
        f'<th class="side pred">{escape(prediction_label)}</th>'
        f"</tr></thead>"
        f"<tbody>{''.join(split_rows)}</tbody>"
        "</table>"
    )
    unified_table = (
        '<table class="ocrscout-diff-unified">'
        "<thead><tr>"
        f'<th class="ln"></th>'
        f'<th class="ln"></th>'
        f'<th class="side">'
        f'<span class="lab base">{escape(baseline_label)}</span>'
        f'<span class="lab-sep">↔</span>'
        f'<span class="lab pred">{escape(prediction_label)}</span>'
        "</th>"
        "</tr></thead>"
        f"<tbody>{''.join(unified_rows)}</tbody>"
        "</table>"
    )
    minimap = (
        '<div class="ocrscout-diff-minimap" aria-hidden="true">'
        + "".join(minimap_rows)
        + "</div>"
    )
    return (
        '<div class="ocrscout-diff" data-mode="split" data-changes-only="0">'
        + header
        + '<div class="ocrscout-diff-body">'
        + '<div class="ocrscout-diff-pane">'
        + split_table
        + unified_table
        + "</div>"
        + minimap
        + "</div>"
        + "</div>"
    )


def _render_header(
    result: TextComparisonResult,
    prediction_label: str,
    baseline_label: str,
) -> str:
    cer_html = (
        f'<span class="stat cer">CER {result.cer:.3f}</span>'
        if result.cer is not None
        else '<span class="stat cer dim">CER —</span>'
    )
    wer_html = (
        f'<span class="stat wer">WER {result.wer:.3f}</span>'
        if result.wer is not None
        else '<span class="stat wer dim">WER —</span>'
    )
    return (
        '<div class="ocrscout-diff-header">'
        '<div class="diff-stats">'
        f'<span class="stat similarity">{result.similarity:.1f}% similar</span>'
        f'<span class="stat added">+{result.lines_added}</span>'
        f'<span class="stat removed">−{result.lines_removed}</span>'
        f'<span class="stat dim">{result.lines_unchanged} unchanged</span>'
        f"{cer_html}{wer_html}"
        '<span class="diff-legend">'
        f'<span class="legend-chip base">■ {escape(baseline_label)}</span>'
        f'<span class="legend-chip pred">■ {escape(prediction_label)}</span>'
        f'<span class="legend-chip word-key">▪ word diff</span>'
        "</span>"
        "</div>"
        '<div class="diff-controls">'
        '<div class="diff-mode" role="radiogroup" aria-label="Diff mode">'
        '<button type="button" class="mode-btn active" data-mode="split">Split</button>'
        '<button type="button" class="mode-btn" data-mode="unified">Unified</button>'
        "</div>"
        '<label class="diff-toggle">'
        '<input type="checkbox" class="changes-only-toggle"> Changes only'
        "</label>"
        "</div>"
        "</div>"
    )


def _build_rows(
    result: TextComparisonResult,
) -> tuple[list[str], list[str], list[str]]:
    """Build split/unified/minimap row HTML in one pass over line_opcodes.

    Equal-line runs share a CSS class so the "changes only" toggle can
    collapse them via a single attribute switch on the parent.

    Index convention: i indexes ``base_lines`` (left, removed/red),
    j indexes ``pred_lines`` (right, added/green).
    """
    base = result.base_lines
    pred = result.pred_lines
    inline = result.inline_word_opcodes

    split_rows: list[str] = []
    unified_rows: list[str] = []
    minimap_rows: list[str] = []

    base_n = 1
    pred_n = 1
    for tag, i1, i2, j1, j2 in result.line_opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                line = base[i1 + k]
                bn = base_n + k
                pn = pred_n + k
                escaped = _escape_line(line)
                split_rows.append(
                    f'<tr class="row equal">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="line equal">{escaped}</td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line equal">{escaped}</td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row equal">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line equal"><span class="marker"> </span>{escaped}</td>'
                    f"</tr>"
                )
                minimap_rows.append('<span class="mm equal"></span>')
            base_n += i2 - i1
            pred_n += j2 - j1
        elif tag == "delete":
            for k in range(i2 - i1):
                line = base[i1 + k]
                bn = base_n + k
                escaped = _escape_line(line)
                split_rows.append(
                    f'<tr class="row delete">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="line delete">{escaped}</td>'
                    f'<td class="ln"></td>'
                    f'<td class="line empty"></td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row delete">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="ln"></td>'
                    f'<td class="line delete"><span class="marker">−</span>{escaped}</td>'
                    f"</tr>"
                )
                minimap_rows.append('<span class="mm delete"></span>')
            base_n += i2 - i1
        elif tag == "insert":
            for k in range(j2 - j1):
                line = pred[j1 + k]
                pn = pred_n + k
                escaped = _escape_line(line)
                split_rows.append(
                    f'<tr class="row insert">'
                    f'<td class="ln"></td>'
                    f'<td class="line empty"></td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line insert">{escaped}</td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row insert">'
                    f'<td class="ln"></td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line insert"><span class="marker">+</span>{escaped}</td>'
                    f"</tr>"
                )
                minimap_rows.append('<span class="mm insert"></span>')
            pred_n += j2 - j1
        elif tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            # Paired modified lines: word-level highlights on both sides.
            for k in range(pairs):
                base_idx = i1 + k
                pred_idx = j1 + k
                bn = base_n + k
                pn = pred_n + k
                base_line = base[base_idx]
                pred_line = pred[pred_idx]
                opcodes = inline.get(str(base_idx), [])
                left_html, right_html = _render_word_pair_html(
                    base_line, pred_line, opcodes,
                )
                split_rows.append(
                    f'<tr class="row replace">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="line delete">{left_html}</td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line insert">{right_html}</td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row delete">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="ln"></td>'
                    f'<td class="line delete"><span class="marker">−</span>{left_html}</td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row insert">'
                    f'<td class="ln"></td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line insert"><span class="marker">+</span>{right_html}</td>'
                    f"</tr>"
                )
                minimap_rows.append('<span class="mm replace"></span>')
            # Overflow (uneven pair) flushed as solo delete/insert lines.
            for k in range(pairs, i2 - i1):
                base_idx = i1 + k
                bn = base_n + k
                escaped = _escape_line(base[base_idx])
                split_rows.append(
                    f'<tr class="row delete">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="line delete">{escaped}</td>'
                    f'<td class="ln"></td>'
                    f'<td class="line empty"></td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row delete">'
                    f'<td class="ln">{bn}</td>'
                    f'<td class="ln"></td>'
                    f'<td class="line delete"><span class="marker">−</span>{escaped}</td>'
                    f"</tr>"
                )
                minimap_rows.append('<span class="mm delete"></span>')
            for k in range(pairs, j2 - j1):
                pred_idx = j1 + k
                pn = pred_n + k
                escaped = _escape_line(pred[pred_idx])
                split_rows.append(
                    f'<tr class="row insert">'
                    f'<td class="ln"></td>'
                    f'<td class="line empty"></td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line insert">{escaped}</td>'
                    f"</tr>"
                )
                unified_rows.append(
                    f'<tr class="row insert">'
                    f'<td class="ln"></td>'
                    f'<td class="ln">{pn}</td>'
                    f'<td class="line insert"><span class="marker">+</span>{escaped}</td>'
                    f"</tr>"
                )
                minimap_rows.append('<span class="mm insert"></span>')
            base_n += i2 - i1
            pred_n += j2 - j1

    return split_rows, unified_rows, minimap_rows


def _render_word_pair_html(
    base_line: str,
    pred_line: str,
    inline_opcodes: list[tuple[str, int, int, int, int]],
) -> tuple[str, str]:
    """Render the left/right HTML for a `replace` line pair with word-level
    highlighting. Falls back to the whole line tinted when no opcodes.

    Index convention: ``i`` indexes baseline tokens (left), ``j`` indexes
    prediction tokens (right).
    """
    from ocrscout.viewer.diff import tokenize

    base_toks = tokenize(base_line)
    pred_toks = tokenize(pred_line)

    if not inline_opcodes:
        return _escape_line(base_line), _escape_line(pred_line)

    left_parts: list[str] = []
    right_parts: list[str] = []
    for tag, i1, i2, j1, j2 in inline_opcodes:
        left_segment = " ".join(base_toks[i1:i2])
        right_segment = " ".join(pred_toks[j1:j2])
        if tag == "equal":
            left_parts.append(escape(left_segment))
            right_parts.append(escape(right_segment))
        elif tag == "delete":
            left_parts.append(
                f'<span class="word-delete">{escape(left_segment)}</span>'
            )
        elif tag == "insert":
            right_parts.append(
                f'<span class="word-insert">{escape(right_segment)}</span>'
            )
        elif tag == "replace":
            if left_segment:
                left_parts.append(
                    f'<span class="word-delete">{escape(left_segment)}</span>'
                )
            if right_segment:
                right_parts.append(
                    f'<span class="word-insert">{escape(right_segment)}</span>'
                )

    left_html = " ".join(p for p in left_parts if p) or "&nbsp;"
    right_html = " ".join(p for p in right_parts if p) or "&nbsp;"
    return left_html, right_html


def _escape_line(line: str) -> str:
    """Escape an HTML line, replacing leading whitespace with non-breaking
    spaces so indentation reads correctly inside table cells."""
    if not line:
        return "&nbsp;"
    stripped = line.lstrip()
    leading = len(line) - len(stripped)
    return ("&nbsp;" * leading) + escape(stripped)


def _render_word_only_html(
    result: TextComparisonResult,
    prediction_label: str,
    baseline_label: str,
) -> str:
    """Fallback for old parquet-loaded results without line opcodes.

    Renders the legacy word-level table; users of older runs still get a
    diff, just without the line gutter or unified mode.
    """
    rows: list[str] = []
    for tag, i1, i2, j1, j2 in result.opcodes:
        left = _tokens_to_html(result.base_tokens[i1:i2])
        right = _tokens_to_html(result.pred_tokens[j1:j2])
        if tag == "equal":
            rows.append(
                f'<tr class="row equal"><td class="line">{left}</td>'
                f'<td class="line">{right}</td></tr>'
            )
        elif tag == "delete":
            rows.append(
                f'<tr class="row delete"><td class="line delete">{left}</td>'
                f'<td class="line empty"></td></tr>'
            )
        elif tag == "insert":
            rows.append(
                f'<tr class="row insert"><td class="line empty"></td>'
                f'<td class="line insert">{right}</td></tr>'
            )
        elif tag == "replace":
            rows.append(
                f'<tr class="row replace"><td class="line delete">{left}</td>'
                f'<td class="line insert">{right}</td></tr>'
            )
    header = _render_header(result, prediction_label, baseline_label)
    return (
        '<div class="ocrscout-diff legacy-word-diff" data-mode="split">'
        + header
        + '<div class="ocrscout-diff-body">'
        '<div class="ocrscout-diff-pane">'
        '<table class="ocrscout-diff-split">'
        "<thead><tr>"
        f'<th class="side base">{escape(baseline_label)}</th>'
        f'<th class="side pred">{escape(prediction_label)}</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></div></div>"
    )


def _tokens_to_html(tokens: list[str]) -> str:
    parts: list[str] = []
    for tok in tokens:
        if "\n" in tok:
            parts.append("<br>" * tok.count("\n"))
        else:
            parts.append(escape(tok) + " ")
    return "".join(parts).rstrip() or "&nbsp;"


def _wrong_result_html(result: ComparisonResult) -> str:
    return (
        f"<p>TextComparisonRenderer cannot render "
        f"<code>{escape(type(result).__name__)}</code>.</p>"
    )


# --------------------------------------------------------------------- standalone


_STANDALONE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
<script>{js}</script>
</body>
</html>
"""

# Embedded copy of the diff CSS — kept in lockstep with viewer.css so the
# standalone HTML page (served by `inspect --compare --html`) renders
# identically to the Gradio Compare view. Update both when changing styles.
_DIFF_CSS = """
:root { color-scheme: light dark; }
body {
  margin: 0; padding: 0;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  background: #f6f8fa;
  color: #1f2328;
}
.ocrscout-diff {
  display: flex; flex-direction: column;
  height: 100vh;
  background: #fff;
}
.ocrscout-diff-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap;
  padding: 0.6rem 1rem;
  background: #fff;
  border-bottom: 1px solid #d0d7de;
  position: sticky; top: 0; z-index: 5;
}
.diff-stats { display: flex; gap: 0.65rem; align-items: center; flex-wrap: wrap; }
.diff-stats .stat {
  font-size: 0.8rem;
  font-weight: 500;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  background: #eaeef2;
  color: #1f2328;
  border: 1px solid #d0d7de;
}
.diff-stats .stat.similarity { background: #eef6ff; border-color: #b6e3ff; color: #0969da; }
.diff-stats .stat.added { background: #e6ffec; border-color: #a6e9b5; color: #1a7f37; }
.diff-stats .stat.removed { background: #ffeef0; border-color: #ffabba; color: #cf222e; }
.diff-stats .stat.dim { color: #59636e; }
.diff-legend {
  display: inline-flex; gap: 0.5rem; align-items: center;
  margin-left: 0.25rem;
  padding-left: 0.65rem;
  border-left: 1px solid #d0d7de;
  font-size: 0.75rem;
}
.legend-chip { font-weight: 600; }
.legend-chip.base { color: #cf222e; }
.legend-chip.pred { color: #1a7f37; }
.legend-chip.word-key { color: #59636e; font-weight: 400; font-style: italic; }
.diff-controls { display: flex; align-items: center; gap: 0.75rem; }
.diff-mode { display: inline-flex; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; }
.diff-mode .mode-btn {
  border: 0; background: #fff; padding: 0.3rem 0.7rem;
  font-size: 0.8rem; cursor: pointer; color: #1f2328;
  border-right: 1px solid #d0d7de;
}
.diff-mode .mode-btn:last-child { border-right: 0; }
.diff-mode .mode-btn.active { background: #0969da; color: #fff; }
.diff-toggle {
  display: inline-flex; align-items: center; gap: 0.3rem;
  font-size: 0.8rem; color: #1f2328; cursor: pointer;
}
.ocrscout-diff-body {
  display: flex;
  flex: 1;
  overflow: hidden;
  position: relative;
}
.ocrscout-diff-pane {
  flex: 1;
  overflow: auto;
  background: #fff;
}
.ocrscout-diff-pane table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 0.85rem;
  line-height: 1.45;
}
.ocrscout-diff-pane thead th {
  position: sticky; top: 0; z-index: 1;
  text-align: left;
  padding: 0.4rem 0.6rem;
  background: #f6f8fa;
  border-bottom: 1px solid #d0d7de;
  font-size: 0.75rem;
  font-weight: 600;
  color: #59636e;
}
.ocrscout-diff-pane th.ln { width: 3rem; text-align: right; padding-right: 0.5rem; color: #8c959f; }
.ocrscout-diff-pane th.side.base { color: #cf222e; }
.ocrscout-diff-pane th.side.pred { color: #1a7f37; }
.ocrscout-diff-pane th.side .lab { font-weight: 600; }
.ocrscout-diff-pane th.side .lab.base { color: #cf222e; }
.ocrscout-diff-pane th.side .lab.pred { color: #1a7f37; }
.ocrscout-diff-pane th.side .lab-sep { color: #8c959f; padding: 0 0.5rem; font-weight: 400; }
.ocrscout-diff-pane td.ln {
  width: 3rem;
  text-align: right;
  padding: 0 0.5rem;
  color: #8c959f;
  user-select: none;
  border-right: 1px solid #eaeef2;
  font-variant-numeric: tabular-nums;
  font-size: 0.75rem;
  vertical-align: top;
  background: #f6f8fa;
}
.ocrscout-diff-pane td.line {
  padding: 0 0.6rem;
  white-space: pre-wrap;
  word-break: break-word;
  vertical-align: top;
}
.ocrscout-diff-pane td.line.empty { background: #f6f8fa; }
.ocrscout-diff-pane td.line.equal { color: #1f2328; }
.ocrscout-diff-pane td.line.delete {
  border-left: 3px solid #cf222e;
}
.ocrscout-diff-pane td.line.insert {
  border-left: 3px solid #1a7f37;
}
.ocrscout-diff-pane td.line .marker {
  display: inline-block;
  width: 1ch;
  color: #59636e;
  font-weight: 600;
  margin-right: 0.4rem;
}
.ocrscout-diff-pane tr.delete td.line.delete .marker { color: #cf222e; }
.ocrscout-diff-pane tr.insert td.line.insert .marker { color: #1a7f37; }
.ocrscout-diff-pane td.line .word-delete {
  background: rgba(207, 34, 46, 0.25);
  color: #82071e;
  font-weight: 600;
  border-radius: 3px;
  padding: 0 2px;
}
.ocrscout-diff-pane td.line .word-insert {
  background: rgba(26, 127, 55, 0.28);
  color: #044317;
  font-weight: 600;
  border-radius: 3px;
  padding: 0 2px;
}

/* Mode visibility: split / unified are siblings in the same container.
   The toggle on .ocrscout-diff[data-mode] flips which is rendered. */
.ocrscout-diff[data-mode="split"] .ocrscout-diff-unified { display: none; }
.ocrscout-diff[data-mode="unified"] .ocrscout-diff-split { display: none; }

/* Changes-only filter — collapse equal rows. */
.ocrscout-diff[data-changes-only="1"] tr.equal { display: none; }

/* Minimap */
.ocrscout-diff-minimap {
  width: 90px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #f6f8fa;
  border-left: 1px solid #d0d7de;
  overflow-y: auto;
  padding: 0.4rem 0;
}
.ocrscout-diff-minimap .mm {
  display: block;
  width: 100%;
  height: 3px;
  margin: 0.5px 0;
}
.ocrscout-diff-minimap .mm.equal { background: #eaeef2; }
.ocrscout-diff-minimap .mm.delete { background: #cf222e; }
.ocrscout-diff-minimap .mm.insert { background: #1a7f37; }
.ocrscout-diff-minimap .mm.replace {
  background: linear-gradient(90deg, #cf222e 50%, #1a7f37 50%);
}

@media (prefers-color-scheme: dark) {
  body { background: #0d1117; color: #c9d1d9; }
  .ocrscout-diff, .ocrscout-diff-pane { background: #0d1117; color: #c9d1d9; }
  .ocrscout-diff-header { background: #161b22; border-color: #30363d; }
  .diff-stats .stat { background: #21262d; border-color: #30363d; color: #c9d1d9; }
  .diff-stats .stat.similarity { background: #0c2d6b; border-color: #1f6feb; color: #79c0ff; }
  .diff-stats .stat.added { background: #033a16; border-color: #196c2e; color: #56d364; }
  .diff-stats .stat.removed { background: #67060c; border-color: #8e1519; color: #ff7b72; }
  .diff-mode { border-color: #30363d; }
  .diff-mode .mode-btn { background: #161b22; color: #c9d1d9; border-color: #30363d; }
  .diff-mode .mode-btn.active { background: #1f6feb; color: #fff; }
  .ocrscout-diff-pane thead th { background: #161b22; color: #8b949e; border-color: #30363d; }
  .ocrscout-diff-pane td.ln { background: #161b22; color: #6e7681; border-color: #21262d; }
  .ocrscout-diff-pane td.line.empty { background: #161b22; }
  .ocrscout-diff-pane td.line.equal { color: #c9d1d9; }
  .ocrscout-diff-pane td.line.delete { border-left-color: #f85149; }
  .ocrscout-diff-pane td.line.insert { border-left-color: #3fb950; }
  .ocrscout-diff-pane td.line .word-delete { background: rgba(248, 81, 73, 0.30); color: #ffa198; }
  .ocrscout-diff-pane td.line .word-insert { background: rgba(63, 185, 80, 0.30); color: #56d364; }
  .ocrscout-diff-minimap { background: #161b22; border-color: #30363d; }
  .ocrscout-diff-minimap .mm.equal { background: #21262d; }
  .ocrscout-diff-minimap .mm.delete { background: #f85149; }
  .ocrscout-diff-minimap .mm.insert { background: #3fb950; }
  .ocrscout-diff-minimap .mm.replace {
    background: linear-gradient(90deg, #f85149 50%, #3fb950 50%);
  }
}
"""

_DIFF_JS = """
(function() {
  function bind(root) {
    const modeButtons = root.querySelectorAll('.diff-mode .mode-btn');
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        const diff = btn.closest('.ocrscout-diff');
        if (!diff) return;
        diff.dataset.mode = mode;
        diff.querySelectorAll('.diff-mode .mode-btn').forEach(b => {
          b.classList.toggle('active', b === btn);
        });
      });
    });
    const toggles = root.querySelectorAll('.changes-only-toggle');
    toggles.forEach(toggle => {
      toggle.addEventListener('change', () => {
        const diff = toggle.closest('.ocrscout-diff');
        if (!diff) return;
        diff.dataset.changesOnly = toggle.checked ? '1' : '0';
      });
    });
    const minimaps = root.querySelectorAll('.ocrscout-diff-minimap');
    minimaps.forEach(mm => {
      mm.addEventListener('click', (e) => {
        const diff = mm.closest('.ocrscout-diff');
        if (!diff) return;
        const pane = diff.querySelector('.ocrscout-diff-pane');
        if (!pane) return;
        const rect = mm.getBoundingClientRect();
        const ratio = (e.clientY - rect.top) / rect.height;
        pane.scrollTo({ top: ratio * pane.scrollHeight, behavior: 'smooth' });
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => bind(document));
  } else {
    bind(document);
  }
  // Re-bind when Gradio re-renders the comparison HTML.
  if (typeof window !== 'undefined') {
    window.ocrscoutBindDiff = bind;
  }
})();
"""

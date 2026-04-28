"""Renderer for ``DocumentComparisonResult`` — chip cards + stacked bar.

Replaces the bare HTML table with a more scannable presentation:

* Chip cards for the headline metrics (Headings / Tables / Pictures), each
  with directional arrows and color-coded deltas.
* A horizontal stacked bar comparing the prediction's vs the baseline's
  item-type distribution.
* A "missing structure" callout when one side has 0 of something the
  other side has many of (the load-bearing signal for "this model
  silently dropped tables").

Same DOM is emitted for the standalone HTML page (`render_html`) and
for the Gradio Compare embed (`render_gradio`); the standalone page just
wraps it with inline CSS that mirrors viewer.css.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from rich.table import Table

from ocrscout.comparisons.document import DocumentComparisonResult
from ocrscout.interfaces.comparison import ComparisonRenderer, ComparisonResult

if TYPE_CHECKING:
    from rich.console import Console


# Colors used for the stacked-bar segments. Stable across labels so the
# same label keeps the same color across pages — useful when scanning a
# whole run for "missing tables" patterns.
_LABEL_COLORS: dict[str, str] = {
    "title": "#7c4dff",
    "section_header": "#3d5afe",
    "page_header": "#9e9e9e",
    "page_footer": "#9e9e9e",
    "subtitle_level_1": "#3d5afe",
    "subtitle_level_2": "#3d5afe",
    "paragraph": "#26a69a",
    "list_item": "#00897b",
    "caption": "#8e24aa",
    "footnote": "#9e9e9e",
    "formula": "#5e35b1",
    "code": "#1e88e5",
    "reference": "#558b2f",
    "__tables__": "#d81b60",
    "__pictures__": "#fb8c00",
}
_FALLBACK_COLOR = "#90a4ae"


class DocumentComparisonRenderer(ComparisonRenderer):
    name = "document"

    def render_html(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, DocumentComparisonResult):
            return _wrong_result(result)
        return _STANDALONE_TEMPLATE.format(
            title=escape(f"{prediction_label} ↔ {baseline_label}"),
            body=_render_doc_html(result, prediction_label, baseline_label),
            css=_DOC_CSS,
        )

    def render_gradio(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, DocumentComparisonResult):
            return _wrong_result(result)
        return _render_doc_html(result, prediction_label, baseline_label)

    def render_terminal(
        self, result, *, prediction_label, baseline_label, console: Console
    ) -> None:
        if not isinstance(result, DocumentComparisonResult):
            console.print(
                f"[yellow]DocumentRenderer can't render {type(result).__name__}[/]"
            )
            return
        console.print(
            f"\n[bold cyan]=== document comparison[/bold cyan]   "
            f"[red]── {prediction_label}[/red]   "
            f"[green]── {baseline_label}[/green]"
        )
        console.print(
            f"[dim]heading Δ {result.heading_count_delta:+d}  ·  "
            f"table Δ {result.table_count_delta:+d}  ·  "
            f"picture Δ {result.picture_count_delta:+d}[/dim]\n"
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("label")
        table.add_column(prediction_label, justify="right")
        table.add_column(baseline_label, justify="right")
        table.add_column("Δ", justify="right")
        labels = sorted(set(result.item_counts_pred) | set(result.item_counts_base))
        for label in labels:
            p = result.item_counts_pred.get(label, 0)
            b = result.item_counts_base.get(label, 0)
            delta = p - b
            delta_str = (
                f"[red]{delta:+d}[/red]" if delta < 0
                else f"[green]+{delta}[/green]" if delta > 0
                else "0"
            )
            table.add_row(_pretty_label(label), str(p), str(b), delta_str)
        console.print(table)


# --------------------------------------------------------------------- HTML


def _render_doc_html(
    result: DocumentComparisonResult,
    prediction_label: str,
    baseline_label: str,
) -> str:
    chips = _render_headline_chips(result)
    stacked = _render_stacked_bars(
        result.item_counts_pred,
        result.item_counts_base,
        prediction_label,
        baseline_label,
    )
    callouts = _render_missing_callouts(result)
    return (
        '<div class="ocrscout-doc-cmp">'
        + chips
        + stacked
        + callouts
        + "</div>"
    )


def _render_headline_chips(result: DocumentComparisonResult) -> str:
    chips: list[str] = []
    for label, pred_n, base_n, delta in [
        (
            "Headings",
            sum(
                n for lab, n in result.item_counts_pred.items()
                if lab in _HEADING_KEYS
            ),
            sum(
                n for lab, n in result.item_counts_base.items()
                if lab in _HEADING_KEYS
            ),
            result.heading_count_delta,
        ),
        (
            "Tables",
            result.item_counts_pred.get("__tables__", 0),
            result.item_counts_base.get("__tables__", 0),
            result.table_count_delta,
        ),
        (
            "Pictures",
            result.item_counts_pred.get("__pictures__", 0),
            result.item_counts_base.get("__pictures__", 0),
            result.picture_count_delta,
        ),
    ]:
        klass = (
            "delta-pos" if delta > 0
            else "delta-neg" if delta < 0
            else "delta-zero"
        )
        delta_str = f"Δ {delta:+d}" if delta != 0 else "Δ 0"
        chips.append(
            f'<div class="chip {klass}">'
            f'<span class="label">{escape(label)}</span>'
            f'<span class="vals">{pred_n}'
            f'<span class="arrow">→</span>{base_n}</span>'
            f'<span class="delta">{delta_str}</span>'
            "</div>"
        )
    return f'<div class="chip-row">{"".join(chips)}</div>'


def _render_stacked_bars(
    pred: dict[str, int],
    base: dict[str, int],
    pred_label: str,
    base_label: str,
) -> str:
    pred_total = sum(pred.values()) or 1
    base_total = sum(base.values()) or 1
    labels = sorted(set(pred) | set(base))

    def _bar(counts: dict[str, int], total: int) -> str:
        segs: list[str] = []
        for label in labels:
            n = counts.get(label, 0)
            if n == 0:
                continue
            pct = (n / total) * 100
            color = _LABEL_COLORS.get(label, _FALLBACK_COLOR)
            segs.append(
                f'<span class="seg" '
                f'style="width:{pct:.2f}%;background:{color};" '
                f'title="{escape(_pretty_label(label))}: {n} ({pct:.0f}%)"></span>'
            )
        return '<div class="stacked-bar">' + "".join(segs) + "</div>"

    return (
        '<div class="stacked-bar-row">'
        f'<span class="label">{escape(pred_label)}</span>'
        + _bar(pred, pred_total)
        + "</div>"
        '<div class="stacked-bar-row">'
        f'<span class="label">{escape(base_label)}</span>'
        + _bar(base, base_total)
        + "</div>"
    )


def _render_missing_callouts(result: DocumentComparisonResult) -> str:
    """Surface big asymmetries — when one side has 0 of something the other
    has 3+ of, the user wants that called out, not buried in a table."""
    callouts: list[str] = []
    for label, pretty in (
        ("__tables__", "tables"),
        ("__pictures__", "pictures"),
    ):
        p = result.item_counts_pred.get(label, 0)
        b = result.item_counts_base.get(label, 0)
        if p == 0 and b >= 3:
            callouts.append(
                f'<div class="callout">Prediction has no {pretty}; '
                f"baseline has {b}.</div>"
            )
        elif b == 0 and p >= 3:
            callouts.append(
                f'<div class="callout">Baseline has no {pretty}; '
                f"prediction has {p}.</div>"
            )
    if abs(result.heading_count_delta) >= 5:
        sign = "more" if result.heading_count_delta > 0 else "fewer"
        callouts.append(
            '<div class="callout">'
            f"Prediction has {abs(result.heading_count_delta)} {sign} headings "
            "than the baseline."
            "</div>"
        )
    return "".join(callouts)


_HEADING_KEYS: frozenset[str] = frozenset({
    "title",
    "section_header",
    "subtitle_level_1",
    "subtitle_level_2",
    "subtitle_level_3",
    "subtitle_level_4",
    "subtitle_level_5",
    "page_header",
})


def _pretty_label(label: str) -> str:
    if label == "__tables__":
        return "tables"
    if label == "__pictures__":
        return "pictures"
    return label


def _wrong_result(result: ComparisonResult) -> str:
    return (
        f"<p>DocumentComparisonRenderer cannot render "
        f"<code>{escape(type(result).__name__)}</code>.</p>"
    )


_STANDALONE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>{body}</body>
</html>
"""

# Embedded copy mirrors the .ocrscout-doc-cmp rules in viewer.css.
_DOC_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: 1rem; color: #1f2328; background: #f6f8fa;
}
.ocrscout-doc-cmp { display: flex; flex-direction: column; gap: 0.6rem; max-width: 960px; }
.ocrscout-doc-cmp .chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ocrscout-doc-cmp .chip {
  padding: 0.4rem 0.6rem;
  border-radius: 6px;
  background: #fff;
  border: 1px solid #d0d7de;
  display: flex; flex-direction: column;
  min-width: 110px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.8rem;
}
.ocrscout-doc-cmp .chip .label {
  font-weight: 600; color: #59636e; text-transform: capitalize;
  margin-bottom: 0.15rem;
}
.ocrscout-doc-cmp .chip .vals { font-size: 0.85rem; }
.ocrscout-doc-cmp .chip .vals .arrow { margin: 0 0.3rem; color: #59636e; }
.ocrscout-doc-cmp .chip .delta { margin-top: 0.2rem; font-size: 0.75rem; font-weight: 600; }
.ocrscout-doc-cmp .chip.delta-pos { border-left: 3px solid #1a7f37; }
.ocrscout-doc-cmp .chip.delta-pos .delta { color: #1a7f37; }
.ocrscout-doc-cmp .chip.delta-neg { border-left: 3px solid #cf222e; }
.ocrscout-doc-cmp .chip.delta-neg .delta { color: #cf222e; }
.ocrscout-doc-cmp .chip.delta-zero { border-left: 3px solid #d0d7de; }
.ocrscout-doc-cmp .chip.delta-zero .delta { color: #6c757d; }
.ocrscout-doc-cmp .stacked-bar {
  display: flex; height: 18px;
  border-radius: 4px; overflow: hidden;
  border: 1px solid #d0d7de;
}
.ocrscout-doc-cmp .stacked-bar .seg { display: block; height: 100%; }
.ocrscout-doc-cmp .stacked-bar-row {
  display: grid; grid-template-columns: 8rem 1fr;
  gap: 0.5rem; align-items: center;
  font-size: 0.8rem; padding: 0.15rem 0;
}
.ocrscout-doc-cmp .stacked-bar-row .label { color: #59636e; text-align: right; }
.ocrscout-doc-cmp .callout {
  padding: 0.4rem 0.6rem;
  border-left: 3px solid #d29922;
  background: rgba(210, 153, 34, 0.10);
  border-radius: 0 4px 4px 0;
  font-size: 0.85rem;
}
"""

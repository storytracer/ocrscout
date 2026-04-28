"""Renderer for ``LayoutComparisonResult`` — IoU bars + region legend.

Replaces the bare table with a more scannable layout:

* A summary legend (matched / missing / extra) so the user can read the
  comparison at a glance.
* Per-category horizontal IoU bars (0..1, gradient red→amber→green) so
  it's obvious which categories agree and which don't.

The viewer's Compare mode swaps the page-image AnnotatedImage overlay
to a comparison overlay (matched green, missing red, hallucinated yellow)
in parallel with this renderer — the image lives in the existing image
column, not in the comparison panel itself, so the comparison panel
stays compact.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from rich.table import Table

from ocrscout.comparisons.layout import LayoutComparisonResult
from ocrscout.interfaces.comparison import ComparisonRenderer, ComparisonResult

if TYPE_CHECKING:
    from rich.console import Console


class LayoutComparisonRenderer(ComparisonRenderer):
    name = "layout"

    def render_html(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, LayoutComparisonResult):
            return _wrong_result(result)
        return _STANDALONE_TEMPLATE.format(
            title=escape(f"{prediction_label} ↔ {baseline_label}"),
            body=_render_layout_html(result, prediction_label, baseline_label),
            css=_LAYOUT_CSS,
        )

    def render_gradio(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, LayoutComparisonResult):
            return _wrong_result(result)
        return _render_layout_html(result, prediction_label, baseline_label)

    def render_terminal(
        self, result, *, prediction_label, baseline_label, console: Console
    ) -> None:
        if not isinstance(result, LayoutComparisonResult):
            console.print(
                f"[yellow]LayoutRenderer can't render {type(result).__name__}[/]"
            )
            return
        console.print(
            f"\n[bold cyan]=== layout comparison[/bold cyan]   "
            f"[red]── {prediction_label}[/red]   "
            f"[green]── {baseline_label}[/green]"
        )
        console.print(
            f"[dim]iou mean {result.iou_mean:.3f}  ·  "
            f"matched {result.matched_regions}  ·  "
            f"missing {result.unmatched_base}  ·  "
            f"extra {result.unmatched_pred}[/dim]\n"
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("category")
        table.add_column("mean IoU", justify="right")
        for cat in sorted(result.iou_per_category):
            iou = result.iou_per_category[cat]
            table.add_row(cat, f"{iou:.3f}")
        console.print(table)


def _render_layout_html(
    result: LayoutComparisonResult,
    prediction_label: str,
    baseline_label: str,
) -> str:
    legend = (
        '<div class="legend">'
        f'<span class="chip matched"><span class="dot"></span>matched {result.matched_regions}</span>'
        f'<span class="chip missing"><span class="dot"></span>missing {result.unmatched_base}</span>'
        f'<span class="chip extra"><span class="dot"></span>extra {result.unmatched_pred}</span>'
        f'<span class="chip"><b>IoU mean {result.iou_mean:.3f}</b></span>'
        "</div>"
    )
    iou_rows: list[str] = []
    for cat in sorted(result.iou_per_category):
        iou = result.iou_per_category[cat]
        pct = max(0.0, min(1.0, iou)) * 100
        iou_rows.append(
            f'<div class="iou-row">'
            f'<span class="label">{escape(cat)}</span>'
            f'<span class="bar"><span style="width:{pct:.1f}%;"></span></span>'
            f'<span class="val">{iou:.3f}</span>'
            "</div>"
        )
    return (
        '<div class="ocrscout-layout-cmp">'
        + legend
        + '<div class="iou-list">'
        + "".join(iou_rows)
        + "</div>"
        + "</div>"
    )


def _wrong_result(result: ComparisonResult) -> str:
    return (
        f"<p>LayoutComparisonRenderer cannot render "
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

_LAYOUT_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: 1rem; color: #1f2328; background: #f6f8fa;
}
.ocrscout-layout-cmp { display: flex; flex-direction: column; gap: 0.5rem; max-width: 720px; }
.ocrscout-layout-cmp .legend {
  display: flex; gap: 0.6rem; font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 0.8rem; flex-wrap: wrap;
}
.ocrscout-layout-cmp .legend .chip {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.15rem 0.55rem; border-radius: 999px;
  background: #fff; border: 1px solid #d0d7de;
}
.ocrscout-layout-cmp .legend .chip .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; }
.ocrscout-layout-cmp .legend .matched .dot { background: #1a7f37; }
.ocrscout-layout-cmp .legend .missing .dot { background: #cf222e; }
.ocrscout-layout-cmp .legend .extra .dot { background: #d29922; }
.ocrscout-layout-cmp .iou-list { display: flex; flex-direction: column; gap: 0.25rem; }
.ocrscout-layout-cmp .iou-row {
  display: grid; grid-template-columns: 8rem 1fr 3rem;
  gap: 0.5rem; align-items: center;
  font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 0.8rem;
}
.ocrscout-layout-cmp .iou-row .label { color: #59636e; }
.ocrscout-layout-cmp .iou-row .bar {
  height: 8px; border-radius: 4px;
  background: #eaeef2; overflow: hidden;
}
.ocrscout-layout-cmp .iou-row .bar > span {
  display: block; height: 100%;
  background: linear-gradient(90deg, #cf222e 0%, #d29922 50%, #1a7f37 100%);
}
.ocrscout-layout-cmp .iou-row .val { text-align: right; font-variant-numeric: tabular-nums; }
"""

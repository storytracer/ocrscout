"""Renderer for ``LayoutComparisonResult`` — per-category IoU summary.

Today the renderer keeps things text-only: a Rich table for terminal and
a small HTML/Gradio table of category → mean IoU. A future enhancement
would overlay the matched / unmatched regions on the source image
(SVG-on-canvas), but that requires pulling the page image — which inspect
doesn't have at hand. Out of scope for this round.
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
        return _LAYOUT_HTML_TEMPLATE.format(
            prediction=escape(prediction_label),
            baseline=escape(baseline_label),
            iou_mean=f"{result.iou_mean:.3f}",
            matched=result.matched_regions,
            unmatched_pred=result.unmatched_pred,
            unmatched_base=result.unmatched_base,
            body=_render_per_cat_rows_html(result.iou_per_category),
        )

    def render_gradio(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, LayoutComparisonResult):
            return _wrong_result(result)
        body = _render_per_cat_rows_html(result.iou_per_category)
        return (
            f'<div class="ocrscout-layout-cmp">'
            f"<p>iou mean <b>{result.iou_mean:.3f}</b>  ·  "
            f"matched {result.matched_regions}  ·  "
            f"unmatched in {escape(prediction_label)} {result.unmatched_pred}  ·  "
            f"unmatched in {escape(baseline_label)} {result.unmatched_base}</p>"
            "<table class='ocrscout-layout-cmp-table'>"
            "<thead><tr><th>category</th><th>mean IoU</th></tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
            "</div>"
        )

    def render_terminal(
        self, result, *, prediction_label, baseline_label, console: Console
    ) -> None:
        if not isinstance(result, LayoutComparisonResult):
            console.print(f"[yellow]LayoutRenderer can't render {type(result).__name__}[/]")
            return
        console.print(
            f"\n[bold cyan]=== layout comparison[/bold cyan]   "
            f"[red]── {prediction_label}[/red]   "
            f"[green]── {baseline_label}[/green]"
        )
        console.print(
            f"[dim]iou mean {result.iou_mean:.3f}  ·  "
            f"matched {result.matched_regions}  ·  "
            f"unmatched in {prediction_label} {result.unmatched_pred}  ·  "
            f"unmatched in {baseline_label} {result.unmatched_base}[/dim]\n"
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("category")
        table.add_column("mean IoU", justify="right")
        for cat in sorted(result.iou_per_category):
            iou = result.iou_per_category[cat]
            table.add_row(cat, f"{iou:.3f}")
        console.print(table)


def _render_per_cat_rows_html(per_cat: dict[str, float]) -> str:
    rows: list[str] = []
    for cat in sorted(per_cat):
        rows.append(
            f"<tr><td>{escape(cat)}</td><td>{per_cat[cat]:.3f}</td></tr>"
        )
    return "\n".join(rows)


def _wrong_result(result: ComparisonResult) -> str:
    return (
        f"<p>LayoutComparisonRenderer cannot render "
        f"<code>{escape(type(result).__name__)}</code>.</p>"
    )


_LAYOUT_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ocrscout layout comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          padding: 1rem; color: #212529; background: #f8f9fa; }}
  header h1 {{ font-size: 1rem; }}
  header .stats {{ color: #6c757d; font-size: 0.9rem; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; }}
  th, td {{ padding: 0.4rem 1rem; border-bottom: 1px solid #dee2e6; }}
  th {{ text-align: left; }}
</style>
</head>
<body>
<header>
  <h1>layout comparison: {prediction} vs {baseline}</h1>
  <div class="stats">
    iou mean {iou_mean}  ·  matched {matched}  ·
    unmatched in {prediction} {unmatched_pred}  ·
    unmatched in {baseline} {unmatched_base}
  </div>
</header>
<table>
  <thead><tr><th>category</th><th>mean IoU</th></tr></thead>
  <tbody>
{body}
  </tbody>
</table>
</body>
</html>
"""

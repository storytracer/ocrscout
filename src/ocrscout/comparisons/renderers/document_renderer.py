"""Renderer for ``DocumentComparisonResult`` — side-by-side item-count table."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from rich.table import Table

from ocrscout.comparisons.document import DocumentComparisonResult
from ocrscout.interfaces.comparison import ComparisonRenderer, ComparisonResult

if TYPE_CHECKING:
    from rich.console import Console


class DocumentComparisonRenderer(ComparisonRenderer):
    name = "document"

    def render_html(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, DocumentComparisonResult):
            return _wrong_result(result)
        return _DOC_HTML_TEMPLATE.format(
            prediction=escape(prediction_label),
            baseline=escape(baseline_label),
            heading_delta=result.heading_count_delta,
            table_delta=result.table_count_delta,
            picture_delta=result.picture_count_delta,
            body=_render_count_table_html(
                result.item_counts_pred,
                result.item_counts_base,
            ),
        )

    def render_gradio(self, result, *, prediction_label, baseline_label) -> str:
        if not isinstance(result, DocumentComparisonResult):
            return _wrong_result(result)
        body = _render_count_table_html(
            result.item_counts_pred, result.item_counts_base
        )
        return (
            '<div class="ocrscout-doc-cmp">'
            "<table class='ocrscout-doc-cmp-table'>"
            "<thead><tr>"
            f"<th>label</th>"
            f"<th>{escape(prediction_label)}</th>"
            f"<th>{escape(baseline_label)}</th>"
            f"<th>Δ</th>"
            "</tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
            "</div>"
        )

    def render_terminal(
        self, result, *, prediction_label, baseline_label, console: Console
    ) -> None:
        if not isinstance(result, DocumentComparisonResult):
            console.print(f"[yellow]DocumentRenderer can't render {type(result).__name__}[/]")
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


def _pretty_label(label: str) -> str:
    if label == "__tables__":
        return "(tables)"
    if label == "__pictures__":
        return "(pictures)"
    return label


def _render_count_table_html(pred: dict[str, int], base: dict[str, int]) -> str:
    labels = sorted(set(pred) | set(base))
    rows: list[str] = []
    for label in labels:
        p = pred.get(label, 0)
        b = base.get(label, 0)
        delta = p - b
        delta_class = "neg" if delta < 0 else "pos" if delta > 0 else "neu"
        rows.append(
            f"<tr><td>{escape(_pretty_label(label))}</td>"
            f"<td>{p}</td><td>{b}</td>"
            f'<td class="{delta_class}">{delta:+d}</td></tr>'
        )
    return "\n".join(rows)


def _wrong_result(result: ComparisonResult) -> str:
    return (
        f"<p>DocumentComparisonRenderer cannot render "
        f"<code>{escape(type(result).__name__)}</code>.</p>"
    )


_DOC_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ocrscout document comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          padding: 1rem; color: #212529; background: #f8f9fa; }}
  header h1 {{ font-size: 1rem; }}
  header .stats {{ color: #6c757d; font-size: 0.9rem; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; }}
  th, td {{ padding: 0.4rem 1rem; border-bottom: 1px solid #dee2e6; }}
  th {{ text-align: left; }}
  td.neg {{ color: #b02a37; }}
  td.pos {{ color: #146c43; }}
  td.neu {{ color: #6c757d; }}
</style>
</head>
<body>
<header>
  <h1>document comparison: {prediction} vs {baseline}</h1>
  <div class="stats">
    heading Δ {heading_delta:+d}  ·  table Δ {table_delta:+d}  ·
    picture Δ {picture_delta:+d}
  </div>
</header>
<table>
  <thead><tr>
    <th>label</th><th>{prediction}</th><th>{baseline}</th><th>Δ</th>
  </tr></thead>
  <tbody>
{body}
  </tbody>
</table>
</body>
</html>
"""

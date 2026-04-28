"""Renderer for ``TextComparisonResult`` — the opcode-aligned word diff.

Replaces the rendering surface previously in ``viewer/diff.py``. The result
already carries ``opcodes`` + ``pred_tokens`` + ``base_tokens``, so the
renderer is pure formatting — no recomputation. The HTML output is the
same side-by-side opcode-aligned table that inspect's ``--html`` serves
and the viewer's Compare mode embeds; updating the styling here updates
both surfaces.
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
        body = _render_diff_rows(result)
        return _DIFF_HTML_TEMPLATE.format(
            page_id="",
            prediction=escape(prediction_label),
            baseline=escape(baseline_label),
            similarity=f"{result.similarity:.1f}",
            common=result.common,
            removed=result.removed,
            added=result.added,
            body=body,
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
        body = _render_diff_rows(result)
        return (
            '<div class="ocrscout-diff-scroll">'
            '<table class="ocrscout-diff-table">'
            "<thead><tr>"
            f'<th class="left">{escape(prediction_label)}</th>'
            f'<th class="right">{escape(baseline_label)}</th>'
            "</tr></thead>"
            f"<tbody>{body}</tbody>"
            "</table>"
            "</div>"
        )

    def render_terminal(
        self,
        result: ComparisonResult,
        *,
        prediction_label: str,
        baseline_label: str,
        console: Console,
    ) -> None:
        if not isinstance(result, TextComparisonResult):
            console.print(f"[yellow]TextRenderer can't render {type(result).__name__}[/]")
            return
        rendered = Text()
        for tag, i1, i2, j1, j2 in result.opcodes:
            if tag == "equal":
                for tok in result.pred_tokens[i1:i2]:
                    _emit_token(rendered, tok, style="")
            elif tag == "delete":
                for tok in result.pred_tokens[i1:i2]:
                    _emit_token(rendered, tok, style="red strike")
            elif tag == "insert":
                for tok in result.base_tokens[j1:j2]:
                    _emit_token(rendered, tok, style="green")
            elif tag == "replace":
                for tok in result.pred_tokens[i1:i2]:
                    _emit_token(rendered, tok, style="red strike")
                for tok in result.base_tokens[j1:j2]:
                    _emit_token(rendered, tok, style="green")

        console.print(
            f"\n[bold cyan]=== diff[/bold cyan]   "
            f"[red strike]── {prediction_label}[/red strike]   "
            f"[green]── {baseline_label}[/green]"
        )
        console.print(
            f"[dim]similarity {result.similarity:.1f}%  ·  "
            f"common {result.common} words  ·  "
            f"removed {result.removed}  ·  added {result.added}"
            + (f"  ·  cer {result.cer:.3f}" if result.cer is not None else "")
            + (f"  ·  wer {result.wer:.3f}" if result.wer is not None else "")
            + "[/dim]\n"
        )
        console.print(rendered)


def _emit_token(text_obj: Text, token: str, *, style: str) -> None:
    """Append a diff token to a Rich Text, preserving newlines and word spacing."""
    if "\n" in token:
        text_obj.append(token)
    else:
        text_obj.append(token + " ", style=style)


def _tokens_to_html(tokens: list[str]) -> str:
    """Join word tokens with single spaces, mapping newline runs to ``<br>``."""
    parts: list[str] = []
    for tok in tokens:
        if "\n" in tok:
            parts.append("<br>" * tok.count("\n"))
        else:
            parts.append(escape(tok) + " ")
    return "".join(parts).rstrip()


def _render_diff_rows(result: TextComparisonResult) -> str:
    rows_html: list[str] = []
    for tag, i1, i2, j1, j2 in result.opcodes:
        if tag == "equal":
            text = _tokens_to_html(result.pred_tokens[i1:i2])
            rows_html.append(
                f'<tr class="equal"><td class="left">{text}</td>'
                f'<td class="right">{text}</td></tr>'
            )
        elif tag == "delete":
            left = _tokens_to_html(result.pred_tokens[i1:i2])
            rows_html.append(
                f'<tr class="delete"><td class="left">{left}</td>'
                f'<td class="right"></td></tr>'
            )
        elif tag == "insert":
            right = _tokens_to_html(result.base_tokens[j1:j2])
            rows_html.append(
                f'<tr class="insert"><td class="left"></td>'
                f'<td class="right">{right}</td></tr>'
            )
        elif tag == "replace":
            left = _tokens_to_html(result.pred_tokens[i1:i2])
            right = _tokens_to_html(result.base_tokens[j1:j2])
            rows_html.append(
                f'<tr class="replace"><td class="left">{left}</td>'
                f'<td class="right">{right}</td></tr>'
            )
    return "\n".join(rows_html)


def _wrong_result_html(result: ComparisonResult) -> str:
    return (
        f"<p>TextComparisonRenderer cannot render "
        f"<code>{escape(type(result).__name__)}</code>.</p>"
    )


_DIFF_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ocrscout text comparison</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    line-height: 1.55;
    color: #212529;
    background: #f8f9fa;
  }}
  header {{
    position: sticky; top: 0; z-index: 2;
    background: #fff;
    border-bottom: 1px solid #dee2e6;
    padding: 0.75rem 1.25rem;
  }}
  header h1 {{
    font-size: 0.95rem; font-weight: 600; margin: 0 0 0.25rem;
    color: #495057;
  }}
  header .stats {{ font-size: 0.85rem; color: #6c757d; }}
  header .removed {{ color: #b02a37; }}
  header .added {{ color: #146c43; }}
  table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
  thead th {{
    position: sticky; top: 3.4rem; z-index: 1;
    background: #fff;
    text-align: left; padding: 0.5rem 1.25rem;
    border-bottom: 2px solid #dee2e6;
    font-weight: 600; font-size: 0.85rem;
    color: #495057;
  }}
  thead th.left {{ border-right: 1px solid #e9ecef; color: #b02a37; }}
  thead th.right {{ color: #146c43; }}
  td {{
    padding: 0.4rem 1.25rem;
    vertical-align: top;
    width: 50%;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  td.left {{ border-right: 1px solid #e9ecef; }}
  tr.equal td {{ color: #6c757d; }}
  tr.delete td.left, tr.replace td.left {{
    background: #fdecec; color: #842029;
  }}
  tr.insert td.right, tr.replace td.right {{
    background: #e6f4ea; color: #0a3622;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0d1117; color: #c9d1d9; }}
    header, thead th {{
      background: #161b22; border-color: #30363d; color: #c9d1d9;
    }}
    header h1, header .stats {{ color: #8b949e; }}
    td.left {{ border-right-color: #21262d; }}
    tr.equal td {{ color: #8b949e; }}
    tr.delete td.left, tr.replace td.left {{
      background: #4d1f24; color: #ffa198;
    }}
    tr.insert td.right, tr.replace td.right {{
      background: #1f3324; color: #7ee787;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>{page_id}</h1>
  <div class="stats">
    similarity {similarity}%  ·  common {common} words  ·
    <span class="removed">removed {removed}</span>  ·
    <span class="added">added {added}</span>
  </div>
</header>
<table>
  <thead><tr><th class="left">{prediction}</th><th class="right">{baseline}</th></tr></thead>
  <tbody>
{body}
  </tbody>
</table>
</body>
</html>
"""

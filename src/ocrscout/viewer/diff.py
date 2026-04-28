"""Reusable pairwise word-level diff renderer shared by `inspect` and `viewer`.

Built on ``difflib.SequenceMatcher``. The HTML output is the same
side-by-side opcode-aligned table the ``ocrscout inspect --diff a,b --html``
command serves and the viewer's Diff mode embeds, so the look stays
consistent across both call sites.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from html import escape

# Word-level diff tokenizer: \S+ captures a word, \n+ captures a paragraph
# break. Inline whitespace is dropped (we re-render single spaces between
# words), but newline runs survive as their own tokens so paragraphs stay
# visible in the diff output.
DIFF_TOKEN_RE = re.compile(r"\S+|\n+")


def tokenize(text: str) -> list[str]:
    """Word + paragraph-break tokenization for diff alignment."""
    return DIFF_TOKEN_RE.findall(text)


@dataclass(frozen=True)
class DiffStats:
    similarity: float  # 0..100
    common: int
    removed: int
    added: int


def compute_diff(
    text_a: str, text_b: str
) -> tuple[list[tuple[str, int, int, int, int]], list[str], list[str], DiffStats]:
    """Run a word-level diff and return opcodes, both token lists, and stats."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    # difflib types the tag as a Literal; widen to str so callers can iterate.
    opcodes: list[tuple[str, int, int, int, int]] = [
        (str(tag), i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes()
    ]
    similarity = matcher.ratio() * 100
    common = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal")
    removed = sum(
        i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("delete", "replace")
    )
    added = sum(
        j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("insert", "replace")
    )
    return opcodes, tokens_a, tokens_b, DiffStats(similarity, common, removed, added)


def tokens_to_html(tokens: list[str]) -> str:
    """Join word tokens with single spaces, mapping newline runs to ``<br>``."""
    parts: list[str] = []
    for tok in tokens:
        if "\n" in tok:
            parts.append("<br>" * tok.count("\n"))
        else:
            parts.append(escape(tok) + " ")
    return "".join(parts).rstrip()


def render_diff_rows(
    opcodes: list[tuple[str, int, int, int, int]],
    tokens_a: list[str],
    tokens_b: list[str],
) -> str:
    """Render diff opcodes as opcode-aligned ``<tr>`` rows.

    Each opcode becomes one ``<tr>``: ``equal`` rows show identical text on
    both sides; ``delete``/``insert`` rows highlight one side and leave the
    other empty; ``replace`` rows highlight both sides with their respective
    diverging text.
    """
    rows_html: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            text = tokens_to_html(tokens_a[i1:i2])
            rows_html.append(
                f'<tr class="equal"><td class="left">{text}</td>'
                f'<td class="right">{text}</td></tr>'
            )
        elif tag == "delete":
            left = tokens_to_html(tokens_a[i1:i2])
            rows_html.append(
                f'<tr class="delete"><td class="left">{left}</td>'
                f'<td class="right"></td></tr>'
            )
        elif tag == "insert":
            right = tokens_to_html(tokens_b[j1:j2])
            rows_html.append(
                f'<tr class="insert"><td class="left"></td>'
                f'<td class="right">{right}</td></tr>'
            )
        elif tag == "replace":
            left = tokens_to_html(tokens_a[i1:i2])
            right = tokens_to_html(tokens_b[j1:j2])
            rows_html.append(
                f'<tr class="replace"><td class="left">{left}</td>'
                f'<td class="right">{right}</td></tr>'
            )
    return "\n".join(rows_html)


def render_diff_page(
    text_a: str,
    text_b: str,
    *,
    page_id: str,
    model_a: str,
    model_b: str,
) -> str:
    """Build a fully self-contained side-by-side diff HTML document.

    Used by ``ocrscout inspect --diff a,b --html`` (which serves it via a
    one-shot HTTP server) and by the viewer's ``--html`` export action.
    """
    opcodes, tokens_a, tokens_b, stats = compute_diff(text_a, text_b)
    body = render_diff_rows(opcodes, tokens_a, tokens_b)
    return _DIFF_HTML_TEMPLATE.format(
        page_id=escape(page_id),
        model_a=escape(model_a),
        model_b=escape(model_b),
        similarity=f"{stats.similarity:.1f}",
        common=stats.common,
        removed=stats.removed,
        added=stats.added,
        body=body,
    )


def render_diff_table_fragment(
    text_a: str,
    text_b: str,
    *,
    model_a: str,
    model_b: str,
) -> tuple[str, DiffStats]:
    """Build just the diff table (not a full HTML page) for embedding in Gradio.

    Returns the HTML fragment plus the per-pair stats, so the caller can
    render its own header/legend in Gradio components.
    """
    opcodes, tokens_a, tokens_b, stats = compute_diff(text_a, text_b)
    body = render_diff_rows(opcodes, tokens_a, tokens_b)
    fragment = (
        '<div class="ocrscout-diff-scroll">'
        '<table class="ocrscout-diff-table">'
        '<thead><tr>'
        f'<th class="left">{escape(model_a)}</th>'
        f'<th class="right">{escape(model_b)}</th>'
        '</tr></thead>'
        f'<tbody>{body}</tbody>'
        '</table>'
        '</div>'
    )
    return fragment, stats


_DIFF_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ocrscout diff: {page_id}</title>
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
  <thead><tr><th class="left">{model_a}</th><th class="right">{model_b}</th></tr></thead>
  <tbody>
{body}
  </tbody>
</table>
</body>
</html>
"""

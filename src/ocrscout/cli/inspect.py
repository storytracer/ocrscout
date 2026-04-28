"""`ocrscout inspect` — side-by-side comparison of a previous run's output."""

from __future__ import annotations

import difflib
import http.server
import json
import re
import socket
import threading
import webbrowser
from html import escape
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ocrscout.cli import app

# Word-level diff tokenizer: \S+ captures a word, \n+ captures a paragraph
# break. Inline whitespace is dropped (we re-render single spaces between
# words), but newline runs survive as their own tokens so paragraphs stay
# visible in the diff output.
_DIFF_TOKEN_RE = re.compile(r"\S+|\n+")


@app.command("inspect")
def inspect(
    output_dir: Path = typer.Argument(
        ..., help="A previous run's --output-dir (must contain results.parquet)."
    ),
    page: str | None = typer.Option(
        None, "--page", "-p",
        help="Show the full per-model markdown for one page_id instead of "
             "the summary table.",
    ),
    diff: str | None = typer.Option(
        None, "--diff",
        help="Show an inline word-level diff between two models for the page "
             "given by --page. Format: 'modelA,modelB'. Words only in modelA "
             "render red strikethrough; words only in modelB render green; "
             "common words render in default text color. Requires --page.",
    ),
    html: bool = typer.Option(
        False, "--html",
        help="With --diff, serve a side-by-side HTML diff via a one-shot local "
             "HTTP server and open it in your default browser. No files are "
             "written. Press Ctrl-C to stop the server.",
    ),
    snippet_length: int = typer.Option(
        80, "--snippet-length",
        help="Characters of text preview shown in the summary table.",
    ),
) -> None:
    """Read a previous run's results.parquet and print a per-page comparison."""
    parquet_path = output_dir / "results.parquet"
    if not parquet_path.is_file():
        rprint(f"[red]No results.parquet at {parquet_path}[/red]")
        raise typer.Exit(code=1)

    rows = _load_rows(parquet_path)
    if not rows:
        rprint(f"[yellow]{parquet_path} is empty.[/yellow]")
        return

    if diff is not None:
        if page is None:
            rprint("[red]--diff requires --page <page_id>[/red]")
            raise typer.Exit(code=1)
        parts = [s.strip() for s in diff.split(",") if s.strip()]
        if len(parts) != 2:
            rprint(
                f"[red]--diff expects exactly two comma-separated model names; "
                f"got {parts!r}[/red]"
            )
            raise typer.Exit(code=1)
        if parts[0] == parts[1]:
            rprint(f"[red]--diff: model A and model B are the same ({parts[0]!r}); "
                   f"nothing to compare[/red]")
            raise typer.Exit(code=1)
        _show_page_diff(
            rows, page_id=page, model_a=parts[0], model_b=parts[1],
            output_dir=output_dir, html=html,
        )
    elif html:
        rprint("[red]--html requires --diff (it serves a diff viewer)[/red]")
        raise typer.Exit(code=1)
    elif page is not None:
        _show_page(rows, page_id=page, output_dir=output_dir)
    else:
        _show_summary(rows, snippet_length=snippet_length)


def _load_rows(parquet_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path)
    out: list[dict[str, Any]] = []
    for i in range(table.num_rows):
        metrics_raw = table["metrics_json"][i].as_py()
        metrics = json.loads(metrics_raw) if metrics_raw else {}
        out.append({
            "page_id": table["page_id"][i].as_py(),
            "source_uri": table["source_uri"][i].as_py(),
            "output_format": table["output_format"][i].as_py(),
            "document_json": table["document_json"][i].as_py(),
            "error": table["error"][i].as_py(),
            "model": metrics.get("model", "?"),
            "metrics": metrics,
        })
    return out


def _show_summary(rows: list[dict[str, Any]], *, snippet_length: int) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["page_id"], r["model"]))

    table = Table(title="ocrscout inspect — per-(page, model) summary")
    table.add_column("page_id", style="bold")
    table.add_column("model")
    table.add_column("items", justify="right")
    table.add_column("chars", justify="right")
    table.add_column("s/page", justify="right")
    table.add_column("snippet", overflow="fold")

    last_page: str | None = None
    for r in rows_sorted:
        m = r["metrics"]
        items = _fmt_int(m.get("item_count"))
        chars = _fmt_int(m.get("text_length"))
        s_per_page = _fmt_seconds(m.get("run_seconds_per_page"))
        snippet = _snippet_from_doc(r["document_json"], snippet_length)

        # Visually group rows by page: only show page_id on the first row of each group.
        page_label = r["page_id"] if r["page_id"] != last_page else ""
        last_page = r["page_id"]
        table.add_row(page_label, r["model"], items, chars, s_per_page, snippet)

    Console().print(table)


def _show_page(
    rows: list[dict[str, Any]], *, page_id: str, output_dir: Path
) -> None:
    matches = [r for r in rows if r["page_id"] == page_id]
    if not matches:
        all_pages = sorted({r["page_id"] for r in rows})
        rprint(f"[red]No rows for page_id={page_id!r}.[/red]")
        rprint(f"[dim]Available: {all_pages}[/dim]")
        raise typer.Exit(code=1)

    matches.sort(key=lambda r: r["model"])
    text_dir = output_dir / "text"
    for r in matches:
        rprint(
            f"\n[bold cyan]=== {r['page_id']}  ·  {r['model']} "
            f"({r['output_format']}) ===[/bold cyan]"
        )
        m = r["metrics"]
        rprint(
            f"[dim]items={m.get('item_count')}  chars={m.get('text_length')}  "
            f"s/page={_fmt_seconds(m.get('run_seconds_per_page'))}[/dim]"
        )
        markdown = _markdown_for(r, text_dir=text_dir)
        if markdown:
            rprint(markdown)
        else:
            rprint("[yellow](no rendered text available)[/yellow]")


def _show_page_diff(
    rows: list[dict[str, Any]],
    *,
    page_id: str,
    model_a: str,
    model_b: str,
    output_dir: Path,
    html: bool,
) -> None:
    """Compute a word-level diff between two models; render to terminal or HTML."""
    page_rows = {r["model"]: r for r in rows if r["page_id"] == page_id}
    if not page_rows:
        all_pages = sorted({r["page_id"] for r in rows})
        rprint(f"[red]No rows for page_id={page_id!r}.[/red]")
        rprint(f"[dim]Available: {all_pages}[/dim]")
        raise typer.Exit(code=1)
    missing = [m for m in (model_a, model_b) if m not in page_rows]
    if missing:
        available = sorted(page_rows)
        rprint(
            f"[red]No row for model(s) {missing!r} on page {page_id!r}.[/red]"
        )
        rprint(f"[dim]Available for this page: {available}[/dim]")
        raise typer.Exit(code=1)

    text_dir = output_dir / "text"
    text_a = _markdown_for(page_rows[model_a], text_dir=text_dir)
    text_b = _markdown_for(page_rows[model_b], text_dir=text_dir)
    if not text_a or not text_b:
        rprint(
            "[yellow]Cannot diff: one or both models produced no rendered text.[/yellow]"
        )
        raise typer.Exit(code=1)

    tokens_a = _DIFF_TOKEN_RE.findall(text_a)
    tokens_b = _DIFF_TOKEN_RE.findall(text_b)
    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
    opcodes = matcher.get_opcodes()
    similarity = matcher.ratio() * 100
    common = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal")
    removed = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("delete", "replace"))
    added = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("insert", "replace"))

    if html:
        body = _render_diff_html(
            opcodes, tokens_a, tokens_b,
            page_id=page_id, model_a=model_a, model_b=model_b,
            similarity=similarity, common=common, removed=removed, added=added,
        )
        _serve_diff_html(body, page_id=page_id, model_a=model_a, model_b=model_b)
        return

    _render_diff_terminal(
        opcodes, tokens_a, tokens_b,
        page_id=page_id, model_a=model_a, model_b=model_b,
        similarity=similarity, common=common, removed=removed, added=added,
    )


def _render_diff_terminal(
    opcodes: list[tuple[str, int, int, int, int]],
    tokens_a: list[str],
    tokens_b: list[str],
    *,
    page_id: str,
    model_a: str,
    model_b: str,
    similarity: float,
    common: int,
    removed: int,
    added: int,
) -> None:
    rendered = Text()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for tok in tokens_a[i1:i2]:
                _emit_diff_token(rendered, tok, style="")
        elif tag == "delete":
            for tok in tokens_a[i1:i2]:
                _emit_diff_token(rendered, tok, style="red strike")
        elif tag == "insert":
            for tok in tokens_b[j1:j2]:
                _emit_diff_token(rendered, tok, style="green")
        elif tag == "replace":
            for tok in tokens_a[i1:i2]:
                _emit_diff_token(rendered, tok, style="red strike")
            for tok in tokens_b[j1:j2]:
                _emit_diff_token(rendered, tok, style="green")

    console = Console()
    console.print(
        f"\n[bold cyan]=== {page_id}  ·  diff[/bold cyan]   "
        f"[red strike]── {model_a}[/red strike]   [green]── {model_b}[/green]"
    )
    console.print(
        f"[dim]similarity {similarity:.1f}%  ·  common {common} words  ·  "
        f"removed {removed}  ·  added {added}[/dim]\n"
    )
    console.print(rendered)


def _emit_diff_token(text_obj: Text, token: str, *, style: str) -> None:
    """Append a diff token to a Rich Text, preserving newlines and word spacing."""
    if "\n" in token:
        # Paragraph break — emit unstyled so it isn't visually colored.
        text_obj.append(token)
    else:
        text_obj.append(token + " ", style=style)


def _render_diff_html(
    opcodes: list[tuple[str, int, int, int, int]],
    tokens_a: list[str],
    tokens_b: list[str],
    *,
    page_id: str,
    model_a: str,
    model_b: str,
    similarity: float,
    common: int,
    removed: int,
    added: int,
) -> str:
    """Build a self-contained HTML page with a side-by-side diff table.

    Each opcode becomes one ``<tr>``: ``equal`` rows show identical text on
    both sides; ``delete``/``insert`` rows highlight one side and leave the
    other empty; ``replace`` rows highlight both sides with their respective
    diverging text. Cells are aligned by row, so reading either column
    top-to-bottom yields that model's full transcription with disagreements
    visually marked.
    """
    rows_html: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            text = _tokens_to_html(tokens_a[i1:i2])
            rows_html.append(
                f'<tr class="equal"><td class="left">{text}</td>'
                f'<td class="right">{text}</td></tr>'
            )
        elif tag == "delete":
            left = _tokens_to_html(tokens_a[i1:i2])
            rows_html.append(
                f'<tr class="delete"><td class="left">{left}</td>'
                f'<td class="right"></td></tr>'
            )
        elif tag == "insert":
            right = _tokens_to_html(tokens_b[j1:j2])
            rows_html.append(
                f'<tr class="insert"><td class="left"></td>'
                f'<td class="right">{right}</td></tr>'
            )
        elif tag == "replace":
            left = _tokens_to_html(tokens_a[i1:i2])
            right = _tokens_to_html(tokens_b[j1:j2])
            rows_html.append(
                f'<tr class="replace"><td class="left">{left}</td>'
                f'<td class="right">{right}</td></tr>'
            )
    return _DIFF_HTML_TEMPLATE.format(
        page_id=escape(page_id),
        model_a=escape(model_a),
        model_b=escape(model_b),
        similarity=f"{similarity:.1f}",
        common=common,
        removed=removed,
        added=added,
        body="\n".join(rows_html),
    )


def _tokens_to_html(tokens: list[str]) -> str:
    """Join word tokens with single spaces, mapping newline runs to ``<br>``."""
    parts: list[str] = []
    for tok in tokens:
        if "\n" in tok:
            parts.append("<br>" * tok.count("\n"))
        else:
            parts.append(escape(tok) + " ")
    return "".join(parts).rstrip()


def _serve_diff_html(
    html: str, *, page_id: str, model_a: str, model_b: str
) -> None:
    """Serve ``html`` from a one-shot localhost HTTP server, open in browser."""
    body = html.encode("utf-8")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/diff", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass  # silence default request logging — Rich prints what we want

    # Bind to 0.0.0.0 so the diff is reachable from other machines on the
    # same LAN — useful when ocrscout is running on a headless GPU box and
    # you want to view the diff in your laptop's browser. The server has
    # no auth and accepts any GET; treat your network as trusted.
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
    port = server.server_address[1]
    local_url = f"http://127.0.0.1:{port}/"
    lan_ip = _detect_lan_ip()
    lan_url = f"http://{lan_ip}:{port}/" if lan_ip else None

    rprint(
        f"\n[bold cyan]Diff: {page_id}[/bold cyan]   "
        f"[red strike]── {model_a}[/red strike]   [green]── {model_b}[/green]"
    )
    if lan_url:
        rprint(f"[cyan]Serving on LAN at  {lan_url}[/cyan]")
        rprint(f"[cyan]      …and locally at  {local_url}[/cyan]")
    else:
        rprint(f"[cyan]Serving at {local_url}[/cyan]")
    rprint("[dim]Opening localhost URL in your default browser. "
           "Press Ctrl-C to stop.[/dim]")

    threading.Thread(target=lambda: webbrowser.open(local_url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        rprint("\n[dim]Diff server stopped.[/dim]")
    finally:
        server.server_close()


def _detect_lan_ip() -> str | None:
    """Best-effort LAN-reachable IP for the host running this server.

    Uses the standard "open a UDP socket to a public IP and inspect the
    local endpoint" trick — no packets are actually sent; we just ask the
    kernel which interface would route to the destination, then read back
    its address. Returns ``None`` if no usable interface can be detected
    (e.g. fully offline host).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        try:
            ip = socket.gethostbyname(socket.gethostname())
            # Skip uninformative loopback fallbacks; caller will say "localhost".
            return ip if ip and not ip.startswith("127.") else None
        except OSError:
            return None


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


def _markdown_for(row: dict[str, Any], *, text_dir: Path) -> str:
    """Prefer the on-disk markdown sidecar; fall back to re-rendering."""
    stem = Path(row["page_id"]).stem.replace("/", "_").replace("\\", "_")
    sidecar = text_dir / f"{stem}.{row['model']}.md"
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8")
    doc_json = row["document_json"]
    if not doc_json:
        return ""
    try:
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument.model_validate_json(doc_json)
        return doc.export_to_markdown()
    except Exception:  # noqa: BLE001
        return ""


def _snippet_from_doc(document_json: str | None, length: int) -> str:
    """Extract a short single-line preview from the serialized DoclingDocument."""
    if not document_json:
        return "(empty)"
    try:
        data = json.loads(document_json)
    except json.JSONDecodeError:
        return "(invalid JSON)"
    pieces: list[str] = []
    for t in data.get("texts") or []:
        text = (t.get("text") or "").strip()
        if text:
            pieces.append(text)
        if sum(len(p) for p in pieces) >= length:
            break
    joined = " · ".join(pieces).replace("\n", " ")
    if len(joined) > length:
        return joined[: length - 1] + "…"
    return joined or "(no text items)"


def _fmt_int(value: Any) -> str:
    return str(value) if isinstance(value, int) else "—"


def _fmt_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.2f}"

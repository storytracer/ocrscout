"""`ocrscout inspect` — side-by-side comparison of a previous run's output."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ocrscout.cli import app
from ocrscout.viewer.diff import compute_diff, render_diff_page


@app.command("inspect")
def inspect(
    output_dir: Path = typer.Argument(
        ..., help="A previous run's --output-dir (must contain data/train-*.parquet)."
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
    """Read a previous run's parquet shards and print a per-page comparison."""
    from ocrscout.exports.layout import find_parquet_files

    if not find_parquet_files(output_dir):
        rprint(f"[red]No data/train-*.parquet under {output_dir}[/red]")
        raise typer.Exit(code=1)

    rows = _load_rows(output_dir)
    if not rows:
        rprint(f"[yellow]{output_dir} contains no rows.[/yellow]")
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
            html=html,
        )
    elif html:
        rprint("[red]--html requires --diff (it serves a diff viewer)[/red]")
        raise typer.Exit(code=1)
    elif page is not None:
        _show_page(rows, page_id=page)
    else:
        _show_summary(rows, snippet_length=snippet_length)


def _load_rows(output_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    from ocrscout.exports.layout import parquet_data_files

    ds = load_dataset(
        "parquet",
        data_files=parquet_data_files(output_dir),
        split="train",
    )
    out: list[dict[str, Any]] = []
    for raw in ds:
        metrics_raw = raw.get("metrics_json") or ""
        try:
            metrics = json.loads(metrics_raw) if metrics_raw else {}
        except json.JSONDecodeError:
            metrics = {}
        out.append({
            "page_id": raw["page_id"],
            "model": raw["model"],
            "source_uri": raw.get("source_uri"),
            "output_format": raw.get("output_format"),
            "document_json": raw.get("document_json"),
            "markdown": raw.get("markdown"),
            "error": raw.get("error"),
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


def _show_page(rows: list[dict[str, Any]], *, page_id: str) -> None:
    matches = [r for r in rows if r["page_id"] == page_id]
    if not matches:
        all_pages = sorted({r["page_id"] for r in rows})
        rprint(f"[red]No rows for page_id={page_id!r}.[/red]")
        rprint(f"[dim]Available: {all_pages}[/dim]")
        raise typer.Exit(code=1)

    matches.sort(key=lambda r: r["model"])
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
        markdown = r.get("markdown") or ""
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

    text_a = page_rows[model_a].get("markdown") or ""
    text_b = page_rows[model_b].get("markdown") or ""
    if not text_a or not text_b:
        rprint(
            "[yellow]Cannot diff: one or both models produced no rendered text.[/yellow]"
        )
        raise typer.Exit(code=1)

    if html:
        body = render_diff_page(
            text_a, text_b,
            page_id=page_id, model_a=model_a, model_b=model_b,
        )
        _serve_diff_html(body, page_id=page_id, model_a=model_a, model_b=model_b)
        return

    opcodes, tokens_a, tokens_b, stats = compute_diff(text_a, text_b)
    _render_diff_terminal(
        opcodes, tokens_a, tokens_b,
        page_id=page_id, model_a=model_a, model_b=model_b,
        similarity=stats.similarity,
        common=stats.common,
        removed=stats.removed,
        added=stats.added,
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

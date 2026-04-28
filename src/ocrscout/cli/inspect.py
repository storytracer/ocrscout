"""`ocrscout inspect` — read a previous run's output and surface comparisons.

The summary view lists per-(page, model) stats with comparison-summary
columns lifted from the parquet's flat metric columns (``text_similarity``,
``layout_iou_mean``, etc.). The page view dumps each model's markdown and,
when present, the page's reference text alongside its provenance.

The compare view dispatches through the comparison-renderer registry, so
its output stays in lockstep with the viewer's Compare mode.
"""

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

from ocrscout import registry
from ocrscout.cli import app
from ocrscout.interfaces.comparison import (
    BaselineView,
    ComparisonResult,
    PredictionView,
)
from ocrscout.types import ReferenceProvenance

REFERENCE_PSEUDO_MODEL = "reference"


@app.command("inspect")
def inspect(
    output_dir: Path = typer.Argument(
        ..., help="A previous run's --output-dir (must contain data/train-*.parquet)."
    ),
    page: str | None = typer.Option(
        None, "--page", "-p",
        help="Show the full per-model markdown for one file_id (or page_id, "
             "for older runs) instead of the summary table.",
    ),
    compare: str | None = typer.Option(
        None, "--compare",
        help="Run a comparison between two artifacts for --page. Format "
             "'A,B'. Each side can be a model name or the literal "
             "`reference` to pull the row's reference_text + provenance. "
             "Comparison type defaults to `text`; override with "
             "--comparison-type.",
    ),
    comparison_type: str = typer.Option(
        "text", "--comparison-type",
        help="Comparison name from the registry (`text`, `document`, "
             "`layout`). Each is rendered via the matching "
             "ComparisonRenderer.",
    ),
    html: bool = typer.Option(
        False, "--html",
        help="With --compare, serve the rendered comparison as a self-"
             "contained HTML page from a one-shot local HTTP server "
             "and open it in your default browser. No files are "
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

    if compare is not None:
        if page is None:
            rprint("[red]--compare requires --page <file_id>[/red]")
            raise typer.Exit(code=1)
        parts = [s.strip() for s in compare.split(",") if s.strip()]
        if len(parts) != 2:
            rprint(
                f"[red]--compare expects exactly two comma-separated names; "
                f"got {parts!r}[/red]"
            )
            raise typer.Exit(code=1)
        if parts[0] == parts[1]:
            rprint(
                f"[red]--compare: side A and side B are the same "
                f"({parts[0]!r}); nothing to compare[/red]"
            )
            raise typer.Exit(code=1)
        _show_page_compare(
            rows,
            page_id=page,
            label_a=parts[0],
            label_b=parts[1],
            comparison_type=comparison_type,
            html=html,
        )
    elif html:
        rprint("[red]--html requires --compare (it serves a comparison viewer)[/red]")
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
        # Older parquets predate the file_id column — fall back to page_id
        # so existing runs still inspect cleanly.
        file_id = raw.get("file_id") or raw["page_id"]
        out.append({
            "file_id": file_id,
            "page_id": raw["page_id"],
            "model": raw["model"],
            "source_uri": raw.get("source_uri"),
            "output_format": raw.get("output_format"),
            "document_json": raw.get("document_json"),
            "markdown": raw.get("markdown"),
            "text": raw.get("text"),
            "reference_text": raw.get("reference_text"),
            "reference_provenance_json": raw.get("reference_provenance_json"),
            "comparisons_json": raw.get("comparisons_json"),
            "error": raw.get("error"),
            "metrics": metrics,
            # Flat metric columns — pulled out so the summary table can render
            # without parsing comparisons_json on every row.
            "text_similarity": raw.get("text_similarity"),
            "text_cer": raw.get("text_cer"),
            "text_wer": raw.get("text_wer"),
            "document_heading_count_delta": raw.get("document_heading_count_delta"),
            "document_table_count_delta": raw.get("document_table_count_delta"),
            "document_picture_count_delta": raw.get("document_picture_count_delta"),
            "layout_iou_mean": raw.get("layout_iou_mean"),
        })
    return out


# Map flat-column name -> short header rendered in the summary table.
_FLAT_COLUMN_HEADERS: dict[str, str] = {
    "text_similarity": "text sim",
    "text_cer": "cer",
    "text_wer": "wer",
    "document_heading_count_delta": "Δ headings",
    "document_table_count_delta": "Δ tables",
    "document_picture_count_delta": "Δ pictures",
    "layout_iou_mean": "iou mean",
}


def _show_summary(rows: list[dict[str, Any]], *, snippet_length: int) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["file_id"], r["model"]))

    # Only show comparison columns that have at least one non-null value.
    active_flat = [
        col for col in _FLAT_COLUMN_HEADERS
        if any(r.get(col) is not None for r in rows_sorted)
    ]

    table = Table(title="ocrscout inspect — per-(page, model) summary")
    table.add_column("file_id", style="bold")
    table.add_column("model")
    table.add_column("items", justify="right")
    table.add_column("chars", justify="right")
    table.add_column("s/page", justify="right")
    for col in active_flat:
        table.add_column(_FLAT_COLUMN_HEADERS[col], justify="right")
    table.add_column("snippet", overflow="fold")

    last_file: str | None = None
    for r in rows_sorted:
        m = r["metrics"]
        items = _fmt_int(m.get("item_count"))
        chars = _fmt_int(m.get("text_length"))
        s_per_page = _fmt_seconds(m.get("run_seconds_per_page"))
        snippet = _snippet_from_doc(r["document_json"], snippet_length)

        file_label = r["file_id"] if r["file_id"] != last_file else ""
        last_file = r["file_id"]
        cells = [file_label, r["model"], items, chars, s_per_page]
        for col in active_flat:
            cells.append(_fmt_metric(col, r.get(col)))
        cells.append(snippet)
        table.add_row(*cells)

    Console().print(table)


def _resolve_id(rows: list[dict[str, Any]], requested: str) -> str | None:
    """Match the user-supplied ``--page`` value against file_id first
    (the new canonical identifier), then page_id as a fallback for older
    runs / source-side debugging. Returns the file_id of the first match,
    or ``None`` when nothing matches."""
    for r in rows:
        if r["file_id"] == requested:
            return r["file_id"]
    for r in rows:
        if r["page_id"] == requested:
            return r["file_id"]
    return None


def _show_page(rows: list[dict[str, Any]], *, page_id: str) -> None:
    file_id = _resolve_id(rows, page_id)
    if file_id is None:
        all_files = sorted({r["file_id"] for r in rows})
        rprint(f"[red]No rows for {page_id!r}.[/red]")
        rprint(f"[dim]Available file_ids: {all_files}[/dim]")
        raise typer.Exit(code=1)
    matches = [r for r in rows if r["file_id"] == file_id]

    matches.sort(key=lambda r: r["model"])
    for r in matches:
        rprint(
            f"\n[bold cyan]=== {r['file_id']}  ·  {r['model']} "
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

    reference = next(
        (r.get("reference_text") for r in matches if r.get("reference_text")),
        None,
    )
    if reference:
        provenance = _parse_provenance(
            next(
                (r.get("reference_provenance_json") for r in matches if r.get("reference_provenance_json")),
                None,
            )
        )
        prov_label = _format_provenance(provenance)
        rprint(
            f"\n[bold magenta]=== {file_id}  ·  reference "
            f"({len(reference)} chars{prov_label}) ===[/bold magenta]"
        )
        rprint(reference)


def _show_page_compare(
    rows: list[dict[str, Any]],
    *,
    page_id: str,
    label_a: str,
    label_b: str,
    comparison_type: str,
    html: bool,
) -> None:
    """Run a Comparison between the two named artifacts and dispatch render."""
    file_id = _resolve_id(rows, page_id)
    if file_id is None:
        all_files = sorted({r["file_id"] for r in rows})
        rprint(f"[red]No rows for {page_id!r}.[/red]")
        rprint(f"[dim]Available file_ids: {all_files}[/dim]")
        raise typer.Exit(code=1)
    page_rows = {r["model"]: r for r in rows if r["file_id"] == file_id}
    page_id = file_id  # downstream uses this for display

    missing = [
        m for m in (label_a, label_b)
        if m != REFERENCE_PSEUDO_MODEL and m not in page_rows
    ]
    if missing:
        available = sorted(page_rows) + [REFERENCE_PSEUDO_MODEL]
        rprint(
            f"[red]No row for model(s) {missing!r} on page {page_id!r}.[/red]"
        )
        rprint(f"[dim]Available for this page: {available}[/dim]")
        raise typer.Exit(code=1)

    pred = _build_view(page_rows, label_a, kind="prediction")
    base = _build_view(page_rows, label_b, kind="baseline")
    if pred is None or base is None:
        rprint(
            f"[yellow]Cannot run {comparison_type} comparison on {page_id!r}: "
            f"missing data for one side.[/yellow]"
        )
        raise typer.Exit(code=1)

    try:
        cmp_cls = registry.get("comparisons", comparison_type)
    except Exception as e:
        rprint(
            f"[red]Unknown --comparison-type {comparison_type!r}; "
            f"available: {registry.list('comparisons')}[/red]"
        )
        raise typer.Exit(code=1) from e
    try:
        renderer_cls = registry.get("comparison_renderers", comparison_type)
    except Exception as e:
        rprint(
            f"[red]No renderer registered for comparison "
            f"{comparison_type!r}.[/red]"
        )
        raise typer.Exit(code=1) from e

    cmp = cmp_cls()
    result = cmp.compare(pred, base)
    if result is None:
        rprint(
            f"[yellow]{comparison_type} comparison returned no result — "
            f"likely missing required input on one side (requires "
            f"{getattr(cmp, 'requires', set())!r}).[/yellow]"
        )
        raise typer.Exit(code=1)

    renderer = renderer_cls()
    if html:
        body = renderer.render_html(
            result, prediction_label=label_a, baseline_label=label_b,
        )
        _serve_html(body, page_id=page_id, label_a=label_a, label_b=label_b)
        return

    renderer.render_terminal(
        result,
        prediction_label=label_a,
        baseline_label=label_b,
        console=Console(),
    )


def _build_view(
    page_rows: dict[str, dict[str, Any]],
    label: str,
    *,
    kind: str,
) -> PredictionView | BaselineView | None:
    """Translate a model name (or the ``reference`` pseudo-model) into the
    appropriate Pydantic view object. Both view types are interchangeable
    on the sides of a Comparison; ``kind`` only affects which class is
    returned (so the caller can keep its naming clean).
    """
    if label == REFERENCE_PSEUDO_MODEL:
        any_row = next(iter(page_rows.values()))
        text = any_row.get("reference_text")
        if not text:
            return None
        provenance = _parse_provenance(any_row.get("reference_provenance_json"))
        # Render comparisons use the file_id as the human identifier.
        view_id = any_row.get("file_id") or any_row["page_id"]
        if kind == "baseline":
            return BaselineView(
                page_id=view_id, label=label, text=text, provenance=provenance,
            )
        return PredictionView(page_id=view_id, label=label, text=text)
    row = page_rows.get(label)
    if row is None:
        return None
    text = row.get("text") or row.get("markdown")
    document = _parse_document(row.get("document_json"))
    view_id = row.get("file_id") or row["page_id"]
    if kind == "baseline":
        return BaselineView(
            page_id=view_id, label=label, text=text, document=document,
        )
    return PredictionView(
        page_id=view_id, label=label, text=text, document=document,
    )


def _parse_document(document_json: str | None) -> Any:
    """Lazy-deserialize a DoclingDocument. Returns ``None`` on any failure
    so the caller can fall back to text-only comparisons."""
    if not document_json:
        return None
    try:
        from docling_core.types.doc.document import DoclingDocument
    except ImportError:
        return None
    try:
        return DoclingDocument.model_validate_json(document_json)
    except Exception:  # noqa: BLE001
        return None


def _parse_provenance(raw: str | None) -> ReferenceProvenance | None:
    if not raw:
        return None
    try:
        return ReferenceProvenance.model_validate_json(raw)
    except Exception:  # noqa: BLE001
        return None


def _format_provenance(provenance: ReferenceProvenance | None) -> str:
    if provenance is None:
        return ""
    bits: list[str] = [provenance.method]
    if provenance.engine:
        bits.append(provenance.engine)
    if provenance.confidence is not None:
        bits.append(f"conf={provenance.confidence:.2f}")
    return ", " + ", ".join(bits)


def _serve_html(html: str, *, page_id: str, label_a: str, label_b: str) -> None:
    """Serve ``html`` from a one-shot localhost HTTP server, open in browser."""
    body = html.encode("utf-8")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/diff", "/index.html", "/compare"):
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
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
    port = server.server_address[1]
    local_url = f"http://127.0.0.1:{port}/"
    lan_ip = _detect_lan_ip()
    lan_url = f"http://{lan_ip}:{port}/" if lan_ip else None

    rprint(
        f"\n[bold cyan]Compare: {page_id}[/bold cyan]   "
        f"[red strike]── {label_a}[/red strike]   "
        f"[green]── {label_b}[/green]"
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
        rprint("\n[dim]Compare server stopped.[/dim]")
    finally:
        server.server_close()


def _detect_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        try:
            ip = socket.gethostbyname(socket.gethostname())
            return ip if ip and not ip.startswith("127.") else None
        except OSError:
            return None


def _snippet_from_doc(document_json: str | None, length: int) -> str:
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


def _fmt_metric(column: str, value: Any) -> str:
    if value is None:
        return "—"
    if column == "text_similarity":
        return f"{float(value):.1f}%"
    if column in ("text_cer", "text_wer", "layout_iou_mean"):
        return f"{float(value):.3f}"
    if column.endswith("_delta"):
        try:
            n = int(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{n:+d}"
    return str(value)


# Used by ComparisonResult deserialization round-trip in callers — safe to
# import name even if unused locally.
__all__ = ["ComparisonResult"]

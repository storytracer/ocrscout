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

    _show_runtime_header(output_dir)

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
            # Hardware/cost context. Timing data lives in `metrics_json`
            # (loaded above); the cost-callback's `elapsed_seconds` /
            # `litellm_cost` / `gpu_time_cost` flat columns aren't read by
            # the inspector — the dispatcher's `run_seconds_total` /
            # `run_seconds_per_page` are the single source of truth.
            "gpu_type": raw.get("gpu_type"),
            "provider": raw.get("provider"),
            "cost_per_hour": raw.get("cost_per_hour"),
            "kv_cache_memory_bytes": raw.get("kv_cache_memory_bytes"),
            "concurrent_requests": raw.get("concurrent_requests"),
            "region_concurrency": raw.get("region_concurrency"),
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


def _show_runtime_header(output_dir: Path) -> None:
    """Render the autoscaler context from ``<output_dir>/runtime.yaml`` if
    present. No-op for older runs that predate the file."""
    from ocrscout.runtime import read_runtime_context

    try:
        ctx = read_runtime_context(output_dir)
    except Exception as e:  # noqa: BLE001
        rprint(f"[yellow](runtime.yaml present but unreadable: {e})[/yellow]")
        return
    if ctx is None:
        return

    gpu_label = (
        f"{ctx.gpu.name} (free "
        f"{_fmt_gib(ctx.gpu.free_bytes_at_launch)} / total "
        f"{_fmt_gib(ctx.gpu.total_bytes)})"
        if ctx.gpu
        else "no GPU recorded"
    )
    table = Table(title=f"Run context — {gpu_label}", show_header=False)
    table.add_column("k", style="dim")
    table.add_column("v")
    table.add_row(
        "runner",
        f"{ctx.runner.name}  "
        f"gpu_budget={ctx.runner.gpu_budget:.2f}  "
        f"parallel-models={ctx.runner.parallel_models}"
        + (
            f"  batch-concurrency={ctx.runner.batch_concurrency_override}"
            if ctx.runner.batch_concurrency_override is not None
            else ""
        ),
    )
    if ctx.autoscale:
        for name, rec in ctx.autoscale.profiles.items():
            tag = " [dim](explicit)[/dim]" if rec.explicit_kv_in_yaml else ""
            # Each profile has one of the two populated (per-backend
            # ceiling fix). Render whichever is non-zero.
            concurrency = rec.concurrent_requests or rec.region_concurrency
            table.add_row(
                name + tag,
                f"concurrency={concurrency}  "
                f"kv={_fmt_gib(rec.kv_cache_memory_bytes)}  "
                f"max_model_len={rec.max_model_len}",
            )
    Console().print(table)


def _fmt_gib(n: int | None) -> str:
    if n is None or n <= 0:
        return "—"
    return f"{n / (1024 ** 3):.1f} GiB"


def _show_aggregate(rows: list[dict[str, Any]]) -> None:
    """Print one row per model with run-level totals + batch throughput.

    Every timing number comes from ``metrics_json`` (the dispatcher's
    stamped per-model stats) — never from the cost callback's per-page
    ``elapsed_seconds`` column. The dispatcher already produces the
    canonical numbers (``run_seconds_total``, ``run_seconds_per_page``);
    duplicating them via sum/median of per-page values just adds a
    second source of truth that drifts when backends fan requests out
    concurrently.

    Designed for eyeballing two runs side by side (e.g. DGX vs cloud
    GPU): one table per output dir, each cell directly comparable to the
    same cell in the other table.
    """
    # Group rows by model (one parquet row per (page, model)).
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    aggregates: list[dict[str, Any]] = []
    for model in sorted(by_model):
        model_rows = by_model[model]

        # `run_seconds_total` is stamped uniformly per model by the
        # dispatcher (see cli/run.py); pick the first row that has it.
        run_seconds_total: float | None = None
        for r in model_rows:
            v = (r.get("metrics") or {}).get("run_seconds_total")
            if v is not None:
                run_seconds_total = float(v)
                break

        failed = sum(1 for r in model_rows if r.get("error"))
        succeeded = len(model_rows) - failed
        gpu_type = next(
            (r.get("gpu_type") for r in model_rows
             if r.get("gpu_type") and r.get("gpu_type") != "unknown"),
            None,
        )
        provider = next(
            (r.get("provider") for r in model_rows
             if r.get("provider") and r.get("provider") != "local"),
            None,
        )
        cost_per_hour = next(
            (r.get("cost_per_hour") for r in model_rows
             if r.get("cost_per_hour")),
            None,
        )
        # `total` is the dispatcher's actual wall clock for this model;
        # `avg` divides by visible rows so it matches what the dispatcher
        # prints at end-of-run (total / pages_ok).
        total = run_seconds_total
        avg = (
            run_seconds_total / len(model_rows)
            if run_seconds_total is not None and model_rows else None
        )
        est_cost = (
            total / 3600.0 * cost_per_hour
            if total is not None and cost_per_hour is not None
            else None
        )
        # Each row has exactly one of the two populated depending on the
        # backend (litellm → concurrent_requests, layout_chat →
        # region_concurrency). Coalesce so the table renders the actual
        # value the backend used.
        concurrency = next(
            (r.get("concurrent_requests") or r.get("region_concurrency")
             for r in model_rows
             if r.get("concurrent_requests") or r.get("region_concurrency")),
            None,
        )
        kv_bytes = next(
            (r.get("kv_cache_memory_bytes") for r in model_rows
             if r.get("kv_cache_memory_bytes")),
            None,
        )
        aggregates.append({
            "model": model,
            "succeeded": succeeded,
            "failed": failed,
            "total": total,
            "avg": avg,
            "gpu_type": gpu_type,
            "provider": provider,
            "cost_per_hour": cost_per_hour,
            "est_cost": est_cost,
            "concurrency": concurrency,
            "kv_bytes": kv_bytes,
        })

    show_gpu = any(a["gpu_type"] or a["provider"] for a in aggregates)
    show_cost = any(a["cost_per_hour"] is not None for a in aggregates)
    show_concurrency = any(
        a["concurrency"] or a["kv_bytes"] for a in aggregates
    )

    table = Table(title="run aggregate — batch throughput per model")
    table.add_column("model", style="bold")
    table.add_column("pages", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("total", justify="right")
    table.add_column("avg/page", justify="right")
    if show_gpu:
        table.add_column("gpu", style="dim")
    if show_concurrency:
        table.add_column("concur", justify="right", style="dim")
        table.add_column("kv", justify="right", style="dim")
    if show_cost:
        table.add_column("$/hr", justify="right", style="dim")
        table.add_column("est $", justify="right", style="dim")

    for a in aggregates:
        cells = [
            a["model"],
            _fmt_int(a["succeeded"]),
            f"[red]{a['failed']}[/red]" if a["failed"] else "0",
            _fmt_seconds_human(a["total"]),
            _fmt_seconds_human(a["avg"]),
        ]
        if show_gpu:
            gpu_label = (
                f"{a['gpu_type'] or '?'} ({a['provider'] or '?'})"
                if (a["gpu_type"] or a["provider"]) else "—"
            )
            cells.append(gpu_label)
        if show_concurrency:
            cells.append(
                str(a["concurrency"]) if a["concurrency"] else "—"
            )
            cells.append(_fmt_gib(a["kv_bytes"]))
        if show_cost:
            cells.append(
                f"{a['cost_per_hour']:.2f}"
                if a["cost_per_hour"] is not None else "—"
            )
            cells.append(
                f"{a['est_cost']:.4f}"
                if a["est_cost"] is not None else "—"
            )
        table.add_row(*cells)

    Console().print(table)


def _show_summary(rows: list[dict[str, Any]], *, snippet_length: int) -> None:
    _show_aggregate(rows)
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
        # Throughput-based per-page time stamped by the dispatcher
        # (same value across all rows of a model — single source of truth
        # with the run aggregate above).
        s_per_page = _fmt_seconds_human(m.get("run_seconds_per_page"))
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
        s_per_page = _fmt_seconds_human(m.get("run_seconds_per_page"))
        rprint(
            f"[dim]items={m.get('item_count')}  chars={m.get('text_length')}  "
            f"s/page={s_per_page}[/dim]"
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


def _fmt_seconds_human(value: Any) -> str:
    """Compact human-friendly duration: ``42.10s`` / ``5m 42s`` / ``1h 14m``.

    Used by the run-aggregate table where totals can span seconds → hours.
    Per-page tables keep using :func:`_fmt_seconds` (always 2-decimal s)
    because the per-page magnitudes there are predictable and direct
    comparison matters.
    """
    if not isinstance(value, (int, float)):
        return "—"
    if value < 60:
        return f"{value:.2f}s"
    if value < 3600:
        m, s = divmod(int(round(value)), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(round(value)), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


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

"""`ocrscout inspect` — side-by-side comparison of a previous run's output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from ocrscout.cli import app


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

    if page is not None:
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

"""`ocrscout report` — render a results directory into a report."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ocrscout.cli import app


@app.command("report")
def report(
    results_dir: Path = typer.Argument(..., help="Directory of `ocrscout run` results."),
    fmt: str = typer.Option(
        "html", "--format", "-f", help="Reporter name (html|markdown|terminal)."
    ),
    out: Path = typer.Option(
        Path("./report.html"), "--out", "-o", help="Where to write the report."
    ),
) -> None:
    """Render results from a previous run."""
    rprint(f"[bold]Would render {results_dir} → {out} via reporter {fmt!r}.[/bold]")
    rprint(
        "[yellow](stub) reporter execution arrives in phase 8 of the roadmap; "
        "see CLAUDE.md.[/yellow]"
    )

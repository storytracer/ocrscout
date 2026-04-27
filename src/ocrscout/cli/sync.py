"""`ocrscout sync` — refresh auto-generated profiles from uv-scripts/ocr."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.sync import sync_profiles


@app.command("sync")
def sync(
    scripts_dir: Path | None = typer.Option(
        None,
        "--scripts-dir",
        help="Local checkout of uv-scripts/ocr to introspect instead of fetching.",
    ),
    no_fetch: bool = typer.Option(
        False,
        "--no-fetch",
        help="Skip fetching from the Hub; expects --scripts-dir or a populated cache.",
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="HF Hub revision/commit to pin."
    ),
) -> None:
    """Snapshot the upstream scripts and write auto profiles to the user cache."""
    fetch = not no_fetch and scripts_dir is None
    result = sync_profiles(scripts_dir=scripts_dir, fetch=fetch, revision=revision)
    rprint(f"[bold]Sync complete[/bold] — read scripts from {result.scripts_dir}")
    rprint(f"  wrote: {len(result.written)}")
    rprint(f"  skipped (curated wins): {result.skipped or '-'}")
    if result.errored:
        rprint(f"[red]  errored: {len(result.errored)}[/red]")
        for path, msg in result.errored:
            rprint(f"    [red]{path.name}: {msg}[/red]")

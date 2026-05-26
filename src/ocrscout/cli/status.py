"""`ocrscout status` — show the active runner's state."""

from __future__ import annotations

import typer
from rich import print as rprint
from rich.table import Table

from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.log import setup_logging
from ocrscout.registry import registry


def _humanize_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60}m"


@app.command("status")
def status(
    runner: str | None = typer.Option(
        None, "--runner",
        help="Runner to query. Defaults to whatever's in ~/.ocrscout/state.yaml.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Print a one-shot snapshot of the active runner."""
    setup_logging(verbosity=verbose, quiet=quiet)

    state = state_mod.read_state()
    runner_name = runner or (state.runner if state else None) or "local"
    try:
        runner_cls = registry.get("runners", runner_name)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(str(e)) from e

    snapshot = runner_cls().status()
    rprint(
        f"[bold]{snapshot.runner}[/bold] state="
        f"[cyan]{snapshot.state}[/cyan]  "
        f"models=[cyan]{','.join(snapshot.models) or '—'}[/cyan]"
    )
    details = snapshot.details or {}
    phase = details.get("phase")
    phase_updated_at = details.get("phase_updated_at")
    if phase == "launching":
        if details.get("stale_launching"):
            rprint(
                f"[bold yellow]warning:[/bold yellow] state.yaml shows phase="
                f"'launching' from {phase_updated_at} — the launcher likely "
                f"crashed.\n  Run [cyan]ocrscout down --force[/cyan] to scan "
                f"ports and clean up."
            )
        else:
            rprint(
                f"  phase: [cyan]launching[/cyan] (since {phase_updated_at})"
            )
    elif phase == "tearing_down":
        rprint(f"  phase: [yellow]tearing_down[/yellow] (since {phase_updated_at})")
    if snapshot.proxy_url:
        rprint(f"  proxy: [cyan]{snapshot.proxy_url}[/cyan]")
    rprint(f"  uptime: {_humanize_uptime(snapshot.uptime_seconds)}")
    rprint(
        f"  pages: {snapshot.pages_done} done, {snapshot.pages_failed} failed"
    )
    rprint(f"  cumulative cost: ${snapshot.cumulative_cost:.4f}")

    procs = snapshot.details.get("processes") if snapshot.details else None
    if procs:
        table = Table(title="Daemons", show_lines=False)
        table.add_column("name", style="cyan")
        table.add_column("pid", justify="right")
        table.add_column("port", justify="right")
        table.add_column("alive")
        for p in procs:
            table.add_row(
                str(p.get("name", "")),
                str(p.get("pid") or "—"),
                str(p.get("port") or "—"),
                "yes" if p.get("alive") else "no",
            )
        rprint(table)

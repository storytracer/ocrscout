"""`ocrscout logs` — tail logs from the active runner's daemons or a job."""

from __future__ import annotations

import typer

from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.log import setup_logging
from ocrscout.registry import registry


@app.command("logs")
def logs(
    job_id: str | None = typer.Argument(
        None,
        help="Optional job id (from `ocrscout submit`). When omitted, tails "
             "every daemon log managed by the active runner.",
    ),
    runner: str | None = typer.Option(
        None, "--runner",
        help="Runner whose logs to tail. Defaults to state.yaml's runner.",
    ),
    follow: bool = typer.Option(
        True, "--follow/--no-follow", "-f",
        help="Stream new log lines as they're written (tail -F). With "
             "--no-follow, print the current tail and exit.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Tail daemon (and optionally per-job) logs."""
    setup_logging(verbosity=verbose, quiet=quiet)

    state = state_mod.read_state()
    runner_name = runner or (state.runner if state else None) or "local"
    try:
        runner_cls = registry.get("runners", runner_name)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(str(e)) from e
    runner_cls().logs(job_id=job_id, follow=follow)

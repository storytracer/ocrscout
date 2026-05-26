"""`ocrscout down` — tear down the active runner's stack."""

from __future__ import annotations

import logging

import typer

from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.log import setup_logging
from ocrscout.registry import registry

log = logging.getLogger(__name__)


@app.command("down")
def down(
    runner: str | None = typer.Option(
        None, "--runner",
        help="Runner to tear down. Defaults to whatever's in state.yaml.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """SIGTERM every daemon for the active runner, then clear state."""
    setup_logging(verbosity=verbose, quiet=quiet)

    state = state_mod.read_state()
    runner_name = runner or (state.runner if state else None)
    if runner_name is None:
        log.info("No active runner; nothing to tear down.")
        return

    try:
        runner_cls = registry.get("runners", runner_name)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(str(e)) from e
    runner_cls().down()

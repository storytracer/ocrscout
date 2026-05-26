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
    force: bool = typer.Option(
        False, "--force",
        help="Aggressive teardown: scan the LiteLLM proxy port (4000) and "
             "the vLLM port range (8000-8031) and SIGTERM anything "
             "listening, even with no state.yaml. Use this after a "
             "Ctrl-C'd `ocrscout run` leaves orphaned vllm-serve "
             "processes holding GPU memory. LocalRunner only.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """SIGTERM every daemon for the active runner, then clear state."""
    setup_logging(verbosity=verbose, quiet=quiet)

    state = state_mod.read_state()
    runner_name = runner or (state.runner if state else None)

    if force:
        # --force scans local ports, so it's LocalRunner-specific. When
        # state.yaml is missing we assume the user means local (the
        # common case after a Ctrl-C'd `ocrscout run`); when another
        # runner is recorded as active we reject the flag so we don't
        # accidentally kill an unrelated local process while a remote
        # runner is in flight.
        if runner_name not in (None, "local"):
            raise typer.BadParameter(
                f"--force only applies to the local runner (active "
                f"runner is {runner_name!r})."
            )
        from ocrscout.runners.local import LocalRunner

        LocalRunner().down(force=True)
        return

    if runner_name is None:
        log.info(
            "No active runner; nothing to tear down. "
            "(Use --force to scan ports for orphans.)"
        )
        return

    try:
        runner_cls = registry.get("runners", runner_name)
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(str(e)) from e
    runner_cls().down()

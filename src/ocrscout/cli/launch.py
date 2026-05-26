"""`ocrscout launch` — provision the active runner's compute stack.

Reads the user's selected runner from ``--runner`` (or, by default, the
``default_runner`` field in ``~/.ocrscout/config.yaml``; ultimately
``local``), resolves the named profiles, and calls
``Runner.launch(...)``. Persists everything subsequent CLI commands need
to ``~/.ocrscout/state.yaml`` so ``submit``/``status``/``down`` can run
later without re-passing flags.
"""

from __future__ import annotations

import logging

import typer
from rich import print as rprint

from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.errors import ProfileNotFound, RunnerError, ScoutError
from ocrscout.log import setup_logging
from ocrscout.registry import registry

log = logging.getLogger(__name__)


@app.command("launch")
def launch(
    models: str = typer.Option(
        ..., "--models", "-m",
        help="Comma-separated profile names. Mixed runtimes are fine: the "
             "Runner spawns vLLM serves for runtime=vllm profiles, adds "
             "model_list entries for runtime=hosted, and leaves runtime=cpu "
             "alone (they run in-process via the dispatcher).",
    ),
    runner: str | None = typer.Option(
        None, "--runner",
        help="Which Runner to use (local/skypilot/hf). Defaults to "
             "default_runner from ~/.ocrscout/config.yaml, or 'local'.",
    ),
    gpu_type: str | None = typer.Option(
        None, "--gpu",
        help="GPU type label stamped onto every output row (e.g. L4, "
             "GB10). Falls back to the value in ~/.ocrscout/config.yaml.",
    ),
    workers: int = typer.Option(
        1, "--workers",
        help="Number of remote workers (skypilot/hf only; ignored for local).",
    ),
    base_port: int = typer.Option(
        8000, "--base-port",
        help="First TCP port for vLLM serves (local only).",
    ),
    proxy_port: int = typer.Option(
        4000, "--proxy-port",
        help="LiteLLM proxy port (local only).",
    ),
    gpu_budget: float = typer.Option(
        0.85, "--gpu-budget",
        help="Maximum total GPU memory the stack may collectively claim "
             "(local only). Per-model KV is set by the profile's "
             "kv_cache_memory_bytes; this bounds the sum + overhead.",
    ),
    ready_timeout: float = typer.Option(
        600.0, "--ready-timeout",
        help="Seconds to wait for every spawned daemon to report ready.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Launch the active runner's compute stack."""
    setup_logging(verbosity=verbose, quiet=quiet)

    runner_name = runner or state_mod.read_config().default_runner
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise typer.BadParameter("--models must list at least one profile")

    try:
        runner_cls = registry.get("runners", runner_name)
    except ScoutError as e:
        raise typer.BadParameter(str(e)) from e
    runner_obj = runner_cls()

    try:
        handle = runner_obj.launch(
            models=model_list,
            gpu_type=gpu_type,
            workers=workers,
            base_port=base_port,
            proxy_port=proxy_port,
            gpu_budget=gpu_budget,
            ready_timeout=ready_timeout,
        )
    except ProfileNotFound as e:
        raise typer.BadParameter(str(e)) from e
    except RunnerError as e:
        log.error("Launch failed: %s", e)
        raise typer.Exit(code=1) from e

    rprint(f"[bold green]ocrscout {runner_name} ready[/bold green]")
    rprint(f"  proxy: [cyan]{handle.proxy_url}[/cyan]")
    rprint(f"  models: [cyan]{','.join(model_list)}[/cyan]")
    if handle.endpoints:
        for ep in handle.endpoints:
            pid_part = f" (pid {ep.pid})" if ep.pid is not None else ""
            rprint(f"    - {ep.model}: {ep.url}{pid_part}")

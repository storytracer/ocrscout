"""`ocrscout serve` — long-lived managed multi-server stack.

Spawns one ``vllm serve`` per vllm-source profile (and a LiteLLM proxy on top
when N >= 2), prints the proxy URL, then blocks until SIGINT. On Ctrl-C,
tears down the whole stack cleanly.

Pair with ``ocrscout run --server-url <printed-url>`` in another terminal to
drive runs against warm models without paying startup cost per run.
"""

from __future__ import annotations

import signal
import time

import typer
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.errors import ManagedServerError, ProfileNotFound
from ocrscout.managed import gpu_state_lines, managed_servers
from ocrscout.profile import resolve


@app.command("serve")
def serve(
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated curated profile names."
    ),
    gpu_budget: float = typer.Option(
        0.85,
        "--gpu-budget",
        help="Total fraction of GPU memory available to managed vllm-serves "
             "(equally split across them). 0.85 leaves 15%% for the OS and "
             "other processes.",
    ),
    base_port: int = typer.Option(
        8000,
        "--base-port",
        help="First vllm-serve port; subsequent vllm-serves use base+1, base+2, …",
    ),
    proxy_port: int = typer.Option(
        4000,
        "--proxy-port",
        help="LiteLLM proxy port (only used when 2+ vllm models are managed).",
    ),
    ready_timeout: float = typer.Option(
        600.0,
        "--ready-timeout",
        help="Per-server seconds to wait for /v1/models to return 200.",
    ),
) -> None:
    """Start managed vllm-serves + LiteLLM proxy; print URL; block until Ctrl-C."""
    names = [m.strip() for m in models.split(",") if m.strip()]
    if not names:
        raise typer.BadParameter("--models must list at least one profile name")

    profiles = []
    for name in names:
        try:
            profiles.append(resolve(name))
        except ProfileNotFound as e:
            raise typer.BadParameter(str(e)) from e

    vllm_profiles = [p for p in profiles if p.source == "vllm"]
    if not vllm_profiles:
        raise typer.BadParameter(
            "ocrscout serve only manages vllm-source profiles; the requested "
            f"set has none. Got sources: {sorted({p.source for p in profiles})}"
        )

    rprint(
        f"[bold]Starting managed stack[/bold] — {len(vllm_profiles)} vllm model(s), "
        f"GPU budget {gpu_budget:.0%} (~{gpu_budget / len(vllm_profiles):.2f} per model)."
    )

    try:
        with managed_servers(
            profiles,
            gpu_budget=gpu_budget,
            base_port=base_port,
            proxy_port=proxy_port,
            ready_timeout=ready_timeout,
        ) as handle:
            rprint("\n[bold green]Managed multi-server stack ready.[/bold green]")
            rprint(f"  proxy:  [bold]{handle.proxy_url}[/bold]"
                   f" ({len(vllm_profiles)} model{'s' if len(vllm_profiles) != 1 else ''})")
            rprint("  models:")
            for model_id, url in handle.server_urls.items():
                rprint(f"    - {model_id}  [dim]({url})[/dim]")
            rprint(f"  logs:   [dim]{handle.log_dir}[/dim]")
            for line in gpu_state_lines():
                rprint(f"  [dim]{line}[/dim]")
            rprint("\nRun against this stack with:")
            rprint(
                f"  [cyan]ocrscout run --models <names> --server-url {handle.proxy_url}[/cyan]\n"
            )
            rprint("[dim]Press Ctrl-C to stop.[/dim]")

            _block_until_signal()
            rprint("\n[dim]Tearing down managed stack...[/dim]")
    except ManagedServerError as e:
        rprint(f"[red]Managed stack failed: {e}[/red]")
        raise typer.Exit(code=1) from e


def _block_until_signal() -> None:
    """Block in the foreground until SIGINT/SIGTERM, then return."""
    # signal.pause() is POSIX-only; the loop fallback is portable.
    if hasattr(signal, "pause"):
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass
    else:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

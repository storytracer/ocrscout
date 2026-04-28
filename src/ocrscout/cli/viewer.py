"""`ocrscout viewer` — interactive Gradio inspector for a previous run.

Same input as `ocrscout inspect` (a results directory containing
``data/train-*.parquet``), but launches a long-lived Gradio app on a local
port so a user can walk pages, toggle models, switch view modes, and share
URLs.

Gradio is a heavy dependency, so it's gated behind the ``viewer`` optional
extra. If the import fails, this command prints the install hint and exits
non-zero rather than carrying a hard dep in the core install.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.log import setup_logging

log = logging.getLogger(__name__)


@app.command("viewer")
def viewer(
    output_dir: Path = typer.Argument(
        ..., help="A previous run's --output-dir (must contain data/train-*.parquet)."
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host",
        help="Interface to bind on. Use 0.0.0.0 to expose on the LAN.",
    ),
    port: int = typer.Option(
        7860, "--port",
        help="Local port; gradio will pick the next free port if this is taken.",
    ),
    share: bool = typer.Option(
        False, "--share",
        help="Request a public *.gradio.live tunnel from gradio.",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open",
        help="Open the URL in the default browser on startup.",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True,
        help="Increase log verbosity. -v adds timestamps; -vv adds module:line.",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet",
        help="Suppress informational logging; show only warnings/errors.",
    ),
) -> None:
    """Launch the Gradio inspector against a previous run's output directory."""
    setup_logging(verbosity=verbose, quiet=quiet)

    from ocrscout.exports.layout import find_parquet_files

    if not find_parquet_files(output_dir):
        rprint(f"[red]No data/train-*.parquet under {output_dir}[/red]")
        raise typer.Exit(code=1)

    try:
        from ocrscout.viewer.app import build_app
    except ImportError as e:
        rprint(
            "[red]The viewer command needs the optional `viewer` extra "
            f"(missing: {e.name}).[/red]\n"
            "[yellow]Install it with:  pip install 'ocrscout[viewer]'  "
            "(or `uv pip install 'ocrscout[viewer]'`).[/yellow]"
        )
        raise typer.Exit(code=1) from e

    demo = build_app(output_dir)
    log.info("viewer: launching gradio on %s:%d", host, port)
    # css/head are attached to the Blocks by build_app — passed to launch()
    # rather than the constructor because Gradio 6.0 moves them there.
    launch_kwargs: dict = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "inbrowser": open_browser,
        "quiet": quiet,
    }
    css = getattr(demo, "ocrscout_css", None)
    head = getattr(demo, "ocrscout_head", None)
    if css is not None:
        launch_kwargs["css"] = css
    if head is not None:
        launch_kwargs["head"] = head
    demo.launch(**launch_kwargs)

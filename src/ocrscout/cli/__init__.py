"""Typer CLI entrypoint for `ocrscout`."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="ocrscout",
    help="Scout frontier OCR models on your data, your hardware, your terms.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Side-effect imports register the sub-commands on `app`.
from ocrscout.cli import apply as _apply  # noqa: E402, F401
from ocrscout.cli import inspect as _inspect  # noqa: E402, F401
from ocrscout.cli import introspect as _introspect  # noqa: E402, F401
from ocrscout.cli import publish as _publish  # noqa: E402, F401
from ocrscout.cli import report as _report  # noqa: E402, F401
from ocrscout.cli import run as _run  # noqa: E402, F401
from ocrscout.cli import serve as _serve  # noqa: E402, F401
from ocrscout.cli import viewer as _viewer  # noqa: E402, F401


def main() -> None:
    app()


__all__ = ["app", "main"]

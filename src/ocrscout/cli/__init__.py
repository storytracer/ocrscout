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

# Side-effect imports register the sub-commands on `app`. Order is purely
# cosmetic — typer surfaces commands in declaration order in --help.
from ocrscout.cli import apply as _apply  # noqa: E402, F401
from ocrscout.cli import benchmark as _benchmark  # noqa: E402, F401
from ocrscout.cli import costs as _costs  # noqa: E402, F401
from ocrscout.cli import down as _down  # noqa: E402, F401
from ocrscout.cli import inspect as _inspect  # noqa: E402, F401
from ocrscout.cli import introspect as _introspect  # noqa: E402, F401
from ocrscout.cli import launch as _launch  # noqa: E402, F401
from ocrscout.cli import layout as _layout  # noqa: E402, F401
from ocrscout.cli import logs as _logs  # noqa: E402, F401
from ocrscout.cli import normalize as _normalize  # noqa: E402, F401
from ocrscout.cli import ocr as _ocr  # noqa: E402, F401
from ocrscout.cli import publish as _publish  # noqa: E402, F401
from ocrscout.cli import report as _report  # noqa: E402, F401
from ocrscout.cli import run as _run  # noqa: E402, F401
from ocrscout.cli import sample as _sample  # noqa: E402, F401
from ocrscout.cli import source as _source  # noqa: E402, F401
from ocrscout.cli import status as _status  # noqa: E402, F401
from ocrscout.cli import submit as _submit  # noqa: E402, F401
from ocrscout.cli import viewer as _viewer  # noqa: E402, F401

# Hidden subcommand spawned by LocalRunner.submit; never invoked directly.
from ocrscout.cli import _worker as _worker  # noqa: E402, F401


def main() -> None:
    app()


__all__ = ["app", "main"]

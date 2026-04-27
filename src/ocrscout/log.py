"""Logging setup for the ocrscout CLI — stdlib only.

Single coherent ingest path: all package modules log via
``logging.getLogger("ocrscout.…")``; one ``StreamHandler`` writes them with a
verbosity-controlled format. CLI commands call :func:`setup_logging` exactly
once at entry.

No Rich, no fancy formatting, no markup parsing. Plain text, one logical
line per record, no width-based wrapping. The summary table in ``ocrscout
run`` and the ready banner in ``ocrscout serve`` still use ``rich.print``
because those are *presentation* artifacts (always shown, even at ``-q``) —
but log records themselves stay boring on purpose.

Verbosity levels:

* ``-q`` → WARNING and above only.
* default → INFO: bare message, no timestamp or level prefix.
* ``-v`` → VERBOSE (15): timestamp + level prefix.
* ``-vv`` → DEBUG: timestamp + level + ``module:lineno`` prefix.
"""

from __future__ import annotations

import logging
import sys

VERBOSE = 15  # between INFO (20) and DEBUG (10)
logging.addLevelName(VERBOSE, "VERBOSE")

_PACKAGE_LOGGER = "ocrscout"

_FMT_DEFAULT = "%(message)s"
_FMT_VERBOSE = "%(asctime)s %(levelname)-7s %(message)s"
_FMT_DEBUG = "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d  %(message)s"
_TIME_FMT = "%H:%M:%S"


def setup_logging(verbosity: int = 0, *, quiet: bool = False) -> None:
    """Configure the ``ocrscout`` logger tree.

    Idempotent — replaces any pre-existing handler so re-invocations don't
    double-log. Other libraries' loggers are not touched.

    Parameters
    ----------
    verbosity
        0 = default (INFO). 1 = ``-v`` (VERBOSE). 2+ = ``-vv`` (DEBUG).
    quiet
        Override that demotes the level to WARNING. Useful for ``-q``.
    """
    if quiet:
        level = logging.WARNING
        fmt = _FMT_DEFAULT
    elif verbosity <= 0:
        level = logging.INFO
        fmt = _FMT_DEFAULT
    elif verbosity == 1:
        level = VERBOSE
        fmt = _FMT_VERBOSE
    else:
        level = logging.DEBUG
        fmt = _FMT_DEBUG

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_TIME_FMT))

    root = logging.getLogger(_PACKAGE_LOGGER)
    # Idempotent reset.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    # Don't bubble to the root logger — it would double-log if anyone else
    # configures basicConfig downstream.
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return the ocrscout-namespaced logger for ``name``.

    Pass the module's ``__name__`` (e.g. ``"ocrscout.backends.vllm"``).
    The package prefix is preserved as-is — this helper is just a centralized
    naming convention.
    """
    return logging.getLogger(name)

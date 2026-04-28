"""Gradio-based interactive viewer for ocrscout results.

Loaded lazily by the `ocrscout viewer` CLI command — gradio and polars are
only needed for the viewer optional extra and aren't pulled in by core.
"""

from __future__ import annotations

__all__ = ["build_app"]


def build_app(*args, **kwargs):
    """Lazy proxy for ``ocrscout.viewer.app.build_app``.

    Importing this submodule pulls in ``gradio``; the proxy keeps `ocrscout
    -h` and other commands fast on installs without the ``viewer`` extra.
    """
    from ocrscout.viewer.app import build_app as _build

    return _build(*args, **kwargs)

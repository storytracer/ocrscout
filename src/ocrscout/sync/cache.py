"""Cache directory layout for ocrscout."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """Return the ocrscout cache root.

    Honors ``OCRSCOUT_CACHE_DIR`` if set, then ``XDG_CACHE_HOME``, falling back
    to ``~/.cache/ocrscout``.
    """
    explicit = os.environ.get("OCRSCOUT_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "ocrscout"


def profiles_cache_dir() -> Path:
    return cache_root() / "profiles"


def scripts_cache_dir() -> Path:
    return cache_root() / "uv-scripts-ocr"

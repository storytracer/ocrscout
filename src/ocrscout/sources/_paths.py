"""Per-source directory layout under ``~/.ocrscout/sources/``.

Each source owns a subdirectory:

```
~/.ocrscout/sources/<name>/
    info.yaml         # SourceInfo (provisioning record)
    catalog/          # raw upstream cache (TSVs, parquets, manifests)
    derived/          # intermediate artifacts produced by refresh
```

The root sits under :func:`ocrscout.state.state_dir` so every persistent
ocrscout artifact (state, configs, source caches) lives under one tree —
distinct from ``~/.cache/ocrscout/`` (the legacy / sync-script cache root
in :mod:`ocrscout.sync.cache`).

Override the root with the ``OCRSCOUT_SOURCES_DIR`` env var; otherwise it
defaults to ``state_dir() / "sources"``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ocrscout.state import state_dir
from ocrscout.sync.cache import cache_root


def sources_dir() -> Path:
    """Compute the sources root. Pure path — no mkdir.

    Honors ``OCRSCOUT_SOURCES_DIR``; falls back to
    ``state_dir() / "sources"``. Callers about to *write* should
    explicitly ``mkdir(parents=True, exist_ok=True)`` on the leaf path
    they're writing to (or rely on :func:`ocrscout.state._atomic_write_yaml`
    which mkdirs the parent for you).
    """
    raw = os.environ.get("OCRSCOUT_SOURCES_DIR")
    return Path(raw).expanduser() if raw else state_dir() / "sources"


def source_dir(name: str) -> Path:
    return sources_dir() / name


def catalog_dir(name: str) -> Path:
    return source_dir(name) / "catalog"


def derived_dir(name: str) -> Path:
    return source_dir(name) / "derived"


def info_path(name: str) -> Path:
    return source_dir(name) / "info.yaml"


def legacy_cache_dir(name: str) -> Path:
    """Pre-source-admin cache location at ``~/.cache/ocrscout/<name>/``.

    Surfaced by ``ocrscout source <name> info`` as a one-line nudge when
    the directory exists, so users notice they have a legacy cache to
    remove. Never auto-migrated — caches are rebuildable.
    """
    return cache_root() / name

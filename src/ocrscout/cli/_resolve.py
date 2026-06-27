"""Shared source-resolution for the stage CLI commands.

A ``--source`` value that points at a stage parquet (a ``*.parquet`` file, or a
directory holding ``data/<stage>-*.parquet`` shards) is served by the ``pages``
adapter; anything else is a real source adapter (``hf_dataset`` path, ``bhl``,
…). Centralised here so ``run`` / ``sample`` / ``layout`` / ``ocr`` all detect
parquet inputs the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ocrscout.types import AdapterRef


def parse_source_args(raw: list[str]) -> dict[str, Any]:
    """Parse repeated ``--source-arg key=value`` strings into a kwargs dict.

    Values are JSON-parsed when possible (so list/dict/int/bool literals work)
    and fall back to plain strings otherwise.
    """
    parsed: dict[str, Any] = {}
    for entry in raw:
        if "=" not in entry:
            raise typer.BadParameter(f"--source-arg expects 'key=value', got {entry!r}")
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"--source-arg has empty key in {entry!r}")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def parse_comparisons_flag(raw: str | None) -> list[str] | None:
    """Decode ``--comparisons``: ``None`` → default-on; ``"none"`` → opt-out
    (``[]``); else a comma-separated whitelist."""
    if raw is None:
        return None
    raw = raw.strip()
    if raw.lower() == "none":
        return []
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return names or None


def is_stage_parquet(source: str | None) -> bool:
    """True when ``source`` points at a materialized stage parquet."""
    if not source:
        return False
    if "://" in source:
        # Remote sources are real adapters (hf_dataset / bhl), never local
        # stage parquets.
        return False
    p = Path(source)
    if p.is_file():
        return p.suffix == ".parquet"
    if p.is_dir():
        from ocrscout.io.source_parquet import resolve_stage_files

        return bool(resolve_stage_files(p))
    return False


def build_source_ref(
    *,
    source: str | None,
    source_name: str,
    source_args: dict[str, Any],
    sample: int | None = None,
    seed: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> AdapterRef:
    """Build the source ``AdapterRef`` for a stage command.

    A ``--source`` pointing at a stage parquet routes to the ``pages`` adapter;
    otherwise the default ``hf_dataset`` adapter consumes it as a ``path`` and
    corpus-bound adapters (``bhl``) ignore it. ``sample`` / ``seed`` /
    ``start_idx`` / ``end_idx`` are threaded into the constructor kwargs;
    adapters that don't use them ignore them (all sources use ``extra="ignore"``).
    """
    args = dict(source_args)
    if source_name == "hf_dataset" and is_stage_parquet(source):
        source_name = "pages"
        args["path"] = source
    elif source_name == "hf_dataset":
        args["path"] = source
    if sample is not None:
        args.setdefault("sample", sample)
    if seed is not None:
        args.setdefault("seed", seed)
    if start_idx is not None:
        args.setdefault("start_idx", start_idx)
    if end_idx is not None:
        args.setdefault("end_idx", end_idx)
    return AdapterRef(name=source_name, args=args)

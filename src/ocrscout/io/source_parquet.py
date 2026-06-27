"""Resolve and read a stage-parquet pointer for the ``pages`` source adapter.

When ``--source`` points at a materialized stage parquet (a file, a dir holding
``data/<stage>-*.parquet``, or a glob), the ``pages`` source adapter and the
``precomputed`` detector read the page-identity columns back through here. Any
stage artifact works (each is a superset of the page columns).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ocrscout.io import paths
from ocrscout.io.rows import PageRow

# Most upstream first — prefer the leanest artifact that carries page identity.
_PREFERENCE = (paths.PAGES, paths.LAYOUT, paths.RAW, paths.TRAIN)
_PAGE_COLUMNS = tuple(PageRow.model_fields) + ("extra_json",)


def resolve_stage_files(path: str | Path) -> list[Path]:
    """Resolve a parquet pointer to a concrete shard list (``[]`` if none)."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        for base in (p / paths.DATA_DIR, p):
            if not base.is_dir():
                continue
            for prefix in _PREFERENCE:
                shards = sorted(base.glob(f"{prefix}-*.parquet"))
                if shards:
                    return shards
        return []
    matches = sorted(Path().glob(str(path)))
    return [m for m in matches if m.is_file()]


def read_stage_rows(files: list[Path]) -> list[dict[str, Any]]:
    """Read the page-identity columns from stage parquet ``files`` in order,
    decoding ``extra_json`` back into an ``extra`` dict."""
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in files:
        names = set(pq.read_schema(str(path)).names)
        cols = [c for c in _PAGE_COLUMNS if c in names]
        for rec in pq.read_table(str(path), columns=cols).to_pylist():
            extra_json = rec.pop("extra_json", None)
            rec["extra"] = json.loads(extra_json) if extra_json else {}
            rows.append(rec)
    return rows

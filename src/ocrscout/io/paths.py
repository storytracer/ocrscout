"""On-disk layout for ocrscout's staged parquet artifacts.

Every stage writes ``<output_dir>/data/<prefix>-NNNNN.parquet`` (HuggingFace
``data/<split>-*`` convention), so one output dir holds every stage's output
side by side and a published dataset round-trips byte-identically to a local
run. ``train`` is the final results split; ``pages`` / ``layout`` / ``raw`` are
the intermediate stage artifacts; ``volumes`` is the per-volume sidecar.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

DATA_DIR: Final = "data"

# Stage prefix → on-disk shard prefix. ``train`` keeps the historical name so
# existing readers (inspect/viewer/costs/publish) keep working unchanged.
PAGES: Final = "pages"
LAYOUT: Final = "layout"
RAW: Final = "raw"
TRAIN: Final = "train"
VOLUMES: Final = "volumes"

# Upstream → downstream order; used when resolving "the most upstream stage
# present" for a parquet source pointer (every artifact carries the page
# identity columns, so any one reconstructs a PageImage).
STAGE_ORDER: Final = (PAGES, LAYOUT, RAW, TRAIN)

_SHARD_RE = re.compile(r"-(\d+)(?:-of-\d+)?\.parquet$")


def shard_name(prefix: str, index: int) -> str:
    return f"{prefix}-{index:05d}.parquet"


def data_dir(output_dir: Path | str) -> Path:
    return Path(output_dir) / DATA_DIR


def glob_for(prefix: str) -> str:
    """Glob (relative to output_dir) for a stage's shards."""
    return f"{DATA_DIR}/{prefix}-*.parquet"


def find_shards(output_dir: Path | str, prefix: str) -> list[Path]:
    """All ``<prefix>-*.parquet`` shards under ``output_dir/data/``, sorted."""
    d = data_dir(output_dir)
    if not d.is_dir():
        return []
    return sorted(d.glob(f"{prefix}-*.parquet"))


def next_shard_index(output_dir: Path | str, prefix: str) -> int:
    """Next free shard number, continuing after any shards already present."""
    max_idx = -1
    for p in find_shards(output_dir, prefix):
        m = _SHARD_RE.search(p.name)
        if m is not None:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

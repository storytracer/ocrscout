"""On-disk layout for ocrscout's parquet output.

Follows the HuggingFace Hub convention used by ``Dataset.push_to_hub``:

* one ``data/`` subdirectory under the repo root,
* parquet files named ``<split>-NNNNN-of-NNNNN.parquet``,
* a single ``train`` split (the only split ocrscout produces today).

Applying the same layout to local run output (under ``--output-dir``) means
``snapshot_download`` of a published dataset is byte-identical to a fresh
local run, so the viewer / inspect commands work on either without
branching.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = "data"
SPLIT = "train"
DEFAULT_BASENAME = f"{SPLIT}-00000-of-00001.parquet"
DATA_GLOB = f"{DATA_DIR}/{SPLIT}-*.parquet"

VOLUMES_PREFIX = "volumes"
DEFAULT_VOLUMES_BASENAME = f"{VOLUMES_PREFIX}-00000-of-00001.parquet"
VOLUMES_GLOB = f"{DATA_DIR}/{VOLUMES_PREFIX}-*.parquet"

# Decoupled-stage intermediate artifacts. Each lives under the same ``data/``
# directory as the final ``train-*.parquet`` so a single output dir can hold
# every stage's output side by side, discoverable with parallel globs.
PAGES_PREFIX = "pages"
LAYOUT_PREFIX = "layout"
RAW_PREFIX = "raw"
STAGE_PREFIXES: tuple[str, ...] = (PAGES_PREFIX, LAYOUT_PREFIX, RAW_PREFIX)


def stage_glob(prefix: str) -> str:
    """Glob (relative to output_dir) for a stage's parquet shards."""
    return f"{DATA_DIR}/{prefix}-*.parquet"


def find_stage_files(output_dir: Path, prefix: str) -> list[Path]:
    """All ``<prefix>-*.parquet`` shards under ``output_dir/data/``, sorted."""
    data_dir = output_dir / DATA_DIR
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob(f"{prefix}-*.parquet"))


def parquet_dest(output_dir: Path) -> Path:
    """Path the writer should create."""
    return output_dir / DATA_DIR / DEFAULT_BASENAME


def volumes_dest(output_dir: Path) -> Path:
    """Path the volumes-sidecar writer should create."""
    return output_dir / DATA_DIR / DEFAULT_VOLUMES_BASENAME


def volumes_dest_for_pages(pages_dest: Path | str) -> Path:
    """Derive the volumes-sidecar path from the per-page parquet path.

    Mirrors the per-page basename's shard-of-total suffix so the two files sit
    side-by-side under ``data/`` and can be discovered with parallel globs.
    """
    pages_path = Path(pages_dest)
    name = pages_path.name
    if name.startswith(f"{SPLIT}-"):
        suffix = name[len(SPLIT) :]
        return pages_path.with_name(f"{VOLUMES_PREFIX}{suffix}")
    return pages_path.with_name(DEFAULT_VOLUMES_BASENAME)


def find_parquet_files(output_dir: Path) -> list[Path]:
    """All parquet shards present under ``output_dir/data/``, sorted."""
    data_dir = output_dir / DATA_DIR
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob(f"{SPLIT}-*.parquet"))


def find_volumes_files(output_dir: Path) -> list[Path]:
    """All ``volumes-*.parquet`` shards present under ``output_dir/data/``, sorted."""
    data_dir = output_dir / DATA_DIR
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob(f"{VOLUMES_PREFIX}-*.parquet"))


def parquet_data_files(output_dir: Path) -> str:
    """``data_files`` argument for ``datasets.load_dataset("parquet", ...)``.

    Always returns a glob string so single- and multi-shard outputs both
    work without the caller needing to enumerate.
    """
    return str(output_dir / DATA_GLOB)


def volumes_data_files(output_dir: Path) -> str:
    """``data_files`` argument for loading the volumes sidecar via ``datasets``."""
    return str(output_dir / VOLUMES_GLOB)

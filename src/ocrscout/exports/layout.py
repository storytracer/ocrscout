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


def parquet_dest(output_dir: Path) -> Path:
    """Path the writer should create."""
    return output_dir / DATA_DIR / DEFAULT_BASENAME


def find_parquet_files(output_dir: Path) -> list[Path]:
    """All parquet shards present under ``output_dir/data/``, sorted."""
    data_dir = output_dir / DATA_DIR
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob(f"{SPLIT}-*.parquet"))


def parquet_data_files(output_dir: Path) -> str:
    """``data_files`` argument for ``datasets.load_dataset("parquet", ...)``.

    Always returns a glob string so single- and multi-shard outputs both
    work without the caller needing to enumerate.
    """
    return str(output_dir / DATA_GLOB)

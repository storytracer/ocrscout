"""Unified typed parquet IO for the staged pipeline.

One row-model family (:mod:`ocrscout.io.rows`), one schema derivation
(:mod:`ocrscout.io.schema`), one sharded writer/reader (:mod:`ocrscout.io.shards`),
a per-output-dir facade (:class:`ParquetStore`), and resume cursors read from
the shards (:mod:`ocrscout.io.resume`). Replaces the two divergent writers in
``ocrscout.exports``.
"""

from __future__ import annotations

from ocrscout.io.resume import ResumeMode, ResumeTracker
from ocrscout.io.rows import (
    CostColumns,
    LayoutRow,
    PageRow,
    RawRow,
    ResultRow,
    StageRow,
    VolumeRow,
)
from ocrscout.io.schema import columns_for, features_for
from ocrscout.io.shards import ShardReader, ShardWriter
from ocrscout.io.store import ARTIFACTS, ParquetStore

__all__ = [
    "ARTIFACTS",
    "CostColumns",
    "LayoutRow",
    "PageRow",
    "ParquetStore",
    "RawRow",
    "ResultRow",
    "ResumeMode",
    "ResumeTracker",
    "ShardReader",
    "ShardWriter",
    "StageRow",
    "VolumeRow",
    "columns_for",
    "features_for",
]

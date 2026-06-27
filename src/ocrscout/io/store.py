"""ParquetStore: the per-output-dir IO facade stages talk to.

Maps each stage artifact (``pages`` / ``layout`` / ``raw`` / ``train`` /
``volumes``) to its typed row model and hands out writers, readers, and resume
trackers. A stage never globs parquet or names a schema itself — it asks the
store for its artifact.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ocrscout.io import paths
from ocrscout.io.resume import ResumeMode, ResumeTracker
from ocrscout.io.rows import (
    LayoutRow,
    PageRow,
    RawRow,
    ResultRow,
    StageRow,
    VolumeRow,
)
from ocrscout.io.shards import ShardReader, ShardWriter
from ocrscout.types import Volume

# Artifact prefix → typed row model. Single source of truth for "which schema
# does this on-disk artifact use."
ARTIFACTS: dict[str, type[StageRow]] = {
    paths.PAGES: PageRow,
    paths.LAYOUT: LayoutRow,
    paths.RAW: RawRow,
    paths.TRAIN: ResultRow,
    paths.VOLUMES: VolumeRow,
}


class ParquetStore:
    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)

    def _row_type(self, prefix: str) -> type[StageRow]:
        try:
            return ARTIFACTS[prefix]
        except KeyError:
            raise ValueError(f"unknown stage artifact {prefix!r}") from None

    def writer(self, prefix: str, *, batch_size: int = 1000) -> ShardWriter:
        return ShardWriter(
            self.output_dir, self._row_type(prefix), prefix, batch_size=batch_size
        )

    def reader(self, prefix: str) -> ShardReader:
        return ShardReader(self.output_dir, self._row_type(prefix), prefix)

    def resume(self, prefix: str, mode: ResumeMode) -> ResumeTracker:
        return ResumeTracker.from_output(
            self.output_dir, self._row_type(prefix), prefix, mode
        )

    def has(self, prefix: str) -> bool:
        return bool(paths.find_shards(self.output_dir, prefix))

    def write_volumes(self, volumes: Iterable[Volume]) -> int:
        rows = [VolumeRow.from_volume(v) for v in volumes]
        if not rows:
            return 0
        with self.writer(paths.VOLUMES) as w:
            for row in rows:
                w.write(row)
        return len(rows)

"""Generic sharded parquet writer/reader over typed stage-row models.

``ShardWriter[RowT]`` and ``ShardReader[RowT]`` are the single IO primitive for
every stage — one writer, one reader, symmetric, generic over the row model.
They replace ``exports/stages.py``'s ``StageWriter`` and the bespoke pyarrow
projections scattered across the resume helpers.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

from datasets import Dataset

from ocrscout.io import paths
from ocrscout.io.rows import StageRow
from ocrscout.io.schema import features_for

RowT = TypeVar("RowT", bound=StageRow)

_DEFAULT_BATCH_SIZE = 1000


class ShardWriter(Generic[RowT]):
    """Append typed rows to ``data/<prefix>-NNNNN.parquet``.

    Buffers in memory and flushes a fresh shard every ``batch_size`` rows; the
    final partial batch flushes on ``close()``. Shard numbering continues after
    any shards already present, so a resumed stage never clobbers prior output.
    Thread-safe — OCR / detector pools write from many workers.
    """

    def __init__(
        self,
        output_dir: Path | str,
        row_type: type[RowT],
        prefix: str,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._row_type = row_type
        self._prefix = prefix
        self._features = features_for(row_type)
        self._batch_size = max(1, batch_size)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._index = paths.next_shard_index(self._output_dir, prefix)

    def write(self, row: RowT) -> None:
        with self._lock:
            self._buffer.append(row.to_arrow())
            full = len(self._buffer) >= self._batch_size
        if full:
            self._flush()

    def close(self) -> None:
        self._flush()

    def __enter__(self) -> ShardWriter[RowT]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _flush(self) -> None:
        with self._lock:
            batch = self._buffer
            self._buffer = []
            index = self._index
            self._index += 1
        if not batch:
            return
        d = paths.data_dir(self._output_dir)
        d.mkdir(parents=True, exist_ok=True)
        ds = Dataset.from_list(batch, features=self._features)
        ds.to_parquet(str(d / paths.shard_name(self._prefix, index)))


class ShardReader(Generic[RowT]):
    """Read typed rows from every ``data/<prefix>-*.parquet`` shard."""

    def __init__(
        self, output_dir: Path | str, row_type: type[RowT], prefix: str
    ) -> None:
        self._output_dir = Path(output_dir)
        self._row_type = row_type
        self._prefix = prefix

    @property
    def shards(self) -> list[Path]:
        return paths.find_shards(self._output_dir, self._prefix)

    def __iter__(self) -> Iterator[RowT]:
        import pyarrow.parquet as pq

        for shard in self.shards:
            table = pq.read_table(str(shard))
            for record in table.to_pylist():
                yield self._row_type.from_arrow(record)  # type: ignore[misc]

    def project(self, *columns: str) -> Iterator[dict[str, Any]]:
        """Yield only the named columns (cheap; for resume cursors)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        for shard in self.shards:
            try:
                table = pq.read_table(str(shard), columns=list(columns))
            except (OSError, pa.ArrowInvalid):
                continue
            yield from table.to_pylist()

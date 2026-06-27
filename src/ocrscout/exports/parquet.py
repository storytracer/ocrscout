"""ParquetExportAdapter — the registered ``parquet`` export target.

Thin adapter over the unified :mod:`ocrscout.io` layer, kept so the
``ExportAdapter`` registry contract (and any third-party export target) still
has a built-in. Row building, schema, sharding, and resume all live in ``io``
now — this just maps ``ExportRecord`` → ``ResultRow`` → the train shard writer.
The pipeline's NormalizeStage writes through ``io`` directly; this adapter is
for out-of-pipeline / third-party use of the contract.
"""

from __future__ import annotations

from pathlib import Path

from ocrscout.errors import ScoutError
from ocrscout.interfaces.export import ExportAdapter
from ocrscout.io import ParquetStore, ResultRow
from ocrscout.io.paths import DATA_DIR
from ocrscout.types import ExportRecord, Volume


def _resolve_output_dir(dest: str | Path) -> Path:
    """Accept ``output_dir``, ``output_dir/data``, or a legacy
    ``output_dir/data/<file>.parquet`` path and return ``output_dir``."""
    p = Path(dest)
    if p.suffix == ".parquet":
        return p.parent.parent
    if p.name == DATA_DIR:
        return p.parent
    return p


class ParquetExportAdapter(ExportAdapter):
    name = "parquet"

    def __init__(self, dest: str | Path | None = None, *, batch_size: int = 1000) -> None:
        self._batch_size = batch_size
        self._store: ParquetStore | None = None
        self._writer = None
        self._volumes: list[Volume] = []
        if dest is not None:
            self.open(str(dest))

    def open(self, dest: str) -> None:
        self._store = ParquetStore(_resolve_output_dir(dest))
        self._writer = self._store.writer("train", batch_size=self._batch_size)
        self._volumes = []

    def write(self, record: ExportRecord) -> None:
        if self._writer is None:
            raise ScoutError("ParquetExportAdapter.write() called before open()")
        self._writer.write(ResultRow.from_export_record(record))

    def write_volume(self, volume: Volume) -> None:
        self._volumes.append(volume)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._store is not None and self._volumes:
            self._store.write_volumes(self._volumes)
            self._volumes = []

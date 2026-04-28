"""ExportAdapter ABC: writes a stream of ExportRecord objects to a destination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.types import ExportRecord, Volume


class ExportAdapter(ABC):
    """Sink for a stream of ``ExportRecord`` rows.

    Implementations should buffer / batch internally and flush on ``close()``.
    Use as a context manager to guarantee close on exit.

    ``write_volume`` carries the per-volume bibliographic metadata that
    accompanies sources with an ``iter_volumes()`` implementation. Adapters
    that don't care about volumes can leave the default no-op.
    """

    name: ClassVar[str]

    @abstractmethod
    def open(self, dest: str) -> None: ...

    @abstractmethod
    def write(self, record: ExportRecord) -> None: ...

    def write_volume(self, volume: Volume) -> None:
        """Persist a single ``Volume`` row. Default: drop it."""
        return None

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> ExportAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

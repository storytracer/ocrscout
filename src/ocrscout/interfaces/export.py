"""ExportAdapter ABC: writes a stream of ExportRecord objects to a destination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.types import ExportRecord


class ExportAdapter(ABC):
    """Sink for a stream of ``ExportRecord`` rows.

    Implementations should buffer / batch internally and flush on ``close()``.
    Use as a context manager to guarantee close on exit.
    """

    name: ClassVar[str]

    @abstractmethod
    def open(self, dest: str) -> None: ...

    @abstractmethod
    def write(self, record: ExportRecord) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> ExportAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

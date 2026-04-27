"""SourceAdapter ABC: yields PageImage objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

from ocrscout.types import PageImage


class SourceAdapter(ABC):
    """Yields ``PageImage`` objects from a source (directory, dataset, IIIF, ...)."""

    name: ClassVar[str]

    @abstractmethod
    def iter_pages(self) -> Iterator[PageImage]:
        """Yield pages in iteration order."""

    def __len__(self) -> int:  # type: ignore[override]
        """Return the number of pages, or raise ``TypeError`` if unknown."""
        raise TypeError(f"{type(self).__name__} does not support len()")

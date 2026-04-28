"""SourceAdapter ABC: yields PageImage objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

from ocrscout.types import PageImage, Volume


class SourceAdapter(ABC):
    """Yields ``PageImage`` objects from a source (directory, dataset, IIIF, ...)."""

    name: ClassVar[str]

    @abstractmethod
    def iter_pages(self) -> Iterator[PageImage]:
        """Yield pages in iteration order."""

    def iter_volumes(self) -> Iterator[Volume]:
        """Yield the ``Volume`` objects referenced by ``iter_pages()``.

        Default: empty. Sources with a bibliographic-unit concept (BHL items,
        IA items, HathiTrust volumes, IIIF manifests, multi-page PDFs)
        override this so the run loop can write a ``volumes-NNNNN.parquet``
        sidecar joinable to the per-page results on ``volume_id``.
        """
        return iter(())

    def __len__(self) -> int:  # type: ignore[override]
        """Return the number of pages, or raise ``TypeError`` if unknown."""
        raise TypeError(f"{type(self).__name__} does not support len()")

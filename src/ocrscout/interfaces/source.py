"""SourceAdapter ABC: yields PageImage objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

from ocrscout.types import PageImage, Volume

if TYPE_CHECKING:
    from ocrscout.interfaces.source_action import SourceAction


class SourceAdapter(ABC):
    """Yields ``PageImage`` objects from a source (directory, dataset, IIIF, ...)."""

    name: ClassVar[str]

    actions: ClassVar[list[type["SourceAction"]]] = []
    """Source-specific admin verbs exposed under ``ocrscout source <name> <verb>``.

    See :class:`ocrscout.interfaces.source_action.SourceAction`. The
    universal ``info`` and ``clear`` verbs are added by the CLI driver
    automatically — adapters only declare their domain-specific verbs
    here.
    """

    cache_subdir: ClassVar[str | None] = None
    """Subdirectory name under ``~/.ocrscout/sources/`` for this source's
    cache + ``info.yaml``. Defaults to :attr:`name` when ``None``.
    """

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

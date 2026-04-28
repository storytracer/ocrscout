"""ReferenceAdapter ABC: returns ground-truth content for a page."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.types import PageImage, Reference


class ReferenceAdapter(ABC):
    """Returns ground-truth ``Reference`` for a ``PageImage`` (or ``None``).

    The adapter receives the full ``PageImage`` so it can reach for
    ``volume_id``, ``source_uri``, and ``extra`` when the lookup needs more
    than just ``page_id`` — e.g., BHL OCR URLs derive from both ``ItemID``
    (``volume_id``) and ``PageID`` (``page_id``).
    """

    name: ClassVar[str]

    @abstractmethod
    def get(self, page: PageImage) -> Reference | None: ...

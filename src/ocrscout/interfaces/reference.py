"""ReferenceAdapter ABC: returns ground-truth content for a page."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.types import Reference


class ReferenceAdapter(ABC):
    """Returns ground-truth ``Reference`` for a given ``page_id`` (or ``None``)."""

    name: ClassVar[str]

    @abstractmethod
    def get(self, page_id: str) -> Reference | None: ...

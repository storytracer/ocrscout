"""LayoutDetector ABC: emits typed regions for a page image.

Detectors describe; they do not crop. The layout-aware backend consumes the
returned regions, crops the source image per region, and dispatches one OCR
call per region.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.types import LayoutRegion, PageImage


class LayoutDetector(ABC):
    """Detects typed regions on a page image."""

    name: ClassVar[str]

    @abstractmethod
    def detect(self, page: PageImage) -> list[LayoutRegion]: ...

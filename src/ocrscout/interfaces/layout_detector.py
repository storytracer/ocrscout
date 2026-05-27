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

    def warm_up(self) -> None:
        """Eagerly load any deferred state (model weights, processor,
        torch context). Default no-op for detectors with no lazy state.

        Backends that fan-out detection across worker threads should
        call this **serially** on each instance before spawning workers.
        Concurrent first-loads of the same HF model race inside
        ``transformers``/``accelerate`` (meta tensors, partial init);
        warming up serially closes that window because every subsequent
        load hits the warm HF cache and an idempotent fast path."""

    @abstractmethod
    def detect(self, page: PageImage) -> list[LayoutRegion]: ...

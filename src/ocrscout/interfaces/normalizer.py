"""Normalizer ABC: converts a RawOutput into a DoclingDocument."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


class Normalizer(ABC):
    """Turns a model's raw output (markdown / DocTags / layout JSON) into a
    ``DoclingDocument``."""

    name: ClassVar[str]
    output_format: ClassVar[str]

    @abstractmethod
    def normalize(
        self, raw: RawOutput, page: PageImage, profile: ModelProfile
    ) -> DoclingDocument: ...

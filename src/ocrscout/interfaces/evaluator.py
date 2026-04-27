"""Evaluator ABC: scores a prediction against a reference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ocrscout.types import Reference

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


class Evaluator(ABC):
    """Compares a predicted ``DoclingDocument`` to a ``Reference`` and emits
    one or more named scores."""

    name: ClassVar[str]

    @abstractmethod
    def score(
        self, prediction: DoclingDocument, reference: Reference
    ) -> dict[str, float]: ...

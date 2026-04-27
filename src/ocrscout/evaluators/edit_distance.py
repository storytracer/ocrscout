"""EditDistanceEvaluator: character-level edit distance vs. plain-text reference.

Stub — arrives in phase 4 of the roadmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocrscout.interfaces.evaluator import Evaluator
from ocrscout.types import Reference

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


class EditDistanceEvaluator(Evaluator):
    name = "edit_distance"

    def score(
        self, prediction: DoclingDocument, reference: Reference
    ) -> dict[str, float]:
        raise NotImplementedError("EditDistanceEvaluator is not implemented in v0.")

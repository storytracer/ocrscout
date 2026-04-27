"""VlmJudgeEvaluator: pairwise VLM judging with ELO ratings.

Stub — arrives in phase 8 of the roadmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocrscout.interfaces.evaluator import Evaluator
from ocrscout.types import Reference

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


class VlmJudgeEvaluator(Evaluator):
    name = "vlm_judge"

    def score(
        self, prediction: DoclingDocument, reference: Reference
    ) -> dict[str, float]:
        raise NotImplementedError("VlmJudgeEvaluator is not implemented in v0.")

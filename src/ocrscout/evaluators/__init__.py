"""Evaluators: score predictions against references."""

from ocrscout.evaluators.edit_distance import EditDistanceEvaluator
from ocrscout.evaluators.vlm_judge import VlmJudgeEvaluator

__all__ = ["EditDistanceEvaluator", "VlmJudgeEvaluator"]

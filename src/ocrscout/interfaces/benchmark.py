"""Benchmark ABC: bundles a source, reference, and evaluator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.interfaces.evaluator import Evaluator
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.interfaces.source import SourceAdapter


class Benchmark(ABC):
    """A reproducible benchmark: a source dataset + ground truth + scoring."""

    name: ClassVar[str]

    @abstractmethod
    def source(self) -> SourceAdapter: ...

    @abstractmethod
    def reference(self) -> ReferenceAdapter: ...

    @abstractmethod
    def evaluator(self) -> Evaluator: ...

    def canonical_score(self, scores: dict[str, float]) -> float:
        """Reduce per-page scores to one headline number; default is mean."""
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

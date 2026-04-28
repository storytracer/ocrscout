"""Benchmark ABC: bundles a source, reference, and comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocrscout.interfaces.comparison import Comparison
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.interfaces.source import SourceAdapter


class Benchmark(ABC):
    """A reproducible benchmark: a source dataset + reference + comparisons.

    Note "reference" — not "ground truth". A benchmark measures agreement
    against whatever artifact the reference adapter yields, which may itself
    be OCR (legacy or otherwise). Use ``Reference.provenance`` to interpret
    whether a comparison number reflects accuracy or consistency.
    """

    name: ClassVar[str]

    @abstractmethod
    def source(self) -> SourceAdapter: ...

    @abstractmethod
    def reference(self) -> ReferenceAdapter: ...

    @abstractmethod
    def comparisons(self) -> list[Comparison]: ...

    def canonical_summary(self, summaries: dict[str, float]) -> float:
        """Reduce per-page summary metrics to one headline number; default is mean."""
        if not summaries:
            return 0.0
        return sum(summaries.values()) / len(summaries)

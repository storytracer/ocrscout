"""Abstract base classes for ocrscout components."""

from ocrscout.interfaces.backend import ModelBackend
from ocrscout.interfaces.benchmark import Benchmark
from ocrscout.interfaces.comparison import (
    BaselineView,
    Comparison,
    ComparisonRenderer,
    ComparisonResult,
    PredictionView,
)
from ocrscout.interfaces.export import ExportAdapter
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.interfaces.reporter import Reporter
from ocrscout.interfaces.source import SourceAdapter

__all__ = [
    "BaselineView",
    "Benchmark",
    "Comparison",
    "ComparisonRenderer",
    "ComparisonResult",
    "ExportAdapter",
    "ModelBackend",
    "Normalizer",
    "PredictionView",
    "ReferenceAdapter",
    "Reporter",
    "SourceAdapter",
]

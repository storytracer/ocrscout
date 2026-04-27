"""Abstract base classes for ocrscout components."""

from ocrscout.interfaces.backend import ModelBackend
from ocrscout.interfaces.benchmark import Benchmark
from ocrscout.interfaces.evaluator import Evaluator
from ocrscout.interfaces.export import ExportAdapter
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.interfaces.reporter import Reporter
from ocrscout.interfaces.source import SourceAdapter

__all__ = [
    "Benchmark",
    "Evaluator",
    "ExportAdapter",
    "ModelBackend",
    "Normalizer",
    "ReferenceAdapter",
    "Reporter",
    "SourceAdapter",
]

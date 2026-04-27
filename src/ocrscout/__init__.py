"""ocrscout — scout frontier OCR models on your data, your hardware, your terms."""

from ocrscout._version import __version__
from ocrscout.errors import (
    BackendError,
    IntrospectionError,
    NormalizerError,
    PipelineError,
    ProfileError,
    RegistryError,
    ScoutError,
)
from ocrscout.interfaces import (
    Benchmark,
    Evaluator,
    ExportAdapter,
    ModelBackend,
    Normalizer,
    ReferenceAdapter,
    Reporter,
    SourceAdapter,
)
from ocrscout.metrics import MetricsCollector
from ocrscout.profile import ModelProfile, dump_profile, load_profile, resolve
from ocrscout.registry import Registry, registry
from ocrscout.types import (
    AdapterRef,
    BackendInvocation,
    ExportRecord,
    PageImage,
    PipelineConfig,
    RawOutput,
    Reference,
    RunMetrics,
)

__all__ = [
    "AdapterRef",
    "BackendError",
    "BackendInvocation",
    "Benchmark",
    "Evaluator",
    "ExportAdapter",
    "ExportRecord",
    "IntrospectionError",
    "MetricsCollector",
    "ModelBackend",
    "ModelProfile",
    "Normalizer",
    "NormalizerError",
    "PageImage",
    "PipelineConfig",
    "PipelineError",
    "ProfileError",
    "RawOutput",
    "Reference",
    "ReferenceAdapter",
    "Registry",
    "RegistryError",
    "Reporter",
    "RunMetrics",
    "ScoutError",
    "SourceAdapter",
    "__version__",
    "dump_profile",
    "load_profile",
    "registry",
    "resolve",
]

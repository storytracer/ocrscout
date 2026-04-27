"""Smoke test: every public symbol resolves; no import cycles."""

from __future__ import annotations


def test_top_level_imports() -> None:
    import ocrscout  # noqa: F401
    from ocrscout import (
        AdapterRef,
        BackendError,
        BackendInvocation,
        Benchmark,
        Evaluator,
        ExportAdapter,
        ExportRecord,
        IntrospectionError,
        MetricsCollector,
        ModelBackend,
        ModelProfile,
        Normalizer,
        NormalizerError,
        PageImage,
        PipelineConfig,
        PipelineError,
        ProfileError,
        RawOutput,
        Reference,
        ReferenceAdapter,
        Registry,
        RegistryError,
        Reporter,
        RunMetrics,
        ScoutError,
        SourceAdapter,
        __version__,
        dump_profile,
        load_profile,
        registry,
        resolve,
    )

    assert __version__
    assert isinstance(registry, Registry)
    # Sanity-check that ABCs are real classes:
    for cls in (
        Benchmark, Evaluator, ExportAdapter, ModelBackend, Normalizer,
        ReferenceAdapter, Reporter, SourceAdapter,
    ):
        assert isinstance(cls, type)
    for cls in (
        AdapterRef, BackendInvocation, ExportRecord, MetricsCollector,
        ModelProfile, PageImage, PipelineConfig, RawOutput, Reference, RunMetrics,
    ):
        assert isinstance(cls, type)
    # Errors:
    assert issubclass(ProfileError, ScoutError)
    assert issubclass(BackendError, ScoutError)
    assert issubclass(NormalizerError, ScoutError)
    assert issubclass(IntrospectionError, ScoutError)
    assert issubclass(PipelineError, ScoutError)
    assert issubclass(RegistryError, ScoutError)
    # Functions:
    assert callable(load_profile) and callable(dump_profile) and callable(resolve)


def test_subpackage_imports() -> None:
    """Importing every subpackage shouldn't trigger import errors or cycles."""
    import ocrscout.backends  # noqa: F401
    import ocrscout.benchmarks  # noqa: F401
    import ocrscout.cli  # noqa: F401
    import ocrscout.evaluators  # noqa: F401
    import ocrscout.exports  # noqa: F401
    import ocrscout.interfaces  # noqa: F401
    import ocrscout.normalizers  # noqa: F401
    import ocrscout.pipeline  # noqa: F401
    import ocrscout.references  # noqa: F401
    import ocrscout.reporters  # noqa: F401
    import ocrscout.sources  # noqa: F401
    import ocrscout.sync  # noqa: F401

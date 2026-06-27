"""Registry construction helpers shared by stages.

Centralizes the "resolve a name from the registry, then instantiate" pattern
that was copy-pasted across the CLI commands, so stages stay declarative.
"""

from __future__ import annotations

from ocrscout.interfaces.comparison import Comparison
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.profile import ModelProfile, resolve
from ocrscout.registry import registry
from ocrscout.types import AdapterRef


def build_source(ref: AdapterRef) -> SourceAdapter:
    return registry.get("sources", ref.name)(**ref.args)


def build_reference(ref: AdapterRef | None) -> ReferenceAdapter | None:
    if ref is None:
        return None
    return registry.get("references", ref.name)(**ref.args)


def resolve_comparisons(
    names: list[str] | None, *, has_reference: bool
) -> list[Comparison]:
    """Materialize ``Comparison`` instances.

    ``None`` + reference present → all registered; ``None`` without a reference
    → none; ``[]`` → none (explicit opt-out); otherwise the named ones.
    """
    if names is None:
        names = list(registry.list("comparisons")) if has_reference else []
    return [registry.get("comparisons", n)() for n in names]


def resolve_normalizer(profile: ModelProfile, override: str | None = None) -> Normalizer:
    return registry.get("normalizers", override or profile.normalizer)()


def resolve_profile(model: str) -> ModelProfile:
    return resolve(model)

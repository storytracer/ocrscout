"""Exception hierarchy for ocrscout."""

from __future__ import annotations


class ScoutError(Exception):
    """Base class for all ocrscout errors."""


class ProfileError(ScoutError):
    """Raised when a model profile cannot be loaded, parsed, or resolved."""


class ProfileNotFound(ProfileError):
    """Raised when no curated profile exists for the requested model name."""


class RegistryError(ScoutError):
    """Raised when registry lookup or registration fails."""


class BackendError(ScoutError):
    """Raised when a model backend fails to prepare or execute."""


class ManagedServerError(ScoutError):
    """Raised when ocrscout's managed multi-server orchestration fails.

    Covers GPU-budget rejection, vllm-serve startup timeout/failure, and
    LiteLLM proxy startup failure. Teardown of any partially-started subprocesses
    happens before this is raised.
    """


class NormalizerError(ScoutError):
    """Raised when normalizing a raw model output to DoclingDocument fails."""


class IntrospectionError(ScoutError):
    """Raised when an HF script cannot be statically introspected."""


class PipelineError(ScoutError):
    """Raised when a pipeline configuration is invalid or execution fails."""

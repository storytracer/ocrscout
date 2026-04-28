"""Model profiles: schema, YAML I/O, and curated-only resolution.

Profiles are hand-curated YAMLs shipped at ``src/ocrscout/profiles/``. The
upstream HF uv-scripts/ocr scripts are reference material we read with
Claude Code when authoring a new profile — they are no longer executed.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ocrscout.errors import ProfileError, ProfileNotFound

# ``source`` selects a backend by registry name. We keep the type as a plain
# ``str`` so adding a backend doesn't require touching this schema — the
# registry validates the name at run time when it resolves the backend class.
ProfileSource = str
OutputFormat = Literal["markdown", "doctags", "layout_json", "docling_document"]

DEFAULT_VLLM_ENGINE_ARGS: dict[str, Any] = {
    # Trim CUDA graph capture sizes — vLLM's default is [1..512], but with
    # backend_args.concurrent_requests defaulting to ~16 we will never see
    # batches above ~32. Capturing fewer graphs saves ~1 GiB of GPU memory
    # (absorbed by KV cache) and ~2s of startup per engine. Override
    # per-profile by setting ``cudagraph_capture_sizes`` in
    # ``vllm_engine_args``.
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32],
    # Every shipped OCR VLM ships custom modeling code on the Hub and
    # requires this. Override to ``false`` only for a profile that pins a
    # model with a stock architecture.
    "trust_remote_code": True,
}


def effective_vllm_engine_args(profile: ModelProfile) -> dict[str, Any]:
    """Profile's ``vllm_engine_args`` merged on top of ``DEFAULT_VLLM_ENGINE_ARGS``."""
    return {**DEFAULT_VLLM_ENGINE_ARGS, **(profile.vllm_engine_args or {})}


class ModelProfile(BaseModel):
    """Describes how to invoke an OCR model and what to expect back.

    All profiles are curated. Fields are grouped by concern:

    Identity
        ``name``, ``model_id``, ``model_size``, ``upstream_script``
    Output
        ``output_format``, ``normalizer``, ``has_bboxes``, ``has_layout``,
        ``category_mapping``
    Prompting
        ``prompt_templates`` (mode → template), ``preferred_prompt_mode``,
        ``chat_template_content_format``
    vLLM
        ``vllm_engine_args``, ``sampling_args``, ``vllm_version``,
        ``server_url``
    Free-form
        ``backend_args`` (anything a custom backend might need),
        ``metadata`` (informational notes, prompt-mode catalogs, etc.)
    """

    model_config = ConfigDict(extra="forbid")

    # Identity
    name: str
    source: ProfileSource
    model_id: str
    model_size: str | None = None
    upstream_script: str | None = None
    """Informational reference, e.g. ``"uv-scripts/ocr/dots-mocr.py"``.

    Never executed; recorded so ``ocrscout introspect`` can re-fetch the source
    when revisiting the curation.
    """

    # Output
    output_format: OutputFormat
    normalizer: str
    has_bboxes: bool = False
    has_layout: bool = False
    category_mapping: dict[str, str] = Field(default_factory=dict)

    # Prompting
    prompt_templates: dict[str, str] = Field(default_factory=dict)
    """Mode name → prompt string. The chosen mode is ``preferred_prompt_mode``.

    Values may contain ``{width}``/``{height}`` placeholders that the backend
    fills in per page.
    """
    preferred_prompt_mode: str | None = None
    chat_template_content_format: str | None = None
    """Passed to ``vllm.LLM.chat(chat_template_content_format=...)``.

    Some models (notably ``dots.mocr``) need ``"string"``; default ``None``
    lets vLLM auto-detect.
    """

    # vLLM
    vllm_engine_args: dict[str, Any] = Field(default_factory=dict)
    """kwargs passed to ``vllm.LLM(...)``.

    Examples: ``max_model_len``, ``gpu_memory_utilization``,
    ``trust_remote_code``, ``dtype``, ``limit_mm_per_prompt``.
    """
    sampling_args: dict[str, Any] = Field(default_factory=dict)
    """kwargs passed to ``vllm.SamplingParams(...)``.

    Examples: ``temperature``, ``top_p``, ``max_tokens``.
    """
    vllm_version: str = ">=0.15.1"
    """Version specifier used by ``uv run --with "vllm{vllm_version}"`` for
    subprocess mode. Ignored in server mode."""
    server_url: str | None = None
    """Optional OpenAI-compatible base URL (e.g. ``http://localhost:8000/v1``).

    If set (or the ``OCRSCOUT_VLLM_URL`` env var is set at run time), the
    backend prefers HTTP server mode over spawning a vLLM subprocess.
    """

    # Free-form
    backend_args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Layout-aware orchestration. Consumed by ``layout_chat`` and any future
    # layout-aware backend. Optional and additive — empty/None for every
    # existing profile.
    layout_detector: str | None = None
    """Registry name of a ``LayoutDetector`` (e.g. ``"pp-doclayout-v3"``).

    Required when ``source == "layout_chat"``.
    """
    layout_detector_args: dict[str, Any] = Field(default_factory=dict)
    """Constructor kwargs for the layout detector (device, score_threshold,
    revision, etc.). Detector-specific."""
    prompt_mode_per_category: dict[str, str] = Field(default_factory=dict)
    """Detector-native category label → prompt-template key.

    Keyed on the **detector's** raw category (not the docling-mapped one) so
    the lookup happens before ``category_mapping`` is applied. Categories not
    listed fall back to ``preferred_prompt_mode``.
    """

    @model_validator(mode="after")
    def _validate_layout_chat(self) -> ModelProfile:
        if self.source == "layout_chat":
            if not self.layout_detector:
                raise ValueError(
                    "source='layout_chat' requires layout_detector to be set"
                )
            if self.output_format != "layout_json":
                raise ValueError(
                    "source='layout_chat' requires output_format='layout_json' "
                    f"(got {self.output_format!r})"
                )
            if self.normalizer != "layout_json":
                raise ValueError(
                    "source='layout_chat' requires normalizer='layout_json' "
                    f"(got {self.normalizer!r})"
                )
        return self


def load_profile(path: str | Path) -> ModelProfile:
    """Load a profile from a YAML file."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ProfileError(f"cannot read profile file {p}: {e}") from e
    return load_profile_from_str(text, source_hint=str(p))


def load_profile_from_str(text: str, *, source_hint: str = "<string>") -> ModelProfile:
    """Parse a YAML string into a ModelProfile."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ProfileError(f"invalid YAML in {source_hint}: {e}") from e
    if not isinstance(data, dict):
        raise ProfileError(f"profile {source_hint} must be a YAML mapping")
    try:
        return ModelProfile.model_validate(data)
    except ValidationError as e:
        raise ProfileError(f"invalid profile in {source_hint}: {e}") from e


def dump_profile(profile: ModelProfile, path: str | Path) -> None:
    """Write a profile to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = profile.model_dump(mode="json", exclude_defaults=False)
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _curated_path(name: str) -> Path | None:
    try:
        anchor = files("ocrscout.profiles") / f"{name}.yaml"
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    if not anchor.is_file():
        return None
    return Path(str(anchor))


def resolve(name: str) -> ModelProfile:
    """Resolve a curated profile by name.

    Curated profiles are the only kind that exist; if the YAML isn't shipped
    in ``src/ocrscout/profiles/``, raise ``ProfileNotFound`` with a hint
    pointing at ``ocrscout introspect`` for drafting a new one.
    """
    curated = _curated_path(name)
    if curated is None:
        available = list_curated()
        raise ProfileNotFound(
            f"No curated profile for {name!r}. "
            f"Available: {available or '(none)'}.\n"
            f"To draft a new profile from an upstream HF script, run "
            f"`ocrscout introspect {name}` and refine the output before "
            f"saving it under src/ocrscout/profiles/."
        )
    return load_profile(curated)


def list_curated() -> list[str]:
    """List the names of all curated profiles shipped with the package."""
    try:
        root = files("ocrscout.profiles")
    except ModuleNotFoundError:
        return []
    names: list[str] = []
    for entry in root.iterdir():
        n = entry.name
        if n.endswith(".yaml") and not n.startswith("_"):
            names.append(n[:-5])
    return sorted(names)

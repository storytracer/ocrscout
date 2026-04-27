"""Model profiles: schema, YAML I/O, and three-tier resolution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocrscout.errors import ProfileError

ProfileSource = Literal["hf_scripts", "openai_api", "tesseract", "docling", "custom"]
OutputFormat = Literal["markdown", "doctags", "layout_json", "docling_document"]
ProfileTier = Literal["curated", "auto", "fallback"]


class ModelProfile(BaseModel):
    """Describes how to invoke an OCR model and what to expect back.

    Three tiers exist (see ``resolve``): curated YAMLs ship with the package,
    auto YAMLs are written by ``ocrscout sync``, and a synthetic fallback is
    constructed when neither is found.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: ProfileSource
    script: str | None = None
    repo: str | None = None
    model_id: str
    output_format: OutputFormat
    normalizer: str
    preferred_prompt_mode: str | None = None
    has_bboxes: bool = False
    has_layout: bool = False
    model_size: str | None = None
    category_mapping: dict[str, str] = Field(default_factory=dict)
    backend_args: dict[str, Any] = Field(default_factory=dict)
    # Raw extra args inserted between `uv run` and the script path when invoking
    # subprocess backends. Use to pin packages declared by the upstream script's
    # PEP 723 block, add an index URL, change the index strategy, etc. — e.g.
    # `["--with", "vllm==0.11.2", "--extra-index-url",
    #   "https://download.pytorch.org/whl/cu129", "--index-strategy",
    #   "unsafe-best-match"]`.
    uv_args: list[str] = Field(default_factory=list)
    # Extra environment variables merged into the subprocess for hf_scripts
    # backends. Use for upstream-provided bypass knobs like
    # FLASHINFER_DISABLE_VERSION_CHECK that would otherwise need a per-script
    # patch.
    env: dict[str, str] = Field(default_factory=dict)
    tier: ProfileTier = "curated"
    metadata: dict[str, Any] = Field(default_factory=dict)


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


def _auto_path(name: str, *, cache_dir: Path | None = None) -> Path:
    from ocrscout.sync.cache import profiles_cache_dir

    base = cache_dir if cache_dir is not None else profiles_cache_dir()
    return base / f"{name}.yaml"


def resolve(name: str, *, cache_dir: Path | None = None) -> ModelProfile:
    """Resolve a profile by name across all three tiers.

    Order: curated (shipped with the package) → auto (in user cache) → fallback
    (synthesized: assumes markdown output, model_id == name).
    """
    curated = _curated_path(name)
    if curated is not None:
        profile = load_profile(curated)
        # tier defaults to "curated"; honor whatever the file says.
        return profile

    auto = _auto_path(name, cache_dir=cache_dir)
    if auto.is_file():
        profile = load_profile(auto)
        return profile

    return ModelProfile(
        name=name,
        source="hf_scripts",
        model_id=name,
        output_format="markdown",
        normalizer="markdown",
        tier="fallback",
    )


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

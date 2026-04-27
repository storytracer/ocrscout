"""Profile sync: introspect HF uv-scripts/ocr and write auto-generated YAMLs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ocrscout.profile import ModelProfile, dump_profile, list_curated
from ocrscout.sync.cache import profiles_cache_dir, scripts_cache_dir
from ocrscout.sync.fetch import fetch_scripts
from ocrscout.sync.introspect import HfScriptInfo, introspect_hf_script

log = logging.getLogger(__name__)

__all__ = [
    "HfScriptInfo",
    "SyncResult",
    "introspect_hf_script",
    "profile_from_script",
    "sync_profiles",
]


@dataclass
class SyncResult:
    scripts_dir: Path
    written: list[Path]
    skipped: list[str]  # names shadowed by curated profiles
    errored: list[tuple[Path, str]]


def sync_profiles(
    *,
    scripts_dir: str | Path | None = None,
    fetch: bool = True,
    cache_dir: Path | None = None,
    revision: str | None = None,
) -> SyncResult:
    """Refresh auto-generated profiles in ``~/.cache/ocrscout/profiles/``.

    If ``scripts_dir`` is given, scripts are read from there. Otherwise (and if
    ``fetch=True``), the upstream ``uv-scripts/ocr`` is snapshotted into
    ``~/.cache/ocrscout/uv-scripts-ocr/`` first.
    """
    if scripts_dir is not None:
        src = Path(scripts_dir)
    elif fetch:
        src = fetch_scripts(revision=revision)
    else:
        src = scripts_cache_dir()
    if not src.is_dir():
        raise FileNotFoundError(f"scripts directory not found: {src}")

    out_dir = Path(cache_dir) if cache_dir is not None else profiles_cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    curated = set(list_curated())

    written: list[Path] = []
    skipped: list[str] = []
    errored: list[tuple[Path, str]] = []

    for script_path in sorted(src.glob("*.py")):
        try:
            info = introspect_hf_script(script_path)
        except Exception as e:  # noqa: BLE001
            errored.append((script_path, str(e)))
            log.warning("failed to introspect %s: %s", script_path, e)
            continue

        profile = profile_from_script(info)
        if profile.name in curated:
            skipped.append(profile.name)
            continue

        target = out_dir / f"{profile.name}.yaml"
        dump_profile(profile, target)
        written.append(target)

    return SyncResult(scripts_dir=src, written=written, skipped=skipped, errored=errored)


def profile_from_script(info: HfScriptInfo) -> ModelProfile:
    """Build an auto-tier ``ModelProfile`` from a static introspection result."""
    name = info.path.stem
    output_format = _guess_output_format(info)
    normalizer = output_format

    return ModelProfile(
        name=name,
        source="hf_scripts",
        script=f"uv-scripts/ocr/{info.path.name}",
        repo="uv-scripts/ocr",
        model_id=info.default_model or f"unknown/{name}",
        output_format=output_format,
        normalizer=normalizer,
        preferred_prompt_mode=_pick_prompt_mode(info),
        has_bboxes=output_format in {"layout_json", "doctags"},
        has_layout=output_format in {"layout_json", "doctags"},
        tier="auto",
        metadata={
            "prompt_modes": sorted(info.prompt_templates.keys()),
            "dependencies": list(info.dependencies),
            "requires_python": info.requires_python,
            "imports": sorted(info.imports),
        },
    )


def _guess_output_format(info: HfScriptInfo) -> str:
    keys = set(info.prompt_templates)
    if "doctags" in keys or "docling_core" in info.imports:
        return "doctags"
    if "layout-all" in keys or "layout-only" in keys or "layout" in keys:
        return "layout_json"
    return "markdown"


def _pick_prompt_mode(info: HfScriptInfo) -> str | None:
    keys = info.prompt_templates
    if not keys:
        return None
    for preferred in ("layout-all", "doctags", "ocr", "general"):
        if preferred in keys:
            return preferred
    return next(iter(keys))

"""`ocrscout introspect` — draft a curated profile YAML from an upstream script.

This is a Claude-Code-assisted workflow: the command does the boring
mechanical part (download the script, parse argparse defaults, extract
``PROMPT_TEMPLATES``, list dependencies via ``ast.parse``) and prints a YAML
skeleton with TODO markers. The human (or Claude Code reading the upstream
source alongside) fills in the vLLM engine knobs, sampling params, and
output format.

The script is **never executed**.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from ocrscout.cli import app
from ocrscout.errors import IntrospectionError
from ocrscout.sync.cache import scripts_cache_dir
from ocrscout.sync.fetch import fetch_scripts
from ocrscout.sync.introspect import HfScriptInfo, introspect_hf_script


@app.command("introspect")
def introspect(
    name: str = typer.Argument(
        ...,
        help="Script stem (e.g. 'dots-ocr') or path to a local .py file.",
    ),
    no_fetch: bool = typer.Option(
        False,
        "--no-fetch",
        help="Don't pull from the Hub; expect the script in the local cache.",
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="HF Hub revision/commit to pin when fetching."
    ),
) -> None:
    """Print a draft curated YAML for an upstream uv-scripts/ocr script.

    Pipe to a file under ``src/ocrscout/profiles/<name>.yaml`` and refine
    the TODO markers by reading the upstream script source.
    """
    script_path = _locate_script(name=name, fetch=not no_fetch, revision=revision)
    try:
        info = introspect_hf_script(script_path)
    except IntrospectionError as e:
        typer.echo(f"Failed to introspect {script_path}: {e}", err=True)
        raise typer.Exit(code=1) from e

    draft = _draft_yaml(info)
    typer.echo(draft)
    typer.echo(
        f"\n# Drafted from {script_path}.\n"
        f"# Refine the TODOs and save under src/ocrscout/profiles/{info.path.stem}.yaml.",
        err=True,
    )


def _locate_script(*, name: str, fetch: bool, revision: str | None) -> Path:
    p = Path(name)
    if p.suffix == ".py" and p.is_file():
        return p

    cache = scripts_cache_dir()
    target = cache / f"{name}.py"
    if target.is_file() and not fetch:
        return target
    if fetch:
        fetch_scripts(revision=revision)
        if target.is_file():
            return target
    raise typer.BadParameter(
        f"Cannot locate script {name!r}. Tried {target} and (if --no-fetch was "
        "not given) a Hub fetch."
    )


def _draft_yaml(info: HfScriptInfo) -> str:
    """Build a draft YAML string with TODO markers for the human to resolve."""
    name = info.path.stem
    output_format = _guess_output_format(info)
    is_vllm = "vllm" in info.imports or any("vllm" in d for d in info.dependencies)
    source = "vllm" if is_vllm else "custom"

    draft: dict = {
        "name": name,
        "source": source,
        "model_id": info.default_model or f"TODO-set-model-id  # script default unknown",
        "model_size": "TODO  # e.g. 3B",
        "upstream_script": f"uv-scripts/ocr/{info.path.name}",
        "output_format": output_format,
        "normalizer": output_format if output_format != "docling_document" else "passthrough",
        "has_bboxes": output_format in {"layout_json", "doctags"},
        "has_layout": output_format in {"layout_json", "doctags"},
    }

    if info.prompt_templates:
        preferred = _pick_prompt_mode(info)
        draft["preferred_prompt_mode"] = preferred or "TODO"
        draft["prompt_templates"] = dict(info.prompt_templates)

    if is_vllm:
        draft["chat_template_content_format"] = (
            "TODO  # set to 'string' for dots.mocr-style models, else null"
        )
        draft["vllm_engine_args"] = {
            "trust_remote_code": True,
            "max_model_len": "TODO  # script default; bump to ~24000 for dense pages",
            "gpu_memory_utilization": 0.8,
        }
        draft["sampling_args"] = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": "TODO  # match max_model_len",
        }
        draft["vllm_version"] = ">=0.15.0"
    else:
        draft["backend_args"] = {}

    draft["metadata"] = {
        "notes": (
            f"Drafted by `ocrscout introspect {name}` from "
            f"uv-scripts/ocr/{info.path.name}. Resolve TODO markers by "
            "reading the upstream script source. Dependencies declared by "
            f"the upstream script's PEP 723 block: {sorted(info.dependencies)}."
        ),
    }

    if info.prompt_templates:
        draft["metadata"]["prompt_modes"] = sorted(info.prompt_templates.keys())

    return yaml.safe_dump(draft, sort_keys=False, allow_unicode=True, width=10**9)


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

"""`ocrscout submit` — fire-and-forget pipeline submission.

Builds a ``PipelineConfig`` from the user's flags and hands it to the
active ``Runner.submit(...)``. Returns immediately with a job id;
``ocrscout status`` and ``ocrscout logs <job_id>`` track progress
afterwards, and the launching shell can be closed without affecting the
in-flight work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.errors import RunnerError, ScoutError
from ocrscout.exports.layout import parquet_dest
from ocrscout.log import setup_logging
from ocrscout.registry import registry
from ocrscout.types import AdapterRef, PipelineConfig

log = logging.getLogger(__name__)


@app.command("submit")
def submit(
    source: str | None = typer.Option(
        None, "--source", "-s",
        help="Image source: a local directory, a fsspec URL "
             "(s3://..., gs://..., https://..., hf://...), or an HF Hub "
             "dataset id (org/name). Ignored by adapters with a fixed "
             "corpus (e.g. `bhl`).",
    ),
    source_name: str = typer.Option(
        "hf_dataset", "--source-name",
        help="Source adapter name.",
    ),
    source_arg: list[str] = typer.Option(
        None, "--source-arg",
        help="Repeatable key=value kwargs for the source adapter.",
    ),
    models: str | None = typer.Option(
        None, "--models", "-m",
        help="Comma-separated profile names. Defaults to whatever the "
             "active runner was launched with.",
    ),
    output_dir: Path = typer.Option(
        Path("./data/results"), "--output-dir", "-o",
        help="Where the worker writes data/*.parquet + progress.json.",
    ),
    pages: int | None = typer.Option(
        None, "--pages",
        help="Take a random subset of N pages (no-op if N >= total).",
    ),
    start_idx: int | None = typer.Option(
        None, "--start-idx",
        help="Start index for source partitioning (worker takes pages "
             "[start_idx, end_idx)). For SkyPilot/HF this is set "
             "automatically per worker rank.",
    ),
    end_idx: int | None = typer.Option(
        None, "--end-idx",
        help="End index for source partitioning (exclusive).",
    ),
    seed: int = typer.Option(42, "--seed"),
    reference: str | None = typer.Option(None, "--reference"),
    reference_path: Path | None = typer.Option(None, "--reference-path"),
    comparisons: str | None = typer.Option(None, "--comparisons"),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip page ids already present in <output-dir>/progress.json.",
    ),
    export: str = typer.Option("parquet", "--export"),
    runner: str | None = typer.Option(
        None, "--runner",
        help="Runner to submit through. Defaults to whatever's in state.yaml.",
    ),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
) -> None:
    """Submit a fire-and-forget pipeline run to the active runner."""
    setup_logging(verbosity=verbose, quiet=quiet)

    state = state_mod.read_state()
    runner_name = runner or (state.runner if state else None) or "local"
    if state is None or state.runner != runner_name:
        raise typer.BadParameter(
            f"No active {runner_name!r} runner. Run `ocrscout launch "
            f"--models ...` first."
        )

    model_list = (
        [m.strip() for m in models.split(",") if m.strip()]
        if models
        else list(state.models)
    )
    if not model_list:
        raise typer.BadParameter(
            "no models specified and the active runner has none"
        )

    source_args: dict[str, Any] = _parse_source_args(source_arg or [])
    if source_name == "hf_dataset":
        source_args["path"] = source
    if pages is not None:
        source_args.setdefault("sample", pages)
        source_args.setdefault("seed", seed)
    if start_idx is not None:
        source_args["start_idx"] = start_idx
    if end_idx is not None:
        source_args["end_idx"] = end_idx

    comparison_names = _parse_comparisons_flag(comparisons)

    cfg = PipelineConfig(
        name="submit",
        source=AdapterRef(name=source_name, args=source_args),
        reference=(
            AdapterRef(
                name=reference,
                args={"root": str(reference_path)} if reference_path else {},
            )
            if reference
            else None
        ),
        comparisons=comparison_names,
        models=model_list,
        export=AdapterRef(
            name=export, args={"dest": str(parquet_dest(output_dir))}
        ),
        sample=pages,
        output_dir=output_dir,
    )

    try:
        runner_cls = registry.get("runners", runner_name)
    except ScoutError as e:
        raise typer.BadParameter(str(e)) from e

    try:
        handle = runner_cls().submit(config=cfg, resume=resume)
    except RunnerError as e:
        log.error("Submit failed: %s", e)
        raise typer.Exit(code=1) from e

    rprint(f"[bold green]submitted[/bold green] job=[cyan]{handle.job_id}[/cyan]")
    rprint(f"  output: [cyan]{handle.output_dir}[/cyan]")
    rprint(
        "  follow: [cyan]ocrscout logs "
        f"{handle.job_id}[/cyan]   status: [cyan]ocrscout status[/cyan]"
    )


def _parse_source_args(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise typer.BadParameter(
                f"--source-arg {raw!r} must be key=value"
            )
        key, _, value = raw.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _parse_comparisons_flag(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) == 1 and parts[0].lower() == "none":
        return []
    return parts

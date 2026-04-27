"""`ocrscout run` — one-shot: pick model(s), point at images, get scores."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.types import AdapterRef, PipelineConfig


@app.command("run")
def run(
    source: Path = typer.Option(
        ..., "--source", "-s", help="Directory of images to OCR."
    ),
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated profile names."
    ),
    reference: str | None = typer.Option(
        None, "--reference", help="Reference adapter name (e.g. plain_text)."
    ),
    reference_path: Path | None = typer.Option(
        None, "--reference-path", help="Path passed to the reference adapter."
    ),
    sample: int | None = typer.Option(
        None, "--sample", help="Limit to first N pages."
    ),
    benchmark: str | None = typer.Option(
        None, "--benchmark", help="Run a registered benchmark instead of --source."
    ),
    output_dir: Path = typer.Option(
        Path("./ocrscout-results"), "--output-dir", "-o",
        help="Where to write results and the generated pipeline.yaml.",
    ),
    export: str = typer.Option(
        "parquet", "--export", help="Export adapter name."
    ),
) -> None:
    """Run multiple OCR models against a source and emit a comparison."""
    if benchmark is None and source is None:
        raise typer.BadParameter("--source or --benchmark is required")

    cfg = PipelineConfig(
        name="run",
        source=AdapterRef(name="local", args={"path": str(source)}),
        reference=(
            AdapterRef(
                name=reference,
                args={"root": str(reference_path)} if reference_path else {},
            )
            if reference
            else None
        ),
        models=[m.strip() for m in models.split(",") if m.strip()],
        export=AdapterRef(name=export, args={"dest": str(output_dir / "results.parquet")}),
        sample=sample,
        output_dir=output_dir,
    )

    rprint("[bold]Resolved pipeline config:[/bold]")
    rprint(cfg.model_dump(mode="json"))

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_yaml = output_dir / "pipeline.yaml"
    pipeline_yaml.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    rprint(f"[dim]Wrote {pipeline_yaml} for reproducibility.[/dim]")

    if benchmark:
        rprint(f"[yellow](stub) would run benchmark {benchmark!r}[/yellow]")
    rprint(
        "[yellow](stub) backend execution arrives in phase 2 of the roadmap; "
        "see CLAUDE.md.[/yellow]"
    )

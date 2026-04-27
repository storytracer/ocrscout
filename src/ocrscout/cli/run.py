"""`ocrscout run` — one-shot: pick model(s), point at images, get scores."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import typer
import yaml
from rich import print as rprint
from rich.table import Table

from ocrscout import registry
from ocrscout.cli import app
from ocrscout.errors import BackendError, NormalizerError
from ocrscout.metrics import MetricsCollector
from ocrscout.profile import resolve
from ocrscout.types import AdapterRef, ExportRecord, PipelineConfig

log = logging.getLogger(__name__)


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
    text: bool = typer.Option(
        True, "--text/--no-text",
        help="Also write a `text/<page>.<model>.md` rendering of every result "
             "alongside the parquet (uses DoclingDocument.export_to_markdown).",
    ),
    server_url: str | None = typer.Option(
        None, "--server-url",
        help="OpenAI-compatible vLLM server URL (e.g. http://localhost:8000/v1). "
             "When set, vllm-source profiles use HTTP server mode instead of "
             "spawning a `uv run --with vllm` subprocess. Equivalent to setting "
             "OCRSCOUT_VLLM_URL.",
    ),
) -> None:
    """Run multiple OCR models against a source and emit a comparison."""
    if benchmark is None and source is None:
        raise typer.BadParameter("--source or --benchmark is required")

    if server_url:
        # Set the env var so backends pick it up (and so it propagates to
        # any subprocess invoked downstream).
        os.environ["OCRSCOUT_VLLM_URL"] = server_url
        rprint(f"[dim]Using vLLM server at {server_url}[/dim]")

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

    text_dir = (output_dir / "text") if text else None
    _execute(cfg, text_dir=text_dir)


def _execute(cfg: PipelineConfig, *, text_dir: Path | None = None) -> None:
    source_cls = registry.get("sources", cfg.source.name)
    source = source_cls(**cfg.source.args)
    pages = list(source.iter_pages())
    if cfg.sample is not None:
        pages = pages[: cfg.sample]
    page_by_id = {p.page_id: p for p in pages}
    rprint(f"[bold]Loaded {len(pages)} page(s) from {cfg.source.args.get('path')!r}.[/bold]")
    if not pages:
        from ocrscout.sources.local import _SUPPORTED_SUFFIXES

        rprint(
            f"[red]No images found at {cfg.source.args.get('path')!r}. "
            f"LocalSourceAdapter recognizes {sorted(_SUPPORTED_SUFFIXES)}.[/red]"
        )
        raise typer.Exit(code=1)

    if cfg.reference is not None:
        ref_cls = registry.get("references", cfg.reference.name)
        _ = ref_cls(**cfg.reference.args)  # reference adapter is not yet wired into the loop

    exporter_cls = registry.get("exports", cfg.export.name)
    exporter = exporter_cls()
    exporter.open(cfg.export.args["dest"])

    metrics = MetricsCollector(pipeline_id=cfg.name)
    summary_rows: list[tuple[str, int, int, float]] = []

    try:
        for model_name in cfg.models:
            ok, failed, run_seconds = _run_one_model(
                model_name=model_name,
                pages=pages,
                page_by_id=page_by_id,
                normalizer_overrides=cfg.normalizer_overrides,
                exporter=exporter,
                metrics=metrics,
                text_dir=text_dir,
            )
            summary_rows.append((model_name, ok, failed, run_seconds))
    finally:
        exporter.close()
        metrics.finish()

    _print_summary(summary_rows, dest=cfg.export.args["dest"])
    if text_dir is not None:
        rprint(f"[dim]Wrote per-(page, model) markdown to {text_dir}[/dim]")


def _run_one_model(
    *,
    model_name: str,
    pages: list,
    page_by_id: dict,
    normalizer_overrides: dict,
    exporter,
    metrics: MetricsCollector,
    text_dir: Path | None = None,
) -> tuple[int, int, float]:
    try:
        profile = resolve(model_name)
        backend = registry.get("backends", profile.source)()
        normalizer_name = normalizer_overrides.get(model_name, profile.normalizer)
        normalizer = registry.get("normalizers", normalizer_name)()
    except Exception as e:
        rprint(f"[red]{model_name}: setup failed: {e}[/red]")
        return 0, len(pages), 0.0

    rprint(
        f"[bold cyan]\n=== {model_name} "
        f"(backend={profile.source}, normalizer={normalizer_name}) ===[/bold cyan]"
    )

    try:
        with metrics.stage(f"{model_name}.prepare"):
            inv = backend.prepare(profile, pages)
        with metrics.stage(f"{model_name}.run"):
            raws = list(backend.run(inv))
    except BackendError as e:
        rprint(f"[red]{model_name} backend failed: {e}[/red]")
        return 0, len(pages), metrics.stage_seconds.get(f"{model_name}.run", 0.0)

    prepare_seconds = metrics.stage_seconds.get(f"{model_name}.prepare", 0.0)
    run_seconds_total = metrics.stage_seconds.get(f"{model_name}.run", 0.0)
    pages_attempted = max(len(raws), 1)
    run_seconds_per_page = run_seconds_total / pages_attempted

    ok = 0
    failed = 0
    for raw in raws:
        page = page_by_id.get(raw.page_id)
        if page is None:
            failed += 1
            log.warning("model %s returned unknown page_id %r", model_name, raw.page_id)
            continue
        if raw.error:
            failed += 1
            log.warning("model %s page %s reported error: %s", model_name, raw.page_id, raw.error)
            continue
        try:
            t_norm0 = time.perf_counter()
            with metrics.stage(f"{model_name}.normalize"):
                doc = normalizer.normalize(raw, page, profile)
            normalize_seconds = time.perf_counter() - t_norm0
        except NormalizerError as e:
            failed += 1
            log.warning("normalizer failed for %s/%s: %s", model_name, raw.page_id, e)
            continue

        item_count, text_length, markdown = _doc_stats(doc)

        record = ExportRecord(
            page=page,
            document=doc,
            raw=raw,
            metrics={
                "model": model_name,
                "prepare_seconds": round(prepare_seconds, 4),
                "run_seconds_total": round(run_seconds_total, 4),
                "run_seconds_per_page": round(run_seconds_per_page, 4),
                "normalize_seconds": round(normalize_seconds, 4),
                "tokens": raw.tokens,
                "item_count": item_count,
                "text_length": text_length,
            },
            scores={},
        )
        exporter.write(record)
        ok += 1

        if text_dir is not None:
            _write_markdown(text_dir, page.page_id, model_name, markdown)

    metrics.add_pages(ok=ok, failed=failed)
    run_seconds = metrics.stage_seconds.get(f"{model_name}.run", 0.0)
    rprint(
        f"[green]{model_name}:[/green] {ok}/{ok + failed} ok in "
        f"{run_seconds:.1f}s"
    )
    return ok, failed, run_seconds


def _print_summary(
    summary_rows: list[tuple[str, int, int, float]], *, dest: str
) -> None:
    table = Table(title="ocrscout run — per-model summary")
    table.add_column("model", style="bold")
    table.add_column("pages_ok", justify="right")
    table.add_column("pages_failed", justify="right")
    table.add_column("run_seconds", justify="right")
    table.add_column("s/page", justify="right")
    for name, ok, failed, secs in summary_rows:
        per_page = f"{secs / ok:.2f}" if ok > 0 else "—"
        table.add_row(name, str(ok), str(failed), f"{secs:.1f}", per_page)
    rprint(table)
    rprint(f"[dim]Wrote {dest}[/dim]")


def _doc_stats(doc) -> tuple[int, int, str]:
    """Return (item_count, text_length, markdown) for a DoclingDocument.

    `item_count` sums texts + pictures + tables; `text_length` is the total
    character count of all text items; `markdown` is the
    ``export_to_markdown()`` rendering. Each is computed defensively so a
    missing attribute degrades to a zero/empty rather than crashing the run.
    """
    try:
        text_length = sum(len(t.text or "") for t in (doc.texts or []))
    except Exception:  # noqa: BLE001
        text_length = 0
    try:
        item_count = (
            len(doc.texts or [])
            + len(doc.pictures or [])
            + len(doc.tables or [])
        )
    except Exception:  # noqa: BLE001
        item_count = 0
    try:
        markdown = doc.export_to_markdown()
    except Exception as e:  # noqa: BLE001
        log.warning("export_to_markdown failed: %s", e)
        markdown = ""
    return item_count, text_length, markdown


def _write_markdown(text_dir: Path, page_id: str, model_name: str, markdown: str) -> None:
    """Write `<text_dir>/<sanitized_page_stem>.<model>.md`.

    page_id may contain slashes (subdir-relative) and the original suffix
    (e.g. `.jp2`); we strip the suffix and replace path separators with `_`
    so the filename is flat and predictable.
    """
    stem = Path(page_id).stem.replace("/", "_").replace("\\", "_")
    target = text_dir / f"{stem}.{model_name}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")

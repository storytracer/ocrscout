"""`ocrscout run` — one-shot: pick model(s), point at images, get scores."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import typer
import yaml
from rich import print as rprint
from rich.table import Table

from ocrscout import registry
from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.cost import recorder as cost_recorder
from ocrscout.errors import BackendError, NormalizerError, ProfileNotFound, RunnerError
from ocrscout.exports.layout import parquet_dest
from ocrscout.interfaces.comparison import (
    BaselineView,
    Comparison,
    ComparisonResult,
    PredictionView,
    aggregate_summaries,
)
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.log import VERBOSE, setup_logging
from ocrscout.metrics import MetricsCollector
from ocrscout.profile import resolve
from ocrscout.runners.local import LocalRunner
from ocrscout.state import GpuConfig
from ocrscout.types import AdapterRef, ExportRecord, PipelineConfig

log = logging.getLogger(__name__)


@app.command("run")
def run(
    source: str | None = typer.Option(
        None, "--source", "-s",
        help="Image source: a local directory, a fsspec URL "
             "(s3://bucket/prefix/, gs://..., https://...), or a "
             "HuggingFace Hub dataset id (org/name). Used by the default "
             "`hf_dataset` adapter; ignored by source adapters with a "
             "fixed corpus (e.g. `bhl`).",
    ),
    source_name: str = typer.Option(
        "hf_dataset", "--source-name",
        help="Source adapter name. Default `hf_dataset` covers local "
             "dirs, fsspec URLs, and HF Hub IDs via `--source`. Use `bhl` "
             "to sample the Biodiversity Heritage Library S3 bucket "
             "(305K volumes, 67M pages); `--source` is then ignored. "
             "Per-adapter knobs (e.g. `languages`, `pages_per_volume`, "
             "`volumes`) go through `--source-arg key=value`.",
    ),
    source_arg: list[str] = typer.Option(
        None, "--source-arg",
        help="Repeatable `key=value` kwargs forwarded to the source "
             "adapter constructor. Values are parsed as JSON when "
             "possible, plain string otherwise. Example: "
             "`--source-arg 'languages=[\"eng\"]' --source-arg pages_per_volume=4 --source-arg volumes=10`.",
    ),
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated profile names."
    ),
    reference: str | None = typer.Option(
        None, "--reference", help="Reference adapter name (e.g. plain_text, bhl_ocr)."
    ),
    reference_path: Path | None = typer.Option(
        None, "--reference-path", help="Path passed to the reference adapter."
    ),
    comparisons: str | None = typer.Option(
        None, "--comparisons",
        help="Comma-separated comparison names to run per (page, model) "
             "(e.g. `text`, `text,document`, `text,document,layout`). When "
             "omitted and a reference adapter is configured, all built-in "
             "comparisons whose `requires` set is satisfied by the data "
             "fire by default. Pass `none` to skip comparisons entirely.",
    ),
    sample: int | None = typer.Option(
        None, "--sample",
        help="Take a random subset of N pages (no-op if N >= total). "
             "Sources that can list cheaply (local dirs, fsspec URLs, HF "
             "Hub) only fetch the sampled subset — they do not download "
             "the whole prefix. Use --seed for reproducibility.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="RNG seed for --sample. Defaults to 42 so re-running the same "
             "command picks the same pages — pin to a different integer for "
             "a different reproducible subset, or override per-run as needed.",
    ),
    benchmark: str | None = typer.Option(
        None, "--benchmark", help="Run a registered benchmark instead of --source."
    ),
    output_dir: Path = typer.Option(
        Path("./data/results"), "--output-dir", "-o",
        help="Where to write results and the generated pipeline.yaml.",
    ),
    export: str = typer.Option(
        "parquet", "--export", help="Export adapter name."
    ),
    gpu_budget: float = typer.Option(
        0.85, "--gpu-budget",
        help="Maximum total GPU memory the local stack will collectively "
             "claim, as a fraction of total VRAM. Per-model KV cache is set "
             "by `vllm_engine_args.kv_cache_memory_bytes` in each profile; "
             "this flag bounds the sum + per-model overhead.",
    ),
    base_port: int = typer.Option(
        8000, "--base-port",
        help="First TCP port for vLLM serves (LocalRunner only).",
    ),
    proxy_port: int = typer.Option(
        4000, "--proxy-port",
        help="LiteLLM proxy port (LocalRunner only).",
    ),
    keep_up: bool = typer.Option(
        False, "--keep-up",
        help="After the pipeline finishes, leave the runner stack up so "
             "subsequent `ocrscout submit` calls don't pay another launch "
             "cost. Default: tear it down (ephemeral run).",
    ),
    parallel_models: int | None = typer.Option(
        None, "--parallel-models", "-P",
        help="Number of models to run concurrently. Default: 1 (sequential), "
             "which gives each model the full GPU and produces uncontended "
             "per-model s/page numbers for benchmarking. Total wall-clock is "
             "~equivalent to running them in parallel on a single GPU since "
             "the GPU is the bottleneck either way. Raise this only if you "
             "have separate GPUs per model or genuinely want concurrent "
             "execution at the cost of comparable per-model timings.",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True,
        help="Increase log verbosity. -v shows VERBOSE-level events (timings, "
             "URLs); -vv shows DEBUG (subprocess argvs, full GPU process listings, "
             "module paths).",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet",
        help="Suppress informational logging; only warnings/errors and the "
             "final summary table are shown.",
    ),
) -> None:
    """Run multiple OCR models against a source and emit a comparison."""
    setup_logging(verbosity=verbose, quiet=quiet)
    if benchmark is None and source is None and source_name == "hf_dataset":
        raise typer.BadParameter(
            "--source is required when using the default `hf_dataset` "
            "source adapter. Pass --source-name to select a different "
            "adapter (e.g. `bhl`), or use --benchmark."
        )

    source_args: dict[str, Any] = _parse_source_args(source_arg or [])
    if source_name == "hf_dataset":
        # Only the default fsspec/HF-Hub adapter consumes `--source` as a
        # path; corpus-bound adapters like `bhl` ignore it (warned below).
        source_args["path"] = source
    elif source is not None:
        log.warning(
            "--source is ignored by the %r source adapter (its corpus "
            "is fixed). Use --source-arg for adapter-specific kwargs.",
            source_name,
        )
    if sample is not None:
        # Sampling is pushed down into the adapter so cheap pre-listing
        # avoids downloading the whole prefix just to discard most of it.
        source_args.setdefault("sample", sample)
        source_args.setdefault("seed", seed)
    comparison_names = _parse_comparisons_flag(comparisons)
    cfg = PipelineConfig(
        name="run",
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
        models=[m.strip() for m in models.split(",") if m.strip()],
        export=AdapterRef(name=export, args={"dest": str(parquet_dest(output_dir))}),
        sample=sample,
        output_dir=output_dir,
    )

    # Concise one-liner at default verbosity; dump the full dict only at -v+.
    log.info(
        "ocrscout run: %d model(s) [%s] · source=%s · output=%s",
        len(cfg.models),
        ",".join(cfg.models),
        cfg.source.args.get("path", "?"),
        cfg.export.args.get("dest", "?"),
    )
    log.log(
        VERBOSE,
        "Resolved pipeline config:\n%s",
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_yaml = output_dir / "pipeline.yaml"
    pipeline_yaml.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    log.log(VERBOSE, "Wrote pipeline.yaml: %s", pipeline_yaml)

    if benchmark:
        log.warning("(stub) would run benchmark %r", benchmark)

    # Resolve parallelism. Default 1 (sequential): the GPU is the bottleneck
    # in single-GPU setups, so concurrent model execution just splits compute
    # and bandwidth and yields slowed, contended per-model s/page numbers
    # that aren't useful for benchmarking. Sequential gives each model the
    # full GPU, produces honest per-model numbers, and total wall-clock is
    # ~equivalent. Override with -P > 1 if you actually have per-model GPUs.
    if parallel_models is None:
        parallel_models = 1
    parallel_models = max(1, min(parallel_models, len(cfg.models)))
    if parallel_models > 1:
        log.info("Running up to %d model(s) concurrently", parallel_models)
    else:
        log.info("Running %d model(s) sequentially", len(cfg.models))

    run_pipeline(
        cfg,
        parallel_models=parallel_models,
        base_port=base_port,
        proxy_port=proxy_port,
        gpu_budget=gpu_budget,
        keep_up=keep_up,
    )


def run_pipeline(
    cfg: PipelineConfig,
    *,
    parallel_models: int = 1,
    base_port: int = 8000,
    proxy_port: int = 4000,
    gpu_budget: float = 0.85,
    keep_up: bool = False,
    resume: bool = False,
) -> None:
    """Launch the required runner (if any), run the pipeline, tear down.

    Shared by ``ocrscout run`` and ``ocrscout apply`` (and any future
    command that takes a fully-resolved ``PipelineConfig``). Auto-launches
    a ``LocalRunner`` when any profile has ``runtime != cpu`` and no
    matching runner is already up. Profiles with ``runtime: cpu`` only
    skip the launch entirely (Docling, Tesseract run in-process).
    """
    try:
        profiles = [resolve(name) for name in cfg.models]
    except ProfileNotFound as e:
        raise typer.BadParameter(str(e)) from e

    needs_proxy = any(p.runtime != "cpu" for p in profiles)
    if not needs_proxy:
        log.info("All profiles are runtime=cpu; running CPU-side backends in-process.")
        _execute(cfg, parallel_models=parallel_models, resume=resume)
        return

    runner = LocalRunner()
    existing_state = state_mod.read_state()
    reused_existing = (
        existing_state is not None and existing_state.runner == "local"
    )

    try:
        handle = runner.launch(
            models=cfg.models,
            base_port=base_port,
            proxy_port=proxy_port,
            gpu_budget=gpu_budget,
            # ``ocrscout run`` is one-shot: tie the stack's lifetime to
            # this process via PR_SET_PDEATHSIG + atexit. ``--keep-up``
            # opts back into the daemonised persistent path so the stack
            # outlives the invocation for follow-up `submit`s. Reuse of
            # an existing persistent stack short-circuits before the
            # ephemeral path runs (see LocalRunner.launch).
            persistent=keep_up,
        )
        os.environ["OCRSCOUT_VLLM_URL"] = handle.proxy_url
        if reused_existing:
            log.info("Reusing already-running local stack")
    except RunnerError as e:
        log.error("Runner launch failed: %s", e)
        raise typer.Exit(code=1) from e

    try:
        _execute(cfg, parallel_models=parallel_models, resume=resume)
    finally:
        if not keep_up and not reused_existing:
            try:
                runner.down()
            except Exception as e:  # noqa: BLE001
                log.warning("Runner teardown reported error: %s", e)


def _execute(
    cfg: PipelineConfig,
    *,
    parallel_models: int = 1,
    resume: bool = False,
) -> None:
    # Resolve the active GPU context once and stamp it onto every row of
    # this run. Env-var overrides (set by SkyPilot/HF job YAMLs) take
    # precedence over ``~/.ocrscout/config.yaml`` so remote workers
    # carry their own pricing context into the output Parquet.
    gpu_cfg = state_mod.read_config().gpu
    log.log(
        VERBOSE,
        "GPU context: type=%s provider=%s cost_per_hour=%.4f",
        gpu_cfg.type, gpu_cfg.provider, gpu_cfg.cost_per_hour,
    )

    source_cls = registry.get("sources", cfg.source.name)
    source = source_cls(**cfg.source.args)
    pages_iter = source.iter_pages()
    if cfg.sample is not None:
        pages = list(itertools.islice(pages_iter, cfg.sample))
    else:
        pages = list(pages_iter)
    if resume:
        # Drop page ids already completed in a prior run (recorded by the
        # parquet adapter into <output_dir>/progress.json). Cheap dedup —
        # the loaded set is bounded by the deterministic sample size.
        from ocrscout.exports.parquet import ProgressTracker as _PT
        progress = _PT(cfg.output_dir)
        done = progress.completed_page_ids()
        if done:
            before = len(pages)
            pages = [p for p in pages if p.page_id not in done]
            log.info(
                "Resume: skipping %d of %d pages already in progress.json",
                before - len(pages), before,
            )
    page_by_id = {p.page_id: p for p in pages}
    source_label = cfg.source.args.get("path") or cfg.source.name
    log.info("Loaded %d page(s) from %r", len(pages), source_label)
    if not pages:
        log.error(
            "No images found from source %r (args=%s). For `hf_dataset` "
            "this means the `imagefolder` builder found no images and the "
            "path is not an HF Hub dataset id. For other adapters, check "
            "their --source-arg filters.",
            cfg.source.name, cfg.source.args,
        )
        raise typer.Exit(code=1)

    reference_adapter: ReferenceAdapter | None = None
    if cfg.reference is not None:
        ref_cls = registry.get("references", cfg.reference.name)
        reference_adapter = ref_cls(**cfg.reference.args)
        log.info("Reference adapter: %s", cfg.reference.name)

    active_comparisons = _resolve_comparisons(
        cfg.comparisons, has_reference=reference_adapter is not None,
    )
    if active_comparisons:
        log.info(
            "Comparisons: %s",
            ", ".join(c.name for c in active_comparisons),
        )

    exporter_cls = registry.get("exports", cfg.export.name)
    exporter = exporter_cls()
    exporter.open(cfg.export.args["dest"])
    # Volumes flow through the source's optional iter_volumes() — the run
    # loop materializes them up-front so the per-volume sidecar parquet
    # exists even if a backend later fails on a particular page.
    volume_count = 0
    for volume in source.iter_volumes():
        exporter.write_volume(volume)
        volume_count += 1
    if volume_count:
        log.info("Loaded %d volume(s) from source", volume_count)

    metrics = MetricsCollector(pipeline_id=cfg.name)
    # results[model_name] -> ModelRunResult. Dict so the summary table can
    # walk cfg.models in order even when parallel runs finish out-of-order.
    results: dict[str, _ModelRunResult] = {}

    def _run(name: str) -> tuple[str, _ModelRunResult]:
        result = _run_one_model(
            model_name=name,
            pages=pages,
            page_by_id=page_by_id,
            normalizer_overrides=cfg.normalizer_overrides,
            exporter=exporter,
            reference_adapter=reference_adapter,
            comparisons=active_comparisons,
            metrics=metrics,
            gpu=gpu_cfg,
        )
        return name, result

    try:
        if parallel_models <= 1:
            for model_name in cfg.models:
                _, run_result = _run(model_name)
                results[model_name] = run_result
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_models) as ex:
                futures = [ex.submit(_run, name) for name in cfg.models]
                for fut in concurrent.futures.as_completed(futures):
                    name, run_result = fut.result()
                    results[name] = run_result
    finally:
        exporter.close()
        metrics.finish()

    summary_rows = [
        (m, results[m]) for m in cfg.models if m in results
    ]
    _print_summary(summary_rows, dest=cfg.export.args["dest"])


@dataclass
class _ModelRunResult:
    ok: int
    failed: int
    run_seconds: float
    comparison_results: list[ComparisonResult] = dc_field(default_factory=list)


def _run_one_model(
    *,
    model_name: str,
    pages: list,
    page_by_id: dict,
    normalizer_overrides: dict,
    exporter,
    reference_adapter: ReferenceAdapter | None,
    comparisons: list[Comparison],
    metrics: MetricsCollector,
    gpu: GpuConfig,
) -> _ModelRunResult:
    try:
        profile = resolve(model_name)
        backend = registry.get("backends", profile.backend)()
        normalizer_name = normalizer_overrides.get(model_name, profile.normalizer)
        normalizer = registry.get("normalizers", normalizer_name)()
    except Exception as e:
        log.error("[%s] setup failed: %s", model_name, e)
        return _ModelRunResult(ok=0, failed=len(pages), run_seconds=0.0)

    log.info(
        "[%s] starting (backend=%s, runtime=%s, normalizer=%s)",
        model_name, profile.backend, profile.runtime, normalizer_name,
    )

    try:
        with metrics.stage(f"{model_name}.prepare"):
            inv = backend.prepare(profile, pages)
        with metrics.stage(f"{model_name}.run"):
            raws = list(backend.run(inv))
    except BackendError as e:
        log.error("[%s] backend failed: %s", model_name, e)
        return _ModelRunResult(
            ok=0,
            failed=len(pages),
            run_seconds=metrics.stage_seconds.get(f"{model_name}.run", 0.0),
        )

    prepare_seconds = metrics.stage_seconds.get(f"{model_name}.prepare", 0.0)
    run_seconds_total = metrics.stage_seconds.get(f"{model_name}.run", 0.0)
    pages_attempted = max(len(raws), 1)
    run_seconds_per_page = run_seconds_total / pages_attempted

    ok = 0
    failed = 0
    accumulated_comparisons: list[ComparisonResult] = []
    for raw in raws:
        page = page_by_id.get(raw.page_id)
        if page is None:
            failed += 1
            log.warning("model %s returned unknown page_id %r", model_name, raw.page_id)
            continue
        if raw.error:
            failed += 1
            log.warning("model %s page %s reported error: %s", model_name, page.file_id, raw.error)
            continue
        try:
            t_norm0 = time.perf_counter()
            with metrics.stage(f"{model_name}.normalize"):
                doc = normalizer.normalize(raw, page, profile)
            normalize_seconds = time.perf_counter() - t_norm0
        except NormalizerError as e:
            failed += 1
            log.warning("normalizer failed for %s/%s: %s", model_name, page.file_id, e)
            continue

        item_count, text_length, markdown, text = _doc_stats(doc)

        reference = None
        if reference_adapter is not None:
            try:
                reference = reference_adapter.get(page)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "reference adapter failed for %s/%s: %s",
                    model_name, page.file_id, e,
                )

        page_comparisons: dict[str, ComparisonResult] = {}
        if reference is not None and comparisons:
            view_pred = PredictionView(
                page_id=page.page_id,
                label=model_name,
                text=text,
                document=doc,
            )
            view_base = BaselineView(
                page_id=page.page_id,
                label=reference_adapter.name if reference_adapter else "reference",
                text=reference.text,
                document=reference.document,
                provenance=reference.provenance,
            )
            for cmp in comparisons:
                try:
                    result = cmp.compare(view_pred, view_base)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "comparison %s failed for %s/%s: %s",
                        cmp.name, model_name, page.file_id, e,
                    )
                    continue
                if result is not None:
                    page_comparisons[cmp.name] = result
                    accumulated_comparisons.append(result)

        # Close the cost recorder's page (if the backend opened one — only
        # LiteLLM-routed backends do). Compute gpu_time_cost from elapsed
        # and the active GpuConfig so the column is comparable across
        # GPUs without re-deriving downstream.
        cost_metrics = cost_recorder.close_page(page.page_id)
        if cost_metrics is not None:
            elapsed_s = cost_metrics.elapsed_seconds
            gpu_time_cost = elapsed_s / 3600.0 * gpu.cost_per_hour
            input_tokens = cost_metrics.input_tokens
            output_tokens = cost_metrics.output_tokens
            litellm_cost = cost_metrics.litellm_cost
        else:
            elapsed_s = None
            gpu_time_cost = None
            input_tokens = None
            output_tokens = None
            litellm_cost = None

        record = ExportRecord(
            page=page,
            model=model_name,
            document=doc,
            raw=raw,
            reference=reference,
            markdown=markdown,
            text=text,
            metrics={
                "prepare_seconds": round(prepare_seconds, 4),
                "run_seconds_total": round(run_seconds_total, 4),
                "run_seconds_per_page": round(run_seconds_per_page, 4),
                "normalize_seconds": round(normalize_seconds, 4),
                "tokens": raw.tokens,
                "item_count": item_count,
                "text_length": text_length,
            },
            comparisons=page_comparisons,
            gpu_type=gpu.type,
            provider=gpu.provider,
            cost_per_hour=gpu.cost_per_hour,
            elapsed_seconds=elapsed_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            litellm_cost=litellm_cost,
            gpu_time_cost=gpu_time_cost,
        )
        exporter.write(record)
        ok += 1

    metrics.add_pages(ok=ok, failed=failed)
    run_seconds = metrics.stage_seconds.get(f"{model_name}.run", 0.0)
    log.info(
        "[%s] done: %d/%d ok in %.1fs",
        model_name, ok, ok + failed, run_seconds,
    )
    return _ModelRunResult(
        ok=ok,
        failed=failed,
        run_seconds=run_seconds,
        comparison_results=accumulated_comparisons,
    )


def _parse_source_args(raw: list[str]) -> dict[str, Any]:
    """Parse repeated `--source-arg key=value` strings into a kwargs dict.

    Values are JSON-parsed when possible (so list/dict/int/bool literals
    work) and fall back to plain strings otherwise.
    """
    parsed: dict[str, Any] = {}
    for entry in raw:
        if "=" not in entry:
            raise typer.BadParameter(
                f"--source-arg expects 'key=value', got {entry!r}"
            )
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"--source-arg has empty key in {entry!r}")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def _parse_comparisons_flag(raw: str | None) -> list[str] | None:
    """Decode the ``--comparisons`` flag into a list of comparison names.

    ``None`` / unset → default-on (resolved later from registry + reference
    state). ``"none"`` → explicit opt-out (returns ``[]``). Otherwise a
    comma-separated whitelist of registered comparison names.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if raw.lower() == "none":
        return []
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if not names:
        return None
    return names


def _resolve_comparisons(
    names: list[str] | None, *, has_reference: bool
) -> list[Comparison]:
    """Materialize Comparison instances from the names list.

    ``None`` + reference present → all built-in comparisons. ``[]`` →
    no comparisons (explicit opt-out). Otherwise the named comparisons.
    Names not in the registry raise a typer.BadParameter so typos surface
    early.
    """
    if names is None:
        if not has_reference:
            return []
        names = list(registry.list("comparisons"))
    elif not names:
        return []
    out: list[Comparison] = []
    for n in names:
        try:
            cls = registry.get("comparisons", n)
        except Exception as e:
            raise typer.BadParameter(
                f"unknown comparison {n!r}; available: "
                f"{registry.list('comparisons')}"
            ) from e
        out.append(cls())
    return out


def _print_summary(
    summary_rows: list[tuple[str, _ModelRunResult]], *, dest: str
) -> None:
    # Summary table is the *deliverable*, not status — always rprinted, never
    # gated by log level. Quiet mode still shows it.
    table = Table(title="ocrscout run — per-model summary")
    table.add_column("model", style="bold")
    table.add_column("pages_ok", justify="right")
    table.add_column("pages_failed", justify="right")
    table.add_column("run_seconds", justify="right")
    table.add_column("s/page", justify="right")

    # Dynamic comparison columns: union of every summary key any model
    # produced. A model with no result for a given key renders ``—``.
    extra_keys: list[str] = sorted({
        key
        for _, run_result in summary_rows
        for r in run_result.comparison_results
        for key in (r.summary or {})
    })
    for key in extra_keys:
        table.add_column(f"{key}_avg", justify="right")

    for name, run_result in summary_rows:
        per_page = (
            f"{run_result.run_seconds / run_result.ok:.2f}" if run_result.ok > 0 else "—"
        )
        row = [
            name,
            str(run_result.ok),
            str(run_result.failed),
            f"{run_result.run_seconds:.1f}",
            per_page,
        ]
        if extra_keys:
            agg = aggregate_summaries(run_result.comparison_results)
            for key in extra_keys:
                if key in agg:
                    mean, n = agg[key]
                    row.append(_format_summary_metric(key, mean, n))
                else:
                    row.append("—")
        table.add_row(*row)
    rprint(table)
    log.info("Wrote %s", dest)


def _format_summary_metric(key: str, mean: float, n: int) -> str:
    # Similarity-like metrics live in 0..100 territory and read better as
    # percentages with one decimal; CER/WER are in 0..1 and read better as
    # three-decimal proportions; everything else falls back to a generic
    # two-decimal float.
    if key == "similarity":
        return f"{mean:.1f}%  (n={n})"
    if key in ("cer", "wer"):
        return f"{mean:.3f}  (n={n})"
    return f"{mean:.2f}  (n={n})"


def _doc_stats(doc) -> tuple[int, int, str, str]:
    """Return (item_count, text_length, markdown, text) for a DoclingDocument.

    `item_count` sums texts + pictures + tables; `text_length` is the total
    character count of all text items; `markdown` is the
    ``export_to_markdown()`` rendering kept for human reading and viewer
    display; `text` is ``export_to_text()`` flattened for direct comparison
    against ``reference_text``. Each is computed defensively so a missing
    attribute degrades to a zero/empty rather than crashing the run.
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
    try:
        text = doc.export_to_text()
    except Exception as e:  # noqa: BLE001
        log.warning("export_to_text failed: %s", e)
        text = ""
    return item_count, text_length, markdown, text

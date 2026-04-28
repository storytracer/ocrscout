"""`ocrscout run` — one-shot: pick model(s), point at images, get scores."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from rich import print as rprint
from rich.table import Table

from ocrscout import registry
from ocrscout.cli import app
from ocrscout.errors import BackendError, ManagedServerError, NormalizerError, ProfileNotFound
from ocrscout.exports.layout import parquet_dest
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.log import VERBOSE, setup_logging
from ocrscout.managed import managed_servers
from ocrscout.metrics import MetricsCollector
from ocrscout.profile import resolve
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
             "Per-adapter knobs (e.g. `languages`, `max_pages_per_volume`) "
             "go through `--source-arg key=value`.",
    ),
    source_arg: list[str] = typer.Option(
        None, "--source-arg",
        help="Repeatable `key=value` kwargs forwarded to the source "
             "adapter constructor. Values are parsed as JSON when "
             "possible, plain string otherwise. Example: "
             "`--source-arg 'languages=[\"eng\"]' --source-arg max_pages_per_volume=4`.",
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
        Path("./ocrscout-results"), "--output-dir", "-o",
        help="Where to write results and the generated pipeline.yaml.",
    ),
    export: str = typer.Option(
        "parquet", "--export", help="Export adapter name."
    ),
    server_url: str | None = typer.Option(
        None, "--server-url",
        help="OpenAI-compatible vLLM server URL (e.g. http://localhost:8000/v1). "
             "When set, vllm-source profiles use HTTP server mode instead of "
             "spawning a `uv run --with vllm` subprocess. Equivalent to setting "
             "OCRSCOUT_VLLM_URL. Mutually exclusive with --managed.",
    ),
    managed: bool = typer.Option(
        False, "--managed",
        help="Spin up one vllm-serve per vllm-source profile (and a LiteLLM "
             "proxy when there are 2+) for the duration of this run, then tear "
             "it down. For long-lived servers, use `ocrscout serve` instead.",
    ),
    gpu_budget: float = typer.Option(
        0.85, "--gpu-budget",
        help="Maximum total GPU memory the managed stack will collectively "
             "claim, as a fraction of total VRAM. Per-model KV cache is set "
             "by `vllm_engine_args.kv_cache_memory_bytes` in each profile; "
             "this flag bounds the sum + per-model overhead. Only used with "
             "--managed.",
    ),
    base_port: int = typer.Option(
        8000, "--base-port",
        help="First port for managed vllm-serves. Only used with --managed.",
    ),
    proxy_port: int = typer.Option(
        4000, "--proxy-port",
        help="LiteLLM proxy port (only used when --managed and N>=2).",
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

    if managed and server_url:
        raise typer.BadParameter(
            "--managed and --server-url are mutually exclusive: --managed "
            "spawns its own server stack and sets the env var itself."
        )

    if server_url:
        # Set the env var so backends pick it up (and so it propagates to
        # any subprocess invoked downstream).
        os.environ["OCRSCOUT_VLLM_URL"] = server_url
        log.info("vLLM server: %s", server_url)

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

    if managed:
        try:
            profiles = [resolve(name) for name in cfg.models]
        except ProfileNotFound as e:
            raise typer.BadParameter(str(e)) from e
        try:
            with managed_servers(
                profiles,
                gpu_budget=gpu_budget,
                base_port=base_port,
                proxy_port=proxy_port,
            ) as handle:
                if handle.proxy_url:
                    os.environ["OCRSCOUT_VLLM_URL"] = handle.proxy_url
                    log.info(
                        "Managed endpoint: %s  (logs: %s)",
                        handle.proxy_url, handle.log_dir,
                    )
                else:
                    log.info(
                        "--managed requested but no vllm profiles; running "
                        "docling/in-process backends only."
                    )
                _execute(cfg, parallel_models=parallel_models)
        except ManagedServerError as e:
            log.error("Managed stack failed: %s", e)
            raise typer.Exit(code=1) from e
    else:
        _execute(cfg, parallel_models=parallel_models)


def _execute(
    cfg: PipelineConfig,
    *,
    parallel_models: int = 1,
) -> None:
    source_cls = registry.get("sources", cfg.source.name)
    source = source_cls(**cfg.source.args)
    pages_iter = source.iter_pages()
    if cfg.sample is not None:
        pages = list(itertools.islice(pages_iter, cfg.sample))
    else:
        pages = list(pages_iter)
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
    # results[model_name] -> (ok, failed, run_seconds). We collect into a dict
    # to preserve cfg.models order for the summary table even when models
    # complete out-of-order under parallelism.
    results: dict[str, tuple[int, int, float]] = {}

    def _run(name: str) -> tuple[str, int, int, float]:
        ok, failed, run_seconds = _run_one_model(
            model_name=name,
            pages=pages,
            page_by_id=page_by_id,
            normalizer_overrides=cfg.normalizer_overrides,
            exporter=exporter,
            reference_adapter=reference_adapter,
            metrics=metrics,
        )
        return name, ok, failed, run_seconds

    try:
        if parallel_models <= 1:
            for model_name in cfg.models:
                _, ok, failed, run_seconds = _run(model_name)
                results[model_name] = (ok, failed, run_seconds)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_models) as ex:
                futures = [ex.submit(_run, name) for name in cfg.models]
                for fut in concurrent.futures.as_completed(futures):
                    name, ok, failed, run_seconds = fut.result()
                    results[name] = (ok, failed, run_seconds)
    finally:
        exporter.close()
        metrics.finish()

    summary_rows = [(m, *results[m]) for m in cfg.models if m in results]
    _print_summary(summary_rows, dest=cfg.export.args["dest"])


def _run_one_model(
    *,
    model_name: str,
    pages: list,
    page_by_id: dict,
    normalizer_overrides: dict,
    exporter,
    reference_adapter: ReferenceAdapter | None,
    metrics: MetricsCollector,
) -> tuple[int, int, float]:
    try:
        profile = resolve(model_name)
        backend = registry.get("backends", profile.source)()
        normalizer_name = normalizer_overrides.get(model_name, profile.normalizer)
        normalizer = registry.get("normalizers", normalizer_name)()
    except Exception as e:
        log.error("[%s] setup failed: %s", model_name, e)
        return 0, len(pages), 0.0

    log.info(
        "[%s] starting (backend=%s, normalizer=%s)",
        model_name, profile.source, normalizer_name,
    )

    try:
        with metrics.stage(f"{model_name}.prepare"):
            inv = backend.prepare(profile, pages)
        with metrics.stage(f"{model_name}.run"):
            raws = list(backend.run(inv))
    except BackendError as e:
        log.error("[%s] backend failed: %s", model_name, e)
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

        reference = None
        if reference_adapter is not None:
            try:
                reference = reference_adapter.get(page)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "reference adapter failed for %s/%s: %s",
                    model_name, raw.page_id, e,
                )

        record = ExportRecord(
            page=page,
            model=model_name,
            document=doc,
            raw=raw,
            reference=reference,
            markdown=markdown,
            metrics={
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

    metrics.add_pages(ok=ok, failed=failed)
    run_seconds = metrics.stage_seconds.get(f"{model_name}.run", 0.0)
    log.info(
        "[%s] done: %d/%d ok in %.1fs",
        model_name, ok, ok + failed, run_seconds,
    )
    return ok, failed, run_seconds


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


def _print_summary(
    summary_rows: list[tuple[str, int, int, float]], *, dest: str
) -> None:
    # Summary table is the *deliverable*, not status — always rprinted, never
    # gated by log level. Quiet mode still shows it.
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
    log.info("Wrote %s", dest)


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

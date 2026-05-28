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

from pydantic import ValidationError

from ocrscout import registry
from ocrscout import state as state_mod
from ocrscout.cli import app
from ocrscout.cost import recorder as cost_recorder
from ocrscout.errors import (
    BackendError,
    NormalizerError,
    ProfileNotFound,
    RegistryError,
    RunnerError,
    ScoutError,
)
from ocrscout.interfaces.source import SourceAdapter
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
        help="Size of the concurrent-model chunk for ephemeral `run`. "
             "Default 1 = strict one-at-a-time: spawn one vLLM serve, run "
             "all pages, tear down, spawn the next. Lowest VRAM (preflight "
             "only checks `max(per-profile)`) at the cost of paying vLLM "
             "cold-start once per model. N>1 keeps N serves resident as a "
             "chunk; preflight rejects the chunk if its total exceeds the "
             "GPU. Ignored under `--keep-up` (which forces single-launch "
             "parallel spawn, today's behavior).",
    ),
    batch_concurrency: int | None = typer.Option(
        None, "--batch-concurrency", min=1,
        help="Override the GPU-aware autoscaler's per-profile concurrency. "
             "Sets `concurrent_requests` (litellm) and `region_concurrency` "
             "(layout_chat) to N, then sizes vLLM's `kv_cache_memory_bytes` "
             "to fit. Refuses if it doesn't fit. Default: auto-derived from "
             "detected GPU capacity.",
    ),
    detector_workers: int | None = typer.Option(
        None, "--detector-workers", min=1,
        help="Override the CPU detector pool size for `backend: layout_chat` "
             "profiles. Each worker owns its own PP-DocLayoutV3 instance and "
             "runs in parallel on CPU. Default: auto-derived from "
             "`sched_getaffinity` (capped at 8).",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Skip pages already present in <output-dir>/data/train-*.parquet "
             "for each model. Resume is per-model: a page already done for "
             "model A is still attempted for model B if B hadn't processed "
             "it yet. Survives a crash anywhere mid-run.",
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
    if detector_workers is not None:
        # The layout_chat backend reads this env var in its
        # `_resolve_detector_workers` precedence chain (profile > state >
        # env > auto). Setting it here covers both ephemeral runs and any
        # subprocess workers the runner spawns, since env vars inherit.
        os.environ["OCRSCOUT_DETECTOR_WORKERS"] = str(detector_workers)
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
    # --seed reaches the adapter regardless of how sample was set (typer
    # --sample flag or --source-arg sample=N), and regardless of whether
    # the adapter pre-samples at all (BHL applies seed to the rank query
    # even without an explicit sample).
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
        resume=resume,
        batch_concurrency=batch_concurrency,
    )


def _export_backend_overrides(handle_extra: dict) -> dict[str, dict[str, int]]:
    """Slim the autoscale context on a RunnerHandle down to the per-profile
    concurrency + KV dict the env-var side channel transports.

    The KV slot lets ``_autoscale_values_for_profile`` stamp the right
    value onto ExportRecord even though the profile YAML no longer
    carries it; ``_state_override`` in the backend only consults
    ``concurrent_requests`` / ``region_concurrency``.
    """
    autoscale = (handle_extra or {}).get("autoscale") or {}
    profiles = autoscale.get("profiles") or {}
    out: dict[str, dict[str, int]] = {}
    for name, rec in profiles.items():
        cr = int(rec.get("concurrent_requests") or 0)
        rc = int(rec.get("region_concurrency") or 0)
        kv = int(rec.get("kv_cache_memory_bytes") or 0)
        if cr > 0 or rc > 0 or kv > 0:
            entry: dict[str, int] = {}
            if cr > 0:
                entry["concurrent_requests"] = cr
            if rc > 0:
                entry["region_concurrency"] = rc
            if kv > 0:
                entry["kv_cache_memory_bytes"] = kv
            out[name] = entry
    return out


def _write_runtime_yaml(
    cfg: PipelineConfig,
    runner,
    handle,
    gpu_budget: float,
    batch_concurrency: int | None,
    parallel_models: int,
) -> None:
    """Persist a per-run ``<output_dir>/runtime.yaml`` capturing the
    autoscaler's decisions for this launch.

    The file is a sibling of ``pipeline.yaml`` already written at
    ``ocrscout run`` start: where pipeline.yaml records *what the user
    asked for*, runtime.yaml records *what the runner decided*. Read by
    ``ocrscout inspect`` to render a Run-context header.

    No-op when ``cfg.output_dir`` isn't set (defensive — every CLI
    entrypoint provides one today).
    """
    if cfg.output_dir is None:
        return
    out_dir = Path(cfg.output_dir)
    if not out_dir.is_dir():
        # `ocrscout run` mkdirs this; if a caller hasn't yet, skip
        # quietly rather than try to second-guess them.
        return

    from ocrscout.runtime import (
        AutoscaleProfileRecord,
        AutoscaleRuntime,
        GpuRuntime,
        RunnerRuntime,
        RuntimeContext,
        now_iso,
        ocrscout_version,
        write_runtime_context,
    )

    autoscale_extra = (handle.extra or {}).get("autoscale") or {}
    gpu_info = autoscale_extra.get("gpu") or {}
    gpu_block: GpuRuntime | None = None
    if gpu_info.get("name"):
        bw_raw = gpu_info.get("memory_bandwidth_gb_s")
        spec_name = gpu_info.get("dbgpu_spec_name")
        gpu_block = GpuRuntime(
            name=str(gpu_info["name"]),
            total_bytes=int(gpu_info.get("total_bytes") or 0),
            free_bytes_at_launch=int(gpu_info.get("free_bytes_at_launch") or 0),
            memory_bandwidth_gb_s=float(bw_raw) if bw_raw is not None else None,
            dbgpu_spec_name=str(spec_name) if spec_name else None,
        )

    autoscale_block: AutoscaleRuntime | None = None
    profile_recs: dict[str, AutoscaleProfileRecord] = {}
    for name, rec in (autoscale_extra.get("profiles") or {}).items():
        profile_recs[name] = AutoscaleProfileRecord(
            explicit_kv_in_yaml=bool(rec.get("explicit_kv_in_yaml", False)),
            overhead_bytes=int(rec.get("overhead_bytes") or 0),
            kv_cache_memory_bytes=int(rec.get("kv_cache_memory_bytes") or 0),
            concurrent_requests=int(rec.get("concurrent_requests") or 0),
            region_concurrency=int(rec.get("region_concurrency") or 0),
            max_model_len=int(rec.get("max_model_len") or 0),
        )
    if profile_recs:
        autoscale_block = AutoscaleRuntime(
            per_token_bytes=int(autoscale_extra.get("per_token_bytes") or 0),
            max_concurrency_ceiling=int(
                autoscale_extra.get("max_concurrency_ceiling") or 0
            ),
            profiles=profile_recs,
        )

    ctx = RuntimeContext(
        ocrscout_version=ocrscout_version(),
        started_at=now_iso(),
        gpu=gpu_block,
        runner=RunnerRuntime(
            name=getattr(runner, "name", "unknown"),
            gpu_budget=gpu_budget,
            batch_concurrency_override=batch_concurrency,
            parallel_models=parallel_models,
        ),
        autoscale=autoscale_block,
    )
    try:
        write_runtime_context(out_dir, ctx)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not write runtime.yaml to %s: %s", out_dir, e)


def _autoscale_values_for_profile(
    profile,
) -> tuple[int | None, int | None, int | None]:
    """Recover the autoscaler's decisions for ``profile`` so they can be
    stamped onto each row's ExportRecord.

    Returns ``(concurrent_requests, region_concurrency, kv_cache_memory_bytes)``.
    Hosted / cpu profiles return ``(None, None, None)`` for the KV slot;
    backend defaults still apply for concurrency.

    Sources: the env-var side channel (set by ``_publish_backend_overrides``
    after each launch) for ephemeral in-process runs, then state.yaml for
    launch+submit+worker. Falls back to whatever the profile YAML carries
    (the explicit-KV escape hatch).
    """
    from ocrscout.backends.litellm import _state_override

    kv: int | None = None
    yaml_kv = (profile.vllm_engine_args or {}).get("kv_cache_memory_bytes")
    if yaml_kv is not None:
        try:
            from ocrscout.runners._preflight import parse_bytes

            kv = parse_bytes(yaml_kv)
        except Exception:  # noqa: BLE001
            kv = None
    # State.yaml carries the autoscaler-computed KV; check both via the
    # env var (in-process) and the state file.
    if kv is None:
        env_raw = os.environ.get("OCRSCOUT_BACKEND_OVERRIDES")
        if env_raw:
            try:
                rec = json.loads(env_raw).get(profile.name, {})
                if rec.get("kv_cache_memory_bytes") is not None:
                    kv = int(rec["kv_cache_memory_bytes"])
            except (ValueError, TypeError):
                pass
    if kv is None:
        try:
            st = state_mod.read_state()
        except Exception:  # noqa: BLE001
            st = None
        if st is not None:
            args_autoscale = (st.args or {}).get("autoscale") or {}
            rec = (args_autoscale.get("profiles") or {}).get(profile.name) or {}
            v = rec.get("kv_cache_memory_bytes")
            if v is not None:
                kv = int(v)

    cr = _state_override(profile.name, "concurrent_requests")
    if cr is None:
        explicit_cr = (profile.backend_args or {}).get("concurrent_requests")
        cr = int(explicit_cr) if explicit_cr is not None else None
    rc = _state_override(profile.name, "region_concurrency")
    if rc is None:
        explicit_rc = (profile.backend_args or {}).get("region_concurrency")
        rc = int(explicit_rc) if explicit_rc is not None else None

    return cr, rc, kv


def _publish_backend_overrides(handle_extra: dict) -> None:
    """Set the ``OCRSCOUT_BACKEND_OVERRIDES`` env var so backends and the
    ExportRecord stamping see the autoscaler's per-profile concurrency.

    Idempotent: empty overrides clears the var so a hosted-only launch
    doesn't leak prior values into the next chunk.
    """
    from ocrscout.backends.litellm import _BACKEND_OVERRIDES_ENV

    overrides = _export_backend_overrides(handle_extra)
    if overrides:
        os.environ[_BACKEND_OVERRIDES_ENV] = json.dumps(overrides)
    else:
        os.environ.pop(_BACKEND_OVERRIDES_ENV, None)


def run_pipeline(
    cfg: PipelineConfig,
    *,
    parallel_models: int = 1,
    base_port: int = 8000,
    proxy_port: int = 4000,
    gpu_budget: float = 0.85,
    keep_up: bool = False,
    resume: bool = False,
    batch_concurrency: int | None = None,
) -> None:
    """Launch the required runner (if any), run the pipeline, tear down.

    Shared by ``ocrscout run`` and ``ocrscout apply``. Auto-launches a
    ``LocalRunner`` for any profile with ``runtime != cpu``.

    Ephemeral runs default to **model-major** dispatch: one vLLM serve up at
    a time (chunk size 1), all pages for that model, teardown, next chunk.
    This way the preflight budget check enforces ``max(per-profile)`` rather
    than ``sum(per-profile)``, so a benchmark matrix only needs to fit the
    largest single model. ``--parallel-models N`` widens the chunk to N
    concurrent serves; the preflight cap applies per chunk. ``--keep-up``
    forces the original single-launch parallel-spawn path (all serves
    resident together) so a follow-up ``submit`` pays no relaunch cost.

    Profiles with ``runtime: cpu`` (e.g. Tesseract) run in-process in a
    trailing pass with no proxy. ``runtime: hosted`` profiles share one
    trailing proxy-only launch (no GPU work involved).
    """
    try:
        profiles = [resolve(name) for name in cfg.models]
    except ProfileNotFound as e:
        raise typer.BadParameter(str(e)) from e

    # Construct the source adapter BEFORE any runner work so that
    # configuration errors (e.g. BHL without a prior `source bhl setup`)
    # surface as a typer.BadParameter without first spinning up the
    # LiteLLM proxy + vLLM serves.
    try:
        source = _construct_source(cfg)
    except (ScoutError, ValidationError, RegistryError) as e:
        raise typer.BadParameter(str(e)) from e

    vllm_profiles = [p for p in profiles if p.runtime == "vllm"]
    hosted_profiles = [p for p in profiles if p.runtime == "hosted"]
    cpu_profiles = [p for p in profiles if p.runtime == "cpu"]

    if not vllm_profiles and not hosted_profiles:
        log.info("All profiles are runtime=cpu; running CPU-side backends in-process.")
        _execute(cfg, source=source, parallel_models=parallel_models, resume=resume)
        return

    runner = LocalRunner()
    existing_state = state_mod.read_state()
    reused_existing = (
        existing_state is not None and existing_state.runner == "local"
    )

    # Single-launch path: today's all-serves-resident behavior. Used when
    # the user explicitly asked for it (--keep-up), when a persistent stack
    # is already up (we reuse it as-is — caller's responsibility for the
    # models to match), or when there's nothing to chunk (≤1 vLLM model).
    single_launch = keep_up or reused_existing or len(vllm_profiles) <= 1

    if single_launch:
        try:
            handle = runner.launch(
                models=cfg.models,
                base_port=base_port,
                proxy_port=proxy_port,
                gpu_budget=gpu_budget,
                persistent=keep_up,
                batch_concurrency=batch_concurrency,
            )
            os.environ["OCRSCOUT_VLLM_URL"] = handle.proxy_url
            _publish_backend_overrides(handle.extra)
            _write_runtime_yaml(cfg, runner, handle, gpu_budget,
                                batch_concurrency, parallel_models)
            if reused_existing:
                log.info("Reusing already-running local stack")
        except RunnerError as e:
            log.error("Runner launch failed: %s", e)
            raise typer.Exit(code=1) from e

        try:
            _execute(cfg, source=source, parallel_models=parallel_models, resume=resume)
        finally:
            _publish_backend_overrides({})
            if not keep_up and not reused_existing:
                try:
                    runner.down()
                except Exception as e:  # noqa: BLE001
                    log.warning("Runner teardown reported error: %s", e)
        return

    # Model-major chunked path. Each chunk = one ephemeral launch with its
    # own LiteLLM proxy + vLLM serve(s); teardown between chunks frees the
    # GPU before the next model loads. The shared `source` and exporter
    # state mean per-chunk output appends into the same parquet shards;
    # resume reads (page, model) pairs from those shards directly.
    chunk_size = max(1, parallel_models)
    vllm_names = [p.name for p in vllm_profiles]
    chunks = list(_iter_model_chunks(vllm_names, chunk_size))
    log.info(
        "Model-major dispatch: %d vLLM chunk(s) of up to %d model(s)%s%s",
        len(chunks), chunk_size,
        f" + {len(hosted_profiles)} hosted" if hosted_profiles else "",
        f" + {len(cpu_profiles)} cpu" if cpu_profiles else "",
    )

    for chunk_idx, chunk_models in enumerate(chunks):
        log.info(
            "vLLM chunk %d/%d: %s",
            chunk_idx + 1, len(chunks), ",".join(chunk_models),
        )
        try:
            handle = runner.launch(
                models=chunk_models,
                base_port=base_port,
                proxy_port=proxy_port,
                gpu_budget=gpu_budget,
                persistent=False,
                batch_concurrency=batch_concurrency,
            )
            os.environ["OCRSCOUT_VLLM_URL"] = handle.proxy_url
            _publish_backend_overrides(handle.extra)
            # Each chunk gets its own runtime.yaml snapshot; last chunk
            # wins on disk. Adequate for the typical "all chunks share
            # the same GPU and the same flags" case.
            _write_runtime_yaml(cfg, runner, handle, gpu_budget,
                                batch_concurrency, parallel_models)
        except RunnerError as e:
            log.error(
                "Runner launch failed for chunk %s: %s", chunk_models, e,
            )
            raise typer.Exit(code=1) from e

        try:
            _execute(
                _cfg_for_chunk(cfg, chunk_models),
                source=source,
                parallel_models=len(chunk_models),
                resume=resume,
            )
        finally:
            _publish_backend_overrides({})
            try:
                runner.down()
            except Exception as e:  # noqa: BLE001
                log.warning("Runner teardown reported error: %s", e)

    if hosted_profiles:
        hosted_names = [p.name for p in hosted_profiles]
        log.info("Hosted chunk: %s", ",".join(hosted_names))
        try:
            handle = runner.launch(
                models=hosted_names,
                base_port=base_port,
                proxy_port=proxy_port,
                gpu_budget=gpu_budget,
                persistent=False,
            )
            os.environ["OCRSCOUT_VLLM_URL"] = handle.proxy_url
            _publish_backend_overrides(handle.extra)
        except RunnerError as e:
            log.error("Runner launch failed for hosted chunk: %s", e)
            raise typer.Exit(code=1) from e
        try:
            _execute(
                _cfg_for_chunk(cfg, hosted_names),
                source=source,
                parallel_models=min(parallel_models, len(hosted_names)),
                resume=resume,
            )
        finally:
            _publish_backend_overrides({})
            try:
                runner.down()
            except Exception as e:  # noqa: BLE001
                log.warning("Runner teardown reported error: %s", e)

    if cpu_profiles:
        cpu_names = [p.name for p in cpu_profiles]
        log.info("CPU chunk: %s", ",".join(cpu_names))
        _execute(
            _cfg_for_chunk(cfg, cpu_names),
            source=source,
            parallel_models=min(parallel_models, len(cpu_names)),
            resume=resume,
        )


def _iter_model_chunks(names: list[str], chunk_size: int) -> list[list[str]]:
    """Slide a window of ``chunk_size`` over ``names`` preserving order.

    Returns a concrete list (not iterator) so callers can ``len()`` it
    for progress logging.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [names[i : i + chunk_size] for i in range(0, len(names), chunk_size)]


def _cfg_for_chunk(cfg: PipelineConfig, model_names: list[str]) -> PipelineConfig:
    """Copy ``cfg`` with ``models`` restricted to ``model_names``.

    Used by the chunked dispatch path so each chunk's ``_execute`` only
    iterates over its own models against the shared source + exporter.
    """
    return cfg.model_copy(update={"models": list(model_names)})


def _construct_source(cfg: PipelineConfig) -> SourceAdapter:
    """Resolve and instantiate the source adapter from the pipeline config.

    Pulled into its own helper so callers can fail fast — adapter
    validators (e.g. BHL's check that ``info.yaml`` is configured) run
    here, before any expensive runner setup.
    """
    source_cls = registry.get("sources", cfg.source.name)
    return source_cls(**cfg.source.args)


def _execute(
    cfg: PipelineConfig,
    *,
    source: SourceAdapter,
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

    # Per-model resume done-set built once from the parquet shards already on
    # disk (the single source of truth — see done_pairs_from_parquet). Each
    # model gets its own done page_ids: a page completed for A but not B is
    # still attempted for B. The filter is applied as a streaming predicate
    # inside ``_run`` so ``_run_one_model`` consumes pages lazily.
    done_pairs: dict[str, set[str]] = {}
    if resume:
        from ocrscout.exports.parquet import done_pairs_from_parquet
        done_pairs = done_pairs_from_parquet(cfg.output_dir)
        total_done = sum(len(s) for s in done_pairs.values())
        if total_done:
            log.info(
                "Resume: %d (page, model) pairs already in %s — filtering per model",
                total_done, cfg.output_dir,
            )

    source_label = cfg.source.args.get("path") or cfg.source.name

    def _fresh_pages_iter() -> "_PeekableIter":
        """Return a fresh page iterator from the source, sample-limited
        and resume-filter-able. Each model gets its own — for BHL the
        DuckDB sample is cached on the adapter so only the S3 fetches
        re-run, but pages stream lazily so resident PIL images stay
        bounded at ~``concurrent_fetches`` per source-side worker pool
        plus the backend's per-batch buffer."""
        it = source.iter_pages()
        if cfg.sample is not None:
            it = itertools.islice(it, cfg.sample)
        return _PeekableIter(it)

    # Peek the first page once to surface an empty-source error before
    # any backend setup happens. The peeked iterator is discarded; each
    # model gets its own fresh stream below.
    probe = _fresh_pages_iter()
    if probe.peek() is None:
        log.error(
            "No images found from source %r (args=%s). For `hf_dataset` "
            "this means the `imagefolder` builder found no images and the "
            "path is not an HF Hub dataset id. For other adapters, check "
            "their --source-arg filters.",
            cfg.source.name, cfg.source.args,
        )
        raise typer.Exit(code=1)
    del probe

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
    # exists even if a backend later fails on a particular page. Volume
    # objects carry no images, so their list is cheap to hold.
    volume_count = 0
    for volume in source.iter_volumes():
        exporter.write_volume(volume)
        volume_count += 1
    if volume_count:
        log.info("Loaded %d volume(s) from source", volume_count)

    log.info("Streaming pages from %r", source_label)

    metrics = MetricsCollector(pipeline_id=cfg.name)
    # results[model_name] -> ModelRunResult. Dict so the summary table can
    # walk cfg.models in order even when parallel runs finish out-of-order.
    results: dict[str, _ModelRunResult] = {}

    def _run(name: str) -> tuple[str, _ModelRunResult]:
        model_done = done_pairs.get(name, set())
        pages_iter = _fresh_pages_iter()
        if model_done:
            log.info("[%s] resume: filtering pages already in parquet", name)
            pages_iter = _filter_done(pages_iter, model_done)
        result = _run_one_model(
            model_name=name,
            pages_iter=pages_iter,
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


class _PeekableIter:
    """Single-item-lookahead wrapper around an iterator.

    Used so ``_execute`` can detect an empty source before spinning up
    any backend without consuming the iterator destructively when not
    empty: ``peek()`` returns the next item (or ``None``) without
    advancing the consumer's view; subsequent iteration sees the
    peeked item first.
    """

    def __init__(self, it: "object") -> None:
        self._it = iter(it)  # type: ignore[arg-type]
        self._head: Any = _UNSET

    def peek(self) -> Any:
        if self._head is _UNSET:
            try:
                self._head = next(self._it)
            except StopIteration:
                return None
        return self._head

    def __iter__(self) -> "_PeekableIter":
        return self

    def __next__(self) -> Any:
        if self._head is not _UNSET:
            item = self._head
            self._head = _UNSET
            return item
        return next(self._it)


_UNSET: Any = object()


def _filter_done(it, done_set):
    """Drop pages whose page_id is already in ``done_set``."""
    for page in it:
        if page.page_id not in done_set:
            yield page


def _chunked(it, size: int):
    """Yield successive lists of up to ``size`` items from ``it``.

    Each chunk is materialised as a list so the backend can size its
    internal thread pool exactly; between chunks the previous chunk's
    PageImages (with their decoded PIL bytes) drop out of scope and
    Python's refcount-GC reclaims the image memory before the next
    chunk is decoded. Without this boundary, the source's
    ThreadPoolExecutor would keep filling its as_completed queue with
    finished PageImages while the backend processed earlier ones —
    O(sample) PIL images resident at peak."""
    if size <= 0:
        raise ValueError(f"_chunked size must be positive, got {size}")
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


@dataclass
class _ModelRunResult:
    ok: int
    failed: int
    run_seconds: float
    comparison_results: list[ComparisonResult] = dc_field(default_factory=list)


_PAGE_CHUNK_BASELINE = 32
"""Minimum chunk size when feeding pages into a backend's ``run()``. Picked
large enough that even a cold model's per-batch overhead (cudagraph
warmup, HTTP keep-alive setup) amortises, small enough that 32 PIL
images (~500 MB on 2000x3000 BHL JP2s) sit comfortably in RAM."""

_PAGE_CHUNK_CONCURRENCY_MULT = 2
"""Multiplier on the backend's autoscaler-decided concurrency. Setting
chunk = ``2 × concurrent_requests`` means while the backend is processing
the back half of a chunk the front half's futures are already done — no
idle workers between chunks — at the cost of ~2× the per-chunk memory
footprint vs. chunk = concurrency. Empirically gives the same total
throughput as the single-batch (pre-streaming) path."""


def _resolve_chunk_size(record_concurrency: int | None,
                        record_region_concurrency: int | None) -> int:
    """Page chunk size for one backend.run() call.

    Covers both whole-page backends (``concurrent_requests``) and the
    region backend (``region_concurrency``) by taking the larger; falls
    back to the baseline when both are unset (e.g. CPU / hosted profiles
    that don't autoscale)."""
    concurrency = max(
        record_concurrency or 0,
        record_region_concurrency or 0,
    )
    return max(
        _PAGE_CHUNK_BASELINE,
        _PAGE_CHUNK_CONCURRENCY_MULT * concurrency,
    )


def _run_one_model(
    *,
    model_name: str,
    pages_iter,
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
        return _ModelRunResult(ok=0, failed=0, run_seconds=0.0)

    # Resolve the autoscaler-decided concurrency / KV so each ExportRecord
    # carries the values that actually shaped the run (queryable per-row
    # without joining external metadata). Same precedence chain the
    # backend uses, so the stamped numbers match what really ran.
    record_concurrency, record_region_concurrency, record_kv = (
        _autoscale_values_for_profile(profile)
    )
    chunk_size = _resolve_chunk_size(record_concurrency, record_region_concurrency)

    log.info(
        "[%s] starting (backend=%s, runtime=%s, normalizer=%s, chunk_size=%d)",
        model_name, profile.backend, profile.runtime, normalizer_name, chunk_size,
    )

    try:
        with metrics.stage(f"{model_name}.prepare"):
            inv = backend.prepare(profile)
    except BackendError as e:
        log.error("[%s] prepare failed: %s", model_name, e)
        return _ModelRunResult(ok=0, failed=0, run_seconds=0.0)

    prepare_seconds = metrics.stage_seconds.get(f"{model_name}.prepare", 0.0)

    ok = 0
    failed = 0
    accumulated_comparisons: list[ComparisonResult] = []
    t_run0 = time.perf_counter()

    try:
        with metrics.stage(f"{model_name}.run"):
            for chunk in _chunked(pages_iter, chunk_size):
                for page, raw in backend.run(inv, chunk):
                    # ``run_seconds_total`` / ``..._per_page`` are stamped
                    # with the cumulative wall-clock since prepare finished.
                    # Early records get understated totals; the final record
                    # has the accurate full-run number. Cumulative-and-rolling
                    # is the streaming-friendly approximation of what used to
                    # be a single post-hoc value applied to every row.
                    elapsed_run = time.perf_counter() - t_run0
                    handled = _process_one(
                        model_name=model_name,
                        page=page,
                        raw=raw,
                        profile=profile,
                        normalizer=normalizer,
                        exporter=exporter,
                        reference_adapter=reference_adapter,
                        comparisons=comparisons,
                        metrics=metrics,
                        gpu=gpu,
                        prepare_seconds=prepare_seconds,
                        run_seconds_so_far=elapsed_run,
                        pages_attempted_so_far=ok + failed + 1,
                        record_kv=record_kv,
                        record_concurrency=record_concurrency,
                        record_region_concurrency=record_region_concurrency,
                        accumulated_comparisons=accumulated_comparisons,
                    )
                    if handled == "ok":
                        ok += 1
                    else:
                        failed += 1
                # Chunk goes out of scope here; refcount-GC reclaims the
                # decoded PIL images before the next chunk's pages get
                # pulled out of the source iterator.
    except BackendError as e:
        log.error("[%s] backend failed: %s", model_name, e)
        run_seconds = metrics.stage_seconds.get(f"{model_name}.run", time.perf_counter() - t_run0)
        return _ModelRunResult(ok=ok, failed=failed, run_seconds=run_seconds)

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


def _process_one(
    *,
    model_name: str,
    page,
    raw,
    profile,
    normalizer,
    exporter,
    reference_adapter: ReferenceAdapter | None,
    comparisons: list[Comparison],
    metrics: MetricsCollector,
    gpu: GpuConfig,
    prepare_seconds: float,
    run_seconds_so_far: float,
    pages_attempted_so_far: int,
    record_kv: int | None,
    record_concurrency: int | None,
    record_region_concurrency: int | None,
    accumulated_comparisons: list[ComparisonResult],
) -> str:
    """Normalize + compare + export one (page, raw). Returns 'ok' or 'failed'.

    Split out of ``_run_one_model`` only so the streaming loop body stays
    legible; no semantic change vs. the previous per-row block."""
    if raw.error:
        log.warning("model %s page %s reported error: %s", model_name, page.file_id, raw.error)
        cost_recorder.close_page(page.page_id)
        return "failed"
    try:
        t_norm0 = time.perf_counter()
        with metrics.stage(f"{model_name}.normalize"):
            doc = normalizer.normalize(raw, page, profile)
        normalize_seconds = time.perf_counter() - t_norm0
    except NormalizerError as e:
        log.warning("normalizer failed for %s/%s: %s", model_name, page.file_id, e)
        cost_recorder.close_page(page.page_id)
        return "failed"

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

    run_seconds_per_page = run_seconds_so_far / max(pages_attempted_so_far, 1)
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
            "run_seconds_total": round(run_seconds_so_far, 4),
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
        kv_cache_memory_bytes=record_kv,
        concurrent_requests=record_concurrency,
        region_concurrency=record_region_concurrency,
    )
    exporter.write(record)
    return "ok"


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

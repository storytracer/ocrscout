"""Per-run ``runtime.yaml`` — what the Runner decided at launch time.

A sibling file to the user's ``<output_dir>/pipeline.yaml``. Where
``pipeline.yaml`` records *what the user asked for* (source, models,
comparisons), this file records *what actually happened at runtime*:
which GPU was detected, what the autoscaler computed for each profile,
and what overrides (``--batch-concurrency``, ``--gpu-budget``) shaped
those decisions.

Loaded by ``ocrscout inspect`` to render a Run context header when the
output directory contains one. Independent of ``state.yaml`` — that's a
*global* current-runner snapshot; this is a *per-run* historical record
that travels with the output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocrscout.errors import StateError
from ocrscout.state import _atomic_write_yaml

RUNTIME_YAML_NAME = "runtime.yaml"


class GpuRuntime(BaseModel):
    """GPU snapshot taken at launch time."""

    model_config = ConfigDict(extra="forbid")

    name: str
    total_bytes: int
    free_bytes_at_launch: int
    memory_bandwidth_gb_s: float | None = None
    """Manufacturer-spec memory bandwidth from dbgpu when the GPU name
    matched a catalog entry; ``None`` for unknown GPUs (autoscaler falls
    back to the per-backend safe ceiling)."""
    dbgpu_spec_name: str | None = None
    """Canonical dbgpu name we matched to (e.g. ``"H100 PCIe 80 GB"``).
    ``None`` when unmatched."""


class RunnerRuntime(BaseModel):
    """Runner-side knobs that shaped autoscale."""

    model_config = ConfigDict(extra="forbid")

    name: str
    gpu_budget: float
    batch_concurrency_override: int | None = None
    parallel_models: int


class AutoscaleProfileRecord(BaseModel):
    """Per-profile autoscale decision record."""

    model_config = ConfigDict(extra="forbid")

    explicit_kv_in_yaml: bool
    """``true`` when the profile YAML declared ``kv_cache_memory_bytes``
    explicitly; the autoscaler honored it without computing one."""
    overhead_bytes: int
    """Estimated non-KV footprint (weights + working slack)."""
    kv_cache_memory_bytes: int
    concurrent_requests: int
    region_concurrency: int
    max_model_len: int


class AutoscaleRuntime(BaseModel):
    """Autoscaler context: which constants + per-profile decisions."""

    model_config = ConfigDict(extra="forbid")

    per_token_bytes: int
    max_concurrency_ceiling: int
    profiles: dict[str, AutoscaleProfileRecord] = Field(default_factory=dict)


class RuntimeContext(BaseModel):
    """Top-level ``runtime.yaml`` payload."""

    model_config = ConfigDict(extra="forbid")

    ocrscout_version: str
    started_at: str
    gpu: GpuRuntime | None = None
    runner: RunnerRuntime
    autoscale: AutoscaleRuntime | None = None
    """``None`` when no ``runtime: vllm`` profiles were in the run
    (hosted-only or cpu-only matrices skip autoscale entirely)."""


def runtime_yaml_path(output_dir: Path) -> Path:
    return Path(output_dir) / RUNTIME_YAML_NAME


def write_runtime_context(output_dir: Path, ctx: RuntimeContext) -> None:
    """Atomically write the runtime context to ``<output_dir>/runtime.yaml``."""
    _atomic_write_yaml(runtime_yaml_path(output_dir), ctx.model_dump(mode="json"))


def read_runtime_context(output_dir: Path) -> RuntimeContext | None:
    """Read ``<output_dir>/runtime.yaml`` or return ``None`` if absent."""
    path = runtime_yaml_path(output_dir)
    if not path.exists():
        return None
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise StateError(f"cannot read {path}: {e}") from e
    if not isinstance(raw, dict):
        raise StateError(f"{path} must contain a YAML mapping")
    try:
        return RuntimeContext.model_validate(raw)
    except ValidationError as e:
        raise StateError(f"invalid runtime context in {path}: {e}") from e


def now_iso() -> str:
    """ISO8601 timestamp in UTC, second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ocrscout_version() -> str:
    """Installed package version, or ``"unknown"`` if not resolvable."""
    try:
        return version("ocrscout")
    except PackageNotFoundError:
        return "unknown"

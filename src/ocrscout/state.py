"""Filesystem persistence for ``~/.ocrscout/``.

Holds three things:

* ``state.yaml`` — the active runner's snapshot (proxy URL, PIDs, models,
  args used at launch). Written by ``Runner.launch()``; cleared by
  ``Runner.down()``. Subsequent CLI commands read it to resolve the
  proxy URL without needing flags repeated.
* ``config.yaml`` — user-level defaults (GPU type / cost / provider /
  preferred runner). Env vars override the GPU block at read time so
  remote workers can stamp their own pricing context.
* ``pids/`` ``logs/`` ``jobs/`` — subdirectories the LocalRunner writes
  daemon PIDs, daemon stdout/stderr, and per-job worker logs into.

All writes are atomic (tmp file in the same directory, then ``os.replace``)
so an interrupted write never leaves a partially-readable state file.

The base directory is ``~/.ocrscout/`` by default; override with the
``OCRSCOUT_STATE_DIR`` env var for tests and multi-environment hosts.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from ocrscout.errors import StateError


class GpuConfig(BaseModel):
    """GPU context. Stamped onto every page record so cross-cell
    benchmark Parquets are self-describing.
    """

    type: str = "unknown"
    cost_per_hour: float = 0.0
    provider: str = "local"


class UserConfig(BaseModel):
    """User-level defaults persisted at ``~/.ocrscout/config.yaml``."""

    gpu: GpuConfig = Field(default_factory=GpuConfig)
    default_runner: str = "local"
    default_output_dir: str | None = None


class ManagedProcess(BaseModel):
    """One daemonised child managed by a Runner.

    For ``LocalRunner`` this carries a PID and port. For remote runners
    ``handle`` carries the scheduler-side identifier (SkyPilot pool name,
    HF job id, …).
    """

    name: str
    pid: int | None = None
    port: int | None = None
    log_path: str | None = None
    handle: str | None = None


class RunnerStateFile(BaseModel):
    """Persisted snapshot of an active runner.

    Written at ``~/.ocrscout/state.yaml`` whenever a ``Runner.launch()``
    succeeds; removed by ``Runner.down()``. Other CLI commands consult
    it to determine which runner is active, what models it serves, and
    where the LiteLLM proxy is reachable.
    """

    runner: str
    models: list[str]
    proxy_url: str | None = None
    processes: list[ManagedProcess] = Field(default_factory=list)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    output_dir: str | None = None
    launched_at: str
    args: dict[str, Any] = Field(default_factory=dict)
    """Original launch kwargs, kept so ``ocrscout run`` can decide whether
    to reuse an already-running stack or tear it down and re-launch with
    a different config."""
    phase: Literal["launching", "ready", "tearing_down"] = "ready"
    """Lifecycle stage for the persistent path. ``write_state_launching``
    flips this to ``launching`` BEFORE any subprocess spawn so a Ctrl-C
    mid-launch leaves a detectable breadcrumb on disk. The post-readiness
    ``mark_phase_ready`` is the only success indicator — a state file
    still at ``launching`` after the readiness deadline means the
    launcher was killed and ``ocrscout down --force`` should be used.
    Defaults to ``ready`` so pre-phase state.yaml files round-trip."""
    phase_updated_at: str | None = None
    """ISO8601 timestamp of the most recent phase transition. Used by
    ``is_stale_launching`` to flag crashed launchers."""
    backend_overrides: dict[str, dict[str, int]] = Field(default_factory=dict)
    """Per-profile backend kwargs the runner decided at launch.

    Outer key is ``profile.name``; inner dict carries
    ``{"concurrent_requests": N, "region_concurrency": N}``. Backends
    running in submitted-worker processes consult this so the GPU-aware
    autoscale decision made at ``ocrscout launch`` time survives the
    launch → submit → worker handoff. Empty for ephemeral runs (no
    handoff; profile mutation is visible in-process)."""


def state_dir() -> Path:
    """Resolve and create the state directory.

    Defaults to ``~/.ocrscout/``; override with ``OCRSCOUT_STATE_DIR``.
    """
    raw = os.environ.get("OCRSCOUT_STATE_DIR")
    p = Path(raw).expanduser() if raw else Path.home() / ".ocrscout"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pid_dir() -> Path:
    p = state_dir() / "pids"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = state_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def jobs_dir() -> Path:
    p = state_dir() / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_file_path() -> Path:
    return state_dir() / "state.yaml"


def config_file_path() -> Path:
    return state_dir() / "config.yaml"


def litellm_config_path() -> Path:
    return state_dir() / "litellm.yaml"


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_state() -> RunnerStateFile | None:
    """Read the active runner's state, or ``None`` if no runner is launched."""
    path = state_file_path()
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise StateError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise StateError(f"{path} must contain a YAML mapping")
    try:
        return RunnerStateFile.model_validate(data)
    except ValidationError as e:
        raise StateError(f"invalid state in {path}: {e}") from e


def write_state(state: RunnerStateFile) -> None:
    _atomic_write_yaml(state_file_path(), state.model_dump(mode="json"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state_launching(state: RunnerStateFile) -> None:
    """Atomic write with ``phase='launching'`` and a fresh timestamp.

    Called BEFORE any subprocess spawn in the persistent launch path so
    a Ctrl-C between this call and the final ``mark_phase_ready`` leaves
    a detectable breadcrumb that ``ocrscout status`` / ``ocrscout down
    --force`` can interpret.
    """
    state.phase = "launching"
    state.phase_updated_at = _now_iso()
    write_state(state)


def update_state_processes(state: RunnerStateFile) -> None:
    """Atomic re-write keeping ``phase`` as-is, refreshing
    ``phase_updated_at``.

    Used by the persistent path to record each ``ManagedProcess``
    incrementally as daemons come up and their PIDs get corrected
    (per design: incremental writes rather than only at the final flip).
    """
    state.phase_updated_at = _now_iso()
    write_state(state)


def mark_phase_ready(state: RunnerStateFile) -> None:
    """Atomic flip to ``phase='ready'`` + refresh timestamp.

    The only success indicator for the persistent launch path. Anything
    short of this on disk means the launcher didn't finish.
    """
    state.phase = "ready"
    state.phase_updated_at = _now_iso()
    write_state(state)


def is_stale_launching(
    state: RunnerStateFile, *, max_age_seconds: float = 900.0
) -> bool:
    """Whether a state file in ``phase='launching'`` looks abandoned.

    Heuristic: ``phase == 'launching'`` AND ``phase_updated_at`` is
    older than ``max_age_seconds`` (default 15 min ≈ 1.5× the default
    ready_timeout). Used by ``ocrscout status`` to surface "launcher
    likely crashed; run ``ocrscout down --force``."
    """
    if state.phase != "launching":
        return False
    if not state.phase_updated_at:
        return False
    try:
        ts = datetime.fromisoformat(state.phase_updated_at)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return age > max_age_seconds


def clear_state() -> None:
    """Remove the state file. Idempotent."""
    try:
        state_file_path().unlink()
    except FileNotFoundError:
        pass


def read_config() -> UserConfig:
    """Read ``~/.ocrscout/config.yaml`` or return defaults if absent.

    Env-var overrides (``OCRSCOUT_GPU_TYPE`` / ``OCRSCOUT_COST_PER_HOUR``
    / ``OCRSCOUT_PROVIDER``) take precedence over file fields for the
    GPU block. This is how SkyPilot/HF workers receive their pricing
    context: the job YAML sets the envs and ocrscout picks them up here.
    """
    path = config_file_path()
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            raise StateError(f"cannot read {path}: {e}") from e
        try:
            config = UserConfig.model_validate(raw)
        except ValidationError as e:
            raise StateError(f"invalid config in {path}: {e}") from e
    else:
        config = UserConfig()

    env_type = os.environ.get("OCRSCOUT_GPU_TYPE")
    env_cost = os.environ.get("OCRSCOUT_COST_PER_HOUR")
    env_provider = os.environ.get("OCRSCOUT_PROVIDER")
    if env_type or env_cost or env_provider:
        config.gpu = GpuConfig(
            type=env_type or config.gpu.type,
            cost_per_hour=float(env_cost) if env_cost else config.gpu.cost_per_hour,
            provider=env_provider or config.gpu.provider,
        )
    return config


def write_config(config: UserConfig) -> None:
    _atomic_write_yaml(config_file_path(), config.model_dump(mode="json"))

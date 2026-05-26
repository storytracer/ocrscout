"""LocalRunner: vLLM + LiteLLM as daemonised processes on this machine.

For development, testing, and single-GPU work (DGX Spark, workstation).
Spawns one ``vllm serve`` per ``runtime: vllm`` profile and one LiteLLM
proxy that fronts them all under ``localhost:<proxy_port>``. PIDs land in
``~/.ocrscout/pids/`` and logs in ``~/.ocrscout/logs/`` so subsequent CLI
invocations can ``status`` / ``logs`` / ``down`` without keeping the
launching shell open.

``runtime: hosted`` profiles are added to the LiteLLM model_list (so the
proxy knows how to route their calls) but spawn nothing locally — they
forward to the provider API via LiteLLM's built-in routing.

``runtime: cpu`` profiles aren't touched here; their backends run
in-process via the dispatcher and don't go through the proxy.

``submit()`` is intentionally a no-op stub in Phase 1: page dispatch
still flows through ``ocrscout run`` for now. Phase 1 step 6 wires
``submit`` to a daemonised worker process and adds the new ``submit``
CLI command.
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import yaml

from ocrscout import state as state_mod
from ocrscout.errors import RunnerError
from ocrscout.exports.layout import find_parquet_files
from ocrscout.interfaces.runner import Runner
from ocrscout.profile import ModelProfile, effective_vllm_engine_args, resolve
from ocrscout.runners._daemon import (
    daemonize_subprocess,
    pid_alive,
    read_pid,
    terminate,
    uptime_seconds,
)
from ocrscout.runners._preflight import preflight_kv_budgets
from ocrscout.state import GpuConfig, ManagedProcess, RunnerStateFile
from ocrscout.types import (
    JobHandle,
    PipelineConfig,
    RunnerEndpoint,
    RunnerHandle,
    RunnerStatus,
)

log = logging.getLogger(__name__)

_DEFAULT_BASE_PORT = 8000
_DEFAULT_PROXY_PORT = 4000
_DEFAULT_GPU_BUDGET = 0.85
_DEFAULT_READY_TIMEOUT = 600.0
_LITELLM_VERSION_SPEC = ">=1.50.0"

# vLLM args owned by the runner (port + GPU memory cap come from launch
# context, not the profile); everything else flows from
# ``effective_vllm_engine_args(profile)``.
_OWNED_ENGINE_KEYS = frozenset({"gpu_memory_utilization", "port"})


class LocalRunner(Runner):
    """Daemonised vLLM + LiteLLM stack on this machine."""

    name = "local"
    requires_local_gpu: ClassVar[bool] = True

    def launch(
        self,
        *,
        models: list[str],
        gpu_type: str | None = None,
        workers: int = 1,
        base_port: int = _DEFAULT_BASE_PORT,
        proxy_port: int = _DEFAULT_PROXY_PORT,
        gpu_budget: float = _DEFAULT_GPU_BUDGET,
        ready_timeout: float = _DEFAULT_READY_TIMEOUT,
        **_: Any,
    ) -> RunnerHandle:
        if not models:
            raise RunnerError("LocalRunner.launch: at least one model required")

        existing = state_mod.read_state()
        if existing is not None and existing.runner == self.name:
            if _matches_existing(existing, models, proxy_port):
                log.info(
                    "LocalRunner: reusing already-launched stack (proxy=%s, models=%s)",
                    existing.proxy_url, ",".join(existing.models),
                )
                return _handle_from_state(existing)
            raise RunnerError(
                f"LocalRunner: a stack is already active (runner={existing.runner!r}, "
                f"models={existing.models!r}); call `ocrscout down` before relaunching "
                f"with a different config."
            )

        profiles = [resolve(name) for name in models]
        vllm_profiles = [p for p in profiles if p.runtime == "vllm"]
        hosted_profiles = [p for p in profiles if p.runtime == "hosted"]

        # Allocate ports up front so failed spawns don't leave port-shifting
        # ambiguity on re-launch.
        vllm_ports = _allocate_ports(base_port, len(vllm_profiles))
        gpu_caps: dict[str, float] = {}

        if vllm_profiles:
            summary, gpu_caps = preflight_kv_budgets(vllm_profiles, gpu_budget)
            log.info(summary)

        processes: list[ManagedProcess] = []
        try:
            for profile, port in zip(vllm_profiles, vllm_ports):
                proc = _spawn_vllm_serve(
                    profile=profile,
                    port=port,
                    gpu_memory_utilization=gpu_caps[profile.name],
                )
                processes.append(proc)

            log.info(
                "Spawned %d vllm-serve daemon(s); waiting up to %.0fs for ready",
                len(processes), ready_timeout,
            )
            _wait_all_ready_serves(processes, ready_timeout)

            litellm_proc = _spawn_litellm_proxy(
                vllm_profiles=vllm_profiles,
                vllm_ports=vllm_ports,
                hosted_profiles=hosted_profiles,
                proxy_port=proxy_port,
            )
            processes.append(litellm_proc)
            log.info("Spawned litellm proxy; waiting for ready")
            _wait_proxy_ready(proxy_port, ready_timeout)
        except BaseException:
            # If anything fails mid-launch, terminate whatever we spawned
            # so we don't leak GPU memory or ports.
            for proc in processes:
                if proc.pid is not None:
                    terminate(proc.pid)
            raise

        proxy_url = f"http://localhost:{proxy_port}/v1"
        gpu_cfg = _gpu_config_for_launch(gpu_type)

        state = RunnerStateFile(
            runner=self.name,
            models=models,
            proxy_url=proxy_url,
            processes=processes,
            gpu=gpu_cfg,
            launched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            args={
                "base_port": base_port,
                "proxy_port": proxy_port,
                "gpu_budget": gpu_budget,
            },
        )
        state_mod.write_state(state)
        log.info("LocalRunner ready: proxy=%s  models=%s", proxy_url, ",".join(models))
        return _handle_from_state(state)

    def submit(
        self,
        *,
        config: PipelineConfig,
        resume: bool = False,
        **_: Any,
    ) -> JobHandle:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            raise RunnerError(
                "LocalRunner.submit: no active stack. Call "
                "`ocrscout launch --models ...` first."
            )
        if resume and config.output_dir is None:
            raise RunnerError(
                "LocalRunner.submit: --resume requires an --output-dir "
                "with an existing progress.json"
            )

        job_id = uuid.uuid4().hex[:12]
        job_dir = state_mod.jobs_dir() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_path = job_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

        # Persist initial job metadata so `ocrscout status` reflects the
        # submission even before the worker writes its first heartbeat.
        (job_dir / "state.yaml").write_text(
            yaml.safe_dump(
                {
                    "status": "submitted",
                    "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "resume": resume,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        # Daemonise ``ocrscout _worker --job <id>``. Use the same
        # interpreter / module entry point ocrscout was invoked with so
        # we don't pin a particular venv path.
        ocrscout_cmd = _ocrscout_invocation()
        cmd = [*ocrscout_cmd, "_worker", "--job", job_id]
        pid = daemonize_subprocess(
            cmd,
            log_path=job_dir / "worker.log",
            pid_path=job_dir / "worker.pid",
        )

        # Update state.yaml's output_dir on the very first submit so
        # subsequent ``status`` calls know where to count rows. Subsequent
        # submits in the same launch reuse the same output_dir or set a
        # new one; we record whatever this submit pointed at.
        state.output_dir = str(config.output_dir) if config.output_dir else state.output_dir
        state_mod.write_state(state)

        log.info("LocalRunner.submit: job %s spawned (pid %d)", job_id, pid)
        return JobHandle(
            job_id=job_id,
            runner=self.name,
            output_dir=str(config.output_dir) if config.output_dir else "",
        )

    def status(self) -> RunnerStatus:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            return RunnerStatus(runner=self.name, state="down")

        all_alive = True
        oldest_uptime: float | None = None
        for proc in state.processes:
            if proc.pid is None:
                continue
            if not pid_alive(proc.pid):
                all_alive = False
                continue
            up = uptime_seconds(proc.pid)
            if up is not None:
                oldest_uptime = up if oldest_uptime is None else max(oldest_uptime, up)

        pages_done = 0
        cumulative_cost = 0.0
        if state.output_dir:
            pages_done, cumulative_cost = _count_pages_and_cost(Path(state.output_dir))

        return RunnerStatus(
            runner=self.name,
            state="ready" if all_alive else "error",
            models=state.models,
            proxy_url=state.proxy_url,
            uptime_seconds=oldest_uptime,
            pages_done=pages_done,
            cumulative_cost=cumulative_cost,
            details={
                "processes": [
                    {
                        "name": p.name,
                        "pid": p.pid,
                        "port": p.port,
                        "alive": pid_alive(p.pid) if p.pid else False,
                    }
                    for p in state.processes
                ],
            },
        )

    def logs(self, job_id: str | None = None, *, follow: bool = True) -> None:
        if job_id is not None:
            job_log = state_mod.jobs_dir() / job_id / "worker.log"
            if not job_log.is_file():
                log.warning("LocalRunner: no log for job %s at %s", job_id, job_log)
                return
            _tail_files([(f"job:{job_id}", job_log)], follow=follow)
            return

        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            log.warning("LocalRunner: no active stack; nothing to tail")
            return
        targets: list[tuple[str, Path]] = []
        for proc in state.processes:
            if proc.log_path:
                targets.append((proc.name, Path(proc.log_path)))
        if not targets:
            log.warning("LocalRunner: no log paths recorded on active stack")
            return
        _tail_files(targets, follow=follow)

    def down(self) -> None:
        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            log.info("LocalRunner: nothing to tear down")
            return
        # Reverse order: proxy first, then upstreams. Matches the original
        # context-manager teardown ordering so partial failures during the
        # window when only some upstreams have shut down don't surface as
        # "model X gone from proxy".
        for proc in reversed(state.processes):
            if proc.pid is None:
                continue
            terminate(proc.pid)
            if proc.name:
                pid_path = state_mod.pid_dir() / f"{proc.name}.pid"
                try:
                    pid_path.unlink()
                except FileNotFoundError:
                    pass
        state_mod.clear_state()
        log.info("LocalRunner: torn down")


# --- spawn helpers ---------------------------------------------------------


def _spawn_vllm_serve(
    *,
    profile: ModelProfile,
    port: int,
    gpu_memory_utilization: float,
) -> ManagedProcess:
    name = _proc_name("vllm", profile.name)
    pid_path = state_mod.pid_dir() / f"{name}.pid"
    log_path = state_mod.log_dir() / f"{name}.log"

    cmd: list[str] = [
        "uv", "run",
        "--with", f"vllm{profile.vllm_version}",
        "--", "vllm", "serve",
        profile.model_id,
        "--port", str(port),
        "--gpu-memory-utilization", f"{gpu_memory_utilization:.4f}",
    ]
    cmd += _engine_args_to_cli(effective_vllm_engine_args(profile))

    log.debug("daemonising %s on port %d -> %s", name, port, log_path)
    pid = daemonize_subprocess(cmd, log_path=log_path, pid_path=pid_path)
    return ManagedProcess(
        name=name, pid=pid, port=port, log_path=str(log_path)
    )


def _spawn_litellm_proxy(
    *,
    vllm_profiles: Sequence[ModelProfile],
    vllm_ports: Sequence[int],
    hosted_profiles: Sequence[ModelProfile],
    proxy_port: int,
) -> ManagedProcess:
    config_path = state_mod.litellm_config_path()
    model_list: list[dict[str, Any]] = []

    for profile, port in zip(vllm_profiles, vllm_ports):
        model_list.append(
            {
                "model_name": profile.name,
                "litellm_params": {
                    "model": f"openai/{profile.model_id}",
                    "api_base": f"http://localhost:{port}/v1",
                    "api_key": "ocrscout-dummy",
                },
            }
        )

    for profile in hosted_profiles:
        # Hosted entries trust the profile's ``model_id`` to be in
        # LiteLLM's expected ``provider/<id>`` form (e.g. ``gemini/...``,
        # ``anthropic/...``). The provider-native API key comes from the
        # user's environment (``GEMINI_API_KEY`` etc.) and LiteLLM picks
        # it up automatically.
        params: dict[str, Any] = {"model": profile.model_id}
        params.update(profile.backend_args or {})
        entry: dict[str, Any] = {"model_name": profile.name, "litellm_params": params}
        pricing = (profile.metadata or {}).get("pricing")
        if isinstance(pricing, dict):
            entry["model_info"] = pricing
        model_list.append(entry)

    config = {"model_list": model_list}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    name = "litellm"
    pid_path = state_mod.pid_dir() / f"{name}.pid"
    log_path = state_mod.log_dir() / f"{name}.log"

    cmd = [
        "uv", "run",
        "--with", f"litellm[proxy]{_LITELLM_VERSION_SPEC}",
        "--", "litellm",
        "--config", str(config_path),
        "--port", str(proxy_port),
        "--num_workers", "1",
    ]
    log.debug("daemonising litellm proxy on port %d -> %s", proxy_port, log_path)
    pid = daemonize_subprocess(cmd, log_path=log_path, pid_path=pid_path)
    return ManagedProcess(
        name=name, pid=pid, port=proxy_port, log_path=str(log_path)
    )


def _engine_args_to_cli(engine_args: dict) -> list[str]:
    """Translate ``vllm_engine_args`` into ``vllm serve`` CLI flags."""
    out: list[str] = []
    for key, value in engine_args.items():
        if key in _OWNED_ENGINE_KEYS or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif isinstance(value, (int, float, str)):
            out += [flag, str(value)]
        elif isinstance(value, list):
            out += [flag, *(str(v) for v in value)]
        elif isinstance(value, dict):
            out += [flag, json.dumps(value)]
        else:
            log.warning(
                "Ignoring vllm_engine_args[%r] of unsupported type %s",
                key, type(value).__name__,
            )
    return out


# --- readiness probes ------------------------------------------------------


def _wait_all_ready_serves(processes: Sequence[ManagedProcess], timeout: float) -> None:
    """Wait until every vllm-serve daemon's ``/v1/models`` returns 200."""
    deadline = time.monotonic() + timeout
    for proc in processes:
        if proc.port is None or proc.pid is None:
            continue
        _wait_one(
            url=f"http://localhost:{proc.port}/v1/models",
            pid=proc.pid,
            label=proc.name,
            log_path=Path(proc.log_path) if proc.log_path else None,
            deadline=deadline,
        )


def _wait_proxy_ready(proxy_port: int, timeout: float) -> None:
    """Probe the LiteLLM proxy's liveliness endpoint."""
    pid_path = state_mod.pid_dir() / "litellm.pid"
    pid = read_pid(pid_path)
    log_path = state_mod.log_dir() / "litellm.log"
    _wait_one(
        url=f"http://localhost:{proxy_port}/health/liveliness",
        pid=pid,
        label="litellm",
        log_path=log_path,
        deadline=time.monotonic() + timeout,
    )


def _wait_one(
    *,
    url: str,
    pid: int | None,
    label: str,
    log_path: Path | None,
    deadline: float,
) -> None:
    """Poll ``url`` until it returns < 400; raise if pid dies or deadline hits."""
    while time.monotonic() < deadline:
        if pid is not None and not pid_alive(pid):
            tail = _tail(log_path, 50) if log_path else ""
            raise RunnerError(
                f"{label}: daemon died before becoming ready\n"
                f"--- last 50 lines of {log_path} ---\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 400:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            pass
        time.sleep(2.0)
    tail = _tail(log_path, 50) if log_path else ""
    raise RunnerError(
        f"{label}: did not respond at {url} within deadline\n"
        f"--- last 50 lines of {log_path} ---\n{tail}"
    )


def _tail(path: Path | None, n: int) -> str:
    if path is None or not path.is_file():
        return "(no log file)"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(error reading log: {e})"
    return "\n".join(text.splitlines()[-n:])


# --- status / dollars / pages ----------------------------------------------


def _count_pages_and_cost(output_dir: Path) -> tuple[int, float]:
    """Sum ``pages_done`` (row count) and ``cumulative_cost`` across parquet shards.

    Falls back to a fast row-count scan when polars isn't importable
    (e.g. minimal CPU-only test environment).
    """
    shards = find_parquet_files(output_dir)
    if not shards:
        return 0, 0.0
    try:
        import polars as pl

        df = pl.scan_parquet([str(s) for s in shards])
        agg = df.select(
            pl.len().alias("rows"),
            (pl.col("litellm_cost").fill_null(0.0).sum()
             + pl.col("gpu_time_cost").fill_null(0.0).sum()).alias("cost"),
        ).collect()
        return int(agg["rows"][0]), float(agg["cost"][0])
    except Exception as e:  # noqa: BLE001
        log.debug("polars row-count failed (%s); skipping cost aggregation", e)
        return 0, 0.0


# --- tail / logs -----------------------------------------------------------


def _tail_files(targets: list[tuple[str, Path]], *, follow: bool) -> None:
    """Tail multiple log files with ``[name]`` prefixes.

    Single-pass when ``follow=False`` (prints the tail of each file and
    returns); otherwise uses ``tail -F`` semantics via a polling loop.
    """
    if not follow:
        for name, path in targets:
            if path.is_file():
                sys.stdout.write(f"--- [{name}] {path} ---\n")
                sys.stdout.write(_tail(path, 200))
                sys.stdout.write("\n")
        return

    positions: dict[Path, int] = {}
    for _, path in targets:
        if path.is_file():
            positions[path] = path.stat().st_size
    try:
        while True:
            any_new = False
            for name, path in targets:
                if not path.is_file():
                    continue
                cur = path.stat().st_size
                last = positions.get(path, cur)
                if cur > last:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(last)
                        for line in f:
                            sys.stdout.write(f"[{name}] {line}")
                            any_new = True
                    positions[path] = cur
                elif cur < last:
                    # File rotated / truncated — reset.
                    positions[path] = cur
            sys.stdout.flush()
            if not any_new:
                time.sleep(0.5)
    except KeyboardInterrupt:
        return


# --- misc helpers ----------------------------------------------------------


def _proc_name(role: str, profile_name: str) -> str:
    """Filename-safe daemon name. Used for PID and log file naming."""
    safe = profile_name.replace("/", "_").replace(" ", "_")
    return f"{role}-{safe}"


def _allocate_ports(base: int, count: int) -> list[int]:
    """Pick ``count`` distinct localhost ports starting from ``base``."""
    out: list[int] = []
    candidate = base
    while len(out) < count:
        if _port_is_free(candidate):
            out.append(candidate)
        candidate += 1
        if candidate - base > 100:
            raise RunnerError(
                f"could not find {count} free ports starting at {base}"
            )
    return out


def _port_is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _gpu_config_for_launch(gpu_type: str | None) -> GpuConfig:
    """Resolve a launch-time GpuConfig: explicit override → config → defaults."""
    base = state_mod.read_config().gpu
    if gpu_type:
        return GpuConfig(
            type=gpu_type,
            cost_per_hour=base.cost_per_hour,
            provider=base.provider,
        )
    return base


def _matches_existing(
    existing: RunnerStateFile, models: list[str], proxy_port: int
) -> bool:
    """Whether an active state file matches the launch we're about to do.

    Same models (order-insensitive) + same proxy port = reuse the stack.
    """
    if set(existing.models) != set(models):
        return False
    if existing.proxy_url is None:
        return False
    expected = f"http://localhost:{proxy_port}/v1"
    return existing.proxy_url == expected


def _ocrscout_invocation() -> list[str]:
    """Argv prefix for spawning a child ``ocrscout`` process.

    Prefers an installed ``ocrscout`` console_script entry over
    ``python -m`` because the entry point sets up the package path
    correctly under all install layouts (uv-tool, pip, editable). Falls
    back to ``sys.executable -m ocrscout`` when the entry script can't be
    located on PATH (e.g. running from a checkout without ``pip install
    -e``).
    """
    ocrscout_path = shutil.which("ocrscout")
    if ocrscout_path:
        return [ocrscout_path]
    return [sys.executable, "-m", "ocrscout"]


def _handle_from_state(state: RunnerStateFile) -> RunnerHandle:
    endpoints = [
        RunnerEndpoint(
            model=proc.name.removeprefix("vllm-"),
            url=f"http://localhost:{proc.port}/v1" if proc.port else "",
            pid=proc.pid,
        )
        for proc in state.processes
        if proc.name.startswith("vllm-") and proc.port is not None
    ]
    return RunnerHandle(
        runner=state.runner,
        proxy_url=state.proxy_url or "",
        endpoints=endpoints,
        extra={"models": state.models},
    )

"""LocalRunner: vLLM + LiteLLM as locally-managed processes.

Two lifecycle modes, selected by the ``persistent`` kwarg on ``launch``:

**Persistent** (``persistent=True``, default — used by ``ocrscout launch``
and ``ocrscout run --keep-up``). Each child is spawned via classic UNIX
double-fork in :mod:`ocrscout.runners._daemon` and reparented to init,
so the stack survives terminal close. PIDs land in ``~/.ocrscout/pids/``
and logs in ``~/.ocrscout/logs/``. ``state.yaml`` tracks a ``phase``
field (``launching`` → ``ready`` → ``tearing_down``) so a Ctrl-C
mid-launch leaves a detectable breadcrumb. After each readiness probe
passes, the runner queries the actual port listener PID via
:func:`ocrscout.runners._ports.resolve_listener_pid` and rewrites the
recorded PID — necessary because ``uv run --with vllm -- vllm serve``
exits its transient resolver subprocess before the real server is up.

**Ephemeral** (``persistent=False`` — used by default ``ocrscout run``
and ``ocrscout apply``). Each child is spawned by
:class:`ocrscout.runners._ephemeral.EphemeralStack` via
``subprocess.Popen`` with ``PR_SET_PDEATHSIG=SIGTERM`` on Linux plus
atexit + signal handlers as cross-platform fallback. The stack dies
when this Python process dies. NO ``state.yaml`` is written; NO PID
files. ``ocrscout submit`` after an ephemeral run errors cleanly with
"no active stack" — submit is stateful, ephemeral opts out.

``runtime: hosted`` profiles are added to the LiteLLM model_list (so
the proxy knows how to route their calls) but spawn nothing locally —
they forward to the provider API via LiteLLM's built-in routing.

``runtime: cpu`` profiles aren't touched here; their backends run
in-process via the dispatcher and don't go through the proxy.

``submit()`` daemonises an ``ocrscout _worker`` process — it requires a
persistent stack (raises ``RunnerError`` if state.yaml is missing).
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
from ocrscout.runners import _ephemeral, _ports
from ocrscout.runners._daemon import (
    daemonize_subprocess,
    pid_alive,
    read_pid,
    terminate,
    uptime_seconds,
)
from ocrscout.runners._preflight import (
    AUTOSCALE_MAX_CONCURRENCY,
    AUTOSCALE_PER_TOKEN_BYTES,
    AutoscaleDecision,
    autoscale_kv_budgets,
    estimate_model_overhead,
    parse_bytes,
    preflight_kv_budgets,
    probe_gpu,
)
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
_PORT_SCAN_RANGE = 32  # how many vLLM ports past base_port to scan in --force

# vLLM args owned by the runner (port + GPU memory cap come from launch
# context, not the profile); everything else flows from
# ``effective_vllm_engine_args(profile)``.
_OWNED_ENGINE_KEYS = frozenset({"gpu_memory_utilization", "port"})


class LocalRunner(Runner):
    """Local vLLM + LiteLLM stack (persistent daemonised or ephemeral)."""

    name = "local"
    requires_local_gpu: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        # Reference to an in-process EphemeralStack when ``launch`` was
        # called with ``persistent=False`` on this instance. None for
        # persistent launches and after a successful ``down``.
        self._ephemeral_stack: _ephemeral.EphemeralStack | None = None

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
        persistent: bool = True,
        batch_concurrency: int | None = None,
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

        if persistent:
            return self._launch_persistent(
                models=models, gpu_type=gpu_type, base_port=base_port,
                proxy_port=proxy_port, gpu_budget=gpu_budget,
                ready_timeout=ready_timeout,
                batch_concurrency=batch_concurrency,
            )
        return self._launch_ephemeral(
            models=models, gpu_type=gpu_type, base_port=base_port,
            proxy_port=proxy_port, gpu_budget=gpu_budget,
            ready_timeout=ready_timeout,
            batch_concurrency=batch_concurrency,
        )

    # --- persistent path -----------------------------------------------------

    def _launch_persistent(
        self,
        *,
        models: list[str],
        gpu_type: str | None,
        base_port: int,
        proxy_port: int,
        gpu_budget: float,
        ready_timeout: float,
        batch_concurrency: int | None,
    ) -> RunnerHandle:
        profiles = [resolve(name) for name in models]
        vllm_profiles = [p for p in profiles if p.runtime == "vllm"]
        hosted_profiles = [p for p in profiles if p.runtime == "hosted"]

        vllm_ports = _allocate_ports(base_port, len(vllm_profiles))
        gpu_caps: dict[str, float] = {}
        autoscale_extra: dict[str, Any] = {}

        if vllm_profiles:
            autoscale_extra = _autoscale_and_apply(
                vllm_profiles=vllm_profiles,
                gpu_budget=gpu_budget,
                batch_concurrency=batch_concurrency,
            )
            summary, gpu_caps = preflight_kv_budgets(vllm_profiles, gpu_budget)
            log.info(summary)

        gpu_cfg = _gpu_config_for_launch(gpu_type)
        proxy_url = f"http://localhost:{proxy_port}/v1"

        # Write the crash breadcrumb BEFORE any spawn. An empty
        # ``processes`` list at this point is intentional — phase ==
        # 'launching' means "do not trust these PIDs"; subsequent
        # ``update_state_processes`` calls append corrected entries as
        # daemons come up.
        backend_overrides = _backend_overrides_from_autoscale(autoscale_extra)
        args_record: dict[str, Any] = {
            "base_port": base_port,
            "proxy_port": proxy_port,
            "gpu_budget": gpu_budget,
        }
        if batch_concurrency is not None:
            args_record["batch_concurrency"] = batch_concurrency
        if autoscale_extra:
            # Stash the full autoscale context so reused launches and
            # subsequent ocrscout commands can render a runtime.yaml
            # without re-probing the GPU.
            args_record["autoscale"] = autoscale_extra
        state = RunnerStateFile(
            runner=self.name,
            models=models,
            proxy_url=proxy_url,
            processes=[],
            gpu=gpu_cfg,
            launched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            args=args_record,
            backend_overrides=backend_overrides,
        )
        state_mod.write_state_launching(state)

        try:
            for profile, port in zip(vllm_profiles, vllm_ports):
                proc = _spawn_vllm_serve(
                    profile=profile,
                    port=port,
                    gpu_memory_utilization=gpu_caps[profile.name],
                )
                log.info(
                    "Spawned vllm serve for %s on port %d (initial pid=%d); "
                    "awaiting readiness",
                    profile.name, port, proc.pid or -1,
                )
                _wait_one(
                    url=f"http://localhost:{port}/v1/models",
                    pid=proc.pid,
                    label=proc.name,
                    log_path=Path(proc.log_path) if proc.log_path else None,
                    deadline=time.monotonic() + ready_timeout,
                )
                _correct_recorded_pid(proc, port=port)
                state.processes.append(proc)
                state_mod.update_state_processes(state)

            log.info("Spawned %d vllm-serve daemon(s); spawning litellm proxy",
                     len(vllm_profiles))

            _write_litellm_config(
                vllm_profiles=vllm_profiles,
                vllm_ports=vllm_ports,
                hosted_profiles=hosted_profiles,
            )
            litellm_proc = _spawn_litellm_proxy(proxy_port=proxy_port)
            log.info("Spawned litellm proxy (initial pid=%d); awaiting readiness",
                     litellm_proc.pid or -1)
            _wait_one(
                url=f"http://localhost:{proxy_port}/health/liveliness",
                pid=litellm_proc.pid,
                label=litellm_proc.name,
                log_path=Path(litellm_proc.log_path) if litellm_proc.log_path else None,
                deadline=time.monotonic() + ready_timeout,
            )
            _correct_recorded_pid(litellm_proc, port=proxy_port)
            state.processes.append(litellm_proc)
            state_mod.update_state_processes(state)
        except BaseException:
            # Mid-launch failure: terminate whatever we spawned so we
            # don't leak GPU memory or ports. The crash breadcrumb
            # (phase='launching') is left on disk so the user can run
            # `ocrscout down --force` if our cleanup itself was
            # interrupted.
            for proc in state.processes:
                if proc.pid is not None:
                    terminate(proc.pid)
            raise

        state_mod.mark_phase_ready(state)
        log.info("LocalRunner ready: proxy=%s  models=%s",
                 proxy_url, ",".join(models))
        return _handle_from_state(state)

    # `_handle_from_state` reads `state.args["autoscale"]` so reused
    # launches and the immediate post-launch handle both carry the
    # autoscale context for downstream consumers (runtime.yaml writer).

    # --- ephemeral path ------------------------------------------------------

    def _launch_ephemeral(
        self,
        *,
        models: list[str],
        gpu_type: str | None,
        base_port: int,
        proxy_port: int,
        gpu_budget: float,
        ready_timeout: float,
        batch_concurrency: int | None,
    ) -> RunnerHandle:
        if self._ephemeral_stack is not None and not self._ephemeral_stack._closed:
            # Model-major chunking on the CLI side always calls down() before
            # the next launch, so this is defense-in-depth: if a caller forgot,
            # we refuse to leak GPU memory by stacking a second set of serves.
            raise RunnerError(
                "LocalRunner: previous ephemeral stack not torn down; "
                "call .down() before launching another chunk."
            )

        profiles = [resolve(name) for name in models]
        vllm_profiles = [p for p in profiles if p.runtime == "vllm"]
        hosted_profiles = [p for p in profiles if p.runtime == "hosted"]
        vllm_ports = _allocate_ports(base_port, len(vllm_profiles))

        gpu_caps: dict[str, float] = {}
        autoscale_extra: dict[str, Any] = {}
        if vllm_profiles:
            autoscale_extra = _autoscale_and_apply(
                vllm_profiles=vllm_profiles,
                gpu_budget=gpu_budget,
                batch_concurrency=batch_concurrency,
            )
            summary, gpu_caps = preflight_kv_budgets(vllm_profiles, gpu_budget)
            log.info(summary)

        stack = _ephemeral.EphemeralStack()
        self._ephemeral_stack = stack

        try:
            for profile, port in zip(vllm_profiles, vllm_ports):
                cmd = _vllm_serve_cmd(
                    profile=profile, port=port,
                    gpu_memory_utilization=gpu_caps[profile.name],
                )
                log_path = state_mod.log_dir() / f"vllm-{_safe(profile.name)}.log"
                stack.spawn(cmd, name=f"vllm-{_safe(profile.name)}",
                            log_path=log_path, port=port)
                log.info(
                    "Ephemeral vllm serve for %s on port %d (pid=%d); "
                    "awaiting readiness",
                    profile.name, port, stack.processes[-1].popen.pid,
                )
                _wait_one(
                    url=f"http://localhost:{port}/v1/models",
                    pid=stack.processes[-1].popen.pid,
                    label=stack.processes[-1].name,
                    log_path=log_path,
                    deadline=time.monotonic() + ready_timeout,
                )

            _write_litellm_config(
                vllm_profiles=vllm_profiles,
                vllm_ports=vllm_ports,
                hosted_profiles=hosted_profiles,
            )
            litellm_cmd = _litellm_proxy_cmd(proxy_port=proxy_port)
            litellm_log = state_mod.log_dir() / "litellm.log"
            stack.spawn(litellm_cmd, name="litellm",
                        log_path=litellm_log, port=proxy_port)
            log.info(
                "Ephemeral litellm proxy (pid=%d); awaiting readiness",
                stack.processes[-1].popen.pid,
            )
            _wait_one(
                url=f"http://localhost:{proxy_port}/health/liveliness",
                pid=stack.processes[-1].popen.pid,
                label="litellm",
                log_path=litellm_log,
                deadline=time.monotonic() + ready_timeout,
            )
        except BaseException:
            stack.terminate_all()
            self._ephemeral_stack = None
            raise

        proxy_url = f"http://localhost:{proxy_port}/v1"
        log.info("LocalRunner ephemeral ready: proxy=%s  models=%s",
                 proxy_url, ",".join(models))
        return _handle_from_ephemeral_stack(
            stack, proxy_url, models, autoscale=autoscale_extra,
        )

    # --- submit / status / logs / down --------------------------------------

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

        ocrscout_cmd = _ocrscout_invocation()
        cmd = [*ocrscout_cmd, "_worker", "--job", job_id]
        pid = daemonize_subprocess(
            cmd,
            log_path=job_dir / "worker.log",
            pid_path=job_dir / "worker.pid",
        )

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

        if state.phase == "ready":
            health = "ready" if all_alive else "error"
        elif state.phase == "launching":
            health = "launching"
        else:
            health = "tearing_down"

        return RunnerStatus(
            runner=self.name,
            state=health,
            models=state.models,
            proxy_url=state.proxy_url,
            uptime_seconds=oldest_uptime,
            pages_done=pages_done,
            cumulative_cost=cumulative_cost,
            details={
                "phase": state.phase,
                "phase_updated_at": state.phase_updated_at,
                "stale_launching": state_mod.is_stale_launching(state),
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

    def down(self, *, force: bool = False, **_: Any) -> None:
        # In-process ephemeral stack? Terminate via the Popen tree.
        if self._ephemeral_stack is not None:
            self._ephemeral_stack.terminate_all()
            self._ephemeral_stack = None
            return

        state = state_mod.read_state()
        if state is None or state.runner != self.name:
            if force:
                self._down_by_port_scan(
                    base_port=_DEFAULT_BASE_PORT, proxy_port=_DEFAULT_PROXY_PORT,
                )
            else:
                log.info("LocalRunner: nothing to tear down")
            return

        # Mark teardown in progress so a concurrent `status` poll doesn't
        # try to reuse a half-killed stack.
        state.phase = "tearing_down"
        state_mod.update_state_processes(state)

        # Reverse order matches the original context-manager teardown
        # ordering: proxy first, then upstreams. Partial-failure windows
        # don't surface "model X gone from proxy" errors.
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

        if force:
            self._down_by_port_scan(
                base_port=_DEFAULT_BASE_PORT, proxy_port=_DEFAULT_PROXY_PORT,
            )

        state_mod.clear_state()
        log.info("LocalRunner: torn down")

    def _down_by_port_scan(self, *, base_port: int, proxy_port: int) -> None:
        """Belt-and-suspenders: SIGTERM anything listening on the runner's
        default port range, regardless of whether ``state.yaml`` recorded
        it. Used by ``ocrscout down --force`` to recover from launches
        whose state.yaml never reached ``phase: ready``.
        """
        scan_ports = [proxy_port] + list(
            range(base_port, base_port + _PORT_SCAN_RANGE)
        )
        listeners = _ports.listeners_on_ports(scan_ports)
        if not listeners:
            log.info("Port scan: no orphan listeners on %s", scan_ports)
            return
        log.warning(
            "Port scan: found orphan listeners %s — terminating",
            listeners,
        )
        for port in listeners:
            _ports.kill_listener_on_port(port)


# --- spawn helpers ---------------------------------------------------------


def _vllm_serve_cmd(
    *,
    profile: ModelProfile,
    port: int,
    gpu_memory_utilization: float,
) -> list[str]:
    """Build the ``uv run … -- vllm serve …`` argv for ``profile``.

    Shared by the persistent and ephemeral paths so both produce
    identical processes.

    ``--host 127.0.0.1`` is non-negotiable: vLLM defaults to binding
    ``0.0.0.0``, which on a public-IP cloud GPU box (e.g. Scaleway,
    Lambda) lets anyone in the world submit inference requests to your
    GPU and pollutes the cost callback with foreign traffic. Observed
    in the wild on a 2500-page benchmark run: internet scanners hit
    the API server hundreds of times over a single overnight session.
    All ocrscout consumers reach the daemon via the loopback
    ``OCRSCOUT_VLLM_URL``, so a strict loopback bind is functionally
    identical from our side. The SkyPilot/HF runners spawn their own
    worker-local LocalRunner stack on the worker pod, so they benefit
    from the same lock-down without any code path needing 0.0.0.0.
    """
    cmd: list[str] = [
        "uv", "run",
        "--with", f"vllm{profile.vllm_version}",
        "--", "vllm", "serve",
        profile.model_id,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", f"{gpu_memory_utilization:.4f}",
    ]
    cmd += _engine_args_to_cli(effective_vllm_engine_args(profile))
    return cmd


def _litellm_proxy_cmd(*, proxy_port: int) -> list[str]:
    """Build the ``uv run … -- litellm --config … --port …`` argv.

    Caller is responsible for having written the config file via
    :func:`_write_litellm_config` first.

    Like vLLM (see :func:`_vllm_serve_cmd`), bind the proxy to
    loopback only. LiteLLM defaults to ``0.0.0.0`` and would otherwise
    accept inference requests from anywhere the host's public
    interface is reachable.
    """
    return [
        "uv", "run",
        "--with", f"litellm[proxy]{_LITELLM_VERSION_SPEC}",
        "--", "litellm",
        "--config", str(state_mod.litellm_config_path()),
        "--host", "127.0.0.1",
        "--port", str(proxy_port),
        "--num_workers", "1",
    ]


def _write_litellm_config(
    *,
    vllm_profiles: Sequence[ModelProfile],
    vllm_ports: Sequence[int],
    hosted_profiles: Sequence[ModelProfile],
) -> Path:
    """Render the LiteLLM proxy config to disk. Shared by both paths."""
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
        # LiteLLM's expected ``provider/<id>`` form (e.g. ``gemini/...``).
        # The provider-native API key comes from the user's environment
        # (``GEMINI_API_KEY`` etc.) and LiteLLM picks it up automatically.
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
    return config_path


def _spawn_vllm_serve(
    *,
    profile: ModelProfile,
    port: int,
    gpu_memory_utilization: float,
) -> ManagedProcess:
    """Persistent-path spawn: daemonise ``vllm serve`` via double-fork."""
    name = _proc_name("vllm", profile.name)
    pid_path = state_mod.pid_dir() / f"{name}.pid"
    log_path = state_mod.log_dir() / f"{name}.log"
    cmd = _vllm_serve_cmd(
        profile=profile, port=port,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    log.debug("daemonising %s on port %d -> %s", name, port, log_path)
    pid = daemonize_subprocess(cmd, log_path=log_path, pid_path=pid_path)
    return ManagedProcess(name=name, pid=pid, port=port, log_path=str(log_path))


def _spawn_litellm_proxy(*, proxy_port: int) -> ManagedProcess:
    """Persistent-path spawn: daemonise the LiteLLM proxy via double-fork."""
    name = "litellm"
    pid_path = state_mod.pid_dir() / f"{name}.pid"
    log_path = state_mod.log_dir() / f"{name}.log"
    cmd = _litellm_proxy_cmd(proxy_port=proxy_port)
    log.debug("daemonising litellm proxy on port %d -> %s", proxy_port, log_path)
    pid = daemonize_subprocess(cmd, log_path=log_path, pid_path=pid_path)
    return ManagedProcess(
        name=name, pid=pid, port=proxy_port, log_path=str(log_path)
    )


def _autoscale_and_apply(
    *,
    vllm_profiles: Sequence[ModelProfile],
    gpu_budget: float,
    batch_concurrency: int | None,
) -> dict[str, Any]:
    """Run the GPU-aware autoscaler over ``vllm_profiles`` (those that
    don't declare an explicit ``kv_cache_memory_bytes``), mutate the
    profile objects in place, and return a JSON-serialisable context
    describing the decisions for downstream consumers (runtime.yaml,
    state.yaml).

    Profiles with an explicit YAML value are honored as-is — the
    autoscaler doesn't override user intent — but they still appear in
    the returned context so consumers can render a complete record.
    """
    needs_scaling = [
        p for p in vllm_profiles
        if (p.vllm_engine_args or {}).get("kv_cache_memory_bytes") is None
    ]
    explicit = [
        p for p in vllm_profiles
        if (p.vllm_engine_args or {}).get("kv_cache_memory_bytes") is not None
    ]

    gpu = probe_gpu(gpu_budget)
    scaled: dict[str, AutoscaleDecision] = {}
    if needs_scaling:
        summary, scaled = autoscale_kv_budgets(
            needs_scaling, gpu_budget,
            batch_concurrency=batch_concurrency, probe=gpu,
        )
        log.info(summary)
        for p in needs_scaling:
            d = scaled[p.name]
            _apply_decision_to_profile(p, d)

    if explicit and batch_concurrency is not None:
        # The override is global; warn the user that profile-declared KV
        # is being respected anyway so they're not surprised.
        for p in explicit:
            log.warning(
                "profile %r declares an explicit kv_cache_memory_bytes; "
                "honoring it and ignoring --batch-concurrency for this profile.",
                p.name,
            )

    profile_records: dict[str, dict[str, Any]] = {}
    for p in vllm_profiles:
        if p.name in scaled:
            d = scaled[p.name]
            # Record concurrency under the key the profile's backend
            # actually reads — the other one stays 0 so the runtime.yaml
            # report reflects what the backend will use, not a fictitious
            # "both at once" value.
            is_layout = p.backend == "layout_chat"
            profile_records[p.name] = {
                "explicit_kv_in_yaml": False,
                "overhead_bytes": d.overhead_bytes,
                "kv_cache_memory_bytes": d.kv_cache_memory_bytes,
                "concurrent_requests": 0 if is_layout else d.concurrent_requests,
                "region_concurrency": d.concurrent_requests if is_layout else 0,
                "max_model_len": d.max_model_len,
            }
        else:
            # Explicit profile: read back the values that will reach vLLM.
            kv_raw = (p.vllm_engine_args or {}).get("kv_cache_memory_bytes")
            cr = (p.backend_args or {}).get("concurrent_requests")
            rc = (p.backend_args or {}).get("region_concurrency")
            max_len = int((p.vllm_engine_args or {}).get("max_model_len") or 0)
            profile_records[p.name] = {
                "explicit_kv_in_yaml": True,
                "overhead_bytes": estimate_model_overhead(p),
                "kv_cache_memory_bytes": parse_bytes(kv_raw) if kv_raw is not None else 0,
                "concurrent_requests": int(cr) if cr is not None else 0,
                "region_concurrency": int(rc) if rc is not None else 0,
                "max_model_len": max_len,
            }

    return {
        "gpu": {
            "name": gpu.name,
            "total_bytes": gpu.total_bytes,
            "free_bytes_at_launch": gpu.free_bytes,
            "cap_bytes": gpu.cap_bytes,
            "memory_bandwidth_gb_s": (
                gpu.spec.memory_bandwidth_gb_s if gpu.spec else None
            ),
            "dbgpu_spec_name": gpu.spec.name if gpu.spec else None,
        },
        "gpu_budget": gpu_budget,
        "batch_concurrency_override": batch_concurrency,
        "per_token_bytes": AUTOSCALE_PER_TOKEN_BYTES,
        "max_concurrency_ceiling": AUTOSCALE_MAX_CONCURRENCY,
        "profiles": profile_records,
    }


def _apply_decision_to_profile(
    profile: ModelProfile, decision: AutoscaleDecision
) -> None:
    """Mutate the profile with the autoscaler's KV + concurrency choices.

    Uses ``setdefault`` on the relevant backend_args key so an explicit
    profile value (rare — most profiles don't set this) still wins. Only
    the key the backend will actually read is set: ``concurrent_requests``
    for ``backend: litellm`` (whole-page batches), ``region_concurrency``
    for ``backend: layout_chat`` (per-page region fan-out). Other backends
    fall through to the litellm key as a sensible default.

    Extends ``cudagraph_capture_sizes`` when the chosen concurrency
    exceeds the largest pre-captured bucket; otherwise vLLM would run
    that batch size eagerly, paying ~30% latency for no benefit.
    """
    eng = dict(profile.vllm_engine_args or {})
    eng["kv_cache_memory_bytes"] = decision.kv_cache_memory_bytes

    capture = list(eng.get("cudagraph_capture_sizes") or [])
    if not capture or decision.concurrent_requests > max(capture):
        # Inherit the profile default's existing buckets and add the
        # chosen concurrency. ``cudagraph_capture_sizes`` is overridable
        # per profile; if the YAML set it explicitly, we extend that list.
        from ocrscout.profile import DEFAULT_VLLM_ENGINE_ARGS

        base = capture or list(DEFAULT_VLLM_ENGINE_ARGS["cudagraph_capture_sizes"])
        if decision.concurrent_requests not in base:
            base.append(decision.concurrent_requests)
        eng["cudagraph_capture_sizes"] = sorted(set(base))

    profile.vllm_engine_args = eng

    backend = dict(profile.backend_args or {})
    if profile.backend == "layout_chat":
        backend.setdefault("region_concurrency", decision.concurrent_requests)
    else:
        backend.setdefault("concurrent_requests", decision.concurrent_requests)
    profile.backend_args = backend


def _backend_overrides_from_autoscale(
    autoscale_extra: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Slim down the autoscale context to the per-profile concurrency
    pair the submit-time worker backends consult through state.yaml.
    Empty when the autoscale context is empty (hosted-only runs)."""
    out: dict[str, dict[str, int]] = {}
    for name, rec in (autoscale_extra.get("profiles") or {}).items():
        cr = int(rec.get("concurrent_requests") or 0)
        rc = int(rec.get("region_concurrency") or 0)
        if cr > 0 or rc > 0:
            out[name] = {
                "concurrent_requests": cr,
                "region_concurrency": rc,
            }
    return out


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


# --- PID correction --------------------------------------------------------


def _correct_recorded_pid(proc: ManagedProcess, *, port: int) -> None:
    """Rewrite ``proc.pid`` to the actual port listener if they differ.

    ``daemonize_subprocess`` records the PID of whatever ``execvp`` ran
    — which for ``uv run --with vllm -- vllm serve …`` is the ``uv run``
    wrapper, a transient resolver process that exits as soon as deps
    are synced. The real server is one of its descendants. After the
    readiness probe passes, we ask the kernel which PID owns ``port``
    and update both the in-memory ``ManagedProcess`` and the on-disk
    PID file so ``ocrscout down`` and ``status`` look at the right
    process.

    No-op when the resolver returns None (e.g. ``ss`` and ``/proc/net/tcp``
    both unavailable) — the recorded PID stays as-is and we rely on
    process-group teardown to reach the leaf.
    """
    listener = _ports.resolve_listener_pid(port)
    if listener is None or listener == proc.pid:
        return
    log.info(
        "%s: correcting recorded pid %s → leaf listener %d (port %d)",
        proc.name, proc.pid, listener, port,
    )
    proc.pid = listener
    if proc.name:
        pid_path = state_mod.pid_dir() / f"{proc.name}.pid"
        try:
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = pid_path.with_suffix(pid_path.suffix + ".tmp")
            tmp.write_text(str(listener), encoding="utf-8")
            tmp.replace(pid_path)
        except OSError as e:
            log.warning("could not update pid file %s: %s", pid_path, e)


# --- readiness probes ------------------------------------------------------


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
                f"{label}: process died before becoming ready\n"
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
    """Sum ``pages_done`` (row count) and ``cumulative_cost`` across parquet shards."""
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
    """Tail multiple log files with ``[name]`` prefixes."""
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
                    positions[path] = cur
            sys.stdout.flush()
            if not any_new:
                time.sleep(0.5)
    except KeyboardInterrupt:
        return


# --- misc helpers ----------------------------------------------------------


def _proc_name(role: str, profile_name: str) -> str:
    """Filename-safe daemon name. Used for PID and log file naming."""
    return f"{role}-{_safe(profile_name)}"


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


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
    Only matches when the existing stack is fully ready (``phase ==
    'ready'``) — a stale ``launching`` shouldn't be reused, that's a
    job for ``ocrscout down --force``.
    """
    if existing.phase != "ready":
        return False
    if set(existing.models) != set(models):
        return False
    if existing.proxy_url is None:
        return False
    expected = f"http://localhost:{proxy_port}/v1"
    return existing.proxy_url == expected


def _ocrscout_invocation() -> list[str]:
    """Argv prefix for spawning a child ``ocrscout`` process."""
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
    extra: dict[str, Any] = {"models": state.models}
    autoscale = (state.args or {}).get("autoscale")
    if isinstance(autoscale, dict):
        extra["autoscale"] = autoscale
    batch_concurrency = (state.args or {}).get("batch_concurrency")
    if isinstance(batch_concurrency, int):
        extra["batch_concurrency"] = batch_concurrency
    return RunnerHandle(
        runner=state.runner,
        proxy_url=state.proxy_url or "",
        endpoints=endpoints,
        extra=extra,
    )


def _handle_from_ephemeral_stack(
    stack: _ephemeral.EphemeralStack,
    proxy_url: str,
    models: list[str],
    *,
    autoscale: dict[str, Any] | None = None,
) -> RunnerHandle:
    endpoints = [
        RunnerEndpoint(
            model=p.name.removeprefix("vllm-"),
            url=f"http://localhost:{p.port}/v1" if p.port else "",
            pid=p.popen.pid,
        )
        for p in stack.processes
        if p.name.startswith("vllm-") and p.port is not None
    ]
    extra: dict[str, Any] = {"models": models, "ephemeral": True}
    if autoscale:
        extra["autoscale"] = autoscale
    return RunnerHandle(
        runner="local",
        proxy_url=proxy_url,
        endpoints=endpoints,
        extra=extra,
    )

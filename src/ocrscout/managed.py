"""Managed multi-server orchestration: vllm-serves + LiteLLM proxy.

Spawns one ``vllm serve`` subprocess per vllm-source profile, equally splitting
``--gpu-budget`` of GPU memory between them. When N >= 2, also spawns a
LiteLLM proxy that fronts the N upstreams under a single OpenAI-compatible URL
(routing requests by ``model`` field to the right port). When N == 1, the
single vllm-serve port is the URL — no proxy needed.

The whole stack is exposed as a context manager: enter to spawn + wait-ready,
exit to terminate everything cleanly. Logs land in
``/tmp/ocrscout-managed-<uuid>/`` and survive teardown for debugging.

This module imports nothing GPU-related itself. The vLLM and LiteLLM
dependencies are pulled into per-subprocess uv-managed venvs via
``uv run --with``, preserving the zero-GPU-deps-in-core rule.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml
from nvitop import Device

from ocrscout.errors import ManagedServerError
from ocrscout.profile import ModelProfile

log = logging.getLogger(__name__)

_DEFAULT_GPU_BUDGET = 0.85
_DEFAULT_BASE_PORT = 8000
_DEFAULT_PROXY_PORT = 4000
_DEFAULT_READY_TIMEOUT = 600.0  # seconds
_TEARDOWN_GRACE = 10.0  # seconds to wait after SIGTERM before SIGKILL
_LITELLM_VERSION = ">=1.50.0"

_PR_SET_PDEATHSIG = 1  # Linux: signal to send when parent dies


def _set_pdeathsig() -> None:
    """preexec_fn: ask the kernel to SIGTERM us if ocrscout's main process dies.

    Without this, an abnormal ocrscout exit (SIGKILL, OOM-killer, screen
    `-X quit` cascading through bash) leaves vllm-serve and the LiteLLM
    proxy as orphans adopted by init, holding GPU memory until the user
    manually kills them. PR_SET_PDEATHSIG (Linux 2.1.57+) cures the immediate
    child case; ``uv run``'s default SIGTERM-forwarding then propagates to
    the actual vllm/litellm python process inside.

    Best-effort: silently no-ops on non-Linux or if libc isn't loadable.
    Safe to call from a forked child before exec().
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):
        # No libc, or no prctl symbol — degrade silently. The graceful
        # teardown path still works for normal exits.
        pass


@dataclass
class _ManagedChild:
    """One spawned subprocess with metadata for teardown and error reporting."""

    label: str  # e.g. "vllm:rednote-hilab/dots.mocr" or "litellm-proxy"
    proc: subprocess.Popen
    log_path: Path
    port: int


@dataclass
class ManagedHandle:
    """What the caller sees while inside ``with managed_servers(...)``."""

    proxy_url: str
    """OpenAI-compatible base URL the caller should set as ``OCRSCOUT_VLLM_URL``."""

    server_urls: dict[str, str] = field(default_factory=dict)
    """``model_id -> http://localhost:<port>/v1`` for each vllm-serve. Useful
    for direct upstream debugging without going through the proxy."""

    log_dir: Path = field(default_factory=Path)
    """Where every child's stdout/stderr lands. Not deleted on teardown."""


@contextmanager
def managed_servers(
    profiles: Sequence[ModelProfile],
    *,
    gpu_budget: float = _DEFAULT_GPU_BUDGET,
    base_port: int = _DEFAULT_BASE_PORT,
    proxy_port: int = _DEFAULT_PROXY_PORT,
    ready_timeout: float = _DEFAULT_READY_TIMEOUT,
) -> Iterator[ManagedHandle]:
    """Spawn vllm-serves + (if N>=2) a LiteLLM proxy; tear down on exit.

    Filters ``profiles`` to those with ``source == "vllm"``. If the filtered
    set is empty, yields a handle with an empty proxy_url — callers should
    detect this and skip the env-var injection.
    """
    vllm_profiles = [p for p in profiles if p.source == "vllm"]

    log_dir = Path(tempfile.mkdtemp(prefix=f"ocrscout-managed-{uuid.uuid4().hex[:6]}-"))
    children: list[_ManagedChild] = []

    if not vllm_profiles:
        # Nothing to manage; yield an empty handle. Callers can no-op on this.
        try:
            yield ManagedHandle(proxy_url="", server_urls={}, log_dir=log_dir)
        finally:
            pass
        return

    try:
        per_model_util, alloc_summary = _compute_allocation(gpu_budget, len(vllm_profiles))
        log.info(alloc_summary)

        # Spawn each vllm-serve sequentially from the main thread. We do NOT
        # parallelize via ThreadPoolExecutor here: PR_SET_PDEATHSIG (set in
        # preexec_fn) triggers when the *thread* that called fork() dies, not
        # the parent process. From a worker thread, the death signal would
        # fire as soon as the worker returns — instantly killing the child
        # we just spawned. Forking is fast; only the model-load that follows
        # is slow, and we already parallelize *that* via the readiness probes.
        for i, profile in enumerate(vllm_profiles):
            port = base_port + i
            children.append(
                _spawn_vllm_serve(
                    profile=profile,
                    port=port,
                    gpu_memory_utilization=per_model_util,
                    log_dir=log_dir,
                )
            )

        log.info(
            "Spawned %d vllm-serve children; waiting up to %.0fs for ready...",
            len(children),
            ready_timeout,
        )
        _wait_all_ready(children, timeout=ready_timeout)
        log.info("All vllm-serves ready.")
        log_gpu_state()

        server_urls = {
            p.model_id: f"http://localhost:{base_port + i}/v1"
            for i, p in enumerate(vllm_profiles)
        }

        if len(vllm_profiles) >= 2:
            proxy_child = _spawn_litellm_proxy(
                vllm_profiles=vllm_profiles,
                base_port=base_port,
                proxy_port=proxy_port,
                log_dir=log_dir,
            )
            children.append(proxy_child)
            _wait_one_ready(proxy_child, timeout=ready_timeout)
            proxy_url = f"http://localhost:{proxy_port}/v1"
        else:
            # N == 1: skip the proxy entirely.
            proxy_url = f"http://localhost:{base_port}/v1"

        handle = ManagedHandle(
            proxy_url=proxy_url,
            server_urls=server_urls,
            log_dir=log_dir,
        )
        yield handle
    finally:
        _teardown(children)


# --- subprocess spawning ---------------------------------------------------


def _spawn_vllm_serve(
    *,
    profile: ModelProfile,
    port: int,
    gpu_memory_utilization: float,
    log_dir: Path,
) -> _ManagedChild:
    label = f"vllm:{profile.model_id}"
    safe = _safe_filename(profile.model_id)
    log_path = log_dir / f"vllm-{safe}.log"

    engine = profile.vllm_engine_args or {}
    max_model_len = engine.get("max_model_len")
    trust_remote_code = bool(engine.get("trust_remote_code", False))

    cmd: list[str] = [
        "uv",
        "run",
        "--with",
        f"vllm{profile.vllm_version}",
        "--",
        "vllm",
        "serve",
        profile.model_id,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.4f}",
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    # dtype, served-model-name, etc. are intentionally left to vllm-serve
    # defaults; if a profile needs them, surface them through vllm_engine_args
    # in a follow-up.

    log.info("Spawning %s on port %d -> %s", label, port, log_path)
    log_fh = open(log_path, "wb", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        # New session so the child doesn't share our terminal; teardown sends
        # signals to the leader, which propagates to its descendants.
        start_new_session=True,
        # PR_SET_PDEATHSIG: the kernel SIGTERMs us when ocrscout dies, so an
        # abnormal exit (SIGKILL, screen quit, OOM) doesn't strand orphans
        # holding GPU memory.
        preexec_fn=_set_pdeathsig,
    )
    return _ManagedChild(label=label, proc=proc, log_path=log_path, port=port)


def _spawn_litellm_proxy(
    *,
    vllm_profiles: Sequence[ModelProfile],
    base_port: int,
    proxy_port: int,
    log_dir: Path,
) -> _ManagedChild:
    config_path = log_dir / "litellm.yaml"
    config = {
        "model_list": [
            {
                "model_name": p.model_id,
                "litellm_params": {
                    # The "openai/" prefix tells LiteLLM to use its
                    # OpenAI-compatible client against api_base, not the actual
                    # OpenAI service.
                    "model": f"openai/{p.model_id}",
                    "api_base": f"http://localhost:{base_port + i}/v1",
                    "api_key": "ocrscout-dummy",
                },
            }
            for i, p in enumerate(vllm_profiles)
        ]
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    log_path = log_dir / "litellm.log"
    cmd = [
        "uv",
        "run",
        "--with",
        f"litellm[proxy]{_LITELLM_VERSION}",
        "--",
        "litellm",
        "--config",
        str(config_path),
        "--port",
        str(proxy_port),
        "--num_workers",
        "1",
    ]

    log.info("Spawning litellm-proxy on port %d -> %s", proxy_port, log_path)
    log_fh = open(log_path, "wb", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=_set_pdeathsig,
    )
    return _ManagedChild(
        label="litellm-proxy",
        proc=proc,
        log_path=log_path,
        port=proxy_port,
    )


# --- readiness probing -----------------------------------------------------


def _wait_all_ready(children: Sequence[_ManagedChild], *, timeout: float) -> None:
    """Wait for each child's /v1/models to return 200, in parallel."""
    deadline = time.monotonic() + timeout
    with ThreadPoolExecutor(max_workers=len(children)) as ex:
        futures = {
            ex.submit(_wait_one_ready, child, timeout=timeout): child
            for child in children
        }
        for fut in futures:
            child = futures[fut]
            try:
                fut.result(timeout=max(1.0, deadline - time.monotonic()))
            except Exception as e:  # noqa: BLE001
                raise ManagedServerError(
                    f"{child.label}: failed to become ready: {e}\n"
                    f"--- last 50 lines of {child.log_path} ---\n"
                    + _tail_log(child.log_path, 50)
                ) from e


def _wait_one_ready(child: _ManagedChild, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{child.port}/v1/models"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        # Surface child-died-during-startup early instead of polling forever.
        rc = child.proc.poll()
        if rc is not None:
            raise ManagedServerError(
                f"{child.label} exited (code {rc}) before becoming ready"
            )
        try:
            with urlrequest.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urlerror.URLError, urlerror.HTTPError, TimeoutError, OSError) as e:
            last_error = e
        time.sleep(2.0)
    raise ManagedServerError(
        f"{child.label} did not respond at {url} within {timeout:.0f}s "
        f"(last error: {last_error})"
    )


# --- teardown --------------------------------------------------------------


def _teardown(children: Sequence[_ManagedChild]) -> None:
    """Terminate everything in reverse order (proxy first, then upstreams).

    SIGTERM, wait up to grace, then SIGKILL. Each child runs in its own
    process group (start_new_session=True), so we signal the group.
    """
    for child in reversed(list(children)):
        if child.proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(child.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                child.proc.terminate()
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + _TEARDOWN_GRACE
    for child in reversed(list(children)):
        remaining = max(0.1, deadline - time.monotonic())
        try:
            child.proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(child.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    child.proc.kill()
                except ProcessLookupError:
                    pass
            try:
                child.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                log.warning("%s did not die after SIGKILL", child.label)


# --- GPU budget ------------------------------------------------------------


_FREE_HEADROOM = 0.95  # never reserve more than this fraction of currently-free VRAM
_PER_MODEL_FLOOR = 0.1  # vLLM's practical minimum for --gpu-memory-utilization


def _compute_allocation(gpu_budget: float, n_models: int) -> tuple[float, str]:
    """Decide per-model ``--gpu-memory-utilization`` and return a one-line summary.

    Semantics: ``gpu_budget`` is the *ceiling* fraction of total VRAM the
    managed stack should claim collectively. We additionally clamp to
    ``_FREE_HEADROOM`` of currently-free VRAM, so a partly-busy GPU degrades
    gracefully instead of being rejected outright. The N models split the
    resulting effective budget equally.

    Raises ``ManagedServerError`` if no devices are visible AND ``gpu_budget``
    is in range, or if the resulting per-model fraction falls below vLLM's
    practical floor (~0.1).
    """
    if not (0.0 < gpu_budget <= 1.0):
        raise ManagedServerError(
            f"--gpu-budget must be in (0, 1]; got {gpu_budget!r}"
        )
    if n_models <= 0:
        raise ManagedServerError("at least one vllm profile is required")

    devices = _cuda_devices()
    if not devices:
        # No NVML / non-CUDA host: we can't enforce anything. Use the budget
        # as-is and let vLLM fail loudly if the math doesn't work out.
        per_model = gpu_budget / n_models
        return per_model, (
            f"No CUDA devices visible; per-model gpu_memory_utilization="
            f"{per_model:.3f} (untrusted, no NVML preflight)."
        )

    dev = devices[0]
    try:
        free_b = int(dev.memory_free())
        total_b = int(dev.memory_total())
    except Exception as e:  # noqa: BLE001
        log.warning(
            "nvitop memory query failed (%s); using --gpu-budget as-is", e
        )
        per_model = gpu_budget / n_models
        return per_model, (
            f"NVML query failed; per-model gpu_memory_utilization="
            f"{per_model:.3f} (no preflight)."
        )

    free_fraction = free_b / total_b if total_b > 0 else 0.0
    free_cap = free_fraction * _FREE_HEADROOM
    effective = min(gpu_budget, free_cap)
    per_model = effective / n_models

    if per_model < _PER_MODEL_FLOOR:
        raise ManagedServerError(
            f"GPU has too little free memory for {n_models} model(s) on "
            f"{dev.name()}: free {_human_bytes(free_b)} / "
            f"total {_human_bytes(total_b)} → effective budget "
            f"{effective:.3f}, per-model {per_model:.3f} which is below "
            f"vLLM's floor of {_PER_MODEL_FLOOR}. Free up GPU memory or "
            f"reduce model count."
        )

    summary = (
        f"GPU {dev.name()}: free {_human_bytes(free_b)} / "
        f"total {_human_bytes(total_b)} (free fraction {free_fraction:.2%}). "
        f"Budget {gpu_budget:.2f} → effective {effective:.3f} → "
        f"per-model gpu_memory_utilization={per_model:.3f} "
        f"({n_models} model{'s' if n_models != 1 else ''})."
    )
    if effective < gpu_budget:
        summary += f" [clamped from {gpu_budget:.2f} to fit free VRAM]"
    return per_model, summary


def _cuda_devices() -> list[Device]:
    """Return the list of visible CUDA devices, gracefully empty on CPU-only hosts."""
    try:
        return list(Device.cuda.all())
    except Exception as e:  # noqa: BLE001
        # nvitop raises if libnvml isn't loadable — treat as no GPU.
        log.warning("nvitop could not enumerate devices (%s); proceeding without preflight.", e)
        return []


def _human_bytes(n: int) -> str:
    """Compact human-readable byte count: 1.2 GiB, 512 MiB, etc."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} {u}"
        f /= 1024.0
    return f"{f:.1f} {units[-1]}"


def gpu_state_lines() -> list[str]:
    """Return a list of human-readable lines describing GPU memory + processes.

    Best-effort — returns ``[]`` if no devices, or if NVML queries fail.
    Intended for both logging (via ``log_gpu_state``) and CLI display
    (e.g. ``ocrscout serve`` showing it under "stack ready").
    """
    out: list[str] = []
    devices = _cuda_devices()
    if not devices:
        return out
    for dev in devices:
        try:
            free = int(dev.memory_free())
            used = int(dev.memory_used())
            total = int(dev.memory_total())
            out.append(
                f"GPU {dev.name()}: {_human_bytes(used)} used  /  "
                f"{_human_bytes(free)} free  /  {_human_bytes(total)} total"
            )
            try:
                procs = dev.processes()
            except Exception:  # noqa: BLE001
                procs = {}
            for pid, p in sorted(procs.items()):
                try:
                    pmem = int(p.gpu_memory())
                    cmd = p.name()
                    out.append(f"  pid={pid:<8} {_human_bytes(pmem):>10}  {cmd}")
                except Exception:  # noqa: BLE001
                    continue
        except Exception as e:  # noqa: BLE001
            log.warning("nvitop telemetry failed for %s: %s", dev.name(), e)
    return out


def log_gpu_state() -> None:
    """Log lines from :func:`gpu_state_lines` at INFO level."""
    for line in gpu_state_lines():
        log.info(line)


# --- helpers ---------------------------------------------------------------


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(s: str) -> str:
    return _FILENAME_SAFE.sub("_", s)


def _tail_log(path: Path, n: int) -> str:
    if not path.is_file():
        return "(no log file)"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"(error reading log: {e})"
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def cleanup_log_dir(log_dir: Path) -> None:
    """Optional helper: callers that want to remove the log dir on success."""
    shutil.rmtree(log_dir, ignore_errors=True)

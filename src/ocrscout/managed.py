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
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml
from nvitop import Device

from ocrscout.errors import ManagedServerError
from ocrscout.log import VERBOSE
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
    # Background reader that tees the child's combined stdout to its log file
    # and (when verbose) to the ocrscout logger. Daemon thread; exits when the
    # child's stream closes after teardown.
    tee_thread: threading.Thread | None = None


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
        # Each profile declares its own absolute KV cache budget via
        # vllm_engine_args.kv_cache_memory_bytes. We preflight the sum (plus
        # a per-model overhead estimate) against available VRAM, then spawn
        # all vllm-serve children in parallel — KV cache size is set
        # absolutely so the cudaMemGetInfo race that previously plagued
        # sibling spawns is gone. We *do* still pass --gpu-memory-utilization
        # as a per-engine cap (sized to that engine's actual footprint, not
        # split equally) so vLLM's startup free-memory check passes; without
        # it, vLLM defaults to 0.9+ and rejects the second engine to spawn.
        summary, engine_caps = _preflight_kv_budgets(vllm_profiles, gpu_budget)
        log.info(summary)

        for i, profile in enumerate(vllm_profiles):
            port = base_port + i
            children.append(
                _spawn_vllm_serve(
                    profile=profile,
                    port=port,
                    gpu_memory_utilization=engine_caps[profile.name],
                    log_dir=log_dir,
                )
            )

        log.info(
            "Spawned %d vllm-serve children; waiting up to %.0fs for ready...",
            len(children), ready_timeout,
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


def _spawn_with_tee(
    *, cmd: list[str], label: str, log_path: Path, port: int
) -> _ManagedChild:
    """Spawn a managed child and tee its combined stdout to (file, logger).

    The child's stdout+stderr is captured via a pipe and read by a daemon
    thread that writes every line to ``log_path`` (always, for post-mortem
    debugging) and emits it on the ocrscout logger at VERBOSE (visible at
    ``-v`` and ``-vv``, suppressed at default and ``-q``). Each line is
    prefixed with ``[label]`` so concurrent children stay distinguishable.

    Same lifecycle protections as the previous direct-to-file spawn:
    ``start_new_session`` for clean `killpg` teardown, ``PR_SET_PDEATHSIG``
    so the child dies with ocrscout on abnormal exit.
    """
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        preexec_fn=_set_pdeathsig,
    )
    tee_thread = threading.Thread(
        target=_tee_loop,
        args=(proc.stdout, log_fh, label),
        name=f"tee[{label}]",
        daemon=True,
    )
    tee_thread.start()
    return _ManagedChild(
        label=label, proc=proc, log_path=log_path, port=port, tee_thread=tee_thread
    )


def _tee_loop(src: IO[str], log_fh: IO[str], label: str) -> None:
    """Read lines from ``src``; write each to ``log_fh`` and emit at VERBOSE.

    Runs in a daemon thread per managed child. Exits when ``src`` closes
    (i.e. when the child terminates). The logger emit is gated by level so
    at default verbosity we pay only the file write — no terminal flooding.
    """
    try:
        for line in src:
            try:
                log_fh.write(line)
                log_fh.flush()
            except Exception:  # noqa: BLE001
                pass  # don't crash the tee on file-side issues
            if log.isEnabledFor(VERBOSE):
                log.log(VERBOSE, "[%s] %s", label, line.rstrip("\n"))
    finally:
        try:
            log_fh.close()
        except Exception:  # noqa: BLE001
            pass


_ENGINE_ARG_OWNED_BY_MANAGED = frozenset({"gpu_memory_utilization", "port"})


def _engine_args_to_cli(engine_args: dict) -> list[str]:
    """Translate a profile's ``vllm_engine_args`` dict into ``vllm serve`` flags.

    Each key becomes ``--<kebab-key>``. Booleans are bare flags (omitted when
    false). Scalars (int/float/str) become ``--flag value``. Dicts are
    JSON-encoded (vLLM accepts a single JSON string for args like
    ``limit_mm_per_prompt``). Lists are splatted as separate positional values
    (``--flag v1 v2 v3``), matching argparse ``nargs='+'`` flags like
    ``--cudagraph-capture-sizes``. ``gpu_memory_utilization`` and ``port``
    are skipped because managed mode owns both (cap is computed per-engine
    from the profile's KV bytes; port is assigned by base_port + i).
    """
    out: list[str] = []
    for key, value in engine_args.items():
        if key in _ENGINE_ARG_OWNED_BY_MANAGED or value is None:
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
    cmd += _engine_args_to_cli(profile.vllm_engine_args or {})

    log.debug("Spawning %s on port %d -> %s", label, port, log_path)
    return _spawn_with_tee(cmd=cmd, label=label, log_path=log_path, port=port)


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

    log.debug("Spawning litellm-proxy on port %d -> %s", proxy_port, log_path)
    return _spawn_with_tee(
        cmd=cmd, label="litellm-proxy", log_path=log_path, port=proxy_port
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
                raise ManagedServerError(f"{child.label}: {e}") from e


def _wait_one_ready(child: _ManagedChild, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{child.port}/v1/models"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        # Surface child-died-during-startup early instead of polling forever.
        rc = child.proc.poll()
        if rc is not None:
            raise ManagedServerError(
                f"{child.label} exited (code {rc}) before becoming ready\n"
                f"--- last 50 lines of {child.log_path} ---\n"
                + _tail_log(child.log_path, 50)
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
        f"(last error: {last_error})\n"
        f"--- last 50 lines of {child.log_path} ---\n"
        + _tail_log(child.log_path, 50)
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
_OVERHEAD_PER_MODEL_BYTES = 8 * 1024 ** 3  # fallback when model_size is missing/unparseable

_BYTE_SUFFIX_MULTIPLIERS = {
    "": 1,
    "k": 1_000, "K": 1024,
    "m": 1_000_000, "M": 1024 ** 2,
    "g": 1_000_000_000, "G": 1024 ** 3,
    "t": 1_000_000_000_000, "T": 1024 ** 4,
}

# Decimal suffixes for parameter counts (these are not byte sizes — model
# parameter counts use SI conventions: "3B" means 3 × 10⁹ params).
_PARAM_COUNT_SUFFIXES = {
    "K": 1_000, "M": 1_000_000,
    "B": 1_000_000_000, "T": 1_000_000_000_000,
}

# Bytes per parameter for common vLLM `dtype` strings. Default to 2 (BF16)
# when dtype is unset or unrecognized — matches vLLM's "auto" behavior for
# most modern models. FP8/INT8 = 1 byte; INT4 = 0.5 byte (rounded up by int()).
_DTYPE_BYTES_PER_PARAM: dict[str, float] = {
    "auto": 2,
    "bfloat16": 2, "bf16": 2,
    "float16": 2, "fp16": 2, "half": 2,
    "float32": 4, "fp32": 4, "float": 4,
    "fp8": 1, "float8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1,
    "int4": 0.5, "uint4": 0.5,
}


def _parse_bytes(value: int | float | str) -> int:
    """Parse a byte count from int or string with vLLM-style suffixes.

    Lowercase suffixes are decimal SI (``k`` = 1,000); uppercase are binary
    (``K`` = 1024). Optional trailing ``b``/``B`` is ignored. Decimal
    multipliers supported (``"1.5G"`` → 1,610,612,736). Plain ints/floats are
    returned as ``int(value)``.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(
            f"cannot parse bytes from {type(value).__name__}: {value!r}"
        )
    s = value.strip()
    if s.endswith(("b", "B")):
        s = s[:-1]
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)$", s)
    if not m:
        raise ValueError(f"cannot parse byte value: {value!r}")
    return int(float(m.group(1)) * _BYTE_SUFFIX_MULTIPLIERS[m.group(2)])


def _parse_model_size(s: str) -> int | None:
    """Parse ``"3B"`` / ``"1.7B"`` / ``"750M"`` to a parameter count.

    Suffixes are decimal SI: ``B`` = 10⁹, ``M`` = 10⁶, ``K`` = 10³, ``T`` =
    10¹². Suffix is case-insensitive. Bare numbers (no suffix) are treated
    as ``B`` (the most common case in HF model cards). Returns ``None`` for
    unrecognized formats.
    """
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KkMmBbTt])?$", s.strip())
    if not m:
        return None
    num = float(m.group(1))
    suf = (m.group(2) or "B").upper()
    return int(num * _PARAM_COUNT_SUFFIXES[suf])


def _estimate_model_overhead(profile: ModelProfile) -> int:
    """Estimate per-engine non-KV memory: weights + working + cudagraph slack.

    Weights: ``params × bytes_per_param``, where ``params`` is parsed from
    ``profile.model_size`` and ``bytes_per_param`` is inferred from
    ``profile.vllm_engine_args.dtype`` (default 2 bytes for BF16/FP16/auto).
    Adds a working-memory slack of ``max(1 GiB, weights // 4)`` to cover
    activations, CUDA graph workspace, and allocator overhead.

    Falls back to ``_OVERHEAD_PER_MODEL_BYTES`` when ``model_size`` is missing
    or unparseable, so preflight stays conservative for under-specified
    profiles.
    """
    if not profile.model_size:
        return _OVERHEAD_PER_MODEL_BYTES
    params = _parse_model_size(profile.model_size)
    if params is None:
        log.warning(
            "profile %r: unparseable model_size %r; falling back to %s overhead",
            profile.name, profile.model_size,
            _human_bytes(_OVERHEAD_PER_MODEL_BYTES),
        )
        return _OVERHEAD_PER_MODEL_BYTES

    dtype = str((profile.vllm_engine_args or {}).get("dtype", "auto")).lower()
    bytes_per_param = _DTYPE_BYTES_PER_PARAM.get(dtype, 2)
    weights = int(params * bytes_per_param)
    # Working memory + cudagraph workspace + allocator slack. ``model_size``
    # typically counts only the LLM parameters, but vLLM also loads vision
    # towers / other modules and reserves transient working memory during
    # the profile step — so we floor at 2 GiB and scale to 50% of weights
    # for larger models. Combined with the 1.25× multiplier in the cap
    # calculation, this keeps us well above observed peak footprints
    # without massively over-budgeting.
    working = max(2 * 1024 ** 3, weights // 2)
    return weights + working


# Safety multiplier on each profile's overhead estimate when computing the
# per-engine --gpu-memory-utilization cap. The cap is what vLLM checks
# against free VRAM at startup; 1.25× leaves 25% headroom for transient
# working-memory spikes during vLLM's profile step.
_ENGINE_CAP_OVERHEAD_MULTIPLIER = 1.25


def _preflight_kv_budgets(
    profiles: Sequence[ModelProfile], gpu_budget: float
) -> tuple[str, dict[str, float]]:
    """Validate per-profile ``kv_cache_memory_bytes`` and compute cap fractions.

    Each managed profile must declare ``vllm_engine_args.kv_cache_memory_bytes``
    (vLLM's absolute KV cache size; suffixes accepted). Per-profile overhead
    is estimated from ``model_size`` + ``dtype`` via
    :func:`_estimate_model_overhead` (falls back to a flat 8 GiB when
    ``model_size`` is missing). Total footprint = sum of (KV + overhead) per
    profile, checked against ``min(total_VRAM × gpu_budget, free_VRAM ×
    _FREE_HEADROOM)``.

    Also computes per-profile ``--gpu-memory-utilization`` caps. Even when
    ``kv_cache_memory_bytes`` is set, vLLM still does a startup check that
    ``gpu_memory_utilization × total_VRAM <= free_VRAM``; defaulting (vLLM's
    0.9+) fails when sibling engines are loading. Each cap is sized to the
    profile's own ``(KV + overhead × 1.25) / total_VRAM``, which is large
    enough for the engine's actual footprint and small enough to fit free
    VRAM during parallel spawn.

    Returns ``(summary, caps)`` where ``caps`` maps ``profile.name`` to the
    cap fraction. Falls back to ``gpu_budget / N`` per engine when NVML is
    unavailable.
    """
    if not (0.0 < gpu_budget <= 1.0):
        raise ManagedServerError(
            f"--gpu-budget must be in (0, 1]; got {gpu_budget!r}"
        )
    if not profiles:
        raise ManagedServerError("at least one vllm profile is required")

    declared: list[tuple[str, int, int]] = []  # (name, kv_bytes, overhead_bytes)
    missing: list[str] = []
    for p in profiles:
        raw = (p.vllm_engine_args or {}).get("kv_cache_memory_bytes")
        if raw is None:
            missing.append(p.name)
            continue
        try:
            kv = _parse_bytes(raw)
        except ValueError as e:
            raise ManagedServerError(
                f"profile {p.name!r}: invalid kv_cache_memory_bytes: {e}"
            ) from e
        declared.append((p.name, kv, _estimate_model_overhead(p)))

    if missing:
        raise ManagedServerError(
            f"managed mode requires per-profile "
            f"vllm_engine_args.kv_cache_memory_bytes; missing on profile(s): "
            f"{missing}. Add a value like `kv_cache_memory_bytes: 16G` to the "
            f"profile YAML."
        )

    total_kv = sum(kv for _, kv, _ in declared)
    total_overhead = sum(o for _, _, o in declared)
    total_required = total_kv + total_overhead

    devices = _cuda_devices()
    fallback_cap = gpu_budget / len(declared)
    if not devices:
        summary = (
            f"No CUDA devices visible; trusting per-profile KV budgets "
            f"({_human_bytes(total_kv)} total, +{_human_bytes(total_overhead)} "
            f"estimated overhead) without preflight."
        )
        return summary, {name: fallback_cap for name, _, _ in declared}

    dev = devices[0]
    try:
        free_b = int(dev.memory_free())
        total_b = int(dev.memory_total())
    except Exception as e:  # noqa: BLE001
        log.warning("nvitop memory query failed (%s); skipping preflight", e)
        summary = (
            f"NVML query failed; trusting per-profile KV budgets "
            f"({_human_bytes(total_kv)} total) without preflight."
        )
        return summary, {name: fallback_cap for name, _, _ in declared}

    free_cap = int(free_b * _FREE_HEADROOM)
    budget_cap = int(total_b * gpu_budget)
    cap = min(free_cap, budget_cap)

    if total_required > cap:
        per_profile = "\n  ".join(
            f"{name}: KV {_human_bytes(kv)} + overhead {_human_bytes(o)}"
            for name, kv, o in declared
        )
        raise ManagedServerError(
            f"per-profile KV budgets exceed available GPU memory on "
            f"{dev.name()}:\n  {per_profile}\n  total = "
            f"{_human_bytes(total_required)} > cap {_human_bytes(cap)} "
            f"(free {_human_bytes(free_b)} × {_FREE_HEADROOM:.0%}, "
            f"budget {gpu_budget:.2f} × total {_human_bytes(total_b)}).\n"
            f"Reduce per-profile kv_cache_memory_bytes, free GPU memory, "
            f"or raise --gpu-budget."
        )

    caps = {
        name: (kv + int(o * _ENGINE_CAP_OVERHEAD_MULTIPLIER)) / total_b
        for name, kv, o in declared
    }

    summary = (
        f"GPU {dev.name()}: per-profile KV totals {_human_bytes(total_kv)} + "
        f"overhead {_human_bytes(total_overhead)} = "
        f"{_human_bytes(total_required)} (cap {_human_bytes(cap)}; "
        f"free {_human_bytes(free_b)} / total {_human_bytes(total_b)})"
    )
    return summary, caps


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
    """Log GPU state at VERBOSE level (-v).

    Per-process listings are noisy enough that they shouldn't appear at the
    default verbosity — the allocation summary is the load-bearing line; this
    is "what's actually allocated right now" telemetry for when you want it.
    """
    for line in gpu_state_lines():
        log.log(VERBOSE, line)


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

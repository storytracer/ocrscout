"""KV-budget preflight and autoscaling for ``runtime: vllm`` profiles.

Two complementary functions, sharing a probe of the active GPU's
``min(total × gpu_budget, free × 0.95)`` headroom:

* :func:`preflight_kv_budgets` — validates that an explicit-KV launch fits.
  Sums each profile's declared ``vllm_engine_args.kv_cache_memory_bytes``
  plus a per-profile overhead estimate (weights from ``model_size`` ×
  bytes-per-param from ``dtype``, plus working slack) and rejects when
  the total exceeds the cap.

* :func:`autoscale_kv_budgets` — computes KV and per-profile concurrency
  from the cap when the profile YAML doesn't declare them. Caller mutates
  the profile objects with the returned decisions before any spawn.

Both are reused by every Runner that has to size the GPU footprint of
a planned launch.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ocrscout.errors import RunnerError
from ocrscout.profile import ModelProfile

log = logging.getLogger(__name__)

_FREE_HEADROOM = 0.95
_OVERHEAD_PER_MODEL_BYTES = 8 * 1024**3
_ENGINE_CAP_OVERHEAD_MULTIPLIER = 1.25

# Autoscaler tuning. Both are deliberately module constants — single edits
# retune the whole fleet. The per-token coefficient is a rule of thumb for
# 3B–7B-class VLMs (derived from
# ``2 × num_layers × num_kv_heads × head_dim × dtype_bytes``); the ceiling
# reflects empirical diminishing returns past 64 concurrent requests from
# vision-encoder cache thrashing and prefill contention.
AUTOSCALE_PER_TOKEN_BYTES = 30_000
AUTOSCALE_MAX_CONCURRENCY = 64

_BYTE_SUFFIX_MULTIPLIERS = {
    "": 1,
    "k": 1_000, "K": 1024,
    "m": 1_000_000, "M": 1024**2,
    "g": 1_000_000_000, "G": 1024**3,
    "t": 1_000_000_000_000, "T": 1024**4,
}

_PARAM_COUNT_SUFFIXES = {
    "K": 1_000, "M": 1_000_000,
    "B": 1_000_000_000, "T": 1_000_000_000_000,
}

_DTYPE_BYTES_PER_PARAM: dict[str, float] = {
    "auto": 2,
    "bfloat16": 2, "bf16": 2,
    "float16": 2, "fp16": 2, "half": 2,
    "float32": 4, "fp32": 4, "float": 4,
    "fp8": 1, "float8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1,
    "int4": 0.5, "uint4": 0.5,
}


def parse_bytes(value: int | float | str) -> int:
    """Parse a byte count from int or string with vLLM-style suffixes."""
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


def parse_model_size(s: str) -> int | None:
    """Parse ``"3B"`` / ``"1.7B"`` / ``"750M"`` to a parameter count."""
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([KkMmBbTt])?$", s.strip())
    if not m:
        return None
    num = float(m.group(1))
    suf = (m.group(2) or "B").upper()
    return int(num * _PARAM_COUNT_SUFFIXES[suf])


def estimate_model_overhead(profile: ModelProfile) -> int:
    """Estimate per-engine non-KV memory: weights + working + cudagraph slack."""
    if not profile.model_size:
        return _OVERHEAD_PER_MODEL_BYTES
    params = parse_model_size(profile.model_size)
    if params is None:
        log.warning(
            "profile %r: unparseable model_size %r; falling back to %s overhead",
            profile.name, profile.model_size,
            human_bytes(_OVERHEAD_PER_MODEL_BYTES),
        )
        return _OVERHEAD_PER_MODEL_BYTES

    dtype = str((profile.vllm_engine_args or {}).get("dtype", "auto")).lower()
    bytes_per_param = _DTYPE_BYTES_PER_PARAM.get(dtype, 2)
    weights = int(params * bytes_per_param)
    working = max(2 * 1024**3, weights // 2)
    return weights + working


def preflight_kv_budgets(
    profiles: Sequence[ModelProfile], gpu_budget: float
) -> tuple[str, dict[str, float]]:
    """Validate per-profile KV budgets and compute per-engine cap fractions.

    Returns ``(summary, caps)``. ``summary`` is a human-readable line for
    the runner log; ``caps`` maps ``profile.name`` to the
    ``--gpu-memory-utilization`` value to pass to ``vllm serve``.

    Raises :class:`ocrscout.errors.RunnerError` when total declared KV +
    overhead would exceed the available GPU memory under ``gpu_budget``.
    """
    if not (0.0 < gpu_budget <= 1.0):
        raise RunnerError(f"--gpu-budget must be in (0, 1]; got {gpu_budget!r}")
    if not profiles:
        raise RunnerError("at least one vllm-runtime profile is required")

    declared: list[tuple[str, int, int]] = []
    missing: list[str] = []
    for p in profiles:
        raw = (p.vllm_engine_args or {}).get("kv_cache_memory_bytes")
        if raw is None:
            missing.append(p.name)
            continue
        try:
            kv = parse_bytes(raw)
        except ValueError as e:
            raise RunnerError(
                f"profile {p.name!r}: invalid kv_cache_memory_bytes: {e}"
            ) from e
        declared.append((p.name, kv, estimate_model_overhead(p)))

    if missing:
        raise RunnerError(
            f"runtime='vllm' profiles must declare "
            f"vllm_engine_args.kv_cache_memory_bytes; missing on: "
            f"{missing}. Add e.g. `kv_cache_memory_bytes: 16G` to the "
            f"profile YAML."
        )

    total_kv = sum(kv for _, kv, _ in declared)
    total_overhead = sum(o for _, _, o in declared)
    total_required = total_kv + total_overhead

    devices = cuda_devices()
    fallback_cap = gpu_budget / len(declared)
    if not devices:
        summary = (
            f"No CUDA devices visible; trusting per-profile KV budgets "
            f"({human_bytes(total_kv)} total, "
            f"+{human_bytes(total_overhead)} estimated overhead) without "
            f"preflight."
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
            f"({human_bytes(total_kv)} total) without preflight."
        )
        return summary, {name: fallback_cap for name, _, _ in declared}

    free_cap = int(free_b * _FREE_HEADROOM)
    budget_cap = int(total_b * gpu_budget)
    cap = min(free_cap, budget_cap)

    if total_required > cap:
        per_profile = "\n  ".join(
            f"{name}: KV {human_bytes(kv)} + overhead {human_bytes(o)}"
            for name, kv, o in declared
        )
        raise RunnerError(
            f"per-profile KV budgets exceed available GPU memory on "
            f"{dev.name()}:\n  {per_profile}\n  total = "
            f"{human_bytes(total_required)} > cap {human_bytes(cap)} "
            f"(free {human_bytes(free_b)} × {_FREE_HEADROOM:.0%}, "
            f"budget {gpu_budget:.2f} × total {human_bytes(total_b)}).\n"
            f"Reduce per-profile kv_cache_memory_bytes, free GPU memory, "
            f"or raise --gpu-budget."
        )

    caps = {
        name: (kv + int(o * _ENGINE_CAP_OVERHEAD_MULTIPLIER)) / total_b
        for name, kv, o in declared
    }

    summary = (
        f"GPU {dev.name()}: per-profile KV totals {human_bytes(total_kv)} + "
        f"overhead {human_bytes(total_overhead)} = "
        f"{human_bytes(total_required)} (cap {human_bytes(cap)}; "
        f"free {human_bytes(free_b)} / total {human_bytes(total_b)})"
    )
    return summary, caps


@dataclass(frozen=True)
class AutoscaleDecision:
    """What the autoscaler decided for one profile."""

    profile_name: str
    kv_cache_memory_bytes: int
    concurrent_requests: int
    overhead_bytes: int
    max_model_len: int


@dataclass(frozen=True)
class GpuProbe:
    """Snapshot of the active GPU at autoscale time."""

    name: str
    total_bytes: int
    free_bytes: int
    cap_bytes: int


def probe_gpu(gpu_budget: float) -> GpuProbe:
    """Read free/total VRAM from the first visible CUDA device and compute
    the autoscale cap. Raises ``RunnerError`` if no GPU is visible —
    autoscale cannot proceed without a real device to size for.
    """
    if not (0.0 < gpu_budget <= 1.0):
        raise RunnerError(f"--gpu-budget must be in (0, 1]; got {gpu_budget!r}")
    devices = cuda_devices()
    if not devices:
        raise RunnerError(
            "autoscale requires a visible CUDA device. None detected — check "
            "CUDA_VISIBLE_DEVICES, the NVIDIA driver, and nvitop installation. "
            "Set kv_cache_memory_bytes / backend_args.concurrent_requests "
            "explicitly on the profile to bypass autoscale."
        )
    dev = devices[0]
    try:
        free_b = int(dev.memory_free())
        total_b = int(dev.memory_total())
    except Exception as e:  # noqa: BLE001
        raise RunnerError(f"nvitop memory query failed: {e}") from e
    cap = min(int(free_b * _FREE_HEADROOM), int(total_b * gpu_budget))
    return GpuProbe(
        name=str(dev.name()), total_bytes=total_b, free_bytes=free_b, cap_bytes=cap
    )


def autoscale_kv_budgets(
    profiles: Sequence[ModelProfile],
    gpu_budget: float,
    *,
    batch_concurrency: int | None = None,
    probe: GpuProbe | None = None,
) -> tuple[str, dict[str, AutoscaleDecision]]:
    """Decide KV cache size + per-profile concurrency from the active GPU.

    Returns ``(summary, decisions)`` where ``decisions[profile.name]`` is
    an :class:`AutoscaleDecision`. The caller mutates each profile's
    ``vllm_engine_args["kv_cache_memory_bytes"]`` and ``backend_args``
    from these values before any spawn.

    KV cap = ``min(free × 0.95, total × gpu_budget)`` minus the sum of
    per-profile weight + working overhead. KV is split proportionally
    across the chunk by ``max_model_len × per_token_bytes`` so chunks
    that mix profiles with very different context lengths each get a
    fair slice. When ``batch_concurrency`` is set, that value is honored
    per profile and KV is derived from it; raises if a profile's slice
    cannot hold the requested concurrency.
    """
    if not profiles:
        raise RunnerError("autoscale_kv_budgets: profiles list is empty")
    if batch_concurrency is not None and batch_concurrency < 1:
        raise RunnerError(
            f"--batch-concurrency must be a positive integer; got {batch_concurrency!r}"
        )

    gpu = probe if probe is not None else probe_gpu(gpu_budget)

    overheads: dict[str, int] = {
        p.name: estimate_model_overhead(p) for p in profiles
    }
    total_overhead = sum(overheads.values())
    available_kv = gpu.cap_bytes - total_overhead
    if available_kv <= 0:
        per_profile = ", ".join(
            f"{p.name}: overhead {human_bytes(overheads[p.name])}"
            for p in profiles
        )
        raise RunnerError(
            f"autoscale: model weights + working overhead "
            f"({human_bytes(total_overhead)}) already exceed the GPU cap "
            f"{human_bytes(gpu.cap_bytes)} on {gpu.name}; nothing left for "
            f"KV cache. Per-profile breakdown: {per_profile}. Reduce "
            f"--parallel-models, raise --gpu-budget, or free GPU memory."
        )

    # Proportional weights for the kv split.
    def _max_len(p: ModelProfile) -> int:
        raw = (p.vllm_engine_args or {}).get("max_model_len")
        if raw is None:
            raise RunnerError(
                f"autoscale: profile {p.name!r} is missing "
                f"vllm_engine_args.max_model_len; autoscale needs it to "
                f"size KV cache. Add it to the profile YAML."
            )
        return int(raw)

    max_lens = {p.name: _max_len(p) for p in profiles}
    weights = {
        name: max_lens[name] * AUTOSCALE_PER_TOKEN_BYTES for name in max_lens
    }
    weight_total = sum(weights.values()) or 1

    decisions: dict[str, AutoscaleDecision] = {}
    for p in profiles:
        kv_slice = int(available_kv * weights[p.name] / weight_total)
        per_request_kv = max_lens[p.name] * AUTOSCALE_PER_TOKEN_BYTES

        if batch_concurrency is not None:
            concurrency = batch_concurrency
            kv = concurrency * per_request_kv
            if kv > kv_slice:
                raise RunnerError(
                    f"autoscale: --batch-concurrency {batch_concurrency} on "
                    f"profile {p.name!r} needs {human_bytes(kv)} of KV cache "
                    f"but only {human_bytes(kv_slice)} is available after "
                    f"weights+overhead on {gpu.name} (cap "
                    f"{human_bytes(gpu.cap_bytes)}). Lower "
                    f"--batch-concurrency, raise --gpu-budget, lower "
                    f"--parallel-models, or pick profiles with smaller "
                    f"max_model_len."
                )
        else:
            max_from_kv = kv_slice // per_request_kv
            concurrency = max(1, min(int(max_from_kv), AUTOSCALE_MAX_CONCURRENCY))
            kv = concurrency * per_request_kv

        decisions[p.name] = AutoscaleDecision(
            profile_name=p.name,
            kv_cache_memory_bytes=kv,
            concurrent_requests=concurrency,
            overhead_bytes=overheads[p.name],
            max_model_len=max_lens[p.name],
        )

    lines = [
        f"GPU {gpu.name}: free {human_bytes(gpu.free_bytes)} / total "
        f"{human_bytes(gpu.total_bytes)}; cap {human_bytes(gpu.cap_bytes)}. "
        f"Auto-scaled {len(decisions)} profile(s):"
    ]
    for name, d in decisions.items():
        lines.append(
            f"  {name} (overhead {human_bytes(d.overhead_bytes)}, "
            f"KV {human_bytes(d.kv_cache_memory_bytes)}, "
            f"concurrency {d.concurrent_requests})"
        )
    return "\n".join(lines), decisions


def cuda_devices() -> list:
    """Return visible CUDA devices, empty on CPU-only hosts."""
    try:
        from nvitop import Device

        return list(Device.cuda.all())
    except Exception as e:  # noqa: BLE001
        log.warning("nvitop could not enumerate devices (%s)", e)
        return []


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} {u}"
        f /= 1024.0
    return f"{f:.1f} {units[-1]}"

"""KV-budget preflight for ``runtime: vllm`` profiles.

Sums each profile's declared ``vllm_engine_args.kv_cache_memory_bytes``
plus a per-profile overhead estimate (weights from ``model_size`` ×
bytes-per-param from ``dtype``, plus a working slack) and rejects the
launch when the total would exceed
``min(total_VRAM × gpu_budget, free_VRAM × 0.95)``. Reused by every
``Runner`` that has to size the GPU footprint of a planned launch
(``LocalRunner`` today; future GPU-aware remote runners can call the
same helper).

Extracted from the previous monolithic ``managed.py`` so step 6's hard
cut can delete that module without leaving Runners unwired.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ocrscout.errors import RunnerError
from ocrscout.profile import ModelProfile

log = logging.getLogger(__name__)

_FREE_HEADROOM = 0.95
_OVERHEAD_PER_MODEL_BYTES = 8 * 1024**3
_ENGINE_CAP_OVERHEAD_MULTIPLIER = 1.25

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

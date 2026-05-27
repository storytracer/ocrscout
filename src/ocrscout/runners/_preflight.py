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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ocrscout.errors import RunnerError
from ocrscout.profile import ModelProfile

if TYPE_CHECKING:
    from dbgpu import GPUSpecification

log = logging.getLogger(__name__)

_FREE_HEADROOM = 0.95
_OVERHEAD_PER_MODEL_BYTES = 8 * 1024**3
_ENGINE_CAP_OVERHEAD_MULTIPLIER = 1.25

# Autoscaler tuning. Deliberately module constants — single edits retune the
# whole fleet. The per-token coefficient is a rule of thumb for 3B–7B-class
# VLMs (derived from ``2 × num_layers × num_kv_heads × head_dim × dtype_bytes``).
AUTOSCALE_PER_TOKEN_BYTES = 30_000

# Bandwidth-derived concurrency cap. When the active GPU can be identified
# via dbgpu, ``concurrency_cap = max(1, floor(bandwidth_gb_s / BW_GB_S_PER_CONCURRENT_REQUEST))``
# is the autoscaler's per-backend ceiling — replacing the empirical
# per-backend constants below. The constant is a first-order proxy for
# "decode-step bandwidth headroom one concurrent in-flight request needs";
# the relationship isn't perfectly linear (cache effects, attention pattern,
# unified vs discrete memory) but bandwidth/N is the right shape. Calibrated
# so the four benchmark GPUs (H100/A10/L4/GB10) at concurrency=16 do not
# regress: H100→136, A10→40, L4→20, GB10→18. Tunable here; recalibrate when
# sweep data is in hand.
BW_GB_S_PER_CONCURRENT_REQUEST = 15

# Safe per-backend fallback ceilings used only when the GPU is unknown to
# dbgpu. The historical empirical values: 64 for ``backend: litellm``
# (whole-page batches stagger naturally, vLLM continuous batching scales),
# 16 for ``backend: layout_chat`` (the original per-page burst pattern
# choked at higher concurrency; cross-page batching has since removed the
# burst, but we keep 16 as the conservative unknown-GPU default).
AUTOSCALE_MAX_CONCURRENCY = 64
AUTOSCALE_MAX_REGION_CONCURRENCY = 16

# Performance target for per-request avg latency, in seconds. Used to derive
# a third concurrency cap (beside KV-fits and the bandwidth/15 ceiling) when
# the GPU is identified via dbgpu. Lower → smaller concurrency, faster
# per-request response; higher → larger concurrency, slower per-request
# response but identical aggregate throughput (decode is bandwidth-bound, so
# total tok/s is fixed by ``bandwidth / model_bytes`` regardless of C — only
# the per-request slice changes). This is *not* a runtime cutoff; it only
# shapes the autoscaler's concurrency choice. The real timeout that fails
# requests is ``backends/litellm.py:_DEFAULT_REQUEST_TIMEOUT`` (600s),
# overridable per profile via ``backend_args.request_timeout``.
AUTOSCALE_TARGET_REQUEST_LATENCY_S = 240.0

# Default planning value for the average output-token count per request,
# expressed as a fraction of the profile's ``sampling_args.max_tokens``.
# Calibrated from the dots-mocr BHL run on H100 (~754 tokens observed avg
# vs. max_tokens=20000 → ratio ~0.04); rounded up to 0.05 for safety.
# Override per profile with ``metadata.expected_output_tokens`` when the
# workload is far from this OCR-page baseline (e.g. SVG mode, long-form
# generation).
AUTOSCALE_DEFAULT_T_PLAN_FRACTION = 0.05

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


@dataclass(frozen=True)
class DeviceProbe:
    """Nvitop-derived properties used to disambiguate dbgpu candidates.

    All fields are optional because nvitop's underlying NVML coverage
    varies by platform/driver — ``pci_bus_id`` / ``multi_processor_count``
    error on ARM CUDA (GB10), ``power_max_limit`` errors on some MIG
    instances, etc. The scorer treats a missing field as a non-vote
    (neither penalty nor reward) so partial coverage still helps.

    ``pcie_link_*`` are used as veto-only signals — a dbgpu candidate
    whose ``bus_interface`` reports a strictly-lower PCIe generation or
    width than the live card is *impossible* and gets eliminated before
    scoring. Equal or higher values don't veto (slot-downlinks happen;
    NVML can mis-report on some integrated SoCs).
    """

    memory_size_gb: float | None = None
    cuda_compute_capability: tuple[int, int] | None = None
    boost_clock_mhz: float | None = None
    pcie_link_generation: int | None = None
    pcie_link_width: int | None = None


def _probe_device_properties(dev: object) -> DeviceProbe:
    """Collect nvitop properties relevant for dbgpu disambiguation.

    Each individual probe is wrapped in try/except because NVML coverage
    isn't uniform; one missing field shouldn't disable the others.
    """
    def _safe(get: Callable[[], object]) -> object | None:
        try:
            v = get()
        except Exception:  # noqa: BLE001
            return None
        if v in (None, "N/A", "Unknown"):
            return None
        return v

    mem_total = _safe(lambda: dev.memory_total())  # type: ignore[attr-defined]
    cc = _safe(lambda: dev.cuda_compute_capability)  # type: ignore[attr-defined]
    max_sm_clock = _safe(lambda: dev.max_sm_clock)  # type: ignore[attr-defined]

    # pynvml exposes PCIe link gen/width directly; nvitop doesn't wrap them.
    # NVML is already initialized by nvitop's prior device enumeration.
    pcie_gen: int | None = None
    pcie_width: int | None = None
    try:
        import pynvml

        physical_index = int(dev.physical_index)  # type: ignore[attr-defined]
        h = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        pcie_gen = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h))
        pcie_width = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(h))
    except Exception:  # noqa: BLE001
        pass

    return DeviceProbe(
        memory_size_gb=(
            float(mem_total) / (1024**3)
            if isinstance(mem_total, (int, float)) else None
        ),
        cuda_compute_capability=(
            (int(cc[0]), int(cc[1]))
            if isinstance(cc, tuple) and len(cc) == 2 else None
        ),
        boost_clock_mhz=(
            float(max_sm_clock)
            if isinstance(max_sm_clock, (int, float)) else None
        ),
        pcie_link_generation=pcie_gen,
        pcie_link_width=pcie_width,
    )


_BUS_INTERFACE_RE = re.compile(r"PCIe\s+(\d+)\.\d+\s+x(\d+)", re.IGNORECASE)


def _parse_bus_interface(s: str | None) -> tuple[int, int] | None:
    """Extract ``(generation, width)`` from a dbgpu ``bus_interface`` string
    like ``"PCIe 4.0 x16"``. Returns ``None`` for unparseable / missing
    strings."""
    if not s:
        return None
    m = _BUS_INTERFACE_RE.search(s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _is_vetoed_by_bus(spec: GPUSpecification, probe: DeviceProbe) -> bool:
    """A candidate is vetoed when its dbgpu-recorded bus capability is
    *strictly less* than what the live card reports — a card can't run
    at a generation/width it doesn't support, so any such candidate is
    the wrong SKU. Equal-or-higher dbgpu values pass (slot-downlinks
    are real; NVML can also under-report on integrated SoCs)."""
    if probe.pcie_link_generation is None and probe.pcie_link_width is None:
        return False
    bus = _parse_bus_interface(spec.bus_interface)
    if bus is None:
        return False
    spec_gen, spec_width = bus
    if (
        probe.pcie_link_generation is not None
        and spec_gen < probe.pcie_link_generation
    ):
        return True
    return (
        probe.pcie_link_width is not None
        and spec_width < probe.pcie_link_width
    )


def _score_candidate(spec: GPUSpecification, probe: DeviceProbe) -> int:
    """Count probe fields that agree with the candidate spec within tolerance.

    Score = 0 → no agreement (probably wrong SKU). Higher score → more
    fields agree (better confidence). Used to break ambiguity among
    word-boundary matches when multiple SKUs share a name token (H100
    PCIe 80 vs 96 GB, A100 PCIe vs SXM4, H200 NVL vs SXM, etc.).
    """
    score = 0

    # 10% memory tolerance — accounts for ECC reservation cutting usable
    # VRAM vs the manufacturer's nominal value. 3% clock tolerance —
    # driver-reported max clocks drift slightly across firmware revisions.
    if (
        probe.memory_size_gb is not None
        and spec.memory_size_gb is not None
        and abs(spec.memory_size_gb - probe.memory_size_gb)
        / probe.memory_size_gb <= 0.10
    ):
        score += 1

    if (
        probe.cuda_compute_capability is not None
        and spec.cuda_major_version is not None
        and spec.cuda_minor_version is not None
        and spec.cuda_major_version == probe.cuda_compute_capability[0]
        and spec.cuda_minor_version == probe.cuda_compute_capability[1]
    ):
        score += 1

    if (
        probe.boost_clock_mhz is not None
        and spec.boost_clock_mhz is not None
        and abs(spec.boost_clock_mhz - probe.boost_clock_mhz)
        / probe.boost_clock_mhz <= 0.03
    ):
        score += 1

    return score


def _lookup_gpu_specs(
    gpu_name: str, device_probe: DeviceProbe | None = None
) -> GPUSpecification | None:
    """Identify a GPU's dbgpu :class:`GPUSpecification` from its name.

    Three-stage matching:

    1. Direct exact lookup (after stripping ``"NVIDIA "`` / ``"Tesla "``
       prefixes) — cheap, unambiguous when nvidia-smi happens to align
       with dbgpu's canonical name.
    2. Word-boundary token match against every catalog entry — the
       normalized input must appear as a complete whitespace-bounded
       token in the spec name. Word-boundary matching distinguishes
       "A10" (matches "A10 PCIe") from "A100" (does not collapse "A10"
       into it).
    3. When word-boundary returns multiple candidates and ``device_probe``
       is populated, score each candidate by how many probe fields agree
       (VRAM, CUDA compute capability, boost clock). Pick the candidate
       with the strictly-best score; tie → pick the lowest-bandwidth one
       as a safe under-estimate.

    dbgpu's bundled ``search()`` uses fuzzy ratio with no quality
    threshold and will happily return e.g. "Mobility FireGL 9000" for
    "BogusGPU-9000" — so we don't call it.

    Returns the matched ``GPUSpecification`` directly (so callers can
    read any field they need), or ``None`` on miss / error / ambiguity
    that the probe couldn't break. Never raises.
    """
    # Strip common vendor prefixes that nvidia-smi includes but dbgpu drops.
    normalized = gpu_name.strip()
    for prefix in ("NVIDIA ", "Tesla "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    try:
        from dbgpu import GPUDatabase
    except ImportError as e:
        log.debug("dbgpu unavailable (%s); skipping bandwidth lookup", e)
        return None

    try:
        db = GPUDatabase.default()
    except Exception as e:  # noqa: BLE001
        log.warning("dbgpu database load failed (%s); skipping bandwidth lookup", e)
        return None

    try:
        spec = db[normalized]
        if spec.memory_bandwidth_gb_s:
            return spec
    except KeyError:
        # Fall through to word-boundary token match.
        pass
    except Exception as e:  # noqa: BLE001
        log.warning(
            "dbgpu direct lookup failed for %r (%s); falling back to safe ceiling",
            gpu_name, e,
        )
        return None

    try:
        token_re = re.compile(rf"(?:^|\s){re.escape(normalized)}(?:\s|$)")
    except re.error:
        return None
    candidates = [
        s for s in db.specs
        if s.memory_bandwidth_gb_s and token_re.search(s.name)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    if device_probe is None:
        log.warning(
            "dbgpu: %r matched %d catalog entries (%s); no nvitop probe data "
            "to disambiguate, falling back to safe concurrency ceiling.",
            gpu_name, len(candidates), [s.name for s in candidates[:3]],
        )
        return None

    # PCIe link gen/width are veto-only: a candidate whose bus_interface
    # is strictly lower than the live card's reported max is impossible.
    survivors = [s for s in candidates if not _is_vetoed_by_bus(s, device_probe)]
    if not survivors:
        log.warning(
            "dbgpu: %r matched %d candidates (%s) but all were vetoed by the "
            "PCIe link probe (gen=%s width=%s); falling back to safe ceiling.",
            gpu_name, len(candidates), [s.name for s in candidates[:3]],
            device_probe.pcie_link_generation, device_probe.pcie_link_width,
        )
        return None
    if len(survivors) == 1:
        return survivors[0]
    candidates = survivors

    scored = [(_score_candidate(s, device_probe), s) for s in candidates]
    best_score = max(score for score, _ in scored)
    if best_score == 0:
        log.warning(
            "dbgpu: %r matched %d candidates (%s) but none agree with the "
            "nvitop probe; ambiguous, falling back to safe ceiling.",
            gpu_name, len(candidates), [s.name for s in candidates[:3]],
        )
        return None
    top = [s for score, s in scored if score == best_score]
    if len(top) > 1:
        chosen = min(top, key=lambda s: s.memory_bandwidth_gb_s)
        log.info(
            "dbgpu: %r → %d candidates tied at score %d (%s); picked %r "
            "(lowest bandwidth) as safe under-estimate.",
            gpu_name, len(top), best_score, [s.name for s in top], chosen.name,
        )
        return chosen
    return top[0]


def _bandwidth_concurrency_cap(spec: GPUSpecification) -> int:
    """Bandwidth-derived concurrency ceiling for a matched GPU."""
    bw = spec.memory_bandwidth_gb_s or 0.0
    return max(1, int(bw / BW_GB_S_PER_CONCURRENT_REQUEST))


def _timeout_concurrency_cap(
    profile: ModelProfile, gpu_spec: GPUSpecification | None
) -> int | None:
    """Concurrency cap that keeps avg request latency near the perf target.

    Memory-bandwidth-bound decode model: under continuous batching, every
    decode step reads the full model weights once and emits one token per
    in-flight sequence. So aggregate throughput ≈ ``bandwidth/model_bytes``
    independent of C, and per-sequence throughput ≈
    ``bandwidth/(model_bytes×C)``. A request emitting ``T_plan`` tokens
    therefore takes ~``T_plan × C × model_bytes / bandwidth`` seconds.
    Solving for C at ``AUTOSCALE_TARGET_REQUEST_LATENCY_S`` yields the cap.

    Returns ``None`` when we lack inputs (unknown GPU, missing model_size,
    no ``max_tokens``) — caller falls back to existing ceilings. The cap
    is not a safety check; the runtime safety check is the LiteLLM HTTP
    timeout in the backend.
    """
    if gpu_spec is None or not gpu_spec.memory_bandwidth_gb_s:
        return None
    max_tokens = (profile.sampling_args or {}).get("max_tokens")
    if not max_tokens:
        return None
    t_plan = (profile.metadata or {}).get("expected_output_tokens")
    if not t_plan:
        t_plan = max(1, int(int(max_tokens) * AUTOSCALE_DEFAULT_T_PLAN_FRACTION))
    # Bandwidth-bound decode reads only the weights per step (KV cache
    # contribution is small compared to weights for 1-7B-class models on
    # typical OCR sequence lengths), so we want pure weight bytes here,
    # not the weights + working overhead that estimate_model_overhead returns.
    if not profile.model_size:
        return None
    params = parse_model_size(profile.model_size)
    if params is None:
        return None
    dtype = str((profile.vllm_engine_args or {}).get("dtype", "auto")).lower()
    bytes_per_param = _DTYPE_BYTES_PER_PARAM.get(dtype, 2)
    model_bytes = int(params * bytes_per_param)
    if model_bytes <= 0:
        return None
    bw_bytes_s = gpu_spec.memory_bandwidth_gb_s * 1e9
    cap = int(
        AUTOSCALE_TARGET_REQUEST_LATENCY_S * bw_bytes_s
        / (int(t_plan) * model_bytes)
    )
    return max(1, cap)


def _concurrency_ceiling_for(
    profile: ModelProfile, gpu_spec: GPUSpecification | None = None
) -> int:
    """Per-profile ceiling on the autoscaler's auto-derived concurrency.

    When the GPU is identified via dbgpu, the ceiling is derived from the
    GPU's memory bandwidth (``bandwidth_gb_s / BW_GB_S_PER_CONCURRENT_REQUEST``);
    decode-step KV scans are bandwidth-bound, so a high-bandwidth GPU
    (H100/HBM) can sustain far more in-flight requests than a low-bandwidth
    one (L4/GDDR6 or GB10/LPDDR5X) before saturation.

    When the GPU is *not* in dbgpu, fall back to a safe per-backend ceiling:
    ``layout_chat`` profiles use ``AUTOSCALE_MAX_REGION_CONCURRENCY`` (16,
    empirically validated across H100/A10/L4/DGX Spark); other backends use
    ``AUTOSCALE_MAX_CONCURRENCY`` (64, whole-page batches stagger naturally).

    Only consulted in the auto-derived branch; an explicit
    ``--batch-concurrency N`` from the user bypasses this entirely — they
    get N regardless of GPU or backend, which is the right call for
    benchmarking.
    """
    if gpu_spec is not None:
        return _bandwidth_concurrency_cap(gpu_spec)
    if profile.backend == "layout_chat":
        return AUTOSCALE_MAX_REGION_CONCURRENCY
    return AUTOSCALE_MAX_CONCURRENCY


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
    spec: GPUSpecification | None = None
    """The matched dbgpu ``GPUSpecification`` when the device's nvitop
    properties uniquely identified a catalog entry. ``None`` for unknown
    or ambiguous GPUs — the autoscaler then falls back to the per-backend
    safe ceiling. Callers can read any field directly (``memory_bandwidth_gb_s``,
    ``memory_type``, ``boost_clock_mhz``, etc.)."""


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
    name = str(dev.name())
    return GpuProbe(
        name=name,
        total_bytes=total_b,
        free_bytes=free_b,
        cap_bytes=cap,
        spec=_lookup_gpu_specs(name, device_probe=_probe_device_properties(dev)),
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
            ceiling = _concurrency_ceiling_for(p, gpu.spec)
            timeout_cap = _timeout_concurrency_cap(p, gpu.spec)
            limits = [int(max_from_kv), ceiling]
            if timeout_cap is not None:
                limits.append(timeout_cap)
            concurrency = max(1, min(limits))
            kv = concurrency * per_request_kv

        decisions[p.name] = AutoscaleDecision(
            profile_name=p.name,
            kv_cache_memory_bytes=kv,
            concurrent_requests=concurrency,
            overhead_bytes=overheads[p.name],
            max_model_len=max_lens[p.name],
        )

    if gpu.spec is not None:
        bw_cap = _bandwidth_concurrency_cap(gpu.spec)
        gpu_line = (
            f"GPU {gpu.name}: free {human_bytes(gpu.free_bytes)} / total "
            f"{human_bytes(gpu.total_bytes)}; cap {human_bytes(gpu.cap_bytes)}; "
            f"bandwidth {gpu.spec.memory_bandwidth_gb_s:.0f} GB/s "
            f"(dbgpu: {gpu.spec.name!r}) → concurrency cap {bw_cap}; "
            f"perf target {AUTOSCALE_TARGET_REQUEST_LATENCY_S:.0f}s avg latency. "
            f"Auto-scaled {len(decisions)} profile(s):"
        )
    else:
        gpu_line = (
            f"GPU {gpu.name}: free {human_bytes(gpu.free_bytes)} / total "
            f"{human_bytes(gpu.total_bytes)}; cap {human_bytes(gpu.cap_bytes)}; "
            f"bandwidth unknown (not in dbgpu) → falling back to per-backend ceiling. "
            f"Auto-scaled {len(decisions)} profile(s):"
        )
    lines = [gpu_line]
    for p in profiles:
        d = decisions[p.name]
        timeout_cap = _timeout_concurrency_cap(p, gpu.spec)
        suffix = ""
        if timeout_cap is not None and timeout_cap == d.concurrent_requests:
            suffix = " [perf-target bound]"
        lines.append(
            f"  {p.name} (overhead {human_bytes(d.overhead_bytes)}, "
            f"KV {human_bytes(d.kv_cache_memory_bytes)}, "
            f"concurrency {d.concurrent_requests}){suffix}"
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

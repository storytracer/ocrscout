"""GPU pricing lookup for remote runners.

Maps GPU type labels (``L4``, ``RTX5000``, ``H100``, …) to bundled
provider pricing. Used by ``SkyPilotRunner`` and ``HuggingFaceRunner`` to
stamp ``cost_per_hour`` / ``provider`` into the SkyPilot job YAML's
``envs:`` block so the worker writes accurate cost columns into Parquet
without the user maintaining a parallel lookup.

User can override per-launch with ``--cost-per-hour`` / ``--provider``.
Falls back to ``(0.0, "unknown")`` when the GPU isn't in the table —
the run still proceeds; ``litellm_cost`` covers token-side accounting in
that case.
"""

from __future__ import annotations

from typing import NamedTuple


class GpuPrice(NamedTuple):
    provider: str
    cost_per_hour: float


# OVHcloud Managed Kubernetes pricing for the GPUs Sebastian's pilot is
# targeting. Pricing is in EUR (matches OVHcloud's billing) and is
# correct as of late 2025; users with different contracts should pass
# explicit ``--cost-per-hour`` on launch to override.
_OVHCLOUD: dict[str, float] = {
    "L4": 0.8925,
    "L40": 1.7800,
    "L40S": 1.7800,
    "A100": 2.9750,
    "H100": 4.4500,
    "RTX5000": 0.8500,
    "RTX6000": 1.9000,
}

# AWS on-demand prices for the same accelerators (us-east-1). Useful as
# a fallback when running SkyPilot against AWS rather than OVHcloud.
_AWS: dict[str, float] = {
    "L4": 0.9000,
    "L40": 1.8000,
    "A100": 3.0600,
    "H100": 4.8800,
    "T4": 0.5260,
    "V100": 3.0600,
    "A10G": 1.0060,
}

# Default lookup ordering. SkyPilot's ``--cloud`` (or the infra: hint in
# the YAML) selects a provider; if the caller doesn't pin one, we
# prefer OVHcloud since that's the configured K8s context for the
# initial pilot.
_PROVIDER_TABLES: dict[str, dict[str, float]] = {
    "ovhcloud": _OVHCLOUD,
    "aws": _AWS,
    "k8s": _OVHCLOUD,  # K8s on OVHcloud — same prices as OVHcloud rentals.
    "kubernetes": _OVHCLOUD,
}


def lookup(gpu_type: str, *, provider: str | None = None) -> GpuPrice:
    """Resolve a GPU label to a ``(provider, cost_per_hour)`` pair.

    When ``provider`` is given, look up only that table. Otherwise fall
    through every known provider in declaration order and return the
    first hit. Unknown GPU types return ``("unknown", 0.0)``.
    """
    if provider is not None:
        table = _PROVIDER_TABLES.get(provider.lower())
        if table is None or gpu_type not in table:
            return GpuPrice(provider=provider, cost_per_hour=0.0)
        return GpuPrice(provider=provider, cost_per_hour=table[gpu_type])

    for prov, table in _PROVIDER_TABLES.items():
        if gpu_type in table:
            return GpuPrice(provider=prov, cost_per_hour=table[gpu_type])
    return GpuPrice(provider="unknown", cost_per_hour=0.0)


def known_gpu_types(provider: str | None = None) -> list[str]:
    """List GPU labels we have pricing for (per-provider or union)."""
    if provider is None:
        out: set[str] = set()
        for table in _PROVIDER_TABLES.values():
            out.update(table.keys())
        return sorted(out)
    table = _PROVIDER_TABLES.get(provider.lower(), {})
    return sorted(table.keys())

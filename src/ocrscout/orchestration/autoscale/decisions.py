"""Typed autoscaler decisions — what the runner decided per profile at launch.

Replaces the untyped ``RunnerHandle.extra["autoscale"]`` dict and the
re-resolution of these values from env vars / ``state.yaml`` / profile YAML
scattered across the old run loop. The runner computes these once at launch and
they travel typed on the ``RunnerContext`` to the OCR stage.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AutoscaleDecision(BaseModel):
    """Per-profile concurrency / KV-cache the autoscaler chose at launch."""

    kv_cache_memory_bytes: int | None = None
    concurrent_requests: int | None = None
    region_concurrency: int | None = None


class AutoscaleContext(BaseModel):
    """All per-profile decisions for one launched stack, keyed by profile name."""

    decisions: dict[str, AutoscaleDecision] = Field(default_factory=dict)

    def for_profile(self, name: str) -> AutoscaleDecision:
        return self.decisions.get(name, AutoscaleDecision())

    @classmethod
    def from_backend_overrides(
        cls, overrides: dict[str, dict[str, int]] | None
    ) -> AutoscaleContext:
        """Adapt ``state.yaml``'s ``backend_overrides`` (the submit→worker
        handoff) into typed decisions."""
        decisions: dict[str, AutoscaleDecision] = {}
        for name, rec in (overrides or {}).items():
            decisions[name] = AutoscaleDecision(
                kv_cache_memory_bytes=rec.get("kv_cache_memory_bytes") or None,
                concurrent_requests=rec.get("concurrent_requests") or None,
                region_concurrency=rec.get("region_concurrency") or None,
            )
        return cls(decisions=decisions)

    @classmethod
    def from_handle_extra(cls, extra: dict | None) -> AutoscaleContext:
        """Adapt the existing ``RunnerHandle.extra['autoscale']['profiles']``
        shape into typed decisions (bridge during the rewrite)."""
        profiles = ((extra or {}).get("autoscale") or {}).get("profiles") or {}
        decisions: dict[str, AutoscaleDecision] = {}
        for name, rec in profiles.items():
            decisions[name] = AutoscaleDecision(
                kv_cache_memory_bytes=rec.get("kv_cache_memory_bytes") or None,
                concurrent_requests=rec.get("concurrent_requests") or None,
                region_concurrency=rec.get("region_concurrency") or None,
            )
        return cls(decisions=decisions)

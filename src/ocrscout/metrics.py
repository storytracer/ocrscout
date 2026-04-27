"""Picklable, context-manager-based metrics collector."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from ocrscout.types import RunMetrics


class MetricsCollector:
    """Collects per-stage timings and pipeline counters.

    Designed to be picklable across subprocess boundaries — internal state is a
    plain dict of JSON-friendly values; no file handles, no logger references.
    """

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id
        self.started_at = datetime.now(UTC)
        self.finished_at: datetime | None = None
        self._stage_seconds: dict[str, float] = {}
        self.pages_ok = 0
        self.pages_failed = 0
        self.tokens = 0
        self.gpu_peak_mb: float | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage. Re-entry sums into the existing total."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed

    def add_pages(self, *, ok: int = 0, failed: int = 0) -> None:
        self.pages_ok += ok
        self.pages_failed += failed

    def add_tokens(self, n: int) -> None:
        self.tokens += n

    def record_gpu_peak(self, mb: float) -> None:
        if self.gpu_peak_mb is None or mb > self.gpu_peak_mb:
            self.gpu_peak_mb = mb

    def finish(self) -> None:
        self.finished_at = datetime.now(UTC)

    def merge(self, other: MetricsCollector) -> None:
        for k, v in other._stage_seconds.items():
            self._stage_seconds[k] = self._stage_seconds.get(k, 0.0) + v
        self.pages_ok += other.pages_ok
        self.pages_failed += other.pages_failed
        self.tokens += other.tokens
        if other.gpu_peak_mb is not None:
            self.record_gpu_peak(other.gpu_peak_mb)

    @property
    def stage_seconds(self) -> dict[str, float]:
        return dict(self._stage_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "stage_seconds": dict(self._stage_seconds),
            "pages_ok": self.pages_ok,
            "pages_failed": self.pages_failed,
            "tokens": self.tokens,
            "gpu_peak_mb": self.gpu_peak_mb,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricsCollector:
        m = cls(d["pipeline_id"])
        m.started_at = datetime.fromisoformat(d["started_at"])
        if d.get("finished_at"):
            m.finished_at = datetime.fromisoformat(d["finished_at"])
        m._stage_seconds = dict(d.get("stage_seconds", {}))
        m.pages_ok = int(d.get("pages_ok", 0))
        m.pages_failed = int(d.get("pages_failed", 0))
        m.tokens = int(d.get("tokens", 0))
        m.gpu_peak_mb = d.get("gpu_peak_mb")
        return m

    def to_run_metrics(self) -> RunMetrics:
        return RunMetrics(
            pipeline_id=self.pipeline_id,
            started_at=self.started_at,
            finished_at=self.finished_at,
            pages_total=self.pages_ok + self.pages_failed,
            pages_ok=self.pages_ok,
            pages_failed=self.pages_failed,
            output_tokens=self.tokens,
            gpu_memory_peak_mb=self.gpu_peak_mb,
            stage_seconds=dict(self._stage_seconds),
        )

    # Pickle: explicit so we never accidentally pull in transient runtime state.
    def __getstate__(self) -> dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Rehydrate without calling __init__ (which would reset started_at).
        self.pipeline_id = state["pipeline_id"]
        self.started_at = datetime.fromisoformat(state["started_at"])
        self.finished_at = (
            datetime.fromisoformat(state["finished_at"]) if state.get("finished_at") else None
        )
        self._stage_seconds = dict(state.get("stage_seconds", {}))
        self.pages_ok = int(state.get("pages_ok", 0))
        self.pages_failed = int(state.get("pages_failed", 0))
        self.tokens = int(state.get("tokens", 0))
        self.gpu_peak_mb = state.get("gpu_peak_mb")

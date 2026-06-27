"""Finalize a page's cost/autoscaler columns after OCR.

Closes the in-process cost recorder for a page and projects its tokens / dollars
/ elapsed plus the GPU context and the autoscaler's per-profile decision into a
typed :class:`CostColumns`. (Was ``cli/run.py:finalize_cost_ctx`` returning an
untyped dict.)
"""

from __future__ import annotations

from ocrscout.cost import recorder
from ocrscout.io import CostColumns
from ocrscout.orchestration.autoscale import AutoscaleDecision
from ocrscout.state import GpuConfig


def finalize_cost_columns(
    page_id: str,
    *,
    gpu: GpuConfig | None,
    decision: AutoscaleDecision | None = None,
) -> CostColumns:
    metrics = recorder.close_page(page_id)
    gpu = gpu or GpuConfig()
    decision = decision or AutoscaleDecision()
    cols = CostColumns(
        gpu_type=gpu.type,
        provider=gpu.provider,
        cost_per_hour=gpu.cost_per_hour,
        kv_cache_memory_bytes=decision.kv_cache_memory_bytes,
        concurrent_requests=decision.concurrent_requests,
        region_concurrency=decision.region_concurrency,
    )
    if metrics is not None:
        cols.elapsed_seconds = metrics.elapsed_seconds
        cols.input_tokens = metrics.input_tokens
        cols.output_tokens = metrics.output_tokens
        cols.litellm_cost = metrics.litellm_cost
        cols.gpu_time_cost = metrics.elapsed_seconds / 3600.0 * gpu.cost_per_hour
    return cols

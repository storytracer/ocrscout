"""Per-page cost tracking via LiteLLM's success_callback.

LiteLLM fires ``success_callback`` after every successful completion call,
with the request/response objects and start/end timestamps. The callback
runs in-process (in ocrscout), not in the proxy — so we get accurate
timing and a process-local place to accumulate per-page metrics without
any IPC.

Correlation happens through the ``metadata`` kwarg passed to
``litellm.completion(metadata={"page_id": ..., ...})`` by the backend.
LiteLLM round-trips it to the callback in ``kwargs["litellm_params"]
["metadata"]``. A page that issues multiple requests (e.g. ``layout_chat``
fans out one call per region) accumulates here and is flushed when the
backend calls :meth:`CostRecorder.close_page`.

Cost numbers are best-effort: ``completion_cost`` returns ``0.0`` when the
proxy serves a model with no pricing configured (typical for self-hosted
vLLM unless ``model_info.input_cost_per_token`` is set in litellm.yaml).
The ``gpu_time_cost`` column (elapsed × cost-per-hour) covers that case —
both columns are populated so downstream queries can pick whichever
matches their accounting model.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class PageCostMetrics(BaseModel):
    """Accumulated cost/timing for one (page_id, model) pair.

    A page that issues multiple requests (layout-aware OCR) sums into the
    same record; ``request_count`` distinguishes per-page-per-request from
    per-page totals.
    """

    page_id: str
    model: str
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    litellm_cost: float = 0.0
    elapsed_seconds: float = 0.0


class CostRecorder:
    """Process-wide per-page cost accumulator.

    Thread-safe: LiteLLM may fire success_callback from worker threads when
    the backend POSTs requests concurrently. Backends call
    :meth:`open_page` before issuing requests for a page, and
    :meth:`close_page` after the page's requests have all completed; the
    callback fills in the middle via :meth:`record_request`.

    Pages not opened are silently ignored by ``record_request`` (defensive
    against callback-firing from libraries that share the same litellm
    import — e.g. an evaluation script that calls LiteLLM directly).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pages: dict[str, PageCostMetrics] = {}

    def open_page(self, page_id: str, model: str) -> None:
        with self._lock:
            self._pages[page_id] = PageCostMetrics(page_id=page_id, model=model)

    def record_request(
        self,
        *,
        page_id: str,
        cost: float,
        tokens_in: int,
        tokens_out: int,
        elapsed_s: float,
    ) -> None:
        with self._lock:
            page = self._pages.get(page_id)
            if page is None:
                # Unknown page (open_page was never called for it). Drop
                # silently — see class docstring.
                return
            page.request_count += 1
            page.input_tokens += tokens_in
            page.output_tokens += tokens_out
            page.litellm_cost += cost
            page.elapsed_seconds += elapsed_s

    def close_page(self, page_id: str) -> PageCostMetrics | None:
        with self._lock:
            return self._pages.pop(page_id, None)

    def peek(self, page_id: str) -> PageCostMetrics | None:
        with self._lock:
            page = self._pages.get(page_id)
            if page is None:
                return None
            return page.model_copy()


# Singleton — backends and the cost callback share state through this. A
# more functional API would thread the recorder through every call site,
# but litellm's callback machinery is itself a global, so matching that
# global keeps the wiring simple.
recorder = CostRecorder()


_CALLBACK_REGISTERED = False
_REGISTER_LOCK = threading.Lock()


def ensure_callback_registered() -> None:
    """Idempotently install ``_ocrscout_cost_callback`` on litellm.

    Safe to call multiple times (only the first call registers).
    Backends call this in their constructor or first-use path; importing
    this module does not register the callback so users who just want the
    types don't pay the cost of the litellm import.
    """
    global _CALLBACK_REGISTERED
    if _CALLBACK_REGISTERED:
        return
    with _REGISTER_LOCK:
        if _CALLBACK_REGISTERED:
            return
        import litellm

        existing = list(getattr(litellm, "success_callback", []) or [])
        if _ocrscout_cost_callback not in existing:
            existing.append(_ocrscout_cost_callback)
            litellm.success_callback = existing
        _CALLBACK_REGISTERED = True


def _ocrscout_cost_callback(
    kwargs: dict[str, Any],
    response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """LiteLLM success_callback: record per-request cost into the recorder.

    Extracts the correlating ``page_id`` from ``kwargs["litellm_params"]
    ["metadata"]``. Pricing comes from ``litellm.completion_cost`` (which
    consults LiteLLM's pricing DB plus any ``model_info`` overrides in the
    proxy config); usage tokens come from the response's ``usage`` dict.
    """
    try:
        meta = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        page_id = meta.get("page_id")
        if page_id is None:
            return

        import litellm

        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # noqa: BLE001
            # completion_cost raises when the model isn't priced. That's
            # the common case for self-hosted vLLM without model_info
            # overrides — fall back to 0 and let the gpu_time_cost column
            # cover infra accounting.
            cost = 0.0

        usage = _extract_usage(response)
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
        elapsed_s = _elapsed_seconds(start_time, end_time)

        recorder.record_request(
            page_id=str(page_id),
            cost=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_s=elapsed_s,
        )
    except Exception as e:  # noqa: BLE001
        # A callback that raises pollutes every subsequent completion call
        # in libraries that share the litellm global. Log and swallow.
        log.debug("ocrscout cost callback failed: %s", e)


def _extract_usage(response: Any) -> dict[str, Any]:
    """Pull ``usage`` off a litellm response (dict or pydantic object)."""
    if response is None:
        return {}
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    # pydantic-ish — use model_dump if available, else __dict__
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(getattr(usage, "__dict__", {}))


def _elapsed_seconds(start: Any, end: Any) -> float:
    """LiteLLM may pass datetimes, floats, or None; coerce to seconds."""
    if start is None or end is None:
        return 0.0
    if isinstance(start, datetime) and isinstance(end, datetime):
        return max(0.0, (end - start).total_seconds())
    try:
        return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return 0.0

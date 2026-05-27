"""LayoutChatBackend: layout-detector-driven, region-level OCR over LiteLLM.

A single producer thread iterates pages and runs the layout detector
serially (one detector instance, GPU-resident); for each detected region
it submits a POST task to a shared ``ThreadPoolExecutor``. A
``BoundedSemaphore`` caps in-flight region POSTs across pages — the same
``region_concurrency`` knob now bounds the cross-page in-flight queue,
not a per-page batch. As each region completes, the worker stores its
block in a per-page assembler under a global lock; when a page's last
region completes, its ``page_id`` is enqueued onto a completion queue
that the generator drains and yields. Pages yield in completion order,
not input order — the caller (``cli/run.py:_run_one_model``) materializes
via ``list()`` and re-keys by ``page_id``, so this is invisible.

Two stalls this design eliminates vs. strict per-page sequencing:

- Detector dead time between pages: vLLM keeps draining region POSTs
  from previously-detected pages while the next page's layout runs.
- Slow-region tail: a page's slowest region no longer holds back its
  own page-yield + every subsequent page's start; it's just one of N
  in-flight items competing for cycles.

Requires an active Runner (LiteLLM proxy + at least one backing vLLM
serve). Subprocess vLLM is unsupported here because the per-region launch
cost would dwarf any inference time.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ocrscout import cost as cost_mod
from ocrscout.backends.litellm import (
    _build_messages,
    _split_sampling,
    _state_override,
)
from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.registry import registry
from ocrscout.types import BackendInvocation, LayoutRegion, PageImage, RawOutput

log = logging.getLogger(__name__)

_DEFAULT_REGION_CONCURRENCY = 8
"""Fallback cross-page region concurrency. Normally filled in by the
autoscaler via ``backend_args.region_concurrency`` (or the runner's
state file for submit-time workers); only applies when both lookups
return nothing."""
_DEFAULT_REQUEST_TIMEOUT = 300.0
_MODELS_PROBE_TIMEOUT = 15.0
# Vertical bucketing for top-then-left reading-order sort. 50 px tolerates
# small same-row jitter without conflating rows on tightly-packed pages.
_READING_ORDER_ROW_PX = 50

_SENTINEL: Any = object()
"""Marker pushed onto ``done_queue`` by the producer's ``finally`` so the
yielder unblocks even when the producer crashed before completing every
page."""


@dataclass
class _PageState:
    """Per-page assembly state shared between the producer (initial write)
    and worker threads (post-completion mutation under ``state_lock``)."""

    page: PageImage
    page_idx: int
    ordered: list[LayoutRegion]
    total_regions: int
    remaining: int
    t_start: float
    results: dict[int, dict[str, Any]] = field(default_factory=dict)
    detector_error: str | None = None


def _resolve_region_concurrency(profile: ModelProfile) -> int:
    """Precedence: explicit profile value > state-file override > default.

    Same semantics as the litellm backend's ``_resolve_concurrent_requests``;
    state-file is the launch → submit → worker handoff path.
    """
    explicit = (profile.backend_args or {}).get("region_concurrency")
    if explicit is not None:
        return int(explicit)
    override = _state_override(profile.name, "region_concurrency")
    if override is not None:
        return override
    return _DEFAULT_REGION_CONCURRENCY


class LayoutChatBackend(ModelBackend):
    """Layout-aware OCR over the LiteLLM proxy."""

    name = "layout_chat"
    requires_runner: ClassVar[bool] = True

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        if not profile.layout_detector:
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} has no layout_detector"
            )
        if not profile.prompt_templates:
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} has no prompt_templates"
            )

        proxy_url = os.environ.get("OCRSCOUT_VLLM_URL") or profile.server_url
        if not proxy_url:
            raise BackendError(
                "LayoutChatBackend requires a LiteLLM proxy URL. Launch a "
                "runner (`ocrscout launch --models ...`) or set "
                "OCRSCOUT_VLLM_URL."
            )
        proxy_url = proxy_url.rstrip("/")

        # Probe ``/models`` so a wrong URL fails up-front rather than after
        # every region 404s silently inside _post_region. The proxy
        # advertises the profile name (not the model_id) when its model_list
        # was generated from the same profile, so check against either.
        served = _list_proxy_models(proxy_url, timeout=_MODELS_PROBE_TIMEOUT)
        if profile.name not in served and profile.model_id not in served:
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} (model_id "
                f"{profile.model_id!r}) is not served by the LiteLLM proxy at "
                f"{proxy_url}; proxy serves {sorted(served)!r}."
            )

        detector_cls = registry.get("layout_detectors", profile.layout_detector)
        try:
            detector = detector_cls(**profile.layout_detector_args)
        except Exception as e:
            raise BackendError(
                f"LayoutChatBackend: failed to instantiate layout detector "
                f"{profile.layout_detector!r}: {e}"
            ) from e

        cost_mod.ensure_callback_registered()

        return BackendInvocation(
            kind="http",
            endpoint=proxy_url,
            profile=profile,
            pages=[p.page_id for p in pages],
            extra={
                "pages_runtime": list(pages),
                "detector": detector,
            },
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        profile = invocation.profile
        proxy_url = invocation.endpoint or ""
        if not proxy_url:
            raise BackendError("LayoutChatBackend.run: missing proxy URL")

        pages: list[PageImage] = list(invocation.extra.get("pages_runtime", []))
        detector = invocation.extra["detector"]
        timeout = float(
            profile.backend_args.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT)
        )
        region_concurrency = max(1, _resolve_region_concurrency(profile))
        sampling = _split_sampling(profile.sampling_args or {})
        prefix = f"[{profile.name}]"
        total_pages = len(pages)

        log.info(
            "%s starting %d page(s) against %s "
            "(region concurrency=%d cross-page, detector=%s)",
            prefix, total_pages, proxy_url, region_concurrency, profile.layout_detector,
        )

        if total_pages == 0:
            return

        t_total = time.perf_counter()

        inflight_sem = threading.BoundedSemaphore(region_concurrency)
        state_lock = threading.Lock()
        page_states: dict[str, _PageState] = {}
        done_queue: queue.Queue[Any] = queue.Queue()
        producer_exc: list[BaseException] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=region_concurrency,
            thread_name_prefix="layout_chat_region",
        ) as executor:

            def _producer_loop() -> None:
                try:
                    for page_idx, page in enumerate(pages, start=1):
                        _produce_one_page(
                            page=page,
                            page_idx=page_idx,
                            total_pages=total_pages,
                            detector=detector,
                            proxy_url=proxy_url,
                            profile=profile,
                            sampling=sampling,
                            timeout=timeout,
                            log_prefix=prefix,
                            page_states=page_states,
                            state_lock=state_lock,
                            inflight_sem=inflight_sem,
                            done_queue=done_queue,
                            executor=executor,
                        )
                except BaseException as e:  # noqa: BLE001
                    producer_exc.append(e)
                finally:
                    done_queue.put(_SENTINEL)

            producer = threading.Thread(
                target=_producer_loop,
                name="layout_chat_producer",
                daemon=True,
            )
            producer.start()

            yielded = 0
            try:
                while yielded < total_pages:
                    item = done_queue.get()
                    if item is _SENTINEL:
                        break
                    state = page_states.pop(item)
                    yield _assemble_raw(state, profile, prefix, total_pages)
                    yielded += 1
            finally:
                producer.join()

        if producer_exc:
            raise producer_exc[0]

        log.info(
            "%s layout-chat finished %d pages in %.1fs",
            prefix, total_pages, time.perf_counter() - t_total,
        )


def _produce_one_page(
    *,
    page: PageImage,
    page_idx: int,
    total_pages: int,
    detector: Any,
    proxy_url: str,
    profile: ModelProfile,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
    log_prefix: str,
    page_states: dict[str, _PageState],
    state_lock: threading.Lock,
    inflight_sem: threading.BoundedSemaphore,
    done_queue: queue.Queue[Any],
    executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    """Producer-side body for one page: detect layout, register page state,
    then submit one task per region under the in-flight semaphore."""
    cost_mod.recorder.open_page(page.page_id, profile.name)
    t_start = time.perf_counter()

    try:
        regions = detector.detect(page)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "%s page %d/%d %s detector FAIL: %s",
            log_prefix, page_idx, total_pages, page.file_id, e,
        )
        page_states[page.page_id] = _PageState(
            page=page,
            page_idx=page_idx,
            ordered=[],
            total_regions=0,
            remaining=0,
            t_start=t_start,
            detector_error=f"{type(e).__name__}: {e}",
        )
        done_queue.put(page.page_id)
        return

    if not regions:
        page_states[page.page_id] = _PageState(
            page=page,
            page_idx=page_idx,
            ordered=[],
            total_regions=0,
            remaining=0,
            t_start=t_start,
        )
        done_queue.put(page.page_id)
        return

    ordered = _sort_reading_order(regions)
    # Publish state BEFORE any submit so worker callbacks always find it.
    page_states[page.page_id] = _PageState(
        page=page,
        page_idx=page_idx,
        ordered=ordered,
        total_regions=len(ordered),
        remaining=len(ordered),
        t_start=t_start,
    )

    for region in ordered:
        inflight_sem.acquire()
        executor.submit(
            _make_region_worker(
                page_id=page.page_id,
                region=region,
                page=page,
                profile=profile,
                proxy_url=proxy_url,
                sampling=sampling,
                timeout=timeout,
                page_states=page_states,
                state_lock=state_lock,
                inflight_sem=inflight_sem,
                done_queue=done_queue,
            )
        )


def _make_region_worker(
    *,
    page_id: str,
    region: LayoutRegion,
    page: PageImage,
    profile: ModelProfile,
    proxy_url: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
    page_states: dict[str, _PageState],
    state_lock: threading.Lock,
    inflight_sem: threading.BoundedSemaphore,
    done_queue: queue.Queue[Any],
) -> Callable[[], None]:
    """Build the closure submitted to the executor for one region.

    The closure runs ``_post_region``, releases the in-flight semaphore in
    ``finally`` (so even an unhandled error frees the slot), then mutates
    the page's state under ``state_lock``; when the page's last region
    completes, it enqueues the page_id onto ``done_queue``."""

    def _work() -> None:
        try:
            block = _post_region(
                region=region,
                page=page,
                profile=profile,
                proxy_url=proxy_url,
                sampling=sampling,
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001  defense-in-depth; _post_region catches its own
            block = _failed_block(region, error=f"{type(e).__name__}: {e}")
        finally:
            inflight_sem.release()
        with state_lock:
            state = page_states[page_id]
            state.results[region.id] = block
            state.remaining -= 1
            is_done = state.remaining == 0
        if is_done:
            done_queue.put(page_id)

    return _work


def _assemble_raw(
    state: _PageState,
    profile: ModelProfile,
    log_prefix: str,
    total_pages: int,
) -> RawOutput:
    """Build the per-page ``RawOutput`` from accumulated worker results."""
    page = state.page
    page_idx = state.page_idx

    if state.detector_error is not None:
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error=f"layout detector failed: {state.detector_error}",
        )

    if state.total_regions == 0:
        log.info(
            "%s page %d/%d %s no regions detected",
            log_prefix, page_idx, total_pages, page.file_id,
        )
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload=json.dumps([]),
        )

    blocks = [state.results[r.id] for r in state.ordered]
    n_failed = sum(1 for b in blocks if b.get("error"))
    elapsed = time.perf_counter() - state.t_start
    if n_failed:
        log.warning(
            "%s page %d/%d %s ok (%d/%d regions in %.1fs, %d failed)",
            log_prefix, page_idx, total_pages, page.file_id,
            state.total_regions - n_failed, state.total_regions, elapsed, n_failed,
        )
    else:
        log.info(
            "%s page %d/%d %s ok (%d regions in %.1fs)",
            log_prefix, page_idx, total_pages, page.file_id,
            state.total_regions, elapsed,
        )

    return RawOutput(
        page_id=page.page_id,
        output_format=profile.output_format,
        payload=json.dumps(blocks),
    )


def _post_region(
    *,
    region: LayoutRegion,
    page: PageImage,
    profile: ModelProfile,
    proxy_url: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    """POST one cropped region through LiteLLM; return a layout-JSON block dict."""
    if page.image is None:
        return _failed_block(region, error="page has no in-memory image")

    crop = page.image.crop(region.bbox)
    prompt_template = _resolve_region_prompt(profile, region)
    prompt = _substitute_region_dims(prompt_template, crop)
    messages = _build_messages(crop, prompt)
    top_level, extra_body = sampling

    import litellm

    try:
        resp = litellm.completion(
            model=profile.name,
            custom_llm_provider="openai",
            api_base=proxy_url,
            api_key=os.environ.get("LITELLM_API_KEY", "ocrscout-dummy"),
            messages=messages,
            timeout=timeout,
            metadata={
                "page_id": page.page_id,
                "region_id": region.id,
                "model_name": profile.name,
            },
            extra_body=extra_body or None,
            **top_level,
        )
    except Exception as e:  # noqa: BLE001
        return _failed_block(region, error=f"{type(e).__name__}: {e}")

    text = _extract_text(resp)
    return _ok_block(region, text=text)


def _extract_text(resp: Any) -> str:
    try:
        choices = getattr(resp, "choices", None) or (
            resp.get("choices") if isinstance(resp, dict) else None
        )
        if not choices:
            return ""
        first = choices[0]
        msg = getattr(first, "message", None) or (
            first.get("message") if isinstance(first, dict) else None
        )
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return content or ""
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""


def _ok_block(region: LayoutRegion, *, text: str) -> dict[str, Any]:
    block: dict[str, Any] = {
        "category": region.category,
        "bbox": list(region.bbox),
        "text": text,
    }
    if region.score is not None:
        block["score"] = region.score
    return block


def _failed_block(region: LayoutRegion, *, error: str) -> dict[str, Any]:
    block = _ok_block(region, text="")
    block["error"] = error
    return block


def _resolve_region_prompt(profile: ModelProfile, region: LayoutRegion) -> str:
    """Pick the prompt template for a region.

    Lookup precedence:
        ``profile.prompt_mode_per_category[region.category]``
        → ``profile.preferred_prompt_mode``
        → first key in ``profile.prompt_templates`` (deterministic fallback).
    """
    templates = profile.prompt_templates
    mode = profile.prompt_mode_per_category.get(region.category)
    if mode is None or mode not in templates:
        mode = profile.preferred_prompt_mode
    if mode is None or mode not in templates:
        mode = next(iter(templates))
    return templates[mode]


def _substitute_region_dims(template: str, crop: Any) -> str:
    """Substitute ``{width}``/``{height}`` against the *region* dimensions."""
    if "{width}" not in template and "{height}" not in template:
        return template
    w, h = crop.size  # PIL .size is (width, height)
    return template.replace("{width}", str(w)).replace("{height}", str(h))


def _sort_reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Order regions for downstream document body assembly.

    If every region carries a non-None ``reading_order`` (the detector
    predicted it — e.g. PP-DocLayoutV3 emits results in reading order), use
    that. Otherwise fall back to a top-then-left bucketed sort that
    tolerates small same-row jitter on tightly-packed pages.
    """
    if regions and all(r.reading_order is not None for r in regions):
        return sorted(regions, key=lambda r: (r.reading_order or 0, r.bbox[1], r.bbox[0]))

    def heuristic_key(r: LayoutRegion) -> tuple[int, float]:
        top = r.bbox[1]
        left = r.bbox[0]
        return (int(round(top / _READING_ORDER_ROW_PX)) * _READING_ORDER_ROW_PX, left)

    return sorted(regions, key=heuristic_key)


def _list_proxy_models(proxy_url: str, *, timeout: float) -> set[str]:
    """GET ``{proxy_url}/models`` and return the served model_name set.

    Uses stdlib ``urllib`` instead of ``requests`` because this is a tiny
    one-shot probe; pulling ``requests`` in just to fail fast on URL
    misconfig isn't worth the dep weight in a backend that otherwise
    talks via ``litellm`` (which manages its own HTTP layer).
    """
    url = f"{proxy_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise BackendError(
            f"LayoutChatBackend: cannot reach LiteLLM proxy at {url}: {e}"
        ) from e
    try:
        return {entry["id"] for entry in data["data"]}
    except (KeyError, TypeError) as e:
        raise BackendError(
            f"LayoutChatBackend: malformed /models response from {url}: {e}"
        ) from e

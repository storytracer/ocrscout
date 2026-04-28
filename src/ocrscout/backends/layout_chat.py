"""LayoutChatBackend: layout-detector-driven, region-level OCR.

For each page:

1. Run a registered ``LayoutDetector`` to find typed regions on the page.
2. Sort regions in a stable reading order (top-then-left, with vertical
   bucketing to tolerate small jitter on same-row items).
3. For each region: crop the source image, pick the prompt template based on
   the region's *detector-native* category (via
   ``profile.prompt_mode_per_category``, falling back to
   ``preferred_prompt_mode``), and POST one chat-completion to the
   OpenAI-compatible endpoint.
4. Compose a ``layout_json`` payload — one block per region with the
   page-coordinate bbox (NOT the crop coordinates) and the OCR text — and
   yield it as a single ``RawOutput`` per page.

Server-mode-only: subprocess vLLM is unsupported here because per-region
subprocess re-spawn would dwarf any inference time.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import requests

from ocrscout.backends._openai_chat import (
    ChatCompletionError,
    build_base_payload,
    build_chat_request_body,
    list_models,
    normalize_endpoint,
    post_chat_completion,
)
from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.registry import registry
from ocrscout.types import BackendInvocation, LayoutRegion, PageImage, RawOutput

log = logging.getLogger(__name__)

_DEFAULT_REGION_CONCURRENCY = 16
_DEFAULT_REQUEST_TIMEOUT = 300.0
_MODELS_PROBE_TIMEOUT = 15.0
# Vertical bucketing for top-then-left reading-order sort. 50 px tolerates
# small same-row jitter without conflating rows on tightly-packed pages.
_READING_ORDER_ROW_PX = 50


class LayoutChatBackend(ModelBackend):
    """Layout-aware OCR backend over an OpenAI-compatible /chat/completions
    endpoint."""

    name = "layout_chat"
    # Tells managed-mode lifecycle that profiles using this backend need a
    # vLLM serve subprocess spawned for them, just like ``source: vllm``.
    requires_managed_vllm: ClassVar[bool] = True

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

        server_url = os.environ.get("OCRSCOUT_VLLM_URL") or profile.server_url
        if not server_url:
            raise BackendError(
                "LayoutChatBackend requires an OpenAI-compatible server. "
                "Set --server-url, --managed, or profile.server_url. "
                "Subprocess vLLM is not supported in layout mode."
            )
        endpoint = normalize_endpoint(server_url)

        # Probe /models before doing anything expensive: catches "wrong
        # --server-url" and "model not loaded on this endpoint" up front,
        # rather than after every region 404s silently inside _post_region.
        session = requests.Session()
        try:
            served = list_models(session, endpoint, timeout=_MODELS_PROBE_TIMEOUT)
        except ChatCompletionError as e:
            session.close()
            raise BackendError(
                f"LayoutChatBackend: failed to query {endpoint}/models for "
                f"profile {profile.name!r}: {e}"
            ) from e
        if profile.model_id not in served:
            session.close()
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} expects model "
                f"{profile.model_id!r} but {endpoint} serves {served!r}. "
                f"Check --server-url (proxy URL when --managed runs multiple models)."
            )

        detector_cls = registry.get("layout_detectors", profile.layout_detector)
        try:
            detector = detector_cls(**profile.layout_detector_args)
        except Exception as e:
            session.close()
            raise BackendError(
                f"LayoutChatBackend: failed to instantiate layout detector "
                f"{profile.layout_detector!r}: {e}"
            ) from e

        return BackendInvocation(
            kind="http",
            endpoint=endpoint,
            profile=profile,
            pages=[p.page_id for p in pages],
            extra={
                "pages_runtime": list(pages),
                "detector": detector,
                "session": session,
            },
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        endpoint = normalize_endpoint(invocation.endpoint or "")
        if not endpoint:
            raise BackendError("LayoutChatBackend.run: missing endpoint URL")

        profile = invocation.profile
        pages: list[PageImage] = list(invocation.extra.get("pages_runtime", []))
        detector = invocation.extra["detector"]
        session: requests.Session = invocation.extra["session"]

        timeout = float(profile.backend_args.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT))
        region_concurrency = max(
            1,
            int(profile.backend_args.get("region_concurrency", _DEFAULT_REGION_CONCURRENCY)),
        )

        base_payload = build_base_payload(profile)
        url = f"{endpoint}/chat/completions"
        prefix = f"[{profile.name}]"

        log.info(
            "%s starting %d page(s) against %s (region concurrency=%d, detector=%s)",
            prefix, len(pages), endpoint, region_concurrency, profile.layout_detector,
        )
        log.debug("%s POST URL: %s", prefix, url)

        try:
            t_total = time.perf_counter()
            for page_idx, page in enumerate(pages, start=1):
                yield self._run_one_page(
                    page=page,
                    page_idx=page_idx,
                    total_pages=len(pages),
                    detector=detector,
                    session=session,
                    url=url,
                    base_payload=base_payload,
                    profile=profile,
                    timeout=timeout,
                    region_concurrency=region_concurrency,
                    log_prefix=prefix,
                )
            log.info(
                "%s layout-chat finished %d pages in %.1fs",
                prefix, len(pages), time.perf_counter() - t_total,
            )
        finally:
            session.close()

    def _run_one_page(
        self,
        *,
        page: PageImage,
        page_idx: int,
        total_pages: int,
        detector: Any,
        session: requests.Session,
        url: str,
        base_payload: dict[str, Any],
        profile: ModelProfile,
        timeout: float,
        region_concurrency: int,
        log_prefix: str,
    ) -> RawOutput:
        try:
            regions = detector.detect(page)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "%s page %d/%d %s detector FAIL: %s",
                log_prefix, page_idx, total_pages, page.file_id, e,
            )
            return RawOutput(
                page_id=page.page_id,
                output_format=profile.output_format,
                payload="",
                error=f"layout detector failed: {type(e).__name__}: {e}",
            )

        if not regions:
            log.info(
                "%s page %d/%d %s no regions detected",
                log_prefix, page_idx, total_pages, page.file_id,
            )
            return RawOutput(
                page_id=page.page_id,
                output_format=profile.output_format,
                payload=json.dumps([]),
            )

        ordered = _sort_reading_order(regions)
        t0 = time.perf_counter()
        results: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=region_concurrency) as ex:
            futures = {
                ex.submit(
                    _post_region,
                    region=region,
                    page=page,
                    profile=profile,
                    session=session,
                    url=url,
                    base_payload=base_payload,
                    timeout=timeout,
                ): region
                for region in ordered
            }
            for fut in concurrent.futures.as_completed(futures):
                region = futures[fut]
                try:
                    block = fut.result()
                except Exception as e:  # noqa: BLE001
                    block = _failed_block(region, error=f"{type(e).__name__}: {e}")
                results[region.id] = block

        # Re-emit blocks in the sorted reading-order so downstream consumers
        # walk the document body in a sensible sequence.
        blocks = [results[r.id] for r in ordered]
        n_failed = sum(1 for b in blocks if b.get("error"))
        elapsed = time.perf_counter() - t0
        if n_failed:
            log.warning(
                "%s page %d/%d %s ok (%d/%d regions in %.1fs, %d failed)",
                log_prefix, page_idx, total_pages, page.file_id,
                len(ordered) - n_failed, len(ordered), elapsed, n_failed,
            )
        else:
            log.info(
                "%s page %d/%d %s ok (%d regions in %.1fs)",
                log_prefix, page_idx, total_pages, page.file_id,
                len(ordered), elapsed,
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
    session: requests.Session,
    url: str,
    base_payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST one cropped region; return a layout-JSON block dict."""
    if page.image is None:
        return _failed_block(region, error="page has no in-memory image")

    crop = page.image.crop(region.bbox)
    prompt_template = _resolve_region_prompt(profile, region)
    prompt = _substitute_region_dims(prompt_template, crop)

    body = build_chat_request_body(crop, prompt, base_payload)
    try:
        text, _tokens = post_chat_completion(session, url, body, timeout=timeout)
    except ChatCompletionError as e:
        return _failed_block(region, error=str(e))

    return _ok_block(region, text=text)


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

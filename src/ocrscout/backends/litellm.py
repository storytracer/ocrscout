"""LiteLLMBackend: full-page OCR over a LiteLLM proxy.

Used by every profile whose ``backend: litellm`` — both ``runtime: vllm``
(local OSS VLMs served by ``vllm serve`` behind the proxy) and
``runtime: hosted`` (Gemini, Anthropic, etc., routed by the proxy's
``model_list``). The two only differ in what the active Runner has wired
up behind the proxy; from the backend's perspective every call is the
same ``litellm.completion(model=..., api_base=proxy_url)``.

LiteLLM's success_callback fires in-process (see ``ocrscout.cost``) and
attributes the request to a page via ``metadata={"page_id": ...}``, so
per-page token / cost / elapsed metrics flow into the Parquet export
without any extra plumbing.

The proxy URL comes from (in precedence order):

1. ``OCRSCOUT_VLLM_URL`` env var (set by ``ocrscout run`` / Local /
   SkyPilot / HF runners as part of launch).
2. ``profile.server_url`` for one-off use against a hand-managed proxy.

Subprocess one-shot mode (the old VllmBackend default) is gone — the
LiteLLM proxy is always in the loop.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

from PIL.Image import Image as PILImage

from ocrscout import cost as cost_mod
from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput

log = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT = 300.0
"""Per-request timeout (seconds) passed to ``litellm.completion``. Five
minutes is already 5–10× the steady-state per-page latency for every
shipped VLM profile on every supported GPU; anything past that is dead
weight, not slow. Combined with :data:`_DEFAULT_NUM_RETRIES`, worst-case
wall-clock per page is roughly ``timeout * (1 + num_retries)``."""
_DEFAULT_NUM_RETRIES = 2
"""How many retries LiteLLM performs on retryable failures (rate limits,
5xx, timeouts). LiteLLM classifies exceptions itself — it doesn't retry
``ContextWindowExceededError`` / ``BadRequestError`` / auth errors,
which would be deterministic re-failures. Two retries with exponential
backoff covers transient vLLM overload + sporadic decode glitches
without ballooning total wall-clock."""
_DEFAULT_CONCURRENT_REQUESTS = 8
"""Fallback per-page client-side concurrency. The GPU-aware autoscaler in
:mod:`ocrscout.runners._preflight` normally fills this in via
``backend_args.concurrent_requests`` (or via the runner's state file for
submit-time workers); this default only applies to hosted profiles, manual
single-shot calls without a Runner, or when both lookups return nothing."""


def _resolve_concurrent_requests(profile: ModelProfile) -> int:
    """Precedence: explicit profile value > state-file override > default.

    State-file override is how the launch → submit → worker handoff
    inherits the autoscaler's decision: ``LocalRunner._launch_persistent``
    writes ``state.backend_overrides[profile.name]["concurrent_requests"]``
    and the worker process (separate Python interpreter, re-resolves
    profiles from YAML) reads it back here.
    """
    explicit = (profile.backend_args or {}).get("concurrent_requests")
    if explicit is not None:
        return int(explicit)
    override = _state_override(profile.name, "concurrent_requests")
    if override is not None:
        return override
    return _DEFAULT_CONCURRENT_REQUESTS


_BACKEND_OVERRIDES_ENV = "OCRSCOUT_BACKEND_OVERRIDES"
"""Env-var carrier for the autoscaler's per-profile concurrency
decisions during in-process (ephemeral) runs. Format is a JSON-encoded
``{profile_name: {key: int}}`` mapping. ``run_pipeline`` sets this
after each ``runner.launch(...)`` so backends and ExportRecord-stamping
see the same numbers without round-tripping through state.yaml.

Persistent / launch + submit + worker handoff uses state.yaml
(``RunnerStateFile.backend_overrides``) instead — workers are separate
Python processes that don't inherit the launching shell's env."""


def _state_override(profile_name: str, key: str) -> int | None:
    """Read the autoscaler's per-profile concurrency decision.

    Sources, in precedence order:

    * ``OCRSCOUT_BACKEND_OVERRIDES`` env var — in-process channel set by
      ``run_pipeline`` after each ephemeral launch.
    * ``state.yaml`` ``backend_overrides`` — cross-process channel set
      by ``LocalRunner._launch_persistent`` for submit-time workers.

    Returns ``None`` when neither is available; the caller falls back to
    the module default. Swallows JSON / state errors so a malformed
    side-channel never blocks the backend from running.
    """
    raw = os.environ.get(_BACKEND_OVERRIDES_ENV)
    if raw:
        try:
            import json as _json

            data = _json.loads(raw)
            v = data.get(profile_name, {}).get(key)
            if v is not None:
                return int(v)
        except (ValueError, TypeError):
            pass

    from ocrscout import state as state_mod

    try:
        state = state_mod.read_state()
    except Exception:  # noqa: BLE001
        return None
    if state is None:
        return None
    rec = state.backend_overrides.get(profile_name)
    if not rec:
        return None
    val = rec.get(key)
    return int(val) if val is not None else None

# Subset of vLLM/OpenAI sampling fields the proxy forwards. Anything outside
# this allowlist is passed through ``extra_body`` so vLLM-specific extensions
# (``top_k``, ``repetition_penalty``, …) still reach the engine.
_TOP_LEVEL_SAMPLING_KEYS: tuple[str, ...] = (
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
)


class LiteLLMBackend(ModelBackend):
    """Full-page OCR through a LiteLLM proxy.

    Drives both ``runtime: vllm`` and ``runtime: hosted`` profiles — the
    Runner shapes what's behind the proxy; the backend code is identical.
    """

    name = "litellm"
    requires_runner: ClassVar[bool] = True
    """Requires an active Runner (LiteLLM proxy + at least one backing
    endpoint). LocalRunner uses this flag to know whether to spawn
    anything for this profile."""

    def prepare(self, profile: ModelProfile) -> BackendInvocation:
        proxy_url = _resolve_proxy_url(profile)
        prompt = _resolve_prompt_template(profile)

        # Register the cost callback once per process; safe to call repeatedly.
        cost_mod.ensure_callback_registered()

        return BackendInvocation(
            kind="http",
            endpoint=proxy_url,
            profile=profile,
            pages=[],
            extra={"prompt": prompt},
        )

    def run(
        self,
        invocation: BackendInvocation,
        pages: Sequence[PageImage],
    ) -> Iterator[tuple[PageImage, RawOutput]]:
        profile = invocation.profile
        proxy_url = invocation.endpoint or ""
        if not proxy_url:
            raise BackendError("LiteLLMBackend.run: no proxy URL on invocation")

        pages_list: list[PageImage] = list(pages)
        if not pages_list:
            return
        prompt: str = invocation.extra["prompt"]
        timeout = float(
            profile.backend_args.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT)
        )
        num_retries = int(
            profile.backend_args.get("num_retries", _DEFAULT_NUM_RETRIES)
        )
        concurrent_requests = max(
            1, _resolve_concurrent_requests(profile),
        )
        sampling = _split_sampling(profile.sampling_args or {})
        prefix = f"[{profile.name}]"
        total = len(pages_list)
        log.info(
            "%s batch of %d page(s) against %s (%d-way concurrent)",
            prefix, total, proxy_url, concurrent_requests,
        )

        completed = 0
        t0 = time.perf_counter()

        # Streaming submit pump: keep at most ``concurrent_requests``
        # futures in flight at any moment, refill on completion. Replaces
        # the previous "submit-all-up-front" pattern that pinned every
        # batch page in a futures dict for the lifetime of the slowest
        # request — that pattern held the per-batch PageImage list resident
        # alongside the in-flight base64 message bodies, and at concurrency
        # 40 over an 80-page chunk it doubled the high-water mark for no
        # throughput benefit (vLLM's continuous batching schedules
        # whichever requests are POSTed, regardless of futures-dict size).
        #
        # Pages yield in completion order, so the orchestrator can export
        # and drop each one as its raw arrives. ``pages_list`` references
        # drain naturally as ``in_flight.pop`` returns each page; the only
        # live PIL bytes are those inside the ``concurrent_requests``
        # workers currently inside ``with page.open_image()`` in
        # ``_post_page`` — a tight, predictable ceiling.
        page_iter = iter(pages_list)
        in_flight: dict[concurrent.futures.Future, PageImage] = {}

        def _submit_one(ex: concurrent.futures.ThreadPoolExecutor) -> bool:
            try:
                p = next(page_iter)
            except StopIteration:
                return False
            fut = ex.submit(
                _post_page,
                page=p,
                profile=profile,
                proxy_url=proxy_url,
                prompt=prompt,
                sampling=sampling,
                timeout=timeout,
                num_retries=num_retries,
            )
            in_flight[fut] = p
            return True

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_requests
        ) as ex:
            for _ in range(concurrent_requests):
                if not _submit_one(ex):
                    break
            while in_flight:
                done, _pending = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    page = in_flight.pop(fut)
                    try:
                        raw = fut.result()
                    except Exception as e:  # noqa: BLE001
                        raw = RawOutput(
                            page_id=page.page_id,
                            output_format=profile.output_format,
                            payload="",
                            error=f"{type(e).__name__}: {e}",
                        )
                    completed += 1
                    if raw.error:
                        log.warning(
                            "%s page %d/%d %s FAIL: %s",
                            prefix, completed, total, page.file_id, raw.error,
                        )
                    else:
                        log.info(
                            "%s page %d/%d %s ok",
                            prefix, completed, total, page.file_id,
                        )
                    # Backfill the freed slot before yielding so the next
                    # POST starts encoding while the orchestrator processes
                    # this one — keeps the GPU saturated across the
                    # normalize + export tail of each page.
                    _submit_one(ex)
                    yield page, raw

        log.info(
            "%s batch of %d pages done in %.1fs",
            prefix, total, time.perf_counter() - t0,
        )


def _post_page(
    *,
    page: PageImage,
    profile: ModelProfile,
    proxy_url: str,
    prompt: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
    num_retries: int,
) -> RawOutput:
    """POST one page through LiteLLM and return ``RawOutput``.

    Cost / tokens / elapsed flow into ``cost.recorder`` via the registered
    success_callback — keyed by the ``page_id`` we stamp into ``metadata``.

    The decoded PIL image lives only inside ``page.open_image()`` — load,
    base64-encode for the message body, POST, release. Memory ceiling is
    one decoded RGB + one JPEG-base64 string per concurrent worker, instead
    of the chunk-wide buffer the previous design held.

    Retries are delegated to LiteLLM via ``num_retries`` — it classifies
    retryable exceptions (rate limits, 5xx, timeouts) and applies its own
    exponential backoff. We don't wrap with tenacity because doing so on
    top of ``num_retries`` would multiply attempts (``tenacity × litellm``)
    and we'd have to re-derive the "is this retryable" classification by
    hand.
    """
    top_level, extra_body = sampling
    try:
        with page.open_image() as img:
            page_prompt = _per_page_prompt(prompt, page)
            messages = _build_messages(img, page_prompt)
    except FileNotFoundError as e:
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error=f"image not found: {e}",
        )
    except Exception as e:  # noqa: BLE001
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error=f"image load failed: {type(e).__name__}: {e}",
        )

    cost_mod.recorder.open_page(page.page_id, profile.name)

    import litellm

    try:
        resp = litellm.completion(
            model=profile.name,
            custom_llm_provider="openai",
            api_base=proxy_url,
            api_key=os.environ.get("LITELLM_API_KEY", "ocrscout-dummy"),
            messages=messages,
            timeout=timeout,
            num_retries=num_retries,
            metadata={"page_id": page.page_id, "model_name": profile.name},
            extra_body=extra_body or None,
            **top_level,
        )
    except Exception as e:  # noqa: BLE001
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error=(
                f"{type(e).__name__}: {e} "
                f"(after {num_retries} retr{'y' if num_retries == 1 else 'ies'})"
            ),
        )
    finally:
        del messages

    text, tokens = _extract_completion(resp)
    return RawOutput(
        page_id=page.page_id,
        output_format=profile.output_format,
        payload=text,
        tokens=tokens,
    )


def _resolve_proxy_url(profile: ModelProfile) -> str:
    """Find the LiteLLM proxy URL or raise a clear error.

    ``OCRSCOUT_VLLM_URL`` is set by every Runner during ``launch()``; the
    ``profile.server_url`` fallback only matters for hand-managed proxy
    setups where ``ocrscout submit`` is invoked outside of a Runner.
    """
    url = os.environ.get("OCRSCOUT_VLLM_URL") or profile.server_url
    if not url:
        raise BackendError(
            f"LiteLLMBackend: no proxy URL for profile {profile.name!r}. "
            "Launch a runner (`ocrscout launch --models ...`) or set "
            "OCRSCOUT_VLLM_URL to a LiteLLM proxy endpoint."
        )
    return url.rstrip("/")


def _resolve_prompt_template(profile: ModelProfile) -> str:
    if not profile.prompt_templates:
        raise BackendError(
            f"LiteLLMBackend: profile {profile.name!r} has no prompt_templates"
        )
    mode = profile.preferred_prompt_mode or next(iter(profile.prompt_templates))
    if mode not in profile.prompt_templates:
        raise BackendError(
            f"LiteLLMBackend: preferred_prompt_mode={mode!r} not in "
            f"prompt_templates ({sorted(profile.prompt_templates)})"
        )
    return profile.prompt_templates[mode]


def _per_page_prompt(template: str, page: PageImage) -> str:
    if "{width}" in template or "{height}" in template:
        return template.replace("{width}", str(page.width)).replace(
            "{height}", str(page.height)
        )
    return template


def _build_messages(image: PILImage, prompt: str) -> list[dict[str, Any]]:
    """OpenAI-format chat message with a base64-encoded image.

    JPEG q80 instead of PNG: an archival scan re-encoded as PNG is 10–30 MB
    on a 3000×4000 page; JPEG q80 of the same content is ~0.5 MB, with no
    measurable accuracy delta on the OCR VLMs ocrscout drives. The smaller
    body shrinks both this process's in-flight footprint and the litellm
    proxy's per-request retention, both of which OOM-killed the orchestrator
    at concurrency=40 before this change.
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=80, optimize=False)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _split_sampling(
    sampling: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split sampling kwargs into top-level OpenAI fields and ``extra_body``.

    ``litellm.completion`` accepts the OpenAI core sampling fields as top-level
    kwargs; vLLM extensions (``top_k``, ``repetition_penalty``, ``min_p``, …)
    need to ride through ``extra_body`` so the proxy can forward them
    untouched.
    """
    top: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in sampling.items():
        if v is None:
            continue
        if k in _TOP_LEVEL_SAMPLING_KEYS:
            top[k] = v
        else:
            extra[k] = v
    return top, extra


def _extract_completion(resp: Any) -> tuple[str, int | None]:
    """Pull the assistant text and completion_tokens count off a litellm response."""
    try:
        choices = getattr(resp, "choices", None)
        if choices is None and isinstance(resp, dict):
            choices = resp.get("choices")
        if not choices:
            return "", None
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is None and isinstance(first, dict):
            msg = first.get("message")
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = content or ""
    except (AttributeError, KeyError, IndexError, TypeError):
        text = ""

    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    tokens: int | None = None
    if usage is not None:
        completion = getattr(usage, "completion_tokens", None)
        if completion is None and isinstance(usage, dict):
            completion = usage.get("completion_tokens")
        if completion is not None:
            try:
                tokens = int(completion)
            except (TypeError, ValueError):
                tokens = None
    return text, tokens

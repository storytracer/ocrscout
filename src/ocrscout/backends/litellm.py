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

_DEFAULT_REQUEST_TIMEOUT = 600.0
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

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        proxy_url = _resolve_proxy_url(profile)
        prompt = _resolve_prompt_template(profile)

        # Register the cost callback once per process; safe to call repeatedly.
        cost_mod.ensure_callback_registered()

        return BackendInvocation(
            kind="http",
            endpoint=proxy_url,
            profile=profile,
            pages=[p.page_id for p in pages],
            extra={
                "pages_runtime": list(pages),
                "prompt": prompt,
            },
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        profile = invocation.profile
        proxy_url = invocation.endpoint or ""
        if not proxy_url:
            raise BackendError("LiteLLMBackend.run: no proxy URL on invocation")

        pages: list[PageImage] = list(invocation.extra.get("pages_runtime", []))
        prompt: str = invocation.extra["prompt"]
        timeout = float(
            profile.backend_args.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT)
        )
        concurrent_requests = max(
            1, _resolve_concurrent_requests(profile),
        )
        sampling = _split_sampling(profile.sampling_args or {})
        prefix = f"[{profile.name}]"
        log.info(
            "%s starting %d page(s) against %s (%d-way concurrent)",
            prefix, len(pages), proxy_url, concurrent_requests,
        )

        results: dict[str, RawOutput] = {}
        completed = 0
        total = len(pages)
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_requests
        ) as ex:
            futures = {
                ex.submit(
                    _post_page,
                    page=page,
                    profile=profile,
                    proxy_url=proxy_url,
                    prompt=prompt,
                    sampling=sampling,
                    timeout=timeout,
                ): page
                for page in pages
            }
            for fut in concurrent.futures.as_completed(futures):
                page = futures[fut]
                try:
                    raw = fut.result()
                except Exception as e:  # noqa: BLE001
                    raw = RawOutput(
                        page_id=page.page_id,
                        output_format=profile.output_format,
                        payload="",
                        error=f"{type(e).__name__}: {e}",
                    )
                results[page.page_id] = raw
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

        log.info(
            "%s finished %d pages in %.1fs",
            prefix, total, time.perf_counter() - t0,
        )

        # Yield in input order so downstream record-writers see the same
        # ordering they emitted source pages in.
        for page in pages:
            yield results[page.page_id]


def _post_page(
    *,
    page: PageImage,
    profile: ModelProfile,
    proxy_url: str,
    prompt: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
) -> RawOutput:
    """POST one page through LiteLLM and return ``RawOutput``.

    Cost / tokens / elapsed flow into ``cost.recorder`` via the registered
    success_callback — keyed by the ``page_id`` we stamp into ``metadata``.
    """
    if page.image is None:
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error="page has no in-memory image",
        )
    top_level, extra_body = sampling
    page_prompt = _per_page_prompt(prompt, page)
    messages = _build_messages(page.image, page_prompt)

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
            metadata={"page_id": page.page_id, "model_name": profile.name},
            extra_body=extra_body or None,
            **top_level,
        )
    except Exception as e:  # noqa: BLE001
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error=f"{type(e).__name__}: {e}",
        )

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
    """OpenAI-format chat message with a base64-encoded image."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
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

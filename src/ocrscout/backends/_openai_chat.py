"""Shared helpers for OpenAI-compatible ``/chat/completions`` endpoints.

Used by ``VllmBackend`` (one POST per page) and ``LayoutChatBackend`` (one POST
per region). Centralises the multipart-image request body, the sampling-arg
forwarding allowlist, and the HTTP-error → useful-detail extraction.

Connection pooling is provided by a single ``requests.Session`` per backend
invocation; per-call timeouts are explicit.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import requests
from PIL.Image import Image as PILImage

from ocrscout.profile import ModelProfile

# vLLM's OpenAI server accepts the standard OpenAI sampling fields plus several
# vLLM extensions (top_k, repetition_penalty, min_p, ...) as top-level keys.
# Forward only this allowlist so a stray YAML key doesn't reach the wire.
_FORWARDED_SAMPLING_KEYS: tuple[str, ...] = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "min_p",
    "seed",
    "stop",
)


class ChatCompletionError(Exception):
    """Raised by :func:`post_chat_completion` on transport or parse failure.

    Callers wrap with whatever per-page or per-region attribution they need.
    """


def build_base_payload(profile: ModelProfile) -> dict[str, Any]:
    """Build the static fields of a chat-completions request.

    Call once per backend invocation; merge per-call ``messages`` on top via
    :func:`build_chat_request_body`.
    """
    base: dict[str, Any] = {"model": profile.model_id}
    sampling = profile.sampling_args or {}
    for key in _FORWARDED_SAMPLING_KEYS:
        if key in sampling:
            base[key] = sampling[key]
    if profile.chat_template_content_format is not None:
        base["chat_template_content_format"] = profile.chat_template_content_format
    return base


def build_chat_request_body(
    image: PILImage, prompt: str, base_payload: dict[str, Any]
) -> dict[str, Any]:
    """Compose a chat-completions request body for one image+prompt pair."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {
        **base_payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


def post_chat_completion(
    session: requests.Session,
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> tuple[str, int | None]:
    """POST ``body`` to ``url``, return ``(response_text, completion_tokens)``.

    Raises :class:`ChatCompletionError` on transport, HTTP, or parse failure
    with the response body included (vLLM returns useful detail under
    ``{"object":"error","message":"..."}`` on 4xx/5xx).
    """
    try:
        resp = session.post(url, json=body, timeout=timeout)
    except requests.RequestException as e:
        raise ChatCompletionError(f"{type(e).__name__}: {e}") from e
    if resp.status_code >= 400:
        body_text = resp.text or resp.reason or ""
        raise ChatCompletionError(f"HTTP {resp.status_code}: {body_text}")
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ChatCompletionError(f"{type(e).__name__} parsing response: {e}") from e
    usage = data.get("usage") or {}
    tokens = usage.get("completion_tokens")
    return (text or "", tokens)


def normalize_endpoint(endpoint: str) -> str:
    """Trim trailing slashes from a user-supplied base URL.

    The OpenAI convention is that the base URL *includes* ``/v1`` —
    ``http://host:port/v1``. We don't auto-prepend because some deployments
    live behind a proxy with a different prefix.
    """
    return endpoint.rstrip("/")


def list_models(
    session: requests.Session, endpoint: str, *, timeout: float
) -> list[str]:
    """GET ``{endpoint}/models`` and return the served model ids.

    Raises :class:`ChatCompletionError` on transport, HTTP, or parse failure.
    Use this to fail fast when a backend is pointed at the wrong server URL.
    """
    url = f"{endpoint.rstrip('/')}/models"
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise ChatCompletionError(f"GET {url}: {type(e).__name__}: {e}") from e
    if resp.status_code >= 400:
        body_text = resp.text or resp.reason or ""
        raise ChatCompletionError(f"GET {url}: HTTP {resp.status_code}: {body_text}")
    try:
        data = resp.json()
        return [entry["id"] for entry in data["data"]]
    except (ValueError, KeyError, TypeError) as e:
        raise ChatCompletionError(f"GET {url}: {type(e).__name__} parsing response: {e}") from e

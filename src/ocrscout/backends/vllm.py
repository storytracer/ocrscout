"""VllmBackend: run vLLM directly, either as a `uv run` subprocess or via HTTP.

Subprocess mode (default) spawns ``uv run --with "vllm{vllm_version}" --with
pillow vllm_runner.py manifest.json out.jsonl``. The runner loads the model,
processes a manifest of ``(page_id, image_path, prompt)`` items, and writes
per-page outputs to a JSONL. The parent never imports vLLM (preserves the
zero-GPU-deps-in-core rule).

Server mode is selected when ``profile.server_url`` is set or
``OCRSCOUT_VLLM_URL`` is in the environment. The backend POSTs to the
OpenAI-compatible ``/chat/completions`` endpoint of an externally-running
``vllm serve`` process.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Iterator, Sequence
from importlib.resources import files
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput

log = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 16
_DEFAULT_REQUEST_TIMEOUT = 300.0  # seconds, per HTTP page in server mode


class VllmBackend(ModelBackend):
    name = "vllm"

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        prompt = _resolve_prompt_template(profile)
        server_url = profile.server_url or os.environ.get("OCRSCOUT_VLLM_URL")

        if server_url:
            # Server mode does no preflight beyond keeping pages in memory; the
            # actual HTTP calls happen in run().
            return BackendInvocation(
                kind="http",
                endpoint=server_url.rstrip("/"),
                profile=profile,
                pages=[p.page_id for p in pages],
                extra={
                    "mode": "server",
                    "pages_runtime": list(pages),
                    "prompt": prompt,
                },
            )

        workdir = Path(tempfile.mkdtemp(prefix=f"ocrscout-vllm-{profile.name}-"))
        image_dir = workdir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        items: list[dict] = []
        for page in pages:
            if page.image is None:
                raise BackendError(
                    f"VllmBackend: page {page.page_id!r} has no in-memory image"
                )
            stem = _safe_stem(page.page_id)
            png_path = image_dir / f"{stem}.png"
            page.image.convert("RGB").save(png_path, format="PNG")
            items.append(
                {
                    "page_id": page.page_id,
                    "image_path": str(png_path),
                    "prompt": _per_page_prompt(prompt, page),
                }
            )

        manifest = {
            "model_id": profile.model_id,
            "vllm_engine_args": dict(profile.vllm_engine_args),
            "sampling_args": dict(profile.sampling_args),
            "chat_template_content_format": profile.chat_template_content_format,
            "batch_size": int(profile.backend_args.get("batch_size", _DEFAULT_BATCH_SIZE)),
            "items": items,
        }
        manifest_path = workdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output_jsonl = workdir / "outputs.jsonl"

        runner_path = _runner_path()
        version_spec = profile.vllm_version or ""
        argv = [
            "uv",
            "run",
            "--with",
            f"vllm{version_spec}",
            "--with",
            "pillow",
            str(runner_path),
            str(manifest_path),
            str(output_jsonl),
        ]

        return BackendInvocation(
            kind="subprocess",
            argv=argv,
            profile=profile,
            pages=[p.page_id for p in pages],
            extra={
                "mode": "subprocess",
                "workdir": str(workdir),
                "manifest_path": str(manifest_path),
                "output_jsonl": str(output_jsonl),
            },
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        mode = invocation.extra.get("mode")
        if mode == "subprocess":
            yield from self._run_subprocess(invocation)
        elif mode == "server":
            yield from self._run_server(invocation)
        else:
            raise BackendError(f"VllmBackend: unknown invocation mode {mode!r}")

    # --- subprocess mode ---------------------------------------------------

    def _run_subprocess(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        if invocation.argv is None:
            raise BackendError("VllmBackend.run: subprocess invocation has no argv")

        workdir = invocation.extra.get("workdir")
        output_jsonl = Path(invocation.extra["output_jsonl"])

        tail_buffer: deque[str] = deque(maxlen=400)
        try:
            print(f"  $ {' '.join(invocation.argv)}", flush=True)
            proc = subprocess.Popen(
                invocation.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ},
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tail_buffer.append(line)
            returncode = proc.wait()
            if returncode != 0:
                raise BackendError(
                    f"VllmBackend: runner exited {returncode}; last lines:\n"
                    + "".join(list(tail_buffer)[-100:])
                )

            yield from self._yield_jsonl(output_jsonl, invocation)
        finally:
            if workdir:
                # Keep the workdir on errors for debugging via --keep-workdir
                # later; for now, always clean up to match HfScriptsBackend.
                import shutil

                shutil.rmtree(workdir, ignore_errors=True)

    def _yield_jsonl(
        self, jsonl_path: Path, invocation: BackendInvocation
    ) -> Iterator[RawOutput]:
        if not jsonl_path.is_file():
            raise BackendError(
                f"VllmBackend: runner did not produce output file {jsonl_path}"
            )

        index: dict[str, dict] = {}
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("malformed JSONL line in runner output: %s", e)
                    continue
                page_id = record.get("page_id")
                if page_id is None:
                    continue
                index[page_id] = record

        for page_id in invocation.pages:
            record = index.get(page_id)
            if record is None:
                yield RawOutput(
                    page_id=page_id,
                    output_format=invocation.profile.output_format,
                    payload="",
                    error="page missing in runner output",
                )
                continue
            if "error" in record and record.get("error"):
                yield RawOutput(
                    page_id=page_id,
                    output_format=invocation.profile.output_format,
                    payload="",
                    error=str(record["error"]),
                )
                continue
            yield RawOutput(
                page_id=page_id,
                output_format=invocation.profile.output_format,
                payload=str(record.get("output", "")),
                tokens=record.get("tokens"),
            )

    # --- server mode -------------------------------------------------------

    def _run_server(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        endpoint = invocation.endpoint or ""
        if not endpoint:
            raise BackendError("VllmBackend: server mode requires endpoint URL")

        profile = invocation.profile
        prompt: str = invocation.extra["prompt"]
        pages: list[PageImage] = list(invocation.extra.get("pages_runtime", []))
        timeout = float(profile.backend_args.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT))

        sampling = dict(profile.sampling_args)
        # Map vLLM SamplingParams keys to the OpenAI chat/completions schema.
        # vLLM's OpenAI-compatible server accepts the OpenAI names (and ignores
        # unknown keys), so this is mostly a passthrough with a renamed token cap.
        request_payload_base: dict = {
            "model": profile.model_id,
        }
        if "max_tokens" in sampling:
            request_payload_base["max_tokens"] = sampling["max_tokens"]
        if "temperature" in sampling:
            request_payload_base["temperature"] = sampling["temperature"]
        if "top_p" in sampling:
            request_payload_base["top_p"] = sampling["top_p"]

        url = f"{endpoint}/chat/completions"
        for page in pages:
            page_prompt = _per_page_prompt(prompt, page)
            try:
                buf = io.BytesIO()
                page.image.convert("RGB").save(buf, format="PNG")
                data_uri = (
                    "data:image/png;base64,"
                    + base64.b64encode(buf.getvalue()).decode()
                )
                payload = {
                    **request_payload_base,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri}},
                                {"type": "text", "text": page_prompt},
                            ],
                        }
                    ],
                }
                req = urlrequest.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlrequest.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                tokens = usage.get("completion_tokens")
                yield RawOutput(
                    page_id=page.page_id,
                    output_format=profile.output_format,
                    payload=text or "",
                    tokens=tokens,
                )
            except (urlerror.URLError, json.JSONDecodeError, KeyError) as e:
                yield RawOutput(
                    page_id=page.page_id,
                    output_format=profile.output_format,
                    payload="",
                    error=f"{type(e).__name__}: {e}",
                )


def _resolve_prompt_template(profile: ModelProfile) -> str:
    if not profile.prompt_templates:
        raise BackendError(
            f"VllmBackend: profile {profile.name!r} has no prompt_templates; "
            "add at least one entry under prompt_templates: in the YAML."
        )
    mode = profile.preferred_prompt_mode
    if mode is None:
        # Fall back to the first key, deterministically.
        mode = next(iter(profile.prompt_templates))
    if mode not in profile.prompt_templates:
        raise BackendError(
            f"VllmBackend: preferred_prompt_mode={mode!r} not in "
            f"prompt_templates ({sorted(profile.prompt_templates)})"
        )
    return profile.prompt_templates[mode]


def _per_page_prompt(template: str, page: PageImage) -> str:
    """Substitute ``{width}`` and ``{height}`` if present.

    Per-page substitution (rather than once at prepare-time) keeps the runner
    free of page metadata it doesn't otherwise need.
    """
    if "{width}" in template or "{height}" in template:
        return template.replace("{width}", str(page.width)).replace(
            "{height}", str(page.height)
        )
    return template


def _safe_stem(page_id: str) -> str:
    return page_id.replace("/", "_").replace("\\", "_")


def _runner_path() -> Path:
    """Return the absolute path to ``vllm_runner.py``.

    The runner lives in ``ocrscout.runners`` (not ``ocrscout.backends``) so
    its parent directory — which Python prepends to ``sys.path`` when running
    the script — does not contain a sibling ``vllm.py`` that would shadow the
    real ``vllm`` package the runner imports.
    """
    anchor = files("ocrscout.runners") / "vllm_runner.py"
    return Path(str(anchor))

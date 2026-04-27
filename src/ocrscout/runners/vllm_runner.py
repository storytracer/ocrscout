"""Standalone runner: invoked inside a `uv run --with vllm...` subprocess.

The parent ``VllmBackend`` writes a manifest of ``(page_id, image_path, prompt)``
triples to disk, then spawns this script under ``uv run`` with vLLM and Pillow
as inline dependencies. Reading happens from the manifest path (argv[1]); the
JSONL of ``{page_id, output, error}`` rows is written to argv[2].

This file imports vLLM and is therefore not safe to import from the ocrscout
core package — it runs only inside the subprocess. The parent passes its
absolute path to ``uv run`` directly; no editable-install of ocrscout happens
in the subprocess venv.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow",
#   "vllm",
# ]
# ///

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ocrscout.vllm_runner")


def _make_chat_message(image_path: str, prompt: str) -> list[dict]:
    """Build a vLLM chat message with the image as a base64 data URL.

    Mirrors ``make_ocr_message`` in upstream HF scripts (e.g. dots-mocr.py:114)
    so behavior matches the scripts we are replacing.
    """
    pil = Image.open(image_path).convert("RGB")
    if "{width}" in prompt and "{height}" in prompt:
        prompt = prompt.replace("{width}", str(pil.width)).replace(
            "{height}", str(pil.height)
        )
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="JSON file with run config + items")
    parser.add_argument("output_jsonl", type=Path, help="Where to write per-page outputs")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_id: str = manifest["model_id"]
    engine_args: dict = manifest.get("vllm_engine_args", {}) or {}
    sampling_args: dict = manifest.get("sampling_args", {}) or {}
    chat_format: str | None = manifest.get("chat_template_content_format")
    items: list[dict] = manifest["items"]
    batch_size: int = int(manifest.get("batch_size", 16))

    log.info(
        "Loading vLLM: model=%s engine_args=%s sampling=%s",
        model_id,
        engine_args,
        sampling_args,
    )
    llm = LLM(model=model_id, **engine_args)
    sp = SamplingParams(**sampling_args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            t0 = time.perf_counter()
            try:
                messages = [_make_chat_message(it["image_path"], it["prompt"]) for it in batch]
            except Exception as e:  # noqa: BLE001
                # If we can't even build messages, mark every item in the batch as failed.
                log.exception("Failed to build messages for batch: %s", e)
                for it in batch:
                    out.write(json.dumps({"page_id": it["page_id"], "error": str(e)}) + "\n")
                continue

            chat_kwargs = {}
            if chat_format is not None:
                chat_kwargs["chat_template_content_format"] = chat_format

            try:
                outputs = llm.chat(messages, sp, **chat_kwargs)
            except Exception as e:  # noqa: BLE001
                log.exception("vLLM batch failed: %s", e)
                for it in batch:
                    out.write(json.dumps({"page_id": it["page_id"], "error": str(e)}) + "\n")
                continue

            for it, output in zip(batch, outputs, strict=True):
                try:
                    text = output.outputs[0].text
                    tokens = len(output.outputs[0].token_ids) if output.outputs else None
                    record = {"page_id": it["page_id"], "output": text, "tokens": tokens}
                except Exception as e:  # noqa: BLE001
                    record = {"page_id": it["page_id"], "error": f"output extraction failed: {e}"}
                out.write(json.dumps(record) + "\n")
                out.flush()

            elapsed = time.perf_counter() - t0
            log.info(
                "batch %d-%d/%d done in %.1fs",
                batch_start,
                batch_start + len(batch),
                len(items),
                elapsed,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

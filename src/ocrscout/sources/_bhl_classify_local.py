"""Local rights classifier for BHL refresh.

In-process, parquet in → parquet out. No HuggingFace Hub involvement;
called directly from
:meth:`ocrscout.sources.bhl.BhlSourceAdapter._run_classify_local` when
the user runs ``ocrscout source bhl refresh --runner local``.

Mirrors the prompt structure and output schema of
``uv-scripts/classification/classify-dataset.py`` (the HF-Jobs
classifier) so the local-runner and hf-runner outputs are
interchangeable downstream — same ``classification`` /
``parsing_success`` / ``reasoning`` columns, same model, same labels.

vLLM/torch/transformers are imported lazily so ``import ocrscout`` (and
every BHL admin verb except this one) stays GPU-agnostic. Install via
``pip install ocrscout[vllm]`` to enable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ocrscout.errors import ScoutError

log = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 4000


def _parse_label_descriptions(desc: str) -> dict[str, str]:
    """Parse ``label1:desc1;label2:desc2`` into a dict.

    Semicolon (not comma) separates pairs because BHL's prompt
    descriptions contain unescaped commas.
    """
    out: dict[str, str] = {}
    for pair in desc.split(";"):
        if ":" not in pair:
            continue
        label, body = pair.split(":", 1)
        out[label.strip()] = body.strip()
    return out


def _build_message(
    text: str,
    labels: list[str],
    descriptions: dict[str, str],
    enable_reasoning: bool,
) -> list[dict[str, str]]:
    if descriptions:
        categories = "Categories:\n" + "\n".join(
            f"- {label}: {descriptions.get(label, '')}" if descriptions.get(label)
            else f"- {label}"
            for label in labels
        )
    else:
        categories = f"Categories: {', '.join(labels)}"

    if enable_reasoning:
        user = (
            "Classify this text into one of these categories:\n\n"
            f"{categories}\n\nText: {text}\n\n"
            "Think through your classification step by step, then provide "
            'your final answer in this JSON format:\n{"label": "your_chosen_label"}'
        )
        system = "You are a helpful classification assistant that thinks step by step."
    else:
        user = (
            f"Classify this text as one of: {', '.join(labels)}\n\n"
            f"Text: {text}\n\nLabel:"
        )
        system = "You are a helpful classification assistant. /no_think"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_reasoning(output: str, valid_labels: list[str]) -> tuple[str | None, bool]:
    """Extract ``{"label": "..."}`` from a reasoning-mode generation."""
    think_end = output.find("</think>")
    json_part = output[think_end + len("</think>"):] if think_end != -1 else output
    start = json_part.find("{")
    if start == -1:
        return None, False
    depth = 0
    end = -1
    for i, ch in enumerate(json_part[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None, False
    try:
        data = json.loads(json_part[start:end])
    except json.JSONDecodeError:
        return None, False
    label = data.get("label")
    if label in valid_labels:
        return label, True
    return None, False


def classify_parquet(
    *,
    input_parquet: Path,
    output_parquet: Path,
    column: str,
    model: str,
    labels: list[str],
    label_descriptions: str = "",
    enable_reasoning: bool = True,
    max_tokens: int = 500,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.85,
) -> None:
    """Classify each row's ``column`` with vLLM and write a new parquet.

    Output schema = input schema + ``classification`` (string) +
    ``parsing_success`` (bool). When ``enable_reasoning`` is true a
    ``reasoning`` (string) column is also written.

    Raises :class:`ScoutError` if the GPU stack (``vllm`` / ``torch`` /
    ``transformers`` / ``pyarrow``) isn't importable — points at the
    ``ocrscout[vllm]`` extra.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ScoutError(
            "`refresh --runner local` needs vllm + transformers + torch + "
            "pyarrow in the active environment. Install via "
            "`pip install ocrscout[vllm]` (you control the torch / CUDA "
            f"variant). Underlying import error: {e}"
        ) from e

    descriptions = _parse_label_descriptions(label_descriptions)

    log.info("[bhl-classify] loading %s", input_parquet)
    table = pq.read_table(input_parquet)
    if column not in table.column_names:
        raise ScoutError(
            f"column {column!r} not in {input_parquet}; "
            f"available: {table.column_names}"
        )
    texts = [str(v) if v is not None else "" for v in table[column].to_pylist()]
    log.info("[bhl-classify] %d rows to classify", len(texts))

    log.info("[bhl-classify] loading tokenizer for %s", model)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    log.info(
        "[bhl-classify] initializing vLLM with %s (gpu_memory_utilization=%.2f)",
        model, gpu_memory_utilization,
    )
    try:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ""
        if "Free memory" in msg or "GPU memory utilization" in msg:
            hint = (
                " — VRAM headroom too tight. Free GPU memory by killing other "
                "CUDA processes, or lower the classifier's "
                "gpu_memory_utilization (currently "
                f"{gpu_memory_utilization:.2f}; try 0.80 / 0.75)."
            )
        raise ScoutError(f"vLLM init failed: {e}{hint}") from e

    if enable_reasoning:
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens * 20,
        )
    else:
        from vllm.sampling_params import GuidedDecodingParams
        sampling = SamplingParams(
            guided_decoding=GuidedDecodingParams(choice=labels),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    prompts: list[str] = []
    valid_indices: list[int] = []
    for idx, text in enumerate(texts):
        if not text or len(text) < MIN_TEXT_LENGTH:
            continue
        clipped = text[:MAX_TEXT_LENGTH]
        prompt = tokenizer.apply_chat_template(
            _build_message(clipped, labels, descriptions, enable_reasoning),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        valid_indices.append(idx)

    if not prompts:
        raise ScoutError(
            f"no rows in {input_parquet} passed the length check; "
            f"min length {MIN_TEXT_LENGTH}, column {column!r}."
        )

    log.info("[bhl-classify] generating %d completions", len(prompts))
    try:
        outputs = llm.generate(prompts, sampling)
    except Exception as e:  # noqa: BLE001
        raise ScoutError(f"vLLM generation failed: {e}") from e

    classifications: list[str | None] = [None] * len(texts)
    reasoning_col: list[str | None] = [None] * len(texts)
    parsing_success: list[bool] = [False] * len(texts)

    for k, output in enumerate(outputs):
        orig = valid_indices[k]
        raw = output.outputs[0].text.strip()
        if enable_reasoning:
            label, ok = _parse_reasoning(raw, labels)
            classifications[orig] = label
            reasoning_col[orig] = raw
            parsing_success[orig] = ok
        else:
            classifications[orig] = raw if raw in labels else None
            parsing_success[orig] = classifications[orig] is not None

    ok_count = sum(1 for v in parsing_success if v)
    log.info(
        "[bhl-classify] %d/%d rows parsed successfully (%.1f%%)",
        ok_count, len(texts), 100.0 * ok_count / max(len(texts), 1),
    )

    out_columns = {name: table[name] for name in table.column_names}
    out_columns["classification"] = pa.array(classifications, type=pa.string())
    out_columns["parsing_success"] = pa.array(parsing_success, type=pa.bool_())
    if enable_reasoning:
        out_columns["reasoning"] = pa.array(reasoning_col, type=pa.string())
    out_table = pa.table(out_columns)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    log.info("[bhl-classify] writing %s", output_parquet)
    pq.write_table(out_table, output_parquet, compression="zstd")

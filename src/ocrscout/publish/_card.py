"""Template renderers for ocrscout publish artefacts.

Frontmatter follows HuggingFace Hub conventions: rendered through
:class:`huggingface_hub.DatasetCardData` and :class:`huggingface_hub.SpaceCardData`
so the Hub viewer (Croissant metadata, dataset preview, "Used by" links)
recognises everything without bespoke parsing.

Four entry points, one per file the publisher writes:

* :func:`render_dataset_readme`
* :func:`render_space_readme`
* :func:`render_space_app_py`
* :func:`render_space_requirements`

No I/O. The publisher writes the returned strings to disk.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from huggingface_hub import DatasetCardData, SpaceCardData

from ocrscout.exports.layout import DATA_GLOB, SPLIT
from ocrscout.exports.schema import RESULTS_FEATURES

if TYPE_CHECKING:
    from ocrscout.publish._stats import ModelStats, PageDisagreement


# --------------------------------------------------------------------- dataset


def render_dataset_readme(
    *,
    repo_id: str,
    n_pages: int,
    n_models: int,
    n_rows: int,
    n_pages_with_errors: int,
    mean_disagreement: float | None,
    median_disagreement: float | None,
    per_model: list[ModelStats],
    top_disagreement: list[PageDisagreement],
    pipeline_yaml: str | None,
    has_image_column: bool,
    size_category: str,
    dataset_size_bytes: int | None,
    ocrscout_version: str,
    generated_at: datetime.datetime,
) -> str:
    """HuggingFace dataset card — frontmatter from ``DatasetCardData`` plus
    a Markdown body with per-model stats, top-disagreement pages, and the
    full ``pipeline.yaml``."""
    repo_short = repo_id.split("/")[-1]
    card_data = DatasetCardData(
        license="other",
        pretty_name=f"ocrscout · {repo_short}",
        size_categories=[size_category],
        task_categories=["image-to-text"],
        tags=["ocr", "ocrscout", "document-ai", "vlm-evaluation"],
        configs=[
            {
                "config_name": "default",
                "data_files": [
                    {"split": SPLIT, "path": DATA_GLOB},
                ],
            }
        ],
        dataset_info=_dataset_info_block(
            n_rows=n_rows,
            has_image_column=has_image_column,
            dataset_size_bytes=dataset_size_bytes,
        ),
    )
    body = _render_dataset_body(
        repo_id=repo_id,
        repo_short=repo_short,
        n_pages=n_pages,
        n_models=n_models,
        n_rows=n_rows,
        n_pages_with_errors=n_pages_with_errors,
        mean_disagreement=mean_disagreement,
        median_disagreement=median_disagreement,
        per_model=per_model,
        top_disagreement=top_disagreement,
        pipeline_yaml=pipeline_yaml,
        has_image_column=has_image_column,
        ocrscout_version=ocrscout_version,
        generated_at=generated_at,
    )
    return f"---\n{card_data.to_yaml()}\n---\n\n{body}"


def _dataset_info_block(
    *,
    n_rows: int,
    has_image_column: bool,
    dataset_size_bytes: int | None,
) -> dict[str, Any]:
    """Build the ``dataset_info`` YAML block from RESULTS_FEATURES, in the
    same shape ``Dataset.push_to_hub`` writes — so the Hub viewer (and any
    Croissant consumer) can read schema + sizes without downloading the
    parquet."""
    features: list[dict[str, Any]] = [
        {"name": name, "dtype": _feature_dtype_yaml(feat)}
        for name, feat in RESULTS_FEATURES.items()
    ]
    if has_image_column:
        features.insert(2, {"name": "image", "dtype": {"_type": "Image"}})
    splits: list[dict[str, Any]] = [{"name": SPLIT, "num_examples": n_rows}]
    if dataset_size_bytes is not None:
        splits[0]["num_bytes"] = int(dataset_size_bytes)
    info: dict[str, Any] = {"features": features, "splits": splits}
    if dataset_size_bytes is not None:
        info["download_size"] = int(dataset_size_bytes)
        info["dataset_size"] = int(dataset_size_bytes)
    return info


def _feature_dtype_yaml(feature: Any) -> Any:
    """``Value("string")`` → ``"string"``; complex features fall back to a
    ``{"_type": ...}`` mapping that mirrors what ``datasets`` writes."""
    cls = type(feature).__name__
    if cls == "Value":
        return getattr(feature, "dtype", "string")
    return {"_type": cls}


def _render_dataset_body(
    *,
    repo_id: str,
    repo_short: str,
    n_pages: int,
    n_models: int,
    n_rows: int,
    n_pages_with_errors: int,
    mean_disagreement: float | None,
    median_disagreement: float | None,
    per_model: list[ModelStats],
    top_disagreement: list[PageDisagreement],
    pipeline_yaml: str | None,
    has_image_column: bool,
    ocrscout_version: str,
    generated_at: datetime.datetime,
) -> str:
    summary = (
        f"# {repo_short}\n\n"
        "OCR-model comparison published with "
        "[ocrscout](https://github.com/storytracer/ocrscout). Each row pairs "
        "one source page with one model's normalized output.\n\n"
        "## Summary\n\n"
        f"- **Pages**: {n_pages}\n"
        f"- **Models**: {n_models}\n"
        f"- **Rows**: {n_rows}\n"
        f"- **Pages with errors**: {n_pages_with_errors}\n"
        f"- **Mean disagreement**: {_fmt_pct(mean_disagreement)}\n"
        f"- **Median disagreement**: {_fmt_pct(median_disagreement)}\n"
        f"- **Source images embedded**: "
        f"{'yes' if has_image_column else 'no (resolve via `source_uri`)'}\n"
        f"- **Generated**: {generated_at.strftime('%Y-%m-%d %H:%M UTC')} "
        f"by ocrscout v{ocrscout_version}\n"
    )
    return (
        summary
        + "\n"
        + _render_per_model_table(per_model)
        + "\n"
        + _render_top_disagreement_table(top_disagreement)
        + "\n"
        + _render_schema_section(has_image_column)
        + "\n"
        + _render_usage_section(repo_id)
        + "\n"
        + _render_pipeline_section(pipeline_yaml)
    )


def _render_per_model_table(models: list[ModelStats]) -> str:
    if not models:
        return "## Per-model metrics\n\n_(no rows)_\n"
    out = ["## Per-model metrics\n"]
    out.append(
        "| Model | Format | Pages OK | Pages errored | Total tokens "
        "| Mean tokens/pg | Mean s/pg | Mean prepare s | Mean chars | Mean items |"
    )
    out.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for m in models:
        out.append(
            "| "
            + " | ".join(
                [
                    _md_escape(m.model),
                    _md_escape(m.output_format or "—"),
                    str(m.pages_ok),
                    str(m.pages_errored),
                    str(m.total_tokens),
                    _fmt_num(m.mean_tokens),
                    _fmt_num(m.mean_run_seconds_per_page, 2),
                    _fmt_num(m.mean_prepare_seconds, 2),
                    _fmt_num(m.mean_text_length, 0),
                    _fmt_num(m.mean_item_count, 1),
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def _render_top_disagreement_table(pages: list[PageDisagreement]) -> str:
    if not pages:
        return (
            "## Top-disagreement pages\n\n"
            "_(only single-model pages — nothing to compare.)_\n"
        )
    out = ["## Top-disagreement pages\n"]
    out.append("Pairwise word-diff distance averaged across all models on the page. ")
    out.append("0 = perfect agreement; 1 = no shared tokens.\n")
    out.append("| Page | Models | Disagreement |")
    out.append("| --- | ---: | ---: |")
    for pd in pages:
        out.append(
            f"| {_md_escape(pd.page_id)} | {pd.n_models} | "
            f"{_fmt_pct(pd.disagreement)} |"
        )
    return "\n".join(out) + "\n"


def _render_pipeline_section(pipeline_yaml: str | None) -> str:
    if not pipeline_yaml:
        return "## Pipeline configuration\n\n_(no `pipeline.yaml` in the run.)_\n"
    return (
        "## Pipeline configuration\n\n"
        "The full ocrscout pipeline that produced this dataset:\n\n"
        "```yaml\n" + pipeline_yaml.rstrip() + "\n```\n"
    )


def _render_schema_section(has_image_column: bool) -> str:
    base = (
        "## Schema\n\n"
        "Each row in `data/train-*.parquet` is one `(page, model)` pair.\n\n"
        "| Column | Type | Notes |\n"
        "| --- | --- | --- |\n"
        "| `page_id` | string | Stable id from the source adapter. |\n"
        "| `model` | string | ocrscout profile name. |\n"
        "| `source_uri` | string | Original image path or URL (provenance). |\n"
    )
    if has_image_column:
        base += "| `image` | image | Source page bytes, embedded by ocrscout publish. |\n"
    base += (
        "| `output_format` | string | `markdown`, `doctags`, `layout_json`, … |\n"
        "| `document_json` | string | Full `DoclingDocument` JSON. |\n"
        "| `markdown` | string | `DoclingDocument.export_to_markdown()`. |\n"
        "| `raw_payload` | string | The model's pre-normalization output. |\n"
        "| `tokens` | int | Generated tokens (when reported). |\n"
        "| `error` | string | Set when the page failed for this model. |\n"
        "| `metrics_json` | string | Per-row JSON metrics (timings, sizes, …). |\n"
        "| `scores_json` | string | Reference-eval scores when an evaluator ran. |\n"
    )
    return base


def _render_usage_section(repo_id: str) -> str:
    return (
        "## Use this dataset\n\n"
        "**Load with `datasets`:**\n\n"
        "```python\n"
        "from datasets import load_dataset\n"
        f'ds = load_dataset("{repo_id}", split="train")\n'
        'print(ds[0]["page_id"], ds[0]["model"])\n'
        "```\n\n"
        "**Browse with the ocrscout viewer:**\n\n"
        "```bash\n"
        "pip install 'ocrscout[viewer] @ git+https://github.com/storytracer/ocrscout.git'\n"
        "python -c \"from huggingface_hub import snapshot_download; "
        f"print(snapshot_download('{repo_id}', repo_type='dataset', "
        "local_dir='./scout-snap'))\"\n"
        "ocrscout viewer ./scout-snap\n"
        "```\n"
    )


# ----------------------------------------------------------------------- space


def render_space_readme(
    *,
    repo_id: str,
    dataset_repo_id: str,
    dataset_revision: str | None,
    n_pages: int,
    n_models: int,
    ocrscout_version: str,
) -> str:
    """HuggingFace Space card — frontmatter via ``SpaceCardData`` so the
    "Used by" link from the dataset to this Space is automatic."""
    title_short = repo_id.split("/")[-1]
    card_data = SpaceCardData(
        title=f"ocrscout viewer — {title_short}",
        sdk="gradio",
        # Matches ocrscout's [viewer] extra floor. gradio 6 moved css/head
        # onto Blocks.launch (which our app.py uses) and dropped the
        # huggingface_hub<1 HfFolder import that breaks older 5.x.
        sdk_version="6.13.0",
        app_file="app.py",
        python_version="3.11",
        pinned=False,
        emoji="🔎",
        colorFrom="indigo",
        colorTo="blue",
        datasets=[dataset_repo_id],
        tags=["ocr", "ocrscout", "gradio", "document-ai"],
        license="other",
    )
    rev_note = f" @ `{dataset_revision}`" if dataset_revision else " (latest)"
    body = (
        f"# {title_short}\n\n"
        f"Side-by-side OCR comparison of **{n_pages} pages** across "
        f"**{n_models} models**, served by "
        "[ocrscout](https://github.com/storytracer/ocrscout)'s Gradio viewer.\n\n"
        f"Data: [`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id})"
        f"{rev_note}.\n\n"
        f"Built by `ocrscout publish space` (v{ocrscout_version}).\n"
    )
    return f"---\n{card_data.to_yaml()}\n---\n\n{body}"


def render_space_app_py(
    *, dataset_repo_id: str, dataset_revision: str | None
) -> str:
    rev_repr = "None" if dataset_revision is None else repr(dataset_revision)
    return (
        '"""Auto-generated by `ocrscout publish space`. '
        'Edit and re-publish to customise."""\n'
        "from pathlib import Path\n\n"
        "from huggingface_hub import snapshot_download\n\n"
        "from ocrscout.viewer.app import build_app\n\n"
        f'DATASET_REPO_ID = "{dataset_repo_id}"\n'
        f"DATASET_REVISION = {rev_repr}\n\n"
        "LOCAL = Path(\n"
        "    snapshot_download(\n"
        "        repo_id=DATASET_REPO_ID,\n"
        '        repo_type="dataset",\n'
        "        revision=DATASET_REVISION,\n"
        "    )\n"
        ")\n"
        "demo = build_app(LOCAL)\n\n"
        'if __name__ == "__main__":\n'
        "    launch_kwargs: dict = {}\n"
        '    css = getattr(demo, "ocrscout_css", None)\n'
        '    head = getattr(demo, "ocrscout_head", None)\n'
        "    if css is not None:\n"
        '        launch_kwargs["css"] = css\n'
        "    if head is not None:\n"
        '        launch_kwargs["head"] = head\n'
        "    demo.launch(**launch_kwargs)\n"
    )


def render_space_requirements(ocrscout_spec: str) -> str:
    return f"{ocrscout_spec}\nhuggingface_hub>=0.25.0\n"


# ------------------------------------------------------------------- helpers


def _fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")

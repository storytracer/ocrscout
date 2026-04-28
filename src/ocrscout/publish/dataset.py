"""Build and push an ocrscout run as a HuggingFace Hub dataset repo.

Every read / write / push goes through the standard ``datasets`` and
``huggingface_hub`` APIs — no manual parquet construction or layout
manipulation:

* :func:`build_dataset_for_publish` — loads the local parquet shards via
  ``datasets.load_dataset`` and (optionally) embeds source images as an
  ``Image()`` column.
* :func:`stage_dataset_locally` — for ``--dry-run``: persist the Dataset
  to ``staging`` via ``Dataset.to_parquet`` so users can inspect what will
  be pushed.
* :func:`push_dataset` — pushes via ``Dataset.push_to_hub`` (canonical HF
  upload: sharding, ``data/<split>-NNNNN-of-NNNNN.parquet`` naming, LFS,
  auto ``dataset_info`` block in README) and then a single follow-up
  ``HfApi.create_commit`` overlays the comprehensive dataset card and the
  run's ``pipeline.yaml``.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, Image, load_dataset
from huggingface_hub import CommitOperationAdd, HfApi
from pydantic import BaseModel, Field

from ocrscout import __version__ as OCRSCOUT_VERSION
from ocrscout.exports.layout import (
    find_parquet_files,
    parquet_data_files,
    parquet_dest,
)
from ocrscout.publish._card import render_dataset_readme
from ocrscout.publish._stats import (
    ModelStats,
    PageDisagreement,
    aggregate_per_model,
    overall_summary,
    top_disagreement_pages,
)
from ocrscout.sources.hf_dataset import read_path_or_url

log = logging.getLogger(__name__)


class DatasetManifest(BaseModel):
    """Summary of what was built — passed to the README renderer and to
    :func:`ocrscout.publish.space.build_space_repo` so the Space card can
    quote page/model counts without reloading the parquet."""

    n_pages: int
    n_models: int
    n_rows: int
    n_pages_with_errors: int
    mean_disagreement: float | None = None
    median_disagreement: float | None = None
    has_image_column: bool = False
    per_model: list[ModelStats] = Field(default_factory=list)
    top_disagreement: list[PageDisagreement] = Field(default_factory=list)
    pipeline_yaml: str | None = None
    dataset_size_bytes: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def size_category(self) -> str:
        """HF Hub `size_categories` bucket for ``n_rows``."""
        n = self.n_rows
        if n < 1_000:
            return "n<1K"
        if n < 10_000:
            return "1K<n<10K"
        if n < 100_000:
            return "10K<n<100K"
        if n < 1_000_000:
            return "100K<n<1M"
        if n < 10_000_000:
            return "1M<n<10M"
        return "n>10M"


def build_dataset_for_publish(
    output_dir: Path,
    *,
    bundle_images: bool,
) -> Dataset:
    """Load the local parquet shards into a ``Dataset`` ready for publishing.

    When ``bundle_images`` is set, each row's ``source_uri`` is fetched
    (via fsspec for remote URLs, plain read for local paths) and embedded
    as an ``Image()`` column so the published dataset is self-contained.
    """
    _require_parquet(output_dir)
    log.info("publish: loading %s", parquet_data_files(output_dir))
    ds = load_dataset(
        "parquet",
        data_files=parquet_data_files(output_dir),
        split="train",
    )
    if bundle_images:
        log.info("publish: fetching source images for %d rows", len(ds))
        ds = _attach_image_column(ds)
        ds = ds.cast_column("image", Image())
    return ds


def stage_dataset_locally(
    ds: Dataset,
    output_dir: Path,
    staging: Path,
) -> Path:
    """Write a dry-run preview of what will be pushed.

    Uses ``Dataset.to_parquet`` (standard ``datasets`` method) for the data,
    placed at the canonical HF location ``data/train-00000-of-00001.parquet``.
    Copies ``pipeline.yaml`` if present. The README is written separately
    by :func:`write_dataset_readme` since it depends on the manifest.
    Returns the path of the written parquet shard.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    parquet_out = parquet_dest(staging)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    log.info("publish: staging %s", parquet_out)
    ds.to_parquet(str(parquet_out))
    _copy_pipeline_yaml(output_dir, staging)
    return parquet_out


def build_manifest(
    ds: Dataset,
    *,
    bundle_images: bool,
    pipeline_yaml: str | None,
    dataset_size_bytes: int | None,
) -> DatasetManifest:
    """Aggregate per-model and per-page statistics from an in-memory Dataset.

    The Dataset rows are projected to dicts with ``select_columns`` (skipping
    the ``image`` column to avoid decoding bytes) and parsed metrics are
    attached so :mod:`ocrscout.publish._stats` helpers can run unmodified.
    """
    summary = overall_summary(_iter_rows_for_stats(ds))
    rows_for_stats = _iter_rows_for_stats(ds)
    return DatasetManifest(
        n_pages=summary["n_pages"],
        n_models=summary["n_models"],
        n_rows=summary["n_rows"],
        n_pages_with_errors=summary["n_pages_with_errors"],
        mean_disagreement=summary["mean_disagreement"],
        median_disagreement=summary["median_disagreement"],
        has_image_column=bundle_images,
        per_model=aggregate_per_model(rows_for_stats),
        top_disagreement=top_disagreement_pages(rows_for_stats, k=10),
        pipeline_yaml=pipeline_yaml,
        dataset_size_bytes=dataset_size_bytes,
    )


def write_dataset_readme(
    staging: Path,
    *,
    repo_id: str,
    manifest: DatasetManifest,
) -> None:
    """Render and write ``staging/README.md`` from the manifest. Called by
    the dry-run path (and indirectly by :func:`push_dataset` to produce the
    bytes uploaded in the follow-up commit)."""
    (staging / "README.md").write_text(_render_readme(repo_id, manifest), encoding="utf-8")


def push_dataset(
    ds: Dataset,
    *,
    output_dir: Path,
    manifest: DatasetManifest,
    repo_id: str,
    private: bool,
    revision: str,
    commit_message: str,
    token: str | None,
) -> str:
    """Push the dataset to HF Hub via the standard ``datasets`` library, then
    overlay our enriched dataset card and ``pipeline.yaml`` in a single
    follow-up commit.

    ``Dataset.push_to_hub`` does the canonical heavy lifting: creates the
    repo if missing, shards the parquet, places it under
    ``data/<split>-NNNNN-of-NNNNN.parquet``, generates a README with
    ``configs.data_files`` and ``dataset_info`` frontmatter, and uses LFS
    where appropriate. We then run a single ``HfApi.create_commit`` that
    overwrites the README with our richer body (per-model stats, top
    disagreement pages, pipeline config) and adds ``pipeline.yaml``.
    """
    log.info("publish: Dataset.push_to_hub → %s", repo_id)
    info = ds.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=commit_message,
        revision=revision,
    )

    operations: list[CommitOperationAdd] = [
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=_render_readme(repo_id, manifest).encode("utf-8"),
        ),
    ]
    pipeline_path = output_dir / "pipeline.yaml"
    if pipeline_path.is_file():
        operations.append(
            CommitOperationAdd(
                path_in_repo="pipeline.yaml",
                path_or_fileobj=str(pipeline_path),
            )
        )

    api = HfApi(token=token)
    log.info("publish: enriching dataset card on %s", repo_id)
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        operations=operations,
        commit_message="ocrscout publish: dataset card + pipeline.yaml",
    )
    return getattr(info, "commit_url", str(info))


# ----------------------------------------------------------------- helpers


def _render_readme(repo_id: str, manifest: DatasetManifest) -> str:
    return render_dataset_readme(
        repo_id=repo_id,
        n_pages=manifest.n_pages,
        n_models=manifest.n_models,
        n_rows=manifest.n_rows,
        n_pages_with_errors=manifest.n_pages_with_errors,
        mean_disagreement=manifest.mean_disagreement,
        median_disagreement=manifest.median_disagreement,
        per_model=manifest.per_model,
        top_disagreement=manifest.top_disagreement,
        pipeline_yaml=manifest.pipeline_yaml,
        has_image_column=manifest.has_image_column,
        size_category=manifest.size_category,
        dataset_size_bytes=manifest.dataset_size_bytes,
        ocrscout_version=OCRSCOUT_VERSION,
        generated_at=datetime.datetime.now(datetime.UTC),
    )


def _require_parquet(output_dir: Path) -> None:
    if not find_parquet_files(output_dir):
        raise FileNotFoundError(
            f"No data/train-*.parquet under {output_dir}. Pass an --output-dir "
            "produced by `ocrscout run`."
        )


def _attach_image_column(ds: Dataset) -> Dataset:
    """Fetch each row's image from ``source_uri`` and add an ``image`` column.

    Bytes are deduped within the dataset — multiple model rows for the same
    page share a single fetch. fsspec handles ``s3://``, ``gs://``,
    ``https://``, ``hf://`` and plain local paths via
    :func:`read_path_or_url`.
    """
    cache: dict[str, bytes | None] = {}

    def fetch(uri: str | None) -> bytes | None:
        if not uri:
            return None
        if uri in cache:
            return cache[uri]
        try:
            data = read_path_or_url(uri)
        except (OSError, ValueError, FileNotFoundError) as e:
            log.warning("publish: cannot fetch %s: %s", uri, e)
            data = None
        cache[uri] = data
        return data

    def add_image(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": [
                ({"bytes": b, "path": None} if b else None)
                for b in (fetch(uri) for uri in batch["source_uri"])
            ]
        }

    return ds.map(add_image, batched=True, batch_size=32)


def _copy_pipeline_yaml(output_dir: Path, staging: Path) -> str | None:
    """Copy ``pipeline.yaml`` into the staging dir; return its text content."""
    src = output_dir / "pipeline.yaml"
    if not src.is_file():
        return None
    dst = staging / "pipeline.yaml"
    shutil.copy2(src, dst)
    try:
        return src.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_pipeline_yaml(output_dir: Path) -> str | None:
    """Read ``pipeline.yaml`` without staging it (for the live push path)."""
    path = output_dir / "pipeline.yaml"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _iter_rows_for_stats(ds: Dataset) -> list[dict[str, Any]]:
    """Materialise the lightweight dict view stats helpers expect.

    Keeps only the columns the aggregators read — explicitly skips ``image``
    (which would otherwise decode a PIL.Image per row).
    """
    keep = {
        "page_id",
        "model",
        "output_format",
        "tokens",
        "error",
        "metrics_json",
        "markdown",
    }
    cols = [c for c in ds.column_names if c in keep]
    out: list[dict[str, Any]] = []
    for row in ds.select_columns(cols):
        metrics_raw = row.get("metrics_json") or ""
        try:
            metrics = json.loads(metrics_raw) if metrics_raw else {}
        except json.JSONDecodeError:
            metrics = {}
        out.append({**row, "metrics": metrics})
    return out

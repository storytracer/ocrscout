"""`ocrscout publish` — push a run to HuggingFace Hub.

Two subcommands:

* ``ocrscout publish dataset <output_dir> --repo-id org/name``
    Pushes a dataset repo containing the run's parquet (with optionally
    embedded source images), ``pipeline.yaml``, and a comprehensive README.
* ``ocrscout publish space <output_dir> --repo-id user/space --dataset-repo-id ...``
    Pushes a Gradio Space whose ``app.py`` runs the ocrscout viewer over a
    previously-published dataset repo.

Both commands accept ``--dry-run`` to write the staging directory under
``<output_dir>/.ocrscout-publish/<dataset|space>/`` without any network
call.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from rich import print as rprint

from ocrscout.cli import app
from ocrscout.log import setup_logging

log = logging.getLogger(__name__)


_DEFAULT_OCRSCOUT_SPEC = (
    "ocrscout[viewer] @ git+https://github.com/storytracer/ocrscout.git@main"
)


publish_app = typer.Typer(
    name="publish",
    help="Push an ocrscout run to HuggingFace Hub.",
    no_args_is_help=True,
)
app.add_typer(publish_app, name="publish")


@publish_app.command("dataset")
def publish_dataset(
    output_dir: Path = typer.Argument(
        ...,
        help="A previous run's --output-dir (must contain data/train-*.parquet).",
    ),
    repo_id: str = typer.Option(
        ..., "--repo-id",
        help="Target dataset repo id, e.g. `org/ocrscout-bhl`.",
    ),
    private: bool = typer.Option(
        False, "--private/--public",
        help="Create the dataset repo as private.",
    ),
    revision: str = typer.Option(
        "main", "--revision",
        help="Branch on the dataset repo to push to.",
    ),
    commit_message: str | None = typer.Option(
        None, "--commit-message",
        help="Override the auto-generated commit message.",
    ),
    bundle_images: bool = typer.Option(
        True, "--bundle-images/--no-bundle-images",
        help="Embed source images as an `image` column. Off makes the "
             "published parquet rely on `source_uri` resolving from outside.",
    ),
    token: str | None = typer.Option(
        None, "--token",
        help="HF token. Defaults to $HF_TOKEN / `huggingface-cli login`.",
        envvar="HF_TOKEN",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Build the staging directory under <output_dir>/.ocrscout-publish/dataset/ "
             "and stop. No network calls.",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True,
        help="Increase log verbosity.",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet",
        help="Suppress informational logging.",
    ),
) -> None:
    """Publish ``<output_dir>`` as a HuggingFace Hub dataset."""
    setup_logging(verbosity=verbose, quiet=quiet)

    from ocrscout.publish.dataset import (
        _read_pipeline_yaml,
        build_dataset_for_publish,
        build_manifest,
        push_dataset,
        stage_dataset_locally,
        write_dataset_readme,
    )

    ds = build_dataset_for_publish(output_dir, bundle_images=bundle_images)

    if dry_run:
        staging = output_dir / ".ocrscout-publish" / "dataset"
        log.info("publish dataset: staging at %s", staging)
        parquet_out = stage_dataset_locally(ds, output_dir, staging)
        manifest = build_manifest(
            ds,
            bundle_images=bundle_images,
            pipeline_yaml=_read_pipeline_yaml(output_dir),
            dataset_size_bytes=parquet_out.stat().st_size,
        )
        write_dataset_readme(staging=staging, repo_id=repo_id, manifest=manifest)
        rprint(f"[green]Staging written to[/green] {staging}")
        rprint(
            f"[dim]Pages:[/dim] {manifest.n_pages}  "
            f"[dim]Models:[/dim] {manifest.n_models}  "
            f"[dim]Bundled images:[/dim] {manifest.has_image_column}"
        )
        rprint(
            "[yellow]Dry run — no network call. Drop --dry-run to push to "
            f"{repo_id}.[/yellow]"
        )
        return

    if not _has_token(token):
        rprint(
            "[red]No HF token found. Pass --token or set HF_TOKEN, or run "
            "`huggingface-cli login`.[/red]"
        )
        raise typer.Exit(code=1)

    manifest = build_manifest(
        ds,
        bundle_images=bundle_images,
        pipeline_yaml=_read_pipeline_yaml(output_dir),
        dataset_size_bytes=None,
    )
    msg = commit_message or (
        f"ocrscout publish: {manifest.n_pages} pages × {manifest.n_models} models"
    )
    url = push_dataset(
        ds,
        output_dir=output_dir,
        manifest=manifest,
        repo_id=repo_id,
        private=private,
        revision=revision,
        commit_message=msg,
        token=token,
    )
    rprint(f"[green]Pushed:[/green] {url}")


@publish_app.command("space")
def publish_space(
    output_dir: Path = typer.Argument(
        ...,
        help="A previous run's --output-dir (used for sanity checks; the "
             "Space's runtime data comes from the dataset repo).",
    ),
    repo_id: str = typer.Option(
        ..., "--repo-id",
        help="Target Space repo id, e.g. `user/ocrscout-bhl-viewer`.",
    ),
    dataset_repo_id: str = typer.Option(
        ..., "--dataset-repo-id",
        help="The dataset repo previously pushed via `publish dataset`.",
    ),
    dataset_revision: str | None = typer.Option(
        None, "--dataset-revision",
        help="Pin the dataset to a specific commit/tag.",
    ),
    ocrscout_spec: str = typer.Option(
        _DEFAULT_OCRSCOUT_SPEC, "--ocrscout-spec",
        help="Pip spec used in requirements.txt for the Space. "
             "Override with e.g. `ocrscout[viewer]==X` once on PyPI.",
    ),
    private: bool = typer.Option(
        False, "--private/--public",
        help="Create the Space as private.",
    ),
    hardware: str = typer.Option(
        "cpu-basic", "--hardware",
        help="HF Spaces hardware tier (cpu-basic, cpu-upgrade, t4-small, …).",
    ),
    token: str | None = typer.Option(
        None, "--token",
        help="HF token. Defaults to $HF_TOKEN / `huggingface-cli login`.",
        envvar="HF_TOKEN",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Build the staging directory under <output_dir>/.ocrscout-publish/space/ "
             "and stop. No network calls.",
    ),
    verbose: int = typer.Option(
        0, "-v", "--verbose", count=True,
        help="Increase log verbosity.",
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet",
        help="Suppress informational logging.",
    ),
) -> None:
    """Publish a Gradio Space that views ``--dataset-repo-id``."""
    setup_logging(verbosity=verbose, quiet=quiet)

    from ocrscout.exports.layout import find_parquet_files
    from ocrscout.publish.dataset import (
        _read_pipeline_yaml,
        build_dataset_for_publish,
        build_manifest,
    )
    from ocrscout.publish.space import build_space_repo, push_space

    if not find_parquet_files(output_dir):
        rprint(f"[red]No data/train-*.parquet under {output_dir}[/red]")
        raise typer.Exit(code=1)

    # Manifest only (no image bundling — Space card just needs counts).
    ds = build_dataset_for_publish(output_dir, bundle_images=False)
    manifest = build_manifest(
        ds,
        bundle_images=False,
        pipeline_yaml=_read_pipeline_yaml(output_dir),
        dataset_size_bytes=None,
    )

    staging = output_dir / ".ocrscout-publish" / "space"
    log.info("publish space: staging at %s", staging)
    build_space_repo(
        staging=staging,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        ocrscout_spec=ocrscout_spec,
        repo_id=repo_id,
        manifest=manifest,
    )

    if dry_run:
        rprint(f"[green]Staging written to[/green] {staging}")
        rprint(
            f"[dim]Pages:[/dim] {manifest.n_pages}  "
            f"[dim]Models:[/dim] {manifest.n_models}  "
            f"[dim]Dataset:[/dim] {dataset_repo_id}"
        )
        rprint(
            "[yellow]Dry run — no network call. Drop --dry-run to push to "
            f"{repo_id}.[/yellow]"
        )
        return

    if not _has_token(token):
        rprint(
            "[red]No HF token found. Pass --token or set HF_TOKEN, or run "
            "`huggingface-cli login`.[/red]"
        )
        raise typer.Exit(code=1)

    url = push_space(
        staging=staging,
        repo_id=repo_id,
        private=private,
        hardware=hardware,
        token=token,
    )
    rprint(f"[green]Pushed:[/green] {url}")


def _has_token(token: str | None) -> bool:
    if token:
        return True
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder

        return bool(HfFolder.get_token())
    except Exception:  # noqa: BLE001
        return False

"""Build and push a Gradio Space that wraps an ocrscout viewer over a
previously-published HuggingFace dataset.

Two phases mirror :mod:`ocrscout.publish.dataset` — :func:`build_space_repo`
materialises ``app.py`` / ``requirements.txt`` / ``README.md`` into a
staging directory, and :func:`push_space` creates the repo and uploads the
folder.

The generated Space's ``app.py`` does ``snapshot_download(<dataset>)`` on
boot and feeds the local directory to :func:`ocrscout.viewer.app.build_app`,
so the Space code is small and the heavy data lives in the dataset repo.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ocrscout import __version__ as OCRSCOUT_VERSION
from ocrscout.publish._card import (
    render_space_app_py,
    render_space_readme,
    render_space_requirements,
)
from ocrscout.publish.dataset import DatasetManifest

log = logging.getLogger(__name__)


def build_space_repo(
    staging: Path,
    *,
    dataset_repo_id: str,
    dataset_revision: str | None,
    ocrscout_spec: str,
    repo_id: str,
    manifest: DatasetManifest,
) -> None:
    """Materialise app.py / requirements.txt / README.md under ``staging``."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / "app.py").write_text(
        render_space_app_py(
            dataset_repo_id=dataset_repo_id,
            dataset_revision=dataset_revision,
        ),
        encoding="utf-8",
    )
    (staging / "requirements.txt").write_text(
        render_space_requirements(ocrscout_spec),
        encoding="utf-8",
    )
    (staging / "README.md").write_text(
        render_space_readme(
            repo_id=repo_id,
            dataset_repo_id=dataset_repo_id,
            dataset_revision=dataset_revision,
            n_pages=manifest.n_pages,
            n_models=manifest.n_models,
            ocrscout_version=OCRSCOUT_VERSION,
        ),
        encoding="utf-8",
    )


def push_space(
    staging: Path,
    repo_id: str,
    *,
    private: bool,
    hardware: str,
    token: str | None,
) -> str:
    """Create the Space repo (idempotent) and upload everything in
    ``staging`` in one commit. Returns the commit URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    log.info("publish: ensuring space repo %s exists", repo_id)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        private=private,
        space_sdk="gradio",
        space_hardware=hardware,
        exist_ok=True,
    )
    log.info("publish: uploading %s → %s (space)", staging, repo_id)
    basename = repo_id.split("/")[-1]
    commit = api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"ocrscout publish space (v{OCRSCOUT_VERSION}): {basename}",
    )
    return getattr(commit, "commit_url", str(commit))

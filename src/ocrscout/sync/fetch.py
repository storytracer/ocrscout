"""Fetch the upstream uv-scripts/ocr scripts via huggingface_hub."""

from __future__ import annotations

from pathlib import Path

from ocrscout.errors import ScoutError
from ocrscout.sync.cache import scripts_cache_dir

UPSTREAM_REPO = "uv-scripts/ocr"
UPSTREAM_REPO_TYPE = "dataset"


def fetch_scripts(*, target_dir: Path | None = None, revision: str | None = None) -> Path:
    """Snapshot-download the upstream scripts and return the local directory.

    Returns the directory containing the ``.py`` script files. Network access
    required.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - core dep, should always exist
        raise ScoutError(f"huggingface_hub is required for sync: {e}") from e

    dest = target_dir if target_dir is not None else scripts_cache_dir()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    local = snapshot_download(
        repo_id=UPSTREAM_REPO,
        repo_type=UPSTREAM_REPO_TYPE,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=["*.py"],
    )
    return Path(local)

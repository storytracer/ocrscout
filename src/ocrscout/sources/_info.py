"""``info.yaml`` schema and atomic-merge helpers for source admin.

Each source has a per-instance state file at
``~/.ocrscout/sources/<name>/info.yaml`` that records *provisioning*:
cache freshness markers (ETags, refresh timestamps), pointers to derived
HF datasets, and the shape of any pre-built intermediate artifacts. This
is the file ``ocrscout source <name> info`` reads to render its summary
and the file every ``SourceAction.run()`` may patch via the
:func:`merge_info` helper.

Schema conventions:

* Top-level uses ``extra="ignore"`` so an older ocrscout reading a newer
  ``info.yaml`` (or vice versa) survives unknown sections.
* Each section sub-model uses ``extra="forbid"`` so typos within a known
  section (``last_refersh``) fail loudly.
* Truly breaking schema changes bump ``schema_version`` + add a migrator
  here keyed on the integer version.

All writes go through :func:`merge_info`, which serializes a read →
deep-merge → re-validate → atomic write cycle behind an ``fcntl.flock``
on the info file. The atomic-write primitive itself is shared with
:mod:`ocrscout.state` so any future audit only has to verify one
implementation.
"""

from __future__ import annotations

import fcntl
from datetime import datetime
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocrscout.errors import StateError
from ocrscout.sources._paths import info_path
from ocrscout.state import _atomic_write_yaml


class CatalogSection(BaseModel):
    """Raw upstream cache state (TSVs, manifests, …)."""

    model_config = ConfigDict(extra="forbid")
    last_refresh: datetime | None = None
    files: dict[str, str] = Field(default_factory=dict)
    """Basename → ETag, e.g. ``{"item.txt.gz": "abc123"}``."""


class RightsSection(BaseModel):
    """Rights / classification pipeline state.

    Two refresh modes share this section:

    * ``--runner local`` runs the classifier on this machine and writes
      its output to ``local_parquet`` (a path under ``derived/``). No HF
      Hub round-trip; ``source_repo`` / ``output_repo`` / ``read_from``
      stay ``None``.
    * ``--runner hf`` runs the classifier as a HuggingFace Job that
      reads ``source_repo`` and writes ``output_repo``; ``read_from`` is
      what the adapter reads at runtime (usually equal to ``output_repo``,
      but ``setup --read-from`` can point at a third-party dataset).

    ``_build_volumes_parquet`` resolves the rights source as
    ``local_parquet`` if present, else falls back to fetching
    ``read_from`` from the Hub.
    """

    model_config = ConfigDict(extra="forbid")
    source_repo: str | None = None
    output_repo: str | None = None
    read_from: str | None = None
    local_parquet: str | None = None
    """Path to a locally-classified rights parquet (``--runner local``)."""
    last_refresh: datetime | None = None
    last_runner: str | None = None  # "local" | "hf"
    combos_classified: int | None = None


class DerivedSection(BaseModel):
    """Intermediate artifacts produced by refresh (pre-join parquets, …)."""

    model_config = ConfigDict(extra="forbid")
    volumes_parquet_mtime: datetime | None = None
    volumes_parquet_rows: int | None = None


class SourceInfo(BaseModel):
    """Root schema for ``~/.ocrscout/sources/<name>/info.yaml``."""

    model_config = ConfigDict(extra="ignore")
    schema_version: int = 1
    source_name: str
    created_at: datetime
    catalog: CatalogSection = Field(default_factory=CatalogSection)
    rights: RightsSection = Field(default_factory=RightsSection)
    derived: DerivedSection = Field(default_factory=DerivedSection)


def load_info(source_name: str) -> SourceInfo | None:
    """Read ``info.yaml`` for ``source_name``, or ``None`` if absent/empty."""
    path = info_path(source_name)
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise StateError(f"cannot read {path}: {e}") from e
    if data is None:
        # Empty or whitespace-only file — treated as "no info" so a
        # stub-touched lock file from merge_info doesn't blow up.
        return None
    if not isinstance(data, dict):
        raise StateError(f"{path} must contain a YAML mapping")
    try:
        return SourceInfo.model_validate(data)
    except ValidationError as e:
        raise StateError(f"invalid info.yaml at {path}: {e}") from e


def dump_info(info: SourceInfo) -> None:
    """Atomically write ``info`` to its on-disk location."""
    _atomic_write_yaml(info_path(info.source_name), info.model_dump(mode="json"))


def merge_info(source_name: str, patch: dict[str, dict[str, Any]]) -> SourceInfo:
    """Read → deep-merge ``patch`` section-by-section → re-validate → write.

    ``patch`` is shaped ``{section_name: {key: value, ...}, ...}`` — the
    return value of every :meth:`SourceAction.run`. Sections not present
    in ``patch`` are preserved verbatim; sections present are shallow-
    merged into their existing values (a key in ``patch`` overrides the
    same key in the stored section, but unmentioned keys survive).

    Held under an ``fcntl.flock`` on the info file so concurrent
    ``ocrscout source <name> <verb>`` invocations can't tear the merge.
    Creates the source directory + a baseline ``SourceInfo`` on first
    write.
    """
    path = info_path(source_name)
    # Ensure the source directory exists so the lock file has a home.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock on a sibling .lock file rather than info.yaml itself: the
    # data file gets replaced by _atomic_write_yaml below (os.replace
    # swaps the inode), which would otherwise leave concurrent
    # invocations holding a lock on a dead inode.
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            current = load_info(source_name)
            if current is None:
                current = SourceInfo(
                    source_name=source_name,
                    created_at=datetime.now().astimezone(),
                )
            data = current.model_dump(mode="json")
            for section, updates in patch.items():
                if section not in data or not isinstance(data[section], dict):
                    data[section] = {}
                data[section].update(updates)
            try:
                merged = SourceInfo.model_validate(data)
            except ValidationError as e:
                raise StateError(
                    f"merged info.yaml for {source_name!r} failed validation: {e}"
                ) from e
            _atomic_write_yaml(path, merged.model_dump(mode="json"))
            return merged
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def clear_info(source_name: str) -> None:
    """Remove the info file. Idempotent."""
    try:
        info_path(source_name).unlink()
    except FileNotFoundError:
        pass

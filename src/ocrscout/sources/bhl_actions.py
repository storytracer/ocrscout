"""BHL source's admin verbs.

Each verb is a :class:`~ocrscout.interfaces.source_action.SourceAction`
subclass listed in :class:`~ocrscout.sources.bhl.BhlSourceAdapter.actions`.
The CLI driver in :mod:`ocrscout.cli.source` walks that list to register
``ocrscout source bhl <verb>`` commands.

Currently shipped:

* :class:`BhlSetupAction` — record where to read rights from (the
  lightweight first-run path; no upstream sync).

Future PRs will add ``refresh``, ``stats``, ``rights-stats``, ``sample``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source_action import SourceAction, SourceActionContext

log = logging.getLogger(__name__)


class BhlSetupFlags(BaseModel):
    """Flags for ``ocrscout source bhl setup``."""

    read_from: str = Field(
        ...,
        description=(
            "HF dataset id to read the rights classification from at OCR "
            "time, e.g. storytracer/bhl_rights_json_classified for the "
            "canonical public dataset."
        ),
    )
    output_repo: str | None = Field(
        None,
        description=(
            "Where future `refresh` invocations will publish your own "
            "classified version. Recorded for convenience; can be omitted "
            "if you only consume an existing dataset."
        ),
    )


class BhlSetupAction(SourceAction):
    """Record read pointer and (optionally) output repo without running classify.

    The lightweight setup path: users who just want to OCR BHL pages
    against an *existing* rights-classified dataset (canonical or
    third-party) point ``--read-from`` at it and skip the heavy
    extract/classify/publish pipeline that ``refresh`` runs. The
    next ``ocrscout run --source bhl …`` invocation will use this
    dataset.
    """

    name = "setup"
    flags = BhlSetupFlags
    description = "Record where to read rights classification from."

    def fill_defaults(self, flags: BhlSetupFlags, info: Any) -> BhlSetupFlags:
        # output_repo is sticky: if a prior setup or refresh recorded one,
        # don't make the user repeat it. read_from is required, so no
        # defaulting there.
        if flags.output_repo is None and info.rights.output_repo:
            flags.output_repo = info.rights.output_repo
        return flags

    def run(
        self,
        flags: BhlSetupFlags,
        ctx: SourceActionContext,
    ) -> dict[str, dict] | None:
        # Lightweight validation: fail fast on a typo'd repo before
        # writing anything. Falls back silently on transient network
        # errors so a flaky connection doesn't block setup.
        _verify_dataset_reachable(flags.read_from)

        log.info(
            "[bhl] setup: read_from=%s, output_repo=%s",
            flags.read_from,
            flags.output_repo or "—",
        )
        # Setup only records pointers. It deliberately does NOT touch
        # rights.last_refresh / last_runner / combos_classified — those
        # belong to the (heavier) `refresh` action.
        patch: dict[str, str | None] = {"read_from": flags.read_from}
        if flags.output_repo is not None:
            patch["output_repo"] = flags.output_repo
        return {"rights": patch}


def _verify_dataset_reachable(repo_id: str) -> None:
    """Best-effort existence check via the HF Hub API.

    Raises :class:`ScoutError` only for clear 404 / not-found responses.
    Other failures (network, auth) are logged and swallowed — setup
    shouldn't be gated on a flaky uplink.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        log.debug("huggingface_hub not installed; skipping setup validation")
        return
    try:
        HfApi().dataset_info(repo_id)
    except RepositoryNotFoundError as e:
        raise ScoutError(
            f"HF dataset {repo_id!r} not found (404). Check the repo id, "
            "or pass --read-from for a different dataset."
        ) from e
    except Exception as e:  # noqa: BLE001
        log.warning(
            "could not validate %s on HF Hub (%s); proceeding anyway", repo_id, e
        )

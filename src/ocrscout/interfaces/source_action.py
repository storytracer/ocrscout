"""``SourceAction`` ABC: one CLI verb per source.

Every source-specific admin verb (``refresh``, ``setup``, ``stats``,
``sample``, …) is a :class:`SourceAction` subclass declared on the
``actions`` ClassVar of its :class:`SourceAdapter`. The CLI driver in
:mod:`ocrscout.cli.source` walks each source's ``actions`` list at
import time and auto-generates ``ocrscout source <name> <action>``
typer commands from each action's Pydantic flags model — adapters
never touch typer themselves.

The action lifecycle, executed once per CLI invocation:

1. CLI parses argv → builds the action's ``flags`` Pydantic model.
2. CLI loads the current :class:`~ocrscout.sources._info.SourceInfo`
   (or stub) from ``info.yaml``.
3. CLI calls :meth:`SourceAction.fill_defaults` so the action can
   substitute sticky defaults from prior runs (e.g. last
   ``--source-repo``).
4. CLI constructs a :class:`SourceActionContext` and calls
   :meth:`SourceAction.run`.
5. If ``run`` returns a non-``None`` patch dict, the CLI merges it into
   ``info.yaml`` via
   :func:`ocrscout.sources._info.merge_info` (atomic, lock-guarded).

Actions never write to ``info.yaml`` directly — they declare what
changed via the return value, the driver persists it. This keeps the
locking + schema-merge concern in exactly one place.

Universal verbs (``info``, ``clear``) are CLI-layer-only and not
modeled as :class:`SourceAction` subclasses — they do nothing
source-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ocrscout.sources._info import SourceInfo


@dataclass(frozen=True)
class SourceActionContext:
    """Read-only handles passed into :meth:`SourceAction.run`.

    Constructed by the CLI driver, never by the action. Carries the
    source's name, its on-disk layout, and a snapshot of the current
    ``info.yaml`` so the action body can read state without re-loading
    or re-locking. The action expresses its updates via the return
    value of ``run``, which the driver merges back through
    :func:`ocrscout.sources._info.merge_info`.
    """

    source_name: str
    source_dir: Path
    catalog_dir: Path
    derived_dir: Path
    info: "SourceInfo"
    verbose: int = 0


class SourceAction(ABC):
    """One admin verb for one source."""

    name: ClassVar[str]
    """CLI verb, e.g. ``"refresh"``."""

    flags: ClassVar[type[BaseModel]]
    """Pydantic v2 model defining the action's CLI flags.

    Each field becomes a ``typer.Option`` via the introspector in
    :mod:`ocrscout.sources._flags_bridge`. Use ``Field(description=...)``
    to set the help text; ``Field(default=...)`` to set the default
    value.
    """

    description: ClassVar[str] = ""
    """Short help text for ``ocrscout source <src> <verb> --help``."""

    def fill_defaults(
        self,
        flags: BaseModel,
        info: "SourceInfo",
    ) -> BaseModel:
        """Substitute unset flags with sticky values from a prior run.

        Default: passthrough. Override when an action has fields whose
        most natural default is "what the user passed last time" — e.g.
        ``--source-repo`` / ``--output-repo`` on BHL refresh, which the
        user shouldn't have to retype every invocation.
        """
        return flags

    @abstractmethod
    def run(
        self,
        flags: BaseModel,
        ctx: SourceActionContext,
    ) -> dict[str, dict] | None:
        """Execute the action.

        Returns a section-keyed patch that the CLI driver merges into
        ``info.yaml`` — e.g. ``{"catalog": {"last_refresh": "..."}}``.
        Sections not present in the patch are left untouched. Return
        ``None`` for actions that are observational only (stats,
        sample) and never alter ``info.yaml``.
        """

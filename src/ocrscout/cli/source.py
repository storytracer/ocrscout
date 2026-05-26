"""``ocrscout source <name> <verb>`` — per-source admin subcommands.

Walks :func:`ocrscout.registry.registry.list("sources")` at import time
and builds a typer sub-app for each registered source. Each source
gets two universal verbs (``info``, ``clear``) wired here, plus one
typer command per :class:`~ocrscout.interfaces.source_action.SourceAction`
declared in the adapter's ``actions`` ClassVar — auto-generated via
:func:`ocrscout.sources._flags_bridge.build_typer_callback`.

The subgroup is intentionally **generic**: this module never imports a
concrete source adapter (no ``bhl`` import). All source-specific
behavior lives in the adapter's actions; the CLI just orchestrates.

Bare ``ocrscout source`` (no name) lists every registered source.
``ocrscout source <name>`` (no verb) defers to typer's ``--help``.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from typing import Any

import typer
from pydantic import BaseModel
from rich import print as rprint
from rich.table import Table

from ocrscout.cli import app
from ocrscout.errors import ScoutError
from ocrscout.interfaces.source_action import SourceAction, SourceActionContext
from ocrscout.log import setup_logging
from ocrscout.registry import registry
from ocrscout.sources._flags_bridge import build_typer_callback
from ocrscout.sources._info import (
    SourceInfo,
    load_info,
    merge_info,
)
from ocrscout.sources._paths import (
    catalog_dir,
    derived_dir,
    legacy_cache_dir,
    source_dir,
    sources_dir,
)

log = logging.getLogger(__name__)


source_app = typer.Typer(
    name="source",
    help="Per-source admin: cache, refresh, stats. See `ocrscout source <name> --help`.",
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(source_app, name="source")


@source_app.callback(invoke_without_command=True)
def _source_root(ctx: typer.Context) -> None:
    """List registered sources when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    names = registry.list("sources")
    if not names:
        rprint("[dim]no source adapters registered.[/dim]")
        return
    table = Table(title="Registered sources", show_lines=False)
    table.add_column("name", style="cyan")
    table.add_column("adapter", style="dim")
    table.add_column("admin verbs")
    for name in names:
        cls = registry.get("sources", name)
        verbs = ["info", "clear"] + [a.name for a in getattr(cls, "actions", [])]
        table.add_row(
            name,
            f"{cls.__module__}.{cls.__name__}",
            ", ".join(verbs),
        )
    rprint(table)


def _resolve_cache_subdir(cls: Any, source_name: str) -> str:
    """Adapters may set ``cache_subdir`` to override the default folder name."""
    sub = getattr(cls, "cache_subdir", None)
    return sub or source_name


def _format_info(source_name: str) -> None:
    """Render ``info.yaml`` + on-disk state for ``ocrscout source <name> info``.

    Pure-read: never creates the source directory. Constructs the path
    via ``sources_dir() / name`` rather than going through ``source_dir``
    (which mkdirs) so a fresh ``info`` invocation doesn't litter
    ``~/.ocrscout/sources/`` with empty dirs.
    """
    sd = sources_dir() / source_name
    info = load_info(source_name)
    legacy = legacy_cache_dir(source_name)

    rprint(f"[bold cyan]{source_name}[/bold cyan]  ({sd})")

    if info is None:
        rprint("  [dim]not configured — no info.yaml.[/dim]")
        rprint(
            "  [dim]Run [/dim]"
            f"[yellow]ocrscout source {source_name} setup ...[/yellow]"
            "[dim] or [/dim]"
            f"[yellow]ocrscout source {source_name} refresh ...[/yellow]"
            "[dim] (see --help) to provision.[/dim]"
        )
    else:
        rprint(f"  created: [cyan]{info.created_at.isoformat()}[/cyan]")
        if info.catalog.last_refresh:
            rprint(f"  catalog: refreshed {info.catalog.last_refresh.isoformat()}, "
                   f"{len(info.catalog.files)} file(s)")
        if info.rights.read_from:
            rprint(f"  rights:  read_from [cyan]{info.rights.read_from}[/cyan]")
        if info.rights.last_refresh:
            rprint(f"           refreshed {info.rights.last_refresh.isoformat()} "
                   f"via runner={info.rights.last_runner}, "
                   f"{info.rights.combos_classified or 0} combos classified")
        if info.derived.volumes_parquet_mtime:
            rprint(f"  derived: volumes.parquet built {info.derived.volumes_parquet_mtime.isoformat()}, "
                   f"{info.derived.volumes_parquet_rows or 0} row(s)")

    if legacy.exists():
        rprint(
            f"  [yellow]note:[/yellow] legacy cache at [dim]{legacy}[/dim] is no longer used; "
            "remove manually."
        )


def _clear_source(source_name: str) -> None:
    """Wipe ``~/.ocrscout/sources/<name>/`` (cache, derived, info.yaml)."""
    sd = sources_dir() / source_name
    if sd.exists():
        shutil.rmtree(sd)
        rprint(f"removed [dim]{sd}[/dim]")
    else:
        rprint(f"[dim]{sd} did not exist — nothing to clear.[/dim]")


def _make_action_invoker(
    source_name: str,
    cls: Any,
    action_cls: type[SourceAction],
) -> Any:
    """Build the closure passed into the flags-bridge wrapper.

    Runs the full action lifecycle when typer dispatches the command:
    load info → fill_defaults → build context → run → merge_info patch.
    """

    def _invoke(flags: BaseModel) -> None:
        info = load_info(source_name)
        if info is None:
            # First run: stub a minimal SourceInfo so fill_defaults / run
            # don't need to special-case None. It's only persisted if the
            # action returns a patch.
            info = SourceInfo(
                source_name=source_name,
                created_at=datetime.now().astimezone(),
            )
        action = action_cls()
        flags = action.fill_defaults(flags, info)
        # Materialise the per-source cache layout now that we're about to
        # invoke an action that may write to it. Pure-read commands
        # (`info`) deliberately don't go through this code path so they
        # don't litter ~/.ocrscout/sources/ with empty dirs.
        sd = source_dir(source_name)
        cd = catalog_dir(source_name)
        dd = derived_dir(source_name)
        for p in (sd, cd, dd):
            p.mkdir(parents=True, exist_ok=True)
        ctx = SourceActionContext(
            source_name=source_name,
            source_dir=sd,
            catalog_dir=cd,
            derived_dir=dd,
            info=info,
        )
        try:
            patch = action.run(flags, ctx)
        except ScoutError as e:
            log.error("%s", e)
            raise typer.Exit(code=1) from e
        if patch:
            merge_info(source_name, patch)

    return _invoke


def _build_source_subapp(source_name: str, cls: Any) -> typer.Typer:
    """Construct ``ocrscout source <name>``'s typer sub-app."""
    sub = typer.Typer(
        name=source_name,
        help=f"Admin for the `{source_name}` source.",
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @sub.command("info", help="Show this source's provisioning state.")
    def _info() -> None:
        setup_logging(verbosity=0, quiet=False)
        _format_info(source_name)

    @sub.command("clear", help="Wipe this source's cache and info.yaml.")
    def _clear() -> None:
        setup_logging(verbosity=0, quiet=False)
        _clear_source(source_name)

    for action_cls in getattr(cls, "actions", []):
        callback = build_typer_callback(
            action_cls.flags,
            _make_action_invoker(source_name, cls, action_cls),
        )
        sub.command(action_cls.name, help=action_cls.description or None)(callback)

    return sub


# Walk the registry once at import time and attach a sub-app per source.
# Registry lookups are lazy-imported, so this triggers loading of every
# built-in source module — fine for a CLI invocation, and exactly the same
# cost the existing `ocrscout run --source <name>` paths already pay.
for _name in registry.list("sources"):
    _cls = registry.get("sources", _name)
    source_app.add_typer(_build_source_subapp(_name, _cls), name=_name)

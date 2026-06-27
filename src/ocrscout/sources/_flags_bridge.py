"""Pydantic v2 → typer callback construction.

Each :class:`~ocrscout.interfaces.source_action.SourceAction` declares
its CLI flags as a Pydantic model on the ``flags`` ClassVar. This
module turns one of those models into a typer-compatible callback so
the CLI driver in :mod:`ocrscout.cli.source` can register the action
under ``ocrscout source <name> <verb>`` without per-action typer
boilerplate.

The bridge supports the field types we use across our actions:

* ``str``, ``int``, ``float``, ``bool``
* ``str | None``, ``int | None``, ``float | None``
* ``Literal["a", "b", ...]``  (rendered as a ``click.Choice``)
* ``list[str]``  (repeatable option)

Fields not falling in these shapes fail loudly at action-registration
time — better a clear error at startup than mysterious typer behavior.
Actions that need exotic types can override
``SourceAction.build_typer_callback`` (escape hatch — not used today).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import typer
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ocrscout.log import setup_logging


def _option_decl(field_name: str) -> str:
    """``source_repo`` → ``--source-repo``."""
    return "--" + field_name.replace("_", "-")


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """If ``annotation`` is ``T | None``, return ``(True, T)``.

    Handles both ``Union[T, None]`` and PEP 604 ``T | None``.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(annotation):
            return True, args[0]
    return False, annotation


def _resolve_typer_type(annotation: Any) -> Any:
    """Map a Pydantic field annotation to the type typer expects.

    Returns the type to use in the constructed function's parameter
    annotation. ``Literal[...]`` falls through directly (typer 0.9+
    renders it as a ``click.Choice``); ``list[str]`` falls through
    (typer makes it a repeatable option).
    """
    _, inner = _is_optional(annotation)
    origin = get_origin(inner)
    if origin is list:
        # typer requires the parameterized list type; pass through.
        return inner
    if origin is Literal:
        return inner
    if inner in (str, int, float, bool):
        return inner
    raise TypeError(
        f"unsupported flags-model field type {annotation!r}; "
        "supported: str, int, float, bool, Literal[...], list[str], "
        "and `T | None` variants. Override SourceAction.build_typer_callback "
        "to handle exotic types."
    )


def _build_option(field_name: str, info: Any, py_type: Any) -> Any:
    """Construct a ``typer.Option`` carrying the field's default + help text."""
    decl = _option_decl(field_name)
    help_text = info.description or ""
    if info.default is PydanticUndefined:
        # Required field — no Pydantic default. Make it a required typer option.
        return typer.Option(..., decl, help=help_text)
    default = info.default
    # bool flags should render as ``--foo / --no-foo``.
    if py_type is bool:
        return typer.Option(default, decl, help=help_text)
    return typer.Option(default, decl, help=help_text)


def build_typer_callback(
    flags_cls: type[BaseModel],
    on_invoke: Callable[[BaseModel], Any],
) -> Callable[..., Any]:
    """Construct a function whose signature mirrors ``flags_cls``'s fields.

    Typer registers commands by introspecting a function's signature
    (parameter names, annotations, and ``typer.Option`` defaults). We
    can't write that function statically because the fields vary per
    action — so we synthesize it via :class:`inspect.Signature` and
    attach it to a wrapper that bundles the kwargs back into a
    validated ``flags_cls`` instance and hands it to ``on_invoke``.

    ``on_invoke`` is the closure where the CLI driver does the
    ``fill_defaults → run → merge_info`` lifecycle.
    """
    params: list[inspect.Parameter] = []
    field_order: list[str] = []
    for name, info in flags_cls.model_fields.items():
        py_type = _resolve_typer_type(info.annotation)
        option = _build_option(name, info, py_type)
        params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=option,
                annotation=py_type,
            )
        )
        field_order.append(name)

    # Inject universal `-v` / `-q` options so every auto-generated action
    # configures stdlib logging on entry. Without this, log.info() calls
    # inside the action body go nowhere (no handler is attached).
    params.append(
        inspect.Parameter(
            name="verbose",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(0, "-v", "--verbose", count=True),
            annotation=int,
        )
    )
    params.append(
        inspect.Parameter(
            name="quiet",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(False, "-q", "--quiet"),
            annotation=bool,
        )
    )

    def _wrapper(**kwargs: Any) -> Any:
        verbose = kwargs.pop("verbose", 0)
        quiet = kwargs.pop("quiet", False)
        setup_logging(verbosity=verbose, quiet=quiet)
        validated = flags_cls.model_validate(kwargs)
        return on_invoke(validated)

    _wrapper.__signature__ = inspect.Signature(parameters=params)  # type: ignore[attr-defined]
    _wrapper.__annotations__ = {p.name: p.annotation for p in params}
    return _wrapper

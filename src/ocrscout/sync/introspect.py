"""Static introspection of HuggingFace ``uv-scripts/ocr`` Python files.

NEVER executes the script. Uses ``ast.parse`` + a regex for the PEP 723 block
+ ``tomllib.loads`` for the captured TOML — the scripts come from an untrusted
source and must not run on the user's machine just to read their metadata.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocrscout.errors import IntrospectionError

_PEP723_RE = re.compile(
    r"^# /// script\s*$(?P<body>.*?)^# ///\s*$",
    re.MULTILINE | re.DOTALL,
)
_LEADING_HASH_RE = re.compile(r"^# ?", re.MULTILINE)


@dataclass
class HfScriptInfo:
    path: Path
    requires_python: str | None = None
    dependencies: list[str] = field(default_factory=list)
    prompt_templates: dict[str, str] = field(default_factory=dict)
    default_model: str | None = None
    default_output_column: str | None = None
    imports: set[str] = field(default_factory=set)


def introspect_hf_script(path: str | Path) -> HfScriptInfo:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise IntrospectionError(f"cannot read {p}: {e}") from e
    try:
        tree = ast.parse(text, filename=str(p))
    except SyntaxError as e:
        raise IntrospectionError(f"{p} is not valid Python: {e}") from e

    pep723 = _parse_pep723(text)
    return HfScriptInfo(
        path=p,
        requires_python=pep723.get("requires-python"),
        dependencies=list(pep723.get("dependencies", []) or []),
        prompt_templates=_find_prompt_templates(tree),
        default_model=_find_argparse_default(tree, "--model"),
        default_output_column=_find_argparse_default(tree, "--output-column"),
        imports=_top_level_imports(tree),
    )


def _parse_pep723(text: str) -> dict[str, Any]:
    m = _PEP723_RE.search(text)
    if not m:
        return {}
    body = m.group("body")
    cleaned = _LEADING_HASH_RE.sub("", body)
    try:
        return tomllib.loads(cleaned)
    except tomllib.TOMLDecodeError:
        return {}


def _find_prompt_templates(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "PROMPT_TEMPLATES"):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key_node, val_node in zip(node.value.keys, node.value.values, strict=True):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            value = _string_value(val_node)
            if value is not None:
                out[key_node.value] = value
        # Use the first match; later assignments would shadow it at runtime,
        # but that pattern is not used in the upstream scripts.
        break
    return out


def _string_value(node: ast.expr) -> str | None:
    """Extract a string literal value from an AST expression node.

    Handles plain string Constants, implicit-concatenation (Python parses
    ``"a" "b"`` as a single Constant), and f-strings (``JoinedStr``) by
    replacing each ``FormattedValue`` with the placeholder ``"{...}"`` so the
    template still tells the caller about the prompt's shape.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue):
                parts.append("{...}")
            else:
                return None
        return "".join(parts)
    return None


def _find_argparse_default(tree: ast.Module, flag: str) -> str | None:
    """Find the ``default=`` argument of ``parser.add_argument(flag, ...)``.

    Returns the literal string value if it's a plain string literal, else
    ``None`` (e.g. for ``default=os.environ.get(...)``).
    """
    top_level_constants = _collect_top_level_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        # Confirm one of the positional args matches the flag.
        flag_match = False
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == flag:
                flag_match = True
                break
        if not flag_match:
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            return _resolve_literal_or_name(kw.value, top_level_constants)
    return None


def _resolve_literal_or_name(node: ast.expr, constants: dict[str, Any]) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        # Booleans and ints could conceivably be defaults but aren't useful here.
        return None
    if isinstance(node, ast.Name) and node.id in constants:
        val = constants[node.id]
        return val if isinstance(val, str) else None
    return None


def _collect_top_level_constants(tree: ast.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Constant):
            out[node.targets[0].id] = node.value.value
    return out


def _top_level_imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".", 1)[0])
    return out

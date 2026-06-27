"""Derive a ``datasets.Features`` schema from a typed stage-row model.

The single source of truth for each stage's parquet columns is its row model in
:mod:`ocrscout.io.rows`; this module turns the model's fields into an Arrow
schema. No more hand-listed ``Features`` dicts kept in lockstep with a separate
row-builder.
"""

from __future__ import annotations

import types
import typing
from functools import cache

from datasets import Features, Value

from ocrscout.io.rows import StageRow


def _unwrap_optional(annotation: object) -> object:
    """Strip ``| None`` / ``Optional[...]`` to the inner type."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _value_for(annotation: object) -> Value:
    inner = _unwrap_optional(annotation)
    if inner is bool:
        return Value("bool")
    if inner is int:
        return Value("int64")
    if inner is float:
        return Value("float64")
    if inner is str:
        return Value("string")
    # Lists/dicts/models are JSON-encoded → handled as string columns elsewhere.
    return Value("string")


@cache
def features_for(row_type: type[StageRow]) -> Features:
    """The ``datasets.Features`` for a stage-row model's on-disk columns.

    Scalar fields map by annotation; rich fields (declared in the model's
    ``json_fields()``) become ``string`` columns under their ``*_json`` name.
    """
    json_fields = row_type.json_fields()
    schema: dict[str, Value] = {}
    for name, field in row_type.model_fields.items():
        if name in json_fields:
            continue
        schema[name] = _value_for(field.annotation)
    for column in json_fields.values():
        schema[column] = Value("string")
    return Features(schema)


def columns_for(row_type: type[StageRow]) -> tuple[str, ...]:
    return tuple(features_for(row_type).keys())

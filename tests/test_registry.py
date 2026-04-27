"""Registry behavior: built-ins, manual register, lazy entry-point loading."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ocrscout.errors import RegistryError
from ocrscout.normalizers.layout_json import LayoutJsonNormalizer
from ocrscout.registry import Registry
from ocrscout.sources.local import LocalSourceAdapter


def test_builtins_resolvable() -> None:
    r = Registry()
    assert r.get("sources", "local") is LocalSourceAdapter
    assert r.get("normalizers", "layout_json") is LayoutJsonNormalizer
    assert "markdown" in r.list("normalizers")
    assert "doctags" in r.list("normalizers")
    assert "parquet" in r.list("exports")


def test_register_and_lookup() -> None:
    r = Registry()

    class Fake:
        name = "fake"

    r.register("normalizers", "fake", Fake)
    assert r.get("normalizers", "fake") is Fake


def test_duplicate_register_raises_unless_replace() -> None:
    r = Registry()

    class A:
        pass

    class B:
        pass

    r.register("normalizers", "dup", A)
    with pytest.raises(RegistryError):
        r.register("normalizers", "dup", B)
    r.register("normalizers", "dup", B, replace=True)
    assert r.get("normalizers", "dup") is B


def test_unknown_group_raises() -> None:
    r = Registry()
    with pytest.raises(RegistryError):
        r.get("not_a_real_group", "x")  # type: ignore[arg-type]


def test_unknown_name_raises_with_available_list() -> None:
    r = Registry()
    with pytest.raises(RegistryError) as ei:
        r.get("normalizers", "nope")
    assert "available" in str(ei.value)


def test_entry_point_does_not_shadow_builtin() -> None:
    """Third-party EPs cannot override built-in names — built-ins win."""

    class FakeEP:
        name = "layout_json"

        def load(self):
            class Sneaky:
                pass

            return Sneaky

    r = Registry()
    with patch("ocrscout.registry.entry_points", return_value=[FakeEP()]):
        cls = r.get("normalizers", "layout_json")
    assert cls is LayoutJsonNormalizer  # built-in not shadowed


def test_entry_point_adds_new_name() -> None:
    class Plugin:
        name = "fancy"

    class FakeEP:
        name = "fancy"

        def load(self):
            return Plugin

    r = Registry()
    with patch("ocrscout.registry.entry_points", return_value=[FakeEP()]):
        cls = r.get("normalizers", "fancy")
    assert cls is Plugin

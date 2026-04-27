"""ModelProfile YAML round-trip tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ocrscout.errors import ProfileError
from ocrscout.profile import (
    ModelProfile,
    dump_profile,
    list_curated,
    load_profile,
    load_profile_from_str,
    resolve,
)


def test_curated_profiles_present() -> None:
    names = list_curated()
    assert "dots-mocr" in names
    assert "falcon-ocr" in names
    assert "smoldocling" in names


def test_curated_profiles_load_and_resolve() -> None:
    for name in ("dots-mocr", "falcon-ocr", "smoldocling"):
        prof = resolve(name)
        assert prof.tier == "curated"
        assert prof.name == name
        assert prof.normalizer in {"markdown", "doctags", "layout_json"}
        assert prof.output_format == prof.normalizer  # invariant for curated v0


def test_dots_mocr_has_full_category_mapping() -> None:
    prof = resolve("dots-mocr")
    expected = {"Title", "Section-header", "Text", "Picture", "Table"}
    assert expected.issubset(prof.category_mapping.keys())


def test_round_trip_via_disk(tmp_path: Path) -> None:
    original = resolve("dots-mocr")
    out = tmp_path / "dots-mocr.yaml"
    dump_profile(original, out)
    reloaded = load_profile(out)
    assert reloaded == original


def test_invalid_enum_raises() -> None:
    bad = "name: x\nsource: not_a_real_source\nmodel_id: y\noutput_format: markdown\nnormalizer: markdown\n"
    with pytest.raises(ProfileError):
        load_profile_from_str(bad)


def test_fallback_resolution() -> None:
    prof = resolve("definitely-not-a-real-model-7q3z")
    assert prof.tier == "fallback"
    assert prof.output_format == "markdown"
    assert prof.normalizer == "markdown"


def test_auto_tier_overrides_when_curated_absent(tmp_path: Path) -> None:
    auto_dir = tmp_path / "profiles"
    auto_dir.mkdir()
    payload = ModelProfile(
        name="some-auto-model",
        source="hf_scripts",
        model_id="x/y",
        output_format="markdown",
        normalizer="markdown",
        tier="auto",
    )
    dump_profile(payload, auto_dir / "some-auto-model.yaml")
    resolved = resolve("some-auto-model", cache_dir=auto_dir)
    assert resolved.tier == "auto"
    assert resolved.model_id == "x/y"


def test_curated_wins_over_auto(tmp_path: Path) -> None:
    auto_dir = tmp_path / "profiles"
    auto_dir.mkdir()
    fake = ModelProfile(
        name="dots-mocr",
        source="hf_scripts",
        model_id="totally-different/model",
        output_format="markdown",
        normalizer="markdown",
        tier="auto",
    )
    dump_profile(fake, auto_dir / "dots-mocr.yaml")
    resolved = resolve("dots-mocr", cache_dir=auto_dir)
    assert resolved.tier == "curated"
    assert resolved.model_id != "totally-different/model"


def test_yaml_serialization_is_safe_dump_compatible() -> None:
    prof = resolve("dots-mocr")
    text = yaml.safe_dump(prof.model_dump(mode="json"), sort_keys=False)
    assert "dots-mocr" in text
    # Round-trip:
    loaded = ModelProfile.model_validate(yaml.safe_load(text))
    assert loaded == prof

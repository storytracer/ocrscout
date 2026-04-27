"""Static introspection of HF uv-scripts."""

from __future__ import annotations

from pathlib import Path

from ocrscout.sync.introspect import introspect_hf_script


def test_introspects_sample_script(sample_script: Path) -> None:
    info = introspect_hf_script(sample_script)
    assert info.requires_python == ">=3.11"
    assert "datasets" in info.dependencies
    assert "vllm>=0.15.1" in info.dependencies

    # Prompt templates: keys preserved, JoinedStr handled gracefully.
    assert set(info.prompt_templates.keys()) == {
        "ocr",
        "layout-all",
        "layout-only",
        "doctags",
    }
    assert "Extract the text" in info.prompt_templates["ocr"]
    # JoinedStr value should still be a string (with our placeholder for the
    # FormattedValue).
    assert isinstance(info.prompt_templates["doctags"], str)

    # Argparse defaults — both literal and via top-level NAME constant.
    assert info.default_model == "rednote-hilab/dots.mocr"
    assert info.default_output_column == "markdown"

    # Imports captured at top dotted level only.
    assert "datasets" in info.imports
    assert "huggingface_hub" in info.imports
    assert "argparse" in info.imports


def test_non_literal_default_returns_none(tmp_path: Path) -> None:
    script = tmp_path / "weird.py"
    script.write_text(
        "import os, argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--model', default=os.environ.get('M', 'fallback'))\n",
        encoding="utf-8",
    )
    info = introspect_hf_script(script)
    assert info.default_model is None


def test_missing_pep723_block_is_tolerated(tmp_path: Path) -> None:
    script = tmp_path / "no-pep723.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--model', default='hello/world')\n",
        encoding="utf-8",
    )
    info = introspect_hf_script(script)
    assert info.requires_python is None
    assert info.dependencies == []
    assert info.default_model == "hello/world"

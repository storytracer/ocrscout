"""CLI smoke tests: every command's --help works; scout dumps a config."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ocrscout.cli import app

runner = CliRunner()


def test_top_level_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for cmd in ("scout", "run", "sync", "report"):
        assert cmd in out


def test_scout_help() -> None:
    result = runner.invoke(app, ["scout", "--help"])
    assert result.exit_code == 0
    for opt in ("--source", "--models", "--reference", "--sample", "--benchmark"):
        assert opt in result.stdout


def test_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_sync_help() -> None:
    result = runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0
    assert "--scripts-dir" in result.stdout
    assert "--no-fetch" in result.stdout


def test_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0


def test_scout_dumps_pipeline_config(tmp_path: Path, images_dir: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "scout",
            "--source", str(images_dir),
            "--models", "dots-mocr,smoldocling",
            "--sample", "2",
            "--output-dir", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    pipeline_yaml = out / "pipeline.yaml"
    assert pipeline_yaml.is_file()

    import yaml

    cfg = yaml.safe_load(pipeline_yaml.read_text())
    assert cfg["name"] == "scout"
    assert cfg["models"] == ["dots-mocr", "smoldocling"]
    assert cfg["source"]["name"] == "local"


def test_run_loads_pipeline_yaml(tmp_path: Path, images_dir: Path) -> None:
    # First produce a pipeline.yaml via scout, then load it via run.
    out = tmp_path / "out"
    runner.invoke(
        app,
        [
            "scout",
            "--source", str(images_dir),
            "--models", "dots-mocr",
            "--output-dir", str(out),
        ],
    )
    yaml_path = out / "pipeline.yaml"
    assert yaml_path.is_file()
    result = runner.invoke(app, ["run", str(yaml_path)])
    assert result.exit_code == 0
    assert "stub" in result.stdout.lower()

"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_script(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_script.py"


@pytest.fixture
def sample_layout(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample_layout.json"


@pytest.fixture
def images_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "images"

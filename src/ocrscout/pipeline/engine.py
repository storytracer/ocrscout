"""PipelineEngine: walks a PipelineConfig DAG of stages.

Stub — arrives in phase 6 of the roadmap.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ocrscout.errors import PipelineError
from ocrscout.types import PipelineConfig


class PipelineEngine:
    """Loads ``pipeline.yaml`` files into ``PipelineConfig`` and (eventually)
    executes them."""

    def load(self, path: str | Path) -> PipelineConfig:
        p = Path(path)
        if not p.is_file():
            raise PipelineError(f"pipeline file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise PipelineError(f"invalid YAML in {p}: {e}") from e
        if not isinstance(data, dict):
            raise PipelineError(f"pipeline {p} must be a YAML mapping")
        try:
            return PipelineConfig.model_validate(data)
        except Exception as e:  # noqa: BLE001
            raise PipelineError(f"invalid pipeline config in {p}: {e}") from e

    def execute(self, config: PipelineConfig) -> None:
        raise NotImplementedError(
            "PipelineEngine.execute is not implemented in v0; arrives in phase 6."
        )

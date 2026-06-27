"""Concrete pipeline stages: sample → layout → ocr → normalize."""

from __future__ import annotations

from ocrscout.pipeline.stages.normalize import NormalizeStage
from ocrscout.pipeline.stages.sample import SampleStage

__all__ = ["NormalizeStage", "SampleStage"]

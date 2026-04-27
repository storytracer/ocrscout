"""Reporter ABC: turns a results directory into a rendered report."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar


class Reporter(ABC):
    """Renders a report from a results directory."""

    name: ClassVar[str]

    @abstractmethod
    def render(self, results_dir: Path, out: Path) -> Path: ...

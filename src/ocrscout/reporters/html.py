"""HtmlReporter: render a results directory as an HTML report.

Stub — arrives in phase 8 of the roadmap.
"""

from __future__ import annotations

from pathlib import Path

from ocrscout.interfaces.reporter import Reporter


class HtmlReporter(Reporter):
    name = "html"

    def render(self, results_dir: Path, out: Path) -> Path:
        raise NotImplementedError("HtmlReporter is not implemented in v0.")

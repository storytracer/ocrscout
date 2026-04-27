"""HfScriptsBackend: run a uv-scripts/ocr Python file in a `uv run` subprocess.

Stub — full subprocess orchestration arrives in phase 2 of the roadmap.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput


class HfScriptsBackend(ModelBackend):
    name = "hf_scripts"

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        raise NotImplementedError(
            "HfScriptsBackend.prepare is not implemented in v0; "
            "subprocess invocation arrives in phase 2."
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        raise NotImplementedError(
            "HfScriptsBackend.run is not implemented in v0; "
            "subprocess invocation arrives in phase 2."
        )

"""DoclingBackend: wrap the full Docling pipeline as one model backend.

Stub — requires the optional ``docling`` extra. Arrives in phase 7 of the
roadmap.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput


class DoclingBackend(ModelBackend):
    name = "docling"

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        raise NotImplementedError(
            "DoclingBackend is not implemented in v0; install the `docling` extra "
            "and wait for phase 7."
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        raise NotImplementedError("DoclingBackend is not implemented in v0.")

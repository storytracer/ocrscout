"""TesseractBackend: invoke the tesseract CLI for legacy OCR comparison.

Stub — arrives in phase 7 of the roadmap.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput


class TesseractBackend(ModelBackend):
    name = "tesseract"

    def prepare(self, profile: ModelProfile) -> BackendInvocation:
        raise NotImplementedError("TesseractBackend is not implemented in v0.")

    def run(
        self,
        invocation: BackendInvocation,
        pages: Sequence[PageImage],
    ) -> Iterator[tuple[PageImage, RawOutput]]:
        raise NotImplementedError("TesseractBackend is not implemented in v0.")

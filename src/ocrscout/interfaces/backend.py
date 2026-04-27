"""ModelBackend ABC: prepares and runs OCR model invocations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput


class ModelBackend(ABC):
    """Wraps a way of invoking an OCR model — subprocess, in-process, or HTTP."""

    name: ClassVar[str]

    @abstractmethod
    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation: ...

    @abstractmethod
    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]: ...

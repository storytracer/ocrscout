"""ModelBackend ABC: prepares and runs OCR model invocations.

Pipeline shape: ``prepare(profile)`` once per model (warms up detectors,
resolves proxy URL, registers cost callback, etc.); ``run(invocation,
pages_batch)`` zero-or-more times with successive page batches, yielding
``(page, raw)`` tuples in completion order. Resources cached on
``invocation.extra`` (detector instances, HTTP sessions) persist across
``run`` calls so per-model warmup cost is paid exactly once.

The split is what bounds memory: the orchestrator chunks
``source.iter_pages()`` into batches sized to the backend's in-flight
concurrency, so at most ``chunk_size`` decoded PIL images are resident
during any single ``run`` call — instead of ``sample`` images held all
at once. See [`_run_one_model`](src/ocrscout/cli/run.py).
"""

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
    def prepare(self, profile: ModelProfile) -> BackendInvocation: ...

    @abstractmethod
    def run(
        self,
        invocation: BackendInvocation,
        pages: Sequence[PageImage],
    ) -> Iterator[tuple[PageImage, RawOutput]]: ...

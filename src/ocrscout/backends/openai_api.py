"""OpenAIApiBackend: call any OpenAI-compatible chat-completions endpoint.

Stub — covers vLLM serve, Ollama, LM Studio, Gemini OpenAI-mode, etc. Arrives
in phase 7 of the roadmap.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput


class OpenAIApiBackend(ModelBackend):
    name = "openai_api"

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        raise NotImplementedError("OpenAIApiBackend is not implemented in v0.")

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        raise NotImplementedError("OpenAIApiBackend is not implemented in v0.")

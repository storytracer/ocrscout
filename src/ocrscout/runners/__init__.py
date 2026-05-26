"""Concrete ``Runner`` implementations and their supporting helpers.

Each ``Runner`` (``LocalRunner``, future ``SkyPilotRunner``,
``HuggingFaceRunner``) orchestrates a vLLM + LiteLLM stack on its target
infrastructure. The supporting modules (``_daemon``, ``_preflight``)
provide shared primitives.

``vllm_runner.py`` (the legacy one-shot subprocess script) is scheduled
for deletion in Phase 1 step 6 — it predates the LiteLLM-always
architecture and no longer has a caller.
"""

from ocrscout.runners.local import LocalRunner

__all__ = ["LocalRunner"]

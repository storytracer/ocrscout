"""Static introspection of HF ``uv-scripts/ocr`` scripts for curated profiles.

The package historically wrote auto-generated profiles to a user cache. The
ocrscout policy is now **curated profiles only**: every model gets a
hand-tuned YAML in ``src/ocrscout/profiles/``. This module is therefore
reduced to its read-only helpers — fetching the upstream script source and
parsing it via ``ast.parse`` — which the ``ocrscout introspect`` CLI uses
to draft a starter YAML for a human (or Claude Code) to refine.

The script source is **never executed**. Untrusted code from the Hub must
not run on the user's machine just to read its metadata.
"""

from __future__ import annotations

from ocrscout.sync.introspect import HfScriptInfo, introspect_hf_script

__all__ = [
    "HfScriptInfo",
    "introspect_hf_script",
]

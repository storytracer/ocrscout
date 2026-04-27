"""Standalone runner scripts spawned in subprocesses by ocrscout backends.

These modules are NOT imported by the ocrscout package itself — they execute
inside ``uv run`` venvs that have heavy GPU dependencies (vllm, torch). Sitting
in their own subpackage keeps them away from sibling shadowing: when Python
adds ``sys.path[0]`` for the runner script, no nearby module name (``vllm.py``,
``docling.py``) collides with the real third-party packages the runner imports.
"""

"""Registry of pluggable components.

Built-ins are registered lazily on first access (avoids import cycles between
the registry module and the concrete adapter modules). Third-party packages
register via entry-point groups: ``ocrscout.<group>``.

Third-party entry-points cannot shadow built-in names — built-ins are the
stable contract.
"""

from __future__ import annotations

import threading
from importlib.metadata import entry_points
from typing import Any, Literal

from ocrscout.errors import RegistryError

EntryGroup = Literal[
    "sources",
    "references",
    "backends",
    "normalizers",
    "exports",
    "evaluators",
    "benchmarks",
    "reporters",
    "layout_detectors",
]

_GROUPS: tuple[EntryGroup, ...] = (
    "sources",
    "references",
    "backends",
    "normalizers",
    "exports",
    "evaluators",
    "benchmarks",
    "reporters",
    "layout_detectors",
)


def _builtin_specs() -> dict[EntryGroup, dict[str, str]]:
    """Map (group, name) -> dotted import path. Resolved lazily so importing
    ``ocrscout.registry`` never imports concrete adapters."""
    return {
        "sources": {
            "hf_dataset": "ocrscout.sources.hf_dataset:HfDatasetSourceAdapter",
            "bhl": "ocrscout.sources.bhl:BhlSourceAdapter",
        },
        "references": {
            "plain_text": "ocrscout.references.plain_text:PlainTextReferenceAdapter",
            "bhl_ocr": "ocrscout.references.bhl_ocr:BhlOcrReferenceAdapter",
        },
        "backends": {
            "vllm": "ocrscout.backends.vllm:VllmBackend",
            "openai_api": "ocrscout.backends.openai_api:OpenAIApiBackend",
            "tesseract": "ocrscout.backends.tesseract:TesseractBackend",
            "docling": "ocrscout.backends.docling:DoclingBackend",
            "layout_chat": "ocrscout.backends.layout_chat:LayoutChatBackend",
        },
        "normalizers": {
            "markdown": "ocrscout.normalizers.markdown:MarkdownNormalizer",
            "doctags": "ocrscout.normalizers.doctags:DocTagsNormalizer",
            "layout_json": "ocrscout.normalizers.layout_json:LayoutJsonNormalizer",
            "passthrough": "ocrscout.normalizers.passthrough:PassthroughNormalizer",
        },
        "exports": {
            "parquet": "ocrscout.exports.parquet:ParquetExportAdapter",
        },
        "evaluators": {
            "edit_distance": "ocrscout.evaluators.edit_distance:EditDistanceEvaluator",
            "vlm_judge": "ocrscout.evaluators.vlm_judge:VlmJudgeEvaluator",
        },
        "benchmarks": {},
        "reporters": {
            "html": "ocrscout.reporters.html:HtmlReporter",
        },
        "layout_detectors": {
            "pp-doclayout-v3": "ocrscout.layout_detectors.pp_doclayout_v3:PpDocLayoutV3Detector",
        },
    }


def _import_dotted(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise RegistryError(f"invalid built-in spec {spec!r} (expected 'mod:Class')")
    import importlib

    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise RegistryError(f"built-in {spec!r} not found: {e}") from e


class Registry:
    """In-memory registry of components, with lazy entry-point discovery."""

    def __init__(self) -> None:
        self._stores: dict[EntryGroup, dict[str, Any]] = {g: {} for g in _GROUPS}
        self._builtin_loaded: set[EntryGroup] = set()
        self._ep_loaded: set[EntryGroup] = set()
        self._lock = threading.RLock()

    # --- public API ---------------------------------------------------------

    def register(
        self,
        group: EntryGroup,
        name: str,
        cls: Any,
        *,
        replace: bool = False,
    ) -> None:
        if group not in self._stores:
            raise RegistryError(f"unknown group {group!r}")
        with self._lock:
            self._ensure_builtins(group)
            store = self._stores[group]
            if name in store and not replace:
                raise RegistryError(
                    f"{group}/{name!r} is already registered (pass replace=True to override)"
                )
            store[name] = cls

    def get(self, group: EntryGroup, name: str) -> Any:
        if group not in self._stores:
            raise RegistryError(f"unknown group {group!r}")
        with self._lock:
            self._ensure_builtins(group)
            self._ensure_entry_points(group)
            try:
                return self._stores[group][name]
            except KeyError as e:
                available = sorted(self._stores[group])
                raise RegistryError(
                    f"no {group} named {name!r}; available: {available}"
                ) from e

    def list(self, group: EntryGroup) -> list[str]:
        if group not in self._stores:
            raise RegistryError(f"unknown group {group!r}")
        with self._lock:
            self._ensure_builtins(group)
            self._ensure_entry_points(group)
            return sorted(self._stores[group])

    def groups(self) -> tuple[EntryGroup, ...]:
        return _GROUPS

    # --- internals ----------------------------------------------------------

    def _ensure_builtins(self, group: EntryGroup) -> None:
        if group in self._builtin_loaded:
            return
        store = self._stores[group]
        for name, spec in _builtin_specs().get(group, {}).items():
            if name not in store:
                store[name] = _import_dotted(spec)
        self._builtin_loaded.add(group)

    def _ensure_entry_points(self, group: EntryGroup) -> None:
        if group in self._ep_loaded:
            return
        eps = entry_points(group=f"ocrscout.{group}")
        for ep in eps:
            if ep.name in self._stores[group]:
                # Built-in or already-registered names cannot be shadowed.
                continue
            try:
                cls = ep.load()
            except Exception:  # noqa: BLE001
                # Don't let a broken third-party plugin break the registry.
                continue
            self._stores[group][ep.name] = cls
        self._ep_loaded.add(group)


registry = Registry()

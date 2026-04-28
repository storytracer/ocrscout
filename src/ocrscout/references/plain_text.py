"""PlainTextReferenceAdapter: one .txt per page, matched by stem."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.types import PageImage, Reference, ReferenceProvenance


class PlainTextReferenceAdapter(ReferenceAdapter):
    """Looks for ``<root>/<page_id_stem><suffix>``.

    For nested ``page_id`` like ``"docA/page1.png"``, the stem ``"docA/page1"``
    is appended to ``root`` to form the lookup path.

    Caller-supplied ``method`` and ``engine`` populate the
    :class:`ReferenceProvenance` on every emitted :class:`Reference` so
    downstream comparison results can be interpreted correctly. The
    default is ``method="unknown"`` because this adapter sees only files
    on disk and has no idea whether they're human-typed, OCR'd, or LLM-
    generated.
    """

    name = "plain_text"

    def __init__(
        self,
        root: str | Path,
        *,
        suffix: str = ".txt",
        method: Literal["human", "ocr", "llm", "mixed", "unknown"] = "unknown",
        engine: str | None = None,
        confidence: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.suffix = suffix
        self._provenance = ReferenceProvenance(
            method=method, engine=engine, confidence=confidence,
        )

    def get(self, page: PageImage) -> Reference | None:
        page_id = page.page_id
        rel = Path(page_id)
        candidate = self.root / rel.with_suffix(self.suffix)
        if not candidate.is_file():
            # Also try the bare stem (last component) as a fallback.
            candidate = self.root / (Path(page_id).stem + self.suffix)
            if not candidate.is_file():
                return None
        text = candidate.read_text(encoding="utf-8")
        return Reference(
            page_id=page_id, text=text, provenance=self._provenance,
        )

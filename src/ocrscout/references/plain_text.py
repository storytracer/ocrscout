"""PlainTextReferenceAdapter: one .txt per page, matched by stem."""

from __future__ import annotations

from pathlib import Path

from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.types import Reference


class PlainTextReferenceAdapter(ReferenceAdapter):
    """Looks for ``<root>/<page_id_stem><suffix>``.

    For nested ``page_id`` like ``"docA/page1.png"``, the stem ``"docA/page1"``
    is appended to ``root`` to form the lookup path.
    """

    name = "plain_text"

    def __init__(self, root: str | Path, *, suffix: str = ".txt") -> None:
        self.root = Path(root)
        self.suffix = suffix

    def get(self, page_id: str) -> Reference | None:
        rel = Path(page_id)
        candidate = self.root / rel.with_suffix(self.suffix)
        if not candidate.is_file():
            # Also try the bare stem (last component) as a fallback.
            candidate = self.root / (Path(page_id).stem + self.suffix)
            if not candidate.is_file():
                return None
        text = candidate.read_text(encoding="utf-8")
        return Reference(page_id=page_id, text=text)

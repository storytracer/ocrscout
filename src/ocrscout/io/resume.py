"""Resume cursors, read straight from the output shards.

The parquet shards are the single source of truth for "what's already done"
(``progress.json`` is demoted to run metadata). ``ResumeMode.PAGE`` keys on
``page_id`` (sample/layout — one row per page); ``ResumeMode.PAGE_MODEL`` keys
on ``(page_id, model)`` (ocr/normalize — one row per page per model, so a page
done for model A is still attempted for model B).

Split out of the writer on purpose: writing a shard and tracking done-ness are
separate concerns (the old ``ParquetExportAdapter`` fused them).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from ocrscout.io.rows import StageRow
from ocrscout.io.shards import ShardReader


class ResumeMode(Enum):
    OFF = "off"
    PAGE = "page"
    PAGE_MODEL = "page_model"


class ResumeTracker:
    """Membership test for already-written rows under one stage's output."""

    def __init__(self, mode: ResumeMode) -> None:
        self._mode = mode
        self._pages: set[str] = set()
        self._pairs: set[tuple[str, str]] = set()

    @classmethod
    def from_output(
        cls,
        output_dir: Path | str,
        row_type: type[StageRow],
        prefix: str,
        mode: ResumeMode,
    ) -> ResumeTracker:
        tracker = cls(mode)
        if mode is ResumeMode.OFF:
            return tracker
        reader = ShardReader(output_dir, row_type, prefix)
        if mode is ResumeMode.PAGE:
            for rec in reader.project("page_id"):
                pid = rec.get("page_id")
                if pid is not None:
                    tracker._pages.add(str(pid))
        else:
            for rec in reader.project("page_id", "model"):
                pid, model = rec.get("page_id"), rec.get("model")
                if pid is not None and model is not None:
                    tracker._pairs.add((str(pid), str(model)))
        return tracker

    def __bool__(self) -> bool:
        return bool(self._pages or self._pairs)

    def seen(self, page_id: str, model: str | None = None) -> bool:
        if self._mode is ResumeMode.OFF:
            return False
        if self._mode is ResumeMode.PAGE:
            return page_id in self._pages
        return (page_id, model or "") in self._pairs

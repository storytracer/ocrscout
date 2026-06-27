"""PrecomputedLayoutDetector: replay regions from a ``layout-*.parquet``.

The Stage-2/3 seam. ``ocrscout layout`` writes detected regions to
``layout-*.parquet``; this detector reads them back keyed by ``page_id`` and
returns them from :meth:`detect`, so the existing ``layout_chat`` OCR path runs
against precomputed layouts **unchanged** — no CPU detector loaded, no model
weights, no torch. ``ocrscout ocr --layout <layout.parquet>`` swaps the
profile's ``layout_detector`` for this one.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

from ocrscout.errors import ScoutError
from ocrscout.interfaces.layout_detector import LayoutDetector
from ocrscout.types import LayoutRegion, PageImage

log = logging.getLogger(__name__)


class PrecomputedLayoutDetector(LayoutDetector):
    """Returns regions loaded from a ``layout-*.parquet`` by ``page_id``."""

    name: ClassVar[str] = "precomputed"

    def __init__(self, path: str | None = None, **_: Any) -> None:
        if not path:
            raise ScoutError(
                "precomputed layout detector requires a `path` to a "
                "layout-*.parquet (or a dir containing one)."
            )
        self.path = path
        self._by_page: dict[str, list[LayoutRegion]] | None = None

    def warm_up(self) -> None:
        if self._by_page is not None:
            return
        from ocrscout.io.source_parquet import resolve_stage_files

        files = resolve_stage_files(self.path)
        if not files:
            raise ScoutError(
                f"precomputed layout detector: no layout parquet at {self.path!r}."
            )
        import pyarrow.parquet as pq

        by_page: dict[str, list[LayoutRegion]] = {}
        dups = 0
        for f in files:
            names = set(pq.read_schema(str(f)).names)
            if "regions_json" not in names:
                continue
            table = pq.read_table(str(f), columns=["page_id", "regions_json"])
            for pid, rj in zip(
                table.column("page_id").to_pylist(),
                table.column("regions_json").to_pylist(),
                strict=False,
            ):
                if pid is None:
                    continue
                pid = str(pid)
                if pid in by_page:
                    dups += 1  # last-wins, matching the detector "tolerate dups" contract
                by_page[pid] = _parse_regions(rj)
        if dups:
            log.warning(
                "precomputed layout: %d duplicate page_id(s) in %s; kept last",
                dups, self.path,
            )
        log.info("precomputed layout: loaded regions for %d page(s)", len(by_page))
        self._by_page = by_page

    def detect(self, page: PageImage) -> list[LayoutRegion]:
        if self._by_page is None:
            self.warm_up()
        assert self._by_page is not None
        regions = self._by_page.get(page.page_id)
        if regions is None:
            log.warning(
                "precomputed layout: no regions for page_id %r; returning empty",
                page.page_id,
            )
            return []
        return regions


def _parse_regions(regions_json: str | None) -> list[LayoutRegion]:
    if not regions_json:
        return []
    try:
        raw = json.loads(regions_json)
    except (TypeError, ValueError):
        return []
    out: list[LayoutRegion] = []
    for item in raw:
        try:
            out.append(LayoutRegion.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    return out

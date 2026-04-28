"""BhlOcrReferenceAdapter: pull plain-text OCR for BHL pages from S3.

BHL's OCR sidecar files live at
``s3://bhl-open-data/ocr/item-{ItemID:06d}/item-{ItemID:06d}-{PageID:08d}-0000.txt``.

Both ``ItemID`` (the page's ``volume_id``) and ``PageID`` (the page's
``page_id``) are required, which is why ``ReferenceAdapter.get`` takes the
full ``PageImage`` rather than just the page id.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ocrscout.errors import ScoutError
from ocrscout.interfaces.reference import ReferenceAdapter
from ocrscout.types import PageImage, Reference

log = logging.getLogger(__name__)

BHL_BUCKET = "bhl-open-data"
BHL_OCR_PREFIX = f"s3://{BHL_BUCKET}/ocr"


class BhlOcrReferenceAdapter(ReferenceAdapter):
    """Fetch BHL plain-text OCR for a page over anonymous S3, with local cache.

    Args:
        cache_dir: Where to mirror fetched OCR files (default
            ``~/.cache/ocrscout/bhl/ocr``). The cache layout mirrors the
            S3 prefix structure so ``rm -rf`` of a single subdirectory
            wipes one item's OCR cleanly.
        storage_options: fsspec kwargs (default ``{"anon": True}``).
    """

    name = "bhl_ocr"

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        storage_options: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.storage_options = (
            dict(storage_options) if storage_options else {"anon": True}
        )

    def get(self, page: PageImage) -> Reference | None:
        if page.volume_id is None:
            log.debug(
                "bhl_ocr: page %r has no volume_id (ItemID); skipping",
                page.page_id,
            )
            return None
        try:
            item_id = int(page.volume_id)
            page_id = int(page.page_id)
        except (TypeError, ValueError):
            log.warning(
                "bhl_ocr: page_id=%r / volume_id=%r are not integers; skipping",
                page.page_id, page.volume_id,
            )
            return None

        item_dir = f"item-{item_id:06d}"
        # The README's "-0000.txt" suffix is misleading: the trailing 4-digit
        # number is the page's SequenceOrder within the item, not a constant.
        # We populate page.sequence from the BHL catalog, so use it directly.
        # If absent, fall back to listing the OCR directory and matching by
        # the {item_id:06d}-{page_id:08d}- prefix.
        if page.sequence is not None:
            filename = (
                f"item-{item_id:06d}-{page_id:08d}-{page.sequence:04d}.txt"
            )
        else:
            filename = self._discover_filename(item_dir, item_id, page_id)
            if filename is None:
                log.debug(
                    "bhl_ocr: no OCR file in %s matching PageID=%d",
                    item_dir, page_id,
                )
                return None

        url = f"{BHL_OCR_PREFIX}/{item_dir}/{filename}"
        cache_path = self.cache_dir / item_dir / filename

        if cache_path.is_file():
            text = cache_path.read_text(encoding="utf-8", errors="replace")
            return Reference(page_id=page.page_id, text=text)

        try:
            text = _fetch_ocr_text(url, self.storage_options)
        except FileNotFoundError:
            log.debug("bhl_ocr: no OCR at %s", url)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("bhl_ocr: fetch failed for %s: %s", url, e)
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return Reference(page_id=page.page_id, text=text)

    def _discover_filename(
        self, item_dir: str, item_id: int, page_id: int
    ) -> str | None:
        try:
            import s3fs
        except ImportError:
            return None
        prefix = f"{BHL_BUCKET}/ocr/{item_dir}/item-{item_id:06d}-{page_id:08d}-"
        try:
            fs = s3fs.S3FileSystem(**self.storage_options)
            matches = [k for k in fs.ls(f"{BHL_BUCKET}/ocr/{item_dir}")
                       if k.startswith(prefix) and k.endswith(".txt")]
        except Exception as e:  # noqa: BLE001
            log.debug("bhl_ocr: listing failed for %s: %s", item_dir, e)
            return None
        if not matches:
            return None
        return matches[0].rsplit("/", 1)[-1]


def _default_cache_dir() -> Path:
    base = os.environ.get("OCRSCOUT_CACHE_DIR") or os.path.expanduser("~/.cache/ocrscout")
    return Path(base) / "bhl" / "ocr"


def _fetch_ocr_text(url: str, storage_options: dict[str, Any]) -> str:
    try:
        import s3fs
    except ImportError as e:
        raise ScoutError(
            "s3fs is required for the bhl_ocr reference adapter; install via "
            "`pip install ocrscout[bhl]`."
        ) from e
    fs = s3fs.S3FileSystem(**storage_options)
    with fs.open(url, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace")

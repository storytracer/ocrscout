"""PagesSourceAdapter: iterate ``PageImage``s from a stage parquet.

This is the Stage-1 decoupling primitive. ``ocrscout sample`` materializes a
``pages-*.parquet`` of page identity + ``source_uri`` (no image bytes); this
adapter reads that back (or a ``layout-*.parquet`` / ``raw-*.parquet`` /
``train-*.parquet``, all of which carry the same identity columns) and yields
lazy ``PageImage``s with the ``image_loader`` reconstructed from ``source_uri``.

Loader reconstruction is keyed on the URI, not on which source produced the
parquet — there are only two decode paths in the codebase:

* ``.jp2`` → :meth:`BhlSourceAdapter._fetch_and_decode_jp2` (handles S3 + the
  imagecodecs/Pillow JPEG2000 decode);
* everything else → :func:`read_path_or_url` + ``PIL.Image.open`` (local files,
  ``s3://`` / ``gs://`` / ``https://`` / ``hf://`` URLs).

``start_idx`` / ``end_idx`` window the deterministic parquet row order, so
SkyPilot / HF ``apply`` workers take non-overlapping slices of a materialized
sample uniformly for *any* upstream source.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from typing import Any, ClassVar

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.interfaces.source_action import SourceAction
from ocrscout.types import PageImage

log = logging.getLogger(__name__)


class PagesSourceAdapter(SourceAdapter, BaseModel):
    """Yields ``PageImage``s reconstructed from a stage parquet.

    Args:
        path: A ``*.parquet`` file, a directory holding ``data/*-*.parquet``
            stage shards, or a glob. The most upstream stage present
            (``pages`` → ``layout`` → ``raw`` → ``train``) is read.
        storage_options: fsspec storage options for re-fetching images from
            ``source_uri``. Anonymous S3 access is defaulted for ``s3://``.
        start_idx / end_idx: half-open ``[start, end)`` window over parquet
            row order, for partitioning across distributed workers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    name: ClassVar[str] = "pages"
    actions: ClassVar[list[type[SourceAction]]] = []

    path: str
    storage_options: dict[str, Any] | None = None
    start_idx: int | None = Field(default=None, ge=0)
    end_idx: int | None = Field(default=None, ge=0)

    def iter_pages(self) -> Iterator[PageImage]:
        from ocrscout.io.source_parquet import read_stage_rows, resolve_stage_files

        files = resolve_stage_files(self.path)
        if not files:
            raise ScoutError(
                f"pages source: no stage parquet found at {self.path!r} "
                "(expected a *.parquet file, or a dir with "
                "data/pages-*.parquet / layout-*.parquet / raw-*.parquet)."
            )
        rows = read_stage_rows(files)
        if self.start_idx is not None or self.end_idx is not None:
            lo = self.start_idx or 0
            hi = self.end_idx if self.end_idx is not None else len(rows)
            rows = rows[lo:hi]

        for row in rows:
            page = self._row_to_page(row)
            if page is not None:
                yield page

    def _row_to_page(self, row: dict[str, Any]) -> PageImage | None:
        return page_from_row(row, storage_options=self.storage_options)


def page_from_row(
    row: dict[str, Any], *, storage_options: dict[str, Any] | None = None
) -> PageImage | None:
    """Reconstruct a lazy ``PageImage`` from a stage-parquet row dict.

    Returns ``None`` (with a warning) when ``source_uri`` is missing — such a
    row's image can't be re-fetched. Used by :class:`PagesSourceAdapter` and by
    the ``ocrscout normalize`` stage (which needs the same reconstruction to
    feed normalizers that look at the image).
    """
    source_uri = row.get("source_uri")
    if not source_uri:
        log.warning(
            "pages source: row %r has no source_uri; cannot reconstruct "
            "image loader, skipping.",
            row.get("page_id"),
        )
        return None
    opts = dict(storage_options) if storage_options else {}
    if source_uri.startswith("s3://"):
        opts.setdefault("anon", True)
    return PageImage(
        page_id=str(row["page_id"]),
        file_id=str(row.get("file_id") or row["page_id"]),
        image_loader=_make_loader(source_uri, opts),
        source_uri=source_uri,
        width=int(row.get("width") or 0),
        height=int(row.get("height") or 0),
        dpi=row.get("dpi"),
        barcode=row.get("barcode"),
        sequence=row.get("sequence"),
        extra=row.get("extra") or {},
    )


def _make_loader(source_uri: str, storage_options: dict[str, Any]):
    """Build a zero-arg ``image_loader`` closure for one ``source_uri``."""
    if source_uri.lower().endswith((".jp2", ".j2k", ".jpx")):
        opts = storage_options

        def _from_jp2() -> Image.Image:
            # Reuse BHL's JPEG2000 decoder (imagecodecs → Pillow fallback);
            # it returns an RGB PIL image. Importing it does not require S3
            # credentials — those are only needed at fetch time.
            from ocrscout.sources.bhl import BhlSourceAdapter

            image, _dpi = BhlSourceAdapter._fetch_and_decode_jp2(source_uri, opts)
            return image

        return _from_jp2

    opts = storage_options

    def _from_uri() -> Image.Image:
        from ocrscout.sources.hf_dataset import read_path_or_url

        data = read_path_or_url(source_uri, opts)
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            img = src.copy()
        return img.convert("RGB") if img.mode != "RGB" else img

    return _from_uri

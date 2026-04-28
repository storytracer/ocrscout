"""BhlSourceAdapter: sample images from the Biodiversity Heritage Library.

BHL's open-data S3 bucket (``s3://bhl-open-data/``) holds 305K volumes and
67M pages — far too large to enumerate. Instead, we read its small TSV
catalogs (``data/{item,page,title}.txt.gz``), join them with a
pre-classified copyright lookup hosted on HuggingFace, apply
rights/language/year filters, rank volumes (and pages within each)
deterministically by ``hash(id || seed)``, then fill ``sample_n`` pages
volume-by-volume up to ``pages_per_volume`` per volume. The JPEG-2000
image bytes are pulled only for the chosen pages. The DuckDB query
streams the gzipped TSVs directly from disk, so RAM stays small even
with the full ``page.txt.gz`` (~67M rows) in scope.

Image URLs follow ``s3://bhl-open-data/images/{BarCode}/{BarCode}_{NNNN}.jp2``;
OCR URLs follow ``s3://bhl-open-data/ocr/item-{ItemID:06d}/item-{ItemID:06d}-{PageID:08d}-0000.txt``
(the OCR adapter is in :mod:`ocrscout.references.bhl_ocr`).
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.types import PageImage, Volume

log = logging.getLogger(__name__)

BHL_BUCKET = "bhl-open-data"
BHL_DATA_PREFIX = f"s3://{BHL_BUCKET}/data"
BHL_IMAGES_PREFIX = f"s3://{BHL_BUCKET}/images"
BHL_WEB_HTTPS_PREFIX = f"https://{BHL_BUCKET}.s3.amazonaws.com/web"

# Pattern for the canonical BHL JP2 source_uri produced by this adapter.
# Anchored on BHL_IMAGES_PREFIX so it stays in lockstep if the prefix moves.
_BHL_JP2_RE = re.compile(
    rf"^{re.escape(BHL_IMAGES_PREFIX)}/(?P<barcode>[^/]+)/(?P=barcode)_(?P<seq>\d{{4}})\.jp2$"
)
_BHL_WEB_SIZES = ("thumb", "small", "medium", "large", "full")


def bhl_web_image_url(source_uri: str, size: str = "full") -> str | None:
    """Map a BHL JP2 ``source_uri`` to its pre-converted WebP HTTPS URL.

    BHL publishes web-friendly WebPs alongside the archival JP2s at
    ``s3://bhl-open-data/web/{barcode}/{barcode}_{seq:04d}_{size}.webp`` for
    sizes ``thumb`` / ``small`` / ``medium`` / ``large`` / ``full``. They're
    publicly readable over HTTPS, so the viewer can hand the URL straight to
    Gradio without auth. Returns ``None`` for any input that doesn't match
    the BHL JP2 form, so non-BHL parquets sail through untouched.
    """
    if size not in _BHL_WEB_SIZES:
        raise ValueError(
            f"unknown BHL web size {size!r}; expected one of {_BHL_WEB_SIZES}"
        )
    if not source_uri:
        return None
    m = _BHL_JP2_RE.match(source_uri)
    if not m:
        return None
    barcode = m.group("barcode")
    seq = m.group("seq")
    return f"{BHL_WEB_HTTPS_PREFIX}/{barcode}/{barcode}_{seq}_{size}.webp"


CATALOG_FILES: tuple[str, ...] = ("item.txt.gz", "page.txt.gz", "title.txt.gz")
COPYRIGHT_DATASET = "storytracer/bhl_copyright_statuses_classified"
COPYRIGHT_PARQUET_PATH = "data/train-00000-of-00001.parquet"


class BhlSourceAdapter(SourceAdapter):
    """Sample pages from the BHL public S3 bucket via its TSV catalogs.

    **Image-numbering convention.** BHL's S3 image filenames use the form
    ``{BarCode}_{SequenceOrder:04d}.jp2``, with matching OCR sidecars at
    ``{ItemID:06d}-{PageID:08d}-{SequenceOrder:04d}.txt``. BHL's README
    notes a historical inconsistency where some items' first image was
    ``_0000.jp2`` instead of ``_0001.jp2``, but empirically that case is
    vanishingly rare (0/100 random items in our last sweep). This adapter
    just trusts the modern convention universally; on the off-chance a
    truly-legacy item slips through, the image fetch 404s and the page
    is cleanly skipped with a warning.
    """

    name = "bhl"

    def __init__(
        self,
        *,
        sample: int,
        seed: int = 42,
        rights: str = "public_domain",
        languages: list[str] | None = None,
        year_range: tuple[int, int] | list[int] | None = None,
        pages_per_volume: int = 8,
        volumes: int | None = None,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
        storage_options: dict[str, Any] | None = None,
        copyright_dataset: str = COPYRIGHT_DATASET,
        copyright_parquet_path: str = COPYRIGHT_PARQUET_PATH,
        **_ignored: Any,
    ) -> None:
        if "max_pages_per_volume" in _ignored:
            raise ScoutError(
                "max_pages_per_volume was renamed to pages_per_volume "
                "(it is now a target distribution, not just a cap)."
            )
        if sample <= 0:
            raise ScoutError("BhlSourceAdapter requires sample >= 1")
        if pages_per_volume <= 0:
            raise ScoutError("BhlSourceAdapter requires pages_per_volume >= 1")
        if volumes is not None:
            if volumes <= 0:
                raise ScoutError("BhlSourceAdapter requires volumes >= 1")
            if volumes * pages_per_volume < sample:
                raise ScoutError(
                    f"volumes={volumes} * pages_per_volume={pages_per_volume} "
                    f"= {volumes * pages_per_volume} < sample={sample}; "
                    "raise volumes or pages_per_volume, or lower sample."
                )
        self.sample_n = int(sample)
        self.seed = int(seed)
        self.rights = str(rights)
        # BHL stores language codes in uppercase (ENG, GER, LAT) but ISO
        # 639-2 codes are conventionally written lowercase elsewhere; accept
        # either and normalize.
        self.languages = (
            [str(x).upper() for x in languages] if languages else None
        )
        if year_range is not None:
            yr = tuple(year_range)
            if len(yr) != 2:
                raise ScoutError(
                    f"year_range must be a 2-tuple [min, max]; got {year_range!r}"
                )
            self.year_range: tuple[int, int] | None = (int(yr[0]), int(yr[1]))
        else:
            self.year_range = None
        self.pages_per_volume = int(pages_per_volume)
        self.volumes = int(volumes) if volumes is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.force_refresh = bool(force_refresh)
        self.storage_options = (
            dict(storage_options) if storage_options else {"anon": True}
        )
        self.copyright_dataset = copyright_dataset
        self.copyright_parquet_path = copyright_parquet_path

        self._sample_rows: list[dict[str, Any]] | None = None
        self._volumes: list[Volume] | None = None

    # --- public ABC ---------------------------------------------------------

    def iter_pages(self) -> Iterator[PageImage]:
        self._ensure_query_run()
        assert self._sample_rows is not None
        for row in self._sample_rows:
            page = self._row_to_page(row)
            if page is not None:
                yield page

    def iter_volumes(self) -> Iterator[Volume]:
        self._ensure_query_run()
        assert self._volumes is not None
        yield from self._volumes

    # --- internals ----------------------------------------------------------

    def _ensure_query_run(self) -> None:
        if self._sample_rows is not None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        catalog_paths = {
            name: self._ensure_catalog_file(name) for name in CATALOG_FILES
        }
        copyright_path = self._ensure_copyright_lookup()
        rows = _run_duckdb_sample(
            item_path=catalog_paths["item.txt.gz"],
            page_path=catalog_paths["page.txt.gz"],
            title_path=catalog_paths["title.txt.gz"],
            copyright_path=copyright_path,
            rights=self.rights,
            languages=self.languages,
            year_range=self.year_range,
            pages_per_volume=self.pages_per_volume,
            volumes_n=self.volumes,
            sample_n=self.sample_n,
            seed=self.seed,
        )
        self._sample_rows = rows
        self._volumes = _rows_to_volumes(rows)
        if self.volumes is not None and len(rows) < self.sample_n:
            log.warning(
                "BHL sample: only %d page(s) available across the %d "
                "requested volume(s); requested %d.",
                len(rows), self.volumes, self.sample_n,
            )
        log.info(
            "BHL sample: %d page(s) across %d volume(s) "
            "(rights=%s, languages=%s, year_range=%s, "
            "pages_per_volume=%d, volumes=%s, seed=%d)",
            len(rows), len(self._volumes), self.rights, self.languages,
            self.year_range, self.pages_per_volume,
            self.volumes if self.volumes is not None else "auto",
            self.seed,
        )

    def _ensure_catalog_file(self, basename: str) -> Path:
        local = self.cache_dir / basename
        url = f"{BHL_DATA_PREFIX}/{basename}"
        etag_file = local.with_suffix(local.suffix + ".etag")
        if local.is_file() and not self.force_refresh:
            remote_etag = _s3_etag(url, self.storage_options)
            cached_etag = etag_file.read_text().strip() if etag_file.is_file() else None
            if remote_etag and cached_etag == remote_etag:
                log.debug("BHL catalog %s is fresh (etag=%s)", basename, remote_etag)
                return local
            log.info(
                "BHL catalog %s is stale (cached etag=%s, remote=%s); refreshing",
                basename, cached_etag, remote_etag,
            )
        log.info("Downloading BHL catalog %s -> %s", url, local)
        _download_s3_file(url, local, self.storage_options)
        remote_etag = _s3_etag(url, self.storage_options)
        if remote_etag:
            etag_file.write_text(remote_etag)
        return local

    def _ensure_copyright_lookup(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - core dep
            raise ScoutError(
                "huggingface_hub is required for the BHL adapter's "
                "copyright lookup; install it via `pip install huggingface-hub`."
            ) from e
        local = hf_hub_download(
            repo_id=self.copyright_dataset,
            filename=self.copyright_parquet_path,
            repo_type="dataset",
            force_download=self.force_refresh,
        )
        return Path(local)

    def _row_to_page(self, row: dict[str, Any]) -> PageImage | None:
        item_id = row.get("ItemID")
        page_id = row.get("PageID")
        bar_code = row.get("BarCode")
        if not (item_id and page_id and bar_code):
            log.warning("BHL row missing identifiers, skipping: %r", row)
            return None
        sequence = _parse_int(row.get("SequenceOrder"))
        if sequence is None:
            log.warning(
                "BHL row missing SequenceOrder for PageID=%s; skipping", page_id,
            )
            return None
        bar_code_s = str(bar_code)
        # Modern-convention image URL: SequenceOrder is the file suffix.
        # See class docstring for the legacy-item caveat.
        url = (
            f"{BHL_IMAGES_PREFIX}/{bar_code_s}/{bar_code_s}_{sequence:04d}.jp2"
        )
        try:
            image, dpi = _fetch_and_decode_jp2(url, self.storage_options)
        except FileNotFoundError:
            log.warning(
                "BHL image %s not found (PageID=%s, SO=%d); skipping. "
                "Likely a phantom catalog entry (foldout/insert without "
                "scan) or a legacy 0-indexed item.",
                url, page_id, sequence,
            )
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("BHL image fetch/decode failed for %s: %s", url, e)
            return None
        width, height = image.size
        filename = f"{bar_code_s}_{sequence:04d}.jp2"
        return PageImage(
            page_id=str(page_id),
            file_id=f"{bar_code_s}/{filename}",
            image=image,
            width=width,
            height=height,
            dpi=dpi,
            source_uri=url,
            barcode=bar_code_s,
            sequence=sequence,
            extra={
                "BarCode": bar_code_s,
                "ItemID": str(item_id),
                "PageID": str(page_id),
                "PageTypeName": row.get("PageTypeName"),
                "PagePrefix": row.get("PagePrefix"),
                "PageNumber": row.get("PageNumber"),
            },
        )


# --- helpers ----------------------------------------------------------------


def _default_cache_dir() -> Path:
    base = os.environ.get("OCRSCOUT_CACHE_DIR") or os.path.expanduser("~/.cache/ocrscout")
    return Path(base) / "bhl"


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"\\N", "NULL"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _s3_etag(url: str, storage_options: dict[str, Any]) -> str | None:
    """Best-effort fetch of an S3 object's ETag for cache invalidation."""
    try:
        import s3fs
    except ImportError:
        log.debug("s3fs not installed; skipping etag check for %s", url)
        return None
    try:
        fs = s3fs.S3FileSystem(**storage_options)
        info = fs.info(url)
        etag = info.get("ETag") or info.get("etag")
        return str(etag).strip('"') if etag else None
    except Exception as e:  # noqa: BLE001
        log.debug("etag lookup failed for %s: %s", url, e)
        return None


def _download_s3_file(url: str, dest: Path, storage_options: dict[str, Any]) -> None:
    try:
        import s3fs
    except ImportError as e:
        raise ScoutError(
            "s3fs is required for the BHL adapter; install via "
            "`pip install ocrscout[bhl]`."
        ) from e
    fs = s3fs.S3FileSystem(**storage_options)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with fs.open(url, "rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp.replace(dest)


def _fetch_and_decode_jp2(
    url: str, storage_options: dict[str, Any]
) -> tuple[Image.Image, int | None]:
    """Fetch JP2 bytes from S3 and decode to a PIL Image.

    Decoder preference: ``imagecodecs`` (ships its own libopenjp2 in the
    wheel, so no system dep) → Pillow's built-in JP2 plugin (works when
    Pillow was built against libopenjp2). Either path returns the image
    in RGB mode for downstream backends.
    """
    try:
        import s3fs
    except ImportError as e:
        raise ScoutError(
            "s3fs is required for the BHL adapter; install via "
            "`pip install ocrscout[bhl]`."
        ) from e
    fs = s3fs.S3FileSystem(**storage_options)
    with fs.open(url, "rb") as f:
        data = f.read()

    try:
        import imagecodecs

        arr = imagecodecs.jpeg2k_decode(data)
        image = Image.fromarray(arr)
    except ImportError:
        image = Image.open(io.BytesIO(data))
        image.load()
    if image.mode != "RGB":
        image = image.convert("RGB")
    dpi: int | None = None
    info_dpi = getattr(image, "info", {}).get("dpi")
    if isinstance(info_dpi, tuple) and info_dpi:
        try:
            dpi = int(round(float(info_dpi[0])))
        except (TypeError, ValueError):
            dpi = None
    return image, dpi


def _run_duckdb_sample(
    *,
    item_path: Path,
    page_path: Path,
    title_path: Path,
    copyright_path: Path,
    rights: str,
    languages: list[str] | None,
    year_range: tuple[int, int] | None,
    pages_per_volume: int,
    volumes_n: int | None,
    sample_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the catalog join + volume-grouped fill under DuckDB.

    Strategy: rank every eligible volume by ``hash(ItemID || seed)``, rank
    pages within each volume the same way, then fill ``sample_n`` rows
    ordered by ``(vrank, prank)`` after capping per-volume contributions
    at ``pages_per_volume``. The first volume fills its quota, then the
    next, and so on — so a ``sample_n=40`` with ``pages_per_volume=4``
    naturally lands in ~10 volumes (more if some are thin). Pass
    ``volumes_n`` to hard-cap the candidate volume pool first.

    All BHL TSV columns are kept as VARCHAR up front (BHL writes ``\\N``
    for null, which DuckDB's strict schema inference otherwise trips over).
    Year filtering extracts the first 4-digit run from ``item.Year`` —
    BHL uses freeform date strings.
    """
    try:
        import duckdb
    except ImportError as e:
        raise ScoutError(
            "duckdb is required for the BHL adapter; install via "
            "`pip install ocrscout[bhl]`."
        ) from e

    conn = duckdb.connect(":memory:")

    def _read_tsv(path: Path) -> str:
        # BHL TSVs have no quoting and use the literal token \N for nulls.
        # Disabling quote/escape stops DuckDB from greedily consuming text
        # across newlines on a stray quote character.
        return (
            f"read_csv('{path}', delim='\\t', header=true, "
            "all_varchar=true, nullstr='\\N', quote='', escape='')"
        )

    where_clauses = ["c.classification = ?", "c.parsing_success = TRUE"]
    params: list[Any] = [rights]
    if languages:
        placeholders = ", ".join("?" for _ in languages)
        where_clauses.append(f"t.LanguageCode IN ({placeholders})")
        params.extend(languages)
    if year_range is not None:
        where_clauses.append(
            "TRY_CAST(REGEXP_EXTRACT(i.Year, '\\d{4}', 0) AS INTEGER) "
            "BETWEEN ? AND ?"
        )
        params.extend([year_range[0], year_range[1]])
    where_sql = " AND ".join(where_clauses)

    volumes_limit_sql = f"LIMIT {volumes_n}" if volumes_n is not None else ""

    sql = f"""
    WITH eligible AS (
        SELECT
            i.ItemID,
            i.BarCode,
            i.TitleID,
            i.CopyrightStatus,
            t.LanguageCode AS Language,
            COALESCE(t.FullTitle, t.ShortTitle) AS Title,
            t.MARCBibID,
            t.TL2Author,
            i.Year AS ItemYear,
            TRY_CAST(REGEXP_EXTRACT(i.Year, '\\d{{4}}', 0) AS INTEGER) AS Year,
            c.classification AS Rights
        FROM {_read_tsv(item_path)} AS i
        LEFT JOIN {_read_tsv(title_path)} AS t USING (TitleID)
        JOIN read_parquet('{copyright_path}') AS c
          ON c.CopyrightStatus = i.CopyrightStatus
        WHERE {where_sql}
    ),
    all_pages AS (
        SELECT
            p.PageID,
            p.ItemID,
            p.SequenceOrder,
            p.PageTypeName,
            p.PagePrefix,
            p.PageNumber,
            e.BarCode,
            e.TitleID,
            e.Title,
            e.TL2Author,
            e.Language,
            e.Year,
            e.Rights
        FROM {_read_tsv(page_path)} AS p
        JOIN eligible e USING (ItemID)
    ),
    volume_pool AS (
        SELECT
            ItemID,
            hash(ItemID || CAST({seed} AS VARCHAR)) AS vrank
        FROM all_pages
        GROUP BY ItemID
    ),
    selected_volumes AS (
        SELECT ItemID, vrank
        FROM volume_pool
        ORDER BY vrank
        {volumes_limit_sql}
    ),
    ranked AS (
        SELECT
            ap.*,
            sv.vrank,
            ROW_NUMBER() OVER (
                PARTITION BY ap.ItemID
                ORDER BY hash(ap.PageID || CAST({seed} AS VARCHAR))
            ) AS prank
        FROM all_pages ap
        JOIN selected_volumes sv USING (ItemID)
    ),
    capped AS (
        SELECT * FROM ranked WHERE prank <= {pages_per_volume}
    )
    SELECT *
    FROM capped
    ORDER BY vrank, prank
    LIMIT {sample_n};
    """
    log.debug("BHL DuckDB sample SQL params=%r", params)
    cursor = conn.execute(sql, params)
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, r, strict=False)) for r in cursor.fetchall()]
    return rows


def _rows_to_volumes(rows: list[dict[str, Any]]) -> list[Volume]:
    """Deduplicate the sampled rows into one ``Volume`` per ``ItemID``."""
    by_item: dict[str, dict[str, Any]] = {}
    page_counts: dict[str, int] = {}
    for row in rows:
        item_id = row.get("ItemID")
        if not item_id:
            continue
        item_id = str(item_id)
        page_counts[item_id] = page_counts.get(item_id, 0) + 1
        if item_id in by_item:
            continue
        by_item[item_id] = row
    volumes: list[Volume] = []
    for item_id, row in by_item.items():
        year = _parse_int(row.get("Year"))
        title = row.get("Title")
        bar_code = row.get("BarCode")
        title_id = row.get("TitleID")
        author = (row.get("TL2Author") or "").strip()
        volumes.append(
            Volume(
                barcode=str(bar_code) if bar_code else item_id,
                title=title or None,
                creators=[author] if author else [],
                language=row.get("Language") or None,
                year=year,
                rights=row.get("Rights") or None,
                page_count=page_counts.get(item_id),
                source_uri=f"https://www.biodiversitylibrary.org/item/{item_id}",
                extra={
                    "BarCode": str(bar_code) if bar_code else None,
                    "TitleID": str(title_id) if title_id else None,
                    "CopyrightStatus": row.get("CopyrightStatus"),
                    "ItemYear": row.get("ItemYear"),
                },
            )
        )
    return volumes

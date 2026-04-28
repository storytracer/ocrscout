"""BhlSourceAdapter: sample images from the Biodiversity Heritage Library.

BHL's open-data S3 bucket (``s3://bhl-open-data/``) holds 305K volumes and
67M pages — far too large to enumerate. Instead, we read its small TSV
catalogs (``data/{item,page,title}.txt.gz``), join them with a
pre-classified copyright lookup hosted on HuggingFace, apply
rights/language/year filters, cap pages per volume, reservoir-sample the
result, and only then pull the JPEG-2000 image bytes for the chosen pages.
The DuckDB query streams the gzipped TSVs directly from disk, so RAM
stays small even with the full ``page.txt.gz`` (~67M rows) in scope.

Image URLs follow ``s3://bhl-open-data/images/{BarCode}/{BarCode}_{NNNN}.jp2``;
OCR URLs follow ``s3://bhl-open-data/ocr/item-{ItemID:06d}/item-{ItemID:06d}-{PageID:08d}-0000.txt``
(the OCR adapter is in :mod:`ocrscout.references.bhl_ocr`).
"""

from __future__ import annotations

import io
import logging
import os
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
        max_pages_per_volume: int = 8,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
        storage_options: dict[str, Any] | None = None,
        copyright_dataset: str = COPYRIGHT_DATASET,
        copyright_parquet_path: str = COPYRIGHT_PARQUET_PATH,
        **_ignored: Any,
    ) -> None:
        if sample <= 0:
            raise ScoutError("BhlSourceAdapter requires sample >= 1")
        if max_pages_per_volume <= 0:
            raise ScoutError("BhlSourceAdapter requires max_pages_per_volume >= 1")
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
        self.max_pages_per_volume = int(max_pages_per_volume)
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
            max_pages_per_volume=self.max_pages_per_volume,
            sample_n=self.sample_n,
            seed=self.seed,
        )
        self._sample_rows = rows
        self._volumes = _rows_to_volumes(rows)
        log.info(
            "BHL sample: %d page(s) across %d volume(s) "
            "(rights=%s, languages=%s, year_range=%s, max_per_volume=%d, seed=%d)",
            len(rows), len(self._volumes), self.rights, self.languages,
            self.year_range, self.max_pages_per_volume, self.seed,
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
            file_id=f"{item_id}/{filename}",
            image=image,
            width=width,
            height=height,
            dpi=dpi,
            source_uri=url,
            volume_id=str(item_id),
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
    max_pages_per_volume: int,
    sample_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the catalog join + per-volume cap + reservoir sample under DuckDB.

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
    ranked AS (
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
            e.Rights,
            ROW_NUMBER() OVER (
                PARTITION BY p.ItemID
                ORDER BY hash(p.PageID || CAST({seed} AS VARCHAR))
            ) AS rn
        FROM {_read_tsv(page_path)} AS p
        JOIN eligible e USING (ItemID)
    ),
    -- Materialize the per-volume cap into its own CTE so the reservoir
    -- sample below only sees already-capped rows. Inlining the cap into
    -- the same SELECT as USING SAMPLE causes DuckDB to apply the WHERE
    -- after the sample (with the window column re-evaluated against the
    -- sampled subset), and the result drains to zero.
    capped AS (
        SELECT * FROM ranked WHERE rn <= {max_pages_per_volume}
    )
    SELECT *
    FROM capped
    USING SAMPLE reservoir({sample_n} ROWS) REPEATABLE({seed});
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
                volume_id=item_id,
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

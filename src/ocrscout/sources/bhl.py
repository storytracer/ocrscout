"""BhlSourceAdapter: sample images from the Biodiversity Heritage Library.

BHL's open-data S3 bucket holds 305K volumes and 67M pages — far too large
to enumerate. Instead, this adapter reads BHL's small TSV catalogs
(``data/{item,page,title}.txt.gz``) and joins them with a *user-configured*
rights-classification dataset (a parquet of unique
``CopyrightStatus`` / ``RightsStatement`` / ``LicenseType`` / ``RightsHolder``
combinations classified as ``public_domain`` vs ``not_public_domain``)
hosted on the HF Hub. The combined query ranks volumes and pages
deterministically by ``hash(id || seed)`` and fills ``sample`` pages
volume-by-volume up to ``pages_per_volume`` per volume. JPEG-2000 image
bytes are pulled only for the chosen pages.

No HF dataset id is hardcoded in this module. The user configures the
rights-classification dataset once via ``ocrscout source bhl setup
--read-from <dataset>`` (or by re-running ``refresh`` in a future PR);
the choice is persisted in ``~/.ocrscout/sources/bhl/info.yaml`` and
resolved at adapter-construction time. Image URLs follow
``s3://bhl-open-data/images/{BarCode}/{BarCode}_{NNNN}.jp2``; OCR sidecars
live under ``s3://bhl-open-data/ocr/item-{ItemID:06d}/`` (read by the
companion adapter in :mod:`ocrscout.references.bhl_ocr`).
"""

from __future__ import annotations

import io
import logging
import re
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.interfaces.source_action import SourceAction, SourceActionContext
from ocrscout.sources._info import SourceInfo, load_info
from ocrscout.sources._paths import source_dir
from ocrscout.types import PageImage, Volume

log = logging.getLogger(__name__)


class BhlSourceAdapter(SourceAdapter, BaseModel):
    """Sample pages from the BHL public S3 bucket via its TSV catalogs.

    **Image-numbering convention.** BHL's S3 image filenames use the form
    ``{BarCode}_{SequenceOrder:04d}.jp2``, with matching OCR sidecars at
    ``{ItemID:06d}-{PageID:08d}-{SequenceOrder:04d}.txt``. BHL's README
    notes a historical inconsistency where some items' first image was
    ``_0000.jp2`` instead of ``_0001.jp2``, but empirically that case is
    vanishingly rare (0/100 random items in our last sweep). This adapter
    trusts the modern convention universally; if a truly-legacy item
    slips through, the image fetch 404s and the page is cleanly skipped
    with a warning.

    **Rights configuration.** Sampling joins the TSVs against a
    rights-classification parquet hosted on HF Hub. The dataset id is
    *not* hardcoded; configure it once with
    ``ocrscout source bhl setup --read-from <dataset>`` (or refresh, in a
    future PR), or pass ``copyright_dataset=<dataset>`` explicitly for a
    one-off run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # --- identity (registry contract) ---
    name: ClassVar[str] = "bhl"

    # --- structural constants (S3 layout, regex, JOIN spec) ---
    BUCKET: ClassVar[str] = "bhl-open-data"
    DATA_PREFIX: ClassVar[str] = "s3://bhl-open-data/data"
    IMAGES_PREFIX: ClassVar[str] = "s3://bhl-open-data/images"
    WEB_HTTPS_PREFIX: ClassVar[str] = "https://bhl-open-data.s3.amazonaws.com/web"
    JP2_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        rf"^{re.escape(IMAGES_PREFIX)}/(?P<barcode>[^/]+)/(?P=barcode)_(?P<seq>\d{{4}})\.jp2$"
    )
    WEB_SIZES: ClassVar[tuple[str, ...]] = ("thumb", "small", "medium", "large", "full")
    CATALOG_FILES: ClassVar[tuple[str, ...]] = (
        "item.txt.gz",
        "page.txt.gz",
        "title.txt.gz",
    )
    COPYRIGHT_PARQUET_PATH: ClassVar[str] = "data/train-00000-of-00001.parquet"
    COPYRIGHT_JOIN_COLUMNS: ClassVar[tuple[str, ...]] = (
        "CopyrightStatus",
        "RightsStatement",
        "LicenseType",
        "RightsHolder",
    )
    # NOTE: deliberately NO `CANONICAL_*` constant naming any HF dataset.

    # --- validated user-facing fields ---
    sample: int = Field(gt=0, description="Number of pages to sample.")
    seed: int = 42
    rights: str = "public_domain"
    languages: list[str] | None = None
    year_range: tuple[int, int] | None = None
    pages_per_volume: int = Field(default=8, gt=0)
    volumes: int | None = Field(default=None, gt=0)
    concurrent_fetches: int = Field(default=16, gt=0)
    cache_dir: Path | None = None
    force_refresh: bool = False  # superseded by `refresh` verb in PR 4; PR 5 removes.
    storage_options: dict[str, Any] = Field(default_factory=lambda: {"anon": True})
    copyright_dataset: str | None = None
    copyright_parquet_path: str = COPYRIGHT_PARQUET_PATH
    start_idx: int | None = Field(default=None, ge=0)
    end_idx: int | None = Field(default=None, ge=0)

    # --- private runtime state (not validated, not persisted) ---
    _cache_dir: Path = PrivateAttr()
    _copyright_dataset: str = PrivateAttr()
    _sample_rows: list[dict[str, Any]] | None = PrivateAttr(default=None)
    _volumes_cached: list[Volume] | None = PrivateAttr(default=None)

    # --- validators ---

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed_kwargs(cls, values: Any) -> Any:
        if isinstance(values, dict) and "max_pages_per_volume" in values:
            raise ValueError(
                "max_pages_per_volume was renamed to pages_per_volume "
                "(it is now a target distribution, not just a cap)."
            )
        return values

    @field_validator("languages")
    @classmethod
    def _uppercase_languages(cls, v: list[str] | None) -> list[str] | None:
        # BHL stores ENG/GER/LAT uppercase; ISO 639-2 is conventionally
        # lowercase. Accept either and normalise.
        return [str(x).upper() for x in v] if v else None

    @model_validator(mode="after")
    def _check_cross_field_constraints(self) -> "BhlSourceAdapter":
        if self.volumes is not None and self.volumes * self.pages_per_volume < self.sample:
            raise ValueError(
                f"volumes={self.volumes} * pages_per_volume={self.pages_per_volume} "
                f"= {self.volumes * self.pages_per_volume} < sample={self.sample}; "
                "raise volumes or pages_per_volume, or lower sample."
            )
        if (
            self.start_idx is not None
            and self.end_idx is not None
            and self.end_idx < self.start_idx
        ):
            raise ValueError(
                f"end_idx ({self.end_idx}) must be >= start_idx ({self.start_idx})"
            )
        return self

    @model_validator(mode="after")
    def _resolve_runtime_state(self) -> "BhlSourceAdapter":
        self._cache_dir = (
            self.cache_dir if self.cache_dir is not None
            else source_dir(self.name) / "catalog"
        )
        if self.copyright_dataset is not None:
            self._copyright_dataset = self.copyright_dataset
        else:
            info = load_info(self.name)
            if info is not None and info.rights.read_from:
                self._copyright_dataset = info.rights.read_from
            else:
                raise ScoutError(
                    f"{type(self).__name__} has no rights-classification dataset "
                    f"configured. Run "
                    f"`ocrscout source {self.name} setup --read-from <dataset>` "
                    f"to point at an existing rights-classification dataset, or "
                    f"`ocrscout source {self.name} refresh --runner local|hf "
                    f"--source-repo <repo> --output-repo <repo>` "
                    f"to maintain your own. "
                    f"Constructor escape hatch: pass `copyright_dataset=<dataset>` explicitly."
                )
        return self

    # --- ABC contract ---

    def iter_pages(self) -> Iterator[PageImage]:
        self._ensure_query_run()
        assert self._sample_rows is not None
        rows = self._sample_rows
        if self.start_idx is not None or self.end_idx is not None:
            lo = self.start_idx if self.start_idx is not None else 0
            hi = self.end_idx if self.end_idx is not None else len(rows)
            rows = rows[lo:hi]
        if self.concurrent_fetches == 1 or len(rows) <= 1:
            for row in rows:
                page = self._row_to_page(row)
                if page is not None:
                    yield page
            return
        # _row_to_page is a network-bound S3 GET + JP2 decode; threading is
        # dramatically faster than the sequential loop. Pages are yielded in
        # completion order — downstream OCR doesn't care about ordering.
        workers = min(self.concurrent_fetches, len(rows))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._row_to_page, row) for row in rows]
            for fut in as_completed(futures):
                page = fut.result()
                if page is not None:
                    yield page

    def iter_volumes(self) -> Iterator[Volume]:
        self._ensure_query_run()
        assert self._volumes_cached is not None
        yield from self._volumes_cached

    # --- public classmethod helpers ---

    @classmethod
    def web_image_url(cls, source_uri: str, size: str = "full") -> str | None:
        """Map a BHL JP2 ``source_uri`` to its pre-converted WebP HTTPS URL.

        BHL publishes web-friendly WebPs alongside the archival JP2s at
        ``s3://bhl-open-data/web/{barcode}/{barcode}_{seq:04d}_{size}.webp``
        for sizes ``thumb`` / ``small`` / ``medium`` / ``large`` / ``full``.
        They're publicly readable over HTTPS, so the viewer can hand the
        URL straight to Gradio without auth. Returns ``None`` for any
        input that doesn't match the BHL JP2 form, so non-BHL parquets
        sail through untouched.
        """
        if size not in cls.WEB_SIZES:
            raise ValueError(
                f"unknown BHL web size {size!r}; expected one of {cls.WEB_SIZES}"
            )
        if not source_uri:
            return None
        m = cls.JP2_PATTERN.match(source_uri)
        if not m:
            return None
        barcode = m.group("barcode")
        seq = m.group("seq")
        return f"{cls.WEB_HTTPS_PREFIX}/{barcode}/{barcode}_{seq}_{size}.webp"

    # --- query / I/O internals ---

    def _ensure_query_run(self) -> None:
        if self._sample_rows is not None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        catalog_paths = {
            name: self._ensure_catalog_file(name) for name in self.CATALOG_FILES
        }
        copyright_path = self._ensure_copyright_lookup()
        rows = self._run_duckdb_sample(
            item_path=catalog_paths["item.txt.gz"],
            page_path=catalog_paths["page.txt.gz"],
            title_path=catalog_paths["title.txt.gz"],
            copyright_path=copyright_path,
            rights=self.rights,
            languages=self.languages,
            year_range=self.year_range,
            pages_per_volume=self.pages_per_volume,
            volumes_n=self.volumes,
            sample_n=self.sample,
            seed=self.seed,
        )
        self._sample_rows = rows
        self._volumes_cached = self._rows_to_volumes(rows)
        if self.volumes is not None and len(rows) < self.sample:
            log.warning(
                "BHL sample: only %d page(s) available across the %d "
                "requested volume(s); requested %d.",
                len(rows), self.volumes, self.sample,
            )
        log.info(
            "BHL sample: %d page(s) across %d volume(s) "
            "(rights=%s, languages=%s, year_range=%s, "
            "pages_per_volume=%d, volumes=%s, seed=%d)",
            len(rows), len(self._volumes_cached), self.rights, self.languages,
            self.year_range, self.pages_per_volume,
            self.volumes if self.volumes is not None else "auto",
            self.seed,
        )

    def _ensure_catalog_file(self, basename: str) -> Path:
        local = self._cache_dir / basename
        url = f"{self.DATA_PREFIX}/{basename}"
        etag_file = local.with_suffix(local.suffix + ".etag")
        if local.is_file() and not self.force_refresh:
            remote_etag = self._s3_etag(url, self.storage_options)
            cached_etag = (
                etag_file.read_text().strip() if etag_file.is_file() else None
            )
            if remote_etag and cached_etag == remote_etag:
                log.debug("BHL catalog %s is fresh (etag=%s)", basename, remote_etag)
                return local
            log.info(
                "BHL catalog %s is stale (cached etag=%s, remote=%s); refreshing",
                basename, cached_etag, remote_etag,
            )
        log.info("Downloading BHL catalog %s -> %s", url, local)
        self._download_s3_file(url, local, self.storage_options)
        remote_etag = self._s3_etag(url, self.storage_options)
        if remote_etag:
            etag_file.write_text(remote_etag)
        return local

    def _ensure_copyright_lookup(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - core dep
            raise ScoutError(
                "huggingface_hub is required for the BHL adapter's "
                "rights lookup; install it via `pip install huggingface-hub`."
            ) from e
        local = hf_hub_download(
            repo_id=self._copyright_dataset,
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
        url = f"{self.IMAGES_PREFIX}/{bar_code_s}/{bar_code_s}_{sequence:04d}.jp2"
        try:
            image, dpi = self._fetch_and_decode_jp2(url, self.storage_options)
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

    # --- S3 + decoding statics ---

    @staticmethod
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

    @staticmethod
    def _download_s3_file(
        url: str, dest: Path, storage_options: dict[str, Any]
    ) -> None:
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

    @staticmethod
    def _fetch_and_decode_jp2(
        url: str, storage_options: dict[str, Any]
    ) -> tuple[Image.Image, int | None]:
        """Fetch JP2 bytes from S3 and decode to a PIL Image.

        Decoder preference: ``imagecodecs`` (ships its own libopenjp2 in the
        wheel, no system dep) → Pillow's built-in JP2 plugin. Either path
        returns the image in RGB mode for downstream backends.
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

    # --- DuckDB query / row reshaping ---

    @classmethod
    def _run_duckdb_sample(
        cls,
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

        Strategy: rank every eligible volume by ``hash(ItemID || seed)``,
        rank pages within each volume the same way, then fill ``sample_n``
        rows ordered by ``(vrank, prank)`` after capping per-volume
        contributions at ``pages_per_volume``. The first volume fills its
        quota, then the next, and so on — ``sample_n=40`` with
        ``pages_per_volume=4`` naturally lands in ~10 volumes (more if some
        are thin). Pass ``volumes_n`` to hard-cap the candidate volume
        pool first.

        All BHL TSV columns are kept as VARCHAR up front (BHL writes ``\\N``
        for null, which DuckDB's strict schema inference otherwise trips
        over). Year filtering extracts the first 4-digit run from
        ``item.Year`` — BHL uses freeform date strings. The 4-field rights
        JOIN uses ``NULLIF(.., '') IS NOT DISTINCT FROM`` to normalise the
        classified parquet's empty-string convention against the TSV's
        ``\\N``→NULL convention.
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

        rights_join_sql = " AND ".join(
            f"NULLIF(c.{col}, '') IS NOT DISTINCT FROM NULLIF(i.{col}, '')"
            for col in cls.COPYRIGHT_JOIN_COLUMNS
        )

        volumes_limit_sql = f"LIMIT {volumes_n}" if volumes_n is not None else ""

        sql = f"""
        WITH eligible AS (
            SELECT
                i.ItemID,
                i.BarCode,
                i.TitleID,
                i.CopyrightStatus,
                i.RightsStatement,
                i.LicenseType,
                i.RightsHolder,
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
              ON {rights_join_sql}
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
                e.Rights,
                e.CopyrightStatus,
                e.RightsStatement,
                e.LicenseType,
                e.RightsHolder,
                e.ItemYear
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

    @staticmethod
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
                        "RightsStatement": row.get("RightsStatement"),
                        "LicenseType": row.get("LicenseType"),
                        "RightsHolder": row.get("RightsHolder"),
                        "ItemYear": row.get("ItemYear"),
                    },
                )
            )
        return volumes

    @staticmethod
    def _verify_dataset_reachable(repo_id: str) -> None:
        """Best-effort existence check on the HF Hub.

        Raises :class:`ScoutError` for clear 404s. Network / auth failures
        log a warning and return — admin verbs shouldn't be gated on a
        flaky uplink.
        """
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import RepositoryNotFoundError
        except ImportError:
            log.debug(
                "huggingface_hub not installed; skipping reachability check"
            )
            return
        try:
            HfApi().dataset_info(repo_id)
        except RepositoryNotFoundError as e:
            raise ScoutError(
                f"HF dataset {repo_id!r} not found (404). Check the repo id "
                "or pass a different value."
            ) from e
        except Exception as e:  # noqa: BLE001
            log.warning(
                "could not validate %s on HF Hub (%s); proceeding anyway",
                repo_id, e,
            )

    # --- inline actions ---

    class Setup(SourceAction):
        """Record where to read the rights-classified parquet from.

        Lightweight first-run path: users who only consume an existing
        rights-classification dataset point ``--read-from`` at it and
        skip the heavy extract/classify/publish pipeline (which lives in
        ``refresh``, landing in PR 4).
        """

        name: ClassVar[str] = "setup"
        description: ClassVar[str] = (
            "Record where to read rights classification from."
        )

        class Flags(BaseModel):
            read_from: str = Field(
                ...,
                description=(
                    "HF dataset id of a rights-classification dataset "
                    "(four-field combo → public_domain/not_public_domain)."
                ),
            )
            output_repo: str | None = Field(
                None,
                description=(
                    "Where future `refresh` invocations will publish your "
                    "own classified version. Recorded for convenience."
                ),
            )

        flags: ClassVar[type[BaseModel]] = Flags

        def fill_defaults(
            self,
            flags: "BhlSourceAdapter.Setup.Flags",
            info: SourceInfo,
        ) -> "BhlSourceAdapter.Setup.Flags":
            # output_repo is sticky: don't make the user repeat a prior choice.
            if flags.output_repo is None and info.rights.output_repo:
                flags.output_repo = info.rights.output_repo
            return flags

        def run(
            self,
            flags: "BhlSourceAdapter.Setup.Flags",
            ctx: SourceActionContext,
        ) -> dict[str, dict] | None:
            BhlSourceAdapter._verify_dataset_reachable(flags.read_from)
            log.info(
                "[bhl] setup: read_from=%s, output_repo=%s",
                flags.read_from,
                flags.output_repo or "—",
            )
            # Setup only records pointers. It deliberately does NOT touch
            # rights.last_refresh / last_runner / combos_classified —
            # those belong to the (heavier) `refresh` action.
            patch: dict[str, str | None] = {"read_from": flags.read_from}
            if flags.output_repo is not None:
                patch["output_repo"] = flags.output_repo
            return {"rights": patch}

    # PR 4 will add `class Refresh(SourceAction): …` here.

    actions: ClassVar[list[type[SourceAction]]] = [Setup]


# --- module-local utility (not a class concern) -----------------------------


def _parse_int(value: Any) -> int | None:
    """Tolerant int parse for BHL's stringly-typed catalog values.

    Returns ``None`` for blanks, the literal token ``\\N``, ``NULL``, or
    any unparseable input. Kept module-local because it's a generic
    string utility, not a BHL concept.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"\\N", "NULL"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None

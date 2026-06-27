"""HfDatasetSourceAdapter: iterate images from any source HF ``datasets`` understands.

Handles three input shapes through one code path:

* HF Hub repo IDs — ``"org/dataset"`` with optional ``subset`` and ``split``.
* fsspec URLs — ``s3://``, ``gs://``, ``https://``, ``hf://`` etc., resolved
  via the ``imagefolder`` builder. Anonymous S3 access is the default
  (BHL, Common Crawl, and most "open data" buckets work without creds).
* Local directories — same ``imagefolder`` path. Replaces the previous
  ``LocalSourceAdapter``.
"""

from __future__ import annotations

import io
import logging
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from PIL import Image
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.interfaces.source_action import SourceAction
from ocrscout.types import PageImage

log = logging.getLogger(__name__)


class HfDatasetSourceAdapter(SourceAdapter, BaseModel):
    """Yields one ``PageImage`` per row of a HuggingFace ``datasets`` Dataset.

    Args:
        path: HF Hub repo id (``"org/dataset"``), fsspec URL
            (``"s3://bucket/prefix/"``, ``"https://..."``), or a local
            directory. URLs and directories load through the
            ``imagefolder`` builder; repo ids load directly.
        split: Dataset split to iterate. Defaults to ``"train"``.
        subset: HF Hub config name (``load_dataset(repo, name=subset)``).
            Ignored for filesystem/URL paths.
        image_column: Override the auto-detected image column. The default
            picks the first column whose feature is an ``Image``.
        id_column: Override the auto-detected id column. Default tries
            ``id``, ``page_id``, ``name``, ``filename``, then falls back
            to the row's source path (when known) and finally a row index.
        streaming: Iterate without materialising the dataset. Defaults to
            ``True`` for fsspec URLs (avoids downloading the whole prefix
            up front, and preserves original paths as ``source_uri``) and
            ``False`` for local dirs and HF Hub repo IDs. Streaming mode
            precludes ``len()``.
        storage_options: Passed through to ``datasets.load_dataset``. For
            anonymous S3 access pass ``{"anon": True}`` (the default when
            ``path`` starts with ``s3://`` and no options are supplied).
        revision: HF Hub revision (commit / branch / tag). Hub paths only.
        sample: When set, yield a random subset of this size.
        seed: RNG seed for ``sample``. ``None`` means a different random
            subset every run; pin to an integer for reproducible scouts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    # --- identity ---
    name: ClassVar[str] = "hf_dataset"

    # --- structural constants ---
    HF_REPO_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    FSSPEC_SCHEMES: ClassVar[frozenset[str]] = frozenset(
        {"s3", "gs", "gcs", "az", "abfs", "abfss", "http", "https", "hf", "file"}
    )
    IMAGE_EXTS: ClassVar[frozenset[str]] = frozenset({
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp",
        ".bmp", ".gif", ".jp2", ".j2k", ".jpx",
    })
    ID_COLUMN_CANDIDATES: ClassVar[tuple[str, ...]] = (
        "page_id", "id", "name", "filename", "file_name",
    )

    # --- validated user-facing fields ---
    path: str
    split: str = "train"
    subset: str | None = None
    image_column: str | None = None
    id_column: str | None = None
    streaming: bool | None = None
    storage_options: dict[str, Any] | None = None
    revision: str | None = None
    sample: int | None = None
    seed: int | None = None

    # --- private runtime state ---
    _dataset: Any | None = PrivateAttr(default=None)

    # --- validators ---

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "HfDatasetSourceAdapter":
        # Auto: stream when the source is a fsspec URL (preserves original
        # paths as source_uri and avoids materialising the whole prefix).
        if self.streaming is None:
            self.streaming = "://" in self.path
        # Anonymous S3 access is the default for s3:// sources. Default it
        # even when storage_options is already a dict (e.g. an empty {} from a
        # serialized config round-trip) — otherwise s3fs builds a *signed*
        # client and a public anonymous bucket fails with NoCredentialsError.
        # An explicit `anon: False` still wins via setdefault.
        if self._scheme(self.path) == "s3":
            if self.storage_options is None:
                self.storage_options = {"anon": True}
            else:
                self.storage_options.setdefault("anon", True)
        return self

    # --- ABC contract ---

    def iter_pages(self) -> Iterator[PageImage]:
        """Yield lazy ``PageImage``s for every row of the HF dataset.

        We cast the image column to undecoded so we get the raw bytes (or
        a path to them) without datasets eagerly decoding the entire
        prefix into PIL Images. The decode itself is deferred into the
        installed ``image_loader`` closure — backends invoke it via
        ``page.open_image()`` and drop the bytes when the with-block
        exits. Width / height are unknown at iter time; backends that
        need them after open_image() see the backfilled values.
        """
        ds = self._build()
        image_col = self.image_column or self._detect_image_column(ds.features)
        id_col = self.id_column or self._detect_id_column(
            ds.features, exclude=image_col
        )

        from datasets import Image as HfImage

        ds = ds.cast_column(image_col, HfImage(decode=False))
        storage_options = dict(self.storage_options) if self.storage_options else {}

        for idx, row in enumerate(ds):
            record = row[image_col]
            raw_path = record.get("path") if isinstance(record, dict) else None
            raw_bytes = record.get("bytes") if isinstance(record, dict) else None
            if raw_bytes is None and raw_path is None:
                log.warning(
                    "hf_dataset: row %d has neither bytes nor path; skipping",
                    idx,
                )
                continue

            page_id = self._page_id(row, id_col, raw_path, idx)
            file_id = self._file_id(raw_path, page_id)
            yield PageImage(
                page_id=page_id,
                file_id=file_id,
                image_loader=_make_hf_loader(raw_bytes, raw_path, storage_options),
                source_uri=raw_path,
            )

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError("hf_dataset(streaming=True) does not support len()")
        ds = self._build()
        try:
            return len(ds)
        except TypeError as e:
            raise TypeError(f"underlying dataset is not sized: {e}") from e

    # --- query / I/O internals ---

    def _build(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover — datasets is a core dep
            raise ScoutError(f"datasets is required for hf_dataset: {e}") from e

        kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.storage_options:
            kwargs["storage_options"] = self.storage_options

        if self._is_hf_repo_id(self.path):
            log.debug(
                "hf_dataset: loading hub repo %r split=%r", self.path, self.split
            )
            if self.subset is not None:
                kwargs["name"] = self.subset
            if self.revision is not None:
                kwargs["revision"] = self.revision
            ds = load_dataset(self.path, **kwargs)
            if self.sample is not None:
                ds = self._sample_hub_dataset(ds, self.sample, self.seed)
        elif "://" in self.path:
            if self.sample is not None:
                files = self._list_image_files_fsspec(self.path, self.storage_options)
                files = self._random_subset(files, self.sample, self.seed)
                log.debug(
                    "hf_dataset: sampled %d remote files (seed=%r) from %r",
                    len(files), self.seed, self.path,
                )
                ds = load_dataset("imagefolder", data_files=files, **kwargs)
            else:
                glob = self.path.rstrip("/") + "/**"
                log.debug("hf_dataset: loading imagefolder data_files=%r", glob)
                ds = load_dataset("imagefolder", data_files=glob, **kwargs)
        else:
            data_dir = str(Path(self.path).expanduser())
            if self.sample is not None:
                files = self._list_image_files_local(data_dir)
                files = self._random_subset(files, self.sample, self.seed)
                log.debug(
                    "hf_dataset: sampled %d local files (seed=%r) from %r",
                    len(files), self.seed, data_dir,
                )
                ds = load_dataset("imagefolder", data_files=files, **kwargs)
            else:
                log.debug("hf_dataset: loading imagefolder data_dir=%r", data_dir)
                ds = load_dataset("imagefolder", data_dir=data_dir, **kwargs)

        self._dataset = ds
        return ds

    # --- path / id construction ---

    @staticmethod
    def _page_id(
        row: dict[str, Any],
        id_col: str | None,
        raw_path: str | None,
        idx: int,
    ) -> str:
        if id_col is not None:
            value = row.get(id_col)
            if value not in (None, ""):
                return str(value)
        if raw_path:
            stem = Path(urlparse(raw_path).path or raw_path).name
            if stem:
                return stem
        return f"row_{idx:06d}"

    def _parent_dir(self) -> str:
        """The grouping segment of file_id — basename of the source root.

        - HF Hub ``org/dataset`` → ``dataset``
        - fsspec URL ``s3://bucket/folder/`` → ``folder`` (or ``bucket`` if no path)
        - Local dir ``/path/to/scans/`` → ``scans``
        """
        if self._is_hf_repo_id(self.path):
            return self.path.split("/", 1)[1]
        if "://" in self.path:
            parsed = urlparse(self.path)
            path_part = (parsed.path or "").rstrip("/")
            if path_part:
                return Path(path_part).name
            return parsed.netloc or self.path
        return Path(self.path).expanduser().name or self.path

    def _file_id(self, raw_path: str | None, page_id: str) -> str:
        if raw_path:
            filename = Path(urlparse(raw_path).path or raw_path).name or page_id
        else:
            filename = page_id
        return f"{self._parent_dir()}/{filename}"

    # --- classmethod / staticmethod helpers (was module-level) ---

    @classmethod
    def _is_hf_repo_id(cls, path: str) -> bool:
        if "://" in path or path.startswith(("/", ".", "~")):
            return False
        return bool(cls.HF_REPO_PATTERN.match(path))

    @staticmethod
    def _scheme(path: str) -> str | None:
        parsed = urlparse(path)
        return parsed.scheme.lower() or None if parsed.scheme else None

    @classmethod
    def _list_image_files_fsspec(
        cls, url: str, storage_options: dict[str, Any] | None
    ) -> list[str]:
        """List image files under a fsspec URL prefix, returning full URLs.

        Uses fsspec's ``find`` (recursive listing). The bytes are not
        fetched — we only need the file paths so we can sample without
        downloading.
        """
        try:
            import fsspec
        except ImportError as e:
            raise ScoutError(
                f"hf_dataset: fsspec needed to list {url!r}: {e}. "
                "Install the relevant extra (e.g. `pip install ocrscout[cloud]` for S3)."
            ) from e

        parsed = urlparse(url)
        fs, _ = fsspec.core.url_to_fs(url, **(storage_options or {}))
        prefix = (parsed.netloc + parsed.path).rstrip("/")
        listing = fs.find(prefix)
        scheme = parsed.scheme
        return [
            f"{scheme}://{p}"
            for p in listing
            if Path(p).suffix.lower() in cls.IMAGE_EXTS
        ]

    @classmethod
    def _list_image_files_local(cls, data_dir: str) -> list[str]:
        root = Path(data_dir).expanduser()
        if not root.exists():
            raise ScoutError(f"hf_dataset: source path does not exist: {root}")
        if root.is_file():
            return [str(root)]
        return [
            str(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in cls.IMAGE_EXTS
        ]

    @staticmethod
    def _random_subset(items: list[str], k: int, seed: int | None) -> list[str]:
        if not items:
            raise ScoutError("hf_dataset: no image files found to sample from.")
        rng = random.Random(seed)
        return rng.sample(items, k=min(k, len(items)))

    @staticmethod
    def _sample_hub_dataset(ds: Any, k: int, seed: int | None) -> Any:
        """Random subset of an HF Hub Dataset / IterableDataset.

        For a sized ``Dataset`` we shuffle indices and ``select(range(k))``
        — no extra downloads. For an ``IterableDataset`` (streaming) we
        use ``shuffle(buffer_size=...)`` then ``take(k)``; this fills a
        buffer of ``max(k, 1024)`` items first (a known cost of streaming
        shuffle).
        """
        try:
            n = len(ds)
        except TypeError:
            buffer_size = max(k, 1024)
            return ds.shuffle(seed=seed, buffer_size=buffer_size).take(k)
        return ds.shuffle(seed=seed).select(range(min(k, n)))

    @staticmethod
    def _detect_image_column(features: Any) -> str:
        from datasets import Image as HfImage

        for name, feature in features.items():
            if isinstance(feature, HfImage):
                return name
        raise ScoutError(
            f"hf_dataset: no Image-typed column found in features "
            f"{list(features)!r}; pass image_column=... to disambiguate."
        )

    @classmethod
    def _detect_id_column(cls, features: Any, *, exclude: str) -> str | None:
        for candidate in cls.ID_COLUMN_CANDIDATES:
            if candidate in features and candidate != exclude:
                return candidate
        return None

    @staticmethod
    def _dpi(img: Image.Image) -> int | None:
        info = img.info.get("dpi")
        if info is None:
            return None
        if isinstance(info, (tuple, list)) and info:
            try:
                return int(round(float(info[0])))
            except (TypeError, ValueError):
                return None
        try:
            return int(round(float(info)))
        except (TypeError, ValueError):
            return None

    # --- no source-specific admin verbs ---
    actions: ClassVar[list[type[SourceAction]]] = []


def _make_hf_loader(
    raw_bytes: bytes | None,
    raw_path: str | None,
    storage_options: dict[str, Any],
):
    """Build the image_loader closure for one HF dataset row.

    Two cases by data shape:
    - Local imagefolder rows have a ``path`` — the closure re-reads from
      disk each call, so the orchestrator never holds raw bytes between
      iter time and load time. Cheap; the OS page cache absorbs the
      re-read.
    - Streaming rows (and rows where datasets inlined the bytes for some
      other reason) have bytes only. The closure has no choice but to
      capture them; lazy loading still avoids the decoded-RGB resident
      cost, which is the larger share for most formats.
    """
    if raw_path is not None and "://" not in raw_path:
        local_path = raw_path

        def _from_path() -> Image.Image:
            with Image.open(local_path) as src:
                src.load()
                img = src.copy()
            return img.convert("RGB") if img.mode != "RGB" else img

        return _from_path

    if raw_path is not None:
        remote_path = raw_path
        opts = storage_options

        def _from_url() -> Image.Image:
            data = read_path_or_url(remote_path, opts)
            with Image.open(io.BytesIO(data)) as src:
                src.load()
                img = src.copy()
            return img.convert("RGB") if img.mode != "RGB" else img

        return _from_url

    captured = raw_bytes

    def _from_bytes() -> Image.Image:
        with Image.open(io.BytesIO(captured)) as src:
            src.load()
            img = src.copy()
        return img.convert("RGB") if img.mode != "RGB" else img

    return _from_bytes


def read_path_or_url(
    path: str, storage_options: dict[str, Any] | None = None
) -> bytes:
    """Fetch raw bytes from a local path or any fsspec-recognised URL.

    Used by :class:`HfDatasetSourceAdapter` (when ``datasets`` hands us a
    path without inlined bytes — typical in streaming mode) and by the
    publish pipeline (when bundling source images into a dataset repo).
    Local paths skip the fsspec import entirely.

    Kept as a module-level public function because the publish pipeline
    ([src/ocrscout/publish/dataset.py](src/ocrscout/publish/dataset.py))
    imports it directly. Promoting to a classmethod would force that
    caller to drag in the whole adapter class for one utility.
    """
    if "://" not in path:
        return Path(path).read_bytes()
    try:
        import fsspec
    except ImportError as e:
        raise ScoutError(
            f"fsspec needed to fetch {path!r}: {e}. "
            "Install the relevant extra (e.g. `pip install ocrscout[cloud]` for S3)."
        ) from e
    with fsspec.open(path, "rb", **(storage_options or {})) as f:
        return f.read()

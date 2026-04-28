"""HfDatasetSourceAdapter: iterate images from any source HF ``datasets`` understands.

Handles three input shapes through one code path:

* HF Hub repo IDs — ``"org/dataset"`` with optional ``subset`` and ``split``.
* fsspec URLs — ``s3://``, ``gs://``, ``https://``, ``hf://`` etc., resolved
  via the ``imagefolder`` builder. Anonymous S3 access is the default (BHL,
  Common Crawl, and most "open data" buckets work without creds).
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
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.types import PageImage

log = logging.getLogger(__name__)

# Matches ``org/dataset`` HF Hub repo IDs. A path with more than one slash
# (or a leading dot/slash) is a filesystem path, not a repo id.
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FSSPEC_SCHEMES = {"s3", "gs", "gcs", "az", "abfs", "abfss", "http", "https", "hf", "file"}

# Image extensions ``imagefolder`` understands. Used when we pre-list files
# ourselves for sampling — ``load_dataset("imagefolder", data_files=[...])``
# expects an explicit set of image-suffixed paths, not a mixed listing.
_IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp",
    ".bmp", ".gif", ".jp2", ".j2k", ".jpx",
})


def _is_hf_repo_id(path: str) -> bool:
    if "://" in path or path.startswith(("/", ".", "~")):
        return False
    return bool(_HF_REPO_RE.match(path))


def _scheme(path: str) -> str | None:
    parsed = urlparse(path)
    return parsed.scheme.lower() or None if parsed.scheme else None


class HfDatasetSourceAdapter(SourceAdapter):
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
            ``id``, ``page_id``, ``name``, ``filename``, then falls back to
            the row's source path (when known) and finally a row index.
        streaming: Iterate without materialising the dataset. Defaults to
            ``True`` for fsspec URLs (avoids downloading the whole prefix
            up front, and preserves original paths as ``source_uri``) and
            ``False`` for local dirs and HF Hub repo IDs. Streaming mode
            precludes ``len()``.
        storage_options: Passed through to ``datasets.load_dataset``. For
            anonymous S3 access pass ``{"anon": True}`` (the default when
            ``path`` starts with ``s3://`` and no options are supplied).
        revision: HF Hub revision (commit / branch / tag). Hub paths only.
        sample: When set, yield a random subset of this size. For fsspec
            URLs and local dirs, files are pre-listed cheaply (fsspec
            ``find`` / ``rglob``) and a random ``k``-subset is selected
            *before* any bytes are fetched — so over S3 only ``sample``
            files are downloaded, not the whole prefix. For HF Hub repo
            IDs, ``Dataset.shuffle(seed).select(range(sample))`` is used
            (cost: index permutation, no extra downloads).
        seed: RNG seed for ``sample``. ``None`` means a different random
            subset every run; pin to an integer for reproducible scouts.
    """

    name = "hf_dataset"

    def __init__(
        self,
        path: str,
        *,
        split: str = "train",
        subset: str | None = None,
        image_column: str | None = None,
        id_column: str | None = None,
        streaming: bool | None = None,
        storage_options: dict[str, Any] | None = None,
        revision: str | None = None,
        sample: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.path = path
        self.split = split
        self.subset = subset
        self.image_column = image_column
        self.id_column = id_column
        # Auto: stream when the source is a fsspec URL (preserves original
        # paths as source_uri and avoids materialising the whole prefix).
        self.streaming = streaming if streaming is not None else ("://" in path)
        self.revision = revision
        self.sample = sample
        self.seed = seed

        if storage_options is None and _scheme(path) == "s3":
            storage_options = {"anon": True}
        self.storage_options = storage_options

        self._dataset: Any | None = None  # cached after first build

    # ------------------------------------------------------------------ public

    def iter_pages(self) -> Iterator[PageImage]:
        ds = self._build()
        image_col = self.image_column or _detect_image_column(ds.features)
        id_col = self.id_column or _detect_id_column(ds.features, exclude=image_col)

        # Cast image column to undecoded so we get {'bytes', 'path'} dicts —
        # we want the path to derive a stable page_id and source_uri, and we
        # control PIL decoding ourselves to keep behaviour identical to the
        # old LocalSourceAdapter (and to avoid datasets' image autocrop).
        from datasets import Image as HfImage

        ds = ds.cast_column(image_col, HfImage(decode=False))

        for idx, row in enumerate(ds):
            record = row[image_col]
            raw_path = record.get("path") if isinstance(record, dict) else None
            raw_bytes = record.get("bytes") if isinstance(record, dict) else None
            if raw_bytes is None and raw_path:
                raw_bytes = _read_path(raw_path, self.storage_options)
            if not raw_bytes:
                log.warning(
                    "hf_dataset: row %d has no image bytes (path=%r); skipping",
                    idx, raw_path,
                )
                continue

            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.load()
                page_image = img.copy()
                w, h = page_image.size
                dpi = _dpi(img)

            page_id = self._page_id(row, id_col, raw_path, idx)
            yield PageImage(
                page_id=page_id,
                image=page_image,
                width=w,
                height=h,
                dpi=dpi,
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

    # ----------------------------------------------------------------- helpers

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

        if _is_hf_repo_id(self.path):
            log.debug("hf_dataset: loading hub repo %r split=%r", self.path, self.split)
            if self.subset is not None:
                kwargs["name"] = self.subset
            if self.revision is not None:
                kwargs["revision"] = self.revision
            ds = load_dataset(self.path, **kwargs)
            if self.sample is not None:
                ds = _sample_hub_dataset(ds, self.sample, self.seed)
        elif "://" in self.path:
            if self.sample is not None:
                files = _list_image_files_fsspec(self.path, self.storage_options)
                files = _random_subset(files, self.sample, self.seed)
                log.debug(
                    "hf_dataset: sampled %d remote files (seed=%r) from %r",
                    len(files), self.seed, self.path,
                )
                ds = load_dataset("imagefolder", data_files=files, **kwargs)
            else:
                # ``imagefolder``'s ``data_dir`` is a local-only argument (it
                # joins onto the cwd); fsspec URLs go through ``data_files``
                # with a glob. ``**`` walks all subdirs; ``imagefolder`` then
                # filters by image extension internally.
                glob = self.path.rstrip("/") + "/**"
                log.debug("hf_dataset: loading imagefolder data_files=%r", glob)
                ds = load_dataset("imagefolder", data_files=glob, **kwargs)
        else:
            data_dir = str(Path(self.path).expanduser())
            if self.sample is not None:
                files = _list_image_files_local(data_dir)
                files = _random_subset(files, self.sample, self.seed)
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

    @staticmethod
    def _page_id(row: dict[str, Any], id_col: str | None, raw_path: str | None, idx: int) -> str:
        if id_col is not None:
            value = row.get(id_col)
            if value not in (None, ""):
                return str(value)
        if raw_path:
            stem = Path(urlparse(raw_path).path or raw_path).name
            if stem:
                return stem
        return f"row_{idx:06d}"


# --------------------------------------------------------------------- module helpers


def _list_image_files_fsspec(url: str, storage_options: dict[str, Any] | None) -> list[str]:
    """List all image files under a fsspec URL prefix, returning full URLs.

    Uses fsspec's ``find`` (recursive listing). The bytes are not fetched —
    we only need the file paths so we can sample without downloading.
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
        if Path(p).suffix.lower() in _IMAGE_EXTS
    ]


def _list_image_files_local(data_dir: str) -> list[str]:
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise ScoutError(f"hf_dataset: source path does not exist: {root}")
    if root.is_file():
        return [str(root)]
    return [
        str(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]


def _random_subset(items: list[str], k: int, seed: int | None) -> list[str]:
    if not items:
        raise ScoutError("hf_dataset: no image files found to sample from.")
    rng = random.Random(seed)
    return rng.sample(items, k=min(k, len(items)))


def _sample_hub_dataset(ds: Any, k: int, seed: int | None) -> Any:
    """Random subset of an HF Hub Dataset / IterableDataset.

    For a sized ``Dataset`` we shuffle indices and ``select(range(k))`` —
    no extra downloads. For an ``IterableDataset`` (streaming) we use
    ``shuffle(buffer_size=...)`` then ``take(k)``; this fills a buffer of
    ``max(k, 1024)`` items first (a known cost of streaming shuffle).
    """
    try:
        n = len(ds)
    except TypeError:
        # IterableDataset — shuffle within a buffer, then take.
        buffer_size = max(k, 1024)
        return ds.shuffle(seed=seed, buffer_size=buffer_size).take(k)
    return ds.shuffle(seed=seed).select(range(min(k, n)))


def _detect_image_column(features: Any) -> str:
    from datasets import Image as HfImage

    for name, feature in features.items():
        if isinstance(feature, HfImage):
            return name
    raise ScoutError(
        f"hf_dataset: no Image-typed column found in features {list(features)!r}; "
        "pass image_column=... to disambiguate."
    )


def _detect_id_column(features: Any, *, exclude: str) -> str | None:
    for candidate in ("page_id", "id", "name", "filename", "file_name"):
        if candidate in features and candidate != exclude:
            return candidate
    return None


def _read_path(path: str, storage_options: dict[str, Any] | None) -> bytes:
    """Fetch raw bytes from a path/URL via fsspec when datasets hands us a path
    without inlined bytes (typical in streaming mode)."""
    if "://" not in path:
        return Path(path).read_bytes()
    try:
        import fsspec
    except ImportError as e:
        raise ScoutError(
            f"hf_dataset: fsspec needed to fetch {path!r}: {e}. "
            "Install the relevant extra (e.g. `pip install ocrscout[cloud]` for S3)."
        ) from e
    with fsspec.open(path, "rb", **(storage_options or {})) as f:
        return f.read()


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

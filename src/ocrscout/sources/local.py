"""LocalSourceAdapter: iterate images from a directory on disk."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from ocrscout.errors import ScoutError
from ocrscout.interfaces.source import SourceAdapter
from ocrscout.types import PageImage

# Pillow reads .jp2/.j2k via the JPEG 2000 plugin (OpenJPEG); commonly available.
_SUPPORTED_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".jp2", ".j2k", ".jpx"}
)


class LocalSourceAdapter(SourceAdapter):
    """Yields one ``PageImage`` per image file under ``path``.

    PDFs are intentionally not supported in v0 — install the ``pdf`` extra and
    rasterize them upstream (or use a future PDF source adapter).
    """

    name = "local"

    def __init__(self, path: str | Path, *, recursive: bool = True) -> None:
        self.root = Path(path).expanduser()
        self.recursive = recursive

    def _iter_files(self) -> Iterator[Path]:
        if not self.root.exists():
            raise ScoutError(f"source path does not exist: {self.root}")
        if self.root.is_file():
            yield self.root
            return
        glob = self.root.rglob("*") if self.recursive else self.root.iterdir()
        for p in sorted(glob):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES:
                yield p

    def iter_pages(self) -> Iterator[PageImage]:
        for path in self._iter_files():
            with Image.open(path) as img:
                img.load()
                w, h = img.size
                page_id = str(path.relative_to(self.root)) if self.root.is_dir() else path.name
                yield PageImage(
                    page_id=page_id,
                    image=img.copy(),
                    width=w,
                    height=h,
                    dpi=_dpi(img),
                    source_uri=str(path),
                )

    def __len__(self) -> int:
        return sum(1 for _ in self._iter_files())


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

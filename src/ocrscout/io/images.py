"""Reconstruct a lazy image loader from a ``source_uri``.

Stage parquets carry only the image *reference* (``source_uri``), never bytes.
When a downstream stage needs the pixels, it rebuilds a zero-arg loader keyed on
the URI — JPEG2000 (``.jp2``/``.j2k``/``.jpx``) via ``imagecodecs`` (its bundled
OpenJPEG, no system lib), everything else via Pillow over fsspec. Self-contained
so the IO layer doesn't depend on any source adapter.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from ocrscout.errors import ScoutError

_JP2_SUFFIXES = (".jp2", ".j2k", ".jpx")


def read_bytes(uri: str, storage_options: dict[str, Any] | None = None) -> bytes:
    """Fetch raw bytes from a local path or any fsspec URL (anon S3 default)."""
    if "://" not in uri:
        return Path(uri).read_bytes()
    try:
        import fsspec
    except ImportError as e:  # pragma: no cover
        raise ScoutError(f"fsspec needed to fetch {uri!r}: {e}") from e
    opts = dict(storage_options or {})
    if uri.startswith("s3://"):
        opts.setdefault("anon", True)
    with fsspec.open(uri, "rb", **opts) as f:
        return f.read()


def decode_image(data: bytes, uri: str) -> Image.Image:
    """Decode image bytes to an RGB PIL image, picking the JP2 path by URI."""
    if uri.lower().endswith(_JP2_SUFFIXES):
        try:
            import imagecodecs

            img = Image.fromarray(imagecodecs.jpeg2k_decode(data))
        except ImportError:
            img = Image.open(io.BytesIO(data))
            img.load()
    else:
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            img = src.copy()
    return img.convert("RGB") if img.mode != "RGB" else img


def loader_for_uri(
    uri: str, storage_options: dict[str, Any] | None = None
) -> Callable[[], Image.Image]:
    """Build a zero-arg ``image_loader`` closure for one ``source_uri``."""
    opts = dict(storage_options) if storage_options else {}

    def _load() -> Image.Image:
        return decode_image(read_bytes(uri, opts), uri)

    return _load

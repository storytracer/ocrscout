"""LocalSourceAdapter walks an image directory."""

from __future__ import annotations

from pathlib import Path

from ocrscout.sources.local import LocalSourceAdapter


def test_iterates_pngs(images_dir: Path) -> None:
    src = LocalSourceAdapter(images_dir)
    pages = list(src.iter_pages())
    assert len(pages) == 2
    for p in pages:
        assert p.width == 100 and p.height == 80
        assert p.source_uri is not None and p.source_uri.endswith(".png")


def test_len_matches_iter(images_dir: Path) -> None:
    src = LocalSourceAdapter(images_dir)
    assert len(src) == sum(1 for _ in src.iter_pages())

"""Normalizers: convert raw model outputs to DoclingDocument."""

from ocrscout.normalizers.doctags import DocTagsNormalizer
from ocrscout.normalizers.layout_json import LayoutJsonNormalizer
from ocrscout.normalizers.markdown import MarkdownNormalizer
from ocrscout.normalizers.passthrough import PassthroughNormalizer

__all__ = [
    "DocTagsNormalizer",
    "LayoutJsonNormalizer",
    "MarkdownNormalizer",
    "PassthroughNormalizer",
]

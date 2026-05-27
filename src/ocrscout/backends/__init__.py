"""Model backends: prepare and run OCR model invocations."""

from ocrscout.backends.layout_chat import LayoutChatBackend
from ocrscout.backends.litellm import LiteLLMBackend
from ocrscout.backends.tesseract import TesseractBackend

__all__ = [
    "LayoutChatBackend",
    "LiteLLMBackend",
    "TesseractBackend",
]

"""Model backends: prepare and run OCR model invocations."""

from ocrscout.backends.docling import DoclingBackend
from ocrscout.backends.openai_api import OpenAIApiBackend
from ocrscout.backends.tesseract import TesseractBackend
from ocrscout.backends.vllm import VllmBackend

__all__ = [
    "DoclingBackend",
    "OpenAIApiBackend",
    "TesseractBackend",
    "VllmBackend",
]

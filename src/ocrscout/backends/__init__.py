"""Model backends: prepare and run OCR model invocations."""

from ocrscout.backends.docling import DoclingBackend
from ocrscout.backends.hf_scripts import HfScriptsBackend
from ocrscout.backends.openai_api import OpenAIApiBackend
from ocrscout.backends.tesseract import TesseractBackend

__all__ = [
    "DoclingBackend",
    "HfScriptsBackend",
    "OpenAIApiBackend",
    "TesseractBackend",
]

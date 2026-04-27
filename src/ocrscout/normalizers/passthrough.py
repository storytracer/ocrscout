"""Passthrough normalizer for backends that produce DoclingDocument directly.

Used by DoclingBackend (and any future backend whose output IS already a
DoclingDocument). Payload must be the JSON serialization produced by
DoclingDocument.model_dump_json().
"""

from __future__ import annotations

from docling_core.types.doc import DoclingDocument

from ocrscout.errors import NormalizerError
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput


class PassthroughNormalizer(Normalizer):
    name = "passthrough"
    output_format = "docling_document"

    def normalize(
        self, raw: RawOutput, page: PageImage, profile: ModelProfile
    ) -> DoclingDocument:
        if raw.output_format != "docling_document":
            raise NormalizerError(
                f"PassthroughNormalizer expects output_format='docling_document', "
                f"got {raw.output_format!r}"
            )
        try:
            return DoclingDocument.model_validate_json(raw.payload)
        except Exception as e:
            raise NormalizerError(
                f"failed to deserialize DoclingDocument for page {page.page_id!r}: {e}"
            ) from e

"""DocTags normalizer — wraps docling-core's built-in loader."""

from __future__ import annotations

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

from ocrscout.errors import NormalizerError
from ocrscout.interfaces.normalizer import Normalizer
from ocrscout.profile import ModelProfile
from ocrscout.types import PageImage, RawOutput


class DocTagsNormalizer(Normalizer):
    name = "doctags"
    output_format = "doctags"

    def normalize(
        self, raw: RawOutput, page: PageImage, profile: ModelProfile
    ) -> DoclingDocument:
        if raw.output_format != "doctags":
            raise NormalizerError(
                f"DocTagsNormalizer expects output_format='doctags', got {raw.output_format!r}"
            )
        # Build a single-page DocTagsDocument, then hand it to docling-core to
        # convert into the rich DoclingDocument representation.
        #
        # Some doctag profiles (dots-mocr, …) embed bbox coordinates relative
        # to the page image; passing the image lets docling-core resolve
        # those into geometry the DoclingDocument exposes. Re-load via
        # ``open_image()`` so the decoded buffer is freed when this returns
        # — the backend already closed its copy by the time the normalizer
        # runs.
        try:
            if page.image is not None or page.image_loader is not None:
                with page.open_image() as img:
                    dtd = DocTagsDocument.from_doctags_and_image_pairs(
                        doctags=[raw.payload],
                        images=[img],
                    )
                    return DoclingDocument.load_from_doctags(
                        dtd, document_name=page.page_id
                    )
            dtd = DocTagsDocument.from_doctags_and_image_pairs(
                doctags=[raw.payload],
                images=None,
            )
            return DoclingDocument.load_from_doctags(dtd, document_name=page.page_id)
        except Exception as e:  # noqa: BLE001
            raise NormalizerError(
                f"failed to load DocTags for page {page.page_id!r}: {e}"
            ) from e

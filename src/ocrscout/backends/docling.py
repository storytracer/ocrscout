"""DoclingBackend: wrap the full Docling pipeline as one model backend.

In-process. Requires the optional ``docling`` extra. Currently wired for
SmolDocling via the docling VLM pipeline; other docling-supported VLMs can be
plugged in by extending ``_build_converter`` to dispatch on ``profile.model_id``.

Output: each page is yielded as a ``RawOutput`` with ``output_format='docling_document'``
and the JSON-serialized ``DoclingDocument`` in ``payload``. Pair with the
``passthrough`` normalizer.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.types import BackendInvocation, PageImage, RawOutput

log = logging.getLogger(__name__)


class DoclingBackend(ModelBackend):
    name = "docling"

    def prepare(
        self, profile: ModelProfile, pages: Sequence[PageImage]
    ) -> BackendInvocation:
        return BackendInvocation(
            kind="in_process",
            callable_path="ocrscout.backends.docling:DoclingBackend.run",
            profile=profile,
            pages=[p.page_id for p in pages],
            extra={"pages_runtime": list(pages)},
        )

    def run(self, invocation: BackendInvocation) -> Iterator[RawOutput]:
        try:
            converter = _build_converter(invocation.profile)
        except ImportError as e:
            raise BackendError(
                "DoclingBackend requires the 'docling' extra: "
                "pip install 'ocrscout[docling]'"
            ) from e
        except Exception as e:
            raise BackendError(
                f"DoclingBackend: failed to build converter for {invocation.profile.name!r}: {e}"
            ) from e

        runtime_pages: list[PageImage] = list(invocation.extra.get("pages_runtime", []))
        if not runtime_pages:
            return

        total = len(runtime_pages)
        for i, page in enumerate(runtime_pages, 1):
            t0 = time.perf_counter()
            print(f"  docling [{i}/{total}] {page.page_id} ...", flush=True)
            try:
                with _resolved_image_path(page) as image_path:
                    result = converter.convert(str(image_path))
                doc = result.document
                # docling sets doc.name from the file stem, which for our
                # temp PNG is meaningless ("tmpXXXX"). Restore the page id
                # so downstream consumers see the source filename.
                doc.name = page.page_id
                elapsed = time.perf_counter() - t0
                items = len(getattr(doc, "texts", []) or [])
                print(
                    f"  docling [{i}/{total}] {page.page_id} ok "
                    f"({elapsed:.1f}s, {items} text items)",
                    flush=True,
                )
                yield RawOutput(
                    page_id=page.page_id,
                    output_format="docling_document",
                    payload=doc.model_dump_json(),
                )
            except Exception as e:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                log.warning("DoclingBackend failed on page %s: %s", page.page_id, e)
                print(
                    f"  docling [{i}/{total}] {page.page_id} FAILED "
                    f"({elapsed:.1f}s): {e}",
                    flush=True,
                )
                yield RawOutput(
                    page_id=page.page_id,
                    output_format="docling_document",
                    payload="",
                    error=str(e),
                )


def _build_converter(profile: ModelProfile) -> Any:
    """Build a docling DocumentConverter wired for the requested model.

    Currently supports SmolDocling (model_id ``ds4sd/SmolDocling-256M-preview``);
    extend this dispatch as more docling-supported VLMs are added.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        VlmPipelineOptions,
        smoldocling_vlm_conversion_options,
    )
    from docling.document_converter import DocumentConverter, ImageFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    name = profile.name.lower()
    model_id = (profile.model_id or "").lower()

    if "smoldocling" in name or "smoldocling" in model_id:
        vlm_opts = smoldocling_vlm_conversion_options
    else:
        raise BackendError(
            f"DoclingBackend has no VLM mapping for profile {profile.name!r} "
            f"(model_id={profile.model_id!r}). Add a dispatch case in "
            f"docling.py:_build_converter."
        )

    pipeline_options = VlmPipelineOptions(vlm_options=vlm_opts)
    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )


class _resolved_image_path:
    """Context manager that yields a filesystem path for ``page``.

    Always materializes ``page.image`` (PIL) as a temp PNG. Docling routes
    inputs via extension-based ``InputFormat`` detection (jpg/jpeg/png/tif/
    tiff/bmp/webp), so handing it the original ``source_uri`` would fail for
    common formats outside that set (e.g. ``.jp2``). Re-encoding via PIL is
    cheap relative to model inference and keeps the suffix predictable.
    """

    def __init__(self, page: PageImage) -> None:
        self._page = page
        self._path: Path | None = None

    def __enter__(self) -> Path:
        if self._page.image is None:
            raise BackendError(
                f"DoclingBackend: page {self._page.page_id!r} has no in-memory image"
            )
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        self._path = Path(tmp.name)
        self._page.image.save(self._path, format="PNG")
        return self._path

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._path is not None:
            try:
                self._path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

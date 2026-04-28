"""PP-DocLayoutV3 layout detector via the official Hugging Face Transformers
integration.

Uses ``AutoModelForObjectDetection`` + ``AutoImageProcessor`` from the
``PaddlePaddle/PP-DocLayoutV3_safetensors`` repo. This is the upstream-
maintained path; we don't roll our own preprocessing, NMS, or scale-back —
``image_processor.post_process_object_detection(target_sizes=...)`` returns
boxes in absolute page-pixel coordinates, already filtered by the model's
internal scoring, and emitted in the model's predicted reading order.

Costs ~600 MB extra in the ``[layout]`` extra (transformers + torch CPU).
That's worth it: a v1-quality custom ONNX path was 200 LOC of fragile
preprocessing and decoding heuristics; this is 80 LOC and works.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ocrscout.errors import BackendError
from ocrscout.interfaces.layout_detector import LayoutDetector
from ocrscout.types import LayoutRegion, PageImage

log = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_DEFAULT_SCORE_THRESHOLD = 0.5
_DEFAULT_MIN_AREA_PX = 64 * 64


class PpDocLayoutV3Detector(LayoutDetector):
    """Hugging Face Transformers PP-DocLayoutV3 detector.

    Constructor kwargs come from ``profile.layout_detector_args``. Pin
    ``revision`` per profile for reproducible benchmarks.
    """

    name: ClassVar[str] = "pp-doclayout-v3"

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        revision: str | None = None,
        device: str = "cpu",
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        min_area_px: int = _DEFAULT_MIN_AREA_PX,
        torch_dtype: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._score_threshold = float(score_threshold)
        self._min_area_px = int(min_area_px)
        self._torch_dtype = torch_dtype
        # Lazy: don't import transformers/torch or fetch weights until the
        # first detect() call — keeps ``import ocrscout`` snappy on hosts
        # without the [layout] extra.
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def detect(self, page: PageImage) -> list[LayoutRegion]:
        if page.image is None:
            raise BackendError(
                f"PpDocLayoutV3Detector: page {page.page_id!r} has no in-memory image"
            )
        self._ensure_loaded()
        torch = self._torch

        rgb = page.image.convert("RGB")
        target_sizes = [(page.height, page.width)]

        with torch.inference_mode():
            inputs = self._processor(images=[rgb], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self._score_threshold,
        )[0]

        scores = results["scores"].detach().cpu().tolist()
        labels = results["labels"].detach().cpu().tolist()
        boxes = results["boxes"].detach().cpu().tolist()
        # ``polygon_points`` is present on PPDocLayoutV3ImageProcessor; on a
        # stock DETR-family processor it isn't. Tolerate both.
        polys_raw = results.get("polygon_points")

        id2label = self._model.config.id2label
        regions: list[LayoutRegion] = []
        for idx, (score, label_id, box) in enumerate(
            zip(scores, labels, boxes, strict=False)
        ):
            x0, y0, x1, y1 = (float(c) for c in box)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            if (x1 - x0) * (y1 - y0) < self._min_area_px:
                continue
            category = id2label.get(int(label_id), f"class_{int(label_id)}")
            polygon = _coerce_polygon(polys_raw, idx) if polys_raw is not None else None
            regions.append(
                LayoutRegion(
                    id=idx,
                    category=str(category),
                    bbox=(x0, y0, x1, y1),
                    score=float(score),
                    polygon=polygon,
                    # The processor emits results in the model's predicted
                    # reading order; preserve that via the index.
                    reading_order=idx,
                )
            )
        return regions

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoImageProcessor,
                AutoModelForObjectDetection,
            )
        except ImportError as e:  # pragma: no cover
            raise BackendError(
                "PpDocLayoutV3Detector needs transformers + torch; "
                "install ocrscout[layout]"
            ) from e

        log.info(
            "loading PP-DocLayoutV3 from %s%s on %s",
            self._model_id,
            f"@{self._revision}" if self._revision else "",
            self._device,
        )
        dtype = _resolve_torch_dtype(torch, self._torch_dtype)
        load_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            load_kwargs["revision"] = self._revision
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        self._processor = AutoImageProcessor.from_pretrained(
            self._model_id, **{k: v for k, v in load_kwargs.items() if k != "dtype"}
        )
        self._model = AutoModelForObjectDetection.from_pretrained(
            self._model_id, **load_kwargs
        )
        self._model.eval()
        self._model.to(self._device)
        self._torch = torch


def _resolve_torch_dtype(torch: Any, name: str) -> Any:
    """Map a string name → torch dtype. ``"auto"`` returns None (let the
    loader pick)."""
    if name in (None, "", "auto"):
        return None
    name = name.lower()
    mapping = {
        "float32": getattr(torch, "float32", None),
        "fp32": getattr(torch, "float32", None),
        "float16": getattr(torch, "float16", None),
        "fp16": getattr(torch, "float16", None),
        "half": getattr(torch, "float16", None),
        "bfloat16": getattr(torch, "bfloat16", None),
        "bf16": getattr(torch, "bfloat16", None),
    }
    dtype = mapping.get(name)
    if dtype is None:
        raise BackendError(
            f"PpDocLayoutV3Detector: unknown torch_dtype={name!r}"
        )
    return dtype


def _coerce_polygon(polys_raw: Any, idx: int) -> list[tuple[float, float]] | None:
    """Best-effort conversion of one entry from ``polygon_points`` into a
    list of (x, y) pairs.

    The transformers processor returns either a tensor of shape
    ``(N, P, 2)`` or a Python list; both work with ``[idx]`` indexing.
    """
    try:
        poly = polys_raw[idx]
    except (IndexError, KeyError, TypeError):
        return None
    if poly is None:
        return None
    if hasattr(poly, "detach"):
        poly = poly.detach().cpu().tolist()
    pts: list[tuple[float, float]] = []
    for pt in poly:
        if pt is None:
            continue
        try:
            x, y = pt[0], pt[1]
        except (IndexError, TypeError):
            continue
        pts.append((float(x), float(y)))
    return pts or None

"""Typed parquet row models for the staged pipeline.

One nested family, columns defined once via inheritance:

    PageRow ─┬─ LayoutRow                 (pages ⊂ layout)
             └─ RawRow ── ResultRow       (pages ⊂ raw ⊂ train)

Rich logical fields (``extra``, ``regions``, ``comparisons``, ``metrics``,
``reference_provenance``) are held as their real Python types in the model and
serialized to ``*_json`` string columns on disk via :meth:`StageRow.to_arrow` /
:meth:`StageRow.from_arrow`. The Arrow schema is derived from these models in
:mod:`ocrscout.io.schema` — there is no hand-maintained ``Features`` dict.

This replaces the two divergent writers (``exports/parquet.py`` ``ExportRecord``
rows and ``exports/stages.py`` dict rows) with a single typed contract.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ocrscout.interfaces.comparison import ComparisonResult
from ocrscout.types import (
    ExportRecord,
    LayoutRegion,
    PageImage,
    RawOutput,
    ReferenceProvenance,
    Volume,
)


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable structure,
    calling ``model_dump(mode="json")`` on any Pydantic model encountered."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _encode_json(value: Any) -> str | None:
    """Encode a rich field to its on-disk JSON string, or ``None`` when empty
    (matches the historical writer: empty dict/list → null column)."""
    if value is None or value == {} or value == []:
        return None
    return json.dumps(_to_jsonable(value))


class StageRow(BaseModel):
    """Base for every stage row. Subclasses declare which logical fields are
    JSON-encoded on disk via ``_OWN_JSON_FIELDS`` (logical name → column name);
    everything else maps to a column of the same name."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # logical field name → on-disk column name. Merged across the MRO.
    _OWN_JSON_FIELDS: ClassVar[dict[str, str]] = {}

    @classmethod
    def json_fields(cls) -> dict[str, str]:
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_OWN_JSON_FIELDS")
            if own:
                merged.update(own)
        return merged

    def to_arrow(self) -> dict[str, Any]:
        """Flatten to an on-disk row dict (rich fields → ``*_json`` strings)."""
        json_fields = self.json_fields()
        out: dict[str, Any] = {}
        for name in type(self).model_fields:
            if name in json_fields:
                continue
            out[name] = getattr(self, name)
        for logical, column in json_fields.items():
            out[column] = _encode_json(getattr(self, logical))
        return out

    @classmethod
    def from_arrow(cls, row: dict[str, Any]) -> StageRow:
        """Rebuild a row model from an on-disk row dict."""
        json_fields = cls.json_fields()
        data: dict[str, Any] = {}
        for name in cls.model_fields:
            if name in json_fields:
                continue
            if name in row:
                data[name] = row[name]
        for logical, column in json_fields.items():
            raw = row.get(column)
            # Empty column → leave the field unset so its model default
            # (default_factory dict/list, or None) applies. Setting None
            # explicitly would fail validation for non-optional containers.
            if raw:
                data[logical] = json.loads(raw)
        return cls.model_validate(data)


class PageRow(StageRow):
    """Page identity — the ``ocrscout sample`` output and the base every
    downstream stage row reconstructs a ``PageImage`` from."""

    _OWN_JSON_FIELDS: ClassVar[dict[str, str]] = {"extra": "extra_json"}

    page_id: str
    file_id: str
    barcode: str | None = None
    sequence: int | None = None
    source_uri: str | None = None
    width: int | None = None
    height: int | None = None
    dpi: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_page(cls, page: PageImage) -> PageRow:
        return cls(
            page_id=page.page_id,
            file_id=page.file_id,
            barcode=page.barcode,
            sequence=page.sequence,
            source_uri=page.source_uri,
            width=page.width or None,
            height=page.height or None,
            dpi=page.dpi,
            extra=dict(page.extra or {}),
        )

    def to_page(
        self, storage_options: dict[str, Any] | None = None
    ) -> PageImage | None:
        """Rebuild a lazy ``PageImage``; ``None`` if there's no ``source_uri``
        to reconstruct the image from."""
        if not self.source_uri:
            return None
        from ocrscout.io.images import loader_for_uri

        return PageImage(
            page_id=self.page_id,
            file_id=self.file_id,
            image_loader=loader_for_uri(self.source_uri, storage_options),
            source_uri=self.source_uri,
            width=self.width or 0,
            height=self.height or 0,
            dpi=self.dpi,
            barcode=self.barcode,
            sequence=self.sequence,
            extra=dict(self.extra or {}),
        )


class LayoutRow(PageRow):
    """``ocrscout layout`` output: page identity + detected regions."""

    _OWN_JSON_FIELDS: ClassVar[dict[str, str]] = {"regions": "regions_json"}

    regions: list[LayoutRegion] = Field(default_factory=list)
    detector: str | None = None
    detect_seconds: float | None = None
    detect_error: str | None = None

    @classmethod
    def from_detection(
        cls,
        page: PageImage,
        regions: list[LayoutRegion],
        *,
        detector: str | None,
        detect_seconds: float | None,
        detect_error: str | None,
    ) -> LayoutRow:
        base = PageRow.from_page(page)
        return cls(
            **base.model_dump(),
            regions=list(regions),
            detector=detector,
            detect_seconds=detect_seconds,
            detect_error=detect_error,
        )

    def to_page(
        self, storage_options: dict[str, Any] | None = None
    ) -> PageImage | None:
        """Rebuild the page and attach the precomputed regions, so a
        layout-aware backend can OCR them without re-detecting."""
        page = super().to_page(storage_options)
        if page is not None:
            page.regions = list(self.regions)
        return page


class CostColumns(BaseModel):
    """Per-page cost / autoscaler context, recorded at OCR time and carried
    verbatim through ``raw`` into ``train`` (never recomputed in normalize)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gpu_type: str | None = None
    provider: str | None = None
    cost_per_hour: float | None = None
    elapsed_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    litellm_cost: float | None = None
    gpu_time_cost: float | None = None
    kv_cache_memory_bytes: int | None = None
    concurrent_requests: int | None = None
    region_concurrency: int | None = None


class RawRow(PageRow, CostColumns):
    """``ocrscout ocr`` output: page identity + ``RawOutput`` + cost columns."""

    model: str
    output_format: str
    raw_payload: str
    tokens: int | None = None
    error: str | None = None

    @classmethod
    def from_output(
        cls,
        page: PageImage,
        model: str,
        raw: RawOutput,
        cost: CostColumns | None = None,
    ) -> RawRow:
        base = PageRow.from_page(page).model_dump()
        cost_data = (cost or CostColumns()).model_dump()
        return cls(
            **base,
            **cost_data,
            model=model,
            output_format=raw.output_format,
            raw_payload=raw.payload,
            tokens=raw.tokens,
            error=raw.error,
        )

    def to_raw_output(self) -> RawOutput:
        return RawOutput(
            page_id=self.page_id,
            output_format=self.output_format,  # type: ignore[arg-type]
            payload=self.raw_payload,
            tokens=self.tokens,
            error=self.error,
        )

    def cost_columns(self) -> CostColumns:
        return CostColumns(**{k: getattr(self, k) for k in CostColumns.model_fields})


# Flat metric column → (comparison name, summary key). Lifted from
# ``comparisons[name].summary[key]`` for ergonomic SQL aggregation.
_FLAT_METRICS: dict[str, tuple[str, str]] = {
    "text_similarity": ("text", "similarity"),
    "text_cer": ("text", "cer"),
    "text_wer": ("text", "wer"),
    "document_heading_count_delta": ("document", "heading_count_delta"),
    "document_table_count_delta": ("document", "table_count_delta"),
    "document_picture_count_delta": ("document", "picture_count_delta"),
    "layout_iou_mean": ("layout", "iou_mean"),
}


class ResultRow(RawRow):
    """``ocrscout normalize`` output: the rich final results row (``train``)."""

    _OWN_JSON_FIELDS: ClassVar[dict[str, str]] = {
        "reference_provenance": "reference_provenance_json",
        "metrics": "metrics_json",
        "comparisons": "comparisons_json",
    }

    document_json: str | None = None
    markdown: str | None = None
    text: str | None = None
    reference_text: str | None = None
    reference_provenance: ReferenceProvenance | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    comparisons: dict[str, ComparisonResult] = Field(default_factory=dict)

    # Flat metric columns, derived from ``comparisons`` (see _FLAT_METRICS).
    text_similarity: float | None = None
    text_cer: float | None = None
    text_wer: float | None = None
    document_heading_count_delta: int | None = None
    document_table_count_delta: int | None = None
    document_picture_count_delta: int | None = None
    layout_iou_mean: float | None = None

    @classmethod
    def from_export_record(cls, rec: ExportRecord) -> ResultRow:
        doc = rec.document
        if hasattr(doc, "model_dump_json"):
            document_json: str | None = doc.model_dump_json()
        elif doc is not None:
            document_json = json.dumps(_to_jsonable(doc))
        else:
            document_json = None

        base = PageRow.from_page(rec.page).model_dump()
        cost_data = CostColumns(
            gpu_type=rec.gpu_type, provider=rec.provider,
            cost_per_hour=rec.cost_per_hour, elapsed_seconds=rec.elapsed_seconds,
            input_tokens=rec.input_tokens, output_tokens=rec.output_tokens,
            litellm_cost=rec.litellm_cost, gpu_time_cost=rec.gpu_time_cost,
            kv_cache_memory_bytes=rec.kv_cache_memory_bytes,
            concurrent_requests=rec.concurrent_requests,
            region_concurrency=rec.region_concurrency,
        ).model_dump()

        row = cls(
            **base,
            **cost_data,
            model=rec.model,
            output_format=rec.raw.output_format,
            raw_payload=rec.raw.payload,
            tokens=rec.raw.tokens,
            error=rec.raw.error,
            document_json=document_json,
            markdown=rec.markdown,
            text=rec.text,
            reference_text=rec.reference.text if rec.reference else None,
            reference_provenance=(
                rec.reference.provenance if rec.reference else None
            ),
            metrics=dict(rec.metrics or {}),
            comparisons=dict(rec.comparisons or {}),
        )
        row._derive_flat_metrics()
        return row

    def _derive_flat_metrics(self) -> None:
        for column, (name, key) in _FLAT_METRICS.items():
            result = self.comparisons.get(name)
            if result is not None:
                value = (result.summary or {}).get(key)
                if value is not None:
                    setattr(self, column, value)


class VolumeRow(StageRow):
    """Per-volume bibliographic sidecar (``volumes-*.parquet``)."""

    _OWN_JSON_FIELDS: ClassVar[dict[str, str]] = {
        "creators": "creators_json",
        "extra": "extra_json",
    }

    barcode: str
    title: str | None = None
    creators: list[str] = Field(default_factory=list)
    language: str | None = None
    year: int | None = None
    rights: str | None = None
    page_count: int | None = None
    source_uri: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_volume(cls, vol: Volume) -> VolumeRow:
        return cls(
            barcode=vol.barcode, title=vol.title, creators=list(vol.creators or []),
            language=vol.language, year=vol.year, rights=vol.rights,
            page_count=vol.page_count, source_uri=vol.source_uri,
            extra=dict(vol.extra or {}),
        )

"""Pure aggregation helpers over the rows of an ocrscout ``results.parquet``.

Two consumers:

* the viewer (``ViewerStore`` calls :func:`compute_page_disagreement` to sort
  and decorate the page list);
* the dataset card renderer (``publish/_card.py``).

Inputs are plain dicts (the same shape ``ViewerStore._rows`` materialises) so
this module has no datasets/polars/pyarrow dependency.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from typing import Any

from ocrscout.viewer.diff import tokenize


@dataclass(frozen=True)
class ModelStats:
    """Aggregated per-model metrics over a run."""

    model: str
    output_format: str | None
    pages_ok: int
    pages_errored: int
    total_tokens: int
    mean_tokens: float | None
    mean_run_seconds_per_page: float | None
    mean_prepare_seconds: float | None
    mean_text_length: float | None
    mean_item_count: float | None


@dataclass(frozen=True)
class PageDisagreement:
    """Per-page summary used for "most divergent pages" tables."""

    file_id: str
    disagreement: float
    n_models: int


def compute_page_disagreement(rows_for_page: list[dict[str, Any]]) -> float:
    """Mean pairwise word-diff distance across the rows of a single page.

    0.0 = all models agreed exactly; 1.0 = no shared tokens. Single-model
    pages return 0.0 (no disagreement to measure). ``rows_for_page`` is
    expected to carry a ``markdown`` field — typically the resolved markdown
    body from :class:`ocrscout.viewer.store.ViewerStore._markdown_for`, but
    any string source works.
    """
    token_streams: list[list[str]] = []
    for r in rows_for_page:
        md = r.get("markdown") or ""
        if md:
            token_streams.append(tokenize(md))
    if len(token_streams) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(token_streams)):
        for j in range(i + 1, len(token_streams)):
            ratio = difflib.SequenceMatcher(
                None, token_streams[i], token_streams[j], autojunk=False
            ).quick_ratio()
            total += 1.0 - ratio
            n += 1
    return total / n if n else 0.0


def aggregate_per_model(rows: list[dict[str, Any]]) -> list[ModelStats]:
    """Group rows by ``model`` and produce one :class:`ModelStats` per group.

    Rows are expected to expose ``model``, ``output_format``, ``error``,
    ``tokens``, and a parsed ``metrics`` dict (with keys
    ``run_seconds_per_page``, ``prepare_seconds``, ``text_length``,
    ``item_count``). Missing keys are tolerated.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r.get("model") or "?", []).append(r)
    out: list[ModelStats] = []
    for model, group in sorted(by_model.items()):
        ok = [r for r in group if not r.get("error")]
        errored = [r for r in group if r.get("error")]
        tokens = [int(r.get("tokens") or 0) for r in ok]
        metrics_list = [r.get("metrics") or {} for r in ok]
        out.append(
            ModelStats(
                model=model,
                output_format=_first_non_null(r.get("output_format") for r in group),
                pages_ok=len(ok),
                pages_errored=len(errored),
                total_tokens=sum(tokens),
                mean_tokens=_mean([t for t in tokens if t]),
                mean_run_seconds_per_page=_mean(
                    [m.get("run_seconds_per_page") for m in metrics_list]
                ),
                mean_prepare_seconds=_mean(
                    [m.get("prepare_seconds") for m in metrics_list]
                ),
                mean_text_length=_mean(
                    [m.get("text_length") for m in metrics_list]
                ),
                mean_item_count=_mean(
                    [m.get("item_count") for m in metrics_list]
                ),
            )
        )
    return out


def _row_file_id(r: dict[str, Any]) -> str:
    """Return the row's file_id, falling back to page_id for older parquets."""
    return r.get("file_id") or r["page_id"]


def page_disagreements(rows: list[dict[str, Any]]) -> list[PageDisagreement]:
    """One :class:`PageDisagreement` per page across the run."""
    by_page: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_page.setdefault(_row_file_id(r), []).append(r)
    out: list[PageDisagreement] = []
    for file_id, group in by_page.items():
        out.append(
            PageDisagreement(
                file_id=file_id,
                disagreement=compute_page_disagreement(group),
                n_models=len({r.get("model") for r in group if r.get("model")}),
            )
        )
    return out


def top_disagreement_pages(
    rows: list[dict[str, Any]], k: int = 10
) -> list[PageDisagreement]:
    """Top-``k`` pages by disagreement, descending. Multi-model pages only."""
    pds = [pd for pd in page_disagreements(rows) if pd.n_models >= 2]
    pds.sort(key=lambda pd: (-pd.disagreement, pd.file_id))
    return pds[:k]


def overall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """High-level run summary used at the top of the dataset card."""
    pages = sorted({_row_file_id(r) for r in rows})
    models = sorted({m for m in (r.get("model") for r in rows) if m})
    pds = page_disagreements(rows)
    multi = [pd for pd in pds if pd.n_models >= 2]
    pages_with_errors = {_row_file_id(r) for r in rows if r.get("error")}
    return {
        "n_pages": len(pages),
        "n_models": len(models),
        "n_rows": len(rows),
        "n_pages_with_errors": len(pages_with_errors),
        "mean_disagreement": _mean([pd.disagreement for pd in multi]),
        "median_disagreement": _median([pd.disagreement for pd in multi]),
    }


# --------------------------------------------------------------------- helpers


def _mean(xs: list[Any]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and not _is_nan(x)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _median(xs: list[Any]) -> float | None:
    floats: list[float] = [
        float(x) for x in xs if isinstance(x, (int, float)) and not _is_nan(x)
    ]
    if not floats:
        return None
    floats.sort()
    mid = len(floats) // 2
    if len(floats) % 2 == 1:
        return floats[mid]
    return 0.5 * (floats[mid - 1] + floats[mid])


def _is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _first_non_null(it):
    for x in it:
        if x is not None:
            return x
    return None

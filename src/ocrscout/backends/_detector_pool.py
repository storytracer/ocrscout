"""Shared CPU layout-detector pool.

Two consumers:

* :class:`~ocrscout.backends.layout_chat.LayoutChatBackend` — fuses detection
  with per-region OCR submission; it imports the *resolution helpers* and the
  reading-order sort from here (single source of truth) but keeps its own
  fused detect→submit hot path untouched.
* ``ocrscout layout`` (the standalone detection stage) — uses :class:`DetectorPool`
  here for detection only, writing regions to ``layout-*.parquet``.

The pool preserves the invariants the fused backend documents: serial
``warm_up()`` before any worker thread spawns (concurrent first-loads race in
transformers/accelerate), one detector instance per worker (the detector class
is not thread-safe), ``daemon=True`` threads, a process-global
``torch.set_num_threads`` budget set once before any forward pass, and
completion-order yielding.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

from ocrscout.backends.litellm import _state_override
from ocrscout.errors import BackendError
from ocrscout.profile import ModelProfile
from ocrscout.registry import registry
from ocrscout.types import LayoutRegion, PageImage

log = logging.getLogger(__name__)

_DEFAULT_RESERVED_CPUS = 2
"""CPUs withheld from the detector pool's autosize: one for the main / yield
thread, one for litellm + HTTP work."""
OCRSCOUT_DETECTOR_WORKERS_ENV = "OCRSCOUT_DETECTOR_WORKERS"
OCRSCOUT_DETECTOR_TORCH_THREADS_ENV = "OCRSCOUT_DETECTOR_TORCH_THREADS"
# Vertical bucketing for the top-then-left reading-order fallback. 50 px
# tolerates small same-row jitter without conflating rows on packed pages.
_READING_ORDER_ROW_PX = 50

_SENTINEL: Any = object()


def _available_cpus() -> int:
    """CPUs visible to this process. ``sched_getaffinity`` respects cgroup and
    ``taskset`` masks; falls back to ``cpu_count`` on platforms without the
    affinity API (macOS)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_detector_workers(profile: ModelProfile) -> int:
    """Precedence: explicit profile value > state-file override > env > auto.

    Auto: ``max(1, (available_cpus - reserved) // 2)`` — half the available
    CPUs become detector workers, the other half become the per-forward
    ``torch.set_num_threads`` budget.
    """
    explicit = (profile.backend_args or {}).get("detector_workers")
    if explicit is not None:
        return max(1, int(explicit))
    override = _state_override(profile.name, "detector_workers")
    if override is not None:
        return max(1, override)
    raw_env = os.environ.get(OCRSCOUT_DETECTOR_WORKERS_ENV)
    if raw_env:
        try:
            return max(1, int(raw_env))
        except ValueError:
            log.warning(
                "%s=%r is not an int; ignoring",
                OCRSCOUT_DETECTOR_WORKERS_ENV, raw_env,
            )
    available = _available_cpus()
    budget = max(1, available - _DEFAULT_RESERVED_CPUS)
    return max(1, budget // 2 or 1)


def _resolve_torch_threads_per_op(profile: ModelProfile, n_workers: int) -> int:
    """``torch.set_num_threads(V)`` value. Profile override → env → auto.

    Auto: leftover CPU budget after the detector workers, split per worker.
    """
    explicit = (profile.backend_args or {}).get("detector_torch_threads")
    if explicit is not None:
        return max(1, int(explicit))
    raw_env = os.environ.get(OCRSCOUT_DETECTOR_TORCH_THREADS_ENV)
    if raw_env:
        try:
            return max(1, int(raw_env))
        except ValueError:
            log.warning(
                "%s=%r is not an int; ignoring",
                OCRSCOUT_DETECTOR_TORCH_THREADS_ENV, raw_env,
            )
    available = _available_cpus()
    budget = max(1, available - _DEFAULT_RESERVED_CPUS)
    return max(1, budget // max(1, n_workers))


def _set_torch_intraop_threads(n: int) -> None:
    """Idempotent, best-effort ``torch.set_num_threads(n)``."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        log.debug("torch not importable; skipping set_num_threads(%d)", n)
        return
    try:
        torch.set_num_threads(max(1, int(n)))
    except Exception as e:  # noqa: BLE001
        log.warning("torch.set_num_threads(%d) failed: %s", n, e)


def sort_reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Order regions for downstream body assembly.

    If every region carries ``reading_order`` (the detector predicted it), use
    that; else fall back to a top-then-left bucketed sort that tolerates small
    same-row jitter.
    """
    if regions and all(r.reading_order is not None for r in regions):
        return sorted(regions, key=lambda r: (r.reading_order or 0, r.bbox[1], r.bbox[0]))

    def heuristic_key(r: LayoutRegion) -> tuple[int, float]:
        top = r.bbox[1]
        left = r.bbox[0]
        return (int(round(top / _READING_ORDER_ROW_PX)) * _READING_ORDER_ROW_PX, left)

    return sorted(regions, key=heuristic_key)


def detect_page_regions(
    page: PageImage, detector: Any
) -> tuple[list[LayoutRegion], str | None]:
    """Detect + reading-order-sort regions for one page; never raises.

    Pins the decoded image onto ``page.image`` (and backfills ``width`` /
    ``height``) before detection, then closes + clears it. Returns
    ``(sorted_regions, error)`` — ``error`` is non-None on image-load or
    detector failure so the caller records a per-page failure without
    aborting siblings.
    """
    try:
        if page.image is None and page.image_loader is not None:
            try:
                page.image = page.image_loader()
            except Exception as e:  # noqa: BLE001
                return [], f"image load failed: {type(e).__name__}: {e}"
        if page.image is not None and (page.width <= 0 or page.height <= 0):
            try:
                w, h = page.image.size
                page.width = int(w)
                page.height = int(h)
            except (TypeError, ValueError):
                pass
        try:
            regions = detector.detect(page)
        except Exception as e:  # noqa: BLE001
            return [], f"{type(e).__name__}: {e}"
        return sort_reading_order(regions or []), None
    finally:
        img = getattr(page, "image", None)
        if img is not None:
            try:
                img.close()
            except Exception:  # noqa: BLE001
                pass
            page.image = None


class DetectorPool:
    """Detect-only pool of N CPU detector workers, completion-order yielding.

    Used by ``ocrscout layout``. Construction instantiates + serially warms up
    one detector instance per worker (raising ``BackendError`` on failure);
    :meth:`map` then streams pages through the workers and yields
    ``(page, regions, detect_seconds, error)`` as each finishes.
    """

    def __init__(self, profile: ModelProfile, *, log_prefix: str | None = None) -> None:
        if not profile.layout_detector:
            raise BackendError(
                f"DetectorPool: profile {profile.name!r} has no layout_detector"
            )
        self.profile = profile
        self.log_prefix = log_prefix or f"[{profile.name}]"
        self.n_workers = _resolve_detector_workers(profile)
        self.torch_threads = _resolve_torch_threads_per_op(profile, self.n_workers)
        detector_cls = registry.get("layout_detectors", profile.layout_detector)
        try:
            self.detectors = [
                detector_cls(**(profile.layout_detector_args or {}))
                for _ in range(self.n_workers)
            ]
        except Exception as e:  # noqa: BLE001
            raise BackendError(
                f"DetectorPool: failed to instantiate layout detector "
                f"{profile.layout_detector!r}: {e}"
            ) from e
        # Serial warm-up before any worker spawns — concurrent first-loads of
        # the same HF model race inside transformers/accelerate.
        for i, det in enumerate(self.detectors):
            try:
                det.warm_up()
            except Exception as e:  # noqa: BLE001
                raise BackendError(
                    f"DetectorPool: detector #{i} warm-up failed: {e}"
                ) from e

    def map(
        self, pages: list[PageImage]
    ) -> Iterator[tuple[PageImage, list[LayoutRegion], float, str | None]]:
        pages = list(pages)
        total = len(pages)
        log.info(
            "%s detecting %d page(s) (detector=%s, workers=%d, torch threads/op=%d)",
            self.log_prefix, total, self.profile.layout_detector,
            self.n_workers, self.torch_threads,
        )
        if total == 0:
            return

        # Process-global; set once before any worker calls detect().
        _set_torch_intraop_threads(self.torch_threads)

        pages_queue: queue.Queue[Any] = queue.Queue()
        for idx, page in enumerate(pages, start=1):
            pages_queue.put((idx, page))
        for _ in range(self.n_workers):
            pages_queue.put(_SENTINEL)

        out_queue: queue.Queue[Any] = queue.Queue()
        cancel_event = threading.Event()
        threads: list[threading.Thread] = []
        for w in range(self.n_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self.detectors[w], pages_queue, out_queue, cancel_event),
                name=f"detector-{w}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        produced = 0
        try:
            # Each page produces exactly one out item (detect_page_regions
            # never raises), so this terminates once every page is consumed.
            while produced < total:
                page, regions, secs, err = out_queue.get()
                produced += 1
                yield page, regions, secs, err
        finally:
            cancel_event.set()
            for t in threads:
                t.join(timeout=15.0)

    def _worker(
        self,
        detector: Any,
        pages_queue: queue.Queue[Any],
        out_queue: queue.Queue[Any],
        cancel_event: threading.Event,
    ) -> None:
        while True:
            if cancel_event.is_set():
                return
            item = pages_queue.get()
            if item is _SENTINEL:
                return
            _idx, page = item
            t0 = time.perf_counter()
            regions, err = detect_page_regions(page, detector)
            out_queue.put((page, regions, time.perf_counter() - t0, err))

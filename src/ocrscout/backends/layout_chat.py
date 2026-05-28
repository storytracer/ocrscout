"""LayoutChatBackend: layout-detector-driven, region-level OCR over LiteLLM.

A pool of ``N`` CPU detector workers ingests pages from a shared
``pages_queue``; each owns its own ``PpDocLayoutV3Detector`` instance
(the detector class is not thread-safe). On completing detection, the
same worker opens the cost record for its page and submits the page's
regions directly onto the region executor under a shared
``BoundedSemaphore`` — keeping the detect → submit hop at microsecond
latency (no intermediate submitter thread). Region workers POST through
LiteLLM, mutate per-page state under a global lock, and enqueue the
state onto a completion queue when the page's last region completes;
the generator drains it and yields. Pages yield in completion order,
not input order — the caller (``cli/run.py:_run_one_model``) materializes
via ``list()`` and re-keys by ``page_id``, so this is invisible.

Three stalls this design eliminates vs. strict per-page sequencing
through one producer:

- Detector dead time between pages: vLLM keeps draining region POSTs
  from previously-detected pages while the next page's layout runs.
- Slow-region tail: a page's slowest region no longer holds back its
  own page-yield + every subsequent page's start; it's just one of M
  in-flight items competing for cycles.
- Detector serialization: ``N`` CPU detector workers run forward
  passes in parallel (PyTorch's ATen kernels release the GIL), so the
  detector phase is no longer the single-threaded floor on s/page on
  fast-vLLM hosts.

Requires an active Runner (LiteLLM proxy + at least one backing vLLM
serve). Subprocess vLLM is unsupported here because the per-region launch
cost would dwarf any inference time.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ocrscout import cost as cost_mod
from ocrscout.backends.litellm import (
    _build_messages,
    _split_sampling,
    _state_override,
)
from ocrscout.errors import BackendError
from ocrscout.interfaces.backend import ModelBackend
from ocrscout.profile import ModelProfile
from ocrscout.registry import registry
from ocrscout.types import BackendInvocation, LayoutRegion, PageImage, RawOutput

log = logging.getLogger(__name__)

_DEFAULT_REGION_CONCURRENCY = 8
"""Fallback cross-page region concurrency. Normally filled in by the
autoscaler via ``backend_args.region_concurrency`` (or the runner's
state file for submit-time workers); only applies when both lookups
return nothing."""
_DEFAULT_RESERVED_CPUS = 2
"""CPUs withheld from the detector pool's autosize: one for the main /
yield thread, one for litellm + HTTP work. Region executor threads are
I/O-bound (HTTP wait on vLLM), so they don't compete for CPU
meaningfully. The auto-derived pool fills the rest, scaling linearly
with ``sched_getaffinity``; no hard cap. Override via
``backend_args.detector_workers``, ``--detector-workers``, or
``OCRSCOUT_DETECTOR_WORKERS`` if a profile-specific value is needed."""
OCRSCOUT_DETECTOR_WORKERS_ENV = "OCRSCOUT_DETECTOR_WORKERS"
"""Runtime override for detector worker count. Same precedence story as
``region_concurrency``: explicit profile > state-file override (set by
the runner) > env > auto."""
OCRSCOUT_DETECTOR_TORCH_THREADS_ENV = "OCRSCOUT_DETECTOR_TORCH_THREADS"
"""Runtime override for ``torch.set_num_threads(V)`` — PyTorch's intra-op
parallelism per detector forward. Default is the leftover CPU budget split
evenly across detector workers (floor 1)."""
_DEFAULT_REQUEST_TIMEOUT = 600.0
_MODELS_PROBE_TIMEOUT = 15.0
# ``elapsed`` above this in ``_assemble_raw`` indicates the page sat in
# ``done_queue`` (or its regions sat in flight) far longer than any
# legitimate request could take — almost certainly the consumer-side
# stalled. Log the page at WARNING so the staleness is immediately
# visible in `-q` runs and grep-friendly in CI output; the default INFO
# "ok" line otherwise buries a 9-hour wait in plain text. The threshold
# is set to twice the default request timeout — anything past 2× the
# longest legitimate in-flight wait can't be explained by ordinary
# slow-region cases.
_STALL_WARN_ELAPSED_S = _DEFAULT_REQUEST_TIMEOUT * 2
# Vertical bucketing for top-then-left reading-order sort. 50 px tolerates
# small same-row jitter without conflating rows on tightly-packed pages.
_READING_ORDER_ROW_PX = 50

_SENTINEL: Any = object()
"""Marker pushed onto ``done_queue`` by the producer's ``finally`` so the
yielder unblocks even when the producer crashed before completing every
page."""


@dataclass
class _PageState:
    """Per-page assembly state shared between the producer (initial write)
    and worker threads (post-completion mutation under ``state_lock``).

    ``results`` is pre-sized and indexed by *position in ``ordered``*, not
    by ``region.id`` — detectors are free to return non-consecutive or
    even duplicate region ids (PP-DocLayoutV3 numbers ids before its size
    filter at [pp_doclayout_v3.py:95-107](src/ocrscout/layout_detectors/pp_doclayout_v3.py#L95-L107),
    so the surviving list isn't guaranteed to be 0..N-1). Keying by
    position makes assembly index-stable regardless."""

    page: PageImage
    page_idx: int
    ordered: list[LayoutRegion]
    total_regions: int
    remaining: int
    t_start: float
    results: list[dict[str, Any] | None] = field(default_factory=list)
    detector_error: str | None = None


def _resolve_region_concurrency(profile: ModelProfile) -> int:
    """Precedence: explicit profile value > state-file override > default.

    Same semantics as the litellm backend's ``_resolve_concurrent_requests``;
    state-file is the launch → submit → worker handoff path.
    """
    explicit = (profile.backend_args or {}).get("region_concurrency")
    if explicit is not None:
        return int(explicit)
    override = _state_override(profile.name, "region_concurrency")
    if override is not None:
        return override
    return _DEFAULT_REGION_CONCURRENCY


def _available_cpus() -> int:
    """CPUs visible to this process. ``sched_getaffinity`` respects cgroup
    and ``taskset`` masks (which a generic ``os.cpu_count()`` does not);
    falls back to ``cpu_count`` on platforms without the affinity API
    (macOS)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_detector_workers(profile: ModelProfile) -> int:
    """Precedence: explicit profile value > state-file override > env > auto.

    Auto: ``max(1, (available_cpus - reserved) // 2)`` — half the
    available CPUs go to detector workers, the other half become
    ``torch.set_num_threads(V)`` budget so each forward pass gets some
    intra-op parallelism. Together that's ``N × V ≈ available -
    reserved`` cores in flight (e.g. 24-core → N=11, V=2; 8-core →
    N=3, V=2). No hard cap — the ``reserved`` constant is the safety
    buffer. Pathologically large boxes (128+ cores) can pin via the
    env / profile overrides if 60+ detector instances is too much
    resident weight.
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

    Auto: leftover CPU budget after the detector workers' own threads,
    split per worker. The product ``n_workers × V`` should land near the
    available core count to avoid oversubscription.
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


class LayoutChatBackend(ModelBackend):
    """Layout-aware OCR over the LiteLLM proxy."""

    name = "layout_chat"
    requires_runner: ClassVar[bool] = True

    def prepare(self, profile: ModelProfile) -> BackendInvocation:
        if not profile.layout_detector:
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} has no layout_detector"
            )
        if not profile.prompt_templates:
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} has no prompt_templates"
            )

        proxy_url = os.environ.get("OCRSCOUT_VLLM_URL") or profile.server_url
        if not proxy_url:
            raise BackendError(
                "LayoutChatBackend requires a LiteLLM proxy URL. Launch a "
                "runner (`ocrscout launch --models ...`) or set "
                "OCRSCOUT_VLLM_URL."
            )
        proxy_url = proxy_url.rstrip("/")

        # Probe ``/models`` so a wrong URL fails up-front rather than after
        # every region 404s silently inside _post_region. The proxy
        # advertises the profile name (not the model_id) when its model_list
        # was generated from the same profile, so check against either.
        served = _list_proxy_models(proxy_url, timeout=_MODELS_PROBE_TIMEOUT)
        if profile.name not in served and profile.model_id not in served:
            raise BackendError(
                f"LayoutChatBackend: profile {profile.name!r} (model_id "
                f"{profile.model_id!r}) is not served by the LiteLLM proxy at "
                f"{proxy_url}; proxy serves {sorted(served)!r}."
            )

        detector_cls = registry.get("layout_detectors", profile.layout_detector)
        n_workers = _resolve_detector_workers(profile)
        try:
            detectors = [
                detector_cls(**profile.layout_detector_args)
                for _ in range(n_workers)
            ]
        except Exception as e:
            raise BackendError(
                f"LayoutChatBackend: failed to instantiate layout detector "
                f"{profile.layout_detector!r}: {e}"
            ) from e

        # Warm up each detector serially BEFORE the worker pool starts.
        # Concurrent first-loads of the same HF model race inside
        # transformers / accelerate — symptoms in the wild: ``Cannot copy
        # out of meta tensor`` on `.to(device)`, partial-init detector
        # instances whose ``_torch`` stays None, and subsequent
        # ``AttributeError: 'NoneType' object has no attribute
        # 'inference_mode'`` on every detect() call. Serial warm-up
        # closes the race: the first load populates the HF cache and
        # global transformers state; every subsequent instance hits the
        # warm path. Cost is paid once per benchmark run, before any
        # page is touched.
        for i, det in enumerate(detectors):
            try:
                det.warm_up()
            except Exception as e:
                raise BackendError(
                    f"LayoutChatBackend: detector #{i} warm-up failed: {e}"
                ) from e

        cost_mod.ensure_callback_registered()

        return BackendInvocation(
            kind="http",
            endpoint=proxy_url,
            profile=profile,
            pages=[],
            extra={"detectors": detectors},
        )

    def run(
        self,
        invocation: BackendInvocation,
        pages: Sequence[PageImage],
    ) -> Iterator[tuple[PageImage, RawOutput]]:
        profile = invocation.profile
        proxy_url = invocation.endpoint or ""
        if not proxy_url:
            raise BackendError("LayoutChatBackend.run: missing proxy URL")

        pages = list(pages)
        detectors: list[Any] = list(invocation.extra["detectors"])
        if not detectors:
            raise BackendError("LayoutChatBackend.run: empty detector pool")
        n_workers = len(detectors)
        timeout = float(
            profile.backend_args.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT)
        )
        region_concurrency = max(1, _resolve_region_concurrency(profile))
        torch_threads = _resolve_torch_threads_per_op(profile, n_workers)
        sampling = _split_sampling(profile.sampling_args or {})
        prefix = f"[{profile.name}]"
        total_pages = len(pages)

        log.info(
            "%s starting %d page(s) against %s "
            "(region concurrency=%d cross-page, detector=%s, "
            "detector workers=%d, torch threads/op=%d)",
            prefix, total_pages, proxy_url, region_concurrency,
            profile.layout_detector, n_workers, torch_threads,
        )

        if total_pages == 0:
            return

        # PyTorch intra-op parallelism is process-global. We set it once
        # here, before any detector worker calls `_ensure_loaded()`, so
        # every forward pass uses the same partitioned thread budget.
        _set_torch_intraop_threads(torch_threads)

        t_total = time.perf_counter()

        inflight_sem = threading.BoundedSemaphore(region_concurrency)
        state_lock = threading.Lock()
        # done_queue carries the _PageState directly (not a lookup key) so
        # workers can hold a closure reference to their own state. Keying
        # by page_id would corrupt state if the input ever contained two
        # pages with the same page_id (observed in BHL): a later page's
        # state would overwrite an earlier page's slot mid-flight, and
        # the earlier page's still-pending workers would write to the new
        # state's results list, masquerading as the new page's regions.
        done_queue: queue.Queue[Any] = queue.Queue()
        # Unbounded — the page list is already materialized in memory by
        # the caller, so buffering page refs here adds no real cost.
        pages_queue: queue.Queue[Any] = queue.Queue()
        for page_idx, page in enumerate(pages, start=1):
            pages_queue.put((page_idx, page))
        # One sentinel per detector worker — each pops exactly one.
        for _ in range(n_workers):
            pages_queue.put(_SENTINEL)

        cancel_event = threading.Event()
        worker_exc: list[BaseException] = []
        done_workers_lock = threading.Lock()
        done_workers = [0]
        # ``pages_completed`` is the source of truth for completion.
        # Either ``_submit_state`` (for empty/detector-error pages) or
        # ``_make_region_worker``'s last-region branch increments it; the
        # one that takes it to ``total_pages`` is responsible for pushing
        # ``_SENTINEL`` to ``done_queue``. This guarantees the sentinel
        # always trails the last real page, so the main thread can't
        # exit early while region work is still in flight.
        pages_completed_lock = threading.Lock()
        pages_completed = [0]
        # ``pages_dispatched`` is incremented at the START of
        # ``_submit_state`` — every page that's been popped from
        # ``pages_queue`` and handed to the region executor (or
        # short-circuited via publish_page for empty/error pages).
        # Used by the main thread's heartbeat: a "no progress" timeout
        # is only a true stall if every dispatched page has also
        # completed. While ``completed < dispatched`` there's still
        # legitimate region work in flight (some pages have 40+ regions
        # and slow regions can sit on the wire for tens of seconds),
        # so the main thread keeps waiting instead of declaring loss.
        pages_dispatched_lock = threading.Lock()
        pages_dispatched = [0]

        # Explicit lifecycle instead of ``with ThreadPoolExecutor(...) as
        # executor``: the context manager calls ``shutdown(wait=True)``
        # with the default ``cancel_futures=False``, which means a wedged
        # main thread (e.g. one that's lost a signal-handler race and is
        # blocked inside a corrupt BufferedWriter) can stall the exit
        # path indefinitely while it waits for futures that nothing is
        # consuming. We pair the executor with a try/finally that calls
        # ``shutdown(wait=True, cancel_futures=True)`` instead: pending
        # futures are dropped so abnormal termination is bounded by the
        # longest in-flight HTTP timeout, and running futures still get
        # awaited so we don't pull state out from under them. On normal
        # completion there are no pending futures (the in-flight
        # semaphore + per-page publish accounting guarantees all regions
        # resolved before the yield loop exits), so this preserves the
        # original wait-for-all-running semantics.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=region_concurrency,
            thread_name_prefix="layout_chat_region",
        )
        try:

            def _mark_dispatched() -> None:
                """Called at the head of ``_submit_state`` for every
                page handed to the region executor (including empty /
                detector-error pages that publish directly). Pairs
                with ``_publish_page``: while ``pages_completed <
                pages_dispatched``, region work is still in flight and
                the main thread must keep waiting."""
                with pages_dispatched_lock:
                    pages_dispatched[0] += 1

            def _publish_page(state: _PageState) -> None:
                """Push a fully-assembled (or empty / detector-failed)
                page to ``done_queue``; whoever takes ``pages_completed``
                to ``total_pages`` also pushes the terminating sentinel.

                The lock is held across both ``put()`` calls so concurrent
                publishers strictly serialize: state_N goes in before
                state_N+1, and ``_SENTINEL`` lands after the final state.
                ``Queue.put`` doesn't block on an unbounded queue, so
                holding the lock is fine.

                Releases the pinned decoded image *before* taking the
                publish lock — once the last region resolved, no other
                thread will read ``state.page.image`` again, and freeing
                it here drops the largest single allocation per page
                before the consumer-side flow (normalizer + exporter)
                even sees the state."""
                img = state.page.image
                if img is not None:
                    try:
                        img.close()
                    except Exception:  # noqa: BLE001
                        pass
                    state.page.image = None
                with pages_completed_lock:
                    pages_completed[0] += 1
                    is_last = pages_completed[0] >= total_pages
                    done_queue.put(state)
                    if is_last:
                        done_queue.put(_SENTINEL)

            detector_threads: list[threading.Thread] = []
            for i, det in enumerate(detectors):
                t = threading.Thread(
                    target=_detector_worker_loop,
                    name=f"layout_chat_detector_{i}",
                    daemon=True,
                    kwargs=dict(
                        worker_idx=i,
                        n_workers=n_workers,
                        detector=det,
                        pages_queue=pages_queue,
                        total_pages=total_pages,
                        proxy_url=proxy_url,
                        profile=profile,
                        sampling=sampling,
                        timeout=timeout,
                        log_prefix=prefix,
                        state_lock=state_lock,
                        inflight_sem=inflight_sem,
                        mark_dispatched=_mark_dispatched,
                        publish_page=_publish_page,
                        executor=executor,
                        cancel_event=cancel_event,
                        worker_exc=worker_exc,
                        done_workers=done_workers,
                        done_workers_lock=done_workers_lock,
                        done_queue=done_queue,
                    ),
                )
                t.start()
                detector_threads.append(t)

            yielded = 0
            try:
                while yielded < total_pages:
                    try:
                        item = done_queue.get(timeout=1.0)
                    except queue.Empty:
                        # Heartbeat: are we genuinely stuck, or just
                        # waiting on slow region work? Region POSTs
                        # against a busy vLLM can take 30-60s per
                        # request, and a single page with 40+ regions
                        # can sit in the executor longer than the
                        # poll window. A "no progress for 1s" silence
                        # alone is NOT a stall signal — it's only a
                        # genuine loss when every dispatched page has
                        # also completed AND the dispatched count
                        # under-runs ``total_pages`` (i.e. some pages
                        # never made it through _submit_state due to
                        # detector crashes).
                        with done_workers_lock:
                            all_workers_done = done_workers[0] >= n_workers
                        if not all_workers_done:
                            continue
                        with pages_completed_lock:
                            completed = pages_completed[0]
                        with pages_dispatched_lock:
                            dispatched = pages_dispatched[0]
                        if completed < dispatched:
                            # Region work still in flight; keep waiting.
                            continue
                        # All dispatched pages have published. If the
                        # dispatched count fell short of total_pages,
                        # some pages were lost — surface it and break.
                        # Otherwise publish_page already pushed the
                        # final _SENTINEL, which we'll see next.
                        if dispatched < total_pages:
                            log.error(
                                "%s detector workers all exited and "
                                "all dispatched pages completed, but "
                                "only %d/%d pages were dispatched "
                                "(%d lost in-flight). Yielded %d so far.",
                                prefix, dispatched, total_pages,
                                total_pages - dispatched, yielded,
                            )
                            break
                        # dispatched == total_pages but no _SENTINEL yet
                        # — shouldn't happen (publish_page pushes it on
                        # the last completion). Defensive break.
                        break
                    if item is _SENTINEL:
                        break
                    yield item.page, _assemble_raw(item, profile, prefix, total_pages)
                    yielded += 1
            finally:
                # Signal cancellation so detector workers blocked on
                # ``pages_queue.get()`` wake up and exit at the top of
                # their loop. Detector-thread join is deliberately
                # deferred to the OUTER finally below — joining before
                # the executor shuts down can deadlock when a worker is
                # blocked on ``inflight_sem.acquire()`` and only a
                # future completion would release the semaphore.
                cancel_event.set()
                for _ in range(n_workers):
                    pages_queue.put(_SENTINEL)
        finally:
            # Bound the executor shutdown so abnormal termination
            # doesn't stall here forever. ``cancel_futures=True`` drops
            # pending futures (the default ``False`` would wait on
            # everything in the queue); ``wait=True`` still awaits
            # running futures so they release ``inflight_sem`` and the
            # cost recorder cleanly. Worst case is bounded by the
            # request timeout (default 600s), not by the unconsumed
            # backlog.
            executor.shutdown(wait=True, cancel_futures=True)
            # Belt-and-braces semaphore drain. ``_submit_state`` now
            # polls ``cancel_event`` in its acquire loop, so workers
            # shouldn't be stranded on the semaphore — but a future
            # code path that bypasses the poll, or a worker that
            # already passed acquire when its future got cancelled
            # (consuming a permit ``_work``'s finally never released),
            # would re-introduce the hang. Releasing ``n_workers``
            # permits is enough to unstick any single-acquire blocker;
            # ``BoundedSemaphore`` raises ``ValueError`` if we exceed
            # the initial count, which is the common case and benign.
            for _ in range(n_workers):
                try:
                    inflight_sem.release()
                except ValueError:
                    pass
            # Bounded join: detector threads are daemon=True, so any
            # that still fail to exit die with the process. Don't let
            # one wedged worker hold the cleanup open indefinitely.
            for t in detector_threads:
                t.join(timeout=15.0)
                if t.is_alive():
                    log.warning(
                        "%s detector thread %s did not exit within 15s; "
                        "abandoning (daemon=True, dies with process)",
                        prefix, t.name,
                    )

        if worker_exc:
            if len(worker_exc) > 1:
                log.error(
                    "%s multiple detector workers crashed (%d total); "
                    "raising the first. All exceptions: %s",
                    prefix, len(worker_exc),
                    [f"{type(e).__name__}: {e}" for e in worker_exc],
                )
            raise worker_exc[0]

        log.info(
            "%s layout-chat finished %d pages in %.1fs",
            prefix, total_pages, time.perf_counter() - t_total,
        )


def _set_torch_intraop_threads(n: int) -> None:
    """Idempotent ``torch.set_num_threads(n)``. Best-effort: if torch
    isn't yet importable (e.g. the ``[layout]`` extra isn't installed
    despite a layout_chat profile somehow making it this far), log and
    move on — the first detector worker's ``_ensure_loaded`` will then
    raise the real ImportError."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        log.debug("torch not importable; skipping set_num_threads(%d)", n)
        return
    try:
        torch.set_num_threads(max(1, int(n)))
    except Exception as e:  # noqa: BLE001
        log.warning("torch.set_num_threads(%d) failed: %s", n, e)


def _detector_worker_loop(
    *,
    worker_idx: int,
    n_workers: int,
    detector: Any,
    pages_queue: queue.Queue[Any],
    total_pages: int,
    proxy_url: str,
    profile: ModelProfile,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
    log_prefix: str,
    state_lock: threading.Lock,
    inflight_sem: threading.BoundedSemaphore,
    mark_dispatched: Callable[[], None],
    publish_page: Callable[[_PageState], None],
    executor: concurrent.futures.ThreadPoolExecutor,
    cancel_event: threading.Event,
    worker_exc: list[BaseException],
    done_workers: list[int],
    done_workers_lock: threading.Lock,
    done_queue: queue.Queue[Any],
) -> None:
    """One detector worker: pop pages, detect, open the cost record,
    submit regions. Each worker owns its own detector instance — the
    detector class is not thread-safe (one shared ``nn.Module`` per
    instance, lazy load has no lock)."""
    try:
        while True:
            if cancel_event.is_set():
                return
            item = pages_queue.get()
            if item is _SENTINEL:
                return
            page_idx, page = item
            state = _detect_one_page(
                page=page,
                page_idx=page_idx,
                total_pages=total_pages,
                detector=detector,
                log_prefix=log_prefix,
            )
            _submit_state(
                state=state,
                profile=profile,
                proxy_url=proxy_url,
                sampling=sampling,
                timeout=timeout,
                state_lock=state_lock,
                inflight_sem=inflight_sem,
                mark_dispatched=mark_dispatched,
                publish_page=publish_page,
                executor=executor,
                cancel_event=cancel_event,
            )
    except BaseException as e:  # noqa: BLE001
        worker_exc.append(e)
        log.exception(
            "%s detector worker %d crashed; other workers continue",
            log_prefix, worker_idx,
        )
    finally:
        with done_workers_lock:
            done_workers[0] += 1
        # NOTE: do NOT push _SENTINEL here. ``publish_page`` is the only
        # authority on completion — it pushes _SENTINEL strictly after
        # the last real page. A detector worker that finishes ``detect``
        # + submits regions returns immediately; the regions may still
        # be in flight in the executor, and pushing _SENTINEL now would
        # let the yielder exit before those regions complete. The
        # crash-recovery path is the main thread's timeout-poll, which
        # detects "all detector workers exited and no progress" and
        # raises ``worker_exc``.


def _detect_one_page(
    *,
    page: PageImage,
    page_idx: int,
    total_pages: int,
    detector: Any,
    log_prefix: str,
) -> _PageState:
    """Run the layout detector for one page. Returns a fully-built
    ``_PageState`` — either with ``ordered`` populated (success), or
    with ``detector_error`` set (per-page detector exception), or with
    ``total_regions == 0`` (empty detection). Never raises; per-page
    failures are kept local so sibling pages still progress.

    Pins the decoded PIL image onto ``page.image`` for the lifetime of
    this page's region fan-out. The detector reads from ``page.image``
    and the region workers crop from it concurrently; ``_publish_page``
    closes + clears it when the last region resolves (or immediately, on
    a detector-error / empty-detection path)."""
    t_start = time.perf_counter()

    if page.image is None and page.image_loader is not None:
        try:
            page.image = page.image_loader()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "%s page %d/%d %s image load FAIL: %s",
                log_prefix, page_idx, total_pages, page.file_id, e,
            )
            return _PageState(
                page=page,
                page_idx=page_idx,
                ordered=[],
                total_regions=0,
                remaining=0,
                t_start=t_start,
                detector_error=f"image load failed: {type(e).__name__}: {e}",
            )

    # Backfill page dims now that the image is in hand — the detector and
    # downstream layout_json normalizer both read ``page.{width,height}``.
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
        log.warning(
            "%s page %d/%d %s detector FAIL: %s",
            log_prefix, page_idx, total_pages, page.file_id, e,
        )
        return _PageState(
            page=page,
            page_idx=page_idx,
            ordered=[],
            total_regions=0,
            remaining=0,
            t_start=t_start,
            detector_error=f"{type(e).__name__}: {e}",
        )

    if not regions:
        return _PageState(
            page=page,
            page_idx=page_idx,
            ordered=[],
            total_regions=0,
            remaining=0,
            t_start=t_start,
        )

    ordered = _sort_reading_order(regions)
    return _PageState(
        page=page,
        page_idx=page_idx,
        ordered=ordered,
        total_regions=len(ordered),
        remaining=len(ordered),
        t_start=t_start,
        results=[None] * len(ordered),
    )


def _submit_state(
    *,
    state: _PageState,
    profile: ModelProfile,
    proxy_url: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
    state_lock: threading.Lock,
    inflight_sem: threading.BoundedSemaphore,
    mark_dispatched: Callable[[], None],
    publish_page: Callable[[_PageState], None],
    executor: concurrent.futures.ThreadPoolExecutor,
    cancel_event: threading.Event,
) -> None:
    """Open the page's cost record, then dispatch region work. For empty
    or detector-failed states, publish the page immediately so the
    yielder still emits a ``RawOutput`` for it.

    ``open_page`` must precede the first ``executor.submit(...)`` —
    region workers won't pick up the task until after submit returns,
    so this ordering guarantees the cost recorder's page slot exists
    before any region's ``record_request`` callback fires.

    ``mark_dispatched`` is called BEFORE any region work could publish,
    so the main thread's ``pages_completed >= pages_dispatched`` check
    can never see completion racing ahead of dispatch.

    ``cancel_event`` is checked between every region's
    ``inflight_sem.acquire()`` so a Ctrl-C arriving mid-page (e.g.
    while we're 30 regions deep into a 100-region page) doesn't strand
    this worker on the semaphore until a future completes. Without
    this check, a worker mid-``_submit_state`` is invisible to the
    outer-loop ``cancel_event`` check and would hold up the bounded
    detector-thread join in ``run()``."""
    cost_mod.recorder.open_page(state.page.page_id, profile.name)
    mark_dispatched()

    if state.total_regions == 0:
        # Empty detection or detector error: nothing to submit; the
        # yielder will assemble an empty (or error-tagged) RawOutput.
        publish_page(state)
        return

    for position, region in enumerate(state.ordered):
        # Polling acquire so cancellation surfaces inside this loop.
        # The 0.5s timeout is short enough that even a 100-region page
        # gives up within a fraction of a second of ``cancel_event``
        # being set, but long enough that steady-state acquires don't
        # spin. The ``while/else`` returns when ``cancel_event`` flips
        # and we never see a successful ``break``.
        while not cancel_event.is_set():
            if inflight_sem.acquire(timeout=0.5):
                break
        else:
            return
        executor.submit(
            _make_region_worker(
                state=state,
                position=position,
                region=region,
                page=state.page,
                profile=profile,
                proxy_url=proxy_url,
                sampling=sampling,
                timeout=timeout,
                state_lock=state_lock,
                inflight_sem=inflight_sem,
                publish_page=publish_page,
            )
        )


def _make_region_worker(
    *,
    state: _PageState,
    position: int,
    region: LayoutRegion,
    page: PageImage,
    profile: ModelProfile,
    proxy_url: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
    state_lock: threading.Lock,
    inflight_sem: threading.BoundedSemaphore,
    publish_page: Callable[[_PageState], None],
) -> Callable[[], None]:
    """Build the closure submitted to the executor for one region.

    The closure runs ``_post_region``, releases the in-flight semaphore in
    ``finally`` (so even an unhandled error frees the slot), then mutates
    the captured ``state`` under ``state_lock``; when the page's last
    region completes, it publishes the state via ``publish_page``."""

    def _work() -> None:
        try:
            block = _post_region(
                region=region,
                page=page,
                profile=profile,
                proxy_url=proxy_url,
                sampling=sampling,
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001  defense-in-depth; _post_region catches its own
            block = _failed_block(region, error=f"{type(e).__name__}: {e}")
        finally:
            inflight_sem.release()
        with state_lock:
            state.results[position] = block
            state.remaining -= 1
            is_done = state.remaining == 0
        if is_done:
            publish_page(state)

    return _work


def _assemble_raw(
    state: _PageState,
    profile: ModelProfile,
    log_prefix: str,
    total_pages: int,
) -> RawOutput:
    """Build the per-page ``RawOutput`` from accumulated worker results."""
    page = state.page
    page_idx = state.page_idx

    if state.detector_error is not None:
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload="",
            error=f"layout detector failed: {state.detector_error}",
        )

    if state.total_regions == 0:
        log.info(
            "%s page %d/%d %s no regions detected",
            log_prefix, page_idx, total_pages, page.file_id,
        )
        return RawOutput(
            page_id=page.page_id,
            output_format=profile.output_format,
            payload=json.dumps([]),
        )

    blocks: list[dict[str, Any]] = []
    missing_positions: list[int] = []
    for position, slot in enumerate(state.results):
        if slot is None:
            # A worker silently dropped its result — this should be unreachable
            # given the lock + remaining-counter invariant. Filling with a
            # failed block keeps the page assembly going and surfaces the
            # diagnostic instead of crashing the whole run.
            missing_positions.append(position)
            blocks.append(
                _failed_block(
                    state.ordered[position],
                    error="worker dropped result (no completion signal)",
                )
            )
        else:
            blocks.append(slot)

    if missing_positions:
        log.error(
            "%s page %d/%d %s BUG: %d/%d region slot(s) are None at assembly "
            "time (positions=%s, region_ids=%s, remaining=%d). Filled with "
            "failed blocks; please report.",
            log_prefix, page_idx, total_pages, page.file_id,
            len(missing_positions), state.total_regions,
            missing_positions,
            [state.ordered[p].id for p in missing_positions],
            state.remaining,
        )

    n_failed = sum(1 for b in blocks if b.get("error"))
    elapsed = time.perf_counter() - state.t_start
    # Promote suspiciously slow pages to WARNING — see
    # ``_STALL_WARN_ELAPSED_S``. Pages that legitimately take this long
    # don't exist in practice (request timeout is 600s and pages run
    # regions concurrently), so an elapsed past 2× that is the strongest
    # signal we get that the consumer-side stalled while regions sat
    # ready in ``done_queue``. Surfacing it at WARNING means even ``-q``
    # runs see the problem.
    stalled = elapsed > _STALL_WARN_ELAPSED_S
    if n_failed:
        log.warning(
            "%s page %d/%d %s ok (%d/%d regions in %.1fs, %d failed)",
            log_prefix, page_idx, total_pages, page.file_id,
            state.total_regions - n_failed, state.total_regions, elapsed, n_failed,
        )
    elif stalled:
        log.warning(
            "%s page %d/%d %s ok (%d regions in %.1fs — STALE: elapsed > %.0fs, "
            "consumer likely stalled while regions waited in done_queue)",
            log_prefix, page_idx, total_pages, page.file_id,
            state.total_regions, elapsed, _STALL_WARN_ELAPSED_S,
        )
    else:
        log.info(
            "%s page %d/%d %s ok (%d regions in %.1fs)",
            log_prefix, page_idx, total_pages, page.file_id,
            state.total_regions, elapsed,
        )

    return RawOutput(
        page_id=page.page_id,
        output_format=profile.output_format,
        payload=json.dumps(blocks),
    )


def _post_region(
    *,
    region: LayoutRegion,
    page: PageImage,
    profile: ModelProfile,
    proxy_url: str,
    sampling: tuple[dict[str, Any], dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    """POST one cropped region through LiteLLM; return a layout-JSON block dict."""
    if page.image is None:
        return _failed_block(region, error="page has no in-memory image")

    crop = page.image.crop(region.bbox)
    prompt_template = _resolve_region_prompt(profile, region)
    prompt = _substitute_region_dims(prompt_template, crop)
    messages = _build_messages(crop, prompt)
    top_level, extra_body = sampling

    import litellm

    try:
        resp = litellm.completion(
            model=profile.name,
            custom_llm_provider="openai",
            api_base=proxy_url,
            api_key=os.environ.get("LITELLM_API_KEY", "ocrscout-dummy"),
            messages=messages,
            timeout=timeout,
            metadata={
                "page_id": page.page_id,
                "region_id": region.id,
                "model_name": profile.name,
            },
            extra_body=extra_body or None,
            **top_level,
        )
    except Exception as e:  # noqa: BLE001
        return _failed_block(region, error=f"{type(e).__name__}: {e}")

    text = _extract_text(resp)
    return _ok_block(region, text=text)


def _extract_text(resp: Any) -> str:
    try:
        choices = getattr(resp, "choices", None) or (
            resp.get("choices") if isinstance(resp, dict) else None
        )
        if not choices:
            return ""
        first = choices[0]
        msg = getattr(first, "message", None) or (
            first.get("message") if isinstance(first, dict) else None
        )
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return content or ""
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""


def _ok_block(region: LayoutRegion, *, text: str) -> dict[str, Any]:
    block: dict[str, Any] = {
        "category": region.category,
        "bbox": list(region.bbox),
        "text": text,
    }
    if region.score is not None:
        block["score"] = region.score
    return block


def _failed_block(region: LayoutRegion, *, error: str) -> dict[str, Any]:
    block = _ok_block(region, text="")
    block["error"] = error
    return block


def _resolve_region_prompt(profile: ModelProfile, region: LayoutRegion) -> str:
    """Pick the prompt template for a region.

    Lookup precedence:
        ``profile.prompt_mode_per_category[region.category]``
        → ``profile.preferred_prompt_mode``
        → first key in ``profile.prompt_templates`` (deterministic fallback).
    """
    templates = profile.prompt_templates
    mode = profile.prompt_mode_per_category.get(region.category)
    if mode is None or mode not in templates:
        mode = profile.preferred_prompt_mode
    if mode is None or mode not in templates:
        mode = next(iter(templates))
    return templates[mode]


def _substitute_region_dims(template: str, crop: Any) -> str:
    """Substitute ``{width}``/``{height}`` against the *region* dimensions."""
    if "{width}" not in template and "{height}" not in template:
        return template
    w, h = crop.size  # PIL .size is (width, height)
    return template.replace("{width}", str(w)).replace("{height}", str(h))


def _sort_reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Order regions for downstream document body assembly.

    If every region carries a non-None ``reading_order`` (the detector
    predicted it — e.g. PP-DocLayoutV3 emits results in reading order), use
    that. Otherwise fall back to a top-then-left bucketed sort that
    tolerates small same-row jitter on tightly-packed pages.
    """
    if regions and all(r.reading_order is not None for r in regions):
        return sorted(regions, key=lambda r: (r.reading_order or 0, r.bbox[1], r.bbox[0]))

    def heuristic_key(r: LayoutRegion) -> tuple[int, float]:
        top = r.bbox[1]
        left = r.bbox[0]
        return (int(round(top / _READING_ORDER_ROW_PX)) * _READING_ORDER_ROW_PX, left)

    return sorted(regions, key=heuristic_key)


def _list_proxy_models(proxy_url: str, *, timeout: float) -> set[str]:
    """GET ``{proxy_url}/models`` and return the served model_name set.

    Uses stdlib ``urllib`` instead of ``requests`` because this is a tiny
    one-shot probe; pulling ``requests`` in just to fail fast on URL
    misconfig isn't worth the dep weight in a backend that otherwise
    talks via ``litellm`` (which manages its own HTTP layer).
    """
    url = f"{proxy_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise BackendError(
            f"LayoutChatBackend: cannot reach LiteLLM proxy at {url}: {e}"
        ) from e
    try:
        return {entry["id"] for entry in data["data"]}
    except (KeyError, TypeError) as e:
        raise BackendError(
            f"LayoutChatBackend: malformed /models response from {url}: {e}"
        ) from e

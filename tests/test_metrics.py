"""MetricsCollector behavior + picklability."""

from __future__ import annotations

import pickle
import time

from ocrscout.metrics import MetricsCollector


def test_stage_timings_accumulate() -> None:
    m = MetricsCollector("p")
    with m.stage("load"):
        time.sleep(0.01)
    with m.stage("load"):
        time.sleep(0.01)
    with m.stage("inference"):
        time.sleep(0.01)
    assert m.stage_seconds["load"] >= 0.018  # two 10ms sleeps
    assert "inference" in m.stage_seconds


def test_counters() -> None:
    m = MetricsCollector("p")
    m.add_pages(ok=3, failed=1)
    m.add_tokens(123)
    m.record_gpu_peak(512.0)
    m.record_gpu_peak(256.0)
    assert m.pages_ok == 3 and m.pages_failed == 1
    assert m.tokens == 123
    assert m.gpu_peak_mb == 512.0  # max-merge


def test_merge_subprocess_metrics() -> None:
    a = MetricsCollector("p")
    b = MetricsCollector("p")
    with a.stage("inf"):
        time.sleep(0.005)
    with b.stage("inf"):
        time.sleep(0.005)
    a.add_pages(ok=1)
    b.add_pages(ok=2, failed=1)
    a.merge(b)
    assert a.pages_ok == 3 and a.pages_failed == 1
    assert a.stage_seconds["inf"] >= 0.008


def test_pickle_round_trip_preserves_state() -> None:
    m = MetricsCollector("pipe-1")
    with m.stage("load"):
        time.sleep(0.005)
    m.add_pages(ok=2)
    m.add_tokens(10)
    m.record_gpu_peak(42.0)

    data = pickle.dumps(m)
    revived = pickle.loads(data)
    assert revived.pipeline_id == "pipe-1"
    assert revived.pages_ok == 2
    assert revived.tokens == 10
    assert revived.gpu_peak_mb == 42.0
    assert revived.stage_seconds["load"] >= 0.004
    assert revived.started_at == m.started_at


def test_to_run_metrics_has_pages_per_hour() -> None:
    m = MetricsCollector("p")
    m.add_pages(ok=10)
    m.finish()
    rm = m.to_run_metrics()
    assert rm.pages_ok == 10
    pph = rm.pages_per_hour
    assert pph is None or pph >= 0

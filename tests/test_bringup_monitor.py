"""ServoHealthMonitor: escalation hooks, failures, skip-when-busy, lifecycle."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from so101 import JOINT_NAMES
from so101.hardware import health as h


def _samples(temperature: int = 30) -> dict[str, h.ServoHealthSample]:
    raw = {
        "Present_Temperature": temperature,
        "Present_Voltage": 50,
        "Present_Load": 0,
        "Present_Current": 10,
        "Status": 0,
    }
    return {
        joint: h.sample_from_raw(joint, raw, source="mock", timestamp=0.0) for joint in JOINT_NAMES
    }


class Reader:
    def __init__(self) -> None:
        self.temperature = 30
        self.fail = False
        self.calls = 0

    def __call__(self) -> dict[str, h.ServoHealthSample]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("bus timeout")
        return _samples(self.temperature)


def _wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_ok_tick_records_a_snapshot_without_callbacks() -> None:
    events: list[str] = []
    monitor = h.ServoHealthMonitor(
        Reader(),
        on_warn=lambda s: events.append("warn"),
        on_critical=lambda s: events.append("critical"),
        clock=lambda: 7.0,
    )
    snapshot = monitor.poll_once()
    assert snapshot is not None and snapshot.evaluation.level == "ok"
    assert snapshot.source == "mock" and snapshot.taken_at == 7.0
    assert monitor.latest == snapshot
    assert events == []


def test_critical_is_level_triggered_and_warn_is_edge_triggered() -> None:
    reader = Reader()
    warns: list[h.HealthSnapshot] = []
    criticals: list[h.HealthSnapshot] = []
    monitor = h.ServoHealthMonitor(reader, on_warn=warns.append, on_critical=criticals.append)
    reader.temperature = 56  # warn (fallback thresholds 55/65)
    monitor.poll_once()
    monitor.poll_once()
    assert len(warns) == 1
    reader.temperature = 70
    monitor.poll_once()
    monitor.poll_once()
    assert len(criticals) == 2
    assert criticals[0].previous_level == "warn"
    reader.temperature = 56
    monitor.poll_once()
    assert len(warns) == 2  # re-entering warn after critical notifies again


def test_handlers_can_be_registered_generically() -> None:
    reader = Reader()
    monitor = h.ServoHealthMonitor(reader)
    seen: list[str] = []

    @monitor.on_critical
    def _collapse_and_fault(snapshot: h.HealthSnapshot) -> None:
        seen.append(snapshot.evaluation.level)

    monitor.on_warn(lambda snapshot: seen.append("warn"))
    reader.temperature = 80
    monitor.poll_once()
    assert seen == ["critical"]


def test_read_failures_escalate_after_consecutive_misses() -> None:
    reader = Reader()
    criticals: list[h.HealthSnapshot] = []
    monitor = h.ServoHealthMonitor(
        reader,
        thresholds=h.HealthThresholds(max_consecutive_failures=3),
        on_critical=criticals.append,
    )
    reader.fail = True
    levels = []
    for _ in range(3):
        snapshot = monitor.poll_once()
        assert snapshot is not None
        levels.append(snapshot.evaluation.level)
    assert levels == ["warn", "warn", "critical"]
    assert snapshot.error is not None and "bus timeout" in snapshot.error
    assert snapshot.consecutive_failures == 3 and snapshot.samples == {}
    assert len(criticals) == 1
    reader.fail = False
    recovered = monitor.poll_once()
    assert recovered is not None and recovered.evaluation.level == "ok"
    assert recovered.consecutive_failures == 0
    assert monitor.stats()["read_failures"] == 3


def test_busy_bus_lock_skips_the_tick_without_reading() -> None:
    reader = Reader()
    lock = threading.Lock()
    monitor = h.ServoHealthMonitor(reader, lock=lock, lock_timeout_s=0.01)
    lock.acquire()
    try:
        started = time.monotonic()
        assert monitor.poll_once() is None
        assert time.monotonic() - started < 0.5
    finally:
        lock.release()
    assert reader.calls == 0
    assert monitor.stats()["skipped_busy"] == 1
    assert monitor.poll_once() is not None
    assert reader.calls == 1


def test_rlock_held_by_the_control_thread_also_skips() -> None:
    reader = Reader()
    lock = threading.RLock()
    held = threading.Event()
    release = threading.Event()

    def control_loop() -> None:
        with lock:
            held.set()
            release.wait(2.0)

    thread = threading.Thread(target=control_loop)
    thread.start()
    try:
        assert held.wait(2.0)
        monitor = h.ServoHealthMonitor(reader, lock=lock, lock_timeout_s=0.01)
        assert monitor.poll_once() is None
    finally:
        release.set()
        thread.join(2.0)
    assert reader.calls == 0


def test_handlers_run_after_the_bus_lock_is_released() -> None:
    reader = Reader()
    reader.temperature = 90
    lock = threading.Lock()
    lock_free_in_handler: list[bool] = []

    def handler(snapshot: h.HealthSnapshot) -> None:
        acquired = lock.acquire(blocking=False)
        lock_free_in_handler.append(acquired)
        if acquired:
            lock.release()

    monitor = h.ServoHealthMonitor(reader, lock=lock, on_critical=handler)
    monitor.poll_once()
    assert lock_free_in_handler == [True]


def test_a_failing_handler_is_counted_not_raised() -> None:
    reader = Reader()
    reader.temperature = 90

    def boom(snapshot: h.HealthSnapshot) -> None:
        raise RuntimeError("handler bug")

    monitor = h.ServoHealthMonitor(reader, on_critical=boom)
    assert monitor.poll_once() is not None
    assert monitor.stats()["callback_errors"] == 1


def test_start_and_stop_are_idempotent_and_restartable() -> None:
    reader = Reader()
    monitor = h.ServoHealthMonitor(reader, poll_hz=2.0)
    assert monitor.stop() is True  # never started
    assert monitor.start() is True
    assert monitor.start() is False
    assert _wait_for(lambda: reader.calls >= 1)  # first tick runs immediately
    assert monitor.running
    assert monitor.stop() is True
    assert monitor.stop() is True
    assert not monitor.running
    calls = reader.calls
    assert monitor.start() is True
    assert _wait_for(lambda: reader.calls > calls)
    assert monitor.stop() is True


def test_stop_from_a_handler_does_not_deadlock() -> None:
    reader = Reader()
    reader.temperature = 90
    stopped = threading.Event()
    monitor: h.ServoHealthMonitor

    def collapse(snapshot: h.HealthSnapshot) -> None:
        monitor.stop()
        stopped.set()

    monitor = h.ServoHealthMonitor(reader, on_critical=collapse)
    monitor.start()
    assert stopped.wait(3.0)
    assert _wait_for(lambda: not monitor.running)
    assert monitor.stop() is True


def test_the_thread_skips_while_control_holds_the_lock() -> None:
    reader = Reader()
    lock = threading.Lock()
    monitor = h.ServoHealthMonitor(reader, lock=lock, lock_timeout_s=0.01, poll_hz=2.0)
    lock.acquire()
    try:
        monitor.start()
        assert _wait_for(lambda: monitor.stats()["skipped_busy"] >= 1)
        assert reader.calls == 0
    finally:
        lock.release()
    assert _wait_for(lambda: reader.calls >= 1)
    assert monitor.stop() is True


@pytest.mark.parametrize("poll_hz", [0.5, 2.5, float("nan")])
def test_poll_rate_is_limited_to_one_or_two_hertz(poll_hz: float) -> None:
    with pytest.raises(ValueError):
        h.ServoHealthMonitor(Reader(), poll_hz=poll_hz)

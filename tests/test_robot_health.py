"""Servo health monitoring inside the server (docs/bring-up.md "Servo health monitor").

Contract: docs/plans/2026-10-01-suite-v02-design.md §2 (servo health) and §5
("Servo health critical -> Collapse + fault"). The ``so101-health`` thread polls
each arm whose bus the manager has open, at ``SO101_HEALTH_POLL_HZ`` on the
injected manager clock. WARN is an event and a status level; CRITICAL calls
``servo_health_critical``. Most tests hold the thread (``health_poll``) and the
watchdog (``supervise``) and drive one pass at a time on frozen time; the
thread tests poll against a deadline.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import JOINT_NAMES  # noqa: E402
from so101.config import get_settings  # noqa: E402
from so101.hardware.health import MockHealthModel  # noqa: E402
from so101.services import robot_manager as rm  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import RobotManager, get_manager, reset_manager  # noqa: E402
from so101.services.schemas import RobotStatus, ServoHealthResponse, Telemetry  # noqa: E402
from so101.services.store import get_store  # noqa: E402

OPERATOR = "test-operator"


def _events(kind: str) -> list[dict[str, Any]]:
    return [event for event in get_store().list_events(limit=500) if event["type"] == kind]


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _armed(manager: RobotManager) -> str:
    lease = manager.acquire_lease("health-test", 600)
    manager.arm_control(lease["id"])
    return str(lease["id"])


def _model(manager: RobotManager, arm: str = "follower") -> MockHealthModel:
    device = manager.follower() if arm == "follower" else manager.leader()
    return device.bus.health_model  # type: ignore[attr-defined,no-any-return]


def _arm_view(manager: RobotManager, arm: str = "follower") -> dict[str, Any]:
    return manager.health_snapshot()["arms"][arm]  # type: ignore[no-any-return]


@pytest.fixture
def manager(manager_clock: Any, supervise: Any) -> Iterator[RobotManager]:
    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    yield manager


# --- activation and visibility ---------------------------------------------------------


def test_polling_is_inactive_while_the_follower_bus_is_closed(manager, health_poll):
    health_poll()
    view = _arm_view(manager)
    assert view["active"] is False and "closed" in view["inactive_reason"]
    assert view["polls"] == 0 and view["level"] is None
    leader = _arm_view(manager, "leader")
    assert leader["active"] is False and "leader" in leader["inactive_reason"]
    summary = manager.status()["health"]
    assert summary == {
        "level": None,
        "active": False,
        "arms": {"follower": None, "leader": None},
        "age_s": None,
        "stale": False,
        "reasons": [],
    }


def test_arming_starts_polling_at_once_and_then_every_period(manager, manager_clock, health_poll):
    _armed(manager)
    health_poll()
    view = _arm_view(manager)
    assert view["active"] is True and view["inactive_reason"] is None
    assert view["polls"] == 1 and view["level"] == "ok" and view["source"] == "mock"
    assert set(view["motors"]) == set(JOINT_NAMES)
    sample = view["motors"]["elbow_flex"]["sample"]
    assert sample["voltage_v"] == pytest.approx(5.0, abs=0.2) and sample["status_raw"] == 0
    assert view["thresholds"]["supply_profile"] == "7.4v"

    health_poll()  # not due yet: the period runs on the manager clock
    assert _arm_view(manager)["polls"] == 1
    manager_clock.advance(1.0 / get_settings().health_poll_hz)
    health_poll()
    assert _arm_view(manager)["polls"] == 2
    manager_clock.advance(30.0)  # a stall polls once, with no burst catch-up
    health_poll()
    health_poll()
    assert _arm_view(manager)["polls"] == 3

    summary = manager.status()["health"]
    assert summary["active"] is True and summary["level"] == "ok"
    assert summary["arms"] == {"follower": "ok", "leader": None}
    telemetry = manager.telemetry()["health"]
    assert telemetry == summary
    RobotStatus(**manager.status())
    Telemetry(**manager.telemetry())


def test_sample_age_and_staleness_follow_the_manager_clock(manager, manager_clock, health_poll):
    _armed(manager)
    health_poll()
    manager_clock.advance(2.5)
    view = _arm_view(manager)
    assert view["sample_age_s"] == pytest.approx(2.5)
    assert manager.status()["health"]["stale"] is False
    manager_clock.advance(1.0)  # older than three 1 s periods
    assert manager.status()["health"]["stale"] is True
    assert manager.status()["health"]["age_s"] == pytest.approx(3.5)


def test_the_health_route_returns_the_snapshot(manager, health_poll):
    _armed(manager)
    health_poll()
    with TestClient(create_app()) as client:
        body = client.get("/api/robot/health").json()
        status = client.get("/api/robot/status").json()
    parsed = ServoHealthResponse(**body)
    assert parsed.active is True and parsed.level == "ok" and parsed.poller_running is True
    assert parsed.poll_hz == get_settings().health_poll_hz and parsed.io_wait_ms == 50.0
    follower = parsed.arms["follower"]
    assert follower.motors["gripper"].sample.motor == "gripper"
    assert follower.sampled_at is not None and follower.thresholds is not None
    assert parsed.unit_provenance["Present_Voltage"].status == "unverified"
    assert status["health"]["arms"]["follower"] == "ok"


def test_a_new_session_starts_a_fresh_monitor(manager, manager_clock, health_poll):
    lease = _armed(manager)
    _model(manager).fail_next_reads(2)
    health_poll()
    manager_clock.advance(1.0)
    health_poll()
    assert _arm_view(manager)["consecutive_failures"] == 2
    manager.release_lease(lease)
    manager.request_stop("collapse", reason="test: end of session")
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    health_poll()
    assert _arm_view(manager)["active"] is False
    assert _arm_view(manager)["consecutive_failures"] == 2  # kept for the record

    _armed(manager)
    manager_clock.advance(1.0)
    health_poll()
    view = _arm_view(manager)
    assert view["activations"] == 2 and view["consecutive_failures"] == 0
    assert view["level"] == "ok" and manager.fault is None


# --- WARN --------------------------------------------------------------------------------


def test_a_warning_is_an_event_and_a_status_level_but_never_a_stop(
    manager, manager_clock, health_poll
):
    _armed(manager)
    _model(manager).set_override("elbow_flex", load_pct=85.0)
    health_poll()

    status = manager.status()
    assert status["health"]["level"] == "warn"
    assert any(
        "elbow_flex" in reason and "load" in reason for reason in status["health"]["reasons"]
    )
    assert manager.control_state()["armed"] is True and status["stop_state"] == "none"
    assert manager.fault is None
    (event,) = _events("robot.health.warn")
    assert event["level"] == "warning" and event["detail"]["arm"] == "follower"
    assert event["detail"]["reasons"][0]["category"] == "load"

    manager_clock.advance(1.0)
    health_poll()  # still warn: no second event
    assert len(_events("robot.health.warn")) == 1

    _model(manager).clear_overrides()
    manager_clock.advance(1.0)
    health_poll()
    assert manager.status()["health"]["level"] == "ok"
    (recovered,) = _events("robot.health.ok")
    assert recovered["detail"] == {"arm": "follower", "previous_level": "warn"}


def test_one_failed_read_is_a_warning(manager, health_poll):
    _armed(manager)
    _model(manager).fail_next_reads(1)
    health_poll()
    view = _arm_view(manager)
    assert view["level"] == "warn" and view["consecutive_failures"] == 1
    assert "injected mock health read failure" in view["error"]
    assert view["read_failures"] == 1 and view["motors"] == {}
    assert manager.control_state()["armed"] is True


# --- CRITICAL ----------------------------------------------------------------------------


def test_critical_temperature_collapses_and_latches_a_fault(manager, health_poll):
    _armed(manager)
    _model(manager).set_override("wrist_flex", temperature_c=66.0)  # critical at 70 - 5
    health_poll()

    assert manager.fault is not None and manager.fault.startswith("servo health critical")
    assert "follower: wrist_flex: temperature 66" in manager.fault
    assert manager.control_state()["armed"] is False
    (critical,) = _events("safety.servo_health_critical")
    assert critical["level"] == "critical" and critical["detail"]["arm"] == "follower"
    assert critical["detail"]["reasons"][0]["category"] == "temperature"
    assert _events("safety.collapse.start")
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    assert manager.status()["torque_state"] == "off"
    assert manager.fault is not None  # only the deliberate reset clears it
    assert _arm_view(manager)["critical_dispatches"] == 1
    assert _arm_view(manager)["level"] == "critical"


def test_consecutive_read_failures_go_critical(manager, manager_clock, health_poll):
    _armed(manager)
    max_failures = get_settings().health_max_consecutive_failures
    _model(manager).fail_next_reads(max_failures)
    for _ in range(max_failures - 1):
        health_poll()
        manager_clock.advance(1.0)
        assert manager.fault is None and _arm_view(manager)["level"] == "warn"
    health_poll()
    assert manager.fault is not None
    assert f"health read failed ({max_failures} consecutive)" in manager.fault
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    assert manager.status()["torque_state"] == "off"


def test_low_supply_voltage_is_critical(manager, health_poll):
    _armed(manager)
    _model(manager).set_override("shoulder_pan", voltage_v=3.9)  # 7.4v band: critical <= 4.0
    health_poll()
    assert manager.fault is not None and "supply 3.9 V" in manager.fault


class _ParkingSleep:
    """A manager sleep that parks the collapse descent until released."""

    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.parked = threading.Event()
        self.go = threading.Event()

    def __call__(self, seconds: float) -> None:
        self.clock.now += max(0.0, seconds)
        self.parked.set()
        self.go.wait(10)


def test_a_repeat_critical_during_its_own_collapse_does_not_dispatch_again(
    manager, manager_clock, health_poll, monkeypatch
):
    gate = _ParkingSleep(manager_clock)
    monkeypatch.setattr(manager, "_sleep", gate)
    _armed(manager)
    _model(manager).set_override("gripper", temperature_c=67.0)
    try:
        health_poll()
        assert gate.parked.wait(5)
        assert manager.status()["stop_state"] == "collapsing"
        manager_clock.advance(1.0)
        health_poll()  # still critical, the descent still runs
        assert _arm_view(manager)["critical_dispatches"] == 1
        assert len(_events("safety.servo_health_critical")) == 1
    finally:
        gate.go.set()
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")


def test_critical_during_an_operator_collapse_still_latches_the_fault(
    manager, manager_clock, health_poll, monkeypatch
):
    gate = _ParkingSleep(manager_clock)
    monkeypatch.setattr(manager, "_sleep", gate)
    _armed(manager)
    health_poll()  # the arm is polled before the operator's Collapse starts
    try:
        assert manager.request_stop("collapse", reason="operator collapse").collapsing
        assert gate.parked.wait(5)
        _model(manager).set_override("gripper", temperature_c=67.0)
        manager_clock.advance(1.0)
        health_poll()
        assert manager.fault is not None and manager.fault.startswith("servo health critical")
    finally:
        gate.go.set()
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    assert manager.fault is not None


def test_critical_while_e_stopped_dispatches_nothing(manager):
    from so101.hardware.health import HealthEvaluation, HealthReason, HealthSnapshot

    reason = HealthReason(category="temperature", level="critical", message="hot")
    snapshot = HealthSnapshot(
        arm="follower",
        taken_at=time.time(),
        evaluation=HealthEvaluation(level="critical", reasons=(reason,), evaluated_at=time.time()),
    )
    manager.estop()
    manager._on_health_critical("follower", snapshot)
    assert _events("safety.servo_health_critical") == []
    assert _arm_view(manager)["critical_dispatches"] == 0


# --- the bus lock -------------------------------------------------------------------------


def test_one_arm_is_read_per_pass(manager, health_poll):
    """A loop never waits behind a follower read and a leader read back to back."""
    _armed(manager)
    leader = manager.leader()
    leader.bus.connect()  # as teleop and recording open it
    try:
        health_poll()
        assert (_arm_view(manager)["polls"], _arm_view(manager, "leader")["polls"]) == (1, 0)
        health_poll()
        assert (_arm_view(manager)["polls"], _arm_view(manager, "leader")["polls"]) == (1, 1)
        assert _arm_view(manager, "leader")["level"] == "ok"
    finally:
        leader.bus.disconnect()


def test_a_busy_bus_skips_the_poll_and_retries_soon(manager, manager_clock, health_poll):
    _armed(manager)
    manager.health_io_wait_s = 0.01
    assert manager._io_lock.acquire(timeout=1)
    try:
        health_poll()
    finally:
        manager._io_lock.release()
    view = _arm_view(manager)
    assert view["skipped_busy"] == 1 and view["polls"] == 0 and view["read_failures"] == 0
    assert "stayed busy for 10 ms" in view["last_skip"]
    manager_clock.advance(0.2)  # the retry comes well before a whole period
    health_poll()
    assert _arm_view(manager)["polls"] == 1


def test_a_bus_that_closed_while_the_poll_waited_is_not_a_failure(manager, health_poll):
    _armed(manager)
    health_poll()
    state = manager._health["follower"]
    follower = manager.follower()
    manager.health_io_wait_s = 5.0
    manager._io_lock.acquire()
    try:
        follower.bus._connected = False  # type: ignore[attr-defined]  # closed under the poll
        thread = threading.Thread(
            target=manager._health_poll, args=("follower", follower, state), daemon=True
        )
        state.next_due = float("-inf")
        thread.start()
        time.sleep(0.01)
    finally:
        manager._io_lock.release()
    thread.join(5)
    assert _arm_view(manager)["polls"] == 1 and _arm_view(manager)["read_failures"] == 0
    follower.bus._connected = True  # type: ignore[attr-defined]


def test_the_poll_reads_under_the_bus_lock_and_handles_after_it(manager, health_poll, monkeypatch):
    released: list[bool] = []
    # The monitor binds its handlers when the arm activates, so patch first.
    monkeypatch.setattr(
        manager, "_on_health_warn", lambda arm, snap: released.append(manager._io_lock.locked())
    )
    _armed(manager)
    bus = manager.follower().bus
    original = bus.read_health  # type: ignore[attr-defined]
    held: list[bool] = []

    def spy() -> Any:
        held.append(manager._io_lock.locked())
        return original()

    monkeypatch.setattr(bus, "read_health", spy)
    _model(manager).set_override("elbow_flex", load_pct=90.0)
    health_poll()
    assert held == [True] and released == [False]


# --- inactive cases ------------------------------------------------------------------------


def test_the_e_stop_owns_the_bus(manager, health_poll):
    _armed(manager)
    manager.estop()
    health_poll()
    view = _arm_view(manager)
    assert view["active"] is False and "E-stop" in view["inactive_reason"]


def test_the_sim_backend_is_cleanly_inactive(manager, health_poll, monkeypatch):
    monkeypatch.setattr(rm, "current_backend", lambda settings: "sim")
    health_poll()
    for arm in ("follower", "leader"):
        view = _arm_view(manager, arm)
        assert view["active"] is False and "SO101_BACKEND=sim" in view["inactive_reason"]
        assert view["level"] is None
    assert manager.fault is None


def test_a_bus_without_health_registers_is_inactive(manager, health_poll, monkeypatch):
    class Silent:
        connected = False
        torque_state = "off"

    monkeypatch.setattr(manager.follower(), "bus", Silent())
    health_poll()
    assert "no servo health registers" in _arm_view(manager)["inactive_reason"]


def test_invalid_thresholds_switch_monitoring_off_visibly(monkeypatch):
    monkeypatch.setenv("SO101_HEALTH_TEMP_CRITICAL_MARGIN_C", "20")  # above the warn margin
    get_settings.cache_clear()
    reset_manager()
    manager = get_manager()
    view = _arm_view(manager)
    assert view["active"] is False and "invalid health thresholds" in view["inactive_reason"]
    assert view["thresholds"] is None
    (event,) = [e for e in _events("robot.health.config_error") if e["detail"]["arm"] == "follower"]
    assert event["level"] == "error"


# --- the real thread -------------------------------------------------------------------------


def test_the_health_thread_polls_an_armed_follower():
    manager = get_manager()
    assert manager.health_snapshot()["poller_running"] is True
    _armed(manager)
    assert _wait_for(lambda: _arm_view(manager)["polls"] >= 1, 5.0)
    assert manager.status()["health"]["arms"]["follower"] == "ok"


def test_the_health_thread_interleaves_with_a_streaming_loop(monkeypatch):
    monkeypatch.setenv("SO101_HEALTH_POLL_HZ", "2")
    get_settings.cache_clear()
    reset_manager()
    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    manager.start_teleop(30, operator_id=OPERATOR)
    try:

        def polled() -> bool:
            manager.operator_heartbeat(OPERATOR)
            return _arm_view(manager)["polls"] >= 3 and _arm_view(manager, "leader")["polls"] >= 3

        assert _wait_for(polled, 10.0)
        seq = manager.telemetry()["sample_seq"]
        assert _wait_for(lambda: manager.telemetry()["sample_seq"] >= seq + 10, 5.0)
    finally:
        assert manager.soft_stop(reason="test end")
    follower, leader = _arm_view(manager), _arm_view(manager, "leader")
    assert follower["level"] == "ok" and leader["level"] == "ok"
    assert follower["read_failures"] == 0 and leader["read_failures"] == 0
    assert manager.fault is None


def test_shutdown_stops_the_health_thread():
    manager = get_manager()
    thread = manager._health_thread
    assert thread is not None and thread.is_alive()
    manager.shutdown()
    assert not thread.is_alive()
    assert manager.health_snapshot()["poller_running"] is False


# --- the synthetic model ------------------------------------------------------------------------


def test_an_hour_of_mock_teleop_never_warns():
    """The always-on monitor must not raise false warnings in a rehearsal."""
    import math

    from so101.hardware.health import HealthThresholds, evaluate

    now = [0.0]
    model = MockHealthModel(clock=lambda: now[0])
    thresholds = HealthThresholds()
    for second in range(3600):
        now[0] = float(second)
        t = float(second)
        goals = {  # the synthetic mock leader's trajectory (hardware/leader.py)
            "shoulder_pan": 40 * math.sin(t * 0.5),
            "shoulder_lift": 30 * math.sin(t * 0.4 + 1),
            "elbow_flex": 35 * math.sin(t * 0.6 + 2),
            "wrist_flex": 25 * math.sin(t * 0.7),
            "wrist_roll": 60 * math.sin(t * 0.3),
            "gripper": 50 + 50 * math.sin(t * 0.9),
        }
        samples = model.sample(goals, goals, torque=True)
        if second % 600 == 0 or second == 3599:
            assert evaluate(samples, thresholds).level == "ok", second
    assert max(sample.temperature_c for sample in samples.values()) < 50.0


def test_the_latency_probe_refuses_without_the_mock_backend_and_measures_with_it():
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_health_latency.py"
    env = {key: value for key, value in os.environ.items() if not key.startswith("SO101_")}
    refused = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=60
    )
    assert refused.returncode == 2 and "SO101_MOCK=1" in refused.stderr
    done = subprocess.run(
        [sys.executable, str(script), "--seconds", "1.2"],
        env={**env, "SO101_MOCK": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "health polling:" in done.stdout and "added worst-case loop wait" in done.stdout

"""Collapse (category 1): a slow fall to the rest pose, then torque off.

Contract: docs/plans/2026-10-01-suite-v02-design.md §5 "Collapse" and
"Precedence". Every test runs on the mock bus at the shipped speed (15 deg/s)
and rate (30 Hz), on the injected manager clock: the descent's pacing sleeps
move that clock, so a descent takes milliseconds of wall time. The physics
version of these checks is tests/test_sim_stops_physics.py.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import JOINT_NAMES  # noqa: E402
from so101.config import get_settings  # noqa: E402
from so101.hardware.rest_pose import RestPoseError  # noqa: E402
from so101.hardware.safety import SafetyViolation  # noqa: E402
from so101.services import robot_manager as rm  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlConflict,
    ControlRejected,
    RobotManager,
    StopKind,
    get_manager,
    reset_manager,
)
from so101.services.store import get_store  # noqa: E402
from so101.simulation.rest_pose import MODEL_REST_POSE  # noqa: E402

OPERATOR = "test-operator"
ARM = [joint for joint in JOINT_NAMES if joint != "gripper"]
SPEED = 15.0
RATE = 30.0


@pytest.fixture
def shipped_speed(monkeypatch):
    """The shipped 15 deg/s and 30 Hz (conftest speeds mock descents up)."""

    monkeypatch.setenv("SO101_COLLAPSE_SPEED_DEG_S", str(SPEED))
    monkeypatch.setattr(RobotManager, "collapse_rate_hz", RATE)
    get_settings.cache_clear()
    reset_manager()


@pytest.fixture
def client(shipped_speed, manager_clock):
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


class Gate:
    """A manager sleep that parks the descent after ``park_after`` ticks.

    Fake time still advances by exactly what was asked, so the descent's
    schedule is unchanged; the test decides when it may continue.
    """

    def __init__(self, clock: Any, park_after: int) -> None:
        self.clock = clock
        self.park_after = park_after
        self.calls = 0
        self.parked = threading.Event()
        self.go = threading.Event()

    def sleep(self, seconds: float) -> None:
        self.clock.now += max(0.0, seconds)
        self.calls += 1
        if self.calls >= self.park_after and not self.go.is_set():
            self.parked.set()
            self.go.wait(10)


def _armed(manager: RobotManager) -> str:
    lease = manager.acquire_lease("collapse-test", 60)
    manager.arm_control(lease["id"])
    return lease["id"]


def _events(kind: str) -> list[dict[str, Any]]:
    return [event for event in get_store().list_events(limit=500) if event["type"] == kind]


def _wait_for(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _finished(manager: RobotManager) -> bool:
    return manager.status()["stop_state"] == "none"


def _spy_goals(manager: RobotManager, monkeypatch, clock) -> list[tuple[float, dict[str, float]]]:
    follower = manager.follower()
    original = follower.send_goal
    goals: list[tuple[float, dict[str, float]]] = []

    def spy(goal: dict[str, float], *, dt: float | None = None) -> dict[str, float]:
        goals.append((clock.now, dict(goal)))
        return original(goal, dt=dt)

    monkeypatch.setattr(follower, "send_goal", spy)
    return goals


# --- the descent -----------------------------------------------------------------------


def test_collapse_lowers_to_the_rest_pose_at_the_cap_then_turns_torque_off(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    lease_id = _armed(manager)
    bus = manager.follower().bus
    gripper_before = bus.read_positions()["gripper"]
    goals = _spy_goals(manager, monkeypatch, manager_clock)

    response = client.post("/api/robot/collapse", json={"reason": "done for the day"})

    assert response.status_code == 200, response.text
    assert "collapsing" in response.json()["detail"]
    assert _wait_for(lambda: _finished(manager))
    # Every joint stays within the cap: the setpoint moves at most speed/rate
    # per tick, all joints arriving together (joint-space interpolation).
    step_limit = SPEED / RATE + 1e-9
    for (t0, before), (t1, after) in zip(goals, goals[1:], strict=False):
        assert t1 - t0 == pytest.approx(1 / RATE)
        for joint in ARM:
            assert abs(after[joint] - before[joint]) <= step_limit, joint
    pairs = list(zip(goals, goals[1:], strict=False))
    peak = {
        joint: max(abs(b[joint] - a[joint]) * RATE for (_, a), (_, b) in pairs) for joint in ARM
    }
    assert max(peak.values()) == pytest.approx(SPEED, rel=1e-6)  # the farthest joint at the cap
    # It ends at rest, with torque confirmed off (C2 read-back).
    final = dict(bus._pos)  # the port is closed; the mock arm's true pose
    for joint in ARM:
        assert final[joint] == pytest.approx(MODEL_REST_POSE[joint], abs=1.0), joint
    assert bus.torque_enabled is False and bus.connected is False
    status = manager.status()
    assert status["torque_state"] == "off" and status["collapse"] is None
    # The gripper follows the Halt grip rule throughout: it never opens.
    gripper_goals = {goal["gripper"] for _, goal in goals}
    assert len(gripper_goals) == 1 and gripper_goals.pop() <= gripper_before + 0.25
    (start,) = _events("safety.collapse.start")
    assert start["detail"]["reason"] == "done for the day"
    assert start["detail"]["rest_source"] == "default"
    assert start["detail"]["speed_deg_s"] == SPEED
    distance = start["detail"]["distance_deg"]
    expected_timeout = min(distance / SPEED + 2.0, 12.0)
    assert start["detail"]["timeout_s"] == pytest.approx(expected_timeout, abs=1e-3)
    (done,) = _events("safety.collapse.done")
    assert done["detail"]["torque_off"] is True and done["detail"]["final_error_deg"] <= 1.0
    # Not latched: no fault, and the arm can be armed again at once.
    assert manager.fault is None
    assert manager.arm_control(lease_id)["armed"] is True


class LateTicks:
    """A manager sleep that makes chosen descent ticks late (a scheduler stall).

    Fake time advances by what was asked plus ``extra`` seconds on every call
    numbered in ``late``, as if the thread woke up that much too late.
    """

    def __init__(self, clock: Any, late: set[int] | None = None, *, extra: float) -> None:
        self.clock = clock
        self.late = late
        self.extra = extra
        self.calls = 0

    def sleep(self, seconds: float) -> None:
        self.calls += 1
        late = self.late is None or self.calls in self.late
        self.clock.now += max(0.0, seconds) + (self.extra if late else 0.0)
        time.sleep(0.0005)


class SlowBusLock:
    """The manager's bus lock, with chosen acquisitions taking ``wait_s`` of fake time.

    Models a descent tick that waited for the bus (another holder) but still
    got it within ``stop_io_timeout_s``.
    """

    def __init__(self, lock: Any, clock: Any, *, slow: set[int], wait_s: float) -> None:
        self._inner = lock
        self.clock = clock
        self.slow = slow
        self.wait_s = wait_s
        self.collapse_acquisitions = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if threading.current_thread().name == "so101-collapse":
            self.collapse_acquisitions += 1
            if self.collapse_acquisitions in self.slow:
                self.clock.now += self.wait_s
        return self._inner.acquire(blocking, timeout)

    def release(self) -> None:
        self._inner.release()

    def locked(self) -> bool:
        return self._inner.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _spy_commanded(manager: RobotManager, monkeypatch) -> list[dict[str, float]]:
    """Every goal the descent sent to ``send_goal`` and what the servos accepted."""

    follower = manager.follower()
    original = follower.send_goal
    accepted: list[dict[str, float]] = []

    def spy(goal: dict[str, float], *, dt: float | None = None) -> dict[str, float]:
        commanded = original(goal, dt=dt)
        accepted.append(dict(commanded))
        return commanded

    monkeypatch.setattr(follower, "send_goal", spy)
    return accepted


def _assert_every_step_within_the_cap(goals: list[dict[str, float]]) -> None:
    step_limit = SPEED / RATE + 1e-9
    assert len(goals) > 2
    for index, (before, after) in enumerate(zip(goals, goals[1:], strict=False)):
        for joint in ARM:
            step = abs(after[joint] - before[joint])
            assert step <= step_limit, f"tick {index + 1}: {joint} stepped {step:.3f} deg"


@pytest.mark.parametrize("delay_s", [0.1, 0.45, 1.0], ids=["100ms", "450ms", "1s"])
def test_a_late_tick_never_makes_the_descent_catch_up(
    client: TestClient, manager_clock, monkeypatch, delay_s: float
):
    """A stalled descent thread resumes at the cap: it does not jump to where the clock says."""

    manager = get_manager()
    _armed(manager)
    manager._sleep = LateTicks(manager_clock, {10}, extra=delay_s).sleep
    goals = _spy_goals(manager, monkeypatch, manager_clock)
    accepted = _spy_commanded(manager, monkeypatch)

    assert manager.request_stop(StopKind.COLLAPSE, reason="late tick").collapsing
    assert _wait_for(lambda: _finished(manager))

    _assert_every_step_within_the_cap([goal for _, goal in goals])
    _assert_every_step_within_the_cap(accepted)
    # The stall made the descent longer, and it still finished at rest.
    (done,) = _events("safety.collapse.done")
    assert done["detail"]["torque_off"] is True
    (start,) = _events("safety.collapse.start")
    assert done["detail"]["duration_s"] >= start["detail"]["duration_s"] + delay_s - 1e-3
    assert _events("safety.collapse.fallback") == []


def test_a_tick_that_waited_for_the_bus_never_makes_the_descent_catch_up(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    _armed(manager)
    # Just inside stop_io_timeout_s (0.5 s), so the descent continues.
    slow = SlowBusLock(manager._io_lock, manager_clock, slow={8, 9, 20}, wait_s=0.45)
    monkeypatch.setattr(manager, "_io_lock", slow)
    goals = _spy_goals(manager, monkeypatch, manager_clock)

    assert manager.request_stop(StopKind.COLLAPSE, reason="busy bus").collapsing
    assert _wait_for(lambda: _finished(manager))

    assert slow.collapse_acquisitions >= 20
    _assert_every_step_within_the_cap([goal for _, goal in goals])
    assert _events("safety.collapse.done") and _events("safety.collapse.fallback") == []


def test_a_goal_the_driver_clipped_is_not_made_up_in_one_jump(
    client: TestClient, manager_clock, monkeypatch
):
    """The per-joint clamp: each step is capped from what the servos last accepted."""

    manager = get_manager()
    _armed(manager)
    bus = manager.follower().bus
    original = bus.write_positions
    writes = {"n": 0}
    held: dict[str, float] = {}

    def clipping(goal: dict[str, float]) -> dict[str, float]:
        # Like a driver's relative-target clip: these ticks' arm goals do not
        # move, and the driver reports the goal it really wrote.
        writes["n"] += 1
        if writes["n"] in (6, 7, 8):
            goal = {**goal, **held}
        commanded = original(goal)
        held.update({joint: commanded[joint] for joint in ARM if joint in commanded})
        return commanded

    monkeypatch.setattr(bus, "write_positions", clipping)
    accepted = _spy_commanded(manager, monkeypatch)

    assert manager.request_stop(StopKind.COLLAPSE, reason="clipped").collapsing
    assert _wait_for(lambda: _finished(manager))

    _assert_every_step_within_the_cap(accepted)
    assert _events("safety.collapse.done")


def test_delays_past_the_bound_end_in_the_timeout_fallback_at_the_cap(
    client: TestClient, manager_clock, monkeypatch
):
    """The bound stays on the real clock: a descent that keeps stalling is cut, not hurried."""

    manager = get_manager()
    _armed(manager)
    manager._sleep = LateTicks(manager_clock, None, extra=0.3).sleep  # every tick late
    goals = _spy_goals(manager, monkeypatch, manager_clock)

    assert manager.request_stop(StopKind.COLLAPSE, reason="stalling").collapsing
    assert _wait_for(lambda: _finished(manager))

    _assert_every_step_within_the_cap([goal for _, goal in goals])
    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "timeout" and fallback["detail"]["torque_off"] is True
    assert manager.fault is not None and "timeout" in manager.fault
    assert manager.follower().bus.torque_enabled is False
    assert _events("safety.collapse.done") == []


def test_status_and_telemetry_describe_a_running_collapse(client: TestClient, manager_clock):
    manager = get_manager()
    _armed(manager)
    gate = Gate(manager_clock, park_after=30)
    manager._sleep = gate.sleep

    assert manager.request_stop(StopKind.COLLAPSE, reason="watch me").collapsing
    assert gate.parked.wait(5)
    try:
        status = manager.status()
        frame = manager.telemetry()
    finally:
        gate.go.set()

    assert status["stop_state"] == "collapsing" and status["torque_state"] == "on"
    collapse = status["collapse"]
    assert collapse["reason"] == "watch me"
    assert 0.0 < collapse["progress"] < 1.0
    assert collapse["eta_s"] > 0
    datetime.fromisoformat(collapse["since"])
    assert frame["stop_state"] == "collapsing" and frame["collapse"]["reason"] == "watch me"
    assert frame["observation_live"] is True
    assert _wait_for(lambda: _finished(manager))


def test_collapse_with_the_motors_off_is_a_no_op(client: TestClient):
    manager = get_manager()

    result = manager.request_stop(StopKind.COLLAPSE, reason="idle")

    assert result.ok and not result.collapsing and result.torque_off
    assert _events("safety.collapse.start") == []
    assert manager.follower().bus.torque_enabled is False


def test_nothing_starts_while_estopped(client: TestClient):
    manager = get_manager()
    _armed(manager)
    manager.estop()

    result = manager.request_stop(StopKind.COLLAPSE, reason="after estop")

    assert result.applied and not result.collapsing and result.torque_off
    assert _events("safety.collapse.start") == []


def test_a_repeat_collapse_joins_the_running_one(client: TestClient, manager_clock):
    manager = get_manager()
    _armed(manager)
    gate = Gate(manager_clock, park_after=5)
    manager._sleep = gate.sleep
    assert manager.request_stop(StopKind.COLLAPSE, reason="first").collapsing
    assert gate.parked.wait(5)
    try:
        again = manager.request_stop(StopKind.COLLAPSE, reason="second")
    finally:
        gate.go.set()

    assert again.ok and again.collapsing
    assert _wait_for(lambda: _finished(manager))
    assert len(_events("safety.collapse.start")) == 1


def test_an_owner_scoped_collapse_with_a_foreign_lease_changes_nothing(client: TestClient):
    manager = get_manager()
    _armed(manager)

    result = manager.request_stop(StopKind.COLLAPSE, reason="stale", lease_id="not-the-lease")

    assert result.applied is False
    assert manager.control_state()["armed"] is True


# --- fallbacks -------------------------------------------------------------------------


def test_a_timeout_falls_back_to_torque_off_with_a_fault(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)
    monkeypatch.setattr(manager.settings, "collapse_timeout_seconds", 1.0)

    assert manager.request_stop(StopKind.COLLAPSE, reason="short bound").collapsing
    assert _wait_for(lambda: _finished(manager))

    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "timeout" and fallback["detail"]["torque_off"] is True
    assert manager.fault is not None and "timeout" in manager.fault
    assert manager.follower().bus.torque_enabled is False
    assert _events("safety.collapse.done") == []


def test_a_stall_falls_back(client: TestClient):
    manager = get_manager()
    _armed(manager)
    manager.follower().bus.smoothing = 0.0  # the arm stops following its goal

    assert manager.request_stop(StopKind.COLLAPSE, reason="blocked").collapsing
    assert _wait_for(lambda: _finished(manager))

    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "stall"
    assert "no progress for 0.5 s" in fallback["detail"]["detail"]
    assert manager.fault is not None


def test_a_bus_error_mid_descent_falls_back(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)
    follower = manager.follower()
    original = follower.read_positions
    calls = {"n": 0}

    def flaky() -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] > 6:
            raise ConnectionError("no status packet")
        return original()

    monkeypatch.setattr(follower, "read_positions", flaky)

    assert manager.request_stop(StopKind.COLLAPSE, reason="flaky").collapsing
    assert _wait_for(lambda: _finished(manager))

    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "bus_error"
    assert follower.bus.torque_enabled is False and manager.fault is not None


def test_a_dead_bus_fails_the_request_itself(client: TestClient):
    manager = get_manager()
    lease_id = _armed(manager)
    manager.follower().bus.fail_io = True  # torque-off cannot confirm either

    response = client.post("/api/control/disarm", json={"lease_id": lease_id})

    assert response.status_code == 409, response.text
    assert "cut physical power" in response.json()["detail"]
    status = manager.status()
    assert status["torque_unconfirmed"] is True and status["stop_state"] == "none"
    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "bus_error"


def test_a_guard_rejection_mid_descent_falls_back(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)
    guard = manager.follower().guard
    original = guard.clamp
    calls = {"n": 0}

    def rejecting(goal, *, dt=None):
        calls["n"] += 1
        if calls["n"] > 3:
            raise SafetyViolation("non-finite command for elbow_flex")
        return original(goal, dt=dt)

    monkeypatch.setattr(guard, "clamp", rejecting)

    assert manager.request_stop(StopKind.COLLAPSE, reason="rejected").collapsing
    assert _wait_for(lambda: _finished(manager))

    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "guard_rejection"
    assert manager.fault is not None


def test_an_arm_far_outside_the_envelope_is_a_guard_rejection_at_once(client: TestClient):
    manager = get_manager()
    _armed(manager)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    bus = manager.follower().bus
    bus._pos["shoulder_lift"] = bus._goal["shoulder_lift"] = -135.0  # 35 deg past -100

    response = client.post("/api/robot/collapse")

    assert response.status_code == 503, response.text
    assert "guard_rejection" in response.json()["detail"]
    assert bus.torque_enabled is False and manager.fault is not None


def test_without_a_rest_pose_the_arm_is_de_energized_where_it_stands(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _armed(manager)

    def unresolvable(settings, *, limits=None):
        raise RestPoseError("the SafetyGuard and model ranges do not overlap")

    monkeypatch.setattr(rm, "resolve_collapse_rest_pose", unresolvable)

    result = manager.request_stop(StopKind.COLLAPSE, reason="no rest")

    assert result.fallback == "no_rest_pose" and result.torque_off
    assert manager.follower().bus.torque_enabled is False
    assert manager.fault is not None and "where it stood" in manager.fault
    assert _events("safety.collapse.start") == []


def test_the_watchdog_backstop_cuts_a_descent_stuck_in_the_driver(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    _armed(manager)
    gate = Gate(manager_clock, park_after=3)
    manager._sleep = gate.sleep
    assert manager.request_stop(StopKind.COLLAPSE, reason="stuck").collapsing
    assert gate.parked.wait(5)
    try:
        deadline = manager._collapse.deadline
        manager_clock.now = deadline + 1.5
        supervise()
    finally:
        gate.go.set()

    assert _finished(manager)
    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "timeout" and "stuck" in fallback["detail"]["detail"]
    assert manager.follower().bus.torque_enabled is False


# --- precedence --------------------------------------------------------------------------


def test_estop_preempts_a_collapse_instantly(client: TestClient, manager_clock, monkeypatch):
    manager = get_manager()
    _armed(manager)
    gate = Gate(manager_clock, park_after=10)
    manager._sleep = gate.sleep
    assert manager.request_stop(StopKind.COLLAPSE, reason="lowering").collapsing
    assert gate.parked.wait(5)
    thread = manager._collapse.thread
    bus = manager.follower().bus
    writes: list[dict[str, float]] = []
    original = bus.write_positions
    monkeypatch.setattr(bus, "write_positions", lambda g: writes.append(g) or original(g))

    response = client.post("/api/robot/estop")
    gate.go.set()
    thread.join(timeout=5)

    assert response.status_code == 200, response.text
    assert writes == [], "the descent never writes after the E-stop"
    assert bus.torque_enabled is False
    status = manager.status()
    assert status["estopped"] is True and status["stop_state"] == "none"
    (estop,) = _events("safety.estop")
    assert estop["detail"]["preempted"] == "collapsing"
    assert _events("safety.collapse.done") == [] and _events("safety.collapse.fallback") == []


def test_a_halt_freezes_a_collapse_and_restarts_the_hold_timer(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    _armed(manager)
    gate = Gate(manager_clock, park_after=20)
    manager._sleep = gate.sleep
    assert manager.request_stop(StopKind.COLLAPSE, reason="lowering").collapsing
    assert gate.parked.wait(5)
    thread = manager._collapse.thread
    bus = manager.follower().bus

    response = client.post("/api/robot/halt", json={"operator_id": OPERATOR})
    held = dict(bus._goal)
    gate.go.set()
    thread.join(timeout=5)

    assert response.status_code == 200, response.text
    assert bus._goal == held, "the descent wrote nothing after the freeze"
    assert bus.torque_enabled is True
    status = manager.status()
    assert status["stop_state"] == "halted"
    expires = datetime.fromisoformat(status["halt"]["hold_expires_at"])
    now = datetime.fromisoformat(status["halt"]["server_time"])
    assert (expires - now).total_seconds() == pytest.approx(60.0, abs=0.5)
    halts = _events("safety.halt")
    assert halts[0]["detail"]["froze_collapse"] == "lowering"


def test_a_scoped_halt_does_not_freeze_a_collapse(client: TestClient, manager_clock):
    """Lease release, expiry, watchdog and error cleanup only revoke authority."""

    manager = get_manager()
    lease_id = _armed(manager)
    gate = Gate(manager_clock, park_after=5)
    manager._sleep = gate.sleep
    assert manager.request_stop(StopKind.COLLAPSE, reason="lowering").collapsing
    assert gate.parked.wait(5)
    try:
        manager.release_lease(lease_id)  # an owner-scoped Halt that ends the lease
        assert manager.status()["stop_state"] == "collapsing"
    finally:
        gate.go.set()

    assert _wait_for(lambda: _finished(manager))
    assert _events("safety.collapse.done")
    assert manager.follower().bus.torque_enabled is False


def test_a_collapse_during_a_halt_proceeds(client: TestClient):
    manager = get_manager()
    _armed(manager)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding

    assert manager.request_stop(StopKind.COLLAPSE, reason="power down").collapsing

    assert _wait_for(lambda: _finished(manager))
    assert _events("safety.collapse.done")


def test_an_older_pending_collapse_cannot_override_a_later_completed_halt(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    lease_id = _armed(manager)
    entered, proceed = threading.Event(), threading.Event()
    original_disarmed_event = manager._disarmed_event

    def gated_disarmed_event(kind: StopKind, reason: str) -> None:
        if threading.current_thread().name == "older-collapse":
            entered.set()
            assert proceed.wait(5), "the test never released the older Collapse"
        original_disarmed_event(kind, reason)

    monkeypatch.setattr(manager, "_disarmed_event", gated_disarmed_event)
    older: list[object] = []

    def collapse() -> None:
        try:
            older.append(manager.request_stop(StopKind.COLLAPSE, reason="older"))
        except Exception as exc:
            older.append(exc)

    worker = threading.Thread(name="older-collapse", target=collapse)
    worker.start()
    assert entered.wait(3), "the older Collapse never reached its pending gap"

    later = manager.request_stop(StopKind.HALT, reason="later Halt")
    assert later.holding
    scoped = manager.request_stop(
        StopKind.HALT,
        reason="scoped Halt after the global winner",
        lease_id=lease_id,
    )
    assert scoped.holding and not scoped.collapsing
    assert manager.status()["stop_state"] == "halted"
    proceed.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert older and getattr(older[0], "applied", True) is False
    status = manager.status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"
    assert _events("safety.collapse.start") == []


def test_release_lease_does_not_supersede_a_pending_collapse(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    lease_id = _armed(manager)
    entered, proceed = threading.Event(), threading.Event()
    original_disarmed_event = manager._disarmed_event

    def gated_disarmed_event(kind: StopKind, reason: str) -> None:
        if threading.current_thread().name == "collapse-before-release":
            entered.set()
            assert proceed.wait(5), "the test never released Collapse"
        original_disarmed_event(kind, reason)

    monkeypatch.setattr(manager, "_disarmed_event", gated_disarmed_event)
    collapse_outcome: list[object] = []

    def collapse() -> None:
        try:
            collapse_outcome.append(manager.request_stop(StopKind.COLLAPSE, reason="owner is done"))
        except Exception as exc:
            collapse_outcome.append(exc)

    worker = threading.Thread(name="collapse-before-release", target=collapse)
    worker.start()
    assert entered.wait(3), "Collapse never reached its pending gap"
    try:
        manager.release_lease(lease_id)
        assert manager.control_state()["lease"] is None
    finally:
        proceed.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert collapse_outcome and getattr(collapse_outcome[0], "collapsing", False)
    assert _wait_for(lambda: _finished(manager))
    status = manager.status()
    assert status["stop_state"] == "none" and status["torque_state"] == "off"
    assert manager.follower().bus.torque_enabled is False
    assert _events("safety.collapse.done")
    assert _events("safety.halt") == []


def test_release_cannot_invalidate_an_already_reserved_owner_collapse(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    lease_id = _armed(manager)
    entered, proceed = threading.Event(), threading.Event()
    original_collapse_stop = manager._collapse_stop

    def gated_collapse_stop(*args: Any, **kwargs: Any):
        if threading.current_thread().name == "reserved-owner-collapse":
            entered.set()
            assert proceed.wait(5), "the test never released the reserved Collapse"
        return original_collapse_stop(*args, **kwargs)

    monkeypatch.setattr(manager, "_collapse_stop", gated_collapse_stop)
    collapse_outcome: list[object] = []

    def collapse() -> None:
        try:
            collapse_outcome.append(
                manager.request_stop(
                    StopKind.COLLAPSE,
                    reason="reserved owner disarm",
                    lease_id=lease_id,
                )
            )
        except Exception as exc:
            collapse_outcome.append(exc)

    worker = threading.Thread(name="reserved-owner-collapse", target=collapse)
    worker.start()
    assert entered.wait(3), "owner Collapse never reserved its transition"
    try:
        manager.release_lease(lease_id)
        assert manager.control_state()["lease"] is None
    finally:
        proceed.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert collapse_outcome and getattr(collapse_outcome[0], "collapsing", False)
    assert _wait_for(lambda: _finished(manager))
    status = manager.status()
    assert status["stop_state"] == "none" and status["torque_state"] == "off"
    assert manager.follower().bus.torque_enabled is False
    assert _events("safety.collapse.done")
    assert _events("safety.halt") == []


def test_arm_cleanup_does_not_cut_a_collapse_established_after_torque_enable(
    client: TestClient, manager_clock, monkeypatch
):
    """A revoked arm defers to the active capped descent that took ownership."""

    manager = get_manager()
    lease_id = manager.acquire_lease("collapse-test", 60)["id"]
    follower = manager.follower()
    gate = Gate(manager_clock, park_after=1)
    manager._sleep = gate.sleep
    enabled, return_from_enable = threading.Event(), threading.Event()
    collapse_revoked, collapse_returned = threading.Event(), threading.Event()
    original_enable = follower.enable_torque
    original_revoke = manager._revoke_authority_locked
    original_deenergize = manager._deenergize_external

    def gated_enable() -> dict[str, float]:
        measured = original_enable()
        enabled.set()
        assert return_from_enable.wait(5), "the test never released torque-enable"
        return measured

    def observed_revoke(**kwargs: Any) -> bool:
        armed = original_revoke(**kwargs)
        if threading.current_thread().name == "collapse-after-enable":
            collapse_revoked.set()
        return armed

    def cleanup_after_collapse(reason: str = "disarm") -> bool:
        if threading.current_thread().name == "arm-before-collapse":
            assert collapse_returned.wait(5), "Collapse did not start before arm cleanup"
        return original_deenergize(reason)

    monkeypatch.setattr(follower, "enable_torque", gated_enable)
    monkeypatch.setattr(manager, "_revoke_authority_locked", observed_revoke)
    monkeypatch.setattr(manager, "_deenergize_external", cleanup_after_collapse)
    arm_outcome: list[object] = []
    collapse_outcome: list[object] = []

    def arm_control() -> None:
        try:
            arm_outcome.append(manager.arm_control(lease_id))
        except Exception as exc:
            arm_outcome.append(exc)

    def collapse() -> None:
        try:
            collapse_outcome.append(manager.request_stop(StopKind.COLLAPSE, reason="after enable"))
        except Exception as exc:
            collapse_outcome.append(exc)
        finally:
            collapse_returned.set()

    arm = threading.Thread(name="arm-before-collapse", target=arm_control)
    arm.start()
    assert enabled.wait(3), "arm_control never enabled torque"
    stop = threading.Thread(name="collapse-after-enable", target=collapse)
    stop.start()
    assert collapse_revoked.wait(3), "Collapse never revoked the arm"
    return_from_enable.set()
    arm.join(timeout=5)
    stop.join(timeout=5)

    assert not arm.is_alive() and not stop.is_alive()
    assert arm_outcome and isinstance(arm_outcome[0], ControlRejected)
    assert collapse_outcome and getattr(collapse_outcome[0], "collapsing", False)
    assert gate.parked.wait(5), "the active descent was cancelled by arming cleanup"
    try:
        status = manager.status()
        assert status["stop_state"] == "collapsing" and status["torque_state"] == "on"
        assert follower.bus.torque_enabled is True
    finally:
        gate.go.set()
    assert _wait_for(lambda: _finished(manager))


def test_nothing_else_starts_while_collapsing(client: TestClient, manager_clock):
    manager = get_manager()
    lease_id = _armed(manager)
    gate = Gate(manager_clock, park_after=5)
    manager._sleep = gate.sleep
    assert manager.request_stop(StopKind.COLLAPSE, reason="lowering").collapsing
    assert gate.parked.wait(5)
    try:
        with pytest.raises(ControlConflict, match="Collapse"):
            manager.arm_control(lease_id)
        with pytest.raises(RuntimeError, match="Collapse"):
            manager.start_teleop(30, operator_id=OPERATOR)
        with pytest.raises(ControlConflict, match="Collapse"):
            manager.clear_fault(operator_id=OPERATOR)
        with pytest.raises(ControlConflict, match="energized"):
            manager.claim_calibration("follower")
    finally:
        gate.go.set()
    assert _wait_for(lambda: _finished(manager))


# --- servo health and shutdown -------------------------------------------------------------


def test_servo_health_critical_collapses_and_latches_a_fault(client: TestClient):
    manager = get_manager()
    lease_id = _armed(manager)

    result = manager.servo_health_critical("elbow_flex at 71 C")

    assert result.collapsing
    assert manager.control_state()["safety_state"] == "fault"
    assert _wait_for(lambda: _finished(manager))
    assert _events("safety.collapse.done"), "a fault-triggered descent still runs to rest"
    assert manager.follower().bus.torque_enabled is False
    # Latched because a fault triggered it: only the deliberate reset clears it.
    assert manager.fault == "servo health critical: elbow_flex at 71 C"
    with pytest.raises(Exception, match="fault"):
        manager.arm_control(lease_id)
    manager.clear_fault(operator_id=OPERATOR)
    assert manager.fault is None
    assert _events("safety.servo_health_critical")


def test_pending_fault_collapse_keeps_precedence_over_a_later_halt(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)
    entered, proceed = threading.Event(), threading.Event()
    original_disarmed_event = manager._disarmed_event

    def gated_disarmed_event(kind: StopKind, reason: str) -> None:
        if threading.current_thread().name == "pending-fault-collapse":
            entered.set()
            assert proceed.wait(5), "the test never released the fault Collapse"
        original_disarmed_event(kind, reason)

    monkeypatch.setattr(manager, "_disarmed_event", gated_disarmed_event)
    outcome: list[object] = []
    worker = threading.Thread(
        name="pending-fault-collapse",
        target=lambda: outcome.append(manager.servo_health_critical("elbow overload")),
    )
    worker.start()
    assert entered.wait(3)

    halt = manager.request_stop(StopKind.HALT, reason="must not override fault reaction")
    assert halt.collapsing
    proceed.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert outcome and getattr(outcome[0], "collapsing", False)
    assert _wait_for(lambda: _finished(manager))
    assert _events("safety.collapse.done")
    assert manager.fault == "servo health critical: elbow overload"


def test_estop_preempted_fault_collapse_does_not_make_halt_report_collapsing(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _armed(manager)
    entered, proceed = threading.Event(), threading.Event()
    original_disarmed_event = manager._disarmed_event

    def gated_disarmed_event(kind: StopKind, reason: str) -> None:
        if threading.current_thread().name == "estop-preempted-fault-collapse":
            entered.set()
            assert proceed.wait(5), "the test never released the preempted Collapse"
        original_disarmed_event(kind, reason)

    monkeypatch.setattr(manager, "_disarmed_event", gated_disarmed_event)
    outcome: list[object] = []
    worker = threading.Thread(
        name="estop-preempted-fault-collapse",
        target=lambda: outcome.append(manager.servo_health_critical("wrist overload")),
    )
    worker.start()
    assert entered.wait(3)
    try:
        manager.estop("preempt pending fault Collapse")
        halt = manager.request_stop(StopKind.HALT, reason="after E-stop")
        assert halt.torque_off and not halt.collapsing
        status = manager.status()
        assert status["estopped"] is True
        assert status["stop_state"] == "none" and status["torque_state"] == "off"
    finally:
        proceed.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert outcome and getattr(outcome[0], "applied", True) is False
    assert _events("safety.collapse.start") == []


def test_a_halt_does_not_freeze_a_fault_triggered_collapse(client: TestClient, manager_clock):
    manager = get_manager()
    _armed(manager)
    gate = Gate(manager_clock, park_after=5)
    manager._sleep = gate.sleep
    assert manager.servo_health_critical("wrist_flex overload").collapsing
    assert gate.parked.wait(5)
    try:
        result = manager.request_stop(StopKind.HALT, reason="operator halt")
        assert result.collapsing and manager.status()["stop_state"] == "collapsing"
    finally:
        gate.go.set()
    assert _wait_for(lambda: _finished(manager))
    assert _events("safety.collapse.done")


def test_servo_health_critical_with_the_motors_off_only_latches(client: TestClient):
    manager = get_manager()

    result = manager.servo_health_critical("gripper status 0x20")

    assert result.ok and not result.collapsing
    assert manager.fault == "servo health critical: gripper status 0x20"


def test_shutdown_is_a_bounded_collapse_then_torque_off(client: TestClient):
    manager = get_manager()
    _armed(manager)

    manager.shutdown()

    assert _events("safety.collapse.start")[0]["detail"]["reason"] == "service shutdown"
    assert _events("safety.collapse.done")
    assert manager.follower().bus.torque_enabled is False
    assert manager.follower().connected is False


@pytest.mark.parametrize("via_shutdown", [False, True], ids=["request", "shutdown"])
def test_collapse_thread_start_failure_falls_back_without_propagating(
    client: TestClient, monkeypatch, via_shutdown: bool
):
    manager = get_manager()
    _armed(manager)
    original_start = threading.Thread.start

    def conditional_start(thread: threading.Thread) -> None:
        if thread.name == "so101-collapse":
            raise RuntimeError("injected collapse thread start failure")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", conditional_start)

    if via_shutdown:
        manager.shutdown()
        result = None
    else:
        result = manager.request_stop(StopKind.COLLAPSE, reason="thread failure")

    if result is not None:
        assert result.fallback == "error" and result.torque_off
    status = manager.status()
    assert status["stop_state"] == "none" and status["torque_state"] == "off"
    assert manager.fault is not None and "thread start failure" in manager.fault
    assert manager.follower().bus.torque_enabled is False
    fallback = _events("safety.collapse.fallback")
    assert fallback and fallback[0]["detail"]["cause"] == "error"
    assert manager.follower().connected is False


def test_shutdown_cuts_torque_when_the_descent_overruns_its_bound(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    _armed(manager)
    monkeypatch.setattr(manager.settings, "collapse_timeout_seconds", 0.2)
    monkeypatch.setattr(rm, "_COLLAPSE_POLICE_MARGIN_S", 0.0)
    gate = Gate(manager_clock, park_after=2)
    manager._sleep = gate.sleep
    started = time.monotonic()
    try:
        manager.shutdown()
    finally:
        gate.go.set()

    assert time.monotonic() - started < 5.0
    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "timeout"
    assert manager.follower().bus.torque_enabled is False
    assert manager.fault is not None  # latched, and persisted for the next start


def test_await_collapse_without_a_running_descent_returns_at_once(client: TestClient):
    manager = get_manager()
    started = time.monotonic()
    manager._await_collapse()
    assert time.monotonic() - started < 0.5


def test_a_lease_that_was_halted_ends_a_collapse_owner_scoped(client: TestClient):
    """``POST /api/control/disarm`` is an owner-scoped Collapse that keeps the lease."""

    manager = get_manager()
    lease_id = _armed(manager)

    response = client.post("/api/control/disarm", json={"lease_id": lease_id})

    assert response.status_code == 200, response.text
    assert response.json()["armed"] is False
    assert response.json()["lease"] is not None
    assert _wait_for(lambda: _finished(manager))
    assert manager.follower().bus.torque_enabled is False


def test_the_commanded_setpoints_never_overshoot_the_rest_pose(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    lease_id = _armed(manager)
    manager.send_external_command(
        lease_id=lease_id,
        source="collapse-test",
        sequence=1,
        sent_at=datetime.now(timezone.utc),
        ttl_ms=500,
        deadman=True,
        targets={"shoulder_pan": 12.0, "wrist_roll": -9.0},
    )
    goals = _spy_goals(manager, monkeypatch, manager_clock)

    assert manager.request_stop(StopKind.COLLAPSE, reason="overshoot").collapsing
    assert _wait_for(lambda: _finished(manager))

    start = goals[0][1]
    for joint in ARM:
        rest = MODEL_REST_POSE[joint]
        low, high = sorted((start[joint], rest))
        for _, goal in goals:
            assert low - 1e-9 <= goal[joint] <= high + 1e-9, joint


class RefusingBusLock(SlowBusLock):
    """The manager's bus lock, with one descent acquisition timing out (another holder)."""

    def __init__(self, lock: Any, clock: Any, *, refuse: int) -> None:
        super().__init__(lock, clock, slow=set(), wait_s=0.0)
        self.refuse = refuse

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if threading.current_thread().name == "so101-collapse":
            self.collapse_acquisitions += 1
            if self.collapse_acquisitions == self.refuse:
                self.clock.now += max(0.0, timeout)
                return False
        return self._inner.acquire(blocking, timeout)


def test_a_tick_that_cannot_get_the_bus_falls_back(client: TestClient, manager_clock, monkeypatch):
    manager = get_manager()
    _armed(manager)
    monkeypatch.setattr(
        manager, "_io_lock", RefusingBusLock(manager._io_lock, manager_clock, refuse=6)
    )

    assert manager.request_stop(StopKind.COLLAPSE, reason="bus held").collapsing
    assert _wait_for(lambda: _finished(manager))

    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "bus_error"
    assert "did not come free within 0.5 s" in fallback["detail"]["detail"]
    assert fallback["detail"]["torque_off"] is True
    assert manager.follower().bus.torque_enabled is False and manager.fault is not None


def test_a_descent_without_a_plan_falls_back_at_once(client: TestClient):
    manager = get_manager()
    _armed(manager)
    run = rm._Collapse(since=0.0, since_wall=time.time(), reason="no plan")

    manager._run_descent(run)

    (fallback,) = _events("safety.collapse.fallback")
    assert fallback["detail"]["cause"] == "error"
    assert "without a plan" in fallback["detail"]["detail"]
    assert manager.follower().bus.torque_enabled is False
    # A second finisher finds the descent claimed and only reports the torque truth.
    assert manager._collapse_fallback(run, "error", "again") is True
    assert len(_events("safety.collapse.fallback")) == 1

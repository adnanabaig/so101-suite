"""Halt (category 2): freeze in place, motors on (design doc §5 "Halt").

Contract: docs/plans/2026-10-01-suite-v02-design.md §5. Every test runs on the
mock bus. Timers run on the injected manager clock (``manager_clock``) and the
watchdog acts only when a test calls ``supervise`` (devex-B4).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlConflict,
    ControlRejected,
    RobotManager,
    StopKind,
    get_manager,
)
from so101.services.store import get_store  # noqa: E402

OPERATOR = "test-operator"
EVIL = "https://evil.example"


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _armed(manager: RobotManager) -> str:
    lease = manager.acquire_lease("halt-test", 30)
    manager.arm_control(lease["id"])
    assert manager.follower().bus.torque_enabled is True
    return lease["id"]


def _command(manager: RobotManager, lease_id: str, sequence: int, **targets: float) -> None:
    manager.send_external_command(
        lease_id=lease_id,
        source="halt-test",
        sequence=sequence,
        sent_at=datetime.now(timezone.utc),
        ttl_ms=500,
        deadman=True,
        targets=targets,
    )


def _events(kind: str) -> list[dict[str, Any]]:
    return [event for event in get_store().list_events(limit=500) if event["type"] == kind]


def _seconds_left(status: dict[str, Any]) -> float:
    halt = status["halt"]
    expires = datetime.fromisoformat(halt["hold_expires_at"])
    now = datetime.fromisoformat(halt["server_time"])
    return (expires - now).total_seconds()


def _spy_reads(manager: RobotManager, monkeypatch) -> list[dict[str, float]]:
    follower = manager.follower()
    original = follower.read_positions
    reads: list[dict[str, float]] = []

    def spy() -> dict[str, float]:
        measured = original()
        reads.append(dict(measured))
        return measured

    monkeypatch.setattr(follower, "read_positions", spy)
    return reads


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class _PauseAfterRelease:
    """A lock proxy that opens one deterministic post-I/O scheduling gap."""

    def __init__(self, inner: Any, thread_name: str, *, release_number: int = 1) -> None:
        self._inner = inner
        self.thread_name = thread_name
        self.release_number = release_number
        self.released = threading.Event()
        self.resume = threading.Event()
        self._paused = False
        self._releases = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._inner.acquire(blocking, timeout)

    def release(self) -> None:
        self._inner.release()
        if threading.current_thread().name != self.thread_name:
            return
        self._releases += 1
        if self._releases == self.release_number and not self._paused:
            self._paused = True
            self.released.set()
            assert self.resume.wait(5), "the test never released the paused operation"

    def locked(self) -> bool:
        return self._inner.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --- holding ---------------------------------------------------------------------


def test_halt_holds_goal_at_a_freshly_read_present(client: TestClient, monkeypatch):
    manager = get_manager()
    lease_id = _armed(manager)
    _command(manager, lease_id, 1, shoulder_pan=10.0, elbow_flex=-8.0)
    bus = manager.follower().bus
    reads = _spy_reads(manager, monkeypatch)

    result = manager.request_stop(StopKind.HALT, reason="operator halt")

    assert result.ok and result.holding and not result.torque_off
    present = reads[-1]
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        assert bus._goal[joint] == pytest.approx(present[joint])
    assert bus.torque_enabled is True and bus.connected is True
    state = manager.control_state()
    assert state["armed"] is False and state["safety_state"] == "disarmed"
    status = manager.status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"
    assert status["halt"]["reason"] == "operator halt"
    assert _seconds_left(status) == pytest.approx(60.0, abs=0.5)
    (event,) = _events("safety.halt")
    assert event["detail"]["holding"] is True
    assert event["detail"]["hold_s"] == 60.0


@pytest.mark.parametrize(
    ("last_goal", "present"),
    [
        (10.0, 30.0),  # closing on an object: keep at most the margin of squeeze
        (28.0, 30.0),  # an existing small squeeze is kept as it is
        (80.0, 30.0),  # opening: never opens further than where it is
        (0.0, 2.0),  # near closed: stays inside the 0..100 gripper range
    ],
    ids=["squeeze-capped", "squeeze-kept", "never-opens", "range"],
)
def test_halt_grip_rule(client: TestClient, monkeypatch, last_goal: float, present: float):
    manager = get_manager()
    _armed(manager)
    follower = manager.follower()
    bus = follower.bus
    bus._pos["gripper"] = bus._goal["gripper"] = present
    follower.last_goal["gripper"] = last_goal
    reads = _spy_reads(manager, monkeypatch)

    assert manager.request_stop(StopKind.HALT, reason="grip").holding

    measured = reads[-1]["gripper"]  # the mock adds a little gripper wobble
    margin = manager.settings.grip_hold_margin
    expected = max(0.0, min(max(last_goal, measured - margin), measured))
    assert bus._goal["gripper"] == pytest.approx(min(expected, measured))
    assert bus._goal["gripper"] <= measured + 1e-9
    assert bus._goal["gripper"] >= min(measured - margin, measured) - 1e-9


def test_halt_never_energizes_motors_that_are_off(client: TestClient):
    manager = get_manager()
    follower = manager.follower()

    idle = manager.request_stop(StopKind.HALT, reason="idle halt")

    assert idle.ok and not idle.holding and idle.torque_off
    # Connected with torque off (arming's first stage) is off as well.
    with manager._io_lock:
        follower.connect(enable_torque=False)
    connected = manager.request_stop(StopKind.HALT, reason="connected but off")
    assert connected.ok and not connected.holding
    assert follower.bus.torque_enabled is False
    assert manager.status()["stop_state"] == "none"
    assert _events("safety.halt") == []


def test_halt_while_estopped_leaves_the_estop_in_charge(client: TestClient):
    manager = get_manager()
    _armed(manager)
    manager.estop()

    result = manager.request_stop(StopKind.HALT, reason="after estop")

    assert result.applied and result.torque_off and not result.holding
    assert manager.status()["stop_state"] == "none"
    assert manager.follower().bus.torque_enabled is False


def test_a_halt_under_a_latched_fault_cuts_torque(client: TestClient):
    manager = get_manager()
    _armed(manager)
    manager._set_fault("encoder dropout", source="test")  # nothing cut torque yet

    result = manager.request_stop(StopKind.HALT, reason="halt under fault")

    assert result.torque_off and not result.holding
    assert manager.follower().bus.torque_enabled is False
    assert manager.status()["stop_state"] == "none"


# --- fallback (category 0) ----------------------------------------------------------


def test_halt_falls_back_to_torque_off_when_the_position_cannot_be_read(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _armed(manager)

    def dead() -> dict[str, float]:
        raise ConnectionError("no status packet")

    monkeypatch.setattr(manager.follower(), "read_positions", dead)

    response = client.post("/api/robot/halt", json={"operator_id": OPERATOR})

    assert response.status_code == 503, response.text
    assert "could not hold" in response.json()["detail"]
    assert "torque cut" in response.json()["detail"]
    assert manager.follower().bus.torque_enabled is False
    assert manager.fault is not None and "Halt could not hold" in manager.fault
    status = manager.status()
    assert status["stop_state"] == "none" and status["torque_state"] == "off"
    (event,) = _events("safety.halt")
    assert event["level"] == "critical" and event["detail"]["fallback"]


def test_halt_falls_back_when_the_goal_cannot_be_written(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)

    def refuse(goal):
        raise ConnectionError("write lost")

    monkeypatch.setattr(manager.follower(), "hold", refuse)

    result = manager.request_stop(StopKind.HALT, reason="write fails")

    assert result.fallback and "write lost" in result.fallback
    assert result.torque_off and not result.ok
    assert manager.fault is not None


def test_halt_falls_back_when_the_bus_does_not_come_free(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)
    monkeypatch.setattr(manager, "stop_io_timeout_s", 0.05)
    assert manager._io_lock.acquire(timeout=1)  # a wedged driver call holds the bus
    try:
        result = manager.request_stop(StopKind.HALT, reason="bus held")
    finally:
        manager._io_lock.release()

    assert result.fallback is not None and "did not come free" in result.fallback
    # The fallback's own torque-off could not reach the bus either: C2.
    assert result.torque_off is False
    status = manager.status()
    assert status["torque_unconfirmed"] is True and status["torque_state"] == "unconfirmed"


def test_halt_falls_back_when_its_io_overruns_the_timeout(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    _armed(manager)
    follower = manager.follower()
    original = follower.read_positions

    def slow_read() -> dict[str, float]:
        manager_clock.advance(manager.stop_io_timeout_s + 0.1)
        return original()

    monkeypatch.setattr(follower, "read_positions", slow_read)

    result = manager.request_stop(StopKind.HALT, reason="slow bus")

    assert result.fallback is not None and "limit" in result.fallback
    assert follower.bus.torque_enabled is False
    assert manager.fault is not None


def test_halt_falls_back_when_torque_is_not_known_off_but_nothing_holds(client: TestClient):
    manager = get_manager()
    _armed(manager)
    bus = manager.follower().bus
    bus.fail_torque_off = True
    with manager._io_lock:
        try:
            bus.disconnect()  # closes with torque unconfirmed
        except Exception:
            pass
    bus.fail_torque_off = False

    result = manager.request_stop(StopKind.HALT, reason="stale bus")

    assert result.fallback is not None and "unconfirmed" in result.fallback
    assert manager.status()["torque_unconfirmed"] is True


# --- hold timer and supervision -------------------------------------------------------


def test_the_hold_timer_expiry_becomes_a_collapse(client: TestClient, manager_clock, supervise):
    manager = get_manager()
    _armed(manager)
    manager.operator_heartbeat(OPERATOR)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding

    for _ in range(59):
        manager_clock.advance(1.0)
        manager.operator_heartbeat(OPERATOR)  # supervised throughout
        supervise()
    assert manager.status()["stop_state"] == "halted"

    manager_clock.advance(1.1)
    manager.operator_heartbeat(OPERATOR)
    supervise()

    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    (start,) = _events("safety.collapse.start")
    assert start["detail"]["reason"] == "halt hold expired after 60 s"
    assert _events("safety.collapse.done")
    assert manager.follower().bus.torque_enabled is False


def test_lost_supervision_collapses_after_the_grace_period(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    _armed(manager)
    manager.operator_heartbeat(OPERATOR)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    timeout = manager.operator_timeout_ms / 1000
    grace = manager.settings.supervision_grace_seconds

    manager_clock.advance(timeout + 0.5)  # the operator's heartbeat went stale
    supervise()
    status = manager.status()
    assert status["stop_state"] == "halted"
    assert status["supervision"]["supervised"] is False
    assert status["supervision"]["lost_at"] is not None
    assert _seconds_left(status) == pytest.approx(grace - 0.5, abs=0.05)

    manager_clock.advance(grace - 0.6)
    supervise()
    assert manager.status()["stop_state"] == "halted"

    manager_clock.advance(0.2)
    supervise()

    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    (start,) = _events("safety.collapse.start")
    assert "supervision lost" in start["detail"]["reason"]


def test_a_halt_with_no_operator_ever_holds_only_the_grace_period(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    _armed(manager)
    with manager._lock:
        manager._operator_heartbeats.clear()  # a CLI-only setup: no browser ever

    assert manager.request_stop(StopKind.HALT, reason="cli halt").holding
    status = manager.status()
    assert status["supervision"] == {
        "supervised": False,
        "lost_at": None,
        "grace_s": manager.settings.supervision_grace_seconds,
    }
    assert _seconds_left(status) == pytest.approx(manager.settings.supervision_grace_seconds)


def test_returning_supervision_restores_the_hold_timer(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    _armed(manager)
    manager.operator_heartbeat(OPERATOR)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding

    manager_clock.advance(manager.operator_timeout_ms / 1000 + 2.0)
    supervise()
    assert _seconds_left(manager.status()) < manager.settings.supervision_grace_seconds

    manager.operator_heartbeat("another-tab")  # any operator browser supervises
    supervise()

    status = manager.status()
    assert status["stop_state"] == "halted"
    assert status["supervision"]["supervised"] is True
    assert _seconds_left(status) > 50


def test_a_repeat_halt_renews_only_for_a_supervising_operator_past_the_origin_guard(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    _armed(manager)
    manager.operator_heartbeat(OPERATOR)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    manager_clock.advance(30.0)
    manager.operator_heartbeat(OPERATOR)

    # A cross-origin repeat still halts (stops are never blocked) but cannot renew.
    foreign = client.post(
        "/api/robot/halt", json={"operator_id": OPERATOR}, headers={"Origin": EVIL}
    )
    assert foreign.status_code == 200 and "holding" in foreign.json()["detail"]
    assert _seconds_left(manager.status()) == pytest.approx(30.0, abs=0.5)
    # No operator, or one without a fresh heartbeat: no renewal either.
    assert client.post("/api/robot/halt").status_code == 200
    manager.request_stop(
        StopKind.HALT, reason="stale", operator_id="ghost-tab", origin_verified=True
    )
    assert _seconds_left(manager.status()) == pytest.approx(30.0, abs=0.5)

    renewed = client.post("/api/robot/halt", json={"operator_id": OPERATOR})

    assert renewed.status_code == 200
    assert _seconds_left(manager.status()) == pytest.approx(60.0, abs=0.5)
    assert any(event["detail"].get("renewed") for event in _events("safety.halt"))


# --- leaving a halt ---------------------------------------------------------------------


def test_rearming_after_a_halt_neither_toggles_torque_nor_moves_the_arm(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    lease_id = _armed(manager)
    _command(manager, lease_id, 1, shoulder_pan=6.0, gripper=40.0)
    assert manager.request_stop(StopKind.HALT, reason="pause").holding
    bus = manager.follower().bus
    held_goal = dict(bus._goal)
    torque_calls: list[bool] = []
    writes: list[dict[str, float]] = []
    set_torque, write_positions = bus.set_torque, bus.write_positions
    monkeypatch.setattr(bus, "set_torque", lambda on: torque_calls.append(on) or set_torque(on))
    monkeypatch.setattr(
        bus, "write_positions", lambda goal: writes.append(dict(goal)) or write_positions(goal)
    )

    state = manager.arm_control(lease_id)

    assert state["armed"] is True
    assert torque_calls == [] and writes == [], "no torque write and no goal write"
    assert bus._goal == held_goal and bus.torque_enabled is True
    assert manager.status()["stop_state"] == "none", "the armed session took over the hold"
    _command(manager, lease_id, 2, shoulder_pan=7.0)  # and it accepts commands


def test_a_loop_end_halts_and_holds_instead_of_sagging(client: TestClient, monkeypatch):
    manager = get_manager()
    manager.start_teleop(30, operator_id=OPERATOR)
    follower = manager.follower()
    assert _wait_for(lambda: follower.connected and follower.bus.torque_enabled)
    disconnects: list[bool] = []
    original = follower.disconnect
    monkeypatch.setattr(follower, "disconnect", lambda: disconnects.append(True) or original())

    response = client.post("/api/teleop/stop")

    assert response.status_code == 200, response.text
    assert "holds" in response.json()["detail"]
    assert disconnects == [], "the loop's end never de-energizes the follower"
    status = manager.status()
    assert status["mode"] == "idle"
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"
    assert status["leader_connected"] is False, "the passive leader is released"


def test_the_eval_runner_stop_between_episodes_holds(client: TestClient):
    manager = get_manager()
    manager.start_teleop(30, operator_id=OPERATOR)
    assert _wait_for(lambda: manager.follower().bus.torque_enabled)

    assert manager.stop() is True

    assert manager.status()["stop_state"] == "halted"
    # The next episode's loop takes the hold over.
    manager.operator_heartbeat(OPERATOR)
    manager.start_teleop(30, operator_id=OPERATOR)
    assert manager.status()["stop_state"] == "none"
    assert manager.soft_stop("done") is True


def test_operator_heartbeat_loss_is_a_halt_then_a_collapse(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    manager.operator_timeout_ms = 100
    manager.operator_heartbeat(OPERATOR)
    started = manager_clock.now
    manager.start_teleop(30, operator_id=OPERATOR)
    # The loop's pacing sleeps move the injected clock past the timeout.
    assert _wait_for(lambda: manager_clock.now - started > 0.2)

    supervise()

    assert _wait_for(lambda: manager.mode.value == "idle")
    status = manager.status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"
    (event,) = _events("runtime.operator_timeout")
    assert "holding position" in event["message"]

    manager_clock.advance(manager.settings.supervision_grace_seconds + 0.2)
    supervise()

    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    (start,) = _events("safety.collapse.start")
    assert "supervision lost" in start["detail"]["reason"]


def test_the_hold_is_sampled_and_a_dead_bus_is_a_category_0_fault(
    client: TestClient, manager_clock, supervise
):
    manager = get_manager()
    _armed(manager)
    manager.operator_heartbeat(OPERATOR)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    seq = manager.telemetry()["sample_seq"]

    manager_clock.advance(0.2)
    manager.operator_heartbeat(OPERATOR)
    supervise()

    frame = manager.telemetry()
    assert frame["sample_seq"] > seq and frame["observation_live"] is True
    assert frame["stop_state"] == "halted" and frame["halt"]["reason"] == "hold"

    manager.follower().bus.fail_io = True
    manager_clock.advance(0.2)
    manager.operator_heartbeat(OPERATOR)
    supervise()

    # The read failure latched a fault; the dead bus then could not confirm
    # torque-off either, which latched over it (C2).
    assert manager.fault is not None and "halt hold lost the bus" in manager.fault
    assert any("while holding" in event["message"] for event in _events("fault"))
    status = manager.status()
    assert status["stop_state"] == "none"
    assert status["torque_unconfirmed"] is True  # the dead bus cannot confirm torque-off


@pytest.mark.parametrize("torque_state", ["on", "unconfirmed"])
def test_closed_port_while_halted_fails_closed(client: TestClient, torque_state: str):
    manager = get_manager()
    _armed(manager)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    halt = manager._halt
    assert halt is not None
    bus = manager.follower().bus
    with manager._lock:
        # The adapter vanished without a confirmed torque-off. Closing a host
        # port does not remove power from an STS3215.
        bus._connected = False
        bus._torque_state = torque_state

    manager._sample_hold(halt)

    status = manager.status()
    assert status["stop_state"] == "none"
    assert status["torque_unconfirmed"] is True
    assert status["torque_state"] == "unconfirmed"
    assert manager.fault is not None and "halt hold lost torque confirmation" in manager.fault


@pytest.mark.parametrize("torque_state", ["on", "unconfirmed"])
def test_repeat_halt_does_not_claim_a_disconnected_hold(client: TestClient, torque_state: str):
    manager = get_manager()
    _armed(manager)
    assert manager.request_stop(StopKind.HALT, reason="first halt").holding
    bus = manager.follower().bus
    with manager._lock:
        bus._connected = False
        bus._torque_state = torque_state

    result = manager.request_stop(StopKind.HALT, reason="repeat halt")

    assert result.holding is False
    assert result.ok is False
    assert result.fallback is not None and "lost torque confirmation" in result.fallback
    assert manager.fault is not None
    assert manager.status()["torque_unconfirmed"] is True


def test_calibration_is_refused_while_the_arm_holds(client: TestClient):
    manager = get_manager()
    lease_id = _armed(manager)
    manager.release_lease(lease_id)  # a Halt that ends the lease
    assert manager.status()["stop_state"] == "halted"

    with pytest.raises(ControlConflict, match="energized and holding"):
        manager.claim_calibration("follower")


def test_a_hold_that_ended_under_it_clears_the_halt(client: TestClient, manager_clock, supervise):
    manager = get_manager()
    _armed(manager)
    manager.operator_heartbeat(OPERATOR)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    with manager._lock:
        manager._follower.bus._torque_state = "off"  # e.g. a servo lost power

    manager_clock.advance(0.2)
    manager.operator_heartbeat(OPERATOR)
    supervise()

    assert manager.status()["stop_state"] == "none"


def test_a_halt_racing_an_arm_keeps_the_hold(client: TestClient, monkeypatch):
    """A global halt that lands while arming takes over a hold leaves the hold."""

    manager = get_manager()
    lease_id = _armed(manager)
    assert manager.request_stop(StopKind.HALT, reason="first").holding
    follower = manager.follower()
    entered, proceed = threading.Event(), threading.Event()
    original = follower.read_positions

    def slow_read() -> dict[str, float]:
        entered.set()
        proceed.wait(2)
        return original()

    monkeypatch.setattr(follower, "read_positions", slow_read)
    outcome: list[object] = []
    worker = threading.Thread(
        target=lambda: outcome.append(_try(lambda: manager.arm_control(lease_id)))
    )
    worker.start()
    assert entered.wait(2)
    with manager._lock:
        manager._authority_epoch += 1  # another stop revoked authority mid-arm
    proceed.set()
    worker.join(timeout=5)

    assert outcome and isinstance(outcome[0], Exception)
    assert follower.bus.torque_enabled is True, "the hold this arm found is kept"


def test_rearm_cannot_finish_between_the_halt_write_and_its_published_state(
    client: TestClient, manager_clock, monkeypatch
):
    """A Halt owns the arm until its hold is visible and the request finishes."""

    manager = get_manager()
    lease_id = _armed(manager)
    gate = _PauseAfterRelease(manager._io_lock, "halt-publication")
    monkeypatch.setattr(manager, "_io_lock", gate)
    halt_outcome: list[object] = []
    worker = threading.Thread(
        name="halt-publication",
        target=lambda: halt_outcome.append(
            _try(lambda: manager.request_stop(StopKind.HALT, reason="publication race"))
        ),
    )
    worker.start()
    assert gate.released.wait(3), "Halt never reached its post-I/O publication gap"
    try:
        rearm = _try(lambda: manager.arm_control(lease_id))
        with pytest.raises(ControlConflict, match="stop is being established"):
            manager.clear_fault(operator_id=OPERATOR)
        with pytest.raises(ControlConflict, match="being established"):
            manager.claim_calibration("follower")
    finally:
        gate.resume.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert isinstance(rearm, (ControlConflict, ControlRejected)), rearm
    assert halt_outcome and getattr(halt_outcome[0], "holding", False)
    state = manager.control_state()
    assert state["armed"] is False
    assert manager.status()["stop_state"] == "halted"
    command = _try(lambda: _command(manager, lease_id, 2, shoulder_pan=9.0))
    assert isinstance(command, ControlRejected), command


def test_halt_fallback_latches_before_a_blocking_post_cut_audit(
    client: TestClient, manager_clock, monkeypatch
):
    """No valid lease can re-energize between a category-0 cut and its fault."""

    manager = get_manager()
    lease_id = _armed(manager)
    follower = manager.follower()

    def fail_hold(_goal: dict[str, float]) -> dict[str, float]:
        raise ConnectionError("injected hold failure")

    monkeypatch.setattr(follower, "hold", fail_hold)
    entered, release = threading.Event(), threading.Event()
    original_event = manager._event

    def blocking_event(event_type: str, message: str, **kwargs: Any) -> None:
        if (
            event_type == "safety.force_deenergized"
            and threading.current_thread().name == "halt-fallback"
        ):
            entered.set()
            assert release.wait(5), "the test never released the post-cut audit"
        original_event(event_type, message, **kwargs)

    monkeypatch.setattr(manager, "_event", blocking_event)
    halt_outcome: list[object] = []
    worker = threading.Thread(
        name="halt-fallback",
        target=lambda: halt_outcome.append(
            _try(lambda: manager.request_stop(StopKind.HALT, reason="fallback race"))
        ),
    )
    worker.start()
    assert entered.wait(3), "the fallback never reached its post-cut audit"
    try:
        rearm = _try(lambda: manager.arm_control(lease_id))
        assert isinstance(rearm, ControlRejected), rearm
        assert manager.fault is not None
        assert follower.bus.torque_enabled is False
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert halt_outcome and getattr(halt_outcome[0], "fallback", None)
    status = manager.status()
    assert status["torque_state"] == "off" and manager.fault is not None


def test_arm_cleanup_does_not_cut_a_halt_established_after_torque_enable(
    client: TestClient, manager_clock, monkeypatch
):
    """A revoked arm defers to the Halt that took ownership of energized hardware."""

    manager = get_manager()
    lease_id = manager.acquire_lease("halt-test", 30)["id"]
    follower = manager.follower()
    enabled, return_from_enable = threading.Event(), threading.Event()
    halt_revoked, halt_finished = threading.Event(), threading.Event()
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
        if threading.current_thread().name == "halt-after-enable":
            halt_revoked.set()
        return armed

    def cleanup_after_halt(reason: str = "disarm") -> bool:
        if threading.current_thread().name == "arm-before-halt":
            assert halt_finished.wait(5), "the Halt did not finish before arm cleanup"
        return original_deenergize(reason)

    monkeypatch.setattr(follower, "enable_torque", gated_enable)
    monkeypatch.setattr(manager, "_revoke_authority_locked", observed_revoke)
    monkeypatch.setattr(manager, "_deenergize_external", cleanup_after_halt)
    arm_outcome: list[object] = []
    halt_outcome: list[object] = []
    arm = threading.Thread(
        name="arm-before-halt",
        target=lambda: arm_outcome.append(_try(lambda: manager.arm_control(lease_id))),
    )

    def halt() -> None:
        try:
            halt_outcome.append(
                _try(lambda: manager.request_stop(StopKind.HALT, reason="after enable"))
            )
        finally:
            halt_finished.set()

    arm.start()
    assert enabled.wait(3), "arm_control never enabled torque"
    stop = threading.Thread(name="halt-after-enable", target=halt)
    stop.start()
    assert halt_revoked.wait(3), "Halt never revoked the arm"
    return_from_enable.set()
    arm.join(timeout=5)
    stop.join(timeout=5)

    assert not arm.is_alive() and not stop.is_alive()
    assert arm_outcome and isinstance(arm_outcome[0], ControlRejected)
    assert halt_outcome and getattr(halt_outcome[0], "holding", False)
    assert follower.bus.torque_enabled is True
    status = manager.status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"


def test_late_arm_cleanup_never_cuts_a_newer_session_that_took_over_the_halt(
    client: TestClient, manager_clock, monkeypatch
):
    """Authority epochs keep stale cleanup away from a later successful re-arm."""

    manager = get_manager()
    lease_id = manager.acquire_lease("halt-test", 30)["id"]
    gate = _PauseAfterRelease(manager._io_lock, "stale-arm-cleanup", release_number=2)
    monkeypatch.setattr(manager, "_io_lock", gate)
    stale_outcome: list[object] = []
    stale = threading.Thread(
        name="stale-arm-cleanup",
        target=lambda: stale_outcome.append(_try(lambda: manager.arm_control(lease_id))),
    )
    stale.start()
    assert gate.released.wait(3), "the first arm never reached its post-enable cleanup gap"

    assert manager.request_stop(StopKind.HALT, reason="new owner").holding
    newer = manager.arm_control(lease_id)
    assert newer["armed"] is True
    gate.resume.set()
    stale.join(timeout=5)

    assert not stale.is_alive()
    assert stale_outcome and isinstance(stale_outcome[0], ControlRejected)
    assert manager.control_state()["armed"] is True
    assert manager.status()["stop_state"] == "none"
    assert manager.follower().bus.torque_enabled is True
    _command(manager, lease_id, 1, shoulder_pan=3.0)


def _try(func):
    try:
        return func()
    except Exception as exc:  # handed to the test
        return exc


def test_a_loops_only_halt_leaves_an_armed_session_alone(client: TestClient):
    """``stop()`` (the eval runner between episodes) never takes over a session."""

    manager = get_manager()
    lease_id = _armed(manager)
    _command(manager, lease_id, 1, shoulder_pan=4.0)

    assert manager.stop() is True

    assert manager.control_state()["armed"] is True
    assert manager.status()["stop_state"] == "none"
    _command(manager, lease_id, 2, shoulder_pan=5.0)


def test_loops_only_halt_blocks_a_new_lease_until_its_hold_is_final(
    client: TestClient, manager_clock, monkeypatch
):
    manager = get_manager()
    manager.start_teleop(30, operator_id=OPERATOR)
    assert _wait_for(lambda: manager.follower().bus.torque_enabled)
    entered, proceed = threading.Event(), threading.Event()
    original_hold = manager._hold_in_place

    def gated_hold(*args: Any, **kwargs: Any):
        entered.set()
        assert proceed.wait(5), "the test never released loops-only Halt"
        return original_hold(*args, **kwargs)

    monkeypatch.setattr(manager, "_hold_in_place", gated_hold)
    stopped: list[object] = []
    worker = threading.Thread(
        name="loops-only-halt",
        target=lambda: stopped.append(_try(manager.stop)),
    )
    worker.start()
    assert entered.wait(3), "loops-only Halt never reached hold publication"
    try:
        lease = _try(lambda: manager.acquire_lease("racing-session", 30))
        arm = _try(lambda: manager.arm_control(lease["id"])) if isinstance(lease, dict) else None
    finally:
        proceed.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert isinstance(lease, ControlConflict), (lease, arm)
    assert stopped == [True]
    assert manager.status()["stop_state"] == "halted"

    lease_id = manager.acquire_lease("after-stop", 30)["id"]
    assert manager.arm_control(lease_id)["armed"] is True
    assert manager.status()["stop_state"] == "none"


def test_loops_only_halt_revalidates_armed_state_before_publishing(
    client: TestClient, manager_clock, monkeypatch
):
    """An external session that appears after the early sample owns the arm."""

    manager = get_manager()
    lease_id = _armed(manager)
    with manager._lock:
        manager._armed = False
        manager._safety_state = "disarmed"
    entered, proceed = threading.Event(), threading.Event()
    original_hold = manager._hold_in_place

    def gated_hold(*args: Any, **kwargs: Any):
        entered.set()
        assert proceed.wait(5), "the test never released loops-only publication"
        return original_hold(*args, **kwargs)

    monkeypatch.setattr(manager, "_hold_in_place", gated_hold)
    outcome: list[object] = []
    worker = threading.Thread(
        target=lambda: outcome.append(
            _try(lambda: manager._dispatch_stop(StopKind.HALT, reason="loop end", loops_only=True))
        )
    )
    worker.start()
    assert entered.wait(3)
    with manager._lock:
        manager._armed = True
        manager._safety_state = "armed"
    proceed.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert outcome and getattr(outcome[0], "loop_stopped", False)
    assert manager.control_state()["armed"] is True
    assert manager.status()["stop_state"] == "none"
    _command(manager, lease_id, 1, shoulder_pan=2.0)


def test_only_motion_owning_stop_transitions_block_external_control(client: TestClient):
    manager = get_manager()
    with manager._lock:
        manager._stop_transitions[1001] = (StopKind.HALT, True, False)
        assert manager._external_stop_in_progress_locked() is False
        manager._stop_transitions[1002] = (StopKind.COLLAPSE, False, False)
        assert manager._external_stop_in_progress_locked() is True
        manager._stop_transitions.clear()


def test_a_hold_timer_collapse_never_stops_a_loop_that_took_the_halt_over(
    client: TestClient, manager_clock
):
    manager = get_manager()
    lease_id = _armed(manager)
    assert manager.request_stop(StopKind.HALT, reason="hold").holding
    with manager._lock:
        stale = manager._halt
    manager.release_lease(lease_id)
    manager.operator_heartbeat(OPERATOR)
    manager.start_teleop(30, operator_id=OPERATOR)  # the loop takes the hold over

    result = manager._dispatch_stop(StopKind.COLLAPSE, reason="late timer", expected_halt=stale)

    assert result.applied is False
    assert manager.mode.value == "teleop"
    assert manager.soft_stop("done") is True

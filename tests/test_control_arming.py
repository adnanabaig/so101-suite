"""Arming preconditions (docs/audit/v0.1/safety.md).

- safety-B10: a follower resting outside the guard envelope can still be armed
  and teleoperated. The measured rest pose is held in place, and commands may
  only move a joint back toward the envelope, never further out. A rest pose
  far outside the envelope still fails closed. Converts safety repro 06.
- safety-B12: observe drops torque, so it is refused while armed. Converts
  safety repro 08.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.hardware.safety import SafetyGuard, SafetyViolation  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlRejected,
    Mode,
    StopKind,
    get_manager,
)
from so101.services.store import get_store  # noqa: E402

OPERATOR = "test-operator"
# repro 06: a 2491-tick shoulder_lift sweep calibrates to +-109.5 deg, so the
# arm folded on its stop reads about -109.5 against a +-100 deg envelope.
REST = -(2491 / 2) * 360.0 / 4095.0


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _rest_lift_at(degrees: float) -> None:
    bus = get_manager().follower().bus
    bus._pos["shoulder_lift"] = degrees
    bus._goal["shoulder_lift"] = degrees


def _lease(client: TestClient) -> str:
    response = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 30})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _command(client: TestClient, lease_id: str, sequence: int, **targets: float):
    return client.post(
        "/api/control/command",
        json={
            "lease_id": lease_id,
            "source": "ui",
            "sequence": sequence,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "ttl_ms": 500,
            "deadman": True,
            "targets": targets,
        },
    )


def _wait_for(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --- safety-B10: resting outside the envelope -------------------------------


def test_safety_b10_repro_06_a_follower_resting_outside_the_envelope_arms(client: TestClient):
    _rest_lift_at(REST)
    lease_id = _lease(client)

    armed = client.post("/api/control/arm", json={"lease_id": lease_id})

    assert armed.status_code == 200, armed.text
    state = client.get("/api/control/state").json()
    assert state["armed"] is True and state["safety_state"] == "armed"
    manager = get_manager()
    assert manager.fault is None
    bus = manager.follower().bus
    assert bus.torque_enabled is True
    # Held in place: the goal is the measured rest pose, not the envelope edge.
    assert bus._goal["shoulder_lift"] == pytest.approx(REST)


def test_safety_b10_repro_06_teleop_runs_from_outside_the_envelope(client: TestClient):
    _rest_lift_at(REST)
    manager = get_manager()

    started = client.post("/api/teleop/start", json={"fps": 30, "operator_id": OPERATOR})
    assert started.status_code == 200, started.text
    seq0 = manager.telemetry()["sample_seq"]
    assert _wait_for(lambda: manager.telemetry()["sample_seq"] > seq0 + 3)

    assert manager.fault is None
    assert client.get("/api/robot/status").json()["mode"] == "teleop"
    assert client.post("/api/teleop/stop").status_code == 200


def test_safety_b10_a_command_further_out_holds_the_joint_in_place(client: TestClient):
    _rest_lift_at(REST)
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200

    response = _command(client, lease_id, 1, shoulder_lift=-120.0)

    assert response.status_code == 200, response.text
    assert response.json()["targets"]["shoulder_lift"] == pytest.approx(REST)
    assert get_manager().fault is None


def test_safety_b10_commands_only_move_back_toward_the_envelope(client: TestClient):
    _rest_lift_at(REST)
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200

    toward = _command(client, lease_id, 1, shoulder_lift=-95.0)
    assert toward.status_code == 200, toward.text
    moved = toward.json()["targets"]["shoulder_lift"]
    assert REST < moved <= -95.0

    seen = [moved]
    for sequence in range(2, 12):
        response = _command(client, lease_id, sequence, shoulder_lift=-95.0)
        assert response.status_code == 200, response.text
        seen.append(response.json()["targets"]["shoulder_lift"])
        time.sleep(0.01)
    assert seen == sorted(seen), "a joint outside the envelope only moves back toward it"
    assert seen[-1] == pytest.approx(-95.0)

    # Back inside, the envelope applies again.
    back_out = _command(client, lease_id, 12, shoulder_lift=-120.0)
    assert back_out.status_code == 200, back_out.text
    assert back_out.json()["targets"]["shoulder_lift"] >= -101.0
    assert get_manager().fault is None


def test_safety_b10_a_rest_pose_far_outside_the_envelope_still_fails_closed(
    client: TestClient,
):
    _rest_lift_at(-140.0)
    lease_id = _lease(client)

    armed = client.post("/api/control/arm", json={"lease_id": lease_id})

    assert armed.status_code == 409, armed.text
    assert "envelope" in armed.json()["detail"]
    assert client.get("/api/control/state").json()["armed"] is False
    assert get_manager().follower().bus.torque_enabled is False


def test_safety_b10_guard_holds_a_rest_pose_and_only_moves_toward_the_envelope():
    guard = SafetyGuard(max_step_deg=100.0, max_velocity_deg_s=10.0)
    guard.seed({"shoulder_lift": -109.5, "elbow_flex": 104.0, "shoulder_pan": 0.0})

    # The hold-in-place prime is accepted.
    guard.record_actual_command({"shoulder_lift": -109.5, "elbow_flex": 104.0})
    assert guard.outside_envelope == {"shoulder_lift": -109.5, "elbow_flex": 104.0}

    # Further out holds; toward the envelope moves at the rate limit.
    held = guard.clamp({"shoulder_lift": -130.0, "elbow_flex": 120.0}, dt=0.1)
    assert held == {"shoulder_lift": -109.5, "elbow_flex": 104.0}
    toward = guard.clamp({"shoulder_lift": 0.0, "elbow_flex": 0.0}, dt=0.1)
    assert toward == {"shoulder_lift": pytest.approx(-108.5), "elbow_flex": pytest.approx(103.0)}
    # Never back out past the last accepted command.
    assert guard.clamp({"shoulder_lift": -109.5}, dt=0.1) == {
        "shoulder_lift": pytest.approx(-108.5)
    }


def test_safety_b10_guard_accepts_a_driver_clip_between_the_rest_pose_and_the_goal():
    guard = SafetyGuard()
    guard.seed({"shoulder_lift": -109.5})
    guard.record_actual_command({"shoulder_lift": -109.5})
    assert guard.clamp({"shoulder_lift": -100.0}, dt=0.1) == {"shoulder_lift": -100.0}

    # LeRobot's max_relative_target clips against the present position, which
    # still lags outside the envelope: that is accepted, beyond the rest pose is not.
    guard.record_actual_command({"shoulder_lift": -104.5})
    with pytest.raises(SafetyViolation, match="out-of-range"):
        guard.record_actual_command({"shoulder_lift": -110.0})


def test_safety_b10_guard_limits_preview_to_the_reachable_range():
    guard = SafetyGuard()
    guard.seed({"shoulder_lift": -109.5, "elbow_flex": 0.0})

    assert guard.effective_limits("shoulder_lift") == (-109.5, 100.0)
    assert guard.effective_limits("elbow_flex") == (-100.0, 99.0)
    guard.seed(None)
    assert guard.effective_limits("shoulder_lift") == (-101.0, 100.0)
    assert guard.outside_envelope == {}


def test_safety_b10_guard_refuses_a_rest_pose_beyond_its_tolerance():
    with pytest.raises(SafetyViolation, match="envelope"):
        SafetyGuard().seed({"shoulder_lift": -140.0})
    with pytest.raises(SafetyViolation, match="envelope"):
        SafetyGuard(rest_tolerance_deg=0.0).seed({"shoulder_lift": -101.5})
    with pytest.raises(ValueError, match="rest_tolerance_deg"):
        SafetyGuard(rest_tolerance_deg=-1.0)


# --- safety-B12: observe while armed -------------------------------------------


def test_safety_b12_repro_08_observe_while_armed_is_refused(client: TestClient):
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    follower = get_manager().follower()

    observed = client.post("/api/control/observe", json={"lease_id": lease_id})

    assert observed.status_code == 409, observed.text
    assert "disarm" in observed.json()["detail"]
    assert follower.connected and follower.bus.torque_enabled is True
    assert client.get("/api/control/state").json()["armed"] is True
    assert _command(client, lease_id, 1, shoulder_pan=1.0).status_code == 200


def test_safety_b12_observe_after_disarm_still_reads_the_pose(client: TestClient):
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    assert client.post("/api/control/disarm", json={"lease_id": lease_id}).status_code == 200
    # The owner disarm is a Collapse: it ends at rest with the motors off.
    assert _wait_for(lambda: get_manager().status()["stop_state"] == "none")

    observed = client.post("/api/control/observe", json={"lease_id": lease_id})

    assert observed.status_code == 200, observed.text
    assert "shoulder_pan.pos" in observed.json()["observation"]
    assert get_manager().follower().bus.torque_enabled is False


def test_observe_while_halted_reads_the_pose_and_keeps_the_hold(client: TestClient):
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    assert client.post("/api/robot/halt").status_code == 200

    observed = client.post("/api/control/observe", json={"lease_id": lease_id})

    assert observed.status_code == 200, observed.text
    assert "shoulder_pan.pos" in observed.json()["observation"]
    # Observe reads with torque off only when it found torque off; disconnecting
    # a halted arm would drop it (design doc §5).
    bus = get_manager().follower().bus
    assert bus.torque_enabled is True and bus.connected is True
    assert get_manager().status()["stop_state"] == "halted"


def test_safety_b12_an_arm_that_lands_while_observe_waits_for_the_bus_wins(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    lease_id = _lease(client)
    acquire = manager._acquire_io_for_request

    def arm_first(what: str) -> None:
        if what == "observe":
            # The arm request wins the race for the bus.
            monkeypatch.setattr(manager, "_acquire_io_for_request", acquire)
            manager.arm_control(lease_id)
        acquire(what)

    monkeypatch.setattr(manager, "_acquire_io_for_request", arm_first)

    observed = client.post("/api/control/observe", json={"lease_id": lease_id})

    assert observed.status_code == 409, observed.text
    follower = manager.follower()
    assert follower.connected and follower.bus.torque_enabled is True
    assert client.get("/api/control/state").json()["armed"] is True
    assert manager._io_lock.acquire(timeout=0.5), "observe released the bus"
    manager._io_lock.release()


# --- a stop that lands while arm_control waits for the bus ------------------------------
#
# arm_control checks its authority under the state lock, then takes the bus
# lock for its I/O. A Halt, Collapse, global stop or fault can complete in
# between; the arm must notice before any torque or goal write, not after.


class _EnergizingWrites:
    """Counts torque enables and goal writes on the mock follower bus."""

    def __init__(self, bus, monkeypatch) -> None:
        self.torque_on = 0
        self.goals = 0
        set_torque, write_positions = bus.set_torque, bus.write_positions

        def counted_set_torque(enabled: bool) -> None:
            self.torque_on += bool(enabled)
            set_torque(enabled)

        def counted_write_positions(goal):
            self.goals += 1
            return write_positions(goal)

        monkeypatch.setattr(bus, "set_torque", counted_set_torque)
        monkeypatch.setattr(bus, "write_positions", counted_write_positions)

    @property
    def total(self) -> int:
        return self.torque_on + self.goals


class _ArmInTheGap:
    """Runs arm_control on a thread and parks it before its ``nth`` bus-lock wait."""

    def __init__(self, manager, monkeypatch, lease_id: str, *, nth: int) -> None:
        self.entered, self.go = threading.Event(), threading.Event()
        self.error: BaseException | None = None
        self.result: dict | None = None
        acquire = manager._acquire_io_for_request
        calls = {"arm": 0}

        def gated(what: str) -> None:
            if what == "arm":
                calls["arm"] += 1
                if calls["arm"] == nth:
                    self.entered.set()
                    assert self.go.wait(5)
            acquire(what)

        monkeypatch.setattr(manager, "_acquire_io_for_request", gated)

        def run() -> None:
            try:
                self.result = manager.arm_control(lease_id)
            except BaseException as exc:  # reported by the test
                self.error = exc

        self.thread = threading.Thread(target=run, name="arm-in-the-gap")
        self.thread.start()
        assert self.entered.wait(5), "arm_control never reached the bus lock"

    def finish(self) -> None:
        self.go.set()
        self.thread.join(5)
        assert not self.thread.is_alive()


def _stops(manager, lease_id: str, manager_clock) -> dict:
    return {
        "halt": lambda: manager.request_stop(StopKind.HALT, reason="operator halt"),
        "collapse": lambda: manager.request_stop(StopKind.COLLAPSE, reason="operator collapse"),
        "lease-release": lambda: manager.release_lease(lease_id),
        "fault": lambda: manager._set_fault("injected bus fault", source="test"),
        "estop": lambda: manager.estop(),
        "lease-lapses": lambda: manager_clock.advance(60),
    }


STOPS = ["halt", "collapse", "lease-release", "fault", "estop", "lease-lapses"]


@pytest.mark.parametrize("stop", STOPS)
def test_a_stop_before_torque_enable_means_no_torque_or_goal_write(
    client: TestClient, manager_clock, supervise, monkeypatch, stop: str
):
    manager = get_manager()
    lease_id = _lease(client)
    gap = _ArmInTheGap(manager, monkeypatch, lease_id, nth=2)  # the torque phase
    bus = manager.follower().bus
    assert bus.connected and bus.torque_enabled is False  # read, primed, not enabled
    writes = _EnergizingWrites(bus, monkeypatch)

    try:
        _stops(manager, lease_id, manager_clock)[stop]()
    finally:
        gap.finish()

    assert isinstance(gap.error, ControlRejected), gap.error
    assert "authority changed" in str(gap.error)
    assert writes.total == 0, f"{writes.torque_on} torque enable(s), {writes.goals} goal write(s)"
    assert bus.torque_enabled is False
    assert manager.status()["torque_state"] == "off"
    assert client.get("/api/control/state").json()["armed"] is False
    assert not [e for e in get_store().list_events(200) if e["type"] == "control.armed"]


@pytest.mark.parametrize("stop", ["halt", "collapse", "lease-release", "fault", "lease-lapses"])
def test_a_stop_before_the_first_bus_access_means_no_bus_write(
    client: TestClient, manager_clock, supervise, monkeypatch, stop: str
):
    manager = get_manager()
    lease_id = _lease(client)
    bus = manager.follower().bus
    writes = _EnergizingWrites(bus, monkeypatch)
    gap = _ArmInTheGap(manager, monkeypatch, lease_id, nth=1)  # before connect primes a goal

    try:
        _stops(manager, lease_id, manager_clock)[stop]()
    finally:
        gap.finish()

    assert isinstance(gap.error, ControlRejected), gap.error
    assert writes.total == 0, f"{writes.torque_on} torque enable(s), {writes.goals} goal write(s)"
    assert bus.connected is False and bus.torque_enabled is False
    assert client.get("/api/control/state").json()["armed"] is False


def test_a_halt_before_a_rearm_takes_the_hold_leaves_the_hold_with_the_halt(
    client: TestClient, manager_clock, supervise, monkeypatch
):
    manager = get_manager()
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    assert manager.request_stop(StopKind.HALT, reason="first halt").holding
    follower = manager.follower()
    taken_over = {"n": 0}
    resume_hold = follower.resume_hold

    def counted_resume_hold():
        taken_over["n"] += 1
        return resume_hold()

    monkeypatch.setattr(follower, "resume_hold", counted_resume_hold)
    gap = _ArmInTheGap(manager, monkeypatch, lease_id, nth=2)
    writes = _EnergizingWrites(follower.bus, monkeypatch)

    try:
        manager.request_stop(StopKind.HALT, reason="operator halt")
    finally:
        gap.finish()

    assert isinstance(gap.error, ControlRejected), gap.error
    assert taken_over["n"] == 0, "the re-arm took over a hold the halt owns"
    assert writes.total == 0
    status = manager.status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"
    assert client.get("/api/control/state").json()["armed"] is False


def test_an_arm_with_no_stop_in_the_gap_still_arms(
    client: TestClient, manager_clock, supervise, monkeypatch
):
    manager = get_manager()
    lease_id = _lease(client)
    gap = _ArmInTheGap(manager, monkeypatch, lease_id, nth=2)
    gap.finish()

    assert gap.error is None and gap.result is not None and gap.result["armed"] is True
    assert manager.follower().bus.torque_enabled is True


@pytest.mark.parametrize(
    ("torque_phase", "change", "reason"),
    [
        (False, lambda m: setattr(m, "_lease", None), "released or replaced"),
        (False, lambda m: m.guard.estop(), "E-STOP"),
        (False, lambda m: setattr(m, "_fault", "boom"), "runtime fault: boom"),
        (False, lambda m: setattr(m, "cal_in_progress", True), "calibration"),
        (False, lambda m: setattr(m, "_stop_state", "collapsing"), "Collapse"),
        (False, lambda m: setattr(m, "mode", Mode.MANUAL), "busy (manual)"),
        (True, lambda m: setattr(m, "_armed", False), "disarmed"),
        (True, lambda m: setattr(m, "mode", Mode.TELEOP), "busy (teleop)"),
        (True, lambda m: setattr(m, "_stop_state", "halted"), "a Halt took the arm"),
        (False, lambda m: setattr(m, "_authority_epoch", m._authority_epoch + 1), "revoked"),
    ],
)
def test_every_lost_authority_has_a_reason(client: TestClient, torque_phase, change, reason):
    manager = get_manager()
    lease_id = _lease(client)
    with manager._lock:
        epoch = manager._authority_epoch
        manager._armed = torque_phase
        before = manager._arm_authority_lost_locked(lease_id, epoch, torque_phase=torque_phase)
        change(manager)
        lost = manager._arm_authority_lost_locked(lease_id, epoch, torque_phase=torque_phase)
        manager.guard.reset()
        manager._fault, manager.cal_in_progress, manager._stop_state = None, False, "none"
        manager.mode, manager._armed = Mode.IDLE, False
    assert before is None
    assert lost is not None and reason in lost


def test_a_manual_mode_from_the_own_lease_does_not_cancel_the_torque_phase(client: TestClient):
    manager = get_manager()
    lease_id = _lease(client)
    with manager._lock:
        manager._armed, manager.mode = True, Mode.MANUAL
        lost = manager._arm_authority_lost_locked(
            lease_id, manager._authority_epoch, torque_phase=True, holding=True
        )
        manager._armed, manager.mode = False, Mode.IDLE
    assert lost is None


def test_an_estop_latched_during_the_torque_enable_write_is_cut_after_it(
    client: TestClient, monkeypatch
):
    """The last check, after the write: a stop that landed during the enable itself."""

    manager = get_manager()
    lease_id = _lease(client)
    follower = manager.follower()
    enable = follower.enable_torque

    def enable_then_estop():
        measured = enable()
        manager.latch_estop("E-stop during the enable", lock_timeout=0.05)
        return measured

    monkeypatch.setattr(follower, "enable_torque", enable_then_estop)

    response = client.post("/api/control/arm", json={"lease_id": lease_id})

    assert response.status_code == 409 and "enabling torque" in response.json()["detail"]
    assert follower.bus.torque_enabled is False
    assert manager.status()["torque_state"] == "off"
    assert client.get("/api/control/state").json()["armed"] is False


def test_a_failed_torque_enable_latches_a_fault_and_cuts_torque(client: TestClient, monkeypatch):
    manager = get_manager()
    lease_id = _lease(client)
    bus = manager.follower().bus
    set_torque = bus.set_torque

    def enable_fails(enabled: bool) -> None:
        if enabled:
            bus._torque, bus._torque_state = True, "on"  # a partial enable
            raise ConnectionError("no status packet")
        set_torque(enabled)

    monkeypatch.setattr(bus, "set_torque", enable_fails)

    response = client.post("/api/control/arm", json={"lease_id": lease_id})

    assert response.status_code == 409
    assert "could not enable follower torque" in response.json()["detail"]
    assert manager.fault is not None and "no status packet" in manager.fault
    assert bus.torque_enabled is False and manager.status()["torque_state"] == "off"

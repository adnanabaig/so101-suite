"""C2 deliberate resets and safety-B8 latch persistence.

- C2 (control-hw brief; safety-B3 reset bodies, docs/audit/v0.1/safety.md):
  ``/api/robot/estop/reset`` and ``/api/robot/fault/reset`` need
  ``{"operator_id": str, "confirm": true}``. Status reports
  ``torque_unconfirmed``. While torque is unconfirmed, fault/reset retries a
  real torque-off with read-back and answers 409 unless the operator
  acknowledges that physical power was cut.
- safety-B8 / devex-B2 (docs/audit/v0.1/devex.md): the E-stop latch, faults
  and the unconfirmed-torque flag survive a restart, including a crash or a
  ``--reload``. Converts safety repro 11.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.hardware.motors import MockBus  # noqa: E402
from so101.services import robot_manager as robot_manager_module  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlConflict,
    RobotManager,
    StopKind,
    get_manager,
)
from so101.services.store import ControlStore, get_store, reset_store  # noqa: E402

OPERATOR = "test-operator"
CONFIRM = {"operator_id": OPERATOR, "confirm": True}
RESETS = ["/api/robot/estop/reset", "/api/robot/fault/reset"]


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _bus(manager: RobotManager | None = None) -> MockBus:
    bus = (manager or get_manager()).follower().bus
    assert isinstance(bus, MockBus)
    return bus


def _status(client: TestClient) -> dict:
    return client.get("/api/robot/status").json()


def _safety_state(client: TestClient) -> str:
    return client.get("/api/control/state").json()["safety_state"]


def _crash_and_restart() -> RobotManager:
    """A new process on the same data dir. A crash runs no shutdown code."""

    old = robot_manager_module._manager
    if old is not None:
        old._watchdog_shutdown.set()  # only so the orphan cannot touch shared state
    robot_manager_module._manager = None
    reset_store()
    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    return manager


def _wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _events(event_type: str) -> list[dict]:
    return [event for event in get_store().list_events(500) if event["type"] == event_type]


def _torque_unconfirmed_estop(client: TestClient) -> None:
    client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5})
    lease = get_manager()._lease
    assert lease is not None
    assert client.post("/api/control/arm", json={"lease_id": lease.id}).status_code == 200
    _bus().fail_torque_off = True
    response = client.post("/api/robot/estop")
    assert response.status_code == 503, response.text
    assert _status(client)["torque_unconfirmed"] is True


# --- C2: reset bodies --------------------------------------------------------


@pytest.mark.parametrize("path", RESETS)
@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"operator_id": OPERATOR},
        {"confirm": True},
        {"operator_id": OPERATOR, "confirm": False},
        {"operator_id": OPERATOR, "confirm": 1},
        {"operator_id": OPERATOR, "confirm": "true"},
        {"operator_id": "", "confirm": True},
    ],
    ids=[
        "no-body",
        "empty",
        "no-confirm",
        "no-operator",
        "confirm-false",
        "confirm-int",
        "confirm-string",
        "blank-operator",
    ],
)
def test_c2_resets_without_a_deliberate_body_are_422(client: TestClient, path: str, body):
    assert client.post("/api/robot/estop").status_code == 200
    manager = get_manager()
    manager._set_fault("injected fault", source="test")

    if body is None:
        response = client.post(path)
    else:
        response = client.post(path, json=body)

    assert response.status_code == 422, response.text
    assert _status(client)["estopped"] is True
    assert manager.fault == "injected fault"


@pytest.mark.parametrize("path", RESETS)
def test_c2_a_form_style_simple_request_cannot_reset(client: TestClient, path: str):
    assert client.post("/api/robot/estop").status_code == 200

    response = client.post(
        path,
        content="operator_id=x&confirm=true",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 422
    assert _status(client)["estopped"] is True


def test_c2_estop_reset_with_a_body_clears_the_latch_and_names_the_operator(
    client: TestClient,
):
    assert client.post("/api/robot/estop").status_code == 200

    response = client.post("/api/robot/estop/reset", json=CONFIRM)

    assert response.status_code == 200, response.text
    assert _status(client)["estopped"] is False
    assert _safety_state(client) == "disarmed"
    assert _events("safety.estop_reset")[0]["detail"]["operator_id"] == OPERATOR


def test_c2_fault_reset_with_a_body_clears_the_fault_and_names_the_operator(
    client: TestClient,
):
    get_manager()._set_fault("injected fault", source="test")

    response = client.post("/api/robot/fault/reset", json=CONFIRM)

    assert response.status_code == 200, response.text
    assert get_manager().fault is None
    assert _safety_state(client) == "disarmed"
    assert _events("fault.cleared")[0]["detail"]["operator_id"] == OPERATOR


# --- C2: three-state torque truth ------------------------------------------


def test_c2_status_reports_torque_unconfirmed(client: TestClient):
    assert _status(client)["torque_unconfirmed"] is False

    _torque_unconfirmed_estop(client)

    status = _status(client)
    assert status["torque_unconfirmed"] is True
    # The port closes anyway: "not connected" never stands in for "torque off".
    assert status["follower_connected"] is False
    assert _bus(get_manager()).torque_state == "unconfirmed"


def test_global_standalone_intent_clears_only_after_fault_reset_confirms_torque_off(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleared: list[tuple[str, str, bool, bool]] = []
    manager = get_manager()
    manager.adopt_global_energized_intent(
        {"purpose": "standalone Halt", "written_at": "2026-10-09T00:00:00+00:00"}
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "clear_global_energized_intent",
        lambda settings, *, purpose, device_name, **kwargs: cleared.append(
            (
                purpose,
                device_name,
                bool(kwargs.get("force_current_aliases")),
                bool(kwargs.get("recover_corrupt_current")),
            )
        ),
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "read_global_energized_intent",
        lambda settings: None,
    )

    assert manager.status()["torque_unconfirmed"] is True
    manager.clear_fault(operator_id=OPERATOR)

    assert cleared and cleared[0][1:] == ("", False, True)
    assert manager.fault is None
    assert manager.status()["torque_unconfirmed"] is False
    saved = get_store().load_latches()
    assert saved is not None and saved["energized"] is False


def test_power_cut_acknowledgement_clears_global_intent_without_a_live_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    manager.adopt_global_energized_intent(
        {"purpose": "crashed server", "written_at": "2026-10-09T00:00:00+00:00"}
    )
    cleared: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(manager, "_retry_torque_off", lambda reason: False)
    monkeypatch.setattr(manager, "_release_ports", lambda: [])
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "clear_global_energized_intent",
        lambda settings, *, purpose, device_name, **kwargs: cleared.append(
            (
                device_name,
                bool(kwargs.get("force_current_aliases")),
                bool(kwargs.get("recover_corrupt_current")),
            )
        ),
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "read_global_energized_intent",
        lambda settings: None,
    )

    manager.clear_fault(operator_id=OPERATOR, power_cut_acknowledged=True)

    assert cleared == [("", True, True)]
    assert manager.fault is None
    assert manager._global_energized_recovery is False
    assert manager._energized_recovery is False


def test_power_cut_ack_cannot_clear_a_newer_recovery_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    manager.adopt_global_energized_intent(
        {"purpose": "old crash", "written_at": "2026-10-09T00:00:00+00:00"}
    )
    clears: list[str] = []
    monkeypatch.setattr(manager, "_retry_torque_off", lambda reason: False)

    def publish_newer_fault_during_release() -> list[str]:
        manager._mark_torque_unconfirmed(
            "newer leader torque fault",
            event="test.newer_torque_fault",
            source="test",
            devices={"leader"},
        )
        return []

    monkeypatch.setattr(manager, "_release_ports", publish_newer_fault_during_release)
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "clear_global_energized_intent",
        lambda settings, **kwargs: clears.append(str(kwargs.get("device_name"))),
    )

    with pytest.raises(ControlConflict, match="global energized marker"):
        manager.clear_fault(operator_id=OPERATOR, power_cut_acknowledged=True)

    assert clears == []
    assert manager.fault == "newer leader torque fault"
    assert "leader" in manager._recovery_devices


def test_leader_only_global_intent_requires_leader_readback_not_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    manager.adopt_global_energized_intent(
        {
            "purpose": "standalone leader calibration",
            "written_at": "2026-10-09T00:00:00+00:00",
            "markers": [{"device_name": "leader"}],
        }
    )
    follower = manager.follower()
    leader = manager.leader()
    leader_attempts = leader.bus.torque_off_attempts  # type: ignore[attr-defined]
    monkeypatch.setattr(
        follower.bus,
        "connect",
        lambda: (_ for _ in ()).throw(AssertionError("follower-only readback was used")),
    )
    cleared: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "clear_global_energized_intent",
        lambda settings, *, purpose, device_name, **kwargs: cleared.append(
            (
                device_name,
                bool(kwargs.get("force_current_aliases")),
                bool(kwargs.get("recover_corrupt_current")),
            )
        ),
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "read_global_energized_intent",
        lambda settings: None,
    )

    manager.clear_fault(operator_id=OPERATOR)

    assert follower.connected is False
    assert leader.bus.torque_off_attempts > leader_attempts  # type: ignore[attr-defined]
    assert cleared == [("leader", False, True)]
    assert manager._recovery_devices == set()


def test_legacy_unscoped_torque_truth_rechecks_both_motor_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_store().save_latches(
        {
            "fault": "legacy unscoped torque failure",
            "safety_state": "fault",
            "torque_unconfirmed": True,
        },
        version=time.time_ns() // 1_000,
    )
    manager = get_manager()
    follower = manager.follower()
    leader = manager.leader()
    opened: list[str] = []
    follower_connect = follower.bus.connect
    leader_connect = leader.bus.connect
    monkeypatch.setattr(
        follower.bus,
        "connect",
        lambda: (opened.append("follower"), follower_connect())[1],
    )
    monkeypatch.setattr(
        leader.bus,
        "connect",
        lambda: (opened.append("leader"), leader_connect())[1],
    )

    manager.clear_fault(operator_id=OPERATOR)

    assert opened == ["follower", "leader"]
    assert manager._recovery_devices == set()


def test_newer_recovery_scope_cannot_be_cleared_by_an_inflight_older_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    follower = manager.follower()
    follower.connect()
    with manager._lock:
        manager._torque_unconfirmed = True
        manager._add_recovery_devices_locked({"follower"})
    disconnect = follower.disconnect

    def disconnect_then_publish_new_leader_fault() -> None:
        disconnect()
        with manager._lock:
            manager._add_recovery_devices_locked({"leader"})

    monkeypatch.setattr(follower, "disconnect", disconnect_then_publish_new_leader_fault)

    confirmed = manager._cut_torque_pass("racing reset", latch_failure=False)

    assert confirmed is False
    assert "leader" in manager._recovery_devices
    assert manager._torque_unconfirmed is True


def test_global_clear_cannot_interleave_between_intent_write_and_epoch_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    monkeypatch.setattr(
        manager,
        "settings",
        manager.settings.model_copy(update={"backend": "real", "mock": False}),
    )
    write_entered = threading.Event()
    release_write = threading.Event()
    clear_entered = threading.Event()

    def blocked_write(*args, **kwargs) -> tuple[dict[str, object], ...]:
        write_entered.set()
        assert release_write.wait(timeout=5)
        return ()

    def observed_clear(*args, **kwargs) -> None:
        clear_entered.set()

    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "write_global_energized_intent",
        blocked_write,
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "clear_global_energized_intent",
        observed_clear,
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "read_global_energized_intent",
        lambda settings: None,
    )
    with manager._lock:
        older_epoch = manager._recovery_epoch
    writer = threading.Thread(
        target=manager._write_global_recovery_intent,
        args=("new leader fault", {"leader"}),
    )
    writer.start()
    assert write_entered.wait(timeout=3)
    clearer = threading.Thread(
        target=manager._clear_global_energized_recovery,
        args=("older confirmed cut",),
        kwargs={"devices": {"follower"}, "expected_recovery_epoch": older_epoch},
    )
    clearer.start()
    assert not clear_entered.wait(timeout=0.05)

    release_write.set()
    writer.join(timeout=5)
    clearer.join(timeout=5)

    assert not writer.is_alive() and not clearer.is_alive()
    assert not clear_entered.is_set()
    assert manager._recovery_devices == {"leader"}


def test_shutdown_final_bus_lock_timeout_publishes_global_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    manager.follower().connect()
    monkeypatch.setattr(
        manager,
        "settings",
        manager.settings.model_copy(update={"backend": "real", "mock": False}),
    )
    published: list[str] = []
    monkeypatch.setattr(manager, "_dispatch_stop", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_await_collapse", lambda: None)
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "write_global_energized_intent",
        lambda settings, *, purpose, device_name: published.append(device_name) or (),
    )

    class BusyLock:
        def acquire(self, timeout: float | None = None) -> bool:
            return False

    monkeypatch.setattr(manager, "_io_lock", BusyLock())

    manager.shutdown()

    assert published == ["follower"]
    assert manager.status()["torque_unconfirmed"] is True
    assert manager._global_energized_recovery is True


def test_restored_recovery_publication_failure_is_visible_and_unsaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    monkeypatch.setattr(
        manager,
        "settings",
        manager.settings.model_copy(update={"backend": "real", "mock": False}),
    )
    with manager._lock:
        manager._torque_unconfirmed = True
        manager._add_recovery_devices_locked({"leader"})
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "write_global_energized_intent",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("intent disk full")),
    )

    manager.publish_restored_recovery_intent()

    assert manager.fault is not None and "host-global energized intent is unsaved" in manager.fault
    assert manager.status()["latch_persistence"] == "unsaved"
    assert manager._global_intent_unsaved_devices == {"leader"}


def test_rearm_reuses_adopted_intent_without_any_disk_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    monkeypatch.setattr(
        manager,
        "settings",
        manager.settings.model_copy(update={"backend": "real", "mock": False}),
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "enable_intent_is_adopted",
        lambda settings, *, device_name: True,
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "write_global_energized_intent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rearm rewrote the durable journal")
        ),
    )
    monkeypatch.setattr(
        manager,
        "_persist_latches",
        lambda: (_ for _ in ()).throw(AssertionError("rearm rewrote local SQLite")),
    )

    manager._prepare_global_energized_intent("rearm existing Halt")

    assert manager._global_energized_recovery is True
    assert manager._recovery_devices == {"follower"}


def test_confirmed_device_scope_is_not_retired_when_global_clear_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    with manager._lock:
        manager._global_energized_recovery = True
        manager._energized_recovery = True
        manager._add_recovery_devices_locked({"follower"})
    monkeypatch.setattr(manager, "_clear_global_energized_recovery", lambda *a, **k: False)

    retired = manager._retire_confirmed_recovery_devices({"follower"}, "test clear failure")

    assert retired is False
    assert manager._recovery_devices == {"follower"}
    assert manager._energized_recovery is True


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_rehearsal_manager_never_writes_global_hardware_intent(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    monkeypatch.setattr(
        manager,
        "settings",
        manager.settings.model_copy(update={"backend": backend}),
    )
    monkeypatch.setattr(
        robot_manager_module.bus_ownership,
        "write_global_energized_intent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a rehearsal touched global hardware intent")
        ),
    )

    manager._prepare_global_energized_intent("rehearsal")

    assert manager._global_energized_recovery is False
    assert manager._energized_recovery is False


def test_safety_b2_a_loop_whose_teardown_cannot_confirm_torque_off_reports_torque_unconfirmed(
    client: TestClient,
):
    manager = get_manager()
    started = client.post("/api/teleop/start", json={"fps": 30, "operator_id": OPERATOR})
    assert started.status_code == 200, started.text
    bus = _bus(manager)
    seq0 = manager.telemetry()["sample_seq"]
    assert _wait_for(lambda: manager.telemetry()["sample_seq"] > seq0)

    bus.fail_io = True  # the follower bus stops answering mid-loop

    assert _wait_for(lambda: manager.mode.value == "idle")
    status = _status(client)
    # The teardown's torque-off was not confirmed. The port closed anyway, and
    # the loop exit still reports unconfirmed, not merely "faulted".
    assert status["follower_connected"] is False
    assert status["torque_unconfirmed"] is True
    assert manager.fault is not None
    saved = get_store().load_latches()
    assert saved is not None and saved["torque_unconfirmed"] is True


def test_c2_fault_reset_retries_a_real_torque_off_and_clears_when_confirmed(
    client: TestClient,
):
    _torque_unconfirmed_estop(client)
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200
    assert _safety_state(client) == "fault"
    bus = _bus()
    assert bus.torque_enabled is True
    bus.fail_torque_off = False  # the bus answers again

    response = client.post("/api/robot/fault/reset", json=CONFIRM)

    assert response.status_code == 200, response.text
    assert bus.torque_enabled is False
    status = _status(client)
    assert status["torque_unconfirmed"] is False
    assert status["follower_connected"] is False
    assert _safety_state(client) == "disarmed"


def test_c2_fault_reset_is_409_while_torque_off_stays_unconfirmed(client: TestClient):
    _torque_unconfirmed_estop(client)
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200

    response = client.post("/api/robot/fault/reset", json=CONFIRM)

    assert response.status_code == 409, response.text
    assert "power_cut_acknowledged" in response.json()["detail"]
    assert _status(client)["torque_unconfirmed"] is True
    assert get_manager().fault is not None
    assert _safety_state(client) == "fault"


def test_c2_power_cut_acknowledgement_clears_an_unconfirmed_torque_fault(
    client: TestClient,
):
    _torque_unconfirmed_estop(client)
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200
    bus = _bus()
    attempts_before = bus.torque_off_attempts

    response = client.post(
        "/api/robot/fault/reset", json={**CONFIRM, "power_cut_acknowledged": True}
    )

    assert response.status_code == 200, response.text
    # The acknowledgement never skips the retry; it only overrides its failure.
    assert bus.torque_off_attempts > attempts_before
    assert _status(client)["torque_unconfirmed"] is False
    assert get_manager().fault is None
    assert _safety_state(client) == "disarmed"
    acknowledged = _events("safety.power_cut_acknowledged")
    assert acknowledged and acknowledged[0]["detail"]["operator_id"] == OPERATOR


def test_c2_power_cut_acknowledgement_must_be_a_real_boolean(client: TestClient):
    _torque_unconfirmed_estop(client)
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200

    response = client.post("/api/robot/fault/reset", json={**CONFIRM, "power_cut_acknowledged": 1})

    assert response.status_code == 422
    assert _status(client)["torque_unconfirmed"] is True


def test_c2_fault_reset_reopens_the_bus_for_the_retry_without_priming_goals(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    bus = _bus(manager)
    assert bus.connected is False
    manager._mark_torque_unconfirmed(
        "torque-off unconfirmed before restart", event="test.injected", source="test"
    )
    writes: list[dict] = []
    original_write = MockBus.write_positions

    def recording_write(self, goal):
        writes.append(dict(goal))
        return original_write(self, goal)

    monkeypatch.setattr(MockBus, "write_positions", recording_write)

    response = client.post("/api/robot/fault/reset", json=CONFIRM)

    assert response.status_code == 200, response.text
    assert bus.torque_off_attempts >= 1
    assert writes == [], "a torque-off retry must not write goal positions"
    assert bus.connected is False
    assert _status(client)["torque_unconfirmed"] is False


def test_c2_fault_reset_retry_with_a_dead_bus_is_409(client: TestClient):
    manager = get_manager()
    bus = _bus(manager)
    manager._mark_torque_unconfirmed(
        "torque-off unconfirmed before restart", event="test.injected", source="test"
    )
    bus.fail_connect = True

    response = client.post("/api/robot/fault/reset", json=CONFIRM)

    assert response.status_code == 409, response.text
    assert _status(client)["torque_unconfirmed"] is True


# --- safety-B8 / devex-B2: latches survive a restart -----------------------


def test_safety_b8_estop_latch_survives_a_crash_restart(client: TestClient):
    """safety repro 11: a restart used to come back disarmed and armable."""

    assert client.post("/api/robot/estop").status_code == 200

    _crash_and_restart()

    status = _status(client)
    assert status["estopped"] is True
    assert _safety_state(client) == "estopped"
    lease = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5}).json()
    arm = client.post("/api/control/arm", json={"lease_id": lease["id"]})
    assert arm.status_code == 409
    assert "E-STOP" in arm.json()["detail"]
    assert client.delete(f"/api/control/lease/{lease['id']}").status_code == 200
    assert _events("safety.latches_restored")

    # Only the deliberate reset clears it, durably.
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200
    _crash_and_restart()
    assert _status(client)["estopped"] is False
    assert _safety_state(client) == "disarmed"


def test_safety_b8_latch_survives_a_clean_restart(client: TestClient):
    assert client.post("/api/robot/estop").status_code == 200

    robot_manager_module.reset_manager()
    reset_store()

    assert _status(client)["estopped"] is True


def test_safety_b8_runtime_fault_survives_a_restart(client: TestClient):
    get_manager()._set_fault("servo overload on elbow_flex", source="test")

    manager = _crash_and_restart()

    assert manager.fault == "servo overload on elbow_flex"
    assert _safety_state(client) == "fault"
    with pytest.raises(RuntimeError, match="servo overload"):
        manager.start_teleop(30, operator_id=OPERATOR)

    assert client.post("/api/robot/fault/reset", json=CONFIRM).status_code == 200
    assert _crash_and_restart().fault is None


def test_safety_b8_unconfirmed_torque_survives_a_restart_and_needs_the_retry(
    client: TestClient,
):
    _torque_unconfirmed_estop(client)

    _crash_and_restart()

    status = _status(client)
    assert status["estopped"] is True
    assert status["torque_unconfirmed"] is True
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200
    assert _safety_state(client) == "fault"
    # The new process reopens the bus; this time it answers and reads back off.
    response = client.post("/api/robot/fault/reset", json=CONFIRM)
    assert response.status_code == 200, response.text
    assert _status(client)["torque_unconfirmed"] is False


@pytest.mark.parametrize(
    ("kind", "path"),
    [(StopKind.HALT, "/api/robot/halt"), (StopKind.COLLAPSE, "/api/robot/collapse")],
    ids=["halt", "collapse"],
)
def test_stops_never_claim_restored_unconfirmed_torque_is_off(
    client: TestClient, kind: StopKind, path: str
):
    manager = get_manager()
    manager._mark_torque_unconfirmed(
        "torque-off was unconfirmed before restart", event="test.injected", source="test"
    )
    manager = _crash_and_restart()
    assert manager.follower().torque_state == "off", "the fresh wrapper has no hardware history"
    assert manager.status()["torque_state"] == "unconfirmed"

    result = manager.request_stop(kind, reason="restored latch")

    assert result.torque_off is False
    assert result.ok is False
    assert manager.status()["torque_state"] == "unconfirmed"
    response = client.post(path, json={})
    assert response.status_code == 503, response.text


def test_safety_b8_an_unreadable_latch_store_fails_closed(client: TestClient, monkeypatch):
    def broken(self):
        raise OSError("disk I/O error")

    monkeypatch.setattr(ControlStore, "load_latches", broken)

    manager = _crash_and_restart()

    assert manager.fault is not None
    assert "safety latches" in manager.fault
    assert _safety_state(client) == "fault"


def test_safety_b8_a_stale_snapshot_never_overwrites_a_newer_one(tmp_path):
    store = ControlStore(tmp_path / "latches.sqlite3")
    newer = {"estopped": True, "fault": "x", "safety_state": "estopped", "torque_unconfirmed": True}
    older = {
        "estopped": False,
        "fault": None,
        "safety_state": "disarmed",
        "torque_unconfirmed": False,
    }

    store.save_latches(newer, version=5)
    store.save_latches(older, version=4)

    saved = store.load_latches()
    assert saved is not None
    assert saved["estopped"] is True
    assert saved["version"] == 5
    assert ControlStore(tmp_path / "latches.sqlite3").load_latches() == saved


def test_safety_b8_versions_keep_rising_across_restarts(client: TestClient):
    assert client.post("/api/robot/estop").status_code == 200
    _crash_and_restart()

    # A reset in the new process must win over the old process's last write.
    assert client.post("/api/robot/estop/reset", json=CONFIRM).status_code == 200
    _crash_and_restart()

    assert _status(client)["estopped"] is False


# --- portability: a briefly unavailable store file (Windows antivirus) ------


def _sqlite_error(message: str, errorname: str | None) -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError(message)
    if errorname is not None:
        exc.sqlite_errorname = errorname  # set by sqlite3 itself on real errors
    return exc


def _flaky_store(monkeypatch, manager: RobotManager, failures: list[BaseException]) -> list[int]:
    calls: list[int] = []
    original = manager.store.save_latches

    def flaky(latches, *, version):
        calls.append(version)
        if failures:
            raise failures.pop(0)
        return original(latches, version=version)

    monkeypatch.setattr(manager.store, "save_latches", flaky)
    return calls


@pytest.mark.parametrize(
    "transient",
    [
        lambda: PermissionError(13, "Access is denied"),
        lambda: _sqlite_error("disk I/O error", "SQLITE_IOERR"),
        lambda: _sqlite_error("unable to open database file", None),
    ],
    ids=["permission-denied", "sqlite-ioerr", "sqlite-cantopen"],
)
def test_safety_b8_a_briefly_unavailable_store_is_retried(
    client: TestClient, monkeypatch, transient
):
    manager = get_manager()
    calls = _flaky_store(monkeypatch, manager, [transient(), transient()])

    manager._set_fault("injected fault", source="test")

    assert len(calls) == 3
    saved = get_store().load_latches()
    assert saved is not None and saved["fault"] == "injected fault"


def test_safety_b8_a_locked_store_is_not_retried_after_its_busy_timeout(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    # SQLite already waited busy_timeout (5 s) before raising this.
    calls = _flaky_store(monkeypatch, manager, [_sqlite_error("database is locked", "SQLITE_BUSY")])

    manager._set_fault("injected fault", source="test")

    assert len(calls) == 1
    assert manager.fault == "injected fault"


def test_safety_b8_a_dead_store_gives_up_quickly_and_the_latch_still_holds(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    calls = _flaky_store(
        monkeypatch, manager, [_sqlite_error("disk I/O error", "SQLITE_IOERR") for _ in range(9)]
    )

    started = time.monotonic()
    manager._set_fault("injected fault", source="test")

    assert len(calls) == 3
    assert time.monotonic() - started < 1.0
    assert manager.fault == "injected fault"
    assert _safety_state(client) == "fault"


# --- a latch the store refused: the fallback marker -------------------------------------


def _marker_path():
    from so101.config import get_settings
    from so101.services.store import control_store_path, latch_marker_path, safety_latch_identity

    settings = get_settings()
    return latch_marker_path(control_store_path(settings), safety_latch_identity(settings))


def _marker_identity() -> dict[str, str]:
    from so101.config import get_settings
    from so101.services.store import safety_latch_identity

    return safety_latch_identity(get_settings()).as_dict()


def _dead_store(monkeypatch, manager: RobotManager) -> list[int]:
    """Every latch write fails, as on a full or failing disk."""

    calls: list[int] = []

    def refuse(latches, *, version):
        calls.append(version)
        raise _sqlite_error("disk I/O error", "SQLITE_IOERR")

    monkeypatch.setattr(manager.store, "save_latches", refuse)
    return calls


def test_safety_b8_a_latch_the_store_refused_still_comes_back_after_a_restart(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _bus(manager)  # the follower exists, so the E-stop cuts a real (mock) bus
    assert get_store().load_latches() is None  # nothing saved yet: a restart would be clear
    _dead_store(monkeypatch, manager)

    assert client.post("/api/robot/estop").status_code == 200

    assert _marker_path().is_file()
    assert _status(client)["latch_persistence"] == "marker"
    (unsaved, *_) = _events("safety.latches_unsaved")
    assert unsaved["level"] == "critical" and unsaved["detail"]["marker_written"] is True
    assert unsaved["detail"]["latches"]["estopped"] is True

    restarted = _crash_and_restart()

    status = _status(client)
    assert status["estopped"] is True
    assert restarted.fault is not None
    assert "could not be saved before the restart" in restarted.fault
    assert _safety_state(client) == "estopped"
    assert status["latch_persistence"] == "marker"
    (restored,) = _events("safety.latches_restored")
    assert restored["level"] == "critical"
    assert restored["detail"]["unsaved_marker"]["latches"]["estopped"] is True


def test_safety_b8_a_refused_fault_comes_back_as_a_fault_after_a_restart(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _dead_store(monkeypatch, manager)

    manager._set_fault("servo 3 overload", source="test")

    restarted = _crash_and_restart()
    assert restarted.fault is not None
    assert "could not be saved before the restart" in restarted.fault
    assert "servo 3 overload" in restarted.fault
    assert _safety_state(client) == "fault"
    lease = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5}).json()
    armed = client.post("/api/control/arm", json={"lease_id": lease["id"]})
    assert armed.status_code == 409 and "fault" in armed.json()["detail"]


def test_safety_b8_a_later_save_removes_the_marker(client: TestClient, monkeypatch):
    manager = get_manager()
    _flaky_store(monkeypatch, manager, [_sqlite_error("disk I/O error", "SQLITE_IOERR")] * 3)

    manager._set_fault("injected fault", source="test")
    assert _marker_path().is_file()

    # The store answers again: the next snapshot (newer) reaches it.
    assert client.post("/api/robot/fault/reset", json=CONFIRM).status_code == 200

    assert not _marker_path().exists()
    assert _status(client)["latch_persistence"] == "saved"
    restarted = _crash_and_restart()
    assert restarted.fault is None and _safety_state(client) == "disarmed"


def test_safety_b8_the_marker_survives_until_the_new_process_saves(client: TestClient, monkeypatch):
    manager = get_manager()
    _dead_store(monkeypatch, manager)
    manager._set_fault("injected fault", source="test")

    _crash_and_restart()
    assert _marker_path().is_file()
    _crash_and_restart()  # a second restart before anyone acted: still faulted
    assert get_manager().fault is not None and _marker_path().is_file()

    response = client.post("/api/robot/fault/reset", json=CONFIRM)

    assert response.status_code == 200, response.text
    assert not _marker_path().exists()
    assert _crash_and_restart().fault is None


def test_safety_b8_when_even_the_marker_fails_status_shows_the_latch_is_unsaved(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _dead_store(monkeypatch, manager)

    def no_disk(path, payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(robot_manager_module, "atomic_write_json", no_disk)

    manager._set_fault("injected fault", source="test")

    assert manager.fault == "injected fault"  # the in-memory latch holds
    assert _safety_state(client) == "fault"
    assert not _marker_path().exists()
    status = _status(client)
    assert status["latch_persistence"] == "unsaved"
    (unsaved,) = _events("safety.latches_unsaved")
    assert unsaved["level"] == "critical"
    assert unsaved["detail"]["marker_written"] is False
    assert "No space left" in unsaved["detail"]["marker_error"]
    assert "only in memory" in unsaved["message"]


def test_safety_b8_a_snapshot_that_latches_nothing_writes_no_marker(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _dead_store(monkeypatch, manager)

    manager._persist_latches()  # nothing latched: the stored row is no less latched

    assert not _marker_path().exists()
    assert _status(client)["latch_persistence"] == "saved"
    assert _events("safety.latches_unsaved") == []


def test_safety_b8_an_unreadable_marker_fails_closed(client: TestClient):
    _marker_path().parent.mkdir(parents=True, exist_ok=True)
    _marker_path().write_bytes(b"")  # a power cut can leave an empty file

    restarted = _crash_and_restart()

    assert restarted.fault is not None and "unreadable" in restarted.fault
    status = _status(client)
    assert status["torque_unconfirmed"] is True and status["latch_persistence"] == "marker"
    # The deliberate reset still works: its torque-off retry confirms, then
    # the saved snapshot removes the marker.
    assert client.post("/api/robot/fault/reset", json=CONFIRM).status_code == 200
    assert not _marker_path().exists()
    assert _status(client)["torque_unconfirmed"] is False


def test_safety_b8_a_marker_older_than_the_stored_snapshot_is_obsolete(client: TestClient):
    manager = get_manager()
    manager._persist_latches()  # a clear snapshot reaches the store
    saved = get_store().load_latches()
    assert saved is not None
    robot_manager_module.atomic_write_json(
        _marker_path(),
        {
            "kind": "so101.latch-unsaved",
            "schema_version": 2,
            "identity": _marker_identity(),
            "version": saved["version"] - 1,
            "latches": {"estopped": False, "fault": "old", "torque_unconfirmed": False},
            "error": "OperationalError: disk I/O error",
        },
    )

    restarted = _crash_and_restart()

    assert restarted.fault is None
    assert not _marker_path().exists()
    assert _status(client)["latch_persistence"] == "saved"


def test_safety_b8_an_unreadable_store_with_a_marker_keeps_both(client: TestClient, monkeypatch):
    manager = get_manager()
    _dead_store(monkeypatch, manager)
    manager._set_fault("injected fault", source="test")

    def broken(self):
        raise OSError("disk I/O error")

    monkeypatch.setattr(ControlStore, "load_latches", broken)
    restarted = _crash_and_restart()

    assert restarted.fault is not None and "safety latches" in restarted.fault
    assert _marker_path().is_file()
    assert _status(client)["latch_persistence"] == "marker"


def test_safety_b8_an_older_writer_never_retires_a_newer_marker(client: TestClient):
    manager = get_manager()
    base = manager._latch_version + 1_000

    manager._latches_unsaved({"fault": "newer"}, base + 20, OSError("disk I/O error"))
    assert _marker_path().is_file() and _status(client)["latch_persistence"] == "marker"

    manager._latches_saved(base + 10)  # an overtaken writer finished late
    assert _marker_path().is_file() and _status(client)["latch_persistence"] == "marker"

    manager._latches_saved(base + 30)  # a newer snapshot reached the store
    assert not _marker_path().exists() and _status(client)["latch_persistence"] == "saved"

    manager._latches_unsaved({"fault": "older"}, base + 25, OSError("disk I/O error"))
    assert not _marker_path().exists(), "a snapshot older than the stored one needs no marker"


def test_safety_b8_a_marker_that_cannot_be_updated_still_faults_the_restart(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    base = manager._latch_version + 1_000
    manager._latches_unsaved({"fault": "first"}, base + 1, OSError("disk I/O error"))

    def no_disk(path, payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(robot_manager_module, "atomic_write_json", no_disk)
    manager._latches_unsaved({"estopped": True}, base + 2, OSError("disk I/O error"))

    assert _status(client)["latch_persistence"] == "marker"
    last = _events("safety.latches_unsaved")[0]  # newest first
    assert last["detail"]["marker_written"] is True and "could not be updated" in last["message"]


def test_safety_b8_a_marker_that_cannot_be_removed_is_retired_at_the_next_start(
    client: TestClient, monkeypatch
):
    from pathlib import Path

    manager = get_manager()
    io_error = _sqlite_error("disk I/O error", "SQLITE_IOERR")
    calls = _flaky_store(monkeypatch, manager, [io_error] * 3)
    manager._set_fault("first fault", source="test")
    assert len(calls) == 3 and _marker_path().is_file()
    unlink = Path.unlink

    def stuck(self, missing_ok=False):
        if self.name.endswith(".latch-unsaved"):
            raise PermissionError(13, "Access is denied")
        return unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", stuck)
    manager._set_fault("second fault", source="test")  # reaches the store
    monkeypatch.setattr(Path, "unlink", unlink)

    assert _marker_path().is_file()  # fail closed: it stays
    assert _status(client)["latch_persistence"] == "saved"
    restarted = _crash_and_restart()
    # The stored snapshot is newer than the marker, so the marker is obsolete.
    assert restarted.fault == "second fault"
    assert not _marker_path().exists()


def test_safety_b8_a_marker_restores_its_unconfirmed_torque(client: TestClient):
    robot_manager_module.atomic_write_json(
        _marker_path(),
        {
            "kind": "so101.latch-unsaved",
            "schema_version": 2,
            "identity": _marker_identity(),
            "version": time.time_ns() // 1_000,
            "latches": {
                "estopped": False,
                "fault": "torque-off failed",
                "torque_unconfirmed": True,
            },
            "error": "OperationalError: disk I/O error",
        },
    )

    restarted = _crash_and_restart()

    assert restarted.fault is not None and "torque-off failed" in restarted.fault
    status = _status(client)
    assert status["torque_unconfirmed"] is True and status["torque_state"] == "unconfirmed"


@pytest.mark.parametrize(
    "write",
    [
        lambda path: path.mkdir(parents=True),
        lambda path: path.write_text('{"kind": "something else", "version": 1, "latches": {}}'),
        lambda path: path.write_text('{"kind": "so101.latch-unsaved", "version": true}'),
    ],
    ids=["a-directory", "another-kind", "bad-version"],
)
def test_safety_b8_any_unreadable_marker_fails_closed(client: TestClient, write):
    _marker_path().parent.mkdir(parents=True, exist_ok=True)
    write(_marker_path())

    restarted = _crash_and_restart()

    assert restarted.fault is not None and "unreadable" in restarted.fault
    assert _status(client)["torque_unconfirmed"] is True

"""Calibration history survives restarts and reports failures without hardware I/O."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.services import robot_manager as manager_module
from so101.services.app import create_app
from so101.services.robot_manager import ControlRejected, get_manager, reset_manager
from so101.services.store import get_store


def test_history_imports_previous_failures_and_filters_unrelated_events(monkeypatch):
    store = get_store()
    store.append_event("calibration.started", "follower calibration started")
    error = (
        "calibration encoder ranges outside 0..4095: {'shoulder_pan': (-542, 2049), "
        "'shoulder_lift': (2034, 4455), 'elbow_flex': (-182, 2168)}; "
        f"post-homing starting readings: {dict.fromkeys(JOINT_NAMES, 2047)}"
    )
    store.append_event("calibration.failed", f"follower calibration failed: {error}")
    for _ in range(505):
        store.append_event("unrelated", "other activity")
    manager = get_manager()
    monkeypatch.setattr(manager, "follower", lambda: pytest.fail("GET opened follower"))
    monkeypatch.setattr(manager, "leader", lambda: pytest.fail("GET opened leader"))
    with TestClient(create_app()) as client:
        response = client.get("/api/calibration/history")
        assert response.status_code == 200
        (attempt,) = response.json()["attempts"]
    assert attempt["result"] == "fail"
    assert attempt["affected_motors"] == ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    assert attempt["error"] == f"follower calibration failed: {error}"
    assert attempt["backend"] is None and attempt["sample_seconds"] is None
    assert len(attempt["log"]) == 2


def test_new_attempt_records_metadata_and_diagnostics_across_restart():
    manager = get_manager()
    manager.claim_calibration("follower", sample_seconds=60)
    manager._calibration_diagnostics = {
        "affected_motors": ["elbow_flex"],
        "starting_positions": dict.fromkeys(JOINT_NAMES, 2047),
        "range_mins": {"elbow_flex": -156},
    }
    manager.finish_calibration("follower", "encoder boundary crossed", saved=False)
    reset_manager()
    with TestClient(create_app()) as client:
        (attempt,) = client.get("/api/calibration/history").json()["attempts"]
    assert attempt["result"] == "fail" and attempt["saved"] is False
    assert attempt["backend"] == "mock" and attempt["sample_seconds"] == 60
    assert attempt["device_id"] == manager.settings.follower_id
    assert attempt["affected_motors"] == ["elbow_flex"]
    assert attempt["diagnostics"]["starting_positions"] == dict.fromkeys(JOINT_NAMES, 2047)
    assert attempt["started_at"] and attempt["finished_at"]
    assert attempt["error"] == "encoder boundary crossed"


def test_history_has_pass_result_after_mock_calibration():
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "leader"}).status_code == 200
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            attempts = client.get("/api/calibration/history").json()["attempts"]
            if attempts and attempts[0]["result"] == "pass":
                break
            time.sleep(0.005)
        else:
            pytest.fail("mock calibration did not finish")
    (attempt,) = attempts
    assert attempt["arm"] == "leader" and attempt["saved"] is True
    assert attempt["error"] is None and attempt["affected_motors"] == []
    assert attempt["sample_seconds"] == 12


def test_worker_failure_retains_encoder_diagnostics_in_history(monkeypatch):
    diagnostics = {
        "affected_motors": ["shoulder_pan"],
        "homing_offsets_requested": {"shoulder_pan": 12},
        "homing_readback": {"shoulder_pan": {"homing_offset": 12}},
        "starting_positions": {"shoulder_pan": 2047},
        "range_mins": {"shoulder_pan": -542},
        "range_maxes": {"shoulder_pan": 2049},
    }

    def failed(bus, arm, **kwargs):
        bus.calibration_diagnostics = diagnostics
        raise ValueError("invalid encoder range for shoulder_pan")

    monkeypatch.setattr(manager_module, "calibrate", failed)
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "follower"}).status_code == 200
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            attempts = client.get("/api/calibration/history").json()["attempts"]
            if attempts and attempts[0]["result"] == "fail":
                break
            time.sleep(0.005)
        else:
            pytest.fail("mock calibration worker did not report its error")
    assert attempts[0]["diagnostics"] == diagnostics
    assert attempts[0]["affected_motors"] == ["shoulder_pan"]
    assert attempts[0]["error"] == "invalid encoder range for shoulder_pan"


def test_file_saved_with_cleanup_failure_is_a_failed_attempt():
    manager = get_manager()
    manager.claim_calibration("follower", sample_seconds=30)
    manager.finish_calibration("follower", "torque-off not confirmed for gripper", saved=True)
    with TestClient(create_app()) as client:
        (attempt,) = client.get("/api/calibration/history").json()["attempts"]
    assert attempt["result"] == "fail" and attempt["saved"] is True
    assert attempt["affected_motors"] == ["gripper"]


def test_history_includes_motors_from_both_sweep_and_cleanup_failures():
    manager = get_manager()
    manager.claim_calibration("follower", sample_seconds=60)
    manager._calibration_diagnostics = {"affected_motors": ["shoulder_pan"]}
    manager.finish_calibration("follower", "torque-off unconfirmed for gripper", saved=False)
    with TestClient(create_app()) as client:
        (attempt,) = client.get("/api/calibration/history").json()["attempts"]
    assert attempt["affected_motors"] == ["shoulder_pan", "gripper"]


def test_unfinished_attempt_is_running_then_incomplete_after_restart():
    manager = get_manager()
    manager.claim_calibration("leader", sample_seconds=60)
    with TestClient(create_app()) as client:
        (attempt,) = client.get("/api/calibration/history").json()["attempts"]
        assert attempt["result"] == "running"
        reset_manager()
        (attempt,) = client.get("/api/calibration/history").json()["attempts"]
        assert attempt["result"] == "incomplete"
        assert attempt["finished_at"] is None


def test_history_write_failure_refuses_before_hardware_is_opened(monkeypatch):
    manager = get_manager()
    for device in (manager.follower(), manager.leader()):
        monkeypatch.setattr(device.bus, "connect", lambda: pytest.fail("hardware opened"))
    append = manager.store.append_event

    def unavailable(kind, *args, **kwargs):
        if kind == "calibration.started":
            raise OSError("disk unavailable")
        return append(kind, *args, **kwargs)

    monkeypatch.setattr(manager.store, "append_event", unavailable)
    with pytest.raises(ControlRejected, match="no hardware opened"):
        manager.claim_calibration("follower", sample_seconds=60)
    assert manager.cal_in_progress is False
    assert not manager.follower().connected and not manager.leader().connected


def test_terminal_history_failure_is_visible_and_never_claims_a_pass(monkeypatch):
    manager = get_manager()
    manager.claim_calibration("leader", sample_seconds=60)
    append = manager.store.append_event

    def unavailable(kind, *args, **kwargs):
        if kind == "calibration.saved":
            raise OSError("disk unavailable")
        return append(kind, *args, **kwargs)

    monkeypatch.setattr(manager.store, "append_event", unavailable)
    manager.finish_calibration("leader", saved=True)
    assert "history persistence failed" in manager.cal_message
    with TestClient(create_app()) as client:
        (attempt,) = client.get("/api/calibration/history").json()["attempts"]
    assert attempt["result"] == "incomplete"


@pytest.mark.parametrize("limit", [0, 201])
def test_history_limit_is_bounded(limit):
    with TestClient(create_app()) as client:
        assert client.get(f"/api/calibration/history?limit={limit}").status_code == 422


def test_history_storage_failure_returns_an_error_instead_of_empty_history(monkeypatch):
    def unavailable(**kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(get_store(), "calibration_events", unavailable)
    with TestClient(create_app()) as client:
        assert client.get("/api/calibration/history").status_code == 503


def test_live_status_is_cached_arm_scoped_and_failure_is_retained(monkeypatch):
    import threading

    manager = get_manager()
    entered, release = threading.Event(), threading.Event()
    frame = {
        "phase": "sampling",
        "sequence": 3,
        "sample_count": 2,
        "elapsed_seconds": 0.01,
        "positions": {"elbow_flex": -197},
        "range_mins": {"elbow_flex": -197},
        "range_maxes": {"elbow_flex": 2047},
    }

    def fake_calibrate(bus, arm, **kwargs):
        assert manager._io_lock.locked()
        kwargs["telemetry"](frame)
        frame["positions"]["elbow_flex"] = 999  # callback owns a copy
        bus.calibration_diagnostics = {"affected_motors": ["elbow_flex"]}
        entered.set()
        assert release.wait(5), "test did not release calibration worker"
        raise ValueError("invalid encoder range for elbow_flex")

    monkeypatch.setattr(manager_module, "calibrate", fake_calibrate)
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "follower"}).status_code == 200
        try:
            assert entered.wait(3)
            # GET completes while the worker is holding the I/O lock.
            live = client.get("/api/calibration/status/follower").json()["live"]
            assert live["positions"]["elbow_flex"] == -197
            assert live["phase"] == "sampling" and live["frozen"] is False
            assert live["attempt_id"] and live["age_seconds"] >= 0
            assert client.get("/api/calibration/status/leader").json()["live"] is None
            live["positions"]["elbow_flex"] = 123  # GET returns a copy too
            assert manager.calibration_live_status("follower")["positions"]["elbow_flex"] == -197
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            attempts = client.get("/api/calibration/history").json()["attempts"]
            if attempts[0]["result"] == "fail":
                break
            time.sleep(0.005)
        else:
            pytest.fail("worker did not finish")
        final = client.get("/api/calibration/status/follower").json()["live"]
        assert final["frozen"] and final["phase"] == "failed"
        assert final["positions"]["elbow_flex"] == -197
        assert final["error"] == "invalid encoder range for elbow_flex"
        assert attempts[0]["diagnostics"]["live"]["phase"] == "failed"
        assert attempts[0]["diagnostics"]["live"]["positions"]["elbow_flex"] == -197
        manager.claim_calibration("leader")
        assert manager.calibration_live_status("leader") is None
        manager.finish_calibration("leader", "test cancelled", saved=False)


def test_native_completion_is_not_pass_until_manager_cleanup_finishes(monkeypatch):
    import threading

    entered, release = threading.Event(), threading.Event()
    original = manager_module.calibrate

    def native_complete(bus, arm, **kwargs):
        result = original(bus, arm, **kwargs)
        kwargs["telemetry"](
            {
                "phase": "complete",
                "sequence": 5,
                "sample_count": 2,
                "elapsed_seconds": 8,
            }
        )
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(manager_module, "calibrate", native_complete)
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "leader"}).status_code == 200
        try:
            assert entered.wait(3)
            pending = client.get("/api/calibration/status/leader").json()
            assert pending["in_progress"] and pending["live"]["phase"] == "saving"
            assert not pending["live"]["frozen"]
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            attempts = client.get("/api/calibration/history").json()["attempts"]
            if attempts[0]["result"] == "pass":
                break
            time.sleep(0.005)
        else:
            pytest.fail("mock completion was not recorded")
        final = client.get("/api/calibration/status/leader").json()
        assert final["live"]["phase"] == "complete" and final["live"]["frozen"]


def test_joint_baseline_is_saved_separately_and_never_reloads_arm_guard(monkeypatch):
    from so101.hardware.calibration import CalibrationResult

    manager = get_manager()
    previous_limits = dict(manager.guard.limits)
    original_guard_loader = manager_module.SO101Follower.guard_from_saved_calibration
    monkeypatch.setattr(
        manager_module.SO101Follower,
        "guard_from_saved_calibration",
        staticmethod(lambda: pytest.fail("joint baseline reloaded full-arm safety limits")),
    )
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/calibration/start",
            json={
                "arm": "follower",
                "sample_seconds": 8,
                "joint": "gripper",
            },
        )
        assert response.status_code == 200, response.text
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            attempts = client.get("/api/calibration/history").json()["attempts"]
            if attempts and attempts[0]["result"] == "pass":
                break
            time.sleep(0.005)
        else:
            pytest.fail("joint baseline did not finish")
        attempt = attempts[0]
        assert attempt["joint"] == "gripper" and attempt["saved"] is True
        status = client.get("/api/calibration/status/follower").json()
        assert not status["has_saved"] and status["ranges"] == {}
        assert set(status["joint_baselines"]) == {"gripper"}
        assert status["joint_baselines"]["gripper"]["backend"] == "mock"
        assert "arm calibration unchanged" in status["message"]
        assert manager.guard.limits == previous_limits
        baseline = CalibrationResult.load_joint("follower", "gripper")
        assert baseline and set(baseline.joints) == {"gripper"}
        assert CalibrationResult.load("follower") is None
        assert not (manager.settings.data_dir / "calibration_leader.json").exists()
    monkeypatch.setattr(
        manager_module.SO101Follower,
        "guard_from_saved_calibration",
        staticmethod(original_guard_loader),
    )
    reset_manager()
    with TestClient(create_app()) as client:
        assert "gripper" in client.get("/api/calibration/status/follower").json()["joint_baselines"]
        assert client.get("/api/calibration/status/leader").json()["joint_baselines"] == {}


@pytest.mark.parametrize("joint", ["unknown", "", "../gripper", 6])
def test_invalid_joint_is_rejected_before_calibration_claim(joint, monkeypatch):
    manager = get_manager()
    monkeypatch.setattr(
        manager, "claim_calibration", lambda *a, **k: pytest.fail("claimed hardware")
    )
    with TestClient(create_app()) as client:
        response = client.post("/api/calibration/start", json={"arm": "follower", "joint": joint})
        assert response.status_code == 422


def test_worker_cannot_calibrate_a_different_joint_from_reservation(monkeypatch):
    from so101.services.robot_manager import ControlConflict

    manager = get_manager()
    manager.claim_calibration("follower", joint="gripper")
    monkeypatch.setattr(
        manager, "follower", lambda: pytest.fail("mismatched joint opened hardware")
    )
    with pytest.raises(ControlConflict, match="joint does not match"):
        manager.run_calibration("follower", 8, joint="wrist_roll")
    manager.finish_calibration("follower", "test cancelled", saved=False)

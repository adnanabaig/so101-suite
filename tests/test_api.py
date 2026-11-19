"""API smoke tests via FastAPI's TestClient (mock hardware)."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        from so101.services.robot_manager import get_manager

        get_manager().operator_heartbeat("test-operator")
        yield c
    # ensure any loop is stopped between tests
    from so101.services.robot_manager import get_manager

    get_manager().stop()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_status_idle(client):
    r = client.get("/api/robot/status")
    assert r.json()["mode"] == "idle"


@pytest.mark.parametrize(
    ("env", "backend", "mock"),
    [
        ({"SO101_MOCK": "1"}, "mock", True),
        ({"SO101_BACKEND": "mock"}, "mock", True),
        ({"SO101_BACKEND": "sim", "SO101_SIM_REALTIME": "0"}, "sim", False),
        ({"SO101_BACKEND": "real"}, "real", False),
        ({"SO101_MOCK": "0"}, "real", False),
    ],
)
def test_status_and_capabilities_name_the_backend(monkeypatch, env, backend, mock):
    """``mock`` is false under sim as well as real; ``backend`` tells them apart."""
    from so101.config import get_settings
    from so101.services.robot_manager import reset_manager

    monkeypatch.delenv("SO101_MOCK", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    reset_manager()
    try:
        client = TestClient(create_app())  # no lifespan: nothing connects or moves
        status = client.get("/api/robot/status").json()
        capabilities = client.get("/api/capabilities").json()
    finally:
        reset_manager()
    assert status["backend"] == capabilities["backend"] == backend
    assert status["mock"] is mock and capabilities["mock"] is mock
    assert status["follower_connected"] is False


def test_diagnostics_point_missing_ports_at_this_oss_names(monkeypatch, tmp_path):
    """Not the Linux-only "/dev/serial/by-id" advice everywhere: the per-OS hint,
    which on Linux leads with the udev names install.sh creates."""
    from so101.bringup.serial_ports import port_naming_hint
    from so101.config import get_settings
    from so101.services.robot_manager import reset_manager
    from so101.services.routers import operations

    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", "real")
    monkeypatch.setenv("SO101_FOLLOWER_PORT", str(tmp_path / "no-follower"))
    monkeypatch.setenv("SO101_LEADER_PORT", str(tmp_path / "no-leader"))
    get_settings.cache_clear()
    reset_manager()
    try:
        client = TestClient(create_app())  # no lifespan: nothing connects

        def serial_check() -> dict:
            checks = client.get("/api/diagnostics").json()["checks"]
            return next(check for check in checks if check["id"] == "serial.paths")

        here = serial_check()
        assert here["status"] == "warn"
        assert port_naming_hint() in here["remediation"]
        assert "Use stable /dev/serial/by-id paths" not in here["remediation"]
        monkeypatch.setattr(operations, "port_naming_hint", lambda: port_naming_hint("linux"))
        linux = serial_check()["remediation"]
        assert "the suite expects /dev/so101-follower and /dev/so101-leader" in linux
        settings_check = next(
            c for c in client.get("/api/diagnostics").json()["checks"] if c["id"] == "settings"
        )
        assert settings_check["status"] == "pass"
        assert "Settings files, later ones win" in settings_check["detail"]
    finally:
        reset_manager()


def test_the_backend_field_is_declared_in_the_openapi_schema(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("RobotStatus", "CapabilitiesResponse"):
        assert schemas[name]["properties"]["backend"]["enum"] == ["mock", "sim", "real"]
        assert "backend" in schemas[name]["required"]


def test_estop_flow(client):
    assert client.post("/api/robot/estop").json()["ok"]
    assert client.get("/api/robot/status").json()["estopped"] is True
    # C2: a reset names the operator and confirms.
    reset = {"operator_id": "test-operator", "confirm": True}
    assert client.post("/api/robot/estop/reset", json=reset).json()["ok"]
    assert client.get("/api/robot/status").json()["estopped"] is False


def test_teleop_start_stop(client):
    assert client.post(
        "/api/teleop/start",
        json={"fps": 60, "operator_id": "test-operator"},
    ).json()["ok"]
    time.sleep(0.2)
    assert client.get("/api/robot/status").json()["mode"] == "teleop"
    tele = client.get("/api/robot/status").json()
    assert tele["mode"] == "teleop"
    assert client.post("/api/teleop/stop").json()["ok"]


def _record_tiny_dataset(repo_id: str) -> None:
    from so101 import JOINT_NAMES
    from so101.data.recorder import EpisodeRecorder, RecordingConfig
    from so101.teleop.controller import TeleopStep

    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=10, camera_keys=[]))
    for index in range(3):
        joints = {f"{joint}.pos": float(index) for joint in JOINT_NAMES}
        recorder.add(TeleopStep(t=100.0 + index / 10, observation=joints, action=joints))
    recorder.save_episode()


def test_training_presets_and_submit(client):
    presets = client.get("/api/training/presets").json()
    assert {p["name"] for p in presets} >= {"act", "diffusion", "smolvla"}
    _record_tiny_dataset("test/x")
    job = client.post(
        "/api/training/jobs",
        json={"repo_id": "test/x", "policy": "act", "steps": 100, "batch_size": 8},
    ).json()
    assert job["status"] in {"pending", "running", "succeeded"}
    # let the mock trainer finish
    for _ in range(50):
        j = client.get(f"/api/training/jobs/{job['id']}").json()
        if j["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert j["status"] == "succeeded"


def test_datasets_empty(client):
    assert client.get("/api/datasets").json() == []


def test_inactive_opaque_camera_returns_503_without_open_stream(client):
    response = client.get("/api/video", params={"cam": "/dev/video0"})
    assert response.status_code == 503
    assert "no fresh frame" in response.json()["detail"]

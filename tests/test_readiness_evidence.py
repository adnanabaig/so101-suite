"""Readiness evidence is tied to the backend that produced it (M7).

Sim, mock and real share one data folder. Nothing a sim or mock run saves,
neither a preflight report nor a calibration, may count as evidence for the
physical arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from so101.bringup import preflight as pf
from so101.config import get_settings
from so101.services.app import create_app
from so101.services.robot_manager import reset_manager


def _use_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    if backend == "mock":
        monkeypatch.setenv("SO101_MOCK", "1")
        monkeypatch.delenv("SO101_BACKEND", raising=False)
    else:
        monkeypatch.delenv("SO101_MOCK", raising=False)
        monkeypatch.setenv("SO101_BACKEND", backend)
    monkeypatch.setenv("SO101_SIM_REALTIME", "0")
    get_settings.cache_clear()
    reset_manager()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from so101 import cli_bringup
    from so101.bringup import ownership
    from so101.bringup.ownership import ServerProbe
    from so101.hardware import lerobot_paths

    # No real server probe, no real LeRobot calibration directory, no real sleeps.
    monkeypatch.setattr(
        ownership,
        "probe_local_server",
        lambda settings: ServerProbe(running=False, url="http://127.0.0.1:1", detail="none"),
    )
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    monkeypatch.setattr(cli_bringup, "_sleep", lambda _: None)
    yield
    reset_manager()


def _client() -> TestClient:
    return TestClient(create_app())  # no lifespan: nothing connects or moves


# --- the latest preflight report is kept per backend ---------------------------------


def _run_preflight(monkeypatch: pytest.MonkeyPatch, backend: str) -> dict:
    _use_backend(monkeypatch, backend)
    response = _client().post("/api/preflight")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["backend"] == backend
    return body


def _latest(monkeypatch: pytest.MonkeyPatch, backend: str, **params: str):
    _use_backend(monkeypatch, backend)
    return _client().get("/api/preflight/latest", params=params)


def test_a_real_backend_server_never_serves_a_rehearsal(monkeypatch):
    pytest.importorskip("mujoco")
    sim = _run_preflight(monkeypatch, "sim")
    assert sim["rehearsal"] is True

    real = _latest(monkeypatch, "real")
    assert real.status_code == 404, real.json()
    assert "for the real backend" in real.json()["detail"]
    assert _latest(monkeypatch, "real", origin="api").status_code == 404

    mock = _run_preflight(monkeypatch, "mock")
    assert _latest(monkeypatch, "real").status_code == 404
    # Each backend keeps its own.
    assert _latest(monkeypatch, "sim").json()["created_at"] == sim["created_at"]
    assert _latest(monkeypatch, "mock").json()["created_at"] == mock["created_at"]


def test_each_backend_serves_its_own_latest_report(monkeypatch):
    real = _run_preflight(monkeypatch, "real")
    assert real["rehearsal"] is False
    mock = _run_preflight(monkeypatch, "mock")
    served = _latest(monkeypatch, "real").json()
    assert served["backend"] == "real" and served["created_at"] == real["created_at"]
    assert _latest(monkeypatch, "mock").json()["created_at"] == mock["created_at"]
    directory = pf.reports_dir(get_settings())
    assert {"latest-real.json", "latest-real-api.json", "latest-mock.json"} <= {
        path.name for path in directory.iterdir()
    }


def test_a_legacy_latest_report_counts_only_for_its_own_backend(monkeypatch):
    mock = _run_preflight(monkeypatch, "mock")
    directory = pf.reports_dir(get_settings())
    # What a run before per-backend copies left: one latest.json for every backend.
    for name in ("latest-mock.json", "latest-mock-api.json"):
        (directory / name).rename(directory / name.replace("-mock", ""))
    assert _latest(monkeypatch, "real").status_code == 404
    assert _latest(monkeypatch, "mock").json()["created_at"] == mock["created_at"]

    legacy = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
    assert legacy["backend"] == "mock"


def test_the_cli_says_where_its_report_went(monkeypatch):
    from typer.testing import CliRunner

    from so101.cli import app

    monkeypatch.setenv("COLUMNS", "300")
    result = CliRunner().invoke(app, ["preflight", "--arm", "follower"])
    saved = Path(get_settings().data_dir) / "preflight" / "latest-mock.json"
    assert str(saved) in result.output.replace("\n", "")
    assert saved.is_file()


# --- a calibration a mock run saved is no calibration of the real arm -------------------


def _save_mock_calibration(arm: str = "follower", **ranges: tuple[float, float]):
    """What a mock-backend calibration saves (device and backend "mock")."""

    from so101.hardware.calibration import JointCalibration, calibrate
    from so101.hardware.motors import MockBus

    result = calibrate(MockBus(name=arm), arm)
    assert result.device_id == "mock" and result.backend == "mock"
    for joint, (lo, hi) in ranges.items():
        result.joints[joint] = JointCalibration(min=lo, max=hi, offset=(lo + hi) / 2)
    result.save()
    return result


def _save_legacy_calibration(arm: str, joints: dict[str, tuple[float, float]]) -> None:
    """A calibration file from before device_id and backend were saved."""

    payload = {
        "arm": arm,
        "created_at": 1.0,
        "joints": {
            joint: {"min": lo, "max": hi, "offset": (lo + hi) / 2}
            for joint, (lo, hi) in joints.items()
        },
    }
    path = get_settings().data_dir / f"calibration_{arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _diagnostic(check_id: str) -> dict:
    checks = _client().get("/api/diagnostics").json()["checks"]
    return next(check for check in checks if check["id"] == check_id)


def test_real_diagnostics_count_a_mock_calibration_as_missing(monkeypatch):
    _save_mock_calibration("follower")
    _use_backend(monkeypatch, "real")
    check = _diagnostic("calibration.follower")
    assert check["status"] == "warn", check
    assert "made on the mock backend" in check["detail"]
    assert "Recalibrate the follower on the real arm" in check["remediation"]

    _use_backend(monkeypatch, "mock")  # a rehearsal may use it
    assert _diagnostic("calibration.follower")["status"] == "pass"


def test_the_calibration_status_route_hides_a_mock_calibration_on_real(monkeypatch):
    _save_mock_calibration("follower")
    _use_backend(monkeypatch, "real")
    status = _client().get("/api/calibration/status/follower").json()
    assert status["has_saved"] is False and status["ranges"] == {}
    assert "recalibrate on the real arm" in status["message"]
    from so101.hardware.calibration import MOCK_CALIBRATION_ON_REAL

    assert status["message"] == MOCK_CALIBRATION_ON_REAL

    _use_backend(monkeypatch, "mock")
    status = _client().get("/api/calibration/status/follower").json()
    assert status["has_saved"] is True and status["device_id"] == "mock"


def test_a_legacy_file_with_only_the_default_limits_is_a_mock_run(monkeypatch):
    from so101 import JOINT_NAMES
    from so101.hardware.safety import DEFAULT_LIMITS

    _save_legacy_calibration("follower", {joint: DEFAULT_LIMITS[joint] for joint in JOINT_NAMES})
    _use_backend(monkeypatch, "real")
    assert _client().get("/api/calibration/status/follower").json()["has_saved"] is False
    assert "made on the mock backend" in _diagnostic("calibration.follower")["detail"]

    measured = {joint: (lo + 3.0, hi - 4.5) for joint, (lo, hi) in DEFAULT_LIMITS.items()}
    _save_legacy_calibration("follower", measured)
    assert _client().get("/api/calibration/status/follower").json()["has_saved"] is True
    assert _diagnostic("calibration.follower")["status"] == "pass"


def test_the_real_follower_guard_ignores_a_mock_calibration(monkeypatch):
    from so101.hardware.follower import SO101Follower
    from so101.hardware.safety import DEFAULT_LIMITS

    _save_mock_calibration("follower", shoulder_pan=(-10.0, 10.0))
    _use_backend(monkeypatch, "real")
    assert SO101Follower.guard_from_saved_calibration().limits == DEFAULT_LIMITS
    _use_backend(monkeypatch, "mock")
    assert SO101Follower.guard_from_saved_calibration().limits["shoulder_pan"] == (-10.0, 10.0)


def test_real_preflight_counts_a_mock_suite_calibration_as_missing(monkeypatch):
    from so101.bringup.lerobot_surface import LeRobotSurface
    from so101.bringup.ownership import ServerProbe

    _save_mock_calibration("follower")
    _use_backend(monkeypatch, "real")
    report = pf.run_cli_preflight(
        get_settings(),
        arms=("follower",),
        server_probe=lambda s: ServerProbe(running=False, url="http://127.0.0.1:1", detail=""),
        lerobot_prober=LeRobotSurface,
        sleep=lambda _: None,
        write_report=False,
    )
    check = next(check for check in report.checks if check.key == "calibration.follower")
    assert "made on the mock backend" in check.message, check.message
    assert check.measured["suite_mock"] is True

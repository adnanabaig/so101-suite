"""Diagnostics, doctor and capabilities under ``SO101_BACKEND=sim``.

The sim backend drives a MuJoCo arm through ``so101.simulation.devices``. It
needs MuJoCo, not LeRobot, and no serial port, calibration or real camera.
Doctor output is sent back for triage, so it must say which backend made it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from so101.config import get_settings
from so101.services.app import create_app
from so101.services.robot_manager import reset_manager

pytest.importorskip("mujoco")


def _use_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", backend)
    monkeypatch.setenv("SO101_SIM_REALTIME", "0")
    get_settings.cache_clear()
    reset_manager()


@pytest.fixture
def sim(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _use_backend(monkeypatch, "sim")
    yield
    reset_manager()


@pytest.fixture
def no_sim_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import mujoco`` fails, so ``so101.simulation.devices`` cannot load."""

    monkeypatch.setitem(sys.modules, "mujoco", None)
    monkeypatch.delitem(sys.modules, "so101.simulation.devices", raising=False)


def _diagnostics() -> dict:
    client = TestClient(create_app())  # no lifespan: nothing connects or moves
    response = client.get("/api/diagnostics")
    assert response.status_code == 200, response.text
    return response.json()


def _check(payload: dict, check_id: str) -> dict:
    return next(check for check in payload["checks"] if check["id"] == check_id)


# --- M6: sim is its own backend, not "Real mode" ------------------------------------


def test_sim_diagnostics_check_the_simulated_arm_not_lerobot(sim):
    payload = _diagnostics()
    backend = _check(payload, "runtime.backend")
    assert backend["status"] == "pass", backend
    assert "SO101_BACKEND=sim" in backend["detail"] and "MuJoCo" in backend["detail"]
    assert "Real mode" not in backend["detail"]
    # SO101_MOCK=1 next to SO101_BACKEND=sim is a configuration error.
    assert not any("SO101_MOCK" in check["remediation"] for check in payload["checks"])


def test_sim_diagnostics_fail_when_the_simulated_arm_cannot_load(sim, no_sim_devices):
    payload = _diagnostics()
    backend = _check(payload, "runtime.backend")
    assert backend["status"] == "fail" and payload["overall"] == "fail"
    assert "cannot load" in backend["detail"] and "mujoco" in backend["detail"]
    assert ".[sim]" in backend["remediation"] and "SO101_MOCK" not in backend["remediation"]


def test_real_backend_advice_never_names_so101_mock(monkeypatch):
    _use_backend(monkeypatch, "real")
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    try:
        backend = _check(_diagnostics(), "runtime.backend")
    finally:
        reset_manager()
    assert backend["status"] == "fail"
    assert "SO101_MOCK" not in backend["remediation"]
    assert "SO101_BACKEND=mock" in backend["remediation"]


# --- L7: nothing used to check MuJoCo under sim -----------------------------------


@pytest.fixture
def broken_mujoco(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import mujoco`` returns a module without MuJoCo's API (a broken install).

    The sim devices then fail to import with AttributeError, not ImportError.
    """

    import types

    monkeypatch.setitem(sys.modules, "mujoco", types.ModuleType("mujoco"))
    monkeypatch.delitem(sys.modules, "so101.simulation.devices", raising=False)


def _preflight() -> dict:
    client = TestClient(create_app())
    response = client.post("/api/preflight", params={"save": "false"})
    assert response.status_code == 200, response.text
    return response.json()


def test_sim_preflight_fails_when_the_simulated_arm_cannot_load(sim, no_sim_devices):
    report = _preflight()
    environment = next(check for check in report["checks"] if check["key"] == "environment")
    assert environment["status"] == "fail", environment
    assert "simulated arm cannot load" in environment["message"]
    assert ".[sim]" in environment["remediation"]
    assert report["overall"] == "fail" and report["exit_code"] == 1


def test_sim_preflight_reports_the_loaded_simulated_arm(sim):
    report = _preflight()
    environment = next(check for check in report["checks"] if check["key"] == "environment")
    assert environment["status"] == "pass", environment
    assert environment["measured"]["sim_devices"] == "loaded"


def test_cli_preflight_under_sim_exits_1_when_the_simulated_arm_cannot_load(sim, no_sim_devices):
    from so101.bringup import preflight as pf
    from so101.bringup.lerobot_surface import LeRobotSurface
    from so101.bringup.ownership import ServerProbe

    report = pf.run_cli_preflight(
        get_settings(),
        server_probe=lambda s: ServerProbe(running=False, url="http://127.0.0.1:1", detail=""),
        lerobot_prober=LeRobotSurface,
        sleep=lambda _: None,
        write_report=False,
    )
    assert report.exit_code == 1 and report.overall == "fail"


def test_a_broken_mujoco_leaves_the_server_up_and_fails_diagnostics(sim, broken_mujoco):
    from so101.hardware.motors import SimBusUnavailable, make_bus

    assert isinstance(make_bus("unused", name="follower"), SimBusUnavailable)
    backend = _check(_diagnostics(), "runtime.backend")
    assert backend["status"] == "fail" and "AttributeError" in backend["detail"]


# --- L8: sim diagnostics run no real-hardware checks -------------------------------


def test_sim_diagnostics_skip_serial_ports_and_calibration(sim, monkeypatch, tmp_path):
    # Configured ports that do not exist: a real-backend run warns about them.
    monkeypatch.setenv("SO101_FOLLOWER_PORT", str(tmp_path / "no-follower"))
    monkeypatch.setenv("SO101_LEADER_PORT", str(tmp_path / "no-leader"))
    get_settings.cache_clear()
    reset_manager()
    payload = _diagnostics()
    ids = [check["id"] for check in payload["checks"]]
    assert "serial.paths" not in ids
    text = " ".join(f"{c['detail']} {c['remediation']}" for c in payload["checks"])
    assert "/dev/tty" not in text and "no-follower" not in text and "no-leader" not in text
    for arm in ("follower", "leader"):
        calibration = _check(payload, f"calibration.{arm}")
        assert calibration["status"] == "pass", calibration
        assert calibration["detail"].startswith("Not applicable on the sim backend")
        assert "ecalibrate" not in calibration["remediation"]
    cameras = _check(payload, "cameras")
    assert "SO101_SIM_CAMERAS" in cameras["remediation"]
    assert "SO101_CAMERAS " not in cameras["remediation"]
    buses = [port for port in payload["ports"] if port["id"] in {"follower", "leader"}]
    assert [port["type"] for port in buses] == ["sim", "sim"]
    assert all(port["path"].startswith("sim:") for port in buses)
    assert all("no serial port" in port["detail"] for port in buses)
    # Nothing on a healthy sim install is worse than the dark cameras' warning.
    assert payload["overall"] == "warn"


# --- L10: doctor and info say which backend they describe ---------------------------


def _doctor_json(monkeypatch: pytest.MonkeyPatch) -> tuple[int, dict]:
    import json

    from typer.testing import CliRunner

    from so101 import platform_info
    from so101.cli import app

    # A cheap platform report: no torch import, no MuJoCo render child.
    report = platform_info.collect(
        platform_info.Probes(torch=lambda: None, dist_version=lambda name: None)
    )
    monkeypatch.setattr(platform_info, "collect", lambda: report)
    result = CliRunner().invoke(app, ["doctor", "--json"])
    return result.exit_code, json.loads(result.stdout)


def test_doctor_json_names_the_sim_backend_and_its_settings(sim, monkeypatch):
    monkeypatch.setenv("SO101_SIM_TASK", "reach")
    monkeypatch.setenv("SO101_SIM_SEED", "7")
    get_settings.cache_clear()
    reset_manager()
    code, payload = _doctor_json(monkeypatch)
    assert payload["backend"] == "sim"
    assert payload["sim"] == {
        "task": "reach",
        "seed": 7,
        "cameras": ["front", "wrist_cam"],
        "image_size": "320x240",
        "realtime": False,
    }
    backend = next(check for check in payload["checks"] if check["id"] == "runtime.backend")
    assert backend["status"] == "pass"
    assert code == (1 if payload["overall"] == "fail" else 0)


def test_doctor_json_names_the_mock_backend(monkeypatch):
    _, payload = _doctor_json(monkeypatch)
    assert payload["backend"] == "mock" and payload["sim"] is None


def _info(monkeypatch: pytest.MonkeyPatch) -> str:
    from typer.testing import CliRunner

    from so101.cli import app

    monkeypatch.setenv("COLUMNS", "300")
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    return result.output


def _row(output: str, key: str) -> str | None:
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.strip("│ ").split("│")]
        if len(cells) == 2 and cells[0] == key:
            return cells[1]
    return None


def test_info_under_sim_shows_the_sim_settings_not_serial_ports(sim, monkeypatch):
    monkeypatch.setenv("SO101_SIM_TASK", "pick_place")
    get_settings.cache_clear()
    output = _info(monkeypatch)
    assert _row(output, "backend") == "sim"
    assert _row(output, "sim task") == "pick_place"
    assert _row(output, "sim cameras") == "front, wrist_cam"
    assert _row(output, "mock mode") is None
    assert _row(output, "follower port") is None and "/dev/" not in output


def test_info_on_the_real_backend_shows_the_ports(monkeypatch):
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", "real")
    monkeypatch.setenv("SO101_FOLLOWER_PORT", "/dev/so101-follower")
    get_settings.cache_clear()
    output = _info(monkeypatch)
    assert _row(output, "backend") == "real"
    assert _row(output, "follower port") == "/dev/so101-follower"
    assert _row(output, "sim task") is None


# --- L9: capabilities under sim rest on the simulated bus ----------------------------


def _capabilities() -> dict[str, dict]:
    payload = TestClient(create_app()).get("/api/capabilities").json()
    assert payload["backend"] == "sim"
    return {item["id"]: item for item in payload["capabilities"]}


def test_sim_capabilities_offer_control_teleop_and_training(sim, monkeypatch):
    from so101.services.routers import capabilities

    monkeypatch.setattr(capabilities, "lerobot_train_available", lambda: False)
    items = _capabilities()
    for capability in ("control", "teleop"):
        assert items[capability]["available"] is True, items[capability]
    assert items["control"]["detail"].startswith("Simulated arm (SO101_BACKEND=sim)")
    assert "scripted expert" in items["teleop"]["detail"]
    assert items["training"]["available"] is True
    assert "bc_baseline" in items["training"]["detail"]
    # Every capability is ready on a healthy sim install (the Dashboard's 9/9).
    assert all(item["available"] for item in items.values()), items


def test_sim_capabilities_without_the_simulated_arm(sim, no_sim_devices):
    items = _capabilities()
    for capability in ("control", "teleop"):
        assert items[capability]["available"] is False
        assert "cannot load" in items[capability]["detail"]

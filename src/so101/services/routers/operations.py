"""Operational diagnostics and persistent audit events."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from ...bringup.serial_ports import permission_remediation, port_naming_hint
from ...config import describe_sources, settings_sources
from ...hardware.backend import (
    SIM_EXTRA_REMEDIATION,
    Backend,
    current_backend,
    sim_devices_problem,
)
from ...hardware.calibration import CalibrationResult, counts_as_missing
from ...hardware.cameras import configured_streams
from ..robot_manager import get_manager
from ..schemas import DiagnosticCheck, DiagnosticsResponse
from ..store import get_store

router = APIRouter(tags=["operations"])


def _check(
    check_id: str,
    label: str,
    status: str,
    detail: str,
    remediation: str = "",
) -> DiagnosticCheck:
    return DiagnosticCheck(
        id=check_id,
        label=label,
        status=status,
        detail=detail,
        remediation=remediation,
    )


def _backend_check(backend: Backend) -> DiagnosticCheck:
    """Whether this process can drive the backend SO101_BACKEND selects.

    Never suggests SO101_MOCK: with SO101_BACKEND=real or sim it is a
    configuration error (config.py), so the advice names SO101_BACKEND instead.
    """
    if backend == "mock":
        return _check(
            "runtime.backend", "Robot backend", "pass", "Explicit deterministic mock backend."
        )
    if backend == "sim":
        problem = sim_devices_problem()
        if problem is not None:
            return _check(
                "runtime.backend",
                "Robot backend",
                "fail",
                f"SO101_BACKEND=sim is selected but the simulated arm cannot load: {problem}.",
                SIM_EXTRA_REMEDIATION,
            )
        mujoco = sys.modules.get("mujoco")
        version = getattr(mujoco, "__version__", "")
        return _check(
            "runtime.backend",
            "Robot backend",
            "pass",
            f"MuJoCo simulation (SO101_BACKEND=sim): MuJoCo {version or '(version unknown)'} "
            "and the simulated devices load. Commands move a simulated arm, not the physical "
            "one; nothing here is evidence about a real arm.",
        )
    if importlib.util.find_spec("lerobot") is not None:
        return _check(
            "runtime.backend", "Robot backend", "pass", "Pinned LeRobot adapter is importable."
        )
    return _check(
        "runtime.backend",
        "Robot backend",
        "fail",
        "Real mode is selected but LeRobot is not installed.",
        "Install `.[robot]`, or rehearse without an arm: SO101_BACKEND=mock, or "
        "SO101_BACKEND=sim for the MuJoCo simulation.",
    )


def _calibration_check(settings: Any, backend: Backend, arm: str) -> DiagnosticCheck:
    """The saved calibration of ``arm``, as far as this backend uses it."""

    if backend == "sim":
        # The simulated arm has no encoders; the calibration route and
        # `so101 calibrate` refuse the sim backend (SIM_CALIBRATION_REFUSED).
        return _check(
            f"calibration.{arm}",
            f"{arm.title()} calibration",
            "pass",
            "Not applicable on the sim backend: the simulated arm has no encoders, so it "
            "is never calibrated. A saved real-arm calibration is left as it is.",
        )
    calibration_error = ""
    try:
        calibration = CalibrationResult.load(arm)
    except Exception as exc:
        calibration = None
        calibration_error = str(exc)
    if counts_as_missing(calibration, backend):
        return _check(
            f"calibration.{arm}",
            f"{arm.title()} calibration",
            "warn",
            f"No calibration of the real {arm}: the saved one was made on the mock backend "
            "(device mock), so it says nothing about this arm.",
            f"Recalibrate the {arm} on the real arm before real motion (the Calibration page, "
            f"or `so101 calibrate {arm}` with the server stopped).",
        )
    calibrated = calibration is not None
    expected_device = settings.follower_id if arm == "follower" else settings.leader_id
    calibration_device = calibration.device_id if calibration is not None else ""
    identity_mismatch = bool(
        calibration and calibration_device and calibration_device not in {"mock", expected_device}
    )
    if identity_mismatch:
        status = "fail"
        detail = (
            f"Calibration belongs to {calibration_device}; attached config expects "
            f"{expected_device}."
        )
    elif calibration_error:
        status = "fail"
        detail = f"Saved calibration is invalid: {calibration_error}"
    else:
        status = "pass" if calibrated or settings.mock else "warn"
        detail = (
            "Mock defaults available."
            if settings.mock and not calibrated
            else (
                f"Saved calibration for {calibration.device_id or 'legacy device'}."
                if calibrated and calibration is not None
                else "No saved calibration found."
            )
        )
    return _check(
        f"calibration.{arm}",
        f"{arm.title()} calibration",
        status,
        detail,
        f"Recalibrate the {arm} before real motion." if status in {"warn", "fail"} else "",
    )


def _bus_port(settings: Any, backend: Backend, arm: str, *, connected: bool) -> dict[str, Any]:
    """One motor bus for the ports list. Under sim it names the simulation, not a port."""

    label = f"{arm.title()} motor bus"
    if backend == "sim":
        return {
            "id": arm,
            "path": f"sim:{arm}",
            "label": f"{label} (simulated)",
            "type": "sim",
            "connected": connected,
            "detail": (
                f"MuJoCo {settings.sim_task} scene, seed {settings.sim_seed}"
                + ("; the scripted expert" if arm == "leader" else "")
                + "; no serial port is opened"
            ),
        }
    device = settings.follower_id if arm == "follower" else settings.leader_id
    return {
        "id": arm,
        "path": settings.follower_port if arm == "follower" else settings.leader_port,
        "label": label,
        "type": "serial",
        "connected": connected,
        "detail": "mock backend" if backend == "mock" else f"configured device {device}",
    }


@router.get("/api/diagnostics", response_model=DiagnosticsResponse)
def diagnostics() -> DiagnosticsResponse:
    manager = get_manager()
    settings = manager.settings
    robot = manager.status()
    control = manager.control_state()
    checks: list[DiagnosticCheck] = []

    if robot["estopped"]:
        checks.append(
            _check(
                "safety",
                "Safety state",
                "fail",
                "Software E-stop is latched.",
                "Inspect the workspace, then explicitly reset E-stop.",
            )
        )
    elif manager.fault:
        checks.append(
            _check(
                "safety",
                "Safety state",
                "fail",
                manager.fault,
                "Correct the device fault before clearing and re-arming.",
            )
        )
    else:
        checks.append(
            _check(
                "safety",
                "Safety state",
                "pass",
                f"Runtime is {control['safety_state']} and no fault is latched.",
            )
        )

    persistence_healthy = get_store().healthy()
    checks.append(
        _check(
            "persistence",
            "Audit persistence",
            "pass" if persistence_healthy else "fail",
            f"SQLite WAL store: {get_store().path}",
            "Check data-directory permissions and free disk space."
            if not persistence_healthy
            else "",
        )
    )
    backend = current_backend(settings)
    checks.append(_backend_check(backend))
    free_gb = shutil.disk_usage(settings.data_dir).free / (1024**3)
    checks.append(
        _check(
            "storage",
            "Storage capacity",
            "pass" if free_gb >= 2 else "warn",
            f"{free_gb:.1f} GiB free in {settings.data_dir}",
            "Free space before collecting video episodes." if free_gb < 2 else "",
        )
    )
    sources = settings_sources(settings)
    checks.append(
        _check(
            "settings",
            "Settings and data folders",
            "warn" if sources["warning"] else "pass",
            describe_sources(sources),
            sources["warning"] or "",
        )
    )
    checks.extend(_calibration_check(settings, backend, arm) for arm in ("follower", "leader"))
    streams = configured_streams(settings)  # credentials stripped
    live_cameras = [camera for camera in streams if manager.get_frame(camera) is not None]
    shared_cameras = []
    if backend == "real":
        from ..mcap_sessions import relay_status

        shared_cameras = [name for name, state in relay_status().items() if state["live"]]
    camera_setting = "SO101_SIM_CAMERAS" if backend == "sim" else "SO101_CAMERAS"
    checks.append(
        _check(
            "cameras",
            "Camera readiness",
            "pass" if live_cameras or shared_cameras else "warn",
            (
                f"Live frames: {', '.join(live_cameras)}"
                if live_cameras
                else f"Live shared camera feeds: {', '.join(shared_cameras)}"
                if shared_cameras
                else f"Configured but not live: {', '.join(streams) or 'none'}"
            ),
            f"Start a camera-using mode and verify fresh frames; configure {camera_setting} "
            "if empty."
            if not live_cameras and not shared_cameras
            else "",
        )
    )
    if backend == "real":  # mock and sim open no serial port
        missing_ports = [
            path
            for path in (settings.follower_port, settings.leader_port)
            if not Path(path).exists()
        ]
        checks.append(
            _check(
                "serial.paths",
                "Serial device paths",
                "warn" if missing_ports else "pass",
                (
                    f"Missing configured paths: {', '.join(missing_ports)}"
                    if missing_ports
                    else "Follower and leader paths exist."
                ),
                (
                    "Plug in and power both arms, check SO101_FOLLOWER_PORT and "
                    f"SO101_LEADER_PORT. {port_naming_hint()} "
                    f"{permission_remediation(missing_ports[0])}"
                )
                if missing_ports
                else "",
            )
        )

    priorities = {"pass": 0, "warn": 1, "fail": 2}
    overall = max(checks, key=lambda item: priorities.get(item.status, 1)).status
    ports: list[dict[str, Any] | str] = [
        _bus_port(settings, backend, arm, connected=robot[f"{arm}_connected"])
        for arm in ("follower", "leader")
    ]
    ports.extend(
        {
            "id": f"camera:{camera}",
            "path": camera,
            "label": f"Camera {camera}",
            "type": "camera",
            "connected": manager.get_frame(camera) is not None,
            "detail": (
                "live frame received"
                if manager.get_frame(camera) is not None
                else "configured; no live frame received"
            ),
        }
        for camera in configured_streams(settings)
    )
    return DiagnosticsResponse(
        generated_at=datetime.now(timezone.utc),
        overall=overall,
        checks=checks,
        ports=ports,
        events=get_store().list_events(limit=20),
    )


@router.get("/api/events")
def events(
    limit: int = Query(default=100, ge=1, le=500),
    after_id: int | None = Query(default=None, ge=0),
) -> dict:
    return {"events": get_store().list_events(limit=limit, after_id=after_id)}

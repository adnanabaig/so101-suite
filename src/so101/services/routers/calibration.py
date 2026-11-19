"""Calibration: kick off a background sweep and poll its progress.

The sweep's bus access goes through ``RobotManager.run_calibration``, under the
manager's bus lock and with its three-state torque truth: the manager stays the
only owner of the motor buses (docs/control-plane-safety.md).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from ... import JOINT_NAMES
from ...hardware.backend import current_backend
from ...hardware.calibration import (
    MOCK_CALIBRATION_ON_REAL,
    SIM_CALIBRATION_REFUSED,
    CalibrationResult,
    affected_motors,
    counts_as_missing,
)
from ...hardware.lerobot_paths import lerobot_calibration_file
from ..calibration_pose import cached_pose, read_calibration_pose, remember_pose
from ..robot_manager import (
    CalibrationTorqueOffUnconfirmed,
    ControlBusy,
    ControlConflict,
    ControlRejected,
    get_manager,
)
from ..schemas import (
    Ack,
    CalibrateRequest,
    CalibrationAttempt,
    CalibrationHistory,
    CalibrationLogEntry,
    CalibrationStatus,
    RequestModel,
)
from ..store import get_store

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


class PoseReadRequest(RequestModel):
    confirm_supported_torque_off: Literal[True]


@router.get("/pose/{arm}")
def last_pose(arm: str) -> dict[str, Any] | None:
    if arm not in {"follower", "leader"}:
        raise HTTPException(404, "arm must be follower or leader")
    return cached_pose(arm)


@router.post("/pose/{arm}")
def pose(arm: str, req: PoseReadRequest) -> dict[str, Any]:
    if arm not in {"follower", "leader"}:
        raise HTTPException(404, "arm must be follower or leader")
    manager = get_manager()
    backend = current_backend(manager.settings)
    if backend != "real":
        return {
            "arm": arm,
            "backend": backend,
            "captured_at": time.time(),
            "calibration_sha256": "mock-preview",
            "complete": True,
            "observation": {f"{joint}.pos": 0.0 for joint in JOINT_NAMES},
            "joints": {
                joint: {
                    "motor_id": index,
                    "ticks": None,
                    "range_min": None,
                    "range_max": None,
                    "issues": ["Preview only; no physical encoder read"],
                }
                for index, joint in enumerate(JOINT_NAMES, 1)
            },
        }
    path = lerobot_calibration_file(arm, getattr(manager.settings, f"{arm}_id"))
    try:
        contents = path.read_bytes()
        calibration = json.loads(contents)
        target: Literal["follower", "leader"] = "follower" if arm == "follower" else "leader"
        result = manager.run_idle_bus_diagnostic(
            target, lambda bus: read_calibration_pose(bus, calibration)
        )
        if path.read_bytes() != contents:
            raise ValueError("Installed calibration changed during the read; retry")
    except ControlBusy as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ControlConflict, ControlRejected, ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    payload = {
        **result,
        "arm": arm,
        "backend": backend,
        "calibration_sha256": hashlib.sha256(contents).hexdigest(),
        "complete": len(result["observation"]) == len(JOINT_NAMES),
    }
    remember_pose(arm, payload)
    return payload


def calibration_history(
    events: list[dict[str, Any]], active_id: str | None, limit: int
) -> list[CalibrationAttempt]:
    attempts: dict[str, CalibrationAttempt] = {}
    pending: dict[str, str] = {}
    for event in events:
        detail = event.get("detail") or {}
        message = str(event["message"])
        match = re.match(r"^(follower|leader) calibration\b", message)
        arm = detail.get("arm") or (match.group(1) if match else "unknown")
        started = event["type"] == "calibration.started"
        key = detail.get("attempt_id") or (
            f"event-{event['id']}" if started else pending.get(arm, f"event-{event['id']}")
        )
        if key not in attempts:
            attempts[key] = CalibrationAttempt(
                id=key,
                event_id=event["id"],
                arm=arm,
                joint=detail.get("joint"),
                backend=detail.get("backend"),
                device_id=detail.get("device_id"),
                sample_seconds=detail.get("sample_seconds"),
                started_at=detail.get("started_at") or (event["created_at"] if started else None),
                result="running" if key == active_id else "incomplete",
            )
        attempt = attempts[key]
        attempt.event_id = event["id"]
        attempt.log.append(
            CalibrationLogEntry(
                timestamp=event["created_at"], level=event["level"], message=message
            )
        )
        if started:
            pending[arm] = key
            continue
        passed = event["type"] == "calibration.saved"
        attempt.result = "pass" if passed else "fail"
        attempt.finished_at = event["created_at"]
        attempt.saved = detail.get("saved", True if passed else None)
        attempt.error = detail.get("error") or (None if passed else message)
        attempt.affected_motors = detail.get("affected_motors") or affected_motors(attempt.error)
        attempt.diagnostics = detail.get("diagnostics") or {}
        if pending.get(arm) == key:
            pending.pop(arm)
    return sorted(attempts.values(), key=lambda attempt: attempt.event_id, reverse=True)[:limit]


def _run_calibration(arm: str, sample_seconds: float, joint: str | None = None) -> None:
    """The worker thread: every outcome, including an unconfirmed torque-off,
    ends as the job's status (``calibration.failed`` with the reason)."""

    mgr = get_manager()
    error: str | None = None
    saved = False
    try:
        mgr.run_calibration(arm, sample_seconds, **({"joint": joint} if joint else {}))
        saved = True
    except CalibrationTorqueOffUnconfirmed as exc:
        error, saved = str(exc), exc.saved
    except Exception as exc:  # surfaced through calibration status + diagnostics events
        error = str(exc) or type(exc).__name__
    finally:
        mgr.finish_calibration(arm, error, saved=saved)


@router.post("/start", response_model=Ack)
def start(req: CalibrateRequest) -> Ack:
    mgr = get_manager()
    if current_backend(mgr.settings) == "sim":
        raise HTTPException(409, SIM_CALIBRATION_REFUSED)
    try:
        mgr.claim_calibration(req.arm, sample_seconds=req.sample_seconds, joint=req.joint)
    except (ControlConflict, ControlRejected) as exc:
        raise HTTPException(409, str(exc)) from exc
    worker = threading.Thread(
        target=_run_calibration, args=(req.arm, req.sample_seconds, req.joint), daemon=True
    )
    try:
        mgr.set_calibration_worker(worker)
        worker.start()
    except Exception as exc:
        mgr.finish_calibration(req.arm, str(exc))
        raise HTTPException(500, f"could not start calibration worker: {exc}") from exc
    return Ack(ok=True, detail=f"calibrating {req.arm}")


@router.get("/history", response_model=CalibrationHistory)
def history(limit: int = Query(default=100, ge=1, le=200)) -> CalibrationHistory:
    """Stored metadata only: never opens a motor bus or changes calibration."""
    manager = get_manager()
    try:
        events = get_store().calibration_events(limit=2 * limit + 1)
    except Exception as exc:
        raise HTTPException(503, "Calibration history could not be read") from exc
    return CalibrationHistory(
        attempts=calibration_history(events, manager.calibration_attempt_id, limit), limit=limit
    )


@router.get("/status/{arm}", response_model=CalibrationStatus)
def status(arm: str) -> CalibrationStatus:
    if arm not in {"follower", "leader"}:
        raise HTTPException(404, "arm must be follower or leader")
    mgr = get_manager()
    try:
        saved_result = CalibrationResult.load(arm)
        load_error = ""
    except Exception as exc:
        saved_result = None
        load_error = f"invalid saved calibration: {exc}"
    if counts_as_missing(saved_result, current_backend(mgr.settings)):
        # A mock run's file on the real backend: no calibration of this arm.
        saved_result = None
        load_error = MOCK_CALIBRATION_ON_REAL
    baselines = {}
    for joint in JOINT_NAMES:
        try:
            baseline = CalibrationResult.load_joint(arm, joint)
            if baseline is None or counts_as_missing(baseline, current_backend(mgr.settings)):
                continue
            if baseline.device_id not in {"mock", getattr(mgr.settings, f"{arm}_id")}:
                continue
            baselines[joint] = {
                "created_at": baseline.created_at,
                "backend": baseline.backend,
                "device_id": baseline.device_id,
                "diagnostics": baseline.diagnostics,
            }
        except Exception as exc:
            baselines[joint] = {"error": f"baseline could not be read: {exc}"}
    return CalibrationStatus(
        joint_baselines=baselines,
        live=mgr.calibration_live_status(arm),
        arm=arm,
        in_progress=mgr.cal_in_progress and mgr.cal_arm == arm,
        fraction=mgr.cal_fraction if mgr.cal_arm == arm else 0.0,
        message=(mgr.cal_message if mgr.cal_arm == arm else "") or load_error,
        has_saved=saved_result is not None,
        device_id=saved_result.device_id if saved_result else "",
        created_at=saved_result.created_at if saved_result else None,
        ranges={
            joint: [calibration.min, calibration.max]
            for joint, calibration in (saved_result.joints.items() if saved_result else [])
        },
    )

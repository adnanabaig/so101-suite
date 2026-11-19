"""Exclusive manual-control lease, arming, heartbeat, and command API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..robot_manager import ControlBusy, ControlConflict, ControlRejected, StopKind, get_manager
from ..schemas import (
    Ack,
    ArmRequest,
    ControlCommandRequest,
    ControlObservation,
    ControlState,
    DisarmRequest,
    LeaseInfo,
    LeaseRequest,
)

router = APIRouter(prefix="/api/control", tags=["control"])


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ControlBusy):
        # safety-B9: the motor bus did not come free in time; nothing was sent.
        return HTTPException(503, str(exc))
    status = 409 if isinstance(exc, (ControlConflict, ControlRejected)) else 400
    return HTTPException(status, str(exc))


@router.get("/state", response_model=ControlState)
def state() -> ControlState:
    return ControlState(**get_manager().control_state())


@router.post("/lease", response_model=LeaseInfo, status_code=201)
def acquire(req: LeaseRequest) -> LeaseInfo:
    try:
        lease = get_manager().acquire_lease(req.source, req.ttl_seconds)
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc
    return LeaseInfo(**lease)


@router.post("/lease/{lease_id}/heartbeat", response_model=LeaseInfo)
def heartbeat(lease_id: str) -> LeaseInfo:
    try:
        lease = get_manager().heartbeat_lease(lease_id)
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc
    return LeaseInfo(**lease)


@router.delete("/lease/{lease_id}", response_model=Ack)
def release(lease_id: str) -> Ack:
    try:
        get_manager().release_lease(lease_id)
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc
    return Ack(ok=True, detail="control lease released; follower disarmed")


@router.post("/arm", response_model=ControlState)
def arm(req: ArmRequest) -> ControlState:
    try:
        return ControlState(**get_manager().arm_control(req.lease_id))
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc


@router.post("/observe", response_model=ControlObservation)
def observe(req: ArmRequest) -> ControlObservation:
    try:
        return ControlObservation(**get_manager().observe_with_lease(req.lease_id))
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc


@router.post("/disarm", response_model=ControlState)
def disarm(req: DisarmRequest | None = None) -> ControlState:
    """Owner-scoped COLLAPSE ("done, power down"; design doc §5).

    Disarms, then lowers the arm to the rest pose at the capped speed and
    turns the motors off; it answers once the descent has started. A missing,
    wrong or expired lease is a 409 that changes nothing (C5). The global stops
    are ``POST /api/robot/halt``, ``/stop``, ``/collapse`` and the E-stop.
    """

    lease_id = req.lease_id if req is not None else None
    if not lease_id:
        raise HTTPException(409, "valid exclusive control lease required")
    manager = get_manager()
    try:
        manager.request_stop(
            StopKind.COLLAPSE, reason="operator disarm", lease_id=lease_id, strict=True
        )
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc
    return ControlState(**manager.control_state())


@router.post("/command")
def command(req: ControlCommandRequest) -> dict:
    try:
        commanded = get_manager().send_external_command(
            lease_id=req.lease_id,
            source=req.source,
            sequence=req.sequence,
            sent_at=req.sent_at,
            ttl_ms=req.ttl_ms,
            deadman=req.deadman,
            targets=req.targets,
        )
    except (ControlConflict, ControlRejected) as exc:
        raise _http_error(exc) from exc
    detail = ", ".join(f"{joint}={value:.2f}" for joint, value in commanded.items())
    return {
        "ok": True,
        "accepted": True,
        "detail": f"command accepted: {detail}",
        "targets": commanded,
    }


@router.post("/arm-command")
def arm_command(req: ControlCommandRequest) -> dict:
    manager = get_manager()
    try:
        # Fail fast on a frame that is already stale, then hand the client's
        # own timestamp to the command, so time spent arming counts against
        # its ttl_ms (safety-B7, docs/audit/v0.1/safety.md).
        manager.validate_command_freshness(req.sent_at, req.ttl_ms)
        manager.arm_control(req.lease_id)
        commanded = manager.send_external_command(
            lease_id=req.lease_id,
            source=req.source,
            sequence=req.sequence,
            sent_at=req.sent_at,
            ttl_ms=req.ttl_ms,
            deadman=req.deadman,
            targets=req.targets,
        )
    except (ControlConflict, ControlRejected) as exc:
        # Error-path cleanup is owner-scoped and never raises, so it cannot
        # mask the original error or stop another owner (safety-B6, C5).
        manager.request_stop(
            StopKind.HALT, reason="atomic first command failed", lease_id=req.lease_id
        )
        raise _http_error(exc) from exc
    detail = ", ".join(f"{joint}={value:.2f}" for joint, value in commanded.items())
    return {
        "ok": True,
        "accepted": True,
        "detail": f"armed and accepted first command: {detail}",
        "targets": commanded,
    }

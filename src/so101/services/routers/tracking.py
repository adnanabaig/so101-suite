"""Confidence-gated hand/camera intent retargeting preview."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ... import JOINT_NAMES
from ..robot_manager import (
    ControlBusy,
    ControlConflict,
    ControlRejected,
    StopKind,
    get_manager,
    lease_tag,
)
from ..schemas import TrackingPreviewRequest, TrackingPreviewResponse
from ..store import get_store

router = APIRouter(prefix="/api/tracking", tags=["tracking"])
MIN_TRACKING_CONFIDENCE = 0.72


class _AuditThrottle:
    """At most one audit event per key per window (sim-B20, docs/audit/v0.1/sim.md).

    Entries sit in a bounded LRU and expire after a TTL, so a burst of new keys
    only evicts the least recently audited ones; it never resets every other
    caller's throttle.
    """

    def __init__(
        self,
        *,
        window_s: float = 5.0,
        ttl_s: float = 60.0,
        capacity: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window_s = window_s
        self.ttl_s = ttl_s
        self.capacity = capacity
        self._clock = clock
        self._lock = threading.Lock()
        self._last_audit: OrderedDict[str, float] = OrderedDict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._last_audit)

    def should_audit(self, key: str, *, force: bool = False) -> bool:
        now = self._clock()
        with self._lock:
            # Oldest first: every audit moves its key to the end.
            while self._last_audit:
                oldest_key, audited_at = next(iter(self._last_audit.items()))
                if now - audited_at < self.ttl_s:
                    break
                del self._last_audit[oldest_key]
            previous = self._last_audit.get(key)
            if not force and previous is not None and now - previous < self.window_s:
                return False
            self._last_audit[key] = now
            self._last_audit.move_to_end(key)
            while len(self._last_audit) > self.capacity:
                self._last_audit.popitem(last=False)
            return True


_AUDIT = _AuditThrottle()


def _audit_key(req: TrackingPreviewRequest, client_host: str | None) -> str:
    """The live lease when the frame names it, otherwise the caller's address.

    Never a client-chosen string, so rotating ``source`` or made-up lease ids
    cannot mint fresh keys (sim-B20).
    """

    if req.lease_id is not None and get_manager().holds_lease(req.lease_id):
        return f"lease:{lease_tag(req.lease_id)}"
    return f"client:{client_host or 'unknown'}"


def _client_host(request: Request) -> str | None:
    return request.client.host if request.client else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _normalized(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else min(max(number, -1.0), 1.0)


def _merged_intent(req: TrackingPreviewRequest) -> dict[str, Any]:
    intent = dict(req.intent)
    if isinstance(req.wrist, dict):
        intent = {**req.wrist, **intent}
    elif isinstance(req.wrist, list):
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        for key, value in zip(axes, req.wrist, strict=False):
            intent.setdefault(key, value)
    if req.pinch is not None:
        intent.setdefault("pinch", req.pinch)
    return intent


def _holds_elbow(req: TrackingPreviewRequest, targets: dict[str, float]) -> bool:
    """A null z is "no depth" (monocular tracking): the elbow holds (C9, sim-B5)."""

    intent = _merged_intent(req)
    return "z" in intent and intent["z"] is None and "elbow_flex" not in targets


def _retarget(req: TrackingPreviewRequest) -> dict[str, float]:
    intent = _merged_intent(req)

    targets: dict[str, float] = {}
    for joint in JOINT_NAMES:
        direct_value = _number(intent.get(joint))
        if direct_value is not None:
            targets[joint] = direct_value

    mappings = {
        "x": ("shoulder_pan", 90.0),
        "normalized_x": ("shoulder_pan", 90.0),
        "y": ("shoulder_lift", -75.0),
        "normalized_y": ("shoulder_lift", -75.0),
        "z": ("elbow_flex", 70.0),
        "pitch": ("wrist_flex", 80.0),
        "roll": ("wrist_roll", 140.0),
        "yaw": ("wrist_roll", 140.0),
    }
    for key, (joint, scale) in mappings.items():
        normalized_value = _normalized(intent.get(key))
        if normalized_value is not None and joint not in targets:
            targets[joint] = normalized_value * scale
    pinch_value = _number(intent.get("pinch"))
    if pinch_value is None:
        pinch_value = _number(intent.get("grip"))
    if pinch_value is not None and "gripper" not in targets:
        targets["gripper"] = (1.0 - min(max(pinch_value, 0.0), 1.0)) * 100.0
    return targets


def _execution_fields(req: TrackingPreviewRequest) -> tuple[str, int, datetime]:
    """lease_id, sequence and sent_at: all required to execute (C9, safety-B7)."""

    if req.lease_id is None or req.sequence is None or req.sent_at is None:
        fields = (("lease_id", req.lease_id), ("sequence", req.sequence), ("sent_at", req.sent_at))
        missing = ", ".join(name for name, value in fields if value is None)
        raise HTTPException(
            422, f"lease_id, sequence and sent_at are required for execution; missing: {missing}"
        )
    return req.lease_id, req.sequence, req.sent_at


def _handle_tracking(
    req: TrackingPreviewRequest,
    *,
    arm_first: bool = False,
    client_host: str | None = None,
) -> TrackingPreviewResponse:
    execute = req.execute or arm_first
    # A malformed execution request is refused before anything else (C9).
    execution = _execution_fields(req) if execute else None
    audit_key = _audit_key(req, client_host)
    if req.confidence < MIN_TRACKING_CONFIDENCE:
        if _AUDIT.should_audit(audit_key):
            get_store().append_event(
                "tracking.rejected",
                "tracking intent rejected below confidence threshold",
                level="warning",
                source=req.source,
                detail={"confidence": req.confidence},
            )
        return TrackingPreviewResponse(
            accepted=False,
            reason=f"confidence below {MIN_TRACKING_CONFIDENCE:.2f} threshold",
            targets={},
            desired_targets={},
            commanded_targets={},
            confidence=req.confidence,
        )

    manager = get_manager()
    try:
        raw = _retarget(req)
        hold_elbow = _holds_elbow(req, raw)
        engaged = manager.engaged_position("elbow_flex") if hold_elbow else None
        if engaged is not None:
            raw["elbow_flex"] = engaged
        targets = manager.preview_targets(raw)
    except ControlRejected as exc:
        return TrackingPreviewResponse(
            accepted=False,
            reason=str(exc),
            targets={},
            desired_targets={},
            commanded_targets={},
            confidence=req.confidence,
        )

    reason = "preview accepted; execution was not requested"
    commanded_targets: dict[str, float] = {}
    if execution is not None:
        lease_id, sequence, sent_at = execution
        try:
            if arm_first:
                manager.validate_command_freshness(sent_at, req.ttl_ms)
                manager.arm_control(lease_id)
                engaged = manager.engaged_position("elbow_flex") if hold_elbow else None
                if engaged is not None:
                    targets = {**targets, "elbow_flex": engaged}
            commanded_targets = manager.send_external_command(
                lease_id=lease_id,
                source=req.source,
                sequence=sequence,
                # The client's own timestamp, also after arming, so ttl_ms
                # bounds the command that runs (safety-B7, sim-B4).
                sent_at=sent_at,
                ttl_ms=req.ttl_ms,
                deadman=req.deadman,
                targets=targets,
            )
        except (ControlConflict, ControlRejected) as exc:
            if arm_first:
                # Owner-scoped, non-strict cleanup (safety-B6, C5).
                manager.request_stop(
                    StopKind.HALT, reason="tracked first command failed", lease_id=lease_id
                )
            # safety-B9: a busy bus is 503; nothing was sent.
            raise HTTPException(503 if isinstance(exc, ControlBusy) else 409, str(exc)) from exc
        reason = "preview accepted and bounded command executed"
    if _AUDIT.should_audit(audit_key, force=arm_first):
        get_store().append_event(
            "tracking.preview",
            reason,
            source=req.source,
            detail={
                "confidence": req.confidence,
                "desired_targets": targets,
                "commanded_targets": commanded_targets,
                "executed": execute,
            },
        )
    return TrackingPreviewResponse(
        accepted=True,
        reason=reason,
        targets=commanded_targets or targets,
        desired_targets=targets,
        commanded_targets=commanded_targets,
        confidence=req.confidence,
    )


@router.post("/preview", response_model=TrackingPreviewResponse)
def preview(req: TrackingPreviewRequest, request: Request) -> TrackingPreviewResponse:
    return _handle_tracking(req, client_host=_client_host(request))


@router.post("/arm-execute", response_model=TrackingPreviewResponse)
def arm_execute(req: TrackingPreviewRequest, request: Request) -> TrackingPreviewResponse:
    """Atomically arm and accept the first fresh tracked-hand command."""

    return _handle_tracking(req, arm_first=True, client_host=_client_host(request))

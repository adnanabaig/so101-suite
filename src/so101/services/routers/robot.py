"""Robot status, servo health, the three stops (E-stop, Halt, Collapse), resets and supervision."""

from __future__ import annotations

import asyncio
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ...config import get_settings
from ..guard import GuardPolicy, read_stop_body
from ..robot_manager import (
    ControlBusy,
    ControlConflict,
    ControlRejected,
    StopKind,
    StopResult,
    get_manager,
)
from ..schemas import (
    Ack,
    FaultResetRequest,
    OperatorHeartbeatRequest,
    ResetRequest,
    RobotStatus,
    ServoHealthResponse,
    StopRequest,
)

router = APIRouter(prefix="/api/robot", tags=["robot"])

# safety-B9 (docs/audit/v0.1/safety.md): E-stop teardown never queues behind
# request handlers on the shared AnyIO threadpool. Threads start on first use.
_ESTOP_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="so101-estop")
# Longest the endpoint waits for the teardown to confirm torque-off before it
# answers 503. Kept under the 5 s timeout of `so101 estop`.
ESTOP_RESPONSE_TIMEOUT_S = 3.0
# A halt never queues behind request handlers on the shared AnyIO threadpool
# either (safety-B9); it has its own small executor.
_HALT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="so101-halt")


@router.get("/status", response_model=RobotStatus)
def status() -> RobotStatus:
    return RobotStatus(**get_manager().status())


@router.get("/health", response_model=ServoHealthResponse)
def health() -> ServoHealthResponse:
    """Servo health as the server-side monitor sees it (read-only; no bus I/O).

    Per arm: whether it is polled or why not, the last sample of every servo
    with its verdict, the thresholds, the sample age and the poll counters.
    A critical verdict has already triggered Collapse + fault; a warning only
    an event.
    """
    return ServoHealthResponse(**get_manager().health_snapshot())


@router.post("/estop", response_model=Ack)
async def estop() -> Ack:
    """Category 0 stop. 200 once torque-off is confirmed on every bus, else 503.

    safety-B9: the latch is set inline on the event loop, so it lands even when
    every request worker thread is blocked. The torque-off teardown then runs on
    a dedicated executor; the latch never waits for it.
    """

    manager = get_manager()
    manager.latch_estop("operator E-stop", lock_timeout=0.05)
    teardown = asyncio.get_running_loop().run_in_executor(_ESTOP_EXECUTOR, manager.estop)
    # asyncio.wait never cancels the teardown; it keeps running past the bound.
    done, _ = await asyncio.wait({teardown}, timeout=ESTOP_RESPONSE_TIMEOUT_S)
    if not done:
        raise HTTPException(
            503,
            "E-STOP is latched, but torque-off is not confirmed yet; "
            "cut physical power if anything still moves",
        )
    try:
        teardown.result()
    except (ControlConflict, ControlRejected) as exc:
        raise HTTPException(503, str(exc)) from exc
    return Ack(ok=True, detail="E-STOP engaged; all motion stopped")


@router.post("/estop/reset", response_model=Ack)
def reset_estop(req: ResetRequest) -> Ack:
    """C2: needs ``{"operator_id": str, "confirm": true}``; anything else is 422."""
    try:
        get_manager().reset_estop(operator_id=req.operator_id)
    except ControlConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return Ack(ok=True, detail="e-stop cleared")


@router.post("/fault/reset", response_model=Ack)
def reset_fault(req: FaultResetRequest) -> Ack:
    """C2: retries an unconfirmed torque-off first; 409 while it stays unconfirmed
    unless the body carries ``"power_cut_acknowledged": true``."""
    try:
        get_manager().clear_fault(
            operator_id=req.operator_id, power_cut_acknowledged=req.power_cut_acknowledged
        )
    except ControlConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ControlBusy as exc:
        # The motor bus did not come free to close a dead port; still latched.
        raise HTTPException(503, str(exc)) from exc
    return Ack(ok=True, detail="runtime fault cleared")


@router.post("/stop", response_model=Ack)
def stop() -> Ack:
    """Legacy global stop: a HALT through the dispatcher (design doc §5)."""
    if not get_manager().soft_stop(reason="POST /api/robot/stop"):
        raise HTTPException(
            500, "the halt did not hold; torque-off attempted and a fault latched (see status)"
        )
    return Ack(ok=True, detail="returned to idle; manual control disarmed; holding if energized")


def _lenient_stop_body(raw: bytes) -> tuple[str | None, str | None]:
    """(operator_id, reason) from a halt body. Never raises: a stop is never refused.

    Anything that is not a JSON object with string fields is ignored. A halt
    with no usable operator_id still halts; it just cannot renew the timer.
    ``raw`` comes from ``read_stop_body``: at most 4 KiB read within 0.1 s, so
    a huge or slow body is ignored rather than waited for.
    """

    try:
        body: Any = json.loads(raw) if raw.strip() else {}
    except (ValueError, RecursionError):
        return None, None
    if not isinstance(body, dict):
        return None, None
    operator = body.get("operator_id")
    reason = body.get("reason")
    operator_id = operator if isinstance(operator, str) and 0 < len(operator) <= 128 else None
    text = reason.strip()[:200] if isinstance(reason, str) and reason.strip() else None
    return operator_id, text


def _passes_origin_guard(request: Request) -> bool:
    """Whether this request would pass the Origin/Host guard (C1).

    ``/api/robot/halt`` is exempt from the guard so a stop is never blocked,
    but only a request that would pass it may renew a halt's hold timer
    (design doc §5).
    """

    try:
        policy = GuardPolicy.from_settings(get_settings())
        verdict = policy.check(
            request.scope.get("headers", ()),
            scheme=request.scope.get("scheme", "http"),
            check_origin=True,
        )
    except Exception:
        return False
    return verdict is None


def _stop_detail(result: StopResult) -> str:
    if result.holding:
        return "halted: motors on, holding position"
    if result.collapsing:
        return "collapsing: lowering to the rest pose, then motors off"
    if get_manager().guard.estopped:
        return "E-stop is latched; the motors are off"
    return "motors are off; nothing to hold or lower"


@router.post("/halt", response_model=Ack)
async def halt(request: Request) -> Ack:
    """Category 2: freeze in place (design doc §5). Exempt from the Origin guard.

    Body (optional, lenient): ``{"operator_id"?, "reason"?}``, read with a
    bound (``read_stop_body``: 4 KiB, 0.1 s); a longer or slower body is
    ignored and the halt happens anyway. 200 while the arm holds (or its
    motors are off); 503 when the hold fell back to a torque cut or the arm is
    not accounted for. A repeat halt renews the hold timer only for an
    operator with a fresh supervision heartbeat whose request passes the
    Origin guard.
    """

    operator_id, reason = _lenient_stop_body(await read_stop_body(request))
    origin_verified = _passes_origin_guard(request)
    result = await asyncio.get_running_loop().run_in_executor(
        _HALT_EXECUTOR,
        functools.partial(
            get_manager().request_stop,
            StopKind.HALT,
            reason=reason or "operator halt",
            operator_id=operator_id,
            origin_verified=origin_verified,
        ),
    )
    if not result.ok:
        raise HTTPException(503, _halt_failure(result))
    return Ack(ok=True, detail=_stop_detail(result))


def _halt_failure(result: StopResult) -> str:
    if result.fallback is not None:
        cut = "torque cut" if result.torque_off else "torque-off UNCONFIRMED, cut physical power"
        return f"the halt could not hold ({result.fallback}); {cut}; a fault is latched"
    if not result.loop_stopped:
        return "a control loop hung; forced torque-off attempted and a fault latched"
    return "the arm is not accounted for: torque-off unconfirmed; cut physical power"


@router.post("/collapse", response_model=Ack)
def collapse(req: StopRequest | None = None) -> Ack:
    """Category 1: lower to the rest pose at a capped speed, then torque off.

    Subject to the Origin guard: it commands motion. Answers once the descent
    has started (watch ``stop_state`` and ``collapse`` in the status); 503 when
    it fell back to an immediate torque cut.
    """

    reason = (req.reason or "").strip() if req is not None else ""
    result = get_manager().request_stop(StopKind.COLLAPSE, reason=reason or "operator collapse")
    if not result.ok:
        raise HTTPException(503, _collapse_failure(result))
    return Ack(ok=True, detail=_stop_detail(result))


def _collapse_failure(result: StopResult) -> str:
    if result.fallback is not None:
        cut = "torque cut" if result.torque_off else "torque-off UNCONFIRMED, cut physical power"
        return (
            f"the collapse fell back to torque-off ({result.fallback}); {cut}; a fault is latched"
        )
    if not result.loop_stopped:
        return "a control loop hung; forced torque-off attempted and a fault latched"
    return "the arm is not accounted for: torque-off unconfirmed; cut physical power"


@router.post("/operator-heartbeat", response_model=Ack)
def operator_heartbeat(req: OperatorHeartbeatRequest) -> Ack:
    get_manager().operator_heartbeat(req.operator_id)
    return Ack(ok=True, detail="operator supervision heartbeat accepted")

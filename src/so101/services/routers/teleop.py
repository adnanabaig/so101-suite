"""Teleoperation control."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..robot_manager import get_manager
from ..schemas import Ack, TeleopRequest

router = APIRouter(prefix="/api/teleop", tags=["teleop"])


@router.post("/start", response_model=Ack)
def start(req: TeleopRequest) -> Ack:
    try:
        get_manager().start_teleop(fps=req.fps, operator_id=req.operator_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return Ack(ok=True, detail=f"teleop started @ {req.fps} fps")


@router.post("/stop", response_model=Ack)
def stop() -> Ack:
    """End of teleop is a HALT (design doc §5)."""
    if not get_manager().soft_stop(reason="teleop stop"):
        raise HTTPException(
            500, "teleop did not stop cleanly; torque-off attempted and a fault latched"
        )
    return Ack(ok=True, detail="teleop stopped; the arm holds (halt)")

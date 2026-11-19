"""Trained-model registry + autonomous rollout control."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, HTTPException

from ...models import list_models
from ...policy.checkpoint import (
    ModelNotFoundError,
    PolicyBackendError,
    preflight_policy,
    resolve_control_fps,
)
from ...training.run_manifest import checkpoint_defaults
from ..robot_manager import get_manager
from ..schemas import Ack, ModelSummary, RolloutRequest

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelSummary])
def index() -> list[ModelSummary]:
    return [ModelSummary(**m.__dict__) for m in list_models()]


def _accepts_task(start_rollout: Any) -> bool:
    try:
        return "task" in inspect.signature(start_rollout).parameters
    except (TypeError, ValueError):
        return False


@router.post("/rollout/start", response_model=Ack)
def rollout_start(req: RolloutRequest) -> Ack:
    # Validate the checkpoint before the control loop exists: a bad path must
    # be a client error, not a runtime fault latched by the rollout thread.
    # Only registered models may run (audit B16).
    try:
        info = preflight_policy(req.model_path, task=req.task, registered_only=True)
        fps = resolve_control_fps(req.fps, info)
    except ModelNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except (PolicyBackendError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    manager = get_manager()
    kwargs: dict[str, Any] = {}
    explicit_task = (req.task or "").strip()
    if explicit_task:
        if _accepts_task(manager.start_rollout):
            kwargs["task"] = info.task
        elif explicit_task != (checkpoint_defaults(info.path).task or ""):
            # The rollout loop loads the policy with the checkpoint's training
            # task; never run a different prompt than the one requested.
            raise HTTPException(
                422,
                "this server's rollout loop applies only the checkpoint's training task; "
                'omit "task" to use it, or run with an explicit task via `so101 run --task`',
            )
    try:
        manager.start_rollout(str(info.path), fps=fps, operator_id=req.operator_id, **kwargs)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    detail = f"rollout started: {info.label} ({info.path}) at {fps} fps"
    if info.task:
        detail += f", task {info.task!r}"
    return Ack(ok=True, detail=detail)


@router.post("/rollout/stop", response_model=Ack)
def rollout_stop() -> Ack:
    if not get_manager().soft_stop():
        raise HTTPException(
            500, "rollout did not stop cleanly; torque-off attempted and a fault latched"
        )
    return Ack(ok=True, detail="rollout stopped; the arm holds (halt)")

"""Recording: enter recording mode, then start/save discrete episodes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...data.dataset import refuse_case_alias
from ...data.recorder import RecordingConfig
from ..robot_manager import get_manager
from ..schemas import Ack, EpisodeResult, StartRecordingRequest

router = APIRouter(prefix="/api/recording", tags=["recording"])


@router.post("/start", response_model=Ack)
def start(req: StartRecordingRequest) -> Ack:
    mgr = get_manager()
    cameras = mgr.cameras()  # its stream keys: credentials stripped, depth after color
    cfg = RecordingConfig(
        repo_id=req.repo_id,
        task=req.task,
        fps=req.fps,
        camera_keys=cameras.streams,
        # Simulated cameras record under their own names, like `so101 sim demos`.
        camera_feature_names=cameras.feature_names if cameras.backend == "sim" else {},
    )
    try:
        refuse_case_alias(req.repo_id)  # the recorder checks again under the dataset lock
        mgr.start_recording(cfg, operator_id=req.operator_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return Ack(ok=True, detail=f"recording mode for {req.repo_id}")


@router.post("/episode/start", response_model=Ack)
def episode_start() -> Ack:
    try:
        get_manager().begin_episode()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return Ack(ok=True, detail="episode recording")


@router.post("/episode/save", response_model=EpisodeResult)
def episode_save() -> EpisodeResult:
    try:
        res = get_manager().save_episode()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return EpisodeResult(**res)


@router.post("/stop", response_model=Ack)
def stop() -> Ack:
    if not get_manager().soft_stop():
        raise HTTPException(
            500, "recording did not stop cleanly; torque-off attempted and a fault latched"
        )
    return Ack(ok=True, detail="recording stopped; the arm holds (halt)")

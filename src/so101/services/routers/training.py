"""Training: presets, submit/list/cancel jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...data.dataset import DatasetBusyError, DatasetNotFoundError
from ...training.configs import POLICY_PRESETS
from ...training.preflight import (
    UnknownPolicyError,
    plan_episodes,
    resolve_episode_selection,
    resolve_preset,
    validate_extra_args,
    validate_training_dataset,
)
from ..jobs import JobPersistenceError, JobStartError, get_runner
from ..schemas import (
    Ack,
    JobSummary,
    PolicyPresetOut,
    StartTrainingRequest,
)

router = APIRouter(prefix="/api/training", tags=["training"])


def _summary(job) -> JobSummary:
    return JobSummary(
        id=job.id,
        repo_id=job.repo_id,
        policy=job.policy,
        status=job.status.value,
        step=job.step,
        steps=job.steps,
        loss=job.loss,
        output_dir=job.output_dir,
        error=job.error,
        progress=job.progress,
        episodes=job.episodes,
    )


@router.get("/presets", response_model=list[PolicyPresetOut])
def presets() -> list[PolicyPresetOut]:
    return [
        PolicyPresetOut(
            name=p.name,
            policy_type=p.policy_type,
            description=p.description,
            default_steps=p.default_steps,
            default_batch_size=p.default_batch_size,
            notes=p.notes,
            trainer=p.trainer,
            requires_torch=p.requires_torch,
        )
        for p in POLICY_PRESETS.values()
    ]


@router.post("/jobs", response_model=JobSummary)
def submit(req: StartTrainingRequest) -> JobSummary:
    try:
        preset = resolve_preset(req.policy)
    except UnknownPolicyError as exc:
        raise HTTPException(400, str(exc)) from exc
    # An omitted budget means the preset's own default, not ACT's.
    steps = req.steps if "steps" in req.model_fields_set else preset.default_steps
    batch_size = (
        req.batch_size if "batch_size" in req.model_fields_set else preset.default_batch_size
    )
    if steps < 1 or batch_size < 1:
        raise HTTPException(422, "steps and batch_size must be positive")
    # Fail synchronously, before a worker thread exists, for a dataset that is
    # missing or cannot be trained on (mock mode included), an override so101
    # owns (output dir, dataset, Hub upload, ...), or an episode selection its
    # outcome labels can't honour (unreadable, or nothing selected). The job
    # checks the labels again when it starts.
    try:
        validate_extra_args(preset, req.extra_args)
        episodes = resolve_episode_selection(req.episodes, req.extra_args)
        info = validate_training_dataset(req.repo_id, preset)
        plan_episodes(req.repo_id, episodes, info)
    except DatasetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except DatasetBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        job = get_runner().submit(
            repo_id=req.repo_id,
            policy=preset.name,
            steps=steps,
            batch_size=batch_size,
            extra_args=req.extra_args,
            include_depth=req.include_depth,
            episodes=episodes,
        )
    except (JobPersistenceError, JobStartError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return _summary(job)


@router.get("/jobs", response_model=list[JobSummary])
def jobs() -> list[JobSummary]:
    return [_summary(j) for j in get_runner().list()]


@router.get("/jobs/{job_id}", response_model=JobSummary)
def job(job_id: str) -> JobSummary:
    j = get_runner().get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return _summary(j)


@router.get("/jobs/{job_id}/logs")
def logs(job_id: str) -> dict:
    j = get_runner().get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return {"id": j.id, "status": j.status.value, "log_tail": j.log_tail}


@router.post("/jobs/{job_id}/cancel", response_model=Ack)
def cancel(job_id: str) -> Ack:
    try:
        found = get_runner().cancel(job_id)
    except JobPersistenceError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not found:
        raise HTTPException(404, "job not found")
    return Ack(ok=True, detail="cancellation requested")

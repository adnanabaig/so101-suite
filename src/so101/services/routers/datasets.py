"""Dataset listing / inspection / deletion / hub push."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...data.dataset import (
    DatasetBusyError,
    delete_dataset,
    list_datasets,
    list_trashed_datasets,
    push_to_hub,
    restore_trashed_dataset,
)
from ...data.inspection import (
    OutcomeLabelsUnreadableError,
    episode_timeline,
    inspect_dataset,
    replay_preview,
    run_quality_control,
    set_episode_outcome,
)
from ..robot_manager import get_manager
from ..schemas import (
    Ack,
    DatasetDetail,
    DatasetQualityReport,
    DatasetSummary,
    EpisodeOutcomeRequest,
    EpisodeTimelineOut,
    PushDatasetRequest,
    RestoreDatasetRequest,
    TrashedDatasetSummary,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetSummary])
def index() -> list[DatasetSummary]:
    return [DatasetSummary(**d.__dict__) for d in list_datasets()]


@router.get("/trash", response_model=list[TrashedDatasetSummary])
def trash_index() -> list[TrashedDatasetSummary]:
    return [TrashedDatasetSummary(**item.__dict__) for item in list_trashed_datasets()]


def _restore(trash_id: str) -> DatasetSummary:
    try:
        restored = restore_trashed_dataset(trash_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except DatasetBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return DatasetSummary(**restored.__dict__)


@router.post("/trash/restore", response_model=DatasetSummary)
def restore_from_body(request: RestoreDatasetRequest) -> DatasetSummary:
    # An opaque trash id can contain percent escapes from the encoded repo id;
    # carrying it in JSON avoids proxy/server path-decoding ambiguity.
    return _restore(request.trash_id)


@router.post("/trash/{trash_id}/restore", response_model=DatasetSummary)
def restore(trash_id: str) -> DatasetSummary:
    return _restore(trash_id)


@router.get(
    "/{repo_id:path}/episodes/{episode}/timeline",
    response_model=EpisodeTimelineOut,
)
def timeline(
    repo_id: str,
    episode: int,
    offset: int = 0,
    limit: int = 300,
    stride: int = 1,
) -> EpisodeTimelineOut:
    try:
        result = episode_timeline(
            repo_id,
            episode,
            offset=offset,
            limit=limit,
            stride=stride,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, f"timeline is unavailable: {exc}") from exc
    return EpisodeTimelineOut(**result.to_dict())


@router.get(
    "/{repo_id:path}/episodes/{episode}/replay-preview",
    response_model=EpisodeTimelineOut,
)
def preview(repo_id: str, episode: int, max_points: int = 240) -> EpisodeTimelineOut:
    try:
        result = replay_preview(repo_id, episode, max_points=max_points)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, f"preview is unavailable: {exc}") from exc
    return EpisodeTimelineOut(**result.to_dict())


@router.post("/{repo_id:path}/episodes/{episode}/outcome", response_model=DatasetDetail)
def label_outcome(
    repo_id: str,
    episode: int,
    request: EpisodeOutcomeRequest,
) -> DatasetDetail:
    """Label an episode (fallback: its episode.json; LeRobot: so101_outcomes.json)."""
    try:
        set_episode_outcome(repo_id, episode, request.outcome, request.notes)
        result = inspect_dataset(repo_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (DatasetBusyError, OutcomeLabelsUnreadableError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return DatasetDetail(**result.to_dict())


@router.get("/{repo_id:path}/qc", response_model=DatasetQualityReport)
def quality_control(repo_id: str) -> DatasetQualityReport:
    try:
        report = run_quality_control(repo_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except DatasetBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:  # a damaged dataset is a conflict, not a 500 (B22)
        raise HTTPException(
            409, f"quality control could not complete: {type(exc).__name__}: {exc}"
        ) from exc
    return DatasetQualityReport(**report.to_dict())


@router.get("/{repo_id:path}", response_model=DatasetDetail)
def detail(repo_id: str) -> DatasetDetail:
    try:
        result = inspect_dataset(repo_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return DatasetDetail(**result.to_dict())


@router.delete("/{repo_id:path}", response_model=Ack)
def remove(repo_id: str) -> Ack:
    try:
        removed = delete_dataset(repo_id)
    except DatasetBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not removed:
        raise HTTPException(404, f"dataset {repo_id!r} not found")
    return Ack(ok=True, detail=f"moved {repo_id} to recoverable dataset trash")


@router.post("/{repo_id:path}/push", response_model=Ack)
def push(repo_id: str, request: PushDatasetRequest) -> Ack:
    """Publish a dataset to the Hugging Face Hub (private unless asked otherwise).

    Takes a JSON body with the operator's id and requires that operator's live
    heartbeat, so a page on another site cannot publish data (audit B8).
    """
    try:
        get_manager().require_operator_heartbeat(request.operator_id)
    except RuntimeError as exc:
        raise HTTPException(403, f"cannot push {repo_id}: {exc}") from exc
    try:
        url = push_to_hub(repo_id, private=request.private)
    except DatasetBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except NotImplementedError as e:
        raise HTTPException(501, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return Ack(ok=True, detail=url)

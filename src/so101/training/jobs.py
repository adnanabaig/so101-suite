"""In-memory model for a training/eval job (the JobRunner persists these)."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    # Waiting for the single trainer slot of its kind (one GPU job at a time).
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # The API server stopped (crash, kill, power cut) before the job finished.
    # A trainer it left running is stopped when the server starts again.
    # Interrupted jobs are never restarted automatically.
    INTERRUPTED = "interrupted"
    # The job's record could not be read after a restart (for example a file
    # left truncated by a power cut). Its real outcome is unknown; the entry
    # keeps whatever could be recovered, the output directory above all.
    CORRUPT = "corrupt"


@dataclass
class TrainingJob:
    repo_id: str
    policy: str
    steps: int
    batch_size: int
    output_dir: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    # The trainer subprocess (lerobot-train), recorded as soon as it starts so
    # a restarted server can find and stop a trainer that outlived it: its
    # pid, its process group (its own session on POSIX) and the start time the
    # OS reports for it (an opaque token; a pid reused by another process has
    # a different one).
    pid: int | None = None
    pgid: int | None = None
    pid_start: str | None = None
    step: int = 0
    loss: float | None = None
    log_tail: list[str] = field(default_factory=list)
    # Latest progress-bar refresh ("Training: 42%|... 420/1000"); kept out of
    # log_tail so carriage-return refreshes cannot flush the useful lines.
    progress: str | None = None
    error: str | None = None
    extra_args: dict[str, str] = field(default_factory=dict)
    # Explicit opt-in for real LeRobot training on a Hugging Face Hub dataset
    # that is not present locally. Every other path requires a local dataset.
    remote_dataset: bool = False
    # Real LeRobot training leaves depth maps out of the policy inputs unless
    # this is set (see trainer.policy_inputs_without_depth).
    include_depth: bool = False
    # Which episodes every preset learns from, by outcome label: "kept"
    # (success and unlabeled), "success" or "all" (see
    # preflight.plan_episodes). Records written before this field restore as
    # "kept". The bc_baseline preset also reads extra_args["episodes"].
    episodes: str = "kept"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

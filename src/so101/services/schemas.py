"""Pydantic request/response models — the REST/WS contract the UI consumes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from ..agents.planner import MOTION_SKILLS, route_instruction
from ..data.dataset import repo_id_problem
from ..hardware.health import HealthReason, HealthThresholds, ServoHealthSample, UnitProvenance


class RequestModel(BaseModel):
    """Base for request bodies: every float field refuses NaN and Infinity.

    StrictJSONMiddleware already rejects those tokens in JSON bodies (stop
    routes excepted); this keeps the models finite for any other caller.
    """

    model_config = ConfigDict(allow_inf_nan=False)


def _coerce_timestamp(value: Any) -> Any:
    """Accept ISO-8601 or browser-friendly Unix seconds/milliseconds."""

    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:  # Date.now() milliseconds
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    return value


# --- robot / telemetry -------------------------------------------------
TorqueStateName = Literal["on", "off", "unconfirmed"]
# What drives the arm (SO101_BACKEND; so101.hardware.backend.current_backend).
BackendName = Literal["mock", "sim", "real"]
StopStateName = Literal["none", "halted", "collapsing"]


class HaltInfo(BaseModel):
    """An active Halt (design doc §5 "Visibility"). Times are ISO-8601 UTC.

    ``hold_expires_at`` is when the halt becomes a Collapse given supervision
    right now (the hold timer, or the grace period while no operator browser
    has a fresh heartbeat). Compare it with ``server_time``, not a local clock.
    """

    since: datetime
    reason: str
    hold_expires_at: datetime
    server_time: datetime


class CollapseInfo(BaseModel):
    """An active Collapse descent. ``progress`` is 0..1 of the distance to rest."""

    since: datetime
    reason: str
    progress: float = Field(ge=0.0, le=1.0)
    eta_s: float = Field(ge=0.0)


class SupervisionInfo(BaseModel):
    """Operator supervision as the hold timer sees it (an addition to §5).

    ``supervised`` while any operator browser's heartbeat is fresh;
    ``lost_at`` is when the newest heartbeat went stale (null while
    supervised or when no operator ever sent one).
    """

    supervised: bool
    lost_at: datetime | None = None
    grace_s: float


HealthLevelName = Literal["ok", "warn", "critical"]


class HealthSummary(BaseModel):
    """Servo health in brief, in every status reply and telemetry frame.

    ``level`` is the worst verdict over the arms being polled, null before
    their first poll or while none is (``active`` false). ``arms`` maps each
    arm to its verdict, null while it is not polled. ``age_s`` is the age of
    the oldest last full sample among the polled arms; ``stale`` flags one
    older than three poll periods. ``reasons`` holds at most three warn or
    critical messages. ``GET /api/robot/health`` has the detail.
    """

    level: HealthLevelName | None = None
    active: bool = False
    arms: dict[str, HealthLevelName | None] = Field(default_factory=dict)
    age_s: float | None = None
    stale: bool = False
    reasons: list[str] = Field(default_factory=list)


class ServoHealthMotor(BaseModel):
    """One servo's last full sample and its own verdict."""

    level: HealthLevelName | None = None
    reasons: list[HealthReason] = Field(default_factory=list)
    sample: ServoHealthSample


class ServoHealthArm(BaseModel):
    """One arm's servo-health polling (docs/bring-up.md "Servo health monitor").

    ``active`` while the server has this arm's bus open and polls it;
    otherwise ``inactive_reason`` says why. ``level``, ``reasons``, ``error``
    and ``consecutive_failures`` are the last poll's verdict (kept after
    polling stops; see ``active``). ``motors`` holds the last full sample of
    each servo. Times are ISO-8601 UTC; ``sample_age_s`` is on the server's
    monotonic clock.
    """

    arm: str
    active: bool
    inactive_reason: str | None = None
    level: HealthLevelName | None = None
    source: Literal["real", "mock", "sim"] | None = None
    polled_at: datetime | None = None
    sampled_at: datetime | None = None
    sample_age_s: float | None = None
    consecutive_failures: int = 0
    error: str | None = None
    reasons: list[HealthReason] = Field(default_factory=list)
    motors: dict[str, ServoHealthMotor] = Field(default_factory=dict)
    thresholds: HealthThresholds | None = None
    polls: int = 0
    read_failures: int = 0
    skipped_busy: int = 0
    last_skip: str | None = None
    last_read_ms: float | None = None
    max_read_ms: float = 0.0
    activations: int = 0
    critical_dispatches: int = 0


class ServoHealthResponse(BaseModel):
    """``GET /api/robot/health``: what the server-side servo-health monitor knows.

    ``poll_hz`` is ``SO101_HEALTH_POLL_HZ``; ``io_wait_ms`` the longest a poll
    waits for the motor bus before it skips. ``unit_provenance`` says which
    register scales are verified (none of the unit scales are).
    """

    active: bool
    level: HealthLevelName | None = None
    stale: bool = False
    poller_running: bool
    poll_hz: float
    io_wait_ms: float
    server_time: datetime
    arms: dict[str, ServoHealthArm]
    unit_provenance: dict[str, UnitProvenance] = Field(default_factory=dict)


class StopRequest(RequestModel):
    """Body of ``POST /api/robot/halt`` and ``/collapse``; both fields optional."""

    operator_id: str | None = Field(default=None, min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=200)


class RobotStatus(BaseModel):
    fault: str | None = None
    mode: str  # idle | teleop | recording | rollout
    follower_connected: bool
    leader_connected: bool
    estopped: bool
    # Kept for older clients. It is false under the MuJoCo sim backend as well
    # as the real one, so only ``backend`` (after ``health``) tells sim from real.
    mock: bool
    cameras: list[str]
    live_cameras: list[str] = Field(default_factory=list)
    camera_frame_age_ms: dict[str, float] = Field(default_factory=dict)
    operation_owner: str | None = None
    fps: float = 0.0
    # C11 item 5: the stop kinds with a real implementation.
    stop_kinds: list[str] = Field(default_factory=lambda: ["estop", "halt", "collapse"])
    # C2 / safety-B2: the third torque state. True after a torque-off that
    # could not be confirmed; only a confirmed torque-off (or an acknowledged
    # power cut on fault reset) clears it.
    torque_unconfirmed: bool = False
    # safety-B8: whether a restart comes back at least as latched as now.
    # "saved": the store has the newest latching snapshot. "marker": the store
    # refused it; the fallback marker <data dir>/control_plane.latch-unsaved
    # makes a restart come back faulted. "unsaved": neither could be written;
    # the latch holds only in memory, so do not restart before checking the arm.
    latch_persistence: Literal["saved", "marker", "unsaved"] = "saved"
    # C7: the checkpoint path driving the current rollout or eval trial.
    active_model: str | None = None
    # Design doc §5 "Visibility". torque_state aggregates every bus the
    # manager owns (unconfirmed > on > off). E-stop and fault keep their own
    # fields; stop_state describes only a Halt's hold or a Collapse descent.
    torque_state: TorqueStateName = "off"
    stop_state: StopStateName = "none"
    halt: HaltInfo | None = None
    collapse: CollapseInfo | None = None
    supervision: SupervisionInfo | None = None
    # Servo health in brief (docs/bring-up.md "Servo health monitor").
    health: HealthSummary | None = None
    # What drives the arm: mock, sim (MuJoCo physics) or real.
    backend: BackendName


class ResetRequest(RequestModel):
    """C2: a deliberate E-stop or fault reset (safety-B3 reset bodies,
    docs/audit/v0.1/safety.md). ``confirm`` must be the JSON literal ``true``;
    ``1`` or ``"true"`` is a 422, as is a missing or blank ``operator_id``.
    """

    operator_id: str = Field(min_length=1, max_length=128)
    confirm: StrictBool

    @field_validator("operator_id")
    @classmethod
    def _operator_named(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("operator_id must name the operator")
        return value

    @field_validator("confirm")
    @classmethod
    def _confirmed(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("confirm must be true")
        return value


class FaultResetRequest(ResetRequest):
    """C2: ``power_cut_acknowledged`` lets a fault reset finish when the
    torque-off retry still cannot be confirmed. The retry always runs first.
    """

    power_cut_acknowledged: StrictBool = False


class Telemetry(BaseModel):
    t: float
    mode: str
    estopped: bool
    fps: float
    observation: dict[str, float]  # follower joints, <joint>.pos
    action: dict[str, float]  # commanded, <joint>.pos
    recording: bool = False
    recording_saving: bool = False
    episode_frames: int = 0
    sample_age_ms: float = 0.0
    sample_seq: int = 0
    # C6: sampled by an active loop, an armed session, a halt's hold or a
    # collapse descent within the last 500 ms.
    observation_live: bool = False
    # Design doc §5 "Visibility": every frame carries the stop state.
    torque_state: TorqueStateName = "off"
    stop_state: StopStateName = "none"
    halt: HaltInfo | None = None
    collapse: CollapseInfo | None = None
    # Servo health in brief (docs/bring-up.md "Servo health monitor").
    health: HealthSummary | None = None


# --- calibration -------------------------------------------------------
class CalibrateRequest(RequestModel):
    joint: (
        Literal[
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
        ]
        | None
    ) = None
    arm: str = Field(pattern="^(follower|leader)$")
    sample_seconds: float = Field(default=12.0, ge=8.0, le=60.0)


class CalibrationStatus(BaseModel):
    joint_baselines: dict[str, dict[str, Any]] = Field(default_factory=dict)
    live: dict[str, Any] | None = None
    arm: str
    in_progress: bool
    fraction: float
    message: str
    has_saved: bool
    device_id: str = ""
    created_at: float | None = None
    ranges: dict[str, list[float]] = Field(default_factory=dict)


class CalibrationLogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class CalibrationAttempt(BaseModel):
    joint: str | None = None
    id: str
    event_id: int
    arm: str
    backend: str | None = None
    device_id: str | None = None
    sample_seconds: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: Literal["pass", "fail", "running", "incomplete"] = "incomplete"
    saved: bool | None = None
    affected_motors: list[str] = Field(default_factory=list)
    error: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    log: list[CalibrationLogEntry] = Field(default_factory=list)


class CalibrationHistory(BaseModel):
    attempts: list[CalibrationAttempt]
    limit: int


# --- teleop ------------------------------------------------------------
class TeleopRequest(RequestModel):
    fps: int = 30
    operator_id: str = Field(min_length=1, max_length=128)


# --- recording ---------------------------------------------------------
class StartRecordingRequest(RequestModel):
    repo_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$",
    )
    task: str = "Perform the task."
    fps: int = 30
    operator_id: str = Field(min_length=1, max_length=128)

    @field_validator("repo_id")
    @classmethod
    def _repo_id_is_not_a_route_name(cls, value: str) -> str:
        # The pattern cannot express the reserved route words (no lookahead).
        problem = repo_id_problem(value)
        if problem is not None:
            raise ValueError(problem)
        return value


class EpisodeResult(BaseModel):
    episode: int
    frames: int
    # Sim backend only (null elsewhere): the task predicate's verdict on the
    # saved episode, and how it was stored ("auto-labeled" or "kept the human
    # label"; grading/storage failures are non-2xx recoverable errors).
    outcome: Literal["success", "failure"] | None = None
    label: str | None = None


# --- datasets ----------------------------------------------------------
class DatasetSummary(BaseModel):
    repo_id: str
    path: str
    format: str
    num_episodes: int
    num_frames: int
    fps: int
    task: str
    camera_keys: list[str]  # color cameras
    depth_keys: list[str] = Field(default_factory=list)  # RealSense depth streams


class DatasetCapability(BaseModel):
    id: str
    available: bool
    level: Literal["full", "partial", "unavailable"]
    detail: str


class EpisodeManifestOut(BaseModel):
    episode: int
    name: str
    frames: int | None
    duration_seconds: float | None
    started_at: float | None = None
    ended_at: float | None = None
    outcome: str = "unlabeled"
    notes: str = ""
    complete: bool = True
    camera_chunks: dict[str, int] = Field(default_factory=dict)
    camera_frames: dict[str, int] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    # "human", "auto:<predicate>" (e.g. auto:sim_predicate) or "" when unlabeled.
    outcome_source: str = ""


class IncompleteCaptureOut(BaseModel):
    id: str
    frames: int
    duration_seconds: float | None
    reason: str
    captured_at: float | None
    camera_chunks: dict[str, int] = Field(default_factory=dict)


class DatasetDetail(DatasetSummary):
    duration_seconds: float
    episodes: list[EpisodeManifestOut]
    incomplete_captures: list[IncompleteCaptureOut]
    outcome_counts: dict[str, int]
    capabilities: list[DatasetCapability]


class TimelinePointOut(BaseModel):
    frame_index: int
    t: float | None
    relative_time_seconds: float
    observation: list[float]
    action: list[float]
    desired_action: list[float]


class EpisodeTimelineOut(BaseModel):
    repo_id: str
    episode: int
    fps: int
    total_frames: int | None
    offset: int
    stride: int
    returned_frames: int
    has_more: bool
    joint_names: list[str]
    actuates_hardware: bool
    capability: DatasetCapability
    points: list[TimelinePointOut]


class DatasetQualityCheck(BaseModel):
    id: str
    label: str
    status: Literal["pass", "warn", "fail", "skip"]
    detail: str
    remediation: str = ""
    affected: int = 0
    examples: list[str] = Field(default_factory=list)


class DatasetQualityReport(BaseModel):
    repo_id: str
    generated_at: float
    overall: Literal["pass", "warn", "fail"]
    checked_frames: int
    checks: list[DatasetQualityCheck]
    capabilities: list[DatasetCapability]


class TrashedDatasetSummary(BaseModel):
    id: str
    repo_id: str
    format: str
    num_episodes: int
    num_frames: int
    deleted_at: float
    size_bytes: int


class RestoreDatasetRequest(RequestModel):
    trash_id: str = Field(min_length=1, max_length=300)


class EpisodeOutcomeRequest(RequestModel):
    outcome: Literal["unlabeled", "success", "failure", "partial", "discarded"]
    notes: str = Field(default="", max_length=2000)


class PushDatasetRequest(RequestModel):
    # A JSON body, not query parameters: a cross-site "simple" request cannot
    # publish a dataset, and the pushing operator needs a live heartbeat (B8).
    operator_id: str = Field(min_length=1, max_length=128)
    private: bool = True


# --- training ----------------------------------------------------------
class StartTrainingRequest(RequestModel):
    repo_id: str
    policy: str = "act"
    # An omitted steps/batch_size takes the preset's default (see
    # GET /api/training/presets); the values below apply to ``act``.
    steps: int = Field(100_000, ge=1, le=10_000_000)
    batch_size: int = Field(8, ge=1, le=4096)
    # Validated per preset (see so101.training.preflight.validate_extra_args).
    extra_args: dict[str, str] = Field(default_factory=dict)
    # LeRobot presets: also feed depth maps to the policy (off by default).
    include_depth: bool = False
    # Additive (v0.2): the episodes every preset learns from, by outcome label.
    # None means "kept" (success and unlabeled); "all" includes failures. A
    # bc_baseline request may give it as extra_args["episodes"] instead.
    episodes: Literal["kept", "success", "all"] | None = None


class JobSummary(BaseModel):
    id: str
    repo_id: str
    policy: str
    status: str
    step: int
    steps: int
    loss: float | None
    output_dir: str
    error: str | None = None
    # Latest trainer progress-bar refresh (kept out of the log tail).
    progress: str | None = None
    # Additive (v0.2): the job's episode selection (kept, success or all).
    episodes: str = "kept"


class PolicyPresetOut(BaseModel):
    name: str
    policy_type: str
    description: str
    default_steps: int
    default_batch_size: int
    notes: str
    # Additive (v0.2): "lerobot" (lerobot-train subprocess, needs the [train]
    # extra) or "numpy_in_process" (the torch-free baseline, trains anywhere).
    trainer: str = "lerobot"
    requires_torch: bool = True


# --- models ------------------------------------------------------------
class ModelSummary(BaseModel):
    name: str
    path: str
    policy_type: str
    is_mock: bool
    checkpoint: str
    # Additive (v0.2): kind (lerobot | numpy_bc_mlp | mock), operator-facing
    # label (e.g. "privileged-state sim baseline") and training provenance.
    kind: str = ""
    label: str = ""
    privileged_state: bool = False
    # True for every language-conditioned checkpoint, including custom policy
    # types whose saved preprocessor contains the tokenizer step.
    needs_task: bool = False
    task: str | None = None
    fps: int | None = None
    dataset: str | None = None


# --- rollout / eval ----------------------------------------------------
class RolloutRequest(RequestModel):
    # A registered model: a name or path from GET /api/models (audit B16).
    model_path: str
    # None runs at the fps of the checkpoint's training data, else 30 (B18).
    fps: int | None = Field(None, ge=1, le=240)
    operator_id: str = Field(min_length=1, max_length=128)
    # Language instruction for VLA policies; None uses the training task (B6).
    task: str | None = Field(None, max_length=1000)


class StartEvalRequest(RequestModel):
    model_path: str
    n_episodes: int = 10
    max_seconds: float = Field(30.0, gt=0, le=3600, allow_inf_nan=False)
    operator_id: str = Field(min_length=1, max_length=128)
    # data-B18: None means the checkpoint's training fps, else 30.
    fps: int | None = Field(None, ge=1, le=240)
    # data-B6: language instruction for VLA policies. None or blank means the
    # checkpoint's training task; a language policy with neither is refused.
    task: str | None = Field(None, max_length=1000)


class EvalScoreRequest(RequestModel):
    """C3: only the session's operator, with a live heartbeat, may score."""

    operator_id: str = Field(min_length=1, max_length=128)
    success: bool


class EvalContinueRequest(RequestModel):
    """C3: confirms the workspace reset before the next autonomous trial."""

    operator_id: str = Field(min_length=1, max_length=128)


class EvalState(BaseModel):
    active: bool
    model_path: str = ""
    episode: int = 0  # one-based current episode, or completed count when done
    n_episodes: int = 0  # requested trials
    # idle | rollout | awaiting_score | awaiting_reset | preparing | finalizing,
    # then terminal done | aborted | failed. An E-stop or fault ends in "aborted".
    phase: str = "idle"
    successes: int = 0
    success_rate: float = 0.0  # successes / scored; interrupted trials excluded
    report_path: str = ""
    # --- additive (v0.2) ---
    scored: int = 0
    interrupted: int = 0
    task: str | None = None
    fps: int | None = None
    error: str = ""  # why the session aborted or failed, or the report write failed


class Ack(BaseModel):
    ok: bool = True
    detail: str = ""


class OperatorHeartbeatRequest(RequestModel):
    operator_id: str = Field(min_length=1, max_length=128)


# --- workstation capabilities -------------------------------------------
class Capability(BaseModel):
    id: str
    label: str
    available: bool
    maturity: str
    detail: str


class CapabilitiesResponse(BaseModel):
    system_version: str
    mock: bool  # false under sim too; ``backend`` tells mock, sim and real apart
    backend: BackendName
    capabilities: list[Capability]


# --- exclusive control plane --------------------------------------------
class LeaseRequest(RequestModel):
    source: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_.:-]+$")
    ttl_seconds: float = Field(default=15.0, ge=0.1, le=60.0)


class LeaseView(BaseModel):
    """The lease as any reader sees it, without the id (C4, safety-B5)."""

    source: str
    expires_at: datetime
    ttl_seconds: float
    server_time: datetime


class LeaseInfo(LeaseView):
    """Only POST /api/control/lease and the lease heartbeat return the id (C4)."""

    id: str


class ArmRequest(RequestModel):
    lease_id: str = Field(min_length=1, max_length=80)


class DisarmRequest(RequestModel):
    """Owner-scoped disarm (C5). A missing or wrong lease is a 409, not a 422."""

    lease_id: str | None = Field(default=None, max_length=80)


class ControlState(BaseModel):
    armed: bool
    safety_state: str
    lease: LeaseView | None
    watchdog_ms: int
    last_command_at: datetime | None


class ControlObservation(BaseModel):
    observation: dict[str, float]
    captured_at: datetime


class ControlCommandRequest(RequestModel):
    lease_id: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=80)
    sequence: int = Field(ge=0)
    sent_at: datetime
    ttl_ms: int = Field(default=500, ge=20, le=5000)
    deadman: bool
    targets: dict[str, float] = Field(min_length=1, max_length=6)

    @field_validator("sent_at", mode="before")
    @classmethod
    def command_timestamp(cls, value: Any) -> Any:
        return _coerce_timestamp(value)

    @field_validator("targets")
    @classmethod
    def finite_targets(cls, targets: dict[str, float]) -> dict[str, float]:
        import math

        if any(not math.isfinite(float(value)) for value in targets.values()):
            raise ValueError("targets must contain only finite values")
        return targets


# --- diagnostics / audit -------------------------------------------------
class DiagnosticCheck(BaseModel):
    id: str
    label: str
    status: str
    detail: str
    remediation: str


class DiagnosticsResponse(BaseModel):
    generated_at: datetime
    overall: str
    checks: list[DiagnosticCheck]
    ports: list[dict[str, Any] | str]
    events: list[dict[str, Any]]


# --- simulation ----------------------------------------------------------
class SimulationResetRequest(RequestModel):
    scenario_id: str = "tabletop_pick"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    randomize: bool = True


class SimulationJointTelemetry(BaseModel):
    source: str
    position_units: str
    velocity_units: str
    measured: dict[str, float]
    target: dict[str, float]
    error: dict[str, float]
    velocity: dict[str, float]


class SimulationState(BaseModel):
    backend: str
    backend_kind: str
    physics: bool
    truth_model: str
    hardware_isolated: bool
    writes_motor_commands: bool
    status: str
    scheduler: str
    playback_rate: float
    fixed_dt_s: float
    fixed_hz: float
    step_count: int
    sim_time_s: float
    generation: int
    updated_at: datetime
    joint_order: list[str]
    joint_limits_deg: dict[str, list[float]]
    telemetry: SimulationJointTelemetry
    kinematics: dict[str, Any]
    scene: dict[str, Any]
    # Additive (v0.2): MuJoCo engine only (contacts, effort, objects, servo); None otherwise.
    physics_state: dict[str, Any] | None = None


class SimulationResetResponse(Ack):
    state: SimulationState


class SimulationTargetRequest(RequestModel):
    targets: dict[str, float] = Field(min_length=1, max_length=6)

    @field_validator("targets")
    @classmethod
    def finite_targets(cls, targets: dict[str, float]) -> dict[str, float]:
        import math

        if any(not math.isfinite(float(value)) for value in targets.values()):
            raise ValueError("targets must contain only finite values")
        return targets


class SimulationStepRequest(RequestModel):
    steps: int = Field(default=1, ge=1, le=10_000)
    targets: dict[str, float] | None = Field(default=None, min_length=1, max_length=6)

    @field_validator("targets")
    @classmethod
    def finite_targets(cls, targets: dict[str, float] | None) -> dict[str, float] | None:
        import math

        if targets is not None and any(
            not math.isfinite(float(value)) for value in targets.values()
        ):
            raise ValueError("targets must contain only finite values")
        return targets


class SimulationPlayRequest(RequestModel):
    rate: float = Field(default=1.0, ge=0.1, le=4.0)


class SimulationTrajectoryWaypoint(RequestModel):
    time_s: float = Field(gt=0.0, le=30.0)
    targets: dict[str, float] = Field(min_length=1, max_length=6)

    @field_validator("targets")
    @classmethod
    def finite_targets(cls, targets: dict[str, float]) -> dict[str, float]:
        import math

        if any(not math.isfinite(float(value)) for value in targets.values()):
            raise ValueError("targets must contain only finite values")
        return targets


class SimulationTrajectoryPreviewRequest(RequestModel):
    target: dict[str, float] | None = Field(default=None, min_length=1, max_length=6)
    duration_s: float | None = Field(default=None, ge=0.01, le=30.0)
    waypoints: list[SimulationTrajectoryWaypoint] = Field(default_factory=list, max_length=128)
    sample_hz: float = Field(default=20.0, ge=1.0, le=240.0)

    @field_validator("target")
    @classmethod
    def finite_target(cls, targets: dict[str, float] | None) -> dict[str, float] | None:
        import math

        if targets is not None and any(
            not math.isfinite(float(value)) for value in targets.values()
        ):
            raise ValueError("target must contain only finite values")
        return targets

    @model_validator(mode="after")
    def one_trajectory_form(self) -> SimulationTrajectoryPreviewRequest:
        if (self.target is None) == (not self.waypoints):
            raise ValueError("provide exactly one of target or waypoints")
        if self.target is not None and self.duration_s is None:
            raise ValueError("duration_s is required with target")
        if self.waypoints and self.duration_s is not None:
            raise ValueError("duration_s comes from waypoint time_s values")
        return self


class SimulationTrajectoryPoint(BaseModel):
    step: int
    time_s: float
    measured: dict[str, float]
    target: dict[str, float]
    error: dict[str, float]
    end_effector: dict[str, Any]


class SimulationTrajectoryPreviewResponse(BaseModel):
    backend: str
    physics: bool
    mutates_state: bool
    source_step_count: int
    fixed_dt_s: float
    requested_duration_s: float
    quantized_duration_s: float
    requested_sample_hz: float
    sample_hz: float
    points: list[SimulationTrajectoryPoint]
    final_reached: bool
    joint_limits_deg: dict[str, list[float]]
    starting_target: dict[str, float]
    # Additive (v0.2): how targets are applied over time, e.g. "step_targets_like_play".
    semantics: str | None = None
    # Additive (v0.2): |target - measured| bound for final_reached, suite units.
    reached_tolerance: float | None = None


class SimulationCapabilitiesResponse(BaseModel):
    active_backend: str
    backend_kind: str
    physics_active: bool
    fixed_step: bool
    hardware_isolated: bool
    writes_motor_commands: bool
    adapters: dict[str, Any]


# --- hand / camera tracking ---------------------------------------------
class TrackingPreviewRequest(RequestModel):
    source: str = Field(default="camera", min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    intent: dict[str, Any] = Field(default_factory=dict)
    # A null axis, usually z from monocular tracking, carries no intent; a null
    # z holds the elbow at its engage-time value (C9, sim-B5 backend half).
    wrist: dict[str, float | None] | list[float | None] | None = None
    pinch: float | None = Field(default=None, ge=0.0, le=1.0)
    execute: bool = False
    lease_id: str | None = None
    sequence: int | None = Field(default=None, ge=0)
    # Required whenever the frame executes (execute=true, or arm-execute):
    # its age bounds the command that runs (C9, safety-B7).
    sent_at: datetime | None = None
    ttl_ms: int = Field(default=500, ge=20, le=5000)
    deadman: bool = False

    @field_validator("sent_at", mode="before")
    @classmethod
    def tracking_timestamp(cls, value: Any) -> Any:
        return _coerce_timestamp(value)


class TrackingPreviewResponse(BaseModel):
    accepted: bool
    reason: str
    targets: dict[str, float]
    desired_targets: dict[str, float] = Field(default_factory=dict)
    commanded_targets: dict[str, float] = Field(default_factory=dict)
    confidence: float


# --- bounded agent proposals --------------------------------------------
class AgentConstraints(RequestModel):
    max_duration_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    require_deadman: bool = False
    allow_motion: bool = False


class CreateAgentProposalRequest(RequestModel):
    instruction: str | None = Field(default=None, min_length=1, max_length=500)
    objective: str | None = Field(default=None, min_length=1, max_length=500)
    skill: str | None = Field(default=None, min_length=1, max_length=80)
    planner: str = Field(default="deterministic", pattern="^(deterministic|configured)$")
    constraints: AgentConstraints = Field(default_factory=AgentConstraints)

    @model_validator(mode="after")
    def infer_simple_request(self) -> CreateAgentProposalRequest:
        simple_request = self.instruction is not None and "constraints" not in self.model_fields_set
        if self.objective is None:
            self.objective = self.instruction
        if self.objective is None:
            raise ValueError("objective or instruction is required")
        if self.skill is None and self.planner == "deterministic":
            # Whole words with a negation guard, shared with DeterministicPlanner
            # (sim-B15, docs/audit/v0.1/sim.md).
            self.skill = route_instruction(self.objective)
        if simple_request:
            # A simple instruction still cannot execute directly: it only
            # creates a bounded proposal, and execution separately requires
            # explicit approval plus an armed exclusive lease.
            self.constraints.allow_motion = self.skill in MOTION_SKILLS
        return self


class ApproveAgentProposalRequest(RequestModel):
    """C8: the operator approves the exact targets they reviewed.

    ``target_digest`` is the proposal's SHA-256 target digest as shown to the
    operator; a proposal whose targets differ cannot be approved with it.
    """

    lease_id: str = Field(min_length=1, max_length=80)
    operator_id: str = Field(min_length=1, max_length=128)
    target_digest: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class ExecuteAgentProposalRequest(RequestModel):
    """C8: execution needs the single-use token returned by approval.

    The v0.1 ``approved: true`` flag is gone (sim-B7, docs/audit/v0.1/sim.md).
    """

    lease_id: str = Field(min_length=1, max_length=80)
    approval_token: str = Field(min_length=1, max_length=128)


class HoldAgentProposalRequest(RequestModel):
    """C8 deadman: sent at least every 300 ms while ``require_deadman`` runs."""

    lease_id: str = Field(min_length=1, max_length=80)

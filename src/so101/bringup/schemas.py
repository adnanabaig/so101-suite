"""Wire shapes for the bring-up kit: preflight reports and HIL runs.

These models are the contract for the wave-2 UI (preflight wizard, health
panel, HIL runner). Field names are stable; change them additively only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..hardware.backend import Backend

CheckStatus = Literal["pass", "warn", "fail", "skip"]
ArmName = Literal["follower", "leader"]
ARMS: tuple[ArmName, ...] = ("follower", "leader")
OverallStatus = Literal["pass", "warn", "fail", "incomplete"]
PreflightOrigin = Literal["cli", "api"]
#: Where a check's measured values came from. "host" covers the local
#: environment (Python, files, disk); "server" a running Suite server.
MeasuredSource = Literal["real", "mock", "sim", "host", "server", "none"]

STATUS_ORDER: dict[str, int] = {"pass": 0, "skip": 1, "warn": 2, "fail": 3}


class CheckResult(BaseModel):
    """One preflight check outcome. ``key`` is unique within a report."""

    id: str = Field(description="Check id, e.g. 'motors'")
    key: str = Field(description="Unique key: '<id>' or '<id>.<arm>'")
    arm: ArmName | None = None
    title: str
    status: CheckStatus
    message: str
    remediation: str = ""
    measured: dict[str, Any] = Field(default_factory=dict)
    blocking: bool = Field(
        default=True,
        description="A skip of a blocking check leaves hardware readiness unproven",
    )
    rehearsal: bool
    source: MeasuredSource = "host"
    references: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class PreflightReport(BaseModel):
    kind: Literal["so101.preflight"] = "so101.preflight"
    schema_version: int = 1
    created_at: str
    so101_version: str
    origin: PreflightOrigin
    backend: Backend
    rehearsal: bool
    arms: list[ArmName]
    bus_access: str = Field(description="opened | refused | deferred | synthetic | unavailable")
    bus_access_detail: str = ""
    overall: OverallStatus
    ready_for_hardware: bool = Field(
        description=(
            "True only for a real-backend run with no fail or blocking skip and a recorded "
            "rest pose validated for the current follower calibration"
        )
    )
    exit_code: int
    counts: dict[str, int]
    checks: list[CheckResult]
    notes: list[str] = Field(default_factory=list)


class PreflightBusFacts(BaseModel):
    """``POST /api/preflight/bus``: one read-only diagnostic pass per arm.

    RobotManager lends each idle bus under its bus lock
    (``run_idle_bus_diagnostic``) or refuses. ``so101 preflight`` asks a
    running server for this and grades the facts itself, so the checks are
    the same as with the server stopped. ``buses`` maps an arm to
    ``so101.bringup.buses.BusFacts`` as JSON; it is empty unless
    ``bus_access`` is ``opened`` or ``synthetic``.
    """

    kind: Literal["so101.preflight_bus"] = "so101.preflight_bus"
    backend: Backend
    arms: list[ArmName]
    bus_access: str = Field(description="opened | refused | deferred | synthetic | unavailable")
    bus_access_detail: str = ""
    remediation: str = ""
    blocking: bool = True
    buses: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PreflightCheckInfo(BaseModel):
    """Catalog entry for ``GET /api/preflight/checks`` (wizard layout)."""

    id: str
    title: str
    per_arm: bool
    needs_bus: bool
    description: str


# --- HIL -----------------------------------------------------------------

HilGroup = Literal["gate", "stop", "verification"]
EvidenceKind = Literal["bool", "number", "text", "choice"]
EvidenceValue = bool | int | float | str
ProbeKind = Literal[
    "none",
    "status",
    "preflight",
    "diagnostics",
    "telemetry",
    "events",
    "estop",
    "halt_hold",
    "collapse",
    "hold_timer",
    "supervision_loss",
]
AttestOutcome = Literal["pass", "fail", "skip"]
StepStatus = Literal["pending", "pass", "fail", "skip"]
RunStatus = Literal["in_progress", "passed", "failed", "incomplete"]


class EvidenceField(BaseModel):
    """One value the operator records when attesting a step."""

    name: str
    label: str
    kind: EvidenceKind
    unit: str | None = None
    required: bool = True
    choices: list[str] = Field(default_factory=list)
    #: A ``pass`` attestation is rejected unless the value equals this.
    pass_equals: bool | str | None = None
    #: A ``pass`` attestation is rejected unless min <= value <= max.
    pass_min: float | None = None
    pass_max: float | None = None
    help: str = ""


class HilStepDefinition(BaseModel):
    id: str
    group: HilGroup
    order: int
    title: str
    source: str = Field(description="Document and item this step comes from")
    quote: str = Field(description="The source wording, verbatim")
    instructions: list[str]
    expected_results: list[str]
    evidence_fields: list[EvidenceField]
    probe: ProbeKind = "none"
    probe_window_s: float | None = None
    probe_help: str = ""
    requires: list[str] = Field(
        default_factory=list, description="Server status fields the probe requires"
    )
    probe_checks: list[str] = Field(
        default_factory=list, description="Preflight check ids a 'preflight' probe grades"
    )
    probe_events: list[str] = Field(
        default_factory=list, description="Audit event type prefixes an 'events' probe looks for"
    )
    known_limit: bool = False


class ProbeResult(BaseModel):
    step_id: str
    probe: ProbeKind
    sequence: int
    status: CheckStatus
    message: str
    # 0/empty identifies a probe saved before the evidence contract was versioned.
    catalog_version: int = 0
    catalog_digest: str = ""
    required_status_fields: list[str] = Field(default_factory=list)
    required_status_fields_present: bool = False
    measured: dict[str, Any] = Field(default_factory=dict)
    backend: Backend
    rehearsal: bool
    started_at: str
    finished_at: str
    artifacts: list[str] = Field(default_factory=list)


def _clean_operator(value: str) -> str:
    text = " ".join(str(value).split())
    if not 1 <= len(text) <= 128:
        raise ValueError("operator must be 1-128 visible characters")
    return text


class Attestation(BaseModel):
    operator: str
    outcome: AttestOutcome
    note: str = Field(default="", max_length=4000)
    evidence: dict[str, EvidenceValue] = Field(default_factory=dict)
    # 0/empty identifies an attestation that must be redone after an upgrade.
    catalog_version: int = 0
    catalog_digest: str = ""
    attested_at: str

    @field_validator("operator")
    @classmethod
    def _check_operator(cls, value: str) -> str:
        return _clean_operator(value)


class HilStepRecord(BaseModel):
    step_id: str
    status: StepStatus = "pending"
    probes: list[ProbeResult] = Field(default_factory=list)
    attestation: Attestation | None = None
    attestation_history: list[Attestation] = Field(default_factory=list)
    updated_at: str | None = None


class HilRun(BaseModel):
    kind: Literal["so101.hil_run"] = "so101.hil_run"
    schema_version: int = 1
    # The exact step/evidence catalog used to create this run. Defaults load
    # pre-versioning runs without silently promoting their evidence.
    catalog_version: int = 0
    catalog_digest: str = ""
    run_id: str
    created_at: str
    so101_version: str
    backend: Backend
    rehearsal: bool
    hil_evidence: bool = Field(
        default=False,
        description="True only after finalize, for a real-backend run whose probes all ran "
        "against the real backend, with at least one step attested pass or fail and no stop "
        "evidence gap",
    )
    operator: str
    note: str = ""
    host: str = ""
    devices: dict[str, str] = Field(default_factory=dict)
    step_ids: list[str]
    steps: dict[str, HilStepRecord]
    status: RunStatus = "in_progress"
    finalized_at: str | None = None
    summary: dict[str, int] = Field(default_factory=dict)
    #: Audit-event id at run start; ``events`` probes read only newer events.
    event_cursor: int | None = None
    #: Set by finalize: why ``hil_evidence`` is false, plus caveats a reviewer
    #: should read even when it is true (e.g. a pass attested without a probe).
    evidence_notes: list[str] = Field(default_factory=list)


class HilRunSummary(BaseModel):
    run_id: str
    created_at: str
    backend: Backend
    rehearsal: bool
    status: RunStatus
    hil_evidence: bool
    operator: str
    finalized_at: str | None
    steps_total: int
    steps_done: int


class ManifestVerification(BaseModel):
    ok: bool
    manifest_present: bool
    checked: int = 0
    missing: list[str] = Field(default_factory=list)
    mismatched: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)
    malformed: list[str] = Field(default_factory=list)


class HilReport(BaseModel):
    run: HilRun
    markdown: str
    manifest: ManifestVerification
    run_dir: str


# --- request bodies --------------------------------------------------------


class StartHilRunRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    operator: str
    note: str = Field(default="", max_length=4000)
    step_ids: list[str] | None = None
    groups: list[HilGroup] | None = None

    @field_validator("operator")
    @classmethod
    def _check_operator(cls, value: str) -> str:
        return _clean_operator(value)


class ProbeRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    duration_s: float | None = Field(default=None, gt=0, le=180)


class AttestRequest(BaseModel):
    # Evidence may carry NaN or Infinity here on purpose: hil.validate_attestation
    # refuses non-finite evidence with a per-field message the operator can act on.

    operator: str
    outcome: AttestOutcome
    note: str = Field(default="", max_length=4000)
    evidence: dict[str, EvidenceValue] = Field(default_factory=dict)

    @field_validator("operator")
    @classmethod
    def _check_operator(cls, value: str) -> str:
        return _clean_operator(value)

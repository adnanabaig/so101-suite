"""HIL runs over HTTP: the steps behind ``so101 hil``, observed by the server.

Probes only read (robot status, telemetry, the audit log, preflight and
diagnostics); the operator triggers every action and attests the result.
Evidence lands under ``<data_dir>/hil/<run_id>/`` and finalize seals it with
``MANIFEST.sha256``. Runs on the mock or sim backend are labeled rehearsals and
are never ``hil_evidence``.

Errors: 404 unknown run or step; 409 finalized run, probe already running, or
report before finalize; 422 ``{"message", "problems"}`` for a rejected request.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Body, HTTPException

from ...bringup.hil import HilError, HilInvalid, HilService, LocalHilClient
from ...bringup.schemas import (
    AttestRequest,
    HilReport,
    HilRun,
    HilRunSummary,
    HilStepDefinition,
    HilStepRecord,
    ProbeRequest,
    ProbeResult,
    StartHilRunRequest,
)
from ..robot_manager import get_manager

router = APIRouter(prefix="/api/hil", tags=["hil"])

#: Test seams: sampled probes otherwise sleep and read the real clock.
_sleep = time.sleep
_monotonic = time.monotonic


def _sleep_now(seconds: float) -> None:
    _sleep(seconds)


def _monotonic_now() -> float:
    return _monotonic()


def _service() -> HilService:
    client = LocalHilClient(get_manager(), sleep=_sleep_now)
    return HilService(client, sleep=_sleep_now, monotonic=_monotonic_now)


def _http(exc: HilError) -> HTTPException:
    if isinstance(exc, HilInvalid):
        return HTTPException(exc.status_code, {"message": exc.message, "problems": exc.problems})
    return HTTPException(exc.status_code, exc.message)


@router.get("/steps", response_model=list[HilStepDefinition])
def list_steps() -> list[HilStepDefinition]:
    """The step catalog in run order, with sources, expected results and evidence fields."""
    return _service().steps()


@router.get("/runs", response_model=list[HilRunSummary])
def list_runs() -> list[HilRunSummary]:
    """Every run under <data_dir>/hil, newest first."""
    return _service().list_runs()


@router.post("/runs", response_model=HilRun, status_code=201)
def start_run(body: StartHilRunRequest) -> HilRun:
    """Start a run. No filter selects every step; ``groups``/``step_ids`` narrow it."""
    try:
        return _service().start_run(body)
    except HilError as exc:
        raise _http(exc) from exc


@router.get("/runs/{run_id}", response_model=HilRun)
def get_run(run_id: str) -> HilRun:
    try:
        return _service().get_run(run_id)
    except HilError as exc:
        raise _http(exc) from exc


@router.post("/runs/{run_id}/steps/{step_id}/probe", response_model=ProbeResult)
def probe_step(
    run_id: str, step_id: str, body: ProbeRequest | None = Body(default=None)
) -> ProbeResult:
    """Run the step's observe-only probe. Sampled probes block for their window (<= 180 s)."""
    try:
        return _service().probe(run_id, step_id, body.duration_s if body else None)
    except HilError as exc:
        raise _http(exc) from exc


@router.post("/runs/{run_id}/steps/{step_id}/attest", response_model=HilStepRecord)
def attest_step(run_id: str, step_id: str, body: AttestRequest) -> HilStepRecord:
    """Record the operator's outcome and evidence. Re-attesting replaces it until finalize."""
    try:
        return _service().attest(run_id, step_id, body)
    except HilError as exc:
        raise _http(exc) from exc


@router.post("/runs/{run_id}/finalize", response_model=HilRun)
def finalize_run(run_id: str) -> HilRun:
    """Grade the run, write report.json/report.md and seal MANIFEST.sha256 (immutable after)."""
    try:
        return _service().finalize(run_id)
    except HilError as exc:
        raise _http(exc) from exc


@router.get("/runs/{run_id}/report", response_model=HilReport)
def get_report(run_id: str) -> HilReport:
    """The finalized report with a fresh manifest verification."""
    try:
        return _service().report(run_id)
    except HilError as exc:
        raise _http(exc) from exc

"""Durable, bounded, approval-gated agent proposal workflow (C8).

A proposal is reviewed, then approved: approval returns a single-use token
bound to the reviewed target digest, the lease and the operator. Execution
redeems the token and returns 202; a server-side loop then drives the skill
to tolerance under the lease, the watchdog, the operator heartbeat and,
optionally, deadman holds. Clients poll ``GET /{id}`` until the proposal is
executed, failed, cancelled or timeout. See agent_executor.py and
docs/control-plane-security.md.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, HTTPException

from ... import JOINT_NAMES
from ...agents import PlannerError, PlannerResponseError, get_planner, planner_status
from ..agent_executor import ExecutionRefused, get_executor, target_digest
from ..robot_manager import ControlError, Mode, RobotManager, get_manager
from ..schemas import (
    Ack,
    ApproveAgentProposalRequest,
    CreateAgentProposalRequest,
    ExecuteAgentProposalRequest,
    HoldAgentProposalRequest,
)
from ..store import get_store

router = APIRouter(prefix="/api/agent/proposals", tags=["agent"])
PROPOSAL_MAX_AGE_SECONDS = 120


def _targets_for_skill(skill: str, allow_motion: bool) -> dict[str, float]:
    if not allow_motion:
        return {}
    normalized = skill.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"open", "open_gripper", "release"}:
        return {"gripper": 80.0}
    if normalized in {"close", "close_gripper", "grasp"}:
        return {"gripper": 15.0}
    if normalized in {"wave", "wrist_wave"}:
        return {"wrist_roll": 30.0}
    if normalized in {"home", "park", "neutral"}:
        return {joint: (50.0 if joint == "gripper" else 0.0) for joint in JOINT_NAMES}
    # Free-form language is never interpreted as motion. Unknown skills remain
    # useful observation-only proposals until a bounded skill adapter exists.
    return {}


def _public(proposal: dict[str, Any]) -> dict[str, Any]:
    return {**proposal, "target_digest": target_digest(proposal["targets"])}


def _require(proposal_id: str) -> dict[str, Any]:
    proposal = get_store().get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(404, "proposal not found")
    return proposal


def _require_reviewable(proposal: dict[str, Any]) -> None:
    """The proposal is still proposed, fresh, and carries bounded motion."""

    if proposal["status"] != "proposed":
        raise HTTPException(409, f"proposal is already {proposal['status']}")
    # The age is on this process's monotonic clock, not the stored wall-clock
    # created_at, which a clock step can move. A proposal from before a
    # restart has no known age, so it is expired.
    age_seconds = get_executor().proposal_age(proposal["id"])
    if age_seconds is None or age_seconds > PROPOSAL_MAX_AGE_SECONDS:
        get_store().transition_proposal(
            proposal["id"],
            expected=("proposed",),
            status="expired",
            result={"age_seconds": None if age_seconds is None else round(age_seconds, 3)},
        )
        get_executor().revoke(proposal["id"])
        raise HTTPException(409, "proposal expired; create and review a fresh proposal")
    if not proposal["constraints"].get("allow_motion"):
        raise HTTPException(409, "proposal constraints do not allow motion")
    if not proposal["targets"]:
        raise HTTPException(409, "proposal contains no bounded motion target")


def _require_safe(manager: RobotManager) -> None:
    if manager.guard.estopped:
        raise HTTPException(409, "E-stop is engaged; reset it before agent motion")
    fault = manager.fault
    if fault:
        raise HTTPException(409, f"runtime fault must be cleared first: {fault}")


def _refused(exc: ExecutionRefused) -> HTTPException:
    return HTTPException(exc.status_code, exc.detail)


@router.get("")
def proposals() -> list[dict]:
    return [_public(proposal) for proposal in get_store().list_proposals()]


@router.get("/planner/status")
def get_planner_status() -> dict:
    return planner_status()


@router.get("/{proposal_id}")
def get_proposal(proposal_id: str) -> dict:
    return _public(_require(proposal_id))


@router.post("", status_code=201)
def create(req: CreateAgentProposalRequest) -> dict:
    assert req.objective is not None
    manager = get_manager()
    telemetry = manager.telemetry()
    control = manager.control_state()
    observation_snapshot = {
        "captured_at": telemetry.get("t"),
        "sample_age_ms": telemetry.get("sample_age_ms"),
        "sample_seq": telemetry.get("sample_seq"),
        "mode": telemetry.get("mode"),
        "joints": telemetry.get("observation", {}),
        "safety_state": control["safety_state"],
        "armed": control["armed"],
        "lease_source": control["lease"]["source"] if control["lease"] else None,
    }
    constraints = req.constraints.model_dump()
    constraints["observation_snapshot"] = observation_snapshot
    if req.planner == "configured":
        try:
            planned = get_planner("configured").propose(
                req.objective,
                observation_snapshot,
            )
        except PlannerResponseError as exc:
            # The planner answered with something unusable (sim-B14,
            # docs/audit/v0.1/sim.md); unconfigured or unreachable stays 503.
            raise HTTPException(502, str(exc)) from exc
        except PlannerError as exc:
            raise HTTPException(503, str(exc)) from exc
        skill = planned.skill
        constraints["max_duration_seconds"] = min(
            constraints["max_duration_seconds"],
            planned.max_duration_seconds,
        )
        constraints["planner"] = {
            "provider": planned.provider,
            "model": planned.model,
            "rationale": planned.rationale,
        }
    else:
        assert req.skill is not None
        skill = req.skill
        constraints["planner"] = {
            "provider": "deterministic",
            "model": "",
            "rationale": "User-selected or deterministic bounded skill.",
        }
    targets = _targets_for_skill(skill, req.constraints.allow_motion)
    if not targets:
        constraints["allow_motion"] = False
    risk = "motion" if targets else "observation_only"
    proposal = get_store().create_proposal(
        objective=req.objective,
        skill=skill,
        constraints=constraints,
        targets=targets,
        risk=risk,
    )
    get_executor().note_proposal(proposal["id"])
    get_store().append_event(
        "agent.proposed",
        req.objective,
        source="agent",
        detail={
            "proposal_id": proposal["id"],
            "skill": skill,
            "risk": risk,
            "planner": constraints["planner"]["provider"],
        },
    )
    return _public(proposal)


@router.post("/{proposal_id}/approve")
def approve(proposal_id: str, req: ApproveAgentProposalRequest) -> dict:
    """Approve the reviewed targets; returns a 30 s single-use token (sim-B7)."""

    proposal = _require(proposal_id)
    _require_reviewable(proposal)
    digest = target_digest(proposal["targets"])
    if not hmac.compare_digest(digest, req.target_digest.lower()):
        raise HTTPException(
            409, "target digest does not match this proposal's targets; review it again"
        )
    manager = get_manager()
    _require_safe(manager)
    try:
        manager.require_operator_heartbeat(req.operator_id)
        manager.heartbeat_lease(req.lease_id)
    except ControlError as exc:
        raise HTTPException(409, str(exc)) from exc
    try:
        token, approval = get_executor().approve(
            proposal_id=proposal_id,
            digest=digest,
            lease_id=req.lease_id,
            operator_id=req.operator_id,
        )
    except ExecutionRefused as exc:
        raise _refused(exc) from exc
    expires_at = approval.expires_at.isoformat()
    get_store().append_event(
        "agent.approved",
        proposal["objective"],
        source="agent",
        detail={"proposal_id": proposal_id, "target_digest": digest, "expires_at": expires_at},
    )
    return {"approval_token": token, "expires_at": expires_at, "proposal": _public(proposal)}


@router.post("/{proposal_id}/execute", status_code=202)
def execute(proposal_id: str, req: ExecuteAgentProposalRequest) -> dict:
    """Redeem an approval and start the bounded server-side run (C8)."""

    proposal = _require(proposal_id)
    executor = get_executor()
    approval = executor.redeem(req.approval_token)
    if approval is None:
        raise HTTPException(
            409, "approval token is invalid, expired or already used; approve again"
        )
    if approval.expired:
        raise HTTPException(409, "approval token expired; approve the proposal again")
    if not (
        hmac.compare_digest(approval.proposal_id, proposal_id)
        and hmac.compare_digest(approval.lease_id.encode(), req.lease_id.encode())
    ):
        raise HTTPException(409, "approval token was issued for a different proposal or lease")
    _require_reviewable(proposal)
    if not hmac.compare_digest(target_digest(proposal["targets"]), approval.target_digest):
        raise HTTPException(409, "proposal targets changed after approval; review it again")
    manager = get_manager()
    _require_safe(manager)
    try:
        manager.require_operator_heartbeat(approval.operator_id)
    except ControlError as exc:
        raise HTTPException(409, str(exc)) from exc
    active = executor.active_proposal_id()
    if active is not None:
        raise HTTPException(409, f"another agent skill is executing ({active})")
    if manager.control_state()["armed"]:
        raise HTTPException(409, "control is already armed; disarm before an agent skill runs")
    if manager.mode is not Mode.IDLE:
        raise HTTPException(409, f"robot is busy ({manager.mode.value})")
    try:
        claimed = executor.start(
            proposal,
            lease_id=req.lease_id,
            operator_id=approval.operator_id,
            manager=manager,
        )
    except ExecutionRefused as exc:
        raise _refused(exc) from exc
    return _public(claimed)


@router.post("/{proposal_id}/hold", response_model=Ack)
def hold(proposal_id: str, req: HoldAgentProposalRequest) -> Ack:
    """Deadman hold: send at least every 300 ms while ``require_deadman`` runs."""

    try:
        get_executor().hold(proposal_id, req.lease_id)
    except ExecutionRefused as exc:
        _require(proposal_id)  # an unknown proposal is a 404, not a 409
        raise _refused(exc) from exc
    return Ack(ok=True, detail="holding")


@router.post("/{proposal_id}/cancel")
def cancel(proposal_id: str) -> dict:
    """Cancel a proposal; a running skill stops at its next tick (sim-B8)."""

    proposal = _require(proposal_id)
    store = get_store()
    executor = get_executor()
    executor.revoke(proposal_id)
    if proposal["status"] == "proposed":
        cancelled = store.transition_proposal(
            proposal_id,
            expected=("proposed",),
            status="cancelled",
            result={"cancelled": True, "stop_reason": "cancelled"},
        )
        if cancelled is not None:
            store.append_event(
                "agent.cancelled",
                proposal["objective"],
                source="agent",
                detail={"proposal_id": proposal_id, "steps": 0},
            )
            return _public(cancelled)
        proposal = _require(proposal_id)  # an execute won the race
    if proposal["status"] != "executing":
        raise HTTPException(409, f"proposal is already {proposal['status']}")
    finished = executor.cancel(proposal_id)
    if finished is False:
        # The skill thread did not stop in time (a hung driver call): cut
        # torque from here and record the partial result.
        executor.abandon(proposal_id, ("cancelled", "cancelled", "cancelled by operator"))
    elif finished is None:
        _cancel_orphan(proposal)
    current = _require(proposal_id)
    return _public(current)


def _cancel_orphan(proposal: dict[str, Any]) -> None:
    """Cancel an ``executing`` proposal that no live run owns.

    That only happens if a run was cut off without recovery. Nothing of the
    agent's is driving the arm, so this does not disarm anyone else's lease.
    """

    result = {
        "reached": False,
        "final_error_deg": {},
        "max_abs_error_deg": None,
        "duration_s": None,
        "steps": None,
        "end_state": "unknown",
        "status": "cancelled",
        "stop_reason": "cancelled",
        "error": "cancelled with no live run",
    }
    cancelled = get_store().transition_proposal(
        proposal["id"], expected=("executing",), status="cancelled", result=result
    )
    if cancelled is None:
        current = _require(proposal["id"])
        raise HTTPException(409, f"proposal is already {current['status']}")
    get_store().append_event(
        "agent.cancelled",
        proposal["objective"],
        level="warning",
        source="agent",
        detail={"proposal_id": proposal["id"], **result},
    )

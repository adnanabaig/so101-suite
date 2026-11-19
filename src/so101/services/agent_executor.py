"""Approval tokens and bounded server-side execution of agent skills (C8).

v0.1 ran an agent skill inside one HTTP request. It sent a single
rate-limited step, cut torque and reported "executed" (sim-B1). Approval was
a boolean in the same request (sim-B7). Cancel raced the motion and left no
record of it (sim-B8), and an unexpected error left the proposal "executing"
for good (sim-B9). See docs/audit/v0.1/sim.md.

This module replaces that flow:

* An approval is a single-use token, valid for 30 s and bound to the
  proposal, the target digest the operator reviewed, the lease and the
  approving operator.
* One skill runs at a time, on a server thread, at the control rate. Every
  step goes through RobotManager's external-command path (lease, increasing
  sequence, TTL, SafetyGuard), so the manager's watchdog and E-stop treat it
  like any other command stream.
* A run ends when every target joint is within tolerance, at the proposal's
  ``max_duration_seconds``, or when supervision stops it: E-stop, a runtime
  fault, a stale operator heartbeat, lapsed deadman holds, a lost lease,
  cancel, server shutdown or an unexpected error.
* Every run ends with an owner-scoped Halt (design doc §5): disarmed, the
  final pose held with torque on, and then the hold timer applies (a Collapse
  to the rest pose after ``SO101_HALT_HOLD_SECONDS``, or sooner without
  supervision). The result's ``end_state`` says which: ``holding``,
  ``collapsing``, ``deenergized``, ``estopped``, ``fault`` or ``unknown``.

The planners in so101.agents never reach this module; it moves the arm only
through RobotManager's public API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .robot_manager import ControlError, RobotManager, StopKind
from .store import ControlStore, get_store, iso_now

log = logging.getLogger(__name__)

CONTROL_HZ = 30.0
ARM_TOLERANCE_DEG = 2.0
GRIPPER_TOLERANCE = 3.0  # gripper units (0-100), not degrees
APPROVAL_TTL_S = 30.0
HOLD_TIMEOUT_S = 0.3
CANCEL_WAIT_S = 2.0
SHUTDOWN_WAIT_S = 3.0
# How long a proposal's creation time is kept: longer than any proposal max
# age. A purged entry reads as "age unknown", which is expired.
PROPOSAL_CLOCK_RETENTION_S = 600.0
TERMINAL_STATUSES = frozenset({"executed", "failed", "cancelled", "timeout"})

_Stop = tuple[str, str, str]  # (status, stop_reason, error)


def target_digest(targets: Mapping[str, Any]) -> str:
    """Lowercase hex SHA-256 of the canonical JSON of ``targets`` (C8).

    Values are floats, so ``80`` and ``80.0`` give the same digest. This is
    the same recipe the UI uses to show the operator what they approve.
    """

    canonical = json.dumps(
        {joint: float(value) for joint, value in targets.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tolerance_for(joint: str) -> float:
    return GRIPPER_TOLERANCE if joint == "gripper" else ARM_TOLERANCE_DEG


def _same(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _token_key(token: str) -> str:
    # Only a hash of each token is kept, so the approval table never holds a
    # usable credential.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class ExecutionRefused(Exception):
    """A request the executor declines; the router maps it to HTTP."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class Approval:
    proposal_id: str
    target_digest: str
    lease_id: str
    operator_id: str
    expires_monotonic: float
    expires_at: datetime

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_monotonic


@dataclass
class _Run:
    proposal_id: str
    objective: str
    skill: str
    lease_id: str
    operator_id: str
    require_deadman: bool
    max_duration_s: float
    manager: RobotManager
    store: ControlStore
    targets: dict[str, float] = field(default_factory=dict)
    started_at: str = ""
    started_monotonic: float = 0.0
    last_hold: float = 0.0
    steps: int = 0
    errors: dict[str, float] = field(default_factory=dict)
    stop: _Stop | None = None
    # Set once the run's result is stored. If the thread hung and the run was
    # recorded on its behalf, the thread exits without a second record, and
    # without disarming a session armed since (see _may_disarm).
    recorded: bool = False
    wake: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


class AgentExecutor:
    """Holds live approvals, proposal creation times and the single active run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._approvals: dict[str, Approval] = {}
        # Monotonic creation time of each proposal this process created.
        self._created: dict[str, float] = {}
        self._run: _Run | None = None
        self._closed = False

    # --- proposal freshness -------------------------------------------------
    def note_proposal(self, proposal_id: str) -> None:
        """Start the freshness clock of a proposal this process just created."""

        with self._lock:
            now = time.monotonic()
            self._created = {
                key: created
                for key, created in self._created.items()
                if now - created <= PROPOSAL_CLOCK_RETENTION_S
            }
            self._created[proposal_id] = now

    def proposal_age(self, proposal_id: str) -> float | None:
        """Seconds since this process created the proposal, on the monotonic clock.

        None if this process did not create it: it was loaded from the
        database after a restart, so its age is unknown. The stored
        ``created_at`` is wall-clock time, which can be stepped, so it is not
        used.
        """

        with self._lock:
            created = self._created.get(proposal_id)
            return None if created is None else time.monotonic() - created

    # --- approvals ----------------------------------------------------------
    def approve(
        self, *, proposal_id: str, digest: str, lease_id: str, operator_id: str
    ) -> tuple[str, Approval]:
        token = secrets.token_urlsafe(32)
        ttl = APPROVAL_TTL_S
        approval = Approval(
            proposal_id=proposal_id,
            target_digest=digest,
            lease_id=lease_id,
            operator_id=operator_id,
            expires_monotonic=time.monotonic() + ttl,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
        )
        with self._lock:
            if self._closed:
                raise ExecutionRefused(503, "server is shutting down")
            self._purge_locked()
            # One live approval per proposal: approving again replaces it.
            self._drop_locked(proposal_id)
            self._approvals[_token_key(token)] = approval
        return token, approval

    def redeem(self, token: str) -> Approval | None:
        """Spend ``token`` and return its approval, even if it has expired.

        Any presentation burns the token, whether or not the request it came
        with succeeds, so a token can never start two runs.
        """

        with self._lock:
            approval = self._approvals.pop(_token_key(token), None)
            self._purge_locked()
        return approval

    def revoke(self, proposal_id: str) -> None:
        with self._lock:
            self._drop_locked(proposal_id)

    def _drop_locked(self, proposal_id: str) -> None:
        self._approvals = {
            key: approval
            for key, approval in self._approvals.items()
            if approval.proposal_id != proposal_id
        }

    def _purge_locked(self) -> None:
        now = time.monotonic()
        self._approvals = {
            key: approval
            for key, approval in self._approvals.items()
            if approval.expires_monotonic > now
        }

    # --- runs -----------------------------------------------------------------
    def active_proposal_id(self) -> str | None:
        with self._lock:
            return self._run.proposal_id if self._run is not None else None

    def start(
        self,
        proposal: dict[str, Any],
        *,
        lease_id: str,
        operator_id: str,
        manager: RobotManager,
    ) -> dict[str, Any]:
        """Claim ``proposal``, arm the follower and start the control loop.

        Arming happens here, on the request thread, so a refused arm is a
        synchronous 409. Returns the claimed (``executing``) proposal.
        """

        constraints = proposal.get("constraints") or {}
        store = get_store()
        run = _Run(
            proposal_id=proposal["id"],
            objective=proposal["objective"],
            skill=proposal["skill"],
            lease_id=lease_id,
            operator_id=operator_id,
            require_deadman=bool(constraints.get("require_deadman")),
            max_duration_s=float(constraints.get("max_duration_seconds", 5.0)),
            manager=manager,
            store=store,
        )
        with self._lock:
            if self._closed:
                raise ExecutionRefused(503, "server is shutting down")
            if self._run is not None:
                raise ExecutionRefused(
                    409, f"another agent skill is executing ({self._run.proposal_id})"
                )
            # Reserve the only slot before any I/O so concurrent requests
            # cannot both reach the arm (sim-B8, docs/audit/v0.1/sim.md).
            self._run = run
        try:
            run.targets = manager.preview_targets(proposal["targets"])
            run.started_at = iso_now()
            claimed = store.transition_proposal(
                run.proposal_id,
                expected=("proposed",),
                status="executing",
                result={"started_at": run.started_at},
            )
        except ControlError as exc:
            self._release(run)
            raise ExecutionRefused(409, str(exc)) from exc
        except BaseException:
            # Nothing is armed yet. A slot kept after an error would refuse
            # every later execute until a restart. If the claim committed
            # before the error, no run owns the proposal: cancel ends it as an
            # orphan, and restart recovery fails it.
            self._release(run)
            raise
        if claimed is None:
            self._release(run)
            current = store.get_proposal(run.proposal_id)
            state = current["status"] if current else "gone"
            raise ExecutionRefused(409, f"proposal is already {state}")
        try:
            manager.arm_control(lease_id)
        except ControlError as exc:
            # arm_control cleans up after itself. Disarming here could cut
            # torque for a different lease holder, so it is not called.
            self._finish(run, ("failed", "arm", str(exc)))
            raise ExecutionRefused(409, str(exc)) from exc
        except BaseException as exc:
            log.exception("agent skill %s: unexpected error while arming", run.proposal_id)
            self._stop_motion(run, "failed")
            self._finish(run, ("failed", "arm", f"{type(exc).__name__}: {exc}"))
            raise
        try:
            self._audit(
                run,
                "agent.executing",
                "info",
                {
                    "proposal_id": run.proposal_id,
                    "skill": run.skill,
                    "targets": dict(run.targets),
                    "require_deadman": run.require_deadman,
                    "max_duration_seconds": run.max_duration_s,
                },
            )
            with self._lock:
                # Together, so hold() never sees a start time without the
                # matching deadman start.
                run.started_monotonic = time.monotonic()
                run.last_hold = run.started_monotonic
            threading.Thread(
                target=self._loop, args=(run,), name=f"agent-skill-{run.proposal_id}", daemon=True
            ).start()
        except BaseException as exc:
            self._stop_motion(run, "failed")
            self._finish(run, ("failed", "error", f"could not start the skill thread: {exc}"))
            raise
        return claimed

    def hold(self, proposal_id: str, lease_id: str) -> None:
        """Deadman hold from the operator's client (C8 ``require_deadman``)."""

        with self._lock:
            run = self._run
            if run is None or run.proposal_id != proposal_id:
                raise ExecutionRefused(409, "proposal is not executing")
            if not _same(run.lease_id, lease_id):
                raise ExecutionRefused(409, "hold must come from the lease that started the skill")
            if run.stop is not None:
                raise ExecutionRefused(409, "skill is stopping")
            now = time.monotonic()
            # The deadman times only a started run. A hold while execute is
            # still arming is a refresh; the start resets the window anyway.
            started = bool(run.started_monotonic)
            if run.require_deadman and started and now - run.last_hold > HOLD_TIMEOUT_S:
                # A hold that arrives after the deadline cannot revive the run
                # just because the loop has not ticked yet.
                run.stop = ("failed", "deadman", _deadman_error())
                run.wake.set()
                raise ExecutionRefused(409, "deadman lapsed; the skill is stopping")
            run.last_hold = now

    def cancel(self, proposal_id: str, *, wait: float = CANCEL_WAIT_S) -> bool | None:
        """Stop the run at its next tick.

        Returns ``None`` when no live run owns the proposal, ``True`` when the
        run has finished and ``False`` when it did not finish within ``wait``.
        """

        with self._lock:
            run = self._run
            if run is None or run.proposal_id != proposal_id:
                return None
            if run.stop is None:
                run.stop = ("cancelled", "cancelled", "cancelled by operator")
        run.wake.set()
        return run.done.wait(wait)

    def abandon(self, proposal_id: str, stop: _Stop) -> bool:
        """Record a run whose thread is not responding, and halt it (owner-scoped).

        The slot stays reserved until the thread really exits, so no new
        skill can start while a hung step might still return.
        """

        with self._lock:
            run = self._run
            if run is None or run.proposal_id != proposal_id or run.recorded:
                return False
            run.stop = run.stop or stop
            run.recorded = True
            outcome = run.stop
        log.error("agent skill %s did not stop; recording it as %s", proposal_id, outcome[0])
        self._stop_motion(run, outcome[0])
        self._record(run, outcome)
        return True

    def shutdown(self, timeout: float = SHUTDOWN_WAIT_S) -> None:
        stop: _Stop = ("failed", "shutdown", "interrupted by server shutdown")
        with self._lock:
            self._closed = True
            self._approvals.clear()
            run = self._run
            if run is not None and run.stop is None:
                run.stop = stop
        if run is None:
            return
        run.wake.set()
        if not run.done.wait(timeout):
            self.abandon(run.proposal_id, stop)

    # --- control loop -----------------------------------------------------------
    def _loop(self, run: _Run) -> None:
        outcome: _Stop = ("failed", "error", "skill loop ended without a result")
        try:
            outcome = self._drive(run)
        except Exception as exc:
            log.exception("agent skill %s failed", run.proposal_id)
            outcome = ("failed", "error", f"{type(exc).__name__}: {exc}")
        finally:
            if self._may_disarm(run):
                self._stop_motion(run, outcome[0])
            self._finish(run, outcome)

    def _drive(self, run: _Run) -> _Stop:
        manager = run.manager
        period = 1.0 / CONTROL_HZ
        deadline = run.started_monotonic + run.max_duration_s
        next_tick = time.monotonic()
        while True:
            run.errors = _measure(manager, run.targets)
            now = time.monotonic()
            stop = self._stop_cause(run, now)
            if stop is not None:
                return stop
            if _within_tolerance(run.errors):
                return ("executed", "target_reached", "")
            if now >= deadline:
                return (
                    "timeout",
                    "max_duration",
                    f"targets not reached within {run.max_duration_s:g} s",
                )
            try:
                manager.execute_approved_targets(run.lease_id, run.targets)
            except ControlError as exc:
                return _classify(manager, str(exc))
            run.steps += 1
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay <= 0:
                # Late: carry on from now rather than bursting to catch up.
                next_tick = time.monotonic()
                continue
            run.wake.wait(delay)

    def _stop_cause(self, run: _Run, now: float) -> _Stop | None:
        manager = run.manager
        with self._lock:
            requested = run.stop
            last_hold = run.last_hold
        if requested is not None:
            return requested
        safety = _safety_stop(manager)
        if safety is not None:
            return safety
        try:
            manager.require_operator_heartbeat(run.operator_id)
        except ControlError:
            return (
                "failed",
                "operator_heartbeat",
                f"operator heartbeat is older than {manager.operator_timeout_ms} ms",
            )
        if run.require_deadman and now - last_hold > HOLD_TIMEOUT_S:
            return ("failed", "deadman", _deadman_error())
        return None

    # --- ending a run -------------------------------------------------------------
    def _may_disarm(self, run: _Run) -> bool:
        """Whether the run's thread may disarm on its way out.

        Not if abandon() took over the run and control is armed. abandon()
        marks the run recorded and then halts, and an arm already in progress
        then fails its authority check. So control armed after that is either
        the run's session, which abandon() is about to halt, or a new session
        under the run's lease, which the owner-scoped halt in _stop_motion
        would end.

        If control is not armed, the thread still halts. When a stuck driver
        write kept abandon()'s halt from holding, the halt fell back to a
        torque cut that could not be confirmed, with a fault latched (so nobody
        can arm) and the arm still energized. Once the write returns, this halt
        finds the fault latched and cuts torque with read-back: a Halt never
        holds under a latched fault.
        """

        with self._lock:
            abandoned = run.recorded
        if not abandoned:
            return True
        try:
            return not run.manager.control_state()["armed"]
        except Exception:
            log.exception("agent skill %s: could not read the control state", run.proposal_id)
            return True

    @staticmethod
    def _stop_motion(run: _Run, status: str) -> None:
        # An owner-scoped Halt (design doc §5): it disarms only the session
        # armed under the run's lease, including one the operator armed again
        # under that lease after a cancel, and never another client's. The
        # final pose is held with torque on; the hold timer applies after.
        # _may_disarm() still guards the late exit after abandon(). A lease
        # that lapsed was already halted by its expiry, so this is a no-op.
        try:
            run.manager.request_stop(
                StopKind.HALT,
                reason=f"agent skill {status}",
                lease_id=run.lease_id,
                operator_id=run.operator_id,
            )
        except Exception:
            log.exception("agent skill %s: halt failed", run.proposal_id)

    def _finish(self, run: _Run, outcome: _Stop) -> None:
        try:
            with self._lock:
                recorded, run.recorded = run.recorded, True
            if recorded:
                log.warning(
                    "agent skill %s stopped after it was recorded (%s)",
                    run.proposal_id,
                    outcome[1],
                )
            else:
                self._record(run, outcome)
        finally:
            self._release(run)

    def _record(self, run: _Run, outcome: _Stop) -> None:
        status, stop_reason, error = outcome
        result = _result(run, status, stop_reason, error, _end_state(run.manager))
        finished = None
        try:
            finished = run.store.transition_proposal(
                run.proposal_id, expected=("executing",), status=status, result=result
            )
        except Exception:
            log.exception("agent skill %s: could not store the result", run.proposal_id)
        detail: dict[str, Any] = {"proposal_id": run.proposal_id, **result}
        if finished is None:
            # The proposal left "executing" some other way. The event still
            # records what the arm did (sim-B8, docs/audit/v0.1/sim.md).
            detail["state_conflict"] = True
        level = "info" if status == "executed" else "warning"
        self._audit(run, f"agent.{status}", level, detail)

    @staticmethod
    def _audit(run: _Run, event_type: str, level: str, detail: dict[str, Any]) -> None:
        # Agent events never carry the lease id, the approval token or the
        # operator id.
        try:
            run.store.append_event(
                event_type, run.objective, level=level, source="agent", detail=detail
            )
        except Exception:
            log.exception("could not persist %s for %s", event_type, run.proposal_id)

    def _release(self, run: _Run) -> None:
        with self._lock:
            if self._run is run:
                self._run = None
        run.done.set()


def _deadman_error() -> str:
    return f"deadman released: no hold within {HOLD_TIMEOUT_S * 1000:.0f} ms"


def _measure(manager: RobotManager, targets: Mapping[str, float]) -> dict[str, float]:
    observation = manager.telemetry().get("observation") or {}
    errors: dict[str, float] = {}
    for joint, target in targets.items():
        value = observation.get(f"{joint}.pos")
        if value is None:
            raise RuntimeError(f"no measured position for {joint}")
        measured = float(value)
        if not math.isfinite(measured):
            raise RuntimeError(f"non-finite measured position for {joint}")
        errors[joint] = abs(measured - target)
    return errors


def _within_tolerance(errors: Mapping[str, float]) -> bool:
    return bool(errors) and all(error <= tolerance_for(joint) for joint, error in errors.items())


def _safety_stop(manager: RobotManager) -> _Stop | None:
    if manager.guard.estopped:
        return ("failed", "estop", "E-stop engaged")
    fault = manager.fault
    if fault:
        return ("failed", "fault", f"runtime fault: {fault}")
    return None


def _classify(manager: RobotManager, message: str) -> _Stop:
    # The manager's state says why a command was refused more reliably than
    # its message: an E-stop also clears the lease, for instance.
    return _safety_stop(manager) or ("failed", "rejected", message)


def _end_state(manager: RobotManager) -> str:
    """How the arm was left: from the manager's own stop and torque truth.

    ``holding`` (a Halt holds the final pose, motors on), ``collapsing`` (a
    Collapse is lowering it), ``deenergized`` (torque confirmed off on every
    bus), ``estopped``, ``fault``, or ``unknown``. Never "deenergized" unless
    the torque truth says off.
    """

    try:
        if manager.guard.estopped:
            return "estopped"
        if manager.fault:
            return "fault"
        status = manager.status()
    except Exception:
        log.exception("could not read the robot state")
        return "unknown"
    if status.get("stop_state") == "halted":
        return "holding"
    if status.get("stop_state") == "collapsing":
        return "collapsing"
    if status.get("torque_state") == "off":
        return "deenergized"
    return "unknown"


def _result(run: _Run, status: str, stop_reason: str, error: str, end_state: str) -> dict:
    errors = {joint: round(value, 3) for joint, value in run.errors.items()}
    duration = time.monotonic() - run.started_monotonic if run.started_monotonic else 0.0
    return {
        "reached": _within_tolerance(run.errors),
        "final_error_deg": errors,
        "max_abs_error_deg": max(errors.values()) if errors else None,
        "duration_s": round(duration, 3),
        "steps": run.steps,
        "end_state": end_state,
        "status": status,
        "stop_reason": stop_reason,
        "error": error,
        "targets": dict(run.targets),
        "tolerance": {"arm_deg": ARM_TOLERANCE_DEG, "gripper": GRIPPER_TOLERANCE},
        "started_at": run.started_at,
        "finished_at": iso_now(),
    }


_executor: AgentExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> AgentExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = AgentExecutor()
        return _executor


def shutdown_executor() -> None:
    """Stop any running skill and drop the executor (app shutdown)."""

    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown()


def reset_executor() -> None:
    """Test helper: stop any run and forget approvals."""

    shutdown_executor()


def recover_interrupted_proposals() -> int:
    """Fail proposals a crash or restart left ``executing`` (sim-B9).

    Called at startup. The torque state after a crash is not known, so the
    end state is "unknown". Returns how many proposals were recovered.
    """

    store = get_store()
    with _executor_lock:
        executor = _executor
    live = executor.active_proposal_id() if executor is not None else None
    recovered = 0
    for proposal in store.list_proposals(500):
        if proposal["status"] != "executing" or proposal["id"] == live:
            continue
        previous = proposal.get("result") or {}
        result: dict[str, Any] = {
            "reached": False,
            "final_error_deg": {},
            "max_abs_error_deg": None,
            "duration_s": None,
            "steps": None,
            "end_state": "unknown",
            "status": "failed",
            "stop_reason": "restart",
            "error": "interrupted by restart",
            "started_at": previous.get("started_at"),
            "finished_at": iso_now(),
        }
        failed = store.transition_proposal(
            proposal["id"], expected=("executing",), status="failed", result=result
        )
        if failed is None:
            continue
        store.append_event(
            "agent.failed",
            proposal["objective"],
            level="warning",
            source="agent",
            detail={"proposal_id": proposal["id"], **result},
        )
        recovered += 1
    return recovered

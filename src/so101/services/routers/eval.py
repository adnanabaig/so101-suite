"""Operator-scored evaluation sessions with durable, self-describing reports.

Gates (contract C3; safety-B1 in docs/audit/v0.1/safety.md):

- ``POST /api/eval/score`` and ``POST /api/eval/continue`` take a JSON body
  naming the operator. Only the operator who started the session may score or
  continue it, and only with a fresh supervision heartbeat. Requiring a JSON
  body also stops a cross-site page from sending these as "simple" requests;
  the C1 guard (services/guard.py) rejects foreign origins before that.
- Continue refuses while the E-stop is latched or a runtime fault is active.
- An E-stop or runtime fault aborts the session: phase ``"aborted"``, with
  ``error`` naming the cause. Resetting the E-stop never resumes it.

Durability (docs/audit/v0.1/data.md):

- data-B7: the app lifespan calls :func:`shutdown_session` before the robot
  manager shuts down, so a restart saves a partial report. Every trial is also
  journaled (fsynced) to ``eval_<ms>_<id>.jsonl`` as soon as it ends. A start
  is refused if its initial journal or audit event cannot be saved; a failed
  trial journal aborts before the next trial and remains in the final report.
- data-B13: ``/start`` preflights and loads the registered policy once and
  answers 404/422 if it cannot; every trial receives that same policy, and the
  trial clock starts at its first action, not at ``start_rollout``.
- data-B14: the report records ``requested_episodes``, ``scored``,
  ``interrupted``, ``phase`` and ``error``; ``success_rate`` counts scored
  trials only; the file is written atomically (temp file, then ``os.replace``)
  as ``eval_<ms>_<uuid12>.json``; episodes are numbered from 1.
- data-B6/B18: ``task`` and ``fps`` default from the checkpoint and reach every
  rollout; explicit request values override those defaults.
- Start, score and continue first persist write-ahead `*.requested` events;
  their completed events are best effort after the mutation. Score also waits
  for the trial journal fsync before its response. Stop remains safety-first
  and reports an audit warning only after motion has stopped.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from fastapi import APIRouter, HTTPException

from ...atomic_io import fsync_directory
from ...config import get_settings
from ...eval.evaluator import EvalReport
from ...policy.checkpoint import (
    ModelNotFoundError,
    PolicyBackendError,
    active_backend,
    preflight_policy,
    resolve_control_fps,
)
from ...policy.server import load_policy
from ..robot_manager import ControlRejected, RobotManager, get_manager
from ..schemas import Ack, EvalContinueRequest, EvalScoreRequest, EvalState, StartEvalRequest
from ..store import get_store

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["eval"])

# Longest wait from start_rollout to the policy's first action (the rollout
# thread receives the policy already loaded by /start). The trial clock starts
# at that action, so loading never eats into max_seconds (data-B13).
FIRST_ACTION_TIMEOUT_S = 120.0
POLL_S = 0.05
STOP_JOIN_S = 5.0

ESTOP_ERROR = "E-stop engaged"
OPERATOR_STOP_ERROR = "stopped by operator"
SHUTDOWN_ERROR = "server shutdown"


class EvalStartError(Exception):
    """``/start`` was refused with an HTTP status other than 409."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class EvalPersistenceError(RuntimeError):
    """Required eval audit or journal state could not be made durable."""


def _required_event(
    event_type: str,
    message: str,
    *,
    level: str = "info",
    detail: dict[str, Any] | None = None,
) -> None:
    """Persist an event before its corresponding eval mutation is committed."""

    try:
        get_store().append_event(
            event_type,
            message,
            level=level,
            source="eval",
            detail=detail,
            durable=True,
        )
    except Exception as exc:
        raise EvalPersistenceError(
            f"evaluation audit event {event_type!r} could not be saved: {exc}"
        ) from exc


def _optional_event(
    event_type: str,
    message: str,
    *,
    level: str = "info",
    detail: dict[str, Any] | None = None,
) -> str | None:
    """Best-effort event for a transition that has already happened safely."""

    try:
        get_store().append_event(
            event_type,
            message,
            level=level,
            source="eval",
            detail=detail,
        )
    except Exception as exc:
        problem = f"evaluation audit event {event_type!r} could not be saved: {exc}"
        log.error(problem)
        return problem
    return None


def _motion_block(manager: RobotManager) -> str:
    """Why autonomous motion must not run now (E-stop or fault), else ``""``."""

    if manager.guard.estopped:
        return ESTOP_ERROR
    fault = manager.fault
    return f"runtime fault: {fault}" if fault else ""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` so ``path`` holds either nothing or the whole report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def _quarantine_report(path: Path) -> Path | None:
    """Move a contradictory terminal report outside the ``eval_*.json`` set."""

    if not path.exists():
        return None
    quarantine = path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex[:8]}")
    try:
        os.replace(path, quarantine)
        fsync_directory(path.parent)
        return quarantine
    except OSError as move_exc:
        log.error("could not quarantine contradictory eval report %s: %s", path, move_exc)
    try:
        path.unlink(missing_ok=True)
        fsync_directory(path.parent)
    except OSError as unlink_exc:
        log.critical("could not remove contradictory eval report %s: %s", path, unlink_exc)
    return None


class _Journal:
    """Required, append-only and fsynced record of one session (data-B7)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: TextIO | None = None
        self.error = ""

    def write(self, event: str, **payload: Any) -> None:
        if self.error:
            raise EvalPersistenceError(self.error)
        try:
            created = not self.path.exists()
            if self._stream is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open("a", encoding="utf-8")
            record = {"event": event, "t": round(time.time(), 3), **payload}
            self._stream.write(json.dumps(record, allow_nan=False) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            if created:
                fsync_directory(self.path.parent)
        except (OSError, TypeError, ValueError) as exc:
            self.error = f"evaluation journal {self.path} could not be saved: {exc}"
            self.close()
            raise EvalPersistenceError(self.error) from exc

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None

    def discard(self) -> None:
        """Remove a journal for a start that was refused before any motion."""

        self.close()
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            log.error("could not remove refused eval journal %s: %s", self.path, exc)


class _EvalSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._score = threading.Event()
        self._score_recorded = threading.Event()
        self._score_record_error = ""
        self._score_abort_error = ""
        self._continue = threading.Event()
        self._rollout_start_finished = threading.Event()
        self._rollout_start_finished.set()
        self._thread: threading.Thread | None = None
        self._starting = False
        self._start_cancelled = False
        self.active = False
        self.phase = "idle"
        self.error = ""
        self.model_path = ""
        self.operator_id = ""
        self.task: str | None = None
        self.fps: int | None = None
        self.episode = 0
        self.n_episodes = 0
        self.max_seconds = 30.0
        self.report_path = ""
        self._session_id = ""
        self._stem = ""
        self._started_at = 0.0
        self._policy: Any | None = None
        self._label = ""
        self._privileged_state = False
        self._source = ""
        self._safety_generation: tuple[int, int, int] | None = None
        self._trials: list[dict[str, Any]] = []
        self._last_success: bool | None = None
        self._stop_reason = ""
        self._rollout_running = False
        self._finished = True
        self._journal: _Journal | None = None

    # --- start -------------------------------------------------------------
    def _check_ready(self, manager: RobotManager, operator_id: str) -> None:
        block = _motion_block(manager)
        if block:
            raise RuntimeError(f"cannot start an eval: {block}")
        if manager.mode.value != "idle":
            raise RuntimeError(f"robot busy ({manager.mode.value})")
        if manager.control_state()["lease"] is not None:
            raise RuntimeError("external control lease is active")
        try:
            manager.require_operator_heartbeat(operator_id)
        except ControlRejected as exc:
            raise RuntimeError(str(exc)) from exc

    def _busy(self) -> bool:
        thread = self._thread
        return self.active or self._starting or bool(thread and thread.is_alive())

    def start(self, req: StartEvalRequest) -> None:
        if req.n_episodes <= 0 or req.n_episodes > 1000:
            raise RuntimeError("n_episodes must be between 1 and 1000")
        # Written so NaN fails it too: every comparison with NaN is False.
        if not (0 < req.max_seconds <= 3600):
            raise RuntimeError("max_seconds must be between 0 and 3600")
        manager = get_manager()
        requested_task = (req.task or "").strip() or None
        with self._lock:
            if self._busy():
                raise RuntimeError("eval already running")
            self._check_ready(manager, req.operator_id)
            start_generation = manager.safety_authority_generation()
            self._starting = True
            self._start_cancelled = False
        try:
            # Resolve a registry entry and every checkpoint-owned default before
            # loading or motion. This is the same preflight used by the rollout
            # API and CLI; an API request cannot run an arbitrary directory.
            try:
                info = preflight_policy(
                    req.model_path,
                    task=requested_task,
                    registered_only=True,
                )
                fps = resolve_control_fps(req.fps, info)
            except ModelNotFoundError as exc:
                raise EvalStartError(404, str(exc)) from exc
            except (PolicyBackendError, ValueError) as exc:
                raise EvalStartError(422, str(exc)) from exc

            # Load before any motion (data-B13): a bad checkpoint is a client
            # error, not a runtime fault latched by the rollout thread. Keep
            # this instance and reset it for each trial; never load per trial.
            try:
                policy = load_policy(info.path, task=info.task)
            except Exception as exc:
                raise EvalStartError(
                    422, f"could not load policy from {str(info.path)!r}: {exc}"
                ) from exc
            with self._lock:
                if self._start_cancelled:
                    raise RuntimeError("eval start was cancelled by a stop request")
                self._check_ready(manager, req.operator_id)
                self._begin(
                    req,
                    model_path=str(info.path),
                    task=info.task,
                    fps=fps,
                    policy=policy,
                    label=str(getattr(policy, "label", info.label)),
                    privileged_state=bool(
                        getattr(policy, "privileged_state", info.privileged_state)
                    ),
                    source=active_backend(),
                    safety_generation=start_generation,
                )
        finally:
            with self._lock:
                self._starting = False

    def _begin(
        self,
        req: StartEvalRequest,
        *,
        model_path: str,
        task: str | None,
        fps: int,
        policy: Any,
        label: str,
        privileged_state: bool,
        source: str,
        safety_generation: tuple[int, int, int],
    ) -> None:
        # Caller holds the lock.
        self._started_at = time.time()
        self._session_id = uuid.uuid4().hex[:12]
        self._stem = f"eval_{int(self._started_at * 1000)}_{self._session_id}"
        self.active = True
        self.phase = "rollout"
        self.error = ""
        self.model_path = model_path
        self.operator_id = req.operator_id
        self.task = task
        self.fps = fps
        self.n_episodes = req.n_episodes
        self.max_seconds = req.max_seconds
        self.episode = 0
        self.report_path = ""
        self._policy = policy
        self._label = label
        self._privileged_state = privileged_state
        self._source = source
        self._safety_generation = safety_generation
        self._trials = []
        self._last_success = None
        self._stop_reason = ""
        self._rollout_running = False
        self._finished = False
        self._score.clear()
        self._score_recorded.clear()
        self._score_record_error = ""
        self._score_abort_error = ""
        self._continue.clear()
        journal = _Journal(get_settings().outputs_dir / f"{self._stem}.jsonl")
        self._journal = journal
        audit_saved = False
        try:
            # Both durable records precede thread.start(): a failed disk or
            # SQLite write refuses the request while no autonomous motion has
            # been signalled or committed.
            journal.write(
                "started",
                session_id=self._session_id,
                model_path=self.model_path,
                requested_episodes=self.n_episodes,
                max_seconds=self.max_seconds,
                fps=fps,
                task=task,
                source=source,
                label=label,
            )
            _required_event(
                "eval.start_requested",
                f"evaluation start authorized for {req.model_path}",
                detail={
                    "session_id": self._session_id,
                    "n_episodes": req.n_episodes,
                    "max_seconds": req.max_seconds,
                    "fps": fps,
                    "task": task,
                    "source": source,
                },
            )
            audit_saved = True
            # FULL SQLite sync can take long enough for authority or safety to
            # change. Recheck after persistence and still before thread.start.
            if get_manager().safety_authority_generation() != safety_generation:
                raise RuntimeError("safety/control authority changed during eval start")
            self._check_ready(get_manager(), req.operator_id)
            thread = threading.Thread(
                target=self._loop,
                name=f"so101-eval-{self._session_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            _optional_event(
                "eval.started",
                f"evaluation thread started for {req.model_path}",
                detail={"session_id": self._session_id},
            )
        except Exception as exc:
            # The session is still hidden behind _lock and no thread was
            # started when persistence or the final readiness check failed.
            refused_session_id = self._session_id
            self.active = False
            self.phase = "idle"
            self.error = ""
            self.model_path = ""
            self.operator_id = ""
            self.task = None
            self.fps = None
            self.episode = 0
            self.n_episodes = 0
            self.max_seconds = 30.0
            self.report_path = ""
            self._session_id = ""
            self._stem = ""
            self._started_at = 0.0
            self._thread = None
            self._policy = None
            self._label = ""
            self._privileged_state = False
            self._source = ""
            self._safety_generation = None
            self._trials = []
            self._last_success = None
            self._stop_reason = ""
            self._rollout_running = False
            self._finished = True
            self._journal = None
            self._score.clear()
            self._score_recorded.clear()
            self._score_record_error = ""
            self._score_abort_error = ""
            self._continue.clear()
            journal.discard()
            if audit_saved:
                _optional_event(
                    "eval.start_refused",
                    f"evaluation did not start after its audit commit: {exc}",
                    level="warning",
                    detail={"session_id": refused_session_id},
                )
            raise

    # --- session thread ----------------------------------------------------
    def _abort_cause(self, manager: RobotManager) -> str:
        with self._lock:
            if self._stop_reason:
                return self._stop_reason
            expected_generation = self._safety_generation
        block = _motion_block(manager)
        if block:
            return block
        if (
            expected_generation is not None
            and manager.safety_authority_generation() != expected_generation
        ):
            return "safety/control authority changed during evaluation"
        return ""

    def _wait(self, event: threading.Event, manager: RobotManager) -> str:
        """Block until ``event`` is set; return the abort cause, or ``""``."""

        while not event.wait(POLL_S):
            cause = self._abort_cause(manager)
            if cause:
                return cause
        return self._abort_cause(manager)

    def _watch_rollout(self, manager: RobotManager, seq0: object) -> tuple[str, float | None]:
        """Supervise one rollout; return (abort cause or "", first-action time)."""

        load_deadline = time.monotonic() + FIRST_ACTION_TIMEOUT_S
        acted_at: float | None = None
        while True:
            cause = self._abort_cause(manager)
            if cause:
                return cause, acted_at
            telemetry = manager.telemetry()
            now = time.monotonic()
            if acted_at is None and telemetry.get("sample_seq") != seq0:
                acted_at = now
            if telemetry.get("mode") != "rollout":
                cause = self._abort_cause(manager) or "rollout stopped before the trial ended"
                return cause, acted_at
            if acted_at is None:
                if now >= load_deadline:
                    return (
                        f"the policy produced no action within {FIRST_ACTION_TIMEOUT_S:g} s",
                        None,
                    )
            elif now - acted_at >= self.max_seconds:
                return "", acted_at
            time.sleep(POLL_S)

    def _record(
        self, number: int, success: bool | None, duration: float, *, interrupted: str = ""
    ) -> None:
        trial: dict[str, Any] = {"episode": number, "success": success, "duration": duration}
        if interrupted:
            trial["interrupted"] = interrupted
        with self._lock:
            if self._finished:
                raise EvalPersistenceError(
                    "evaluation finished before the trial result could be journaled"
                )
            self._trials.append(trial)
            journal = self._journal
            # Keep the result in memory before the required journal write. If
            # fsync fails, _loop aborts before another trial and the terminal
            # report still carries this result and the persistence error.
            if journal is None:
                raise EvalPersistenceError("evaluation journal is unavailable")
            journal.write("trial", **trial)
        if interrupted:
            _optional_event(
                "eval.trial_interrupted",
                interrupted,
                level="warning",
                detail={"session_id": self._session_id, "episode": number},
            )

    def _run_trial(self, manager: RobotManager, number: int) -> str:
        """Run and score one trial; return an abort cause, or ``""`` to go on."""

        with self._lock:
            if self._stop_reason:
                return self._stop_reason
            self.episode = number
            self.phase = "rollout"
        cause = self._abort_cause(manager)
        if cause:
            return cause
        seq0 = manager.telemetry().get("sample_seq")
        started = time.monotonic()
        policy = self._policy
        if policy is None:
            raise RuntimeError("evaluation policy was not loaded")
        fps = self.fps
        if fps is None:
            raise RuntimeError("evaluation control rate was not resolved")
        cause = self._abort_cause(manager)
        if cause:
            return cause
        # Publish the whole start window. Stop can then keep issuing the
        # idempotent manager stop until it either sees the live loop or this
        # call finishes, including while RobotManager's post-start audit blocks.
        with self._lock:
            if self._stop_reason:
                return self._stop_reason
            if (
                self._safety_generation is not None
                and manager.safety_authority_generation() != self._safety_generation
            ):
                return "safety/control authority changed during evaluation"
            self._rollout_running = True
            self._rollout_start_finished.clear()
        try:
            manager.start_rollout(
                self.model_path,
                fps=fps,
                operator_id=self.operator_id,
                policy=policy,
                task=self.task,
            )
        except BaseException as exc:
            # start_rollout may have launched its control thread before a late
            # bookkeeping/audit failure. Stop before classifying it.
            manager.stop()
            with self._lock:
                self._rollout_running = False
            if isinstance(exc, RuntimeError):
                cause = self._abort_cause(manager)
                if cause:
                    return cause
            raise
        finally:
            self._rollout_start_finished.set()
        try:
            cause, acted_at = self._watch_rollout(manager, seq0)
        finally:
            manager.stop()
            with self._lock:
                self._rollout_running = False
        duration = round(time.monotonic() - (acted_at or started), 3)
        if cause:
            self._record(number, None, duration, interrupted=cause)
            return cause

        with self._lock:
            # Clear before publishing awaiting_score so a score is never lost.
            self._score.clear()
            self._last_success = None
            self.phase = "awaiting_score"
        cause = self._wait(self._score, manager)
        with self._lock:
            success = self._last_success
            if success is None:
                # Leave awaiting_score in the same locked block, so a score
                # that arrives now is refused, not accepted for a trial that
                # is about to be recorded as interrupted. _loop then aborts.
                self.phase = "aborted"
        if success is None:
            cause = cause or "trial was not scored"
            self._record(number, None, duration, interrupted=cause)
            return cause
        try:
            self._record(number, success, duration)
            cause = cause or self._abort_cause(manager)
            if cause:
                with self._lock:
                    trial = self._trials[-1]
                    trial["success"] = None
                    trial["interrupted"] = cause
                    journal = self._journal
                    if journal is None:
                        raise EvalPersistenceError("evaluation journal is unavailable")
                    journal.write(
                        "trial_invalidated",
                        episode=number,
                        interrupted=cause,
                    )
        except BaseException as exc:
            with self._lock:
                self._score_record_error = str(exc) or type(exc).__name__
                self._score_recorded.set()
            raise
        with self._lock:
            if cause:
                self._score_abort_error = cause
            self._score_recorded.set()
        if cause:
            return cause
        if number >= self.n_episodes:
            return ""
        with self._lock:
            if self._stop_reason:
                return self._stop_reason
            # A deliberate workspace-reset gate before the next autonomous trial.
            self._continue.clear()
            self.phase = "awaiting_reset"
        return self._wait(self._continue, manager)

    def _loop(self) -> None:
        manager = get_manager()
        phase, error = "done", ""
        try:
            for number in range(1, self.n_episodes + 1):
                cause = self._run_trial(manager, number)
                if cause:
                    phase, error = "aborted", cause
                    break
        except Exception as exc:  # a runtime error becomes a visible failed report
            log.exception("eval session failed")
            phase, error = "failed", str(exc)
            _optional_event("eval.failed", error, level="error")
        finally:
            self._finish(phase, error)

    # --- report ------------------------------------------------------------
    def _report_payload(self, *, phase: str, error: str) -> dict[str, Any]:
        """Build the HTTP report through the CLI's canonical report model."""

        # Caller holds the lock. ``from_trials`` makes interrupted trials
        # neither successes nor failures, computes the Wilson interval over
        # scored trials, and carries the actual backend rather than transport.
        trials = [dict(trial) for trial in self._trials]
        report = EvalReport.from_trials(
            self.model_path,
            trials,
            requested_episodes=self.n_episodes,
            phase=phase,
            error=error,
            source=self._source,
            task=self.task,
            fps=self.fps,
            label=self._label,
            privileged_state=self._privileged_state,
            created_at=self._started_at,
        )
        payload = asdict(report)
        payload.update(
            max_seconds=self.max_seconds,
            session_id=self._session_id,
            started_at=self._started_at,
        )
        return payload

    def _finish(self, phase: str, error: str) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            # Do not publish a terminal phase until its journal and report
            # outcome are known. The UI treats finalizing as active work.
            self.active = True
            self.phase = "finalizing"
            self.error = error
            terminal_phase = phase
            terminal_error = error
            session_id = self._session_id
            path = get_settings().outputs_dir / f"{self._stem}.json"
            # Release checkpoint/GPU state after the terminal report has taken
            # everything descriptive into session fields. A wedged rollout
            # thread keeps its own local reference until it exits.
            self._policy = None
            journal, self._journal = self._journal, None
        with self._lock:
            if self._session_id != session_id:
                return
            report = self._report_payload(phase=terminal_phase, error=terminal_error)
        report_saved = False
        report_failure: Exception | None = None
        try:
            _write_json_atomic(path, report)
        except (OSError, TypeError, ValueError) as exc:
            log.error("could not save eval report %s: %s", path, exc)
            report_failure = exc
            note = f"report not saved: {exc}"
            terminal_error = f"{terminal_error}; {note}" if terminal_error else note
            quarantined = _quarantine_report(path)
            if quarantined is not None:
                terminal_error += f"; quarantined stale report as {quarantined.name}"
        else:
            report_saved = True

        # Record the terminal outcome only after report publication has been
        # attempted, so a journal never claims clean completion before a later
        # report failure is known.
        if journal is not None:
            try:
                journal.write(
                    "finished",
                    phase=terminal_phase,
                    error=terminal_error,
                    scored=report["scored"],
                    successes=report["successes"],
                    report=str(path) if report_saved else None,
                )
            except EvalPersistenceError as exc:
                log.error("eval journal finalization failed: %s", exc)
                note = str(exc)
                if note not in terminal_error:
                    terminal_error = f"{terminal_error}; {note}" if terminal_error else note
                terminal_phase = "failed"
                if report_saved:
                    with self._lock:
                        report = self._report_payload(
                            phase=terminal_phase,
                            error=terminal_error,
                        )
                    try:
                        _write_json_atomic(path, report)
                    except (OSError, TypeError, ValueError) as update_exc:
                        log.error("could not update eval report %s: %s", path, update_exc)
                        report_failure = update_exc
                        report_saved = False
                        update_note = f"failed report update: {update_exc}"
                        terminal_error = f"{terminal_error}; {update_note}"
                        quarantined = _quarantine_report(path)
                        if quarantined is not None:
                            terminal_error += f"; quarantined stale report as {quarantined.name}"
            finally:
                journal.close()

        with self._lock:
            if self._session_id == session_id:
                self.report_path = str(path) if report_saved else ""
                self.active = False
                self.phase = terminal_phase
                self.error = terminal_error
        event_detail = {
            "session_id": session_id,
            "phase": terminal_phase,
            "error": terminal_error,
            "requested_episodes": report["requested_episodes"],
            "completed_episodes": report["n_episodes"],
            "scored": report["scored"],
            "successes": report["successes"],
            "path": str(path),
        }
        if report_saved:
            _optional_event(
                "eval.report_saved",
                f"evaluation {terminal_phase}; report saved",
                detail=event_detail,
            )
        else:
            _optional_event(
                "eval.report_failed",
                f"evaluation {terminal_phase}; report not saved: {report_failure}",
                level="error",
                detail=event_detail,
            )

    # --- operator gates ----------------------------------------------------
    def _require_operator(self, operator_id: str) -> None:
        # Caller holds the lock.
        if operator_id != self.operator_id:
            raise RuntimeError("only the operator who started this eval may score or continue it")
        try:
            get_manager().require_operator_heartbeat(operator_id)
        except ControlRejected as exc:
            raise RuntimeError(str(exc)) from exc

    def score(self, operator_id: str, success: bool) -> None:
        with self._lock:
            if not self.active or self.phase != "awaiting_score":
                raise RuntimeError("not awaiting a score")
            self._require_operator(operator_id)
            cause = self._abort_cause(get_manager())
            if cause:
                self._stop_reason = cause
                self._score.set()
                raise RuntimeError(cause)
            if self._last_success is not None:
                raise RuntimeError("episode was already scored")
            episode = self.episode
            _required_event(
                "eval.score_requested",
                f"episode {episode} score requested as {'success' if success else 'failure'}",
                detail={
                    "session_id": self._session_id,
                    "episode": episode,
                    "success": success,
                },
            )
            cause = self._abort_cause(get_manager())
            if cause:
                self._stop_reason = cause
                self._score.set()
                raise RuntimeError(cause)
            self._score_recorded.clear()
            self._score_record_error = ""
            self._score_abort_error = ""
            self._last_success = success
            self._score.set()
            recorded = self._score_recorded
        # Do not acknowledge a score until its trial line is fsynced. The arm
        # is already stopped while awaiting the score, so an I/O stall cannot
        # extend autonomous motion.
        recorded.wait()
        with self._lock:
            if self._score_abort_error:
                raise RuntimeError(self._score_abort_error)
            if self._score_record_error:
                raise EvalPersistenceError(self._score_record_error)
        _optional_event(
            "eval.scored",
            f"episode {episode} scored {'success' if success else 'failure'}",
            detail={"session_id": self._session_id, "episode": episode, "success": success},
        )

    def continue_next(self, operator_id: str) -> None:
        with self._lock:
            if not self.active or self.phase != "awaiting_reset":
                raise RuntimeError("not awaiting a workspace reset")
            self._require_operator(operator_id)
            cause = self._abort_cause(get_manager())
            if cause:
                self._stop_reason = cause
                self._continue.set()
                raise RuntimeError(f"cannot continue: {cause}")
            episode = self.episode
            _required_event(
                "eval.continue_requested",
                f"operator requested continuation after episode {episode}",
                detail={"session_id": self._session_id, "episode": episode},
            )
            # The durable write can outlive a heartbeat or race a stop.
            self._require_operator(operator_id)
            cause = self._abort_cause(get_manager())
            if cause:
                self._stop_reason = cause
                self._continue.set()
                raise RuntimeError(f"cannot continue: {cause}")
            self.phase = "preparing"
            self._continue.set()
        _optional_event(
            "eval.workspace_reset_confirmed",
            f"operator confirmed reset after episode {episode}",
            detail={"session_id": self._session_id, "episode": episode},
        )

    # --- stop / state ------------------------------------------------------
    def stop(self, reason: str = OPERATOR_STOP_ERROR) -> str | None:
        with self._lock:
            if self._starting:
                self._start_cancelled = True
            if not self.active:
                return None
            if not self._stop_reason:
                self._stop_reason = reason
            thread = self._thread
            rollout_running = self._rollout_running
            self._score.set()
            self._continue.set()
        if rollout_running:
            manager = get_manager()
            while True:
                saw_live_loop = manager.mode.value == "rollout"
                manager.stop()
                if saw_live_loop:
                    break
                if self._rollout_start_finished.is_set():
                    # The start may have completed between the mode sample and
                    # the first stop; one final stop closes that race.
                    manager.stop()
                    break
                self._rollout_start_finished.wait(POLL_S)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=STOP_JOIN_S)
            if thread.is_alive():
                # The session thread is wedged (e.g. in a hung driver call):
                # save what was scored rather than lose it.
                self._finish("aborted", f"{reason}; eval thread did not stop")
        return _optional_event(
            "eval.aborted",
            f"evaluation stopped: {reason}",
            level="warning",
            detail={"session_id": self._session_id},
        )

    def state(self) -> EvalState:
        with self._lock:
            scored = [t for t in self._trials if isinstance(t.get("success"), bool)]
            successes = sum(1 for trial in scored if trial["success"])
            rate = successes / len(scored) if scored else 0.0
            return EvalState(
                active=self.active,
                model_path=self.model_path,
                episode=self.episode,
                n_episodes=self.n_episodes,
                phase=self.phase,
                successes=successes,
                success_rate=round(rate, 3),
                report_path=self.report_path,
                scored=len(scored),
                interrupted=len(self._trials) - len(scored),
                task=self.task,
                fps=self.fps,
                error=self.error,
            )


_session = _EvalSession()


def get_session() -> _EvalSession:
    return _session


def shutdown_session() -> None:
    """Lifespan hook: end an active eval and save its partial report (data-B7)."""

    _session.stop(SHUTDOWN_ERROR)


def reset_session() -> None:
    """Test helper: stop any active eval and start from a fresh session."""

    global _session
    _session.stop(SHUTDOWN_ERROR)
    _session = _EvalSession()


@router.post("/start", response_model=Ack)
def start(req: StartEvalRequest) -> Ack:
    try:
        _session.start(req)
    except EvalStartError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
    except EvalPersistenceError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return Ack(ok=True, detail=f"eval started: {req.n_episodes} episodes")


@router.get("/state", response_model=EvalState)
def state() -> EvalState:
    return _session.state()


@router.post("/score", response_model=EvalState)
def score(req: EvalScoreRequest) -> EvalState:
    try:
        _session.score(req.operator_id, req.success)
    except EvalPersistenceError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _session.state()


@router.post("/continue", response_model=EvalState)
def continue_next(req: EvalContinueRequest) -> EvalState:
    try:
        _session.continue_next(req.operator_id)
    except EvalPersistenceError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _session.state()


@router.post("/stop", response_model=Ack)
def stop() -> Ack:
    audit_warning = _session.stop()
    state = _session.state()
    detail = "eval stopped"
    if state.report_path:
        detail += f"; report saved to {state.report_path}"
    if audit_warning:
        detail += f"; WARNING: {audit_warning}"
    return Ack(ok=True, detail=detail)

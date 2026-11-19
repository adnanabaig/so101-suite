"""Run N evaluation rollouts and aggregate operator-scored success.

Real-robot eval can't be auto-scored, so each rollout's success is reported
by the operator (via the UI/CLI) through ``score_fn``. The harness handles
sequencing, reset prompts, and aggregation.

Reports are self-describing and durable (audit B12, B14):

- every report says where its numbers come from (``source``: ``real``,
  ``sim`` or ``mock``; sim and mock results are never presented as real),
  how many trials were requested, scored and interrupted, the terminal
  ``phase`` and any ``error``;
- the success rate is computed over scored trials only and comes with a
  Wilson 95% interval (``ci95``) over ``n`` scored trials;
- reports are written atomically under millisecond-plus-uuid names, and a
  CLI eval appends every scored trial to a journal (``.jsonl``, fsynced) and
  saves a partial report when it is stopped or fails.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ..atomic_io import fsync_directory
from ..config import get_settings
from ..logging import get_logger
from ..policy.checkpoint import active_backend, resolve_control_fps
from ..policy.runner import PolicyRunner
from ..policy.server import load_policy
from ..training.run_manifest import checkpoint_defaults
from .stats import wilson_interval

log = get_logger(__name__)

REPORT_VERSION = 2
PHASES = ("done", "aborted", "failed")


class EvalAborted(Exception):
    """The operator ended an evaluation early (a declined reset or score prompt)."""


class EvalJournalError(RuntimeError):
    """A required evaluation journal record could not be made durable."""


@dataclass
class EvalConfig:
    model_path: str
    n_episodes: int = 10
    max_seconds: float = 30.0
    # None: the fps of the checkpoint's training data, else 30 (audit B18).
    fps: int | None = None
    # "": the checkpoint's training task; VLA policies need one (audit B6).
    task: str = ""


def _report_stem(created_at: float) -> str:
    return f"eval_{int(created_at * 1000)}_{uuid.uuid4().hex[:8]}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    published = False
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        published = True
        fsync_directory(path.parent)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        if published and path.exists():
            quarantine = path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex[:8]}")
            try:
                os.replace(path, quarantine)
            except OSError as exc:
                log.error("could not quarantine non-durable eval report %s: %s", path, exc)
                try:
                    path.unlink(missing_ok=True)
                except OSError as unlink_exc:
                    log.critical(
                        "could not remove non-durable eval report %s: %s",
                        path,
                        unlink_exc,
                    )
        raise


@dataclass
class EvalReport:
    model_path: str
    n_episodes: int
    successes: int
    success_rate: float
    per_episode: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # --- additive (v0.2); derived when omitted -------------------------------
    # Trials behind ``success_rate`` and ``ci95``. ``from_trials`` counts scored
    # trials only (an interrupted trial is neither a success nor a failure); a
    # report built without ``n`` uses ``n_episodes``, the denominator older
    # producers divide by, so the interval always matches the rate.
    n: int | None = None
    # Wilson 95% interval [low, high] for successes / n.
    ci95: list[float] | None = None
    # Scene seeds per trial, when the environment is seeded (simulation).
    seeds: list[int] = field(default_factory=list)
    # Where the trials ran: "real" hardware, "sim" physics or the "mock" arm.
    # Sim and mock results are never real-robot results.
    source: str = ""
    # None: not recorded by the producer.
    requested_episodes: int | None = None
    scored: int | None = None
    interrupted: int | None = None
    # "done", "aborted" or "failed"; "" when the producer did not record it.
    phase: str = ""
    error: str = ""
    task: str | None = None
    fps: int | None = None
    # The policy implementation that actually ran. In mock mode this names
    # MockPolicy rather than implying that checkpoint weights were loaded.
    label: str = ""
    privileged_state: bool = False
    report_version: int = REPORT_VERSION

    def __post_init__(self) -> None:
        flags = [
            entry["success"]
            for entry in self.per_episode
            if isinstance(entry, dict) and "success" in entry
        ]
        if self.scored is None:
            scored = sum(isinstance(flag, bool) for flag in flags)
            self.scored = scored if flags else self.n_episodes
        if self.interrupted is None:
            self.interrupted = sum(flag is None for flag in flags)
        if self.n is None:
            self.n = self.n_episodes
        for entry in self.per_episode:
            # Older producers append a trailing {"error", "phase"} entry.
            if isinstance(entry, dict) and "success" not in entry:
                if not self.error and isinstance(entry.get("error"), str):
                    self.error = entry["error"]
                if not self.phase and isinstance(entry.get("phase"), str):
                    self.phase = entry["phase"]
        if not self.source:
            self.source = active_backend()
        if self.ci95 is None:
            n = max(0, int(self.n))
            low, high = wilson_interval(min(max(0, int(self.successes)), n), n)
            self.ci95 = [round(low, 4), round(high, 4)]

    @classmethod
    def from_trials(
        cls,
        model_path: str,
        trials: Sequence[dict[str, Any]],
        *,
        requested_episodes: int | None = None,
        phase: str = "done",
        error: str = "",
        seeds: Sequence[int] = (),
        source: str = "",
        task: str | None = None,
        fps: int | None = None,
        label: str = "",
        privileged_state: bool = False,
        created_at: float | None = None,
    ) -> EvalReport:
        """Build a report from per-trial records ``{"episode", "success", ...}``.

        ``success`` is ``True``/``False`` for a scored trial and ``None`` for
        an interrupted one. The rate and interval use scored trials only.
        """

        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        records = [dict(trial) for trial in trials]
        scored = [trial for trial in records if isinstance(trial.get("success"), bool)]
        successes = sum(1 for trial in scored if trial["success"])
        return cls(
            model_path=model_path,
            n_episodes=len(records),
            successes=successes,
            success_rate=successes / len(scored) if scored else 0.0,
            per_episode=records,
            created_at=time.time() if created_at is None else created_at,
            n=len(scored),
            seeds=[int(seed) for seed in seeds],
            source=source,
            requested_episodes=(
                len(records) if requested_episodes is None else int(requested_episodes)
            ),
            scored=len(scored),
            interrupted=len(records) - len(scored),
            phase=phase,
            error=error,
            task=task,
            fps=fps,
            label=label,
            privileged_state=privileged_state,
        )

    def save(self, path: Path | None = None) -> Path:
        """Write the report atomically; the default name is unique per call."""

        path = path or (get_settings().outputs_dir / f"{_report_stem(self.created_at)}.json")
        _write_json_atomic(path, asdict(self))
        return path


class EvalJournal:
    """Required append-only, fsynced JSON-lines record of a saved eval."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: TextIO | None = None
        self.error = ""

    @property
    def opened(self) -> bool:
        return self._stream is not None

    def write(self, event: str, **payload: Any) -> None:
        if self.error:
            raise EvalJournalError(self.error)
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
            problem = f"evaluation journal {self.path} could not be saved: {exc}"
            try:
                self.close()
            except OSError as close_exc:
                problem += f"; closing the failed journal also failed: {close_exc}"
            self.error = problem
            raise EvalJournalError(self.error) from exc

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()


class EvalRun:
    """One evaluation: rollouts, scoring, journal and the final report.

    With ``save=True`` every trial is journaled to
    ``<outputs>/eval_<ms>_<id>.jsonl`` as soon as it is scored, and the report
    is saved to the matching ``.json`` when the run ends, including when it is
    stopped (``phase="aborted"``) or fails (``phase="failed"``) after the first
    trial started. The initial and trial journal writes are required: an
    initial failure refuses before connect, and a trial failure ends the run
    before another rollout while retaining that trial in the JSON report. A
    run that fails before any trial (the policy or the arm would not start)
    leaves no files. ``score_fn`` and ``on_reset`` receive the 0-based episode
    index; reports number episodes from 1, like HTTP evals.
    """

    def __init__(
        self,
        cfg: EvalConfig,
        score_fn: Callable[[int], bool] | None = None,
        on_reset: Callable[[int], None] | None = None,
        *,
        save: bool = False,
        output_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.score_fn = score_fn
        self.on_reset = on_reset
        self.created_at = time.time()
        stem = _report_stem(self.created_at)
        directory = output_dir or get_settings().outputs_dir
        self.report_path: Path | None = directory / f"{stem}.json" if save else None
        self.journal_path: Path | None = directory / f"{stem}.jsonl" if save else None
        self._journal = EvalJournal(self.journal_path) if self.journal_path else None
        self.trials: list[dict[str, Any]] = []
        self.report: EvalReport | None = None
        self.fps: int | None = None
        self.task: str | None = None
        self._policy_label = ""
        self._policy_privileged_state = False
        self.ended_holding = False

    def _journal_write(self, event: str, **payload: Any) -> None:
        if self._journal is None:
            return
        self._journal.write(event, **payload)

    def _record(self, trial: dict[str, Any]) -> None:
        self.trials.append(trial)
        self._journal_write("trial", **trial)

    def run(self) -> EvalReport:
        cfg = self.cfg
        phase, error = "done", ""
        pending: dict[str, Any] | None = None
        runner: PolicyRunner | None = None
        try:
            self.task = cfg.task.strip() or None
            policy = load_policy(cfg.model_path, task=self.task)
            self._policy_label = str(getattr(policy, "label", ""))
            self._policy_privileged_state = bool(getattr(policy, "privileged_state", False))
            defaults = checkpoint_defaults(cfg.model_path)
            self.task = self.task or defaults.task
            self.fps = resolve_control_fps(cfg.fps, defaults)
            runner = PolicyRunner(policy, fps=self.fps)
            self._journal_write(
                "start",
                model_path=cfg.model_path,
                requested_episodes=cfg.n_episodes,
                max_seconds=cfg.max_seconds,
                fps=self.fps,
                task=self.task,
                source=active_backend(),
            )
            runner.connect()
            for ep in range(cfg.n_episodes):
                if self.on_reset:
                    self.on_reset(ep)
                trial: dict[str, Any] = {"episode": ep + 1, "success": None, "steps": 0}
                pending = trial
                result = runner.rollout(duration=cfg.max_seconds)
                trial.update(steps=result.steps, duration=round(result.duration, 3))
                trial["success"] = bool(self.score_fn(ep)) if self.score_fn else True
                pending = None
                self._record(trial)
                log.info(
                    "eval ep %d/%d: %s",
                    ep + 1,
                    cfg.n_episodes,
                    "OK" if trial["success"] else "fail",
                )
        except BaseException as exc:
            aborted = isinstance(exc, (KeyboardInterrupt, EvalAborted))
            phase = "aborted" if aborted else "failed"
            error = str(exc) or type(exc).__name__
            if pending is not None:
                # Started but never scored: not a failure, not a success.
                pending["success"] = None
                pending["interrupted"] = error
                try:
                    self._record(pending)
                except EvalJournalError as journal_exc:
                    phase = "failed"
                    note = str(journal_exc)
                    error = f"{error}; {note}" if note not in error else error
                    raise journal_exc from exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            try:
                if runner is not None:
                    if phase == "done":
                        runner.halt_owned()
                        self.ended_holding = True
                    else:
                        runner.disconnect_owned()
            except BaseException as exc:
                cleanup_error = exc
                phase = "failed"
                note = f"policy teardown failed: {exc or type(exc).__name__}"
                if note not in error:
                    error = f"{error}; {note}" if error else note
            try:
                self.report = self._finish(phase, error)
            except BaseException as finish_exc:
                if cleanup_error is not None:
                    raise cleanup_error from finish_exc
                raise
            if cleanup_error is not None:
                raise cleanup_error
        return self.report

    def _report_label(self) -> tuple[str, bool]:
        if self._policy_label:
            return self._policy_label, self._policy_privileged_state
        from ..policy.checkpoint import inspect_checkpoint

        try:
            info = inspect_checkpoint(self.cfg.model_path)
        except Exception:  # descriptive only; never mask the run's own outcome
            return "", False
        return info.label, info.privileged_state

    def _finish(self, phase: str, error: str) -> EvalReport:
        label, privileged = self._report_label()
        report = EvalReport.from_trials(
            self.cfg.model_path,
            self.trials,
            requested_episodes=self.cfg.n_episodes,
            phase=phase,
            error=error,
            task=self.task,
            fps=self.fps,
            label=label,
            privileged_state=privileged,
            created_at=self.created_at,
        )
        if self._journal is None:
            return report
        if not self.trials and phase != "done":
            # Nothing ran: no report, and no journal of an eval that never was.
            self._journal.close()
            if self.journal_path is not None:
                self.journal_path.unlink(missing_ok=True)
            self.report_path = self.journal_path = None
            return report
        try:
            assert self.report_path is not None
            report.save(self.report_path)
        except (OSError, TypeError, ValueError) as exc:  # never mask the run's own outcome
            log.error("could not save eval report %s: %s", self.report_path, exc)
            self.report_path = None
            report.phase = "failed"
            note = f"report not saved: {exc}"
            if note not in report.error:
                report.error = f"{report.error}; {note}" if report.error else note
        journal_failure: EvalJournalError | None = None
        try:
            self._journal_write(
                "end",
                phase=report.phase,
                error=report.error,
                report=str(self.report_path) if self.report_path else None,
            )
        except EvalJournalError as exc:
            journal_failure = exc
            note = str(exc)
            report.phase = "failed"
            if note not in report.error:
                report.error = f"{report.error}; {note}" if report.error else note
            if self.report_path is not None:
                try:
                    report.save(self.report_path)
                except (OSError, TypeError, ValueError) as save_exc:
                    log.error("could not update eval report %s: %s", self.report_path, save_exc)
                    self.report_path = None
        finally:
            self._journal.close()
        if journal_failure is not None and phase == "done":
            # Let the CLI exit non-zero while keeping the failed report
            # available to its partial-result printer.
            self.report = report
            raise journal_failure
        return report


def evaluate(
    cfg: EvalConfig,
    score_fn: Callable[[int], bool] | None = None,
    on_reset: Callable[[int], None] | None = None,
) -> EvalReport:
    """Run ``cfg.n_episodes`` rollouts and return the report (not saved).

    ``score_fn(episode_idx) -> bool`` reports success per episode (defaults to
    auto-success in mock/headless use). ``on_reset(episode_idx)`` lets a UI
    prompt the operator to reset the scene between rollouts. The policy is
    reset at the start of every episode (audit B11). Use :class:`EvalRun`
    with ``save=True`` for a journaled, crash-safe eval.
    """

    return EvalRun(cfg, score_fn=score_fn, on_reset=on_reset).run()

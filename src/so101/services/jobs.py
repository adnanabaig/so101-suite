"""Runs training jobs in background threads and tracks their progress.

Each job streams ``(step, loss, line)`` from the trainer; we keep the latest
in the ``TrainingJob``, the latest progress-bar refresh in ``progress`` and a
bounded log tail for the UI.

Admission (audit B17):

* Real trainers run one at a time per kind: one ``lerobot-train`` process
  (it owns the GPU or MPS device) and one in-process numpy baseline. Later
  jobs wait with status ``queued`` and can be cancelled while waiting.
* A LeRobot job holds its dataset for reading for the whole run (the trainer
  reads it lazily). On POSIX the trainer inherits its own shared lock
  descriptor, so the exclusion survives even if the API/CLI parent is killed.
  Reads are shared, while recording into the dataset, deleting or restoring
  it is refused. The numpy baseline holds the dataset only while loading it.
* Job records are kept under ``<outputs>/.jobs/`` so the job list survives an
  API restart. Records fsync the file before rename and the directory after,
  so a power cut leaves the old record or the new one. A record that
  still cannot be read is listed with status ``corrupt``, keeping the output
  directory when it can be recovered; the damaged file is left in place.
* A ``lerobot-train`` job is recorded again as soon as its subprocess starts,
  with the trainer's pid, process group and OS start time. A job that was
  unfinished when the server stopped is restored as ``interrupted`` and is
  never restarted automatically. If its trainer outlived the server (SIGKILL,
  a crash) and is verifiably still that trainer, the restored runner holds the
  LeRobot slot, stops the trainer's process group (SIGTERM, then SIGKILL after
  ``orphan_grace_s``), and frees the slot only after stop is verified. A failed
  stop keeps the slot closed, so a new job never shares the GPU with it. A pid
  now used by another process is never signalled
  (:mod:`so101.training.processes`).
* Only the runner that holds ``.jobs/.lock`` submits or acts on unfinished
  records. A runner that cannot prove lock ownership remains read-only. The
  OS releases the lock when its process dies, however it dies, so a free lock
  means the server that wrote them is gone. A second server started on the
  same data folder (uvicorn runs the app's startup and shutdown even when it
  then fails to bind the port) lists another server's live jobs as they are
  and never stops their trainers.
* A status is written to the record before it is shown, so a caller that
  sees a job finish can rely on its record. If a write fails before replacement,
  the last durable non-terminal status remains visible. If replacement happened
  but its directory fsync failed, read-back plus a fresh directory fsync must
  prove durability; otherwise the terminal status stays hidden and the API
  reports degraded persistence. The record directory is fixed at submission,
  and the worker does nothing after final status appears.
"""

from __future__ import annotations

import errno
import json
import re
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import IO, Any

from ..atomic_io import atomic_write_json, durable_makedirs, fsync_directory
from ..config import get_settings
from ..data.dataset import dataset_read_hold, get_dataset
from ..logging import get_logger
from ..training import trainer as trainer_module
from ..training.jobs import JobStatus, TrainingJob
from ..training.preflight import resolve_preset
from ..training.processes import inspect_trainer, stop_process_group
from ..training.run_manifest import RUN_MANIFEST_NAME
from ..training.trainer import is_progress_line, run_training

log = get_logger(__name__)
_LOG_TAIL = 200
_TERMINAL = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.INTERRUPTED,
    JobStatus.CORRUPT,
}
_INTERRUPTED = "interrupted: the API server stopped before this job finished"
_ORPHANED = "orphaned: the trainer is still running after the API server stopped"
# Restored jobs whose trainer outlived the server, with its process group.
_Orphans = list[tuple[TrainingJob, int]]
_JOB_RECORDS = ".jobs"
_RECORDS_LOCK = ".lock"
# A JSON string member of a (possibly truncated) record: "key": "value".
_JSON_STRING = r'"{key}"\s*:\s*("(?:[^"\\]|\\.)*")'


class JobPersistenceError(RuntimeError):
    """A training-job state could not be made durable."""

    def __init__(self, message: str, *, replacement_visible: bool = False) -> None:
        super().__init__(message)
        self.replacement_visible = replacement_visible


class JobStartError(RuntimeError):
    """The durable pending job exists, but its worker thread could not start."""


class JobRunner:
    # A trainer left running by a previous server gets this long to exit on
    # SIGTERM before SIGKILL, then this long to disappear.
    orphan_grace_s: float = 10.0
    orphan_kill_wait_s: float = 5.0

    def __init__(self, *, restore: bool = False) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._record_dirs: dict[str, Path] = {}
        self._lock = threading.Lock()
        # Initial persistence + Thread.start is one admission transaction with
        # shutdown. list/get still use _lock and remain responsive while the
        # initial fsync is in progress.
        self._admission_lock = threading.Lock()
        # One worker per trainer kind; mock (synthetic) jobs are not queued.
        self._slots = {
            "lerobot": threading.Semaphore(1),
            "numpy_in_process": threading.Semaphore(1),
        }
        self._records_lock: IO[bytes] | None = None
        self._records_lock_attempted = False
        self._records_lock_error = "job records ownership has not been acquired"
        self._closed = False
        if restore:
            self._restore()

    def submit(
        self,
        repo_id: str,
        policy: str,
        steps: int,
        batch_size: int,
        extra_args: dict[str, str] | None = None,
        include_depth: bool = False,
        episodes: str = "kept",
    ) -> TrainingJob:
        with self._admission_lock:
            return self._submit_admitted(
                repo_id,
                policy,
                steps,
                batch_size,
                extra_args,
                include_depth,
                episodes,
            )

    def _submit_admitted(
        self,
        repo_id: str,
        policy: str,
        steps: int,
        batch_size: int,
        extra_args: dict[str, str] | None,
        include_depth: bool,
        episodes: str,
    ) -> TrainingJob:
        """Submit while exclusion with shutdown is already held."""

        self._require_submission_owner()
        job = TrainingJob(
            repo_id=repo_id,
            policy=policy,
            steps=steps,
            batch_size=batch_size,
            output_dir="",
            extra_args=extra_args or {},
            include_depth=include_depth,
            episodes=episodes,
        )
        outputs = get_settings().outputs_dir
        job.output_dir = str(outputs / f"{policy}_{time.time_ns()}_{job.id}")
        record_dir = outputs / _JOB_RECORDS
        cancel = threading.Event()
        thread = threading.Thread(target=self._run, args=(job.id,), daemon=True)
        # Persistence may block. Keep the staged job out of every shared map so
        # list/get cannot expose it before its initial record is durable.
        self._persist(job, directory=record_dir)
        with self._lock:
            self._jobs[job.id] = job
            self._record_dirs[job.id] = record_dir
            self._cancel[job.id] = cancel
            self._threads[job.id] = thread
            try:
                thread.start()
            except BaseException as start_error:
                self._threads.pop(job.id, None)
                cancel.set()
                job.finished_at = time.time()
                job.error = (
                    f"worker thread failed to start: {type(start_error).__name__}: {start_error}"
                )
                try:
                    self._publish(job, JobStatus.FAILED)
                except BaseException:
                    self._cancel.pop(job.id, None)
                    self._record_dirs.pop(job.id, None)
                    self._jobs.pop(job.id, None)
                    raise
                if job.status is not JobStatus.FAILED:
                    detail = job.error or "worker start failure was not durably recorded"
                    self._cancel.pop(job.id, None)
                    self._record_dirs.pop(job.id, None)
                    self._jobs.pop(job.id, None)
                    raise JobPersistenceError(detail) from start_error
                if isinstance(start_error, Exception):
                    raise JobStartError(
                        f"training job {job.id}: {job.error or str(start_error)}"
                    ) from start_error
                raise
        return job

    def _slot_for(self, job: TrainingJob) -> threading.Semaphore | None:
        preset = resolve_preset(job.policy)
        if preset.trainer == "numpy_in_process":
            return self._slots["numpy_in_process"]
        if preset.trainer == "lerobot" and not get_settings().mock:
            return self._slots["lerobot"]
        return None

    def _wait_for_slot(self, job: TrainingJob, slot: threading.Semaphore) -> bool:
        """Block until ``slot`` is free; False when the job is cancelled while queued."""

        if slot.acquire(blocking=False):
            return True
        self._publish(job, JobStatus.QUEUED)
        cancel = self._cancel[job.id]
        while not slot.acquire(timeout=0.25):
            if cancel.is_set():
                return False
        return True

    def _run(self, job_id: str) -> None:
        job = self._jobs[job_id]
        cancel = self._cancel[job_id]
        slot: threading.Semaphore | None = None
        slot_held = False
        outcome: JobStatus | None = None
        try:
            preset = resolve_preset(job.policy)
            slot = self._slot_for(job)
            if slot is not None:
                slot_held = self._wait_for_slot(job, slot)
                if not slot_held:
                    outcome = JobStatus.CANCELLED
                    return
            job.started_at = time.time()
            self._publish(job, JobStatus.RUNNING)
            # LeRobot reads the dataset throughout training; the numpy
            # baseline takes its own hold while it loads the data.
            hold_dataset = preset.trainer == "lerobot" and get_dataset(job.repo_id) is not None
            with dataset_read_hold(job.repo_id) if hold_dataset else nullcontext():
                for step, loss, line in run_training(
                    job, cancel_event=cancel, on_start=self._persist
                ):
                    if cancel.is_set():
                        break
                    job.step = step
                    job.loss = None if loss != loss else loss  # NaN guard
                    if is_progress_line(line):
                        job.progress = line.strip()
                        continue
                    job.log_tail.append(line)
                    if len(job.log_tail) > _LOG_TAIL:
                        job.log_tail = job.log_tail[-_LOG_TAIL:]
            outcome = JobStatus.CANCELLED if cancel.is_set() else JobStatus.SUCCEEDED
        except Exception as exc:  # pragma: no cover - depends on trainer
            if cancel.is_set():
                outcome = JobStatus.CANCELLED
            else:
                outcome = JobStatus.FAILED
                job.error = str(exc)
                log.exception("job %s failed", job_id)
        finally:
            if slot is not None and slot_held:
                slot.release()
            job.finished_at = time.time()
            if outcome is None:  # something other than an Exception escaped
                self._persist(job)
            else:
                # The final status appears last: a caller that waits for it
                # finds the record written, and this worker does nothing after.
                self._publish(job, outcome)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ev = self._cancel.get(job_id)
            if not ev:
                return False
            if not self._owns_records_lock():
                detail = self._records_lock_error or "job-record lock ownership was lost"
                raise JobPersistenceError(
                    f"training cancellation refused because this runner is read-only: {detail}"
                )
            ev.set()
        return True

    def get(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TrainingJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel and join every owned trainer before the API process exits."""

        with self._admission_lock:
            self._shutdown_admitted(timeout)

    def _shutdown_admitted(self, timeout: float) -> None:
        """Shut down after every earlier admission has started or failed."""

        with self._lock:
            self._closed = True
            active = [
                (job_id, thread) for job_id, thread in self._threads.items() if thread.is_alive()
            ]
            for job_id, _thread in active:
                cancel = self._cancel.get(job_id)
                if cancel is not None:
                    # Internal teardown owns these local worker objects. It
                    # must remain idempotent after the public mutation lock is
                    # closed, while public/non-owner cancel still fails closed.
                    cancel.set()
            threads = [thread for _job_id, thread in active]
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            self._records_lock_error = (
                "shutdown is retaining job-record ownership because training workers "
                f"are still alive: {alive}"
            )
            log.critical("%s", self._records_lock_error)
            return
        if self._records_lock is not None:
            self._records_lock.close()
            self._records_lock = None
        self._records_lock_attempted = True
        self._records_lock_error = "the job runner has shut down"

    # --- durable job records ----------------------------------------------
    @staticmethod
    def _records_dir() -> Path:
        return get_settings().outputs_dir / _JOB_RECORDS

    def _claim_records_lock(self, directory: Path) -> None:
        """Try once to become the sole writer of ``directory``.

        A failed attempt is sticky.  Retrying later without restoring records
        could admit work beside an unfinished trainer whose owner disappeared
        in between, so this runner stays read-only for its lifetime.
        """

        self._records_lock_attempted = True
        try:
            # Persist the directory entry in outputs before an initial job
            # record can be acknowledged.
            durable_makedirs(directory)
            handle = _lock_records(directory)
        except OSError as exc:
            self._records_lock_error = (
                f"could not acquire {directory / _RECORDS_LOCK}: {type(exc).__name__}: {exc}"
            )
            log.warning("%s", self._records_lock_error)
            return
        if handle is None:
            self._records_lock_error = f"another so101 process owns {directory / _RECORDS_LOCK}"
            return
        self._records_lock = handle
        self._records_lock_error = ""

    def _require_submission_owner(self) -> None:
        """Refuse mutation unless this live runner owns ``.jobs/.lock``."""

        with self._lock:
            if self._closed:
                raise JobPersistenceError(
                    "training submission refused: the job runner is shut down"
                )
            if self._owns_records_lock():
                return
            if not self._records_lock_attempted:
                self._claim_records_lock(self._records_dir())
            if self._owns_records_lock():
                return
            detail = self._records_lock_error or "job-record lock ownership was lost"
            raise JobPersistenceError(
                f"training submission refused because this runner is read-only: {detail}"
            )

    def _owns_records_lock(self) -> bool:
        handle = self._records_lock
        return handle is not None and not handle.closed

    def _publish(self, job: TrainingJob, status: JobStatus) -> bool:
        """Record ``status``, then show it, so a status anyone can see is on disk."""

        try:
            self._persist(job, status=status)
        except JobPersistenceError as exc:
            if status not in _TERMINAL:
                raise
            if exc.replacement_visible:
                detail = (
                    f"terminal status {status.value} was visible on disk but remains hidden "
                    f"because durability could not be confirmed: {exc}"
                )
            else:
                detail = f"could not durably publish terminal status {status.value}: {exc}"
            job.error = f"{job.error}; {detail}" if job.error else detail
            job.progress = detail
            log.error("job %s: %s", job.id, detail)
            return False
        job.status = status
        return True

    def _persist(
        self,
        job: TrainingJob,
        *,
        status: JobStatus | None = None,
        directory: Path | None = None,
    ) -> None:
        """Write the job record durably, raising when durability is not proven.

        ``atomic_write_json`` writes a temp file, fsyncs it, renames it over
        the record and fsyncs the directory, so a power cut leaves either the
        previous record or this one, never a truncated file.
        ``status`` is recorded in place of ``job.status`` (see ``_publish``).
        The directory is the one fixed at submit or restore time, so a worker
        never reads settings that may have changed since.
        """

        path: Path | None = None
        record: dict[str, Any] | None = None
        try:
            destination_dir = directory if directory is not None else self._record_dirs[job.id]
            path = destination_dir / f"{job.id}.json"
            record = job.to_dict()
            if status is not None:
                record["status"] = status.value
            atomic_write_json(path, record)
        except Exception as exc:
            destination = str(path) if path is not None else "its assigned record"
            observed: object = None
            if path is not None:
                try:
                    observed = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    pass
            replacement_visible = record is not None and observed == record
            confirmation_error: Exception | None = None
            if replacement_visible and path is not None:
                try:
                    # A directory-fsync failure happens after rename. Read-back
                    # proves visibility only; a successful explicit directory
                    # fsync is what makes terminal exposure safe.
                    fsync_directory(path.parent)
                except Exception as confirm_error:
                    confirmation_error = confirm_error
                else:
                    return
            visibility = (
                "; the replacement is visible but directory durability is unconfirmed"
                f" ({type(confirmation_error).__name__}: {confirmation_error})"
                if replacement_visible
                else ""
            )
            raise JobPersistenceError(
                f"could not persist training job {job.id} to {destination}: "
                f"{type(exc).__name__}: {exc}{visibility}",
                replacement_visible=replacement_visible,
            ) from exc

    def _restore(self) -> None:
        directory = self._records_dir()
        # Created even on a fresh install: this runner tries to hold the lock
        # from startup, before it can write the first running record.
        self._claim_records_lock(directory)
        if not directory.is_dir():
            return
        owner = self._records_lock is not None
        if not owner:
            log.warning(
                "%s; unfinished jobs are listed read-only and submissions are refused",
                self._records_lock_error,
            )
        known = {item.name for item in fields(TrainingJob)}
        orphans: _Orphans = []
        uncertain: list[TrainingJob] = []
        for path in sorted(directory.glob("*.json")):
            try:
                payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
                record = {key: value for key, value in payload.items() if key in known}
                record["status"] = JobStatus(record.get("status", JobStatus.FAILED.value))
                job = TrainingJob(**record)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                # Listed, not dropped: the operator sees that a job existed and
                # where its output went. The damaged file is left as it is.
                log.warning("unreadable job record %s: %s", path, exc)
                corrupt = _corrupt_job(path, exc)
                self._jobs[corrupt.id] = corrupt
                done = threading.Event()
                done.set()
                self._cancel[corrupt.id] = done
                continue
            self._record_dirs[job.id] = directory
            done = threading.Event()
            done.set()
            self._cancel[job.id] = done
            if owner and job.status not in _TERMINAL:
                check = inspect_trainer(job, module=trainer_module.TRAIN_MODULE)
                if check.state == "ours" and check.pgid is not None:
                    if job.pid is None and check.pid is not None:
                        # The parent died after Popen but before its on_start
                        # callback persisted the identity. Keep what the
                        # process-table recovery proved for the next restart.
                        job.pid = check.pid
                        job.pgid = check.pgid
                        job.pid_start = check.start
                        log.warning("job %s: %s", job.id, check.detail)
                    orphans.append((job, check.pgid))
                elif check.state == "unknown":
                    detail = check.detail or "its trainer could not be inspected"
                    log.error("job %s: %s", job.id, detail)
                    job.error = f"{_ORPHANED}; {detail}"
                    job.finished_at = None
                    job.progress = "Training remains locked until this trainer is verified gone"
                    self._publish(job, JobStatus.RUNNING)
                    uncertain.append(job)
                else:
                    if check.state == "other":
                        log.warning("job %s: %s", job.id, check.detail)
                    self._interrupt(job, check.detail)
            self._jobs[job.id] = job
        if orphans:
            self._stop_orphans_in_background(orphans, keep_slot_closed=bool(uncertain))
        elif uncertain:
            self._hold_unverified_trainers(uncertain)

    def _interrupt(self, job: TrainingJob, detail: str = "") -> None:
        job.error = f"{_INTERRUPTED}; {detail}" if detail else _INTERRUPTED
        job.finished_at = job.finished_at or time.time()
        self._publish(job, JobStatus.INTERRUPTED)

    def _hold_unverified_trainers(self, jobs: Sequence[TrainingJob]) -> None:
        """Keep the GPU slot closed when process inspection was inconclusive."""

        self._slots["lerobot"].acquire()  # a new runner's slot is free
        log.critical(
            "keeping the LeRobot slot closed: trainer ownership is unknown for job(s) %s",
            ", ".join(job.id for job in jobs),
        )

    def _stop_orphans_in_background(
        self, orphans: _Orphans, *, keep_slot_closed: bool = False
    ) -> None:
        """Hold the LeRobot slot until every trainer a previous server left is gone.

        Runs while the runner is built, so no job can be submitted before the
        slot is taken. The jobs show as running, with a note in ``progress``,
        until their trainer has exited; cancelling one (or shutting down) skips
        the rest of the SIGTERM grace period. Failure to verify every stop
        leaves the slot acquired and the record non-terminal, so the next
        process restart rechecks it before opening a fresh slot.
        """

        slot = self._slots["lerobot"]
        slot.acquire()  # a new runner's slot is free
        hurry = threading.Event()
        thread = threading.Thread(
            target=self._stop_orphans,
            args=(orphans, slot, hurry, keep_slot_closed),
            name="so101-stop-orphan-trainer",
            daemon=True,
        )
        for job, pgid in orphans:
            log.warning(
                "job %s: its trainer (pid %s, process group %s) outlived the previous "
                "server; stopping it",
                job.id,
                job.pid,
                pgid,
            )
            job.progress = (
                f"Stopping the trainer the previous server left running (process group {pgid})"
            )
            self._cancel[job.id] = hurry
            self._threads[job.id] = thread
        thread.start()

    def _stop_orphans(
        self,
        orphans: _Orphans,
        slot: threading.Semaphore,
        hurry: threading.Event,
        keep_slot_closed: bool = False,
    ) -> None:
        all_stopped = True
        completed = False
        try:
            for job, pgid in orphans:
                stopped = stop_process_group(
                    pgid,
                    grace_s=self.orphan_grace_s,
                    kill_wait_s=self.orphan_kill_wait_s,
                    hurry=hurry,
                )
                all_stopped = all_stopped and stopped
                job.progress = None
                if stopped:
                    detail = (
                        f"its trainer (process group {pgid}) was still running and was "
                        "stopped when the server started again"
                    )
                else:
                    detail = (
                        f"its trainer (process group {pgid}) could not be stopped; end it "
                        f"(kill -KILL -{pgid}) before training again"
                    )
                    log.error("job %s: %s", job.id, detail)
                if stopped:
                    self._interrupt(job, detail)
                else:
                    # Keep this non-terminal on disk. The next server must
                    # inspect and try to stop the same process again before it
                    # can create a fresh LeRobot slot.
                    job.error = f"{_ORPHANED}; {detail}"
                    job.finished_at = None
                    job.progress = "Training remains locked until this trainer is verified gone"
                    self._publish(job, JobStatus.RUNNING)
            completed = True
        finally:
            if completed and all_stopped and not keep_slot_closed:
                slot.release()
            else:
                log.critical(
                    "keeping the LeRobot slot closed because an orphan trainer is not "
                    "verified stopped"
                )


def _lock_records(directory: Path) -> IO[bytes] | None:
    """Take ``<records>/.lock`` exclusively without waiting; the open file, or None.

    ``flock`` on POSIX and ``msvcrt.locking`` on Windows: both locks belong
    to the open file and end with the process. ``None`` means verified lock
    contention; open failures and lock-service errors such as ``ENOLCK``
    propagate so callers can remain explicitly read-only.
    """

    handle = open(directory / _RECORDS_LOCK, "a+b")  # held open: the lock lives with it
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        held = {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK}
        if exc.errno in held:
            return None
        raise
    return handle


def _salvage_text(text: str, key: str) -> str | None:
    """The string member ``key`` of a (possibly truncated) JSON record, if present whole."""

    match = re.search(_JSON_STRING.format(key=re.escape(key)), text)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except ValueError:
        return None
    return value if isinstance(value, str) and value else None


def _corrupt_job(path: Path, exc: BaseException) -> TrainingJob:
    """A ``corrupt`` entry for an unreadable record, with what can be recovered.

    The id is the file name. The output directory comes from the readable part
    of the record, else from the run directory named ``<policy>_<ns>_<id>``
    beside ``.jobs/``; its run manifest then supplies the dataset.
    """

    job_id = path.stem
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        text = ""
    output_dir = _salvage_text(text, "output_dir")
    if output_dir is None:
        runs = sorted(p for p in path.parent.parent.glob(f"*_{job_id}") if p.is_dir())
        output_dir = str(runs[0]) if len(runs) == 1 else None
    policy = _salvage_text(text, "policy")
    repo_id = _salvage_text(text, "repo_id")
    if output_dir is not None:
        if policy is None and Path(output_dir).name.endswith(f"_{job_id}"):
            policy = Path(output_dir).name.rsplit("_", 2)[0] or None
        if repo_id is None:
            try:
                manifest = json.loads(
                    (Path(output_dir) / RUN_MANIFEST_NAME).read_text(encoding="utf-8")
                )
                found = manifest["dataset"]["repo_id"]
                repo_id = found if isinstance(found, str) and found else None
            except (OSError, ValueError, KeyError, TypeError):
                pass
    try:
        created = path.stat().st_mtime
    except OSError:
        created = time.time()
    where = (
        f"its output directory is {output_dir}"
        if output_dir
        else "no output directory could be recovered"
    )
    return TrainingJob(
        id=job_id,
        repo_id=repo_id or "unknown",
        policy=policy or "unknown",
        steps=0,
        batch_size=0,
        output_dir=output_dir or "",
        status=JobStatus.CORRUPT,
        created_at=created,
        error=(
            f"corrupt job record {path}: {type(exc).__name__}: {exc}. The job's outcome is "
            f"unknown; {where}. Delete the record to remove this entry."
        ),
    )


_runner: JobRunner | None = None
_runner_guard = threading.Lock()


def get_runner() -> JobRunner:
    global _runner
    with _runner_guard:
        if _runner is None or _runner._closed:
            _runner = JobRunner(restore=True)
        return _runner

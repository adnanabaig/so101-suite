"""Training-job durability: records that survive power cuts and server crashes."""

from __future__ import annotations

import errno
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from so101.config import get_settings
from so101.services import jobs as jobs_module
from so101.services.app import create_app
from so101.services.jobs import JobPersistenceError, JobRunner
from so101.training.jobs import JobStatus, TrainingJob
from so101.training.processes import ProcessInfo, TrainerCheck, process_info, process_start_marker
from so101.training.run_manifest import RUN_MANIFEST_NAME


def _fsync_spy(monkeypatch) -> list[str]:
    """Record each fsync as "file" or "dir" (what the fd refers to)."""

    calls: list[str] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        calls.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return calls


def _records_dir():
    return get_settings().outputs_dir / ".jobs"


# --- finding 1: job records are fsynced; an unreadable one is listed --------


def test_job_record_is_fsynced_file_and_directory(monkeypatch):
    runner = JobRunner()
    job = TrainingJob(repo_id="x/y", policy="act", steps=1, batch_size=1, output_dir="o")
    runner._record_dirs[job.id] = _records_dir()
    calls = _fsync_spy(monkeypatch)
    runner._persist(job)
    assert "file" in calls and "dir" in calls, calls
    record = json.loads((_records_dir() / f"{job.id}.json").read_text(encoding="utf-8"))
    assert record["id"] == job.id and record["status"] == "pending"
    assert [p.name for p in _records_dir().iterdir()] == [f"{job.id}.json"]  # no temp left


def test_new_job_records_container_is_parent_fsynced(monkeypatch):
    from so101 import atomic_io

    outputs = get_settings().outputs_dir
    calls: list[Path] = []
    real_fsync_directory = atomic_io.fsync_directory

    def spy(path: Path) -> None:
        calls.append(Path(path))
        real_fsync_directory(path)

    monkeypatch.setattr(atomic_io, "fsync_directory", spy)
    runner = JobRunner(restore=True)
    try:
        assert outputs in calls, ".jobs creation must persist its entry in outputs_dir"
    finally:
        runner.shutdown()


def test_terminal_status_is_not_exposed_when_its_durable_write_fails(monkeypatch):
    runner = JobRunner()
    job = TrainingJob(
        repo_id="x/y",
        policy="bc_baseline",
        steps=1,
        batch_size=1,
        output_dir="o",
        status=JobStatus.RUNNING,
    )
    runner._jobs[job.id] = job
    runner._record_dirs[job.id] = _records_dir()
    runner._persist(job)
    write_json = jobs_module.atomic_write_json

    def fail_terminal(path, payload):
        if payload["status"] == "succeeded":
            raise OSError("injected terminal record EIO")
        return write_json(path, payload)

    monkeypatch.setattr(jobs_module, "atomic_write_json", fail_terminal)
    assert runner._publish(job, JobStatus.SUCCEEDED) is False

    assert job.status is JobStatus.RUNNING
    assert "could not durably publish terminal status succeeded" in (job.error or "")
    assert "terminal record EIO" in (job.error or "")
    assert _record_on_disk(job.id)["status"] == "running"


def test_terminal_replace_then_error_requires_directory_fsync_before_exposure(monkeypatch):
    runner = JobRunner()
    job = TrainingJob(
        repo_id="x/y",
        policy="bc_baseline",
        steps=1,
        batch_size=1,
        output_dir="o",
        status=JobStatus.RUNNING,
    )
    runner._record_dirs[job.id] = _records_dir()
    runner._persist(job)
    write_json = jobs_module.atomic_write_json

    def replace_then_fail(path, payload):
        write_json(path, payload)
        raise OSError("injected post-replace directory fsync EIO")

    monkeypatch.setattr(jobs_module, "atomic_write_json", replace_then_fail)
    assert runner._publish(job, JobStatus.SUCCEEDED) is True

    assert job.status is JobStatus.SUCCEEDED
    assert job.error is None
    assert _record_on_disk(job.id)["status"] == "succeeded"


def test_terminal_stays_hidden_when_post_replace_directory_fsync_still_fails(monkeypatch):
    runner = JobRunner()
    job = TrainingJob(
        repo_id="x/y",
        policy="bc_baseline",
        steps=1,
        batch_size=1,
        output_dir="o",
        status=JobStatus.RUNNING,
    )
    runner._jobs[job.id] = job
    runner._record_dirs[job.id] = _records_dir()
    runner._persist(job)
    write_json = jobs_module.atomic_write_json

    def replace_then_fail(path, payload):
        write_json(path, payload)
        raise OSError("injected post-replace directory fsync EIO")

    def fail_confirmation(_path):
        raise OSError("injected directory fsync confirmation EIO")

    monkeypatch.setattr(jobs_module, "atomic_write_json", replace_then_fail)
    monkeypatch.setattr(jobs_module, "fsync_directory", fail_confirmation)
    assert runner._publish(job, JobStatus.SUCCEEDED) is False

    assert job.status is JobStatus.RUNNING
    assert "remains hidden because durability could not be confirmed" in (job.error or "")
    assert "directory fsync confirmation EIO" in (job.error or "")
    assert _record_on_disk(job.id)["status"] == "succeeded"
    assert runner.get(job.id) is job and runner.get(job.id).status is JobStatus.RUNNING


def test_terminal_persistence_failure_still_releases_the_worker_slot(monkeypatch):
    def one_step(_job, **_kwargs):
        yield 1, 0.1, "done"

    write_json = jobs_module.atomic_write_json

    def fail_terminal(path, payload):
        if payload["status"] == "succeeded":
            raise OSError("injected terminal record EIO")
        return write_json(path, payload)

    monkeypatch.setattr(jobs_module, "run_training", one_step)
    monkeypatch.setattr(jobs_module, "atomic_write_json", fail_terminal)
    runner = JobRunner()
    job = runner.submit("x/y", "bc_baseline", 1, 1)
    runner._threads[job.id].join(timeout=5)

    assert not runner._threads[job.id].is_alive()
    assert job.status is JobStatus.RUNNING
    assert "terminal record EIO" in (job.error or "")
    slot = runner._slots["numpy_in_process"]
    assert slot.acquire(blocking=False), "the failed terminal write must not leak the slot"
    slot.release()
    runner.shutdown()


def test_initial_job_persistence_failure_starts_no_worker_or_phantom_job(monkeypatch):
    def fail_write(_path, _payload):
        raise OSError("injected initial record EIO")

    monkeypatch.setattr(jobs_module, "atomic_write_json", fail_write)
    runner = JobRunner()
    with pytest.raises(RuntimeError, match="initial record EIO"):
        runner.submit("x/y", "bc_baseline", 1, 1)

    assert runner.list() == []
    assert runner._threads == {}
    assert runner._cancel == {}
    assert runner._record_dirs == {}


def test_nonowner_runner_lists_records_but_refuses_mutation(monkeypatch):
    from so101.services.routers import training as training_router

    existing = TrainingJob(
        repo_id="x/old",
        policy="act",
        steps=1,
        batch_size=1,
        output_dir="old",
        status=JobStatus.SUCCEEDED,
    )
    _write_record(existing)
    owner = JobRunner(restore=True)
    observer = JobRunner(restore=True)
    try:
        assert observer.get(existing.id) is not None
        with pytest.raises(JobPersistenceError, match="read-only.*another so101 process"):
            observer.submit("x/new", "act", 1, 1)
        with pytest.raises(JobPersistenceError, match="cancellation refused.*read-only"):
            observer.cancel(existing.id)
        monkeypatch.setattr(training_router, "get_runner", lambda: observer)
        with pytest.raises(HTTPException) as refused:
            training_router.cancel(existing.id)
        assert refused.value.status_code == 503
        assert "cancellation refused" in str(refused.value.detail)
        assert observer._threads == {}
        assert sorted(path.stem for path in _records_dir().glob("*.json")) == [existing.id]
    finally:
        observer.shutdown()
        owner.shutdown()


@pytest.mark.parametrize(
    "lock_error",
    [
        PermissionError(errno.EACCES, "injected lock open failure"),
        OSError(errno.ENOLCK, "injected lock service failure"),
    ],
)
def test_job_lock_operational_failure_leaves_runner_read_only(monkeypatch, lock_error):
    def fail_lock(_directory):
        raise lock_error

    monkeypatch.setattr(jobs_module, "_lock_records", fail_lock)
    runner = JobRunner(restore=True)
    with pytest.raises(JobPersistenceError, match="read-only.*injected lock"):
        runner.submit("x/new", "act", 1, 1)
    assert runner.list() == [] and runner._threads == {}
    runner.shutdown()


def test_job_lock_enolck_is_not_misreported_as_contention(monkeypatch):
    _records_dir().mkdir(parents=True)

    def no_lock_service(*_args):
        raise OSError(errno.ENOLCK, "injected lock service failure")

    if sys.platform == "win32":
        import msvcrt

        monkeypatch.setattr(msvcrt, "locking", no_lock_service)
    else:
        import fcntl

        monkeypatch.setattr(fcntl, "flock", no_lock_service)
    with pytest.raises(OSError, match="lock service failure"):
        jobs_module._lock_records(_records_dir())


def test_submitting_job_is_invisible_until_initial_record_is_durable(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    submitted: list[TrainingJob] = []
    errors: list[BaseException] = []
    pending_id: list[str] = []
    write_json = jobs_module.atomic_write_json

    def blocked_write(path, payload):
        if not entered.is_set():
            pending_id.append(payload["id"])
            entered.set()
            assert release.wait(5), "test did not release the initial record write"
        return write_json(path, payload)

    monkeypatch.setattr(jobs_module, "atomic_write_json", blocked_write)
    monkeypatch.setattr(
        jobs_module,
        "run_training",
        lambda _job, **_kwargs: iter([(1, 0.1, "done")]),
    )
    runner = JobRunner()

    def submit() -> None:
        try:
            submitted.append(runner.submit("x/y", "bc_baseline", 1, 1))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=submit)
    thread.start()
    assert entered.wait(5), "initial record write did not block"
    assert runner.list() == []
    assert pending_id and runner.get(pending_id[0]) is None
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive() and not errors
    assert submitted and runner.get(submitted[0].id) is submitted[0]
    runner.shutdown()


def test_shutdown_serializes_with_blocked_initial_job_admission(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    shutdown_attempted = threading.Event()
    shutdown_done = threading.Event()
    submitted: list[TrainingJob] = []
    errors: list[BaseException] = []
    write_json = jobs_module.atomic_write_json
    first_write = True

    def blocked_initial_write(path, payload):
        nonlocal first_write
        if first_write:
            first_write = False
            entered.set()
            assert release.wait(5), "test did not release initial persistence"
        return write_json(path, payload)

    def wait_for_cancel(_job, *, cancel_event, **_kwargs):
        assert cancel_event is not None
        cancel_event.wait(5)
        if False:  # make this a generator without yielding a training step
            yield 0, 0.0, ""

    runner = JobRunner()
    monkeypatch.setattr(jobs_module, "atomic_write_json", blocked_initial_write)
    monkeypatch.setattr(jobs_module, "run_training", wait_for_cancel)

    def submit() -> None:
        try:
            submitted.append(runner.submit("x/y", "bc_baseline", 1, 1))
        except BaseException as exc:
            errors.append(exc)

    def shutdown() -> None:
        shutdown_attempted.set()
        runner.shutdown(timeout=5)
        shutdown_done.set()

    submit_thread = threading.Thread(target=submit)
    shutdown_thread: threading.Thread | None = None
    submit_thread.start()
    try:
        assert entered.wait(5), "initial persistence did not block"
        shutdown_thread = threading.Thread(target=shutdown)
        shutdown_thread.start()
        assert shutdown_attempted.wait(5)
        assert not shutdown_done.wait(0.1)
        assert runner.list() == []
    finally:
        release.set()
        submit_thread.join(timeout=5)
        if shutdown_thread is not None:
            shutdown_thread.join(timeout=5)
    assert shutdown_thread is not None
    assert not submit_thread.is_alive() and not shutdown_thread.is_alive()
    assert not errors and len(submitted) == 1
    job = submitted[0]
    assert job.status is JobStatus.CANCELLED
    assert _record_on_disk(job.id)["status"] == "cancelled"


def test_shutdown_retains_job_records_lock_while_worker_survives():
    runner = JobRunner(restore=True)
    release = threading.Event()
    worker = threading.Thread(target=release.wait, name="injected-stuck-worker", daemon=True)
    runner._threads["stuck"] = worker
    runner._cancel["stuck"] = threading.Event()
    worker.start()
    contender: JobRunner | None = None
    try:
        runner.shutdown(timeout=0)
        assert worker.is_alive()
        assert runner._owns_records_lock()

        contender = JobRunner(restore=True)
        assert contender._records_lock is None
        with pytest.raises(JobPersistenceError, match="read-only.*another so101 process"):
            contender.submit("x/new", "act", 1, 1)
    finally:
        release.set()
        worker.join(timeout=5)
        runner.shutdown(timeout=1)
        if contender is not None:
            contender.shutdown(timeout=1)

    assert not runner._owns_records_lock()
    handle = jobs_module._lock_records(_records_dir())
    assert handle is not None
    handle.close()


def test_shutdown_is_idempotent_after_completed_worker_and_closed_lock():
    runner = JobRunner(restore=True)
    worker = threading.Thread(target=lambda: None)
    runner._threads["done"] = worker
    runner._cancel["done"] = threading.Event()
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()

    runner.shutdown(timeout=1)
    assert runner._records_lock is None
    runner.shutdown(timeout=1)
    assert runner._records_lock is None


def test_thread_start_failure_is_durably_failed_not_forever_pending(monkeypatch):
    def fail_start(_thread):
        raise RuntimeError("injected thread start failure")

    runner = JobRunner()
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_start)
        with pytest.raises(RuntimeError, match="thread start failure"):
            runner.submit("x/y", "bc_baseline", 1, 1)

    [job] = runner.list()
    assert job.status is JobStatus.FAILED
    assert "worker thread failed to start" in (job.error or "")
    assert _record_on_disk(job.id)["status"] == "failed"
    assert runner._threads == {}


def _write_record(job: TrainingJob) -> None:
    _records_dir().mkdir(parents=True, exist_ok=True)
    (_records_dir() / f"{job.id}.json").write_text(json.dumps(job.to_dict()), encoding="utf-8")


def test_truncated_record_is_listed_as_corrupt_with_its_output_dir():
    outputs = get_settings().outputs_dir
    job = TrainingJob(
        repo_id="user/pick",
        policy="act",
        steps=100,
        batch_size=8,
        output_dir=str(outputs / "act_1_abc123def456"),
        id="abc123def456",
        status=JobStatus.SUCCEEDED,
    )
    _write_record(job)
    path = _records_dir() / f"{job.id}.json"
    text = json.dumps(job.to_dict())
    cut = text.index('"pid"')  # after output_dir, before the end
    path.write_text(text[:cut], encoding="utf-8")

    restored = JobRunner(restore=True).get(job.id)
    assert restored is not None, "a truncated record must not make the job disappear"
    assert restored.status is JobStatus.CORRUPT
    assert restored.output_dir == job.output_dir
    assert restored.repo_id == "user/pick" and restored.policy == "act"
    assert "corrupt job record" in (restored.error or "")
    assert path.read_text(encoding="utf-8") == text[:cut]  # left as it was


def test_empty_record_recovers_the_run_directory_and_shows_in_the_api(monkeypatch):
    """A never-synced new file reads back empty after a power cut (delayed allocation)."""

    outputs = get_settings().outputs_dir
    run = outputs / "bc_baseline_1759000000000000000_0123456789ab"
    run.mkdir(parents=True)
    (run / RUN_MANIFEST_NAME).write_text(
        json.dumps({"dataset": {"repo_id": "user/reach"}}), encoding="utf-8"
    )
    _records_dir().mkdir(parents=True)
    (_records_dir() / "0123456789ab.json").write_bytes(b"")
    (_records_dir() / "fedcba987654.json").write_text("[1, 2", encoding="utf-8")

    monkeypatch.setattr(jobs_module, "_runner", None)  # the API restores from this test's dir
    with TestClient(create_app()) as client:
        listed = {job["id"]: job for job in client.get("/api/training/jobs").json()}
        one = client.get("/api/training/jobs/0123456789ab").json()
    assert listed["0123456789ab"]["status"] == "corrupt"
    assert one["output_dir"] == str(run)
    assert one["policy"] == "bc_baseline" and one["repo_id"] == "user/reach"
    assert "Delete the record" in one["error"]
    nothing = listed["fedcba987654"]
    assert nothing["status"] == "corrupt" and nothing["output_dir"] == ""
    assert "no output directory could be recovered" in nothing["error"]


def test_corrupt_job_cannot_be_cancelled_into_another_state():
    _records_dir().mkdir(parents=True)
    (_records_dir() / "aaaaaaaaaaaa.json").write_text("{", encoding="utf-8")
    runner = JobRunner(restore=True)
    assert runner.cancel("aaaaaaaaaaaa")  # accepted, and a no-op
    runner.shutdown(timeout=1)
    job = runner.get("aaaaaaaaaaaa")
    assert job is not None and job.status is JobStatus.CORRUPT


@pytest.mark.parametrize("payload", ['{"id": "bbbbbbbbbbbb"}', '{"status": "exploded"}', "7"])
def test_parsable_but_invalid_records_are_corrupt_too(payload):
    _records_dir().mkdir(parents=True)
    (_records_dir() / "bbbbbbbbbbbb.json").write_text(payload, encoding="utf-8")
    job = JobRunner(restore=True).get("bbbbbbbbbbbb")
    assert job is not None and job.status is JobStatus.CORRUPT


# --- finding 2: a trainer that outlives the server --------------------------

FAKE_MODULE = "so101_fake_durable_train"
# Stands in for lerobot.scripts.lerobot_train (run with ``python -m``).
FAKE_TRAINER = """
import os, signal, sys, time
from pathlib import Path

args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
if os.environ.get("FAKE_TRAINER_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready = os.environ.get("FAKE_TRAINER_READY")
if ready:
    Path(ready).write_text(str(os.getpid()), encoding="utf-8")
print("INFO step:1 loss:0.500", flush=True)
gate = os.environ.get("FAKE_TRAINER_GATE")
deadline = time.time() + 60
while gate and not Path(gate).exists() and time.time() < deadline:
    time.sleep(0.02)
ckpt = Path(args["output_dir"]) / "checkpoints" / "last" / "pretrained_model"
ckpt.mkdir(parents=True, exist_ok=True)
(ckpt / "config.json").write_text('{"type": "act"}', encoding="utf-8")
print("INFO End of training", flush=True)
"""

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


@pytest.fixture
def fake_trainer(tmp_path, monkeypatch):
    """Real mode with a stand-in trainer module found by find_spec and run with -m."""

    from so101.training import trainer as trainer_module

    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / f"{FAKE_MODULE}.py").write_text(FAKE_TRAINER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [str(modules), os.environ.get("PYTHONPATH")]))
    )
    monkeypatch.setattr(trainer_module, "TRAIN_MODULE", FAKE_MODULE)
    monkeypatch.setattr(trainer_module, "TRAIN_REQUIRES", ())
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    yield modules
    get_settings.cache_clear()


def _lerobot_dataset(repo_id: str) -> None:
    from urllib.parse import quote

    root = get_settings().datasets_dir / f"v2--{quote(repo_id, safe='')}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 10, "fps": 15, "features": {}}),
        encoding="utf-8",
    )


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def _alive(pid: int) -> bool:
    """A process of this user that can still run (EPERM: a zombie on macOS)."""

    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:  # a zombie child of this test still answers kill(0) on Linux
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return waited == 0


def _record_on_disk(job_id: str) -> dict:
    return json.loads((_records_dir() / f"{job_id}.json").read_text(encoding="utf-8"))


def _spawn_orphan(cmd: list[str], env: dict[str, str]) -> int:
    """Start ``cmd`` in its own session from a helper that then exits.

    The child is reparented to init/launchd, as a trainer is when the server
    that started it is killed, so this test process cannot reap it.
    """

    helper = (
        "import json, subprocess, sys\n"
        "cmd = json.loads(sys.argv[1])\n"
        "p = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL,\n"
        "                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(p.pid, flush=True)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", helper, json.dumps(cmd)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return int(out.stdout.strip())


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@posix_only
def test_trainer_pid_and_group_are_recorded_as_soon_as_it_starts(
    fake_trainer, tmp_path, monkeypatch
):
    gate = tmp_path / "gate"
    monkeypatch.setenv("FAKE_TRAINER_GATE", str(gate))
    _lerobot_dataset("test/dur_pid")
    runner = JobRunner()
    job = runner.submit("test/dur_pid", "act", 10, 1)
    try:
        _wait_for(lambda: job.pid is not None and _record_on_disk(job.id).get("pid") == job.pid)
        record = _record_on_disk(job.id)
        assert record["status"] == "running"
        assert record["pgid"] == job.pid  # it leads its own session and group
        assert record["pid_start"] and record["pid_start"] == process_start_marker(job.pid)
    finally:
        gate.touch()
        _wait_for(lambda: job.status is JobStatus.SUCCEEDED)
        runner.shutdown()


def test_trainer_keeps_dataset_read_lock_after_parent_releases_it(
    fake_trainer, tmp_path, monkeypatch
):
    from so101.data.dataset import dataset_operation_lock, dataset_read_hold
    from so101.training import trainer as trainer_module

    if os.name != "posix":
        assert trainer_module._inherited_dataset_read_fd("test/no-posix-inheritance") is None
        return

    repo_id = "test/dur_inherited_dataset_lock"
    _lerobot_dataset(repo_id)
    gate = tmp_path / "trainer-gate"
    monkeypatch.setenv("FAKE_TRAINER_GATE", str(gate))
    job = TrainingJob(
        repo_id=repo_id,
        policy="act",
        steps=10,
        batch_size=1,
        output_dir=str(get_settings().outputs_dir / "inherited-lock"),
    )
    stream = trainer_module.run_training(job)
    operation_lock = dataset_operation_lock(repo_id)
    try:
        with dataset_read_hold(repo_id):
            next(stream)  # Popen completed; the child inherited its own shared-lock fd.

        unexpectedly_acquired = operation_lock.acquire(blocking=False)
        if unexpectedly_acquired:
            operation_lock.release()
        assert not unexpectedly_acquired, (
            "the trainer must keep writers excluded after its parent read hold disappears"
        )
        gate.touch()
        list(stream)
        assert operation_lock.acquire(blocking=False)
        operation_lock.release()
    finally:
        gate.touch(exist_ok=True)
        stream.close()


@posix_only
def test_restored_job_stops_a_live_trainer_before_freeing_the_slot(
    fake_trainer, tmp_path, monkeypatch
):
    _lerobot_dataset("test/dur_orphan")
    outputs = get_settings().outputs_dir
    job = TrainingJob(
        repo_id="test/dur_orphan", policy="act", steps=10, batch_size=1, output_dir=""
    )
    job.output_dir = str(outputs / f"act_1_{job.id}")
    ready = tmp_path / "orphan.ready"
    env = {
        **os.environ,
        "FAKE_TRAINER_IGNORE_TERM": "1",  # needs the SIGKILL after the grace period
        "FAKE_TRAINER_READY": str(ready),
        "FAKE_TRAINER_GATE": str(tmp_path / "never"),
    }
    cmd = [sys.executable, "-m", FAKE_MODULE, f"--output_dir={job.output_dir}", "--steps=10"]
    orphan = _spawn_orphan(cmd, env)
    try:
        _wait_for(ready.exists)
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.pid, job.pgid, job.pid_start = orphan, orphan, process_start_marker(orphan)
        _write_record(job)

        monkeypatch.setattr(JobRunner, "orphan_grace_s", 1.0, raising=False)
        gate = tmp_path / "gate"
        monkeypatch.setenv("FAKE_TRAINER_GATE", str(gate))
        runner = JobRunner(restore=True)
        restored = runner.get(job.id)
        assert restored is not None
        assert restored.status is JobStatus.RUNNING, "still stopping the old trainer"
        assert "Stopping the trainer" in (restored.progress or "")

        new = runner.submit("test/dur_orphan", "act", 10, 1)
        _wait_for(lambda: new.status is JobStatus.QUEUED)
        assert _alive(orphan), "the new job must wait while the old trainer runs"
        _wait_for(lambda: new.started_at is not None, timeout=20)
        assert not _alive(orphan)
        assert restored.status is JobStatus.INTERRUPTED
        assert restored.finished_at is not None and restored.finished_at <= new.started_at
        assert "was stopped" in (restored.error or "")
        assert _record_on_disk(job.id)["status"] == "interrupted"
        gate.touch()
        _wait_for(lambda: new.status is JobStatus.SUCCEEDED)
        runner.shutdown()
    finally:
        _kill_group(orphan)


def test_restore_discovers_trainer_when_parent_died_before_pid_persist(
    fake_trainer, tmp_path, monkeypatch
):
    if not sys.platform.startswith("linux"):
        from so101.training import processes

        missing = TrainingJob(
            repo_id="test/pidless-orphan",
            policy="act",
            steps=10,
            batch_size=1,
            output_dir="missing-pid",
        )
        assert processes.inspect_trainer(missing, module=FAKE_MODULE).state == "unknown"
        return

    outputs = get_settings().outputs_dir
    job = TrainingJob(
        repo_id="test/pidless-orphan",
        policy="act",
        steps=10,
        batch_size=1,
        output_dir="",
        status=JobStatus.RUNNING,
        started_at=time.time(),
    )
    job.output_dir = str(outputs / f"act_1_{job.id}")
    ready = tmp_path / "pidless.ready"
    gate = tmp_path / "never"
    env = {**os.environ, "FAKE_TRAINER_READY": str(ready), "FAKE_TRAINER_GATE": str(gate)}
    cmd = [sys.executable, "-m", FAKE_MODULE, f"--output_dir={job.output_dir}", "--steps=10"]
    orphan = _spawn_orphan(cmd, env)
    release_stop = threading.Event()
    stop_started = threading.Event()
    real_stop = jobs_module.stop_process_group
    runner: JobRunner | None = None
    try:
        _wait_for(ready.exists)
        # This is the durable state in the Popen -> on_start persistence window.
        assert job.pid is None
        _write_record(job)

        def gated_stop(pgid, **kwargs):
            stop_started.set()
            assert release_stop.wait(5), "test did not release orphan cleanup"
            return real_stop(pgid, **kwargs)

        monkeypatch.setattr(jobs_module, "stop_process_group", gated_stop)
        runner = JobRunner(restore=True)
        assert stop_started.wait(5), "the pid-less trainer was not discovered"
        restored = runner.get(job.id)
        assert restored is not None
        assert restored.pid == orphan and restored.pgid == orphan and restored.pid_start
        assert restored.status is JobStatus.RUNNING
        assert not runner._slots["lerobot"].acquire(blocking=False)

        release_stop.set()
        _wait_for(lambda: restored.status is JobStatus.INTERRUPTED)
        _wait_for(lambda: not _alive(orphan))
        assert _record_on_disk(job.id)["pid"] == orphan
    finally:
        release_stop.set()
        if runner is not None:
            runner.shutdown(timeout=2)
        _kill_group(orphan)


def test_failed_orphan_stop_keeps_the_lerobot_slot_closed(monkeypatch):
    runner = JobRunner()
    old = TrainingJob(
        repo_id="test/orphan-stop-failed",
        policy="act",
        steps=10,
        batch_size=1,
        output_dir="old",
        status=JobStatus.RUNNING,
        pid=123,
        pgid=123,
    )
    runner._jobs[old.id] = old
    runner._record_dirs[old.id] = _records_dir()
    runner._cancel[old.id] = threading.Event()
    runner._persist(old)
    monkeypatch.setattr(jobs_module, "stop_process_group", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(JobRunner, "_slot_for", lambda self, _job: self._slots["lerobot"])
    training_started = threading.Event()

    def train(_job, **_kwargs):
        training_started.set()
        yield 1, 0.1, "done"

    monkeypatch.setattr(jobs_module, "run_training", train)
    runner._stop_orphans_in_background([(old, 123)])
    runner._threads[old.id].join(timeout=5)
    assert not runner._threads[old.id].is_alive()
    assert old.status is JobStatus.RUNNING
    first_record = _record_on_disk(old.id)
    assert first_record["status"] == "running"
    assert "Training remains locked" in (first_record["progress"] or "")
    assert not runner._slots["lerobot"].acquire(blocking=False)

    # A process restart must re-inspect this non-terminal record and keep a
    # fresh runner's slot closed when the stop still cannot be verified.
    monkeypatch.setattr(
        jobs_module,
        "inspect_trainer",
        lambda _job, *, module: TrainerCheck("ours", pgid=123, detail=module),
    )
    restored_runner = JobRunner(restore=True)
    restored = restored_runner.get(old.id)
    assert restored is not None
    restored_runner._threads[old.id].join(timeout=5)
    assert not restored_runner._threads[old.id].is_alive()
    assert restored.status is JobStatus.RUNNING

    new = restored_runner.submit("test/new-job", "act", 1, 1)
    _wait_for(lambda: new.status is JobStatus.QUEUED)
    assert not training_started.wait(0.2)
    assert not restored_runner._slots["lerobot"].acquire(blocking=False)
    restored_runner.shutdown(timeout=2)
    runner.shutdown(timeout=2)


def test_process_inspection_eio_is_unknown_not_verified_other(monkeypatch):
    from so101.training import processes

    job = TrainingJob(
        repo_id="test/inspect-eio",
        policy="act",
        steps=1,
        batch_size=1,
        output_dir="out",
        pid=123,
        pgid=123,
    )

    def fail_inspection(_pid):
        raise OSError(errno.EIO, "injected procfs EIO")

    monkeypatch.setattr(processes, "process_info", fail_inspection)
    check = processes.inspect_trainer(job, module=FAKE_MODULE)
    assert check.state == "unknown"
    assert "could not be inspected" in check.detail and "procfs EIO" in check.detail


def test_pidless_discovery_treats_malformed_live_same_uid_proc_as_unknown(monkeypatch):
    from so101.training import processes

    class Stat:
        st_uid = 1000

    job = TrainingJob(
        repo_id="test/pidless-malformed",
        policy="act",
        steps=1,
        batch_size=1,
        output_dir="out",
    )
    monkeypatch.setattr(processes, "POSIX", True)
    monkeypatch.setattr(processes, "_LINUX", True)
    monkeypatch.setattr(processes.os, "listdir", lambda _path: ["123"])
    monkeypatch.setattr(processes.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(processes.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(Path, "stat", lambda _path: Stat())
    monkeypatch.setattr(Path, "read_text", lambda _path, **_kwargs: "malformed live stat")
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"")

    check = processes.inspect_trainer(job, module=FAKE_MODULE)

    assert check.state == "unknown"
    assert "could not be inspected completely" in check.detail
    assert "malformed" in check.detail


@pytest.mark.parametrize("leader", [None, ProcessInfo(123, "Z", "start", "trainer")])
def test_missing_or_zombie_trainer_leader_keeps_live_worker_group_unknown(monkeypatch, leader):
    from so101.training import processes

    job = TrainingJob(
        repo_id="test/worker-group",
        policy="act",
        steps=1,
        batch_size=1,
        output_dir="out",
        pid=123,
        pgid=456,
    )
    monkeypatch.setattr(processes, "process_info", lambda _pid: leader)
    monkeypatch.setattr(processes, "group_alive", lambda pgid: pgid == 456)

    check = processes.inspect_trainer(job, module=FAKE_MODULE)

    assert check.state == "unknown"
    assert "process group 456" in check.detail


@pytest.mark.parametrize("damage", ["eio", "malformed"])
def test_group_alive_treats_procfs_ambiguity_as_live(monkeypatch, damage):
    from so101.training import processes

    monkeypatch.setattr(processes, "_LINUX", True)
    monkeypatch.setattr(processes.os, "killpg", lambda _pgid, _signal: None)
    monkeypatch.setattr(processes.os, "listdir", lambda _path: ["123"])

    def read_stat(_path, **_kwargs):
        if damage == "eio":
            raise OSError(errno.EIO, "injected proc stat EIO")
        return "123 (worker) malformed"

    monkeypatch.setattr(Path, "read_text", read_stat)
    assert processes.group_alive(456) is True


@pytest.mark.parametrize(
    ("returncode", "stdout", "detail"),
    [
        (1, "", "ps could not inspect"),
        (0, "malformed", "malformed process metadata"),
    ],
)
def test_macos_ps_failure_is_unknown(monkeypatch, returncode, stdout, detail):
    from so101.training import processes

    job = TrainingJob(
        repo_id="test/macos-inspection",
        policy="act",
        steps=1,
        batch_size=1,
        output_dir="out",
        pid=123,
        pgid=123,
    )
    monkeypatch.setattr(processes, "POSIX", True)
    monkeypatch.setattr(processes, "_LINUX", False)
    monkeypatch.setattr(processes.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(
        processes.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr="ps error"
        ),
    )

    check = processes.inspect_trainer(job, module=FAKE_MODULE)

    assert check.state == "unknown"
    assert detail in check.detail


def test_unknown_trainer_inspection_keeps_gpu_slot_closed(monkeypatch):
    old = TrainingJob(
        repo_id="test/unknown-inspection",
        policy="act",
        steps=10,
        batch_size=1,
        output_dir="old",
        status=JobStatus.RUNNING,
        pid=123,
        pgid=123,
    )
    _write_record(old)
    monkeypatch.setattr(
        jobs_module,
        "inspect_trainer",
        lambda _job, *, module: TrainerCheck("unknown", detail=f"{module}: injected procfs EIO"),
    )
    monkeypatch.setattr(JobRunner, "_slot_for", lambda self, _job: self._slots["lerobot"])
    training_started = threading.Event()

    def train(_job, **_kwargs):
        training_started.set()
        yield 1, 0.1, "done"

    monkeypatch.setattr(jobs_module, "run_training", train)
    runner = JobRunner(restore=True)
    restored = runner.get(old.id)
    assert restored is not None and restored.status is JobStatus.RUNNING
    assert "procfs EIO" in (restored.error or "")
    assert not runner._slots["lerobot"].acquire(blocking=False)

    new = runner.submit("test/new-after-eio", "act", 1, 1)
    _wait_for(lambda: new.status is JobStatus.QUEUED)
    assert not training_started.wait(0.2)
    assert _record_on_disk(old.id)["status"] == "running"
    runner.shutdown(timeout=2)


def _sleeper(cmd_tail: list[str], *, new_session: bool) -> subprocess.Popen[bytes]:
    script = "import time; time.sleep(60)"
    return subprocess.Popen(
        [sys.executable, "-c", script, *cmd_tail], start_new_session=new_session
    )


@posix_only
@pytest.mark.parametrize("reuse", ["unrelated command", "different start time", "other group"])
def test_a_reused_pid_is_never_signalled(reuse, fake_trainer, monkeypatch):
    outputs = get_settings().outputs_dir
    job = TrainingJob(repo_id="test/reuse", policy="act", steps=10, batch_size=1, output_dir="")
    job.output_dir = str(outputs / f"act_1_{job.id}")
    trainer_like = ["-m", FAKE_MODULE, f"--output_dir={job.output_dir}"]
    if reuse == "unrelated command":
        proc = _sleeper([], new_session=True)
    else:  # looks like the trainer; only the start time or the group tells it apart
        proc = _sleeper(trainer_like, new_session=reuse != "other group")
    try:
        _wait_for(lambda: process_start_marker(proc.pid) is not None)
        if reuse != "unrelated command":  # the check matches "-m <module>" anywhere
            assert FAKE_MODULE in (process_info(proc.pid) or ProcessInfo(0, "", "", "")).cmdline
        job.status = JobStatus.RUNNING
        job.pid, job.pgid = proc.pid, proc.pid
        job.pid_start = process_start_marker(proc.pid)
        if reuse == "different start time":
            job.pid_start = "Thu Jan  1 00:00:00 1970"
        _write_record(job)
        monkeypatch.setattr(JobRunner, "orphan_grace_s", 0.2, raising=False)

        runner = JobRunner(restore=True)
        restored = runner.get(job.id)
        assert restored is not None and restored.status is JobStatus.INTERRUPTED
        assert "left alone" in (restored.error or "")
        time.sleep(0.5)
        assert proc.poll() is None, "an unrelated process was signalled"
        slot = runner._slots["lerobot"]
        assert slot.acquire(blocking=False), "nothing of ours runs, so the slot is free"
        slot.release()
    finally:
        proc.kill()
        proc.wait()


@posix_only
def test_a_second_server_leaves_a_live_servers_trainer_alone(fake_trainer, monkeypatch):
    """Only the runner holding .jobs/.lock acts on unfinished records.

    uvicorn runs the app's startup (restore) before it binds the port, so a
    second server started on the same data folder must not stop the first
    one's trainer, nor rewrite its record.
    """

    assert not _records_dir().exists()  # a fresh install: no records yet
    owner = JobRunner(restore=True)  # the live server
    outputs = get_settings().outputs_dir
    job = TrainingJob(repo_id="test/live", policy="act", steps=10, batch_size=1, output_dir="")
    job.output_dir = str(outputs / f"act_1_{job.id}")
    proc = _sleeper(["-m", FAKE_MODULE, f"--output_dir={job.output_dir}"], new_session=True)
    try:
        _wait_for(lambda: process_start_marker(proc.pid) is not None)
        job.status = JobStatus.RUNNING
        job.pid, job.pgid, job.pid_start = proc.pid, proc.pid, process_start_marker(proc.pid)
        _write_record(job)
        monkeypatch.setattr(JobRunner, "orphan_grace_s", 0.2, raising=False)

        second = JobRunner(restore=True)
        listed = second.get(job.id)
        assert listed is not None and listed.status is JobStatus.RUNNING
        time.sleep(0.5)
        assert proc.poll() is None, "the live server's trainer was signalled"
        assert _record_on_disk(job.id)["status"] == "running"
        second.shutdown(timeout=1)

        owner.shutdown(timeout=1)  # the live server exits (its lock goes with it) ...
        third = JobRunner(restore=True)  # ... and the next start stops the orphan
        _wait_for(lambda: proc.poll() is not None, timeout=10)
        restored = third.get(job.id)
        assert restored is not None
        _wait_for(lambda: restored.status is JobStatus.INTERRUPTED)
        third.shutdown(timeout=1)
    finally:
        proc.kill()
        proc.wait()


def test_windows_never_inspects_or_signals(monkeypatch):
    from so101.training import processes

    monkeypatch.setattr(processes, "POSIX", False)
    job = TrainingJob(repo_id="x/y", policy="act", steps=1, batch_size=1, output_dir="o", pid=1)
    check = processes.inspect_trainer(job, module="m")
    assert check.state == "unknown" and "cannot check" in check.detail
    assert processes.stop_process_group(1, grace_s=0, kill_wait_s=0) is False
    assert processes.own_process_group(1) is None
    assert processes.process_start_marker(1) is None
    no_pid = TrainingJob(repo_id="x/y", policy="act", steps=1, batch_size=1, output_dir="o")
    assert processes.inspect_trainer(no_pid, module="m").state == "unknown"


@posix_only
def test_stop_process_group_escalates_and_reports(tmp_path):
    from so101.training import processes

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "open(__import__('sys').argv[1], 'w').close()\ntime.sleep(60)",
            str(tmp_path / "r"),
        ],
        start_new_session=True,
    )
    try:
        _wait_for((tmp_path / "r").exists)
        reaper = threading.Thread(target=proc.wait, daemon=True)  # reap like init would
        reaper.start()
        hurry = threading.Event()
        hurry.set()  # skip the grace period
        started = time.monotonic()
        assert processes.stop_process_group(proc.pid, grace_s=30, kill_wait_s=5, hurry=hurry)
        assert time.monotonic() - started < 10
        # The group counts as gone once it is only a zombie; the reaper may not
        # have collected it yet, so wait for that before reading the status.
        reaper.join(timeout=10)
        assert not reaper.is_alive()
        assert proc.returncode == -signal.SIGKILL
        assert processes.stop_process_group(proc.pid, grace_s=1, kill_wait_s=1)  # already gone
    finally:
        _kill_group(proc.pid)
        proc.wait()


# --- finding 2: SIGHUP (closing the launcher's terminal) shuts down gracefully


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_hangup_handler_forwards_once_to_the_servers_stop_handler(monkeypatch):
    from so101 import launcher

    seen: list[int] = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: lambda s, f: seen.append(s))
    handler = launcher._HangupStopsServer()
    handler(signal.SIGHUP, None)
    handler(signal.SIGHUP, None)  # a second hang-up, or uvicorn's re-raise afterwards
    assert seen == [signal.SIGHUP]


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_hangup_before_the_server_runs_keeps_the_default_action(monkeypatch):
    from so101 import launcher

    calls: list[tuple] = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, action: calls.append(("set", sig, action)))
    monkeypatch.setattr(signal, "raise_signal", lambda sig: calls.append(("raise", sig)))
    launcher._HangupStopsServer()(signal.SIGHUP, None)
    assert calls == [("set", signal.SIGHUP, signal.SIG_DFL), ("raise", signal.SIGHUP)]


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_hangup_handler_is_installed_only_while_serving(monkeypatch):
    from so101 import launcher

    before = signal.getsignal(signal.SIGHUP)
    seen: list[object] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *a, **kw: seen.append(signal.getsignal(signal.SIGHUP))
    )
    launcher.run_server("127.0.0.1", 1)
    assert isinstance(seen[0], launcher._HangupStopsServer)
    assert signal.getsignal(signal.SIGHUP) == before


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_sighup_stops_a_launched_server_gracefully(tmp_path):
    """Closing the terminal: jobs are cancelled and the lifespan shutdown completes."""

    import httpx

    from so101 import JOINT_NAMES
    from so101.data.recorder import EpisodeRecorder, RecordingConfig
    from so101.teleop.controller import TeleopStep

    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/hup", fps=10, camera_keys=[]))
    for index in range(5):
        joints = {f"{joint}.pos": float(index) for joint in JOINT_NAMES}
        recorder.add(TeleopStep(t=1.0 + index / 10, observation=joints, action=joints))
    recorder.save_episode()

    port = _free_port()
    env = {key: value for key, value in os.environ.items() if key != "SO101_ALLOWED_HOSTS"}
    log = tmp_path / "server.log"
    home = tmp_path / "so101-home"
    command = [sys.executable, "-m", "so101.launcher", "--home", str(home), "--port", str(port)]
    with log.open("wb") as out:
        server = subprocess.Popen(
            [*command, "--no-browser"],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:

            def healthy() -> bool:
                assert server.poll() is None, log.read_text(errors="replace")
                try:
                    return client.get("/api/health").status_code == 200
                except httpx.HTTPError:
                    return False

            _wait_for(healthy, timeout=90)
            body = {"repo_id": "test/hup", "policy": "bc_baseline", "steps": 10_000_000}
            response = client.post("/api/training/jobs", json={**body, "batch_size": 1})
            assert response.status_code == 200, response.text
            job_id = response.json()["id"]
            _wait_for(
                lambda: client.get(f"/api/training/jobs/{job_id}").json()["status"] == "running",
                timeout=30,
            )
        server.send_signal(signal.SIGHUP)
        code = server.wait(timeout=60)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
    text = log.read_text(errors="replace")
    assert code == 0, text
    assert "Application shutdown complete" in text, text
    assert _record_on_disk(job_id)["status"] == "cancelled"


# --- so101 serve handles SIGHUP like the launcher ------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_so101_serve_runs_with_the_hangup_handler_and_its_options(monkeypatch):
    from typer.testing import CliRunner

    from so101 import launcher
    from so101.cli import app

    before = signal.getsignal(signal.SIGHUP)
    seen: list[tuple[object, tuple, dict]] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *a, **kw: seen.append((signal.getsignal(signal.SIGHUP), a, kw))
    )
    result = CliRunner().invoke(app, ["serve", "--host", "127.0.0.2", "--port", "8831"])
    assert result.exit_code == 0, result.output
    ((handler, args, kwargs),) = seen
    assert isinstance(handler, launcher._HangupStopsServer)
    assert args == (launcher.APP,)
    assert kwargs == {"host": "127.0.0.2", "port": 8831, "factory": True}
    assert signal.getsignal(signal.SIGHUP) == before

    # Without options it serves SO101_HOST / SO101_PORT, as before.
    monkeypatch.setenv("SO101_HOST", "127.0.0.3")
    monkeypatch.setenv("SO101_PORT", "8832")
    get_settings.cache_clear()
    seen.clear()
    assert CliRunner().invoke(app, ["serve"]).exit_code == 0
    assert seen[0][2] == {"host": "127.0.0.3", "port": 8832, "factory": True}


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_sighup_stops_so101_serve_gracefully(tmp_path):
    """Closing the terminal `so101 serve` runs in: the lifespan shutdown completes."""

    import httpx

    port = _free_port()
    env = {key: value for key, value in os.environ.items() if key != "SO101_ALLOWED_HOSTS"}
    log = tmp_path / "serve.log"
    command = [sys.executable, "-m", "so101.cli", "serve", "--port", str(port)]
    with log.open("wb") as out:
        server = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env
        )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5
        ) as client:

            def healthy() -> bool:
                assert server.poll() is None, log.read_text(errors="replace")
                try:
                    return client.get("/api/health").status_code == 200
                except httpx.HTTPError:
                    return False

            _wait_for(healthy, timeout=90)
        server.send_signal(signal.SIGHUP)
        code = server.wait(timeout=60)
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
    text = log.read_text(errors="replace")
    assert code == 0, text
    assert "Application shutdown complete" in text, text


# --- finding 3: the run manifest is fsynced; an unreadable one is reported ----


def _mock_run(repo_id: str = "test/manifest_dur"):
    """A mock training run (checkpoint plus so101_run.json) from a recorded dataset."""

    from so101 import JOINT_NAMES
    from so101.data.recorder import EpisodeRecorder, RecordingConfig
    from so101.teleop.controller import TeleopStep
    from so101.training.trainer import run_training

    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=15, camera_keys=[]))
    for index in range(3):
        joints = {f"{joint}.pos": float(index) for joint in JOINT_NAMES}
        recorder.add(TeleopStep(t=1.0 + index / 15, observation=joints, action=joints))
    recorder.save_episode()
    run = get_settings().outputs_dir / "act_manifest_dur"
    job = TrainingJob(repo_id=repo_id, policy="act", steps=20, batch_size=1, output_dir=str(run))
    list(run_training(job))
    return run, run / "checkpoints" / "last" / "pretrained_model"


def test_run_manifest_is_fsynced_file_and_directory(monkeypatch):
    from so101.training.configs import POLICY_PRESETS
    from so101.training.run_manifest import write_run_manifest

    run = get_settings().outputs_dir / "fsync_run"
    job = TrainingJob(repo_id="x/y", policy="act", steps=1, batch_size=1, output_dir=str(run))
    calls = _fsync_spy(monkeypatch)
    path = write_run_manifest(job, POLICY_PRESETS["act"], trainer="mock")
    assert path is not None and json.loads(path.read_text(encoding="utf-8"))["job_id"] == job.id
    assert "file" in calls and "dir" in calls, calls


@pytest.mark.parametrize("damage", ["truncated", "empty", "not an object"])
def test_an_unreadable_run_manifest_is_reported_not_defaulted(damage):
    from typer.testing import CliRunner

    from so101.cli import app as cli_app
    from so101.models import list_models
    from so101.policy.checkpoint import InvalidCheckpointError
    from so101.training.run_manifest import checkpoint_defaults

    run, checkpoint = _mock_run()
    manifest = run / RUN_MANIFEST_NAME
    assert checkpoint_defaults(checkpoint).fps == 15
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        {"truncated": text[: len(text) // 2], "empty": "", "not an object": "[15]"}[damage],
        encoding="utf-8",
    )

    with pytest.raises(InvalidCheckpointError, match="run manifest"):
        checkpoint_defaults(checkpoint)
    for args in (
        ["run", "--duration", "0.02"],
        ["eval", "--episodes", "1", "--max-seconds", "0.02"],
    ):
        result = CliRunner().invoke(cli_app, [args[0], str(checkpoint), *args[1:]], input="y\n")
        output = " ".join(result.output.split())
        assert result.exit_code == 1, output
        assert "run manifest" in output and "can't be read" in output, output
        assert "rollout done" not in output and "success" not in output
    [model] = [m for m in list_models() if m.path == str(checkpoint)]
    assert "run manifest" in model.label  # the registry shows it as unreadable

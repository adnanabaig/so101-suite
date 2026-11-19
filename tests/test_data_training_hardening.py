"""Training launch, progress, admission and run manifests (audit B1, B2, B10, B16, B17,
B20, B23, B24, B26, B28)."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so101 import JOINT_NAMES
from so101.cli import app as cli_app
from so101.config import get_settings
from so101.data.dataset import DatasetBusyError, dataset_read_hold, delete_dataset
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.services.app import create_app
from so101.services.jobs import JobRunner
from so101.teleop.controller import TeleopStep
from so101.training import trainer as trainer_module
from so101.training.configs import POLICY_PRESETS
from so101.training.jobs import JobStatus, TrainingJob
from so101.training.preflight import InvalidTrainingArgsError, validate_extra_args
from so101.training.run_manifest import (
    RUN_MANIFEST_NAME,
    _git_sha,
    checkpoint_defaults,
    read_run_manifest,
)
from so101.training.trainer import (
    build_train_command,
    is_progress_line,
    parse_step,
    run_training,
)

# Stands in for lerobot.scripts.lerobot_train, run with ``python -m`` like the real one.
FAKE_TRAINER = """
import json, os, sys, time
from pathlib import Path

argv = sys.argv[1:]
with open(os.environ["FAKE_TRAINER_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(argv) + "\\n")
args = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
out = Path(args["output_dir"])
if out.exists():  # lerobot-train's TrainPipelineConfig.validate()
    print("FileExistsError: Output directory exists and resume is False", flush=True)
    sys.exit(1)
if args.get("policy.push_to_hub") != "false":
    print("ValueError: 'repo_id' argument missing", flush=True)
    sys.exit(1)
print("INFO Output dir: " + str(out), flush=True)
print("INFO io encoding: " + os.environ.get("PYTHONIOENCODING", "unset"), flush=True)
if os.environ.get("FAKE_TRAINER_BAD_BYTE"):  # not UTF-8: must not end the job
    sys.stdout.buffer.write(b"INFO caf\\xe9 \\xff\\n")
    sys.stdout.flush()
print("Training:  50%|#####     | 500/1000 [00:01<00:01, 500step/s]", flush=True)
print("INFO step:1K smpl:8K ep:2 epch:0.50 loss:0.123 grdn:1.000", flush=True)
gate = os.environ.get("FAKE_TRAINER_GATE")
if gate:
    deadline = time.time() + 20
    while not Path(gate).exists() and time.time() < deadline:
        time.sleep(0.02)
print("Training: 100%|##########| 1000/1000 [00:02<00:00, 500step/s]", flush=True)
ckpt = out / "checkpoints" / "last" / "pretrained_model"
ckpt.mkdir(parents=True)
(ckpt / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
print("INFO End of training", flush=True)
"""


def _record(repo_id: str, episodes: int = 1) -> None:
    for _ in range(episodes):
        recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=10, camera_keys=[]))
        for index in range(3):
            joints = {f"{joint}.pos": float(index) for joint in JOINT_NAMES}
            recorder.add(TeleopStep(t=1.0 + index / 10, observation=joints, action=joints))
        recorder.save_episode()


def _lerobot_dataset(repo_id: str, *, fps: int = 15, task: str = "pick the cube") -> None:
    from urllib.parse import quote

    root = get_settings().datasets_dir / f"v2--{quote(repo_id, safe='')}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 40, "fps": fps, "features": {}}),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8"
    )


FAKE_TRAINER_MODULE = "so101_fake_lerobot_train"


@pytest.fixture
def fake_trainer(tmp_path, monkeypatch):
    """Real mode with a stand-in ``lerobot-train`` that enforces 0.6.1's start-up checks.

    The stand-in is a module the trainer finds with ``find_spec`` and starts
    with ``python -m``, exactly as it does LeRobot's. Nothing goes on PATH.
    """

    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / f"{FAKE_TRAINER_MODULE}.py").write_text(FAKE_TRAINER, encoding="utf-8")
    log = tmp_path / "trainer_argv.jsonl"
    monkeypatch.syspath_prepend(str(modules))  # for find_spec in this process
    monkeypatch.setenv(  # for the child interpreter
        "PYTHONPATH", os.pathsep.join(filter(None, [str(modules), os.environ.get("PYTHONPATH")]))
    )
    monkeypatch.setattr(trainer_module, "TRAIN_MODULE", FAKE_TRAINER_MODULE)
    monkeypatch.setattr(trainer_module, "TRAIN_REQUIRES", ())
    monkeypatch.setenv("FAKE_TRAINER_LOG", str(log))
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    yield log
    get_settings.cache_clear()


def _job(repo_id: str, output_dir, **kwargs) -> TrainingJob:
    return TrainingJob(
        repo_id=repo_id,
        policy=kwargs.pop("policy", "act"),
        steps=kwargs.pop("steps", 1000),
        batch_size=kwargs.pop("batch_size", 8),
        output_dir=str(output_dir),
        **kwargs,
    )


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("line", "current", "expected"),
    [
        ("INFO step:800 smpl:6K ep:20 epch:0.20 loss:0.500", 0, 800),
        ("INFO step:1K smpl:8K ep:20 epch:0.25 loss:0.400", 800, 1000),
        ("INFO step:20K smpl:160K loss:0.1", 0, 20_000),
        ("INFO step:100K loss:0.1", 0, 100_000),
        ("INFO step:1.5M loss:0.1", 0, 1_500_000),
        ("Training:  42%|####2     | 420/1000 [00:10<00:14, 40.1step/s]", 0, 420),
        ("\x1b[ATraining:  43%|####3     | 1430/3300 [00:10<00:14]", 0, 1430),
        # The rounded log form never moves an exact count backwards.
        ("INFO step:1K loss:0.3", 1400, 1400),
        ("[mock] step:5 loss:0.1000", 0, 5),
        ("env_step:9 total_steps:12", 3, 3),
    ],
)
def test_parse_step_handles_lerobot_rounding_and_progress_bar(line, current, expected):
    assert parse_step(line, current) == expected


def test_progress_lines_are_recognised():
    assert is_progress_line("Training:  42%|####2     | 420/1000")
    assert is_progress_line("\x1b[A  Training: 1%")
    assert not is_progress_line("INFO step:1K loss:0.3")


def test_train_command_keeps_so101_flags_last_and_never_pushes(tmp_path):
    # A Hub dataset has no outcome labels here, so it trains with "all".
    job = _job(
        "lerobot/remote_only",
        tmp_path / "run",
        policy="smolvla",
        remote_dataset=True,
        episodes="all",
    )
    job.extra_args = {"policy.n_obs_steps": "2", "optimizer.lr": "1e-4"}
    command = build_train_command(job)
    # The suite's own interpreter runs LeRobot's module: no PATH lookup.
    assert command[:3] == [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    assert "--policy.push_to_hub=false" in command
    assert "--policy.pretrained_path=lerobot/smolvla_base" in command
    owned = {
        "--output_dir": str(tmp_path / "run"),
        "--dataset.repo_id": "lerobot/remote_only",
        "--steps": "1000",
        "--batch_size": "8",
        "--policy.push_to_hub": "false",
        "--wandb.enable": "false",
    }
    first_owned = min(command.index(f"{flag}={value}") for flag, value in owned.items())
    overrides = [command.index("--policy.n_obs_steps=2"), command.index("--optimizer.lr=1e-4")]
    assert max(overrides) < first_owned
    parsed = dict(arg[2:].split("=", 1) for arg in command[3:])  # draccus keeps the last
    assert parsed["output_dir"] == str(tmp_path / "run")

    job.extra_args = {"output_dir": str(tmp_path / "elsewhere")}
    with pytest.raises(InvalidTrainingArgsError):
        build_train_command(job)


def test_pi0_finetunes_from_its_pretrained_base():
    assert POLICY_PRESETS["pi0"].extra_args["policy.pretrained_path"] == "lerobot/pi0_base"


@pytest.mark.parametrize(
    "key",
    [
        "output_dir",
        "dataset.root",
        "dataset.repo_id",
        "policy.push_to_hub",
        "policy.repo_id",
        "policy.type",
        "policy.path",
        "wandb.enable",
        "resume",
        "job_name",
        "steps",
        "batch_size",
        "config_path",
        "env.type",
        "save_checkpoint_to_hub",
        "Policy.lr",
        "policy..lr",
        "--output_dir",
    ],
)
def test_extra_args_reject_so101_owned_or_malformed_keys(key):
    with pytest.raises(InvalidTrainingArgsError):
        validate_extra_args(POLICY_PRESETS["act"], {key: "1"})


def test_extra_args_values_and_allowed_keys(tmp_path):
    act = POLICY_PRESETS["act"]
    allowed = {
        "policy.n_obs_steps": "2",
        "policy.device": "mps",
        "optimizer.lr": "1e-4",
        "scheduler.num_warmup_steps": "100",
        "seed": "7",
        "num_workers": "2",
        "save_freq": "500",
        "policy.pretrained_path": "lerobot/smolvla_base",
    }
    assert validate_extra_args(act, allowed) == allowed
    assert validate_extra_args(act, {"policy.pretrained_path": str(tmp_path)})
    for value in ("attacker/exfil", "https://example.com/model", str(tmp_path / "missing")):
        with pytest.raises(InvalidTrainingArgsError, match="pretrained_path"):
            validate_extra_args(act, {"policy.pretrained_path": value})
    for value in ("", "x" * 513, "a\nb", 5):
        with pytest.raises(InvalidTrainingArgsError):
            validate_extra_args(act, {"seed": value})  # type: ignore[dict-item]


def test_training_api_validates_overrides_and_bounds():
    _record("test/api_bounds")
    body = {"repo_id": "test/api_bounds", "policy": "act", "steps": 10, "batch_size": 1}
    with TestClient(create_app()) as client:
        owned = client.post(
            "/api/training/jobs", json={**body, "extra_args": {"output_dir": "elsewhere"}}
        )
        assert owned.status_code == 422
        assert "not an allowed" in owned.json()["detail"]
        too_many = client.post("/api/training/jobs", json={**body, "steps": 10_000_001})
        assert too_many.status_code == 422
        negative = client.post("/api/training/jobs", json={**body, "batch_size": -3})
        assert negative.status_code == 422


def test_cli_train_rejects_owned_override_and_names_runs_uniquely():
    _record("test/cli_args")
    result = CliRunner().invoke(
        cli_app, ["train", "test/cli_args", "--steps", "5", "--arg", "policy.push_to_hub=true"]
    )
    assert result.exit_code == 1
    assert "not an allowed" in " ".join(result.output.split())
    result = CliRunner().invoke(cli_app, ["train", "test/cli_args", "--arg", "seed"])
    assert result.exit_code == 1
    assert "KEY=VALUE" in result.output

    for _ in range(2):
        result = CliRunner().invoke(cli_app, ["train", "test/cli_args", "--steps", "5"])
        assert result.exit_code == 0, result.output
    runs = [path.name for path in get_settings().outputs_dir.iterdir() if path.name[0] != "."]
    assert len(runs) == 2 and runs[0] != runs[1]


def test_real_training_leaves_output_dir_to_lerobot(fake_trainer, tmp_path):
    _lerobot_dataset("test/lr")
    out = tmp_path / "outputs" / "act_run"
    stream = list(run_training(_job("test/lr", out)))
    assert stream[-1][2] == "INFO End of training"
    assert max(step for step, _loss, _line in stream) == 1000
    assert stream[-1][1] == pytest.approx(0.123)
    argv = json.loads(fake_trainer.read_text(encoding="utf-8").splitlines()[0])
    assert "--policy.push_to_hub=false" in argv
    assert f"--dataset.root={get_settings().datasets_dir / 'v2--test%2Flr'}" in argv

    manifest = read_run_manifest(out / "checkpoints" / "last" / "pretrained_model")
    assert manifest is not None
    assert manifest["trainer"] == "lerobot"
    assert manifest["dataset"]["fps"] == 15
    assert manifest["task"] == "pick the cube"
    assert manifest["backend"] == "real"

    with pytest.raises(FileExistsError):
        list(run_training(_job("test/lr", out)))


def test_real_training_runs_lerobot_with_this_python_and_reads_utf8(
    fake_trainer, tmp_path, monkeypatch
):
    """``python -m <trainer module>`` with sys.executable, in its own session, UTF-8 I/O."""
    import subprocess

    started: list[tuple[list[str], dict]] = []
    real_popen = subprocess.Popen

    def recording_popen(cmd, **kwargs):
        started.append((list(cmd), kwargs))
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(trainer_module.subprocess, "Popen", recording_popen)
    monkeypatch.setenv("FAKE_TRAINER_BAD_BYTE", "1")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))  # nothing found on PATH
    _lerobot_dataset("test/lr_utf8")
    lines = [line for _s, _l, line in run_training(_job("test/lr_utf8", tmp_path / "o" / "u"))]

    # The trainer launch (on macOS, `ps` also runs: it reads the trainer's start time).
    [(cmd, kwargs)] = [(cmd, kwargs) for cmd, kwargs in started if cmd[0] == sys.executable]
    assert cmd[:3] == [sys.executable, "-m", FAKE_TRAINER_MODULE]
    assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace"
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert kwargs["start_new_session"] is True  # the POSIX process-group stop
    assert "INFO io encoding: utf-8" in lines
    assert "INFO caf\ufffd \ufffd" in lines  # replaced, and the job still finished
    assert lines[-1] == "INFO End of training"


def test_training_availability_is_an_import_check_shared_with_capabilities(
    fake_trainer, monkeypatch
):
    """The trainer and /api/capabilities agree, and neither looks at PATH."""
    client = TestClient(create_app())

    def training_capability() -> dict:
        items = client.get("/api/capabilities").json()["capabilities"]
        return next(item for item in items if item["id"] == "training")

    assert trainer_module.lerobot_train_available() is True
    assert training_capability()["available"] is True

    monkeypatch.setattr(trainer_module, "TRAIN_REQUIRES", ("so101_no_such_accelerate",))
    assert trainer_module.lerobot_train_available() is False
    assert training_capability()["available"] is False
    assert "install the training stack" in training_capability()["detail"]

    monkeypatch.setattr(trainer_module, "TRAIN_REQUIRES", ())
    monkeypatch.setattr(trainer_module, "TRAIN_MODULE", "so101_no_such_pkg.scripts.train")
    assert trainer_module.lerobot_train_available() is False  # a missing parent package
    assert training_capability()["available"] is False


def test_the_default_trainer_is_lerobots_module():
    assert trainer_module.TRAIN_MODULE == "lerobot.scripts.lerobot_train"
    assert trainer_module.TRAIN_REQUIRES == ("accelerate",)
    assert trainer_module.train_launcher() == [sys.executable, "-m", trainer_module.TRAIN_MODULE]


def test_closing_the_stream_stops_the_trainer(fake_trainer, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TRAINER_GATE", str(tmp_path / "never"))
    _lerobot_dataset("test/lr_close")
    job = _job("test/lr_close", tmp_path / "outputs" / "closed")
    stream = run_training(job)
    next(stream)
    assert job.pid is not None
    stream.close()  # what Ctrl-C in `so101 train` does via contextlib.closing
    with pytest.raises(ProcessLookupError):
        os.kill(job.pid, 0)


def test_runner_queues_real_jobs_and_shares_the_dataset(fake_trainer, tmp_path, monkeypatch):
    gate = tmp_path / "gate"
    monkeypatch.setenv("FAKE_TRAINER_GATE", str(gate))
    _lerobot_dataset("test/lr_queue")
    runner = JobRunner()
    first = runner.submit("test/lr_queue", "act", 1000, 8)
    _wait_for(lambda: first.step >= 1000)
    second = runner.submit("test/lr_queue", "act", 1000, 8)
    _wait_for(lambda: second.status is JobStatus.QUEUED)

    # Training only reads: a second reader may share the dataset ...
    with dataset_read_hold("test/lr_queue"):
        pass
    # ... but recording into it or deleting it names the training job.
    with pytest.raises(DatasetBusyError, match="training"):
        delete_dataset("test/lr_queue")

    assert runner.cancel(second.id)
    _wait_for(lambda: second.status is JobStatus.CANCELLED)
    assert second.started_at is None
    gate.touch()
    _wait_for(lambda: first.status is JobStatus.SUCCEEDED)
    runner.shutdown()
    assert first.progress is not None and first.progress.startswith("Training: 100%")
    assert not any(line.startswith("Training:") for line in first.log_tail)
    assert "INFO End of training" in first.log_tail

    # Job records survive a restart; a job that was mid-run is reported as such.
    records = get_settings().outputs_dir / ".jobs"
    interrupted = json.loads((records / f"{first.id}.json").read_text(encoding="utf-8"))
    interrupted.update(id="deadbeef0000", status="running", finished_at=None)
    (records / "deadbeef0000.json").write_text(json.dumps(interrupted), encoding="utf-8")
    restored = {job.id: job for job in JobRunner(restore=True).list()}
    assert restored[first.id].status is JobStatus.SUCCEEDED
    assert restored[second.id].status is JobStatus.CANCELLED
    assert restored["deadbeef0000"].status is JobStatus.INTERRUPTED
    assert "interrupted" in (restored["deadbeef0000"].error or "")
    assert "no longer running" in (restored["deadbeef0000"].error or "")  # its trainer exited


def test_a_job_is_recorded_before_it_shows_finished_where_it_was_submitted(tmp_path, monkeypatch):
    """The worker writes the final record before showing the final status and never
    rereads settings. It used to write after, so when a test ended on seeing the
    status, its record (and sometimes the next test's datasets) went to the default
    directories under the working directory."""

    _record("test/record_order")
    persist = JobRunner._persist
    writes: list[tuple[str, JobStatus]] = []

    def spy(self, job, **kwargs):
        persist(self, job, **kwargs)
        writes.append(((kwargs.get("status") or job.status).value, job.status))

    monkeypatch.setattr(JobRunner, "_persist", spy)
    runner = JobRunner()
    job = runner.submit("test/record_order", "act", 5, 1)
    records = get_settings().outputs_dir / ".jobs"
    # What the shared fixture does as soon as a test ends.
    monkeypatch.setenv("SO101_OUTPUTS_DIR", str(tmp_path / "elsewhere"))
    get_settings.cache_clear()
    finished = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
    _wait_for(lambda: job.status in finished)
    runner.shutdown()

    assert job.status is JobStatus.SUCCEEDED, job.error
    record = json.loads((records / f"{job.id}.json").read_text(encoding="utf-8"))
    assert record["status"] == "succeeded" and record["finished_at"] is not None
    assert not (tmp_path / "elsewhere" / ".jobs").exists()
    # The final record was written while the job still showed as running.
    assert writes[-1] == ("succeeded", JobStatus.RUNNING)


def test_mock_training_writes_run_manifest_and_checkpoint_defaults(tmp_path):
    _record("test/manifest", 2)
    out = tmp_path / "outputs" / "act_manifest"
    list(run_training(_job("test/manifest", out, steps=20)))
    checkpoint = out / "checkpoints" / "last" / "pretrained_model"
    assert (out / RUN_MANIFEST_NAME).is_file()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["type"] == "act" and config["mock"] is True
    defaults = checkpoint_defaults(checkpoint)
    assert defaults.task == "Perform the task."
    assert defaults.fps == 10
    assert defaults.policy_type == "act"
    assert defaults.dataset == "test/manifest"
    assert checkpoint_defaults(tmp_path / "nowhere").task is None


def test_run_manifest_reads_the_git_revision_from_files(tmp_path):
    """No ``git`` process: packed and loose refs, a worktree and a detached HEAD."""

    main, feature = "a" * 40, "b" * 40
    git = tmp_path / "repo" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        f"# pack-refs with: peeled\n{main} refs/heads/main\n{feature} refs/heads/wt/feature\n",
        encoding="utf-8",
    )
    source = tmp_path / "repo" / "src" / "so101" / "training" / "run_manifest.py"
    assert _git_sha(source) == main
    (git / "refs" / "heads" / "main").write_text(feature + "\n", encoding="utf-8")
    assert _git_sha(source) == feature  # a loose ref wins over packed-refs

    worktree_meta = git / "worktrees" / "feature"
    worktree_meta.mkdir(parents=True)
    (worktree_meta / "HEAD").write_text("ref: refs/heads/wt/feature\n", encoding="utf-8")
    (worktree_meta / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "feature"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../repo/.git/worktrees/feature\n", encoding="utf-8")
    inside = worktree / "src" / "so101" / "training" / "run_manifest.py"
    assert _git_sha(inside) == feature

    (worktree_meta / "HEAD").write_text(main + "\n", encoding="utf-8")
    assert _git_sha(inside) == main  # detached HEAD
    (worktree_meta / "HEAD").write_text("not a revision\n", encoding="utf-8")
    assert _git_sha(inside) is None

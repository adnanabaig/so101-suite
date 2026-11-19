"""Every preset learns from the episodes its outcome labels select (review finding: LeRobot
training used to learn from failed demonstrations too).

The selection is ``kept`` (success and unlabeled) by default, ``success`` or
``all``. ``lerobot-train`` gets it as ``--dataset.episodes=[...]``: draccus 0.11
(LeRobot 0.6.1's pin) reads a ``list[int]`` field from one ``[0,2,5]`` token and
keeps the last duplicate. The flag is left out when every episode is kept.
Unreadable labels or an empty selection fail the job before ``lerobot-train``
starts, and the run manifest records what was chosen.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so101.cli import app as cli_app
from so101.config import get_settings
from so101.data.inspection import OUTCOMES_SIDECAR, set_episode_outcome
from so101.data.loader import TrainingDataError
from so101.services.app import create_app
from so101.services.jobs import JobRunner, get_runner
from so101.training import trainer as trainer_module
from so101.training.bc_baseline import bc_config_for
from so101.training.configs import POLICY_PRESETS
from so101.training.jobs import JobStatus, TrainingJob
from so101.training.preflight import InvalidTrainingArgsError
from so101.training.run_manifest import RUN_MANIFEST_NAME, checkpoint_defaults
from so101.training.toy_reach import generate_toy_reach_episodes, write_toy_reach_dataset
from so101.training.trainer import build_train_command, run_training

FAKE_MODULE = "so101_fake_episode_trainer"
# Stands in for lerobot.scripts.lerobot_train: records its argv, then trains "instantly".
FAKE_TRAINER = """
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
with open(os.environ["SO101_FAKE_TRAINER_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(argv) + "\\n")
args = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
ckpt = Path(args["output_dir"]) / "checkpoints" / "last" / "pretrained_model"
ckpt.mkdir(parents=True)
(ckpt / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
print("INFO step:5 loss:0.5", flush=True)
"""

FAILURE = {"episode": 1, "outcome": "failure", "reason": "outcome is failure"}
DISCARDED = {"episode": 3, "outcome": "discarded", "reason": "outcome is discarded"}


def _lerobot(repo_id: str, episodes: int = 4, labels: dict[int, str] | None = None) -> Path:
    """A local LeRobot dataset (metadata only) with outcome labels in its sidecar."""

    root = get_settings().datasets_dir / f"v2--{quote(repo_id, safe='')}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": episodes,
                "total_frames": 10 * episodes,
                "fps": 15,
                "features": {},
            }
        ),
        encoding="utf-8",
    )
    for episode, outcome in (labels or {}).items():
        assert set_episode_outcome(repo_id, episode, outcome)
    return root


def _job(repo_id: str, output_dir: Path, **kwargs) -> TrainingJob:
    return TrainingJob(
        repo_id=repo_id,
        policy=kwargs.pop("policy", "act"),
        steps=kwargs.pop("steps", 10),
        batch_size=kwargs.pop("batch_size", 2),
        output_dir=str(output_dir),
        **kwargs,
    )


def _episodes_flags(command: list[str]) -> list[str]:
    return [arg for arg in command if arg.startswith("--dataset.episodes")]


def _last_wins(command: list[str]) -> dict[str, str]:
    """What draccus sees: ``--key=value`` pairs, the last duplicate winning."""

    return dict(arg[2:].split("=", 1) for arg in command[3:])


def _wait_for(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def _real_mode(monkeypatch) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()


@pytest.fixture
def fake_trainer(tmp_path, monkeypatch):
    """Real mode with a stand-in ``lerobot-train`` started like LeRobot's (``python -m``)."""

    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / f"{FAKE_MODULE}.py").write_text(FAKE_TRAINER, encoding="utf-8")
    log = tmp_path / "trainer_argv.jsonl"
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [str(modules), os.environ.get("PYTHONPATH")]))
    )
    monkeypatch.setattr(trainer_module, "TRAIN_MODULE", FAKE_MODULE)
    monkeypatch.setattr(trainer_module, "TRAIN_REQUIRES", ())
    monkeypatch.setenv("SO101_FAKE_TRAINER_LOG", str(log))
    _real_mode(monkeypatch)
    yield log
    get_settings.cache_clear()


def _argvs(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def no_launch(monkeypatch) -> list[list[str]]:
    """Real mode where starting any process fails the test (records the attempt)."""

    launched: list[list[str]] = []

    def refuse(cmd, *args, **kwargs):
        launched.append(list(cmd))
        raise AssertionError(f"a process was started: {cmd}")

    monkeypatch.setattr(trainer_module, "lerobot_train_available", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    _real_mode(monkeypatch)
    yield launched
    get_settings.cache_clear()


# --- the command ------------------------------------------------------------------


def test_the_command_passes_the_kept_episodes_after_every_override(tmp_path):
    root = _lerobot("lab/kept", 5, {0: "success", 1: "failure", 3: "discarded", 4: "partial"})
    job = _job("lab/kept", tmp_path / "run", extra_args={"seed": "3", "policy.n_obs_steps": "2"})
    command = build_train_command(job)
    assert _episodes_flags(command) == ["--dataset.episodes=[0,2]"]
    # Among the suite-owned flags, after every override (draccus keeps the last).
    overrides = max(command.index("--seed=3"), command.index("--policy.n_obs_steps=2"))
    assert overrides < command.index(f"--dataset.root={root}")
    assert command.index(f"--dataset.root={root}") < command.index("--dataset.episodes=[0,2]")
    assert _last_wins(command)["dataset.episodes"] == "[0,2]"

    job.episodes = "success"
    assert _episodes_flags(build_train_command(job)) == ["--dataset.episodes=[0]"]


def test_all_or_a_selection_that_keeps_every_episode_omits_the_flag(tmp_path):
    _lerobot("lab/mixed", 3, {0: "success", 2: "failure"})
    command = build_train_command(_job("lab/mixed", tmp_path / "a", episodes="all"))
    assert _episodes_flags(command) == []
    # ... "all" never even reads the labels: an unreadable file does not block it.
    root = _lerobot("lab/broken_all", 2)
    (root / OUTCOMES_SIDECAR).write_text("{oops", encoding="utf-8")
    command = build_train_command(_job("lab/broken_all", tmp_path / "b", episodes="all"))
    assert _episodes_flags(command) == []

    _lerobot("lab/clean", 3, {1: "success"})  # success and unlabeled: all kept
    assert _episodes_flags(build_train_command(_job("lab/clean", tmp_path / "c"))) == []
    _lerobot("lab/unlabeled", 2)  # no sidecar yet
    assert _episodes_flags(build_train_command(_job("lab/unlabeled", tmp_path / "d"))) == []


def test_extra_args_cannot_override_the_selection(tmp_path, monkeypatch):
    _lerobot("lab/override", 4, {1: "failure", 3: "discarded"})
    job = _job("lab/override", tmp_path / "run", extra_args={"dataset.episodes": "[1,3]"})
    with pytest.raises(InvalidTrainingArgsError, match="not an allowed"):
        build_train_command(job)
    # An override that slipped past validation (a preset's own) comes first and loses.
    monkeypatch.setitem(POLICY_PRESETS["act"].extra_args, "dataset.episodes", "[1,3]")
    command = build_train_command(_job("lab/override", tmp_path / "run"))
    assert _episodes_flags(command) == ["--dataset.episodes=[1,3]", "--dataset.episodes=[0,2]"]
    assert _last_wins(command)["dataset.episodes"] == "[0,2]"


def test_a_hub_dataset_trains_only_with_all(tmp_path):
    job = _job("someone/hub_only", tmp_path / "run", remote_dataset=True)
    with pytest.raises(TrainingDataError, match="Hub dataset has none here.*--episodes all"):
        build_train_command(job)
    job.episodes = "success"
    with pytest.raises(TrainingDataError, match="'success' needs the outcome labels"):
        build_train_command(job)
    job.episodes = "all"
    command = build_train_command(job)
    assert _episodes_flags(command) == []
    assert not any(arg.startswith("--dataset.root=") for arg in command)


def test_an_unknown_selection_is_refused(tmp_path):
    _lerobot("lab/unknown", 2)
    with pytest.raises(InvalidTrainingArgsError, match="episodes must be one of kept, success"):
        build_train_command(_job("lab/unknown", tmp_path / "run", episodes="best"))


# --- failing before launch ----------------------------------------------------------


def _broken_labels(repo_id: str) -> None:
    root = _lerobot(repo_id, 2)
    (root / OUTCOMES_SIDECAR).write_text("{oops", encoding="utf-8")


REFUSALS = [
    pytest.param(_broken_labels, "kept", "outcome labels are unreadable", id="unreadable"),
    pytest.param(
        lambda repo: _lerobot(repo, 2, {0: "failure", 1: "discarded"}),
        "kept",
        "no episodes in .* for selection 'kept'",
        id="nothing-kept",
    ),
    pytest.param(
        lambda repo: _lerobot(repo, 2),
        "success",
        "no episodes in .* for selection 'success'",
        id="nothing-successful",
    ),
]


@pytest.mark.parametrize(("make", "selection", "message"), REFUSALS)
def test_a_selection_the_labels_cannot_honour_fails_before_lerobot_starts(
    make, selection, message, tmp_path, no_launch
):
    make("lab/refused")
    output = tmp_path / "fresh" / "run"
    with pytest.raises(TrainingDataError, match=message):
        list(run_training(_job("lab/refused", output, episodes=selection)))
    assert no_launch == []
    assert not output.parent.exists()  # nothing written either

    # Through the job runner (labels can change between submit and start).
    runner = JobRunner()
    job = runner.submit("lab/refused", "act", 10, 2, episodes=selection)
    _wait_for(lambda: job.status in (JobStatus.FAILED, JobStatus.SUCCEEDED))
    runner.shutdown()
    assert job.status is JobStatus.FAILED
    assert job.error is not None
    assert re.search(message, job.error)
    assert no_launch == []
    assert not Path(job.output_dir).exists()


def test_mock_training_refuses_what_real_training_would(tmp_path):
    _lerobot("lab/mock_refused", 2, {0: "failure", 1: "failure"})
    output = tmp_path / "mock" / "run"
    with pytest.raises(TrainingDataError, match="for selection 'kept'"):
        list(run_training(_job("lab/mock_refused", output)))
    assert not output.exists()


# --- the API and the CLI ------------------------------------------------------------


def test_the_api_passes_the_selection_to_lerobot_train(fake_trainer):
    _lerobot("lab/api", 4, {0: "success", 1: "failure", 3: "discarded"})
    body = {"repo_id": "lab/api", "policy": "act", "steps": 5, "batch_size": 1}
    with TestClient(create_app()) as client:
        response = client.post("/api/training/jobs", json={**body, "episodes": "success"})
        assert response.status_code == 200, response.text
        assert response.json()["episodes"] == "success"
        job_id = response.json()["id"]
        terminal = {"succeeded", "failed", "cancelled"}
        _wait_for(lambda: client.get(f"/api/training/jobs/{job_id}").json()["status"] in terminal)
        done = client.get(f"/api/training/jobs/{job_id}").json()
        assert done["status"] == "succeeded", done
        job = get_runner().get(job_id)
        assert job is not None and job.episodes == "success"

        # Omitted means kept.
        default = client.post("/api/training/jobs", json=body)
        assert default.status_code == 200 and default.json()["episodes"] == "kept"
        default_id = default.json()["id"]
        _wait_for(
            lambda: client.get(f"/api/training/jobs/{default_id}").json()["status"] in terminal
        )

        # Refused synchronously: an unknown value, nothing selected, a conflict.
        bad = client.post("/api/training/jobs", json={**body, "episodes": "best"})
        assert bad.status_code == 422
        _lerobot("lab/api_none", 2)
        none = client.post(
            "/api/training/jobs", json={**body, "repo_id": "lab/api_none", "episodes": "success"}
        )
        assert none.status_code == 422 and "for selection 'success'" in none.json()["detail"]
        conflict = client.post(
            "/api/training/jobs",
            json={
                **body,
                "policy": "bc_baseline",
                "episodes": "success",
                "extra_args": {"episodes": "all"},
            },
        )
        assert conflict.status_code == 422 and "conflicts" in conflict.json()["detail"]

    argvs = _argvs(fake_trainer)
    assert len(argvs) == 2
    assert _episodes_flags(argvs[0]) == ["--dataset.episodes=[0]"]
    assert _episodes_flags(argvs[1]) == ["--dataset.episodes=[0,2]"]
    manifest = json.loads((Path(job.output_dir) / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["episode_selection"] == {
        "selection": "success",
        "episodes": [0],
        "skipped": [
            FAILURE,
            {"episode": 2, "outcome": "unlabeled", "reason": "outcome is unlabeled"},
            DISCARDED,
        ],
    }


def _cli(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli_app, list(args))
    return result.exit_code, " ".join(result.output.split())


def test_the_cli_selection_applies_to_lerobot_presets_and_is_recorded(fake_trainer, tmp_path):
    _lerobot("lab/cli", 4, {0: "success", 1: "failure", 3: "discarded"})
    kept_run, all_run = tmp_path / "out" / "kept", tmp_path / "out" / "all"
    code, text = _cli("train", "lab/cli", "--steps", "5", "--output-dir", str(kept_run))
    assert code == 0, text
    code, text = _cli(
        "train", "lab/cli", "--steps", "5", "--episodes", "all", "--output-dir", str(all_run)
    )
    assert code == 0, text

    kept_argv, all_argv = _argvs(fake_trainer)
    assert _episodes_flags(kept_argv) == ["--dataset.episodes=[0,2]"]
    assert _episodes_flags(all_argv) == []
    kept = json.loads((kept_run / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert kept["trainer"] == "lerobot"
    assert kept["episode_selection"] == {
        "selection": "kept",
        "episodes": [0, 2],
        "skipped": [FAILURE, DISCARDED],
    }
    every = json.loads((all_run / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert every["episode_selection"] == {"selection": "all", "episodes": None, "skipped": []}
    # The manifest's readers are unaffected.
    defaults = checkpoint_defaults(kept_run / "checkpoints" / "last" / "pretrained_model")
    assert (defaults.fps, defaults.dataset) == (15, "lab/cli")


def test_the_cli_refuses_a_selection_before_anything_starts(fake_trainer):
    _lerobot("lab/cli_none", 2, {0: "failure", 1: "partial"})
    code, text = _cli("train", "lab/cli_none", "--steps", "5")
    assert code == 1 and "cannot train" in text and "for selection 'kept'" in text
    code, text = _cli("train", "someone/hub_only", "--hub-dataset", "--steps", "5")
    assert code == 1 and "Hub dataset has none here" in text and "--episodes all" in text
    code, text = _cli("train", "lab/cli_none", "--episodes", "best")
    assert code == 1 and "episodes must be one of" in text
    assert not fake_trainer.exists()  # lerobot-train never started
    assert not any(get_settings().outputs_dir.glob("*"))


# --- bc_baseline, job records and manifests -------------------------------------------


def test_bc_baseline_reads_the_job_field_and_still_the_extra_arg():
    def job(**kwargs) -> TrainingJob:
        return TrainingJob(
            repo_id="x/y", policy="bc_baseline", steps=1, batch_size=1, output_dir="o", **kwargs
        )

    assert bc_config_for(job()).episodes == "kept"
    assert bc_config_for(job(episodes="all")).episodes == "all"
    assert bc_config_for(job(extra_args={"episodes": "success"})).episodes == "success"
    assert bc_config_for(job(episodes="all", extra_args={"episodes": "ALL"})).episodes == "all"
    with pytest.raises(InvalidTrainingArgsError, match="conflicts"):
        bc_config_for(job(episodes="success", extra_args={"episodes": "all"}))


def test_job_records_without_a_selection_restore_as_kept():
    records = get_settings().outputs_dir / ".jobs"
    records.mkdir(parents=True)
    old = {
        "id": "0123456789ab",
        "repo_id": "lab/old",
        "policy": "act",
        "steps": 10,
        "batch_size": 1,
        "output_dir": "/nowhere/act_old",
        "status": "succeeded",
        "created_at": 1.0,
        "extra_args": {},
        "remote_dataset": False,
        "include_depth": False,
    }
    (records / f"{old['id']}.json").write_text(json.dumps(old), encoding="utf-8")
    new = {**old, "id": "ba9876543210", "episodes": "success"}
    (records / f"{new['id']}.json").write_text(json.dumps(new), encoding="utf-8")
    runner = JobRunner(restore=True)
    try:
        restored = {job.id: job for job in runner.list()}
        assert restored[old["id"]].status is JobStatus.SUCCEEDED
        assert restored[old["id"]].episodes == "kept"
        assert restored[new["id"]].episodes == "success"
        assert restored[new["id"]].to_dict()["episodes"] == "success"
    finally:
        runner.shutdown()


def test_mock_and_baseline_manifests_record_the_selection(tmp_path):
    _lerobot("lab/mock", 3, {1: "failure"})
    output = tmp_path / "mock_run"
    list(run_training(_job("lab/mock", output, steps=5)))
    manifest = json.loads((output / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["trainer"] == "mock"
    assert manifest["episode_selection"] == {
        "selection": "kept",
        "episodes": [0, 2],
        "skipped": [FAILURE],
    }

    # The baseline applies the selection as it loads the data and lists the
    # episodes in its checkpoint manifest; the run manifest names the selection.
    write_toy_reach_dataset("toy/select", generate_toy_reach_episodes(4, seed=0))
    output = tmp_path / "bc_run"
    job = _job(
        "toy/select",
        output,
        policy="bc_baseline",
        steps=20,
        batch_size=16,
        episodes="all",
        extra_args={"hidden": "8", "eval_every": "10"},
    )
    list(run_training(job))
    manifest = json.loads((output / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["episode_selection"] == {"selection": "all", "episodes": None, "skipped": None}
    policy = json.loads(
        (output / "checkpoints" / "last" / "pretrained_model" / "policy_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["dataset"]["selection"] == "all"
    assert policy["dataset"]["episodes"] == [0, 1, 2, 3]


def test_checkpoint_defaults_still_read_a_manifest_without_a_selection(tmp_path):
    run = tmp_path / "old_run"
    checkpoint = run / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (run / RUN_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trainer": "lerobot",
                "policy_type": "act",
                "dataset": {"repo_id": "lab/old", "remote": False, "episodes": 4},
                "fps": 30,
                "task": "pick the cube",
            }
        ),
        encoding="utf-8",
    )
    defaults = checkpoint_defaults(checkpoint)
    assert (defaults.task, defaults.fps, defaults.policy_type, defaults.dataset) == (
        "pick the cube",
        30,
        "act",
        "lab/old",
    )

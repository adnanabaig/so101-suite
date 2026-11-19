"""The argv so101 builds must parse and validate with the pinned lerobot-train.

Audit B1/B2: every real training job failed at start-up because the command
was never checked against LeRobot's own parser. These tests feed
``build_train_command`` output through LeRobot's ``parser.wrap`` (the exact
entry point of ``lerobot-train``) and ``TrainPipelineConfig.validate()``.

They need ``lerobot`` and are skipped in the default (torch-free) venv. To run
them, use a venv with ``so101[train]`` or ``lerobot==0.6.1``. No network
access is needed and nothing is uploaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("draccus")
pytest.importorskip("lerobot.configs.train")

import lerobot.policies  # noqa: E402,F401  (registers the --policy.type choices)
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402

from so101.config import get_settings  # noqa: E402
from so101.training.configs import POLICY_PRESETS  # noqa: E402
from so101.training.jobs import TrainingJob  # noqa: E402
from so101.training.preflight import InvalidTrainingArgsError  # noqa: E402
from so101.training.trainer import build_train_command, train_launcher  # noqa: E402


def _parse_like_lerobot_train(cmd: list[str], monkeypatch) -> TrainPipelineConfig:
    """Parse ``cmd`` the way ``lerobot-train`` does, then run its start-up checks."""

    def capture(cfg):
        cfg.validate()  # lerobot_train.train() calls this before touching data
        return cfg

    # parser.wrap reads the config class from the real annotation object; this
    # module uses postponed (string) annotations, so set it explicitly.
    capture.__annotations__ = {"cfg": TrainPipelineConfig}
    # ``python -m lerobot.scripts.lerobot_train ARGS`` gives the module
    # sys.argv == [<its path>, *ARGS]; parser.wrap reads sys.argv[1:].
    launcher = train_launcher()
    assert cmd[: len(launcher)] == launcher
    monkeypatch.setattr(sys, "argv", ["lerobot_train.py", *cmd[len(launcher) :]])
    return parser.wrap()(capture)()


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")


def _job(policy: str, output_dir: Path, **kwargs) -> TrainingJob:
    return TrainingJob(
        repo_id=kwargs.pop("repo_id", "lerobot/so101_contract"),
        policy=policy,
        steps=kwargs.pop("steps", 1234),
        batch_size=kwargs.pop("batch_size", 4),
        output_dir=str(output_dir),
        # The default repo is not local, so it has no outcome labels to select by.
        episodes=kwargs.pop("episodes", "all"),
        **kwargs,
    )


LEROBOT_PRESETS = sorted(name for name, p in POLICY_PRESETS.items() if p.trainer == "lerobot")


@pytest.mark.parametrize("name", LEROBOT_PRESETS)
def test_every_preset_command_parses_and_validates(name, tmp_path, monkeypatch):
    preset = POLICY_PRESETS[name]
    output = tmp_path / "run"
    job = _job(name, output, extra_args={"seed": "3", "log_freq": "50"})
    cfg = _parse_like_lerobot_train(build_train_command(job), monkeypatch)

    assert cfg.policy.type == preset.policy_type
    assert cfg.policy.push_to_hub is False  # B2: 0.6.1 defaults to True
    assert cfg.output_dir == output
    assert not output.exists()  # B1: lerobot-train must create it itself
    assert cfg.steps == 1234
    assert cfg.batch_size == 4
    assert cfg.seed == 3
    assert cfg.log_freq == 50
    assert cfg.dataset.repo_id == "lerobot/so101_contract"
    assert cfg.wandb.enable is False
    pretrained = preset.extra_args.get("policy.pretrained_path")
    if pretrained:
        assert str(cfg.policy.pretrained_path) == pretrained


def test_local_dataset_root_is_passed_through(tmp_path, monkeypatch):
    repo_id = "local/contract"
    root = get_settings().datasets_dir / f"v2--{quote(repo_id, safe='')}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 10, "fps": 30, "features": {}}),
        encoding="utf-8",
    )
    job = _job("act", tmp_path / "run", repo_id=repo_id)
    cfg = _parse_like_lerobot_train(build_train_command(job), monkeypatch)
    assert cfg.dataset.repo_id == repo_id
    assert Path(cfg.dataset.root) == root


def test_kept_episodes_reach_lerobot_as_its_episode_list(tmp_path, monkeypatch):
    """``--dataset.episodes=[0,2]`` parses into DatasetConfig.episodes and wins over a duplicate."""
    from so101.data.inspection import set_episode_outcome

    repo_id = "local/labeled"
    root = get_settings().datasets_dir / f"v2--{quote(repo_id, safe='')}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 4, "total_frames": 40, "fps": 30, "features": {}}),
        encoding="utf-8",
    )
    assert set_episode_outcome(repo_id, 1, "failure")
    assert set_episode_outcome(repo_id, 3, "discarded")
    cmd = build_train_command(_job("act", tmp_path / "run", repo_id=repo_id, episodes="kept"))
    after = len(train_launcher()) + 2  # right after --policy.type/--policy.device
    cmd[after:after] = ["--dataset.episodes=[1,3]"]
    cfg = _parse_like_lerobot_train(cmd, monkeypatch)
    assert cfg.dataset.episodes == [0, 2]
    everything = build_train_command(_job("act", tmp_path / "all", repo_id=repo_id))
    assert _parse_like_lerobot_train(everything, monkeypatch).dataset.episodes is None


def test_owned_flags_win_over_a_smuggled_duplicate(tmp_path, monkeypatch):
    """Draccus keeps the last duplicate, and so101 appends its flags last."""

    output = tmp_path / "run"
    cmd = build_train_command(_job("act", output))
    smuggled = [
        f"--output_dir={tmp_path / 'elsewhere'}",
        "--policy.push_to_hub=true",
        "--steps=1",
        "--dataset.repo_id=someone/else",
    ]
    after = len(train_launcher()) + 2  # right after --policy.type/--policy.device
    cmd[after:after] = smuggled
    cfg = _parse_like_lerobot_train(cmd, monkeypatch)
    assert cfg.output_dir == output
    assert cfg.policy.push_to_hub is False
    assert cfg.steps == 1234
    assert cfg.dataset.repo_id == "lerobot/so101_contract"


def test_existing_output_dir_is_refused_by_lerobot_too(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _parse_like_lerobot_train(build_train_command(_job("act", output)), monkeypatch)


@pytest.mark.parametrize(
    "extra",
    [
        {"policy.push_to_hub": "true"},
        {"output_dir": "elsewhere"},
        {"resume": "true"},
        {"policy.path": "lerobot/act_base"},
    ],
)
def test_dangerous_overrides_never_reach_lerobot(extra, tmp_path):
    with pytest.raises(InvalidTrainingArgsError):
        build_train_command(_job("act", tmp_path / "run", extra_args=extra))

"""``so101 sim``: info, render, demos (EpisodeRecorder + auto labels) and eval."""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

pytest.importorskip("mujoco")

from typer.testing import CliRunner  # noqa: E402

from so101 import JOINT_NAMES, cli_sim  # noqa: E402
from so101.cli import app  # noqa: E402
from so101.config import get_settings  # noqa: E402
from so101.data import inspection  # noqa: E402
from so101.data.dataset import DatasetBusyError, dataset_root  # noqa: E402
from so101.data.recorder import EpisodeRecorder, RecordingConfig  # noqa: E402
from so101.simulation.mjcf import MODEL_COMMIT  # noqa: E402
from so101.simulation.tasks import TASK_NAMES, make_task  # noqa: E402

runner = CliRunner()
NO_GL = "rendering is unavailable"


def sim(*args: str):
    return runner.invoke(app, ["sim", *args])


def meta(repo_id: str) -> dict:
    return json.loads((dataset_root(repo_id) / "meta.json").read_text(encoding="utf-8"))


def episode(repo_id: str, index: int) -> tuple[dict, list[dict]]:
    directory = dataset_root(repo_id) / f"episode_{index:06d}"
    manifest = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    lines = (directory / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    return manifest, [json.loads(line) for line in lines if line.strip()]


def last_run(repo_id: str) -> dict:
    manifest = json.loads(
        (dataset_root(repo_id) / cli_sim.DEMOS_MANIFEST).read_text(encoding="utf-8")
    )
    return manifest["runs"][-1]


def test_info_json():
    result = sim("info", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["model"]["commit"] == MODEL_COMMIT
    assert data["control_hz"] == 30 and data["substeps_per_control_step"] == 8
    assert data["cameras"] == ["front", "wrist_cam"]
    assert [task["name"] for task in data["tasks"]] == list(TASK_NAMES)
    assert data["tasks"][1]["env_state_names"] == list(make_task("pick_lift").env_state_names)
    assert data["backend"] == "mock"  # conftest sets SO101_MOCK=1
    assert set(data["render"]) == {"available", "error", "jpeg"}


def test_render_writes_a_png(tmp_path):
    out = tmp_path / "shots" / "front.png"
    result = sim("render", "--out", str(out), "--task", "pick_lift", "--seed", "1",
                 "--width", "64", "--height", "48")  # fmt: skip
    if result.exit_code == 1 and NO_GL in result.output:
        pytest.skip(result.output)
    assert result.exit_code == 0, result.output
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (64, 48)


@pytest.mark.parametrize(
    "args",
    [
        ("--out", "x.gif"),
        ("--out", "x.png", "--camera", "side"),
        ("--out", "x.png", "--width", "8"),
        ("--out", "x.png", "--task", "stack"),
        ("--out", "x.png", "--seed", "-1"),
    ],
)
def test_render_rejects_bad_parameters(args):
    assert sim("render", *args).exit_code == 2


def test_demos_record_labeled_deterministic_episodes():
    result = sim("demos", "sim/lift", "--task", "pick_lift", "--episodes", "2", "--seed", "5")
    assert result.exit_code == 0, result.output

    info = meta("sim/lift")
    task = make_task("pick_lift")
    assert info["episodes"] == 2 and info["fps"] == 30
    assert info["task"] == task.instruction
    assert info["environment_state_names"] == list(task.env_state_names)
    run = last_run("sim/lift")
    assert run["seeds"] == [5, 6] and run["successes"] == 2
    assert run["features"] == [
        "observation.state",
        "action",
        "desired_action",
        "observation.environment_state",
    ]
    assert run["provenance"]["model"]["commit"] == MODEL_COMMIT
    for index, entry in enumerate(run["episodes"]):
        manifest, frames = episode("sim/lift", index)
        assert manifest["outcome"] == entry["outcome"] == "success"
        assert manifest["outcome_source"] == cli_sim.SIM_OUTCOME_SOURCE
        assert manifest["outcome_detail"]["seed"] == entry["seed"]
        assert manifest["outcome_detail"]["success"] is True
        assert entry["label"] == "auto-labeled"
        assert manifest["frames"] == entry["frames"] == len(frames)
        assert len(frames) <= task.episode_steps()
        for frame in frames:
            assert len(frame["observation.state"]) == len(JOINT_NAMES)
            assert len(frame["action"]) == len(frame["desired_action"]) == len(JOINT_NAMES)
            assert len(frame["observation.environment_state"]) == task.env_state_dim
        # The cube is lifted by the end of a successful episode.
        assert (
            frames[-1]["observation.environment_state"][2]
            > frames[0]["observation.environment_state"][2] + 0.04
        )

    # Episode i used seed + i, so rerunning seed 6 alone reproduces episode 1 exactly.
    again = sim("demos", "sim/again", "--task", "pick_lift", "--episodes", "1", "--seed", "6")
    assert again.exit_code == 0, again.output
    original = (dataset_root("sim/lift") / "episode_000001" / "frames.jsonl").read_bytes()
    rerun = (dataset_root("sim/again") / "episode_000000" / "frames.jsonl").read_bytes()
    assert original == rerun


def test_demos_refuse_to_mix_layouts_in_one_dataset():
    assert sim("demos", "sim/mixed", "--task", "reach", "--episodes", "1").exit_code == 0
    result = sim("demos", "sim/mixed", "--task", "pick_lift", "--episodes", "1")
    assert result.exit_code == 1
    assert "Use a new repo id" in result.output
    assert meta("sim/mixed")["episodes"] == 1
    # The same layout appends.
    assert sim("demos", "sim/mixed", "--task", "reach", "--episodes", "1").exit_code == 0
    assert meta("sim/mixed")["episodes"] == 2
    manifest = json.loads(
        (dataset_root("sim/mixed") / cli_sim.DEMOS_MANIFEST).read_text(encoding="utf-8")
    )
    assert len(manifest["runs"]) == 2


def test_demos_record_camera_frames():
    result = sim("demos", "sim/cams", "--task", "reach", "--episodes", "1",
                 "--cameras", "front", "--image-size", "64x48")  # fmt: skip
    if result.exit_code == 1 and NO_GL in result.output:
        pytest.skip(result.output)
    assert result.exit_code == 0, result.output
    info = meta("sim/cams")
    assert info["camera_keys"] == ["front"]
    assert info["camera_feature_names"] == {"front": "front"}
    assert last_run("sim/cams")["features"][-1] == "observation.images.front"
    manifest, _ = episode("sim/cams", 0)
    chunks = sorted((dataset_root("sim/cams") / "episode_000000" / "images").glob("*/chunk_*.npy"))
    images = np.concatenate([np.load(chunk) for chunk in chunks])
    assert images.shape == (manifest["frames"], 48, 64, 3) and images.dtype == np.uint8
    assert images.std() > 1.0  # a rendered scene, not a blank frame


def test_demos_refuse_to_mix_frame_sizes_in_one_dataset():
    def demos(size: str):
        result = sim("demos", "sim/sizes", "--task", "reach", "--episodes", "1",
                     "--cameras", "front", "--image-size", size)  # fmt: skip
        return result, " ".join(result.output.split())  # undo the console's line wrapping

    first, output = demos("64x48")
    if first.exit_code == 1 and NO_GL in output:
        pytest.skip(output)
    assert first.exit_code == 0, output
    assert meta("sim/sizes")["camera_shapes"] == {"front": [48, 64, 3]}
    refused, output = demos("80x60")
    assert refused.exit_code == 1
    assert "front frames of shape [48, 64, 3]" in output and "--image-size 64x48" in output
    assert meta("sim/sizes")["episodes"] == 1
    assert demos("64x48")[0].exit_code == 0  # the same size appends
    assert meta("sim/sizes")["episodes"] == 2

    # A dataset recorded before camera_shapes existed: the shape comes from its chunks.
    path = dataset_root("sim/sizes") / "meta.json"
    older = meta("sim/sizes")
    del older["camera_shapes"]
    path.write_text(json.dumps(older), encoding="utf-8")
    refused, output = demos("80x60")
    assert refused.exit_code == 1 and "front frames of shape [48, 64, 3]" in output
    assert demos("64x48")[0].exit_code == 0
    info = meta("sim/sizes")
    assert info["episodes"] == 3 and info["camera_shapes"] == {"front": [48, 64, 3]}
    chunks = sorted(dataset_root("sim/sizes").glob("episode_*/images/*/chunk_*.npy"))
    assert {np.load(chunk, mmap_mode="r").shape[1:] for chunk in chunks} == {(48, 64, 3)}


@pytest.mark.parametrize(
    "args",
    [
        ("sim/bad", "--cameras", "side"),
        ("sim/bad", "--image-size", "big"),
        ("sim/bad", "--task", "stack"),
        ("sim/bad", "--episodes", "0"),
        ("sim/bad", "--seed", "-1"),
        ("bad repo id",),
    ],
)
def test_demos_reject_bad_parameters(args):
    assert sim("demos", *args).exit_code == 2


def test_demos_record_as_the_sim_backend_outside_mock(monkeypatch):
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.delenv("SO101_BACKEND", raising=False)
    get_settings.cache_clear()
    assert get_settings().backend == "real"
    seen: list[str | None] = []
    original = cli_sim.label_episode

    def spy(*args, **kwargs):
        seen.append(get_settings().backend)
        return original(*args, **kwargs)

    monkeypatch.setattr(cli_sim, "label_episode", spy)
    result = sim("demos", "sim/backend", "--task", "reach", "--episodes", "1")
    assert result.exit_code == 0, result.output
    assert seen == ["sim"]
    assert last_run("sim/backend")["recorded_as_backend"] == "sim"
    assert "SO101_BACKEND" not in os.environ
    assert get_settings().backend == "real"


def test_interrupted_demo_keeps_no_partial_episode(monkeypatch):
    original = EpisodeRecorder.add
    calls = {"n": 0}

    def interrupt_mid_episode(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeRecorder, "add", interrupt_mid_episode)
    result = sim("demos", "sim/interrupted", "--task", "reach", "--episodes", "2")
    assert result.exit_code == 130
    root = dataset_root("sim/interrupted")
    assert not list(root.glob("episode_*"))
    assert len(list(root.glob("incomplete_*"))) == 1  # the 4 frames, kept apart
    # The dataset is free again.
    EpisodeRecorder(RecordingConfig(repo_id="sim/interrupted", camera_keys=[])).abort("test")


def test_eval_expert_and_hold(tmp_path):
    out = tmp_path / "expert.json"
    result = sim("eval", "--policy", "expert", "--task", "pick_lift", "--episodes", "2",
                 "--seed", "5", "--out", str(out))  # fmt: skip
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["successes"] == 2 and report["seeds"] == [5, 6]
    assert [o["seed"] for o in report["outcomes"]] == [5, 6]
    low, high = report["wilson95"]
    assert 0.3 < low < 0.4 and high == 1.0
    assert report["provenance"]["model"]["commit"] == MODEL_COMMIT
    assert report["real_time_factor"] > 1.0

    hold = sim("eval", "--policy", "hold", "--task", "pick_lift", "--episodes", "2")
    assert hold.exit_code == 0, hold.output
    saved = list((get_settings().outputs_dir / "sim_eval").glob("pick_lift-hold-*.json"))
    assert len(saved) == 1
    report = json.loads(saved[0].read_text(encoding="utf-8"))
    assert report["successes"] == 0 and report["success_rate"] == 0.0


def test_eval_rejects_unknown_policies_and_tasks(tmp_path):
    missing = sim("eval", "--policy", str(tmp_path / "no_model"), "--episodes", "1")
    assert missing.exit_code == 2 and "unsupported model path" in missing.output
    unknown = tmp_path / "model.bin"
    unknown.write_bytes(b"")
    assert sim("eval", "--policy", str(unknown), "--episodes", "1").exit_code == 2
    assert sim("eval", "--task", "stack").exit_code == 2
    negative = sim("eval", "--seed", "-1", "--episodes", "1")
    assert negative.exit_code == 2 and "x>=0" in negative.output


# -- outcome labels ----------------------------------------------------------------------
def test_outcome_source_is_a_valid_auto_source():
    validated = inspection._validated_outcome_source(cli_sim.SIM_OUTCOME_SOURCE)
    assert validated == cli_sim.SIM_OUTCOME_SOURCE != inspection.HUMAN_OUTCOME_SOURCE


def test_label_uses_a_source_aware_setter(monkeypatch):
    calls = []

    def setter(repo_id, episode, outcome, notes="", *, source="human", detail=None):
        calls.append((repo_id, episode, outcome, source, detail))
        return len(calls) == 1  # the second call meets a human label

    monkeypatch.setattr(inspection, "set_episode_outcome", setter)
    detail = {"seed": 3, "success": True}
    assert cli_sim.label_episode("sim/x", 0, True, detail) == "auto-labeled"
    assert cli_sim.label_episode("sim/x", 1, False, detail) == "kept the human label"
    assert calls == [
        ("sim/x", 0, "success", cli_sim.SIM_OUTCOME_SOURCE, detail),
        ("sim/x", 1, "failure", cli_sim.SIM_OUTCOME_SOURCE, detail),
    ]


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("custom-v9"),
        DatasetBusyError("busy"),
        FileNotFoundError("gone"),
        inspection.OutcomeLabelsUnreadableError("so101_outcomes.json is unreadable"),
    ],
)
def test_a_label_that_cannot_be_stored_is_an_error(monkeypatch, error):
    def setter(repo_id, episode, outcome, notes="", *, source="human", detail=None):
        raise error

    monkeypatch.setattr(inspection, "set_episode_outcome", setter)
    with pytest.raises(cli_sim.LabelNotStoredError) as caught:
        cli_sim.label_episode("sim/x", 3, False, {})
    assert caught.value.cause is error and caught.value.outcome == "failure"
    assert "episode 3 of sim/x was saved, but its failure label could not be stored" in str(
        caught.value
    )
    assert "reads as unlabeled" in str(caught.value)


def test_demos_stop_loudly_when_a_label_cannot_be_stored(monkeypatch):
    real = inspection.set_episode_outcome

    def busy_on_the_second(repo_id, episode, outcome, notes="", *, source="human", detail=None):
        if episode == 1:
            raise DatasetBusyError("dataset 'sim/unlabeled' is currently in use")
        return real(repo_id, episode, outcome, notes, source=source, detail=detail)

    # ``sim demos`` retains the recorder's dataset-operation lock across
    # publication and automatic labeling, so it uses the locked writer.
    monkeypatch.setattr(inspection, "_set_episode_outcome_locked", busy_on_the_second)
    result = sim("demos", "sim/unlabeled", "--task", "reach", "--episodes", "3")
    assert result.exit_code == 1
    output = " ".join(result.output.split())  # Rich wraps long lines
    assert "episode 1 of sim/unlabeled was saved, but its success label could not be stored" in (
        output
    )
    assert "stopped: 2 episode(s)" in output
    run = last_run("sim/unlabeled")
    assert [entry["label"] for entry in run["episodes"]] == [
        "auto-labeled",
        "NOT LABELED (dataset 'sim/unlabeled' is currently in use)",
    ]
    assert [e.outcome for e in inspection.inspect_dataset("sim/unlabeled").episodes] == [
        "success",
        "unlabeled",
    ]


# -- the default install records LeRobot datasets (data review finding 1) ------------------
class _FakeLeRobotDataset:
    """LeRobot 0.6.1's dataset writer as far as the recorder uses it (metadata only)."""

    def __init__(self, repo_id, root, **_):
        self.repo_id, self.root = repo_id, root
        info = json.loads((root / "meta" / "info.json").read_text())
        self.fps, self.features = info["fps"], info["features"]
        self.frames: list[dict] = []

    @classmethod
    def create(cls, repo_id, fps, features, root, use_videos=True, **_):
        (root / "meta").mkdir(parents=True, exist_ok=True)
        info = {
            "fps": fps,
            "features": {k: {**v, "shape": list(v["shape"])} for k, v in features.items()},
            "total_episodes": 0,
            "total_frames": 0,
        }
        (root / "meta" / "info.json").write_text(json.dumps(info))
        return cls(repo_id, root)

    @classmethod
    def resume(cls, repo_id, root, **_):
        return cls(repo_id, root)

    @property
    def num_episodes(self) -> int:
        return json.loads((self.root / "meta" / "info.json").read_text())["total_episodes"]

    def add_frame(self, frame) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        path = self.root / "meta" / "info.json"
        info = json.loads(path.read_text())
        info["total_episodes"] += 1
        info["total_frames"] += len(self.frames)
        path.write_text(json.dumps(info))
        self.frames = []

    def finalize(self) -> None:
        pass


def test_demos_on_the_default_install_store_every_label(monkeypatch):
    import sys
    from dataclasses import replace
    from types import ModuleType

    from so101.data.loader import selected_episodes
    from so101.simulation.rollout import EpisodeRunner

    package, datasets = ModuleType("lerobot"), ModuleType("lerobot.datasets")
    datasets.LeRobotDataset = _FakeLeRobotDataset
    package.datasets = datasets
    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.delenv("SO101_MOCK", raising=False)  # a real install, recording as sim
    get_settings.cache_clear()
    run_episode = EpisodeRunner.run

    def every_other_episode_fails(self, policy, seed, **kwargs):
        result = run_episode(self, policy, seed, **kwargs)
        return replace(result, success=seed % 2 == 0)

    monkeypatch.setattr(EpisodeRunner, "run", every_other_episode_fails)
    result = sim("demos", "sim/lerobot", "--task", "reach", "--episodes", "4", "--seed", "0")
    assert result.exit_code == 0, result.output

    detail = inspection.inspect_dataset("sim/lerobot")
    assert detail.summary.format == "lerobot"
    assert [(e.outcome, e.outcome_source) for e in detail.episodes] == [
        ("success", cli_sim.SIM_OUTCOME_SOURCE),
        ("failure", cli_sim.SIM_OUTCOME_SOURCE),
    ] * 2
    assert detail.outcome_counts["unlabeled"] == 0
    assert [entry["label"] for entry in last_run("sim/lerobot")["episodes"]] == ["auto-labeled"] * 4
    # The default selection drops the failed demos, and "success" finds the rest.
    assert selected_episodes("sim/lerobot", "kept")[0] == [0, 2]
    assert selected_episodes("sim/lerobot", "success")[0] == [0, 2]
    stored = json.loads((dataset_root("sim/lerobot") / "so101_outcomes.json").read_text())
    assert stored["episodes"]["1"]["outcome_detail"]["seed"] == 1

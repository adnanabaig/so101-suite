"""Training-data loader and outcome provenance (human labels always win)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.data.dataset import DatasetNotFoundError, get_dataset
from so101.data.inspection import (
    OutcomeLabelsUnreadableError,
    inspect_dataset,
    set_episode_outcome,
)
from so101.data.loader import ENV_STATE_KEY, TrainingDataError, load_training_data
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.teleop.controller import TeleopStep


def _record(repo_id: str, episodes: int, frames: int = 5) -> None:
    for episode in range(episodes):
        recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=10, camera_keys=[]))
        for index in range(frames):
            joints = {f"{joint}.pos": float(index + episode) for joint in JOINT_NAMES}
            recorder.add(TeleopStep(t=1.0 + index / 10, observation=joints, action=joints))
        recorder.save_episode()


def _frames_path(repo_id: str, episode: int):
    info = get_dataset(repo_id)
    assert info is not None
    return Path(info.path) / f"episode_{episode:06d}" / "frames.jsonl"


def _rewrite_frames(repo_id: str, episode: int, mutate) -> None:
    path = _frames_path(repo_id, episode)
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _episode_json(repo_id: str, episode: int) -> dict:
    return json.loads(
        (_frames_path(repo_id, episode).parent / "episode.json").read_text(encoding="utf-8")
    )


def test_auto_label_records_provenance_and_never_overrides_human():
    _record("test/outcomes", 3)
    assert set_episode_outcome("test/outcomes", 0, "failure", "operator bumped the cup")
    assert set_episode_outcome(
        "test/outcomes", 1, "success", source="auto:sim_predicate", detail={"dist_m": 0.004}
    )

    # Automation may not replace a human label ...
    assert not set_episode_outcome("test/outcomes", 0, "success", source="auto:sim_predicate")
    manifest = _episode_json("test/outcomes", 0)
    assert manifest["outcome"] == "failure"
    assert manifest["outcome_source"] == "human"
    assert manifest["notes"] == "operator bumped the cup"

    # ... but a human may replace an automatic label, which moves to the history.
    assert set_episode_outcome("test/outcomes", 1, "failure", "gripper slipped")
    manifest = _episode_json("test/outcomes", 1)
    assert (manifest["outcome"], manifest["outcome_source"]) == ("failure", "human")
    assert manifest["outcome_history"][-1]["source"] == "auto:sim_predicate"
    assert manifest["outcome_history"][-1]["detail"] == {"dist_m": 0.004}
    assert "outcome_detail" not in manifest
    assert not set_episode_outcome("test/outcomes", 1, "success", source="auto:sim_predicate")

    by_episode = {item.episode: item for item in inspect_dataset("test/outcomes").episodes}
    assert by_episode[0].outcome_source == "human"
    assert by_episode[1].outcome_source == "human"
    assert by_episode[2].outcome_source == ""


def test_auto_label_keeps_notes_and_relabels_auto_and_unlabeled():
    _record("test/outcomes_auto", 1)
    assert set_episode_outcome("test/outcomes_auto", 0, "unlabeled", "check camera 2")
    assert set_episode_outcome("test/outcomes_auto", 0, "failure", source="auto:sim_predicate")
    assert set_episode_outcome("test/outcomes_auto", 0, "success", source="auto:sim_predicate")
    manifest = _episode_json("test/outcomes_auto", 0)
    assert manifest["outcome"] == "success"
    assert manifest["notes"] == "check camera 2"
    assert [item["outcome"] for item in manifest["outcome_history"]] == ["unlabeled", "failure"]


def test_legacy_label_without_source_counts_as_human():
    _record("test/outcomes_legacy", 1)
    path = _frames_path("test/outcomes_legacy", 0).parent / "episode.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcome"] = "discarded"
    payload.pop("outcome_source", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not set_episode_outcome("test/outcomes_legacy", 0, "success", source="auto:sim")
    assert inspect_dataset("test/outcomes_legacy").episodes[0].outcome_source == "human"


@pytest.mark.parametrize(
    ("source", "detail"),
    [("robot", None), ("auto:", None), ("auto:Sim", None), ("human", {"x": float("nan")})],
)
def test_outcome_source_and_detail_are_validated(source, detail):
    _record("test/outcomes_bad", 1)
    with pytest.raises(ValueError):
        set_episode_outcome("test/outcomes_bad", 0, "success", source=source, detail=detail)
    assert _episode_json("test/outcomes_bad", 0)["outcome"] == "unlabeled"


def test_loader_selection_follows_outcomes():
    _record("test/select", 4)
    set_episode_outcome("test/select", 0, "success")
    set_episode_outcome("test/select", 1, "failure")
    set_episode_outcome("test/select", 2, "discarded")

    kept = load_training_data("test/select")
    assert [episode.episode for episode in kept.episodes] == [0, 3]
    assert {item.episode: item.reason for item in kept.skipped} == {
        1: "outcome is failure",
        2: "outcome is discarded",
    }
    assert kept.episodes[0].state.shape == (5, 6)
    assert kept.episodes[0].state.dtype == np.float32
    np.testing.assert_allclose(kept.episodes[0].t, np.arange(5) / 10, atol=1e-9)
    assert kept.env_state_dim == 0

    success = load_training_data("test/select", selection="success")
    assert [episode.episode for episode in success.episodes] == [0]
    everything = load_training_data("test/select", selection="all")
    assert [episode.episode for episode in everything.episodes] == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="selection"):
        load_training_data("test/select", selection="best")


def test_loader_skips_unusable_and_quality_failures():
    _record("test/quality", 4)

    def out_of_limits(rows):
        rows[2]["action"][0] = 400.0
        return rows

    def non_finite(rows):
        rows[1]["observation.state"][3] = None
        return rows

    def backwards_time(rows):
        rows[3]["t"] = rows[1]["t"]
        return rows

    _rewrite_frames("test/quality", 0, out_of_limits)
    _rewrite_frames("test/quality", 1, non_finite)
    _rewrite_frames("test/quality", 2, backwards_time)

    kept = load_training_data("test/quality")
    assert [episode.episode for episode in kept.episodes] == [3]
    reasons = {item.episode: item.reason for item in kept.skipped}
    assert reasons[0].startswith("failed quality check") and "shoulder_pan" in reasons[0]
    assert reasons[1].startswith("unusable")
    assert reasons[2].startswith("failed quality check") and "timestamp" in reasons[2]

    # "all" keeps quality failures that are still numerically usable.
    everything = load_training_data("test/quality", selection="all")
    assert [episode.episode for episode in everything.episodes] == [0, 2, 3]


def test_loader_reports_why_nothing_is_usable():
    _record("test/none", 2)
    set_episode_outcome("test/none", 0, "failure")
    set_episode_outcome("test/none", 1, "partial")
    with pytest.raises(TrainingDataError) as excinfo:
        load_training_data("test/none")
    message = str(excinfo.value)
    assert "1 failure, 1 partial" in message
    assert "--episodes all" in message
    with pytest.raises(DatasetNotFoundError):
        load_training_data("test/does-not-exist")


def test_a_corrupt_fallback_manifest_is_never_overwritten_or_trusted():
    _record("test/corrupt-manifest", 1)
    path = _frames_path("test/corrupt-manifest", 0).parent / "episode.json"
    original = b'{"outcome": "failure"'
    path.write_bytes(original)

    inspection = inspect_dataset("test/corrupt-manifest")
    labels = next(item for item in inspection.capabilities if item.id == "outcome_labels")
    manifest = next(item for item in inspection.capabilities if item.id == "manifest_inspection")
    assert not labels.available and labels.level == "unavailable"
    assert manifest.available and manifest.level == "partial"
    assert "outcome labels unreadable" in inspection.episodes[0].issues[0]
    for selection in ("kept", "success"):
        with pytest.raises(TrainingDataError, match="outcome labels are unreadable"):
            load_training_data("test/corrupt-manifest", selection=selection)
    all_episodes = load_training_data("test/corrupt-manifest", selection="all")
    assert [episode.episode for episode in all_episodes.episodes] == [0]
    with pytest.raises(OutcomeLabelsUnreadableError, match="labels were not changed"):
        set_episode_outcome("test/corrupt-manifest", 0, "success")
    assert path.read_bytes() == original


def test_loader_reads_sim_environment_state():
    _record("test/env", 3)
    meta_path = _frames_path("test/env", 0).parent.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["environment_state_names"] = ["cube_x", "cube_y", "cube_z"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def add_env(rows):
        for index, row in enumerate(rows):
            row[ENV_STATE_KEY] = [0.1 * index, 0.2, 0.0]
        return rows

    _rewrite_frames("test/env", 0, add_env)
    _rewrite_frames("test/env", 1, add_env)

    auto = load_training_data("test/env")
    assert [episode.episode for episode in auto.episodes] == [0, 1]
    assert auto.env_state_names == ["cube_x", "cube_y", "cube_z"]
    assert auto.episodes[0].env_state is not None
    assert auto.episodes[0].env_state.shape == (5, 3)
    assert {item.episode: item.reason for item in auto.skipped} == {2: f"no {ENV_STATE_KEY}"}

    state_only = load_training_data("test/env", env_state="ignore")
    assert [episode.episode for episode in state_only.episodes] == [0, 1, 2]
    assert all(episode.env_state is None for episode in state_only.episodes)
    assert state_only.env_state_dim == 0

"""Outcome labels for LeRobot datasets: the so101_outcomes.json sidecar.

LeRobot's metadata has no outcome field, so the suite keeps LeRobot labels in
``<dataset>/so101_outcomes.json`` with the fallback format's rules (human
labels win, history is kept). Inspection, the training-data loader and the
label route read and write it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.data import loader
from so101.data.inspection import (
    OUTCOMES_SIDECAR,
    OutcomeLabelsUnreadableError,
    inspect_dataset,
    set_episode_outcome,
)
from so101.data.loader import TrainingDataError, load_training_data, selected_episodes
from so101.services.app import create_app

AUTO = "auto:sim_predicate"


def _lerobot(repo_id: str, episodes: int):
    root = get_settings().datasets_dir / ("v2--" + repo_id.replace("/", "%2F"))
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "total_episodes": episodes,
                "total_frames": 4 * episodes,
                "features": {"action": {"dtype": "float32", "shape": [6]}},
            }
        )
    )
    return root


def _sidecar(root) -> dict:
    return json.loads((root / OUTCOMES_SIDECAR).read_text(encoding="utf-8"))


def _outcomes(repo_id: str) -> list[tuple[str, str]]:
    return [(e.outcome, e.outcome_source) for e in inspect_dataset(repo_id).episodes]


def test_a_lerobot_episode_label_is_stored_in_the_sidecar_and_read_back():
    root = _lerobot("lab/real", 3)
    assert set_episode_outcome("lab/real", 1, "failure", "gripper slipped")
    stored = _sidecar(root)
    assert stored["schema_version"] == 1 and stored["repo_id"] == "lab/real"
    assert list(stored["episodes"]) == ["1"]
    record = stored["episodes"]["1"]
    assert (record["outcome"], record["outcome_source"], record["notes"]) == (
        "failure",
        "human",
        "gripper slipped",
    )
    assert record["outcome_history"] == [] and isinstance(record["labeled_at"], float)

    inspection = inspect_dataset("lab/real")
    assert _outcomes("lab/real") == [("unlabeled", ""), ("failure", "human"), ("unlabeled", "")]
    assert inspection.episodes[1].notes == "gripper slipped"
    assert inspection.outcome_counts["failure"] == 1 and inspection.outcome_counts["unlabeled"] == 2
    labels = next(c for c in inspection.capabilities if c.id == "outcome_labels")
    assert labels.available and OUTCOMES_SIDECAR in labels.detail
    assert not list(root.glob(".tmp-*"))  # written through a temp file and renamed


def test_sidecar_labels_follow_the_fallback_rules():
    root = _lerobot("lab/rules", 2)
    detail = {"seed": 4, "success": True}
    assert set_episode_outcome("lab/rules", 0, "success", source=AUTO, detail=detail)
    # A human label replaces an automatic one, and the automatic one is history.
    assert set_episode_outcome("lab/rules", 0, "failure", "bumped the cup")
    # Automation never replaces a human label ...
    assert not set_episode_outcome("lab/rules", 0, "success", source=AUTO, detail=detail)
    record = _sidecar(root)["episodes"]["0"]
    assert (record["outcome"], record["outcome_source"]) == ("failure", "human")
    assert "outcome_detail" not in record
    assert [(h["outcome"], h["source"], h.get("detail")) for h in record["outcome_history"]] == [
        ("success", AUTO, detail)
    ]
    # ... but may label an episode a human reset to unlabeled, keeping their notes.
    assert set_episode_outcome("lab/rules", 0, "unlabeled", "check camera 2")
    assert set_episode_outcome("lab/rules", 0, "success", source=AUTO, detail=detail)
    record = _sidecar(root)["episodes"]["0"]
    assert (record["outcome"], record["notes"], record["outcome_detail"]) == (
        "success",
        "check camera 2",
        detail,
    )
    assert len(record["outcome_history"]) == 3
    assert _outcomes("lab/rules") == [("success", AUTO), ("unlabeled", "")]


@pytest.mark.parametrize("episode", [2, -1])
def test_labelling_a_lerobot_episode_that_does_not_exist_is_refused(episode):
    root = _lerobot("lab/range", 2)
    with pytest.raises(FileNotFoundError, match=f"episode {episode} not found"):
        set_episode_outcome("lab/range", episode, "success")
    assert not (root / OUTCOMES_SIDECAR).exists()


def test_selection_follows_sidecar_labels():
    _lerobot("lab/select", 4)
    set_episode_outcome("lab/select", 0, "success")
    set_episode_outcome("lab/select", 1, "failure")
    set_episode_outcome("lab/select", 3, "discarded", source=AUTO)
    assert selected_episodes("lab/select")[0] == [0, 2]
    kept, skipped = selected_episodes("lab/select", "kept")
    assert [(s.episode, s.reason) for s in skipped] == [
        (1, "outcome is failure"),
        (3, "outcome is discarded"),
    ]
    assert selected_episodes("lab/select", "success")[0] == [0]
    assert selected_episodes("lab/select", "all")[0] == [0, 1, 2, 3]
    set_episode_outcome("lab/select", 0, "partial")
    with pytest.raises(TrainingDataError, match="no episodes in 'lab/select' for selection"):
        selected_episodes("lab/select", "success")


class _Table(dict):
    @property
    def column_names(self) -> list[str]:
        return list(self)


def test_the_baseline_loader_trains_only_on_kept_lerobot_episodes(monkeypatch):
    _lerobot("lab/train", 3)
    set_episode_outcome("lab/train", 0, "success")
    set_episode_outcome("lab/train", 1, "failure", source=AUTO)
    joints = [0.0, -10.0, 10.0, 0.0, 0.0, 10.0]
    table = _Table(
        {
            "episode_index": [0, 0, 1, 1, 2, 2],
            "observation.state": [joints] * 6,
            "action": [joints] * 6,
            "timestamp": [0.0, 0.1] * 3,
        }
    )

    class _Dataset:
        hf_dataset = table
        features = {
            key: {"names": [f"{joint}.pos" for joint in JOINT_NAMES]}
            for key in ("observation.state", "action")
        }

    monkeypatch.setattr(loader, "_lerobot_dataset_class", lambda: object)
    monkeypatch.setattr(
        "so101.data.lerobot_local.open_local_lerobot_dataset",
        lambda repo_id, root, dataset_cls=None: _Dataset(),
    )
    kept = load_training_data("lab/train")
    assert [(e.episode, e.outcome) for e in kept.episodes] == [(0, "success"), (2, "unlabeled")]
    assert [(s.episode, s.reason) for s in kept.skipped] == [(1, "outcome is failure")]
    assert kept.episodes[0].outcome_source == "human"
    assert [e.episode for e in load_training_data("lab/train", selection="success").episodes] == [0]
    everything = load_training_data("lab/train", selection="all")
    assert [e.episode for e in everything.episodes] == [0, 1, 2]
    assert np.allclose(everything.episodes[1].action, [joints, joints])


def test_lerobot_loader_applies_timestamp_quality_and_usability_rules(monkeypatch):
    _lerobot("lab/timestamps", 5)
    for episode in range(5):
        set_episode_outcome("lab/timestamps", episode, "success")
    joints = [0.0, -10.0, 10.0, 0.0, 0.0, 10.0]
    table = _Table(
        {
            "episode_index": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "observation.state": [joints] * 10,
            "action": [joints] * 10,
            # Missing/non-finite and duplicate/backwards time are quality
            # failures: strict selections drop them. Explicit ``all`` repairs
            # non-finite time from fps and retains finite unusual time.
            "timestamp": [0.0, 0.1, 0.0, np.nan, 0.0, 0.0, 0.1, 0.0, "bad", 0.1],
        }
    )

    class _Dataset:
        hf_dataset = table
        features = {
            key: {"names": [f"{joint}.pos" for joint in JOINT_NAMES]}
            for key in ("observation.state", "action")
        }

    monkeypatch.setattr(loader, "_lerobot_dataset_class", lambda: object)
    monkeypatch.setattr(
        "so101.data.lerobot_local.open_local_lerobot_dataset",
        lambda repo_id, root, dataset_cls=None: _Dataset(),
    )

    for selection in ("kept", "success"):
        selected = load_training_data("lab/timestamps", selection=selection)
        assert [episode.episode for episode in selected.episodes] == [0]
        assert {item.episode: item.reason for item in selected.skipped} == {
            1: "failed quality check: a frame timestamp is missing or non-finite",
            2: "failed quality check: frame 1 timestamp did not increase",
            3: "failed quality check: frame 1 timestamp did not increase",
            4: "failed quality check: a frame timestamp is missing or non-finite",
        }

    everything = load_training_data("lab/timestamps", selection="all")
    assert [episode.episode for episode in everything.episodes] == [0, 1, 2, 3, 4]
    assert not everything.skipped
    np.testing.assert_allclose(everything.episodes[1].t, [0.0, 0.1])
    np.testing.assert_allclose(everything.episodes[2].t, [0.0, 0.0])
    np.testing.assert_allclose(everything.episodes[3].t, [0.0, -0.1])
    np.testing.assert_allclose(everything.episodes[4].t, [0.0, 0.1])


def test_lerobot_loader_missing_timestamp_column_matches_fallback(monkeypatch):
    _lerobot("lab/missing-timestamps", 1)
    set_episode_outcome("lab/missing-timestamps", 0, "success")
    joints = [0.0, -10.0, 10.0, 0.0, 0.0, 10.0]
    table = _Table(
        {
            "episode_index": [0, 0],
            "observation.state": [joints, joints],
            "action": [joints, joints],
        }
    )

    class _Dataset:
        hf_dataset = table
        features = {
            key: {"names": [f"{joint}.pos" for joint in JOINT_NAMES]}
            for key in ("observation.state", "action")
        }

    monkeypatch.setattr(loader, "_lerobot_dataset_class", lambda: object)
    monkeypatch.setattr(
        "so101.data.lerobot_local.open_local_lerobot_dataset",
        lambda repo_id, root, dataset_cls=None: _Dataset(),
    )

    for selection in ("kept", "success"):
        with pytest.raises(TrainingDataError, match="timestamp is missing or non-finite"):
            load_training_data("lab/missing-timestamps", selection=selection)

    everything = load_training_data("lab/missing-timestamps", selection="all")
    assert [episode.episode for episode in everything.episodes] == [0]
    assert not everything.skipped
    np.testing.assert_allclose(everything.episodes[0].t, [0.0, 0.1])


def test_lerobot_loader_reorders_named_joints_to_canonical_order(monkeypatch):
    _lerobot("lab/joint-order", 1)
    canonical = [float(index + 1) for index in range(len(JOINT_NAMES))]
    canonical_names = [f"{joint}.pos" for joint in JOINT_NAMES]
    state_values = list(reversed(canonical))
    state_names = list(reversed(canonical_names))
    action_values = canonical[2:] + canonical[:2]
    action_names = canonical_names[2:] + canonical_names[:2]
    table = _Table(
        {
            "episode_index": [0, 0],
            "observation.state": [state_values, state_values],
            "action": [action_values, action_values],
            "timestamp": [0.0, 0.1],
        }
    )

    class _Dataset:
        hf_dataset = table
        features = {
            "observation.state": {"names": state_names},
            "action": {"names": action_names},
        }

    monkeypatch.setattr(loader, "_lerobot_dataset_class", lambda: object)
    monkeypatch.setattr(
        "so101.data.lerobot_local.open_local_lerobot_dataset",
        lambda repo_id, root, dataset_cls=None: _Dataset(),
    )

    episode = load_training_data("lab/joint-order").episodes[0]
    np.testing.assert_array_equal(episode.state[0], canonical)
    np.testing.assert_array_equal(episode.action[0], canonical)


@pytest.mark.parametrize(
    "names",
    [None, [f"{joint}.pos" for joint in JOINT_NAMES[:-1]] + ["other.pos"]],
)
def test_lerobot_loader_refuses_untrusted_joint_names(monkeypatch, names):
    _lerobot("lab/joint-names", 1)
    joints = [0.0] * len(JOINT_NAMES)
    table = _Table(
        {
            "episode_index": [0],
            "observation.state": [joints],
            "action": [joints],
        }
    )

    class _Dataset:
        hf_dataset = table
        features = {
            "observation.state": {"names": names},
            "action": {"names": [f"{joint}.pos" for joint in JOINT_NAMES]},
        }

    monkeypatch.setattr(loader, "_lerobot_dataset_class", lambda: object)
    monkeypatch.setattr(
        "so101.data.lerobot_local.open_local_lerobot_dataset",
        lambda repo_id, root, dataset_cls=None: _Dataset(),
    )

    with pytest.raises(TrainingDataError, match="observation.state.*named joints"):
        load_training_data("lab/joint-names")


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"\xff\xfe",
        b'["episodes"]',
        b'{"schema_version": 1, "episodes": {"0": "success"}}',
        b'{"schema_version": 2, "episodes": {}}',
    ],
)
def test_an_unreadable_sidecar_is_never_overwritten_or_trusted(content):
    root = _lerobot("lab/broken", 2)
    (root / OUTCOMES_SIDECAR).write_bytes(content)
    with pytest.raises(OutcomeLabelsUnreadableError, match="labels were not changed"):
        set_episode_outcome("lab/broken", 0, "failure")
    assert (root / OUTCOMES_SIDECAR).read_bytes() == content

    inspection = inspect_dataset("lab/broken")
    labels = next(c for c in inspection.capabilities if c.id == "outcome_labels")
    assert not labels.available and labels.level == "unavailable"
    assert all("outcome labels unreadable" in e.issues[0] for e in inspection.episodes)
    # Every episode reads as unlabeled, so only an explicit "all" may train.
    for selection in ("kept", "success"):
        with pytest.raises(TrainingDataError, match="outcome labels are unreadable"):
            selected_episodes("lab/broken", selection)
        with pytest.raises(TrainingDataError, match="outcome labels are unreadable"):
            load_training_data("lab/broken", selection=selection)
    assert selected_episodes("lab/broken", "all")[0] == [0, 1]


def test_a_semantically_invalid_sidecar_is_never_overwritten_or_trusted():
    root = _lerobot("lab/invalid-label", 2)
    original = json.dumps(
        {
            "schema_version": 1,
            "repo_id": "lab/invalid-label",
            "episodes": {"0": {"outcome": "failur", "outcome_source": "human"}},
        }
    ).encode()
    (root / OUTCOMES_SIDECAR).write_bytes(original)

    for selection in ("kept", "success"):
        with pytest.raises(TrainingDataError, match="outcome labels are unreadable"):
            selected_episodes("lab/invalid-label", selection)
    assert selected_episodes("lab/invalid-label", "all")[0] == [0, 1]
    with pytest.raises(OutcomeLabelsUnreadableError, match="labels were not changed"):
        set_episode_outcome("lab/invalid-label", 0, "failure")
    assert (root / OUTCOMES_SIDECAR).read_bytes() == original


def test_the_label_route_labels_lerobot_datasets():
    root = _lerobot("lab/route", 2)
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/datasets/lab%2Froute/episodes/1/outcome",
            json={"outcome": "failure", "notes": "dropped it"},
        )
        assert response.status_code == 200, response.text
        episode = response.json()["episodes"][1]
        assert (episode["outcome"], episode["outcome_source"], episode["notes"]) == (
            "failure",
            "human",
            "dropped it",
        )
        assert response.json()["outcome_counts"]["failure"] == 1

        missing = client.post(
            "/api/datasets/lab%2Froute/episodes/2/outcome", json={"outcome": "success"}
        )
        assert missing.status_code == 404

        (root / OUTCOMES_SIDECAR).write_text("{oops")
        broken = client.post(
            "/api/datasets/lab%2Froute/episodes/0/outcome", json={"outcome": "success"}
        )
        assert broken.status_code == 409 and "labels were not changed" in broken.json()["detail"]


def test_the_label_route_still_refuses_unknown_formats():
    root = get_settings().datasets_dir / "v2--lab%2Fodd"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps({"repo_id": "lab/odd", "episodes": 1, "format": "custom-v9"})
    )
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/datasets/lab%2Fodd/episodes/0/outcome", json={"outcome": "success"}
        )
    assert response.status_code == 501 and "custom-v9" in response.json()["detail"]

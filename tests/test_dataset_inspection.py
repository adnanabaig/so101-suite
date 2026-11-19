"""Dataset manifests, read-only replay previews, QC, and trash recovery."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.data.dataset import (
    dataset_root,
    delete_dataset,
    get_dataset,
    list_trashed_datasets,
    restore_trashed_dataset,
)
from so101.data.inspection import (
    episode_timeline,
    inspect_dataset,
    run_quality_control,
    set_episode_outcome,
)
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.services.app import create_app
from so101.teleop.controller import TeleopStep


def _joints(value: float) -> dict[str, float]:
    return {f"{joint}.pos": value for joint in JOINT_NAMES}


def _record_dataset(repo_id: str, *, camera: bool = True) -> EpisodeRecorder:
    cameras = ["scene"] if camera else []
    recorder = EpisodeRecorder(
        RecordingConfig(repo_id=repo_id, fps=10, camera_keys=cameras, image_chunk_frames=2)
    )
    for index in range(3):
        images = {"scene": np.full((2, 3, 3), index, dtype=np.uint8)} if camera else None
        recorder.add(
            TeleopStep(
                t=100.0 + index / 10,
                observation=_joints(float(index)),
                action=_joints(float(index + 1)),
                desired_action=_joints(float(index + 2)),
            ),
            images,
        )
    recorder.save_episode()
    return recorder


def test_detail_timeline_outcome_and_qc_are_read_only():
    recorder = _record_dataset("test/inspect")
    set_episode_outcome("test/inspect", 0, "success", "clean pickup")

    detail = inspect_dataset("test/inspect")
    assert detail.summary.num_frames == 3
    assert detail.duration_seconds == 0.2
    assert detail.episodes[0].outcome == "success"
    assert detail.episodes[0].camera_chunks == {"scene": 2}
    assert detail.episodes[0].camera_frames == {"scene": 3}
    assert detail.incomplete_captures == []

    preview = episode_timeline("test/inspect", 0, limit=2)
    assert preview.actuates_hardware is False
    assert preview.returned_frames == 2
    assert preview.has_more is True
    assert preview.points[0].action == [1.0] * len(JOINT_NAMES)
    assert preview.points[0].desired_action == [2.0] * len(JOINT_NAMES)

    report = run_quality_control("test/inspect")
    assert report.overall == "pass"
    assert report.checked_frames == 3
    assert all(check.status == "pass" for check in report.checks)
    assert (recorder.root / "episode_000000" / "episode.json").is_file()


def test_qc_reports_nonfinite_timing_limits_and_camera_mismatch():
    recorder = _record_dataset("test/bad", camera=True)
    frames_path = recorder.root / "episode_000000" / "frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text().splitlines()]
    rows[1]["t"] = rows[0]["t"]
    rows[1]["action"][0] = 999.0
    rows[2]["observation.state"][2] = float("nan")
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # Drop the final one-frame camera chunk.
    (recorder.root / "episode_000000" / "images" / "00_scene" / "chunk_000001.npy").unlink()

    report = run_quality_control("test/bad")
    statuses = {check.id: check.status for check in report.checks}
    assert report.overall == "fail"
    assert statuses["finite"] == "fail"
    assert statuses["timestamps"] == "fail"
    assert statuses["action_limits"] == "fail"
    assert statuses["camera_chunks"] == "fail"


def test_incomplete_capture_is_reported():
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/interrupted", camera_keys=[]))
    recorder.add(TeleopStep(t=1.0, observation=_joints(0), action=_joints(0)))
    recorder.abort("camera disconnected")

    detail = inspect_dataset("test/interrupted")
    # An interrupted first capture has no published meta.json, so it is not a
    # discoverable dataset until at least one complete episode exists.
    assert detail.incomplete_captures[0].reason == "camera disconnected"


def test_recoverable_trash_can_be_listed_and_restored():
    _record_dataset("test/restore", camera=False)
    assert delete_dataset("test/restore") is True
    entries = list_trashed_datasets()
    assert len(entries) == 1
    assert entries[0].repo_id == "test/restore"
    assert entries[0].size_bytes > 0

    restored = restore_trashed_dataset(entries[0].id)
    assert restored.repo_id == "test/restore"
    assert get_dataset("test/restore") is not None
    assert list_trashed_datasets() == []


def test_dataset_api_routes_repo_ids_with_slashes():
    _record_dataset("test/api-inspect", camera=False)
    with TestClient(create_app()) as client:
        detail = client.get("/api/datasets/test/api-inspect")
        assert detail.status_code == 200
        assert detail.json()["episodes"][0]["frames"] == 3

        preview = client.get(
            "/api/datasets/test/api-inspect/episodes/0/replay-preview",
            params={"max_points": 20},
        )
        assert preview.status_code == 200
        assert preview.json()["actuates_hardware"] is False

        qc = client.get("/api/datasets/test/api-inspect/qc")
        assert qc.status_code == 200
        assert qc.json()["overall"] == "pass"

        labeled = client.post(
            "/api/datasets/test/api-inspect/episodes/0/outcome",
            json={"outcome": "partial", "notes": "slipped"},
        )
        assert labeled.status_code == 200
        assert labeled.json()["episodes"][0]["outcome"] == "partial"

        removed = client.delete("/api/datasets/test/api-inspect")
        assert removed.status_code == 200
        trashed = client.get("/api/datasets/trash").json()
        assert len(trashed) == 1
        restored = client.post("/api/datasets/trash/restore", json={"trash_id": trashed[0]["id"]})
        assert restored.status_code == 200
        assert restored.json()["repo_id"] == "test/api-inspect"


def test_lerobot_without_optional_runtime_reports_capabilities(monkeypatch):
    import builtins

    root = dataset_root("test/lerobot-capability")
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 1,
                "total_frames": 12,
                "fps": 30,
                "features": {"action": {"dtype": "float32", "shape": [6]}},
            }
        )
    )
    real_import = builtins.__import__

    def import_without_lerobot(name, *args, **kwargs):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ModuleNotFoundError("optional LeRobot runtime not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_lerobot)
    detail = inspect_dataset("test/lerobot-capability")
    capability = next(item for item in detail.capabilities if item.id == "replay_preview")
    assert capability.available is False
    assert capability.level == "unavailable"

    preview = episode_timeline("test/lerobot-capability", 0)
    assert preview.actuates_hardware is False
    assert preview.capability.available is False
    assert preview.points == []

    report = run_quality_control("test/lerobot-capability")
    assert report.overall == "warn"
    assert any(check.status == "skip" for check in report.checks)

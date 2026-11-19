"""Episode QA distinguishes real capture, software fixtures and damaged samples."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "episode_review", Path(__file__).resolve().parents[1] / "scripts/review_mcap_episodes.py"
)
assert spec and spec.loader
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_default_storage_helper_loads_under_isolated_python(tmp_path):
    profile = tmp_path / "config/rigs/mcap-storage.local.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"output": str(tmp_path / "raw"), "min_free_gib": 0.001}))
    script = str(Path(review.__file__).resolve())
    code = (
        "import runpy,sys; from pathlib import Path; "
        f"entry=runpy.run_path({script!r})['main']; "
        f"entry.__globals__['ROOT']=Path({str(tmp_path)!r}); "
        f"sys.argv=['review','--data-dir',{str(tmp_path)!r}]; "
        "raise SystemExit(entry())"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 1
    assert "No saved episodes yet" in result.stdout, result.stderr


def fixture_data():
    episode = {"status": "saved", "started_at_ns": 0, "ended_at_ns": 6 * 10**9}
    ticks = [
        {
            "timestamp_ns": round(index / 30 * 1e9),
            "seq": index + 1,
            "stream_id": "stream",
            "backend": "real",
            **{
                field: {f"{joint}.pos": float(index) for joint in review.JOINTS}
                for field in ("observation", "action", "desired_action")
            },
        }
        for index in range(180)
    ]
    cameras = {"obsbot": [tick["timestamp_ns"] for tick in ticks]}
    topics = {"/robot/encoder/scene": 100, "/robot/health": 6}
    clocks = {"obsbot": {"host_camera_read_completed"}}
    return episode, ticks, cameras, topics, clocks


def test_complete_capture_requires_human_task_review_and_reports_tracking():
    result = review.assess(*fixture_data())
    assert result["data_checks_passed"] and result["human_task_review"] == "pending"
    assert result["control"]["measured_fps"] == pytest.approx(30)
    assert result["tracking_error"]["gripper"]["unit"] == "percent"
    assert result["camera_target_review"]["all_four_views_ready"] is False


def test_camera_targets_do_not_confuse_primary_integrity_with_all_four_ready():
    episode, ticks, cameras, topics, clocks = fixture_data()
    for name in ("camera_1", "camera_2", "realsense"):
        cameras[name] = list(cameras["obsbot"])
        clocks[name] = {"host_camera_read_completed"}
    result = review.assess(episode, ticks, cameras, topics, clocks)
    assert result["camera_target_review"]["all_four_views_ready"] is True
    cameras["camera_1"] = cameras["camera_1"][::3]
    result = review.assess(episode, ticks, cameras, topics, clocks)
    assert result["data_checks_passed"] is True
    assert result["camera_target_review"]["all_four_views_ready"] is False
    assert result["camera_target_review"]["views"]["camera_1"]["passes"] is False


def test_health_review_retains_warning_identity_without_counting_messages_as_measurements():
    advisories = {}
    health = {
        "arms": {
            "follower": {
                "motors": {
                    "gripper": {"level": "warn", "reasons": ["status 0x20 (unverified)"]},
                    "elbow_flex": {"level": "ok", "reasons": []},
                }
            },
            "leader": {"motors": {"wrist_roll": {"level": "critical", "reasons": ["temperature"]}}},
        }
    }
    review.collect_health_advisories(advisories, health, 100)
    review.collect_health_advisories(advisories, health, 200)
    assert len(advisories) == 2
    gripper = next(value for value in advisories.values() if value["motor"] == "gripper")
    assert gripper["health_message_count"] == 2
    assert gripper["first_log_time_ns"] == 100
    assert gripper["last_log_time_ns"] == 200
    assert gripper["reasons"] == ["status 0x20 (unverified)"]
    assert review.assess(*fixture_data())["data_checks_passed"]


@pytest.mark.parametrize(
    "problem",
    [
        "gap",
        "missing",
        "nan",
        "mock",
        "clock",
        "slow_camera",
        "interrupted",
        "truncated_motor_start",
        "truncated_motor_end",
        "truncated_camera",
    ],
)
def test_corrupted_or_unqualified_capture_is_not_accepted(problem):
    episode, ticks, cameras, topics, clocks = fixture_data()
    if problem == "gap":
        ticks.pop(40)
    elif problem == "missing":
        ticks[30]["action"].pop("wrist_roll.pos")
    elif problem == "nan":
        ticks[30]["observation"]["wrist_roll.pos"] = float("nan")
    elif problem == "mock":
        ticks[30]["backend"] = "mock"
    elif problem == "clock":
        clocks["obsbot"] = {"recorder_arrival"}
    elif problem == "slow_camera":
        cameras["obsbot"] = cameras["obsbot"][::4]
    elif problem == "truncated_motor_start":
        ticks = ticks[30:]
    elif problem == "truncated_motor_end":
        ticks = ticks[:-30]
    elif problem == "truncated_camera":
        cameras["obsbot"] = cameras["obsbot"][:-30]
    else:
        episode["status"] = "interrupted"
    result = review.assess(episode, ticks, cameras, topics, clocks)
    assert not result["data_checks_passed"] and result["issues"]


def test_mock_acceptance_requires_explicit_flag():
    values = fixture_data()
    for tick in values[1]:
        tick["backend"] = "mock"
    assert not review.assess(*values)["data_checks_passed"]
    assert review.assess(*values, allow_mock=True)["data_checks_passed"]


def test_following_metrics_distinguish_tracking_from_clipped_leader_demand():
    values = fixture_data()
    for tick in values[1]:
        tick["observation"]["shoulder_lift.pos"] = 0.0
        tick["action"]["shoulder_lift.pos"] = 5.0
        tick["desired_action"]["shoulder_lift.pos"] = 30.0
    result = review.assess(*values)
    assert result["data_checks_passed"]
    assert result["tracking_error"]["shoulder_lift"]["mean_absolute"] == 5
    following = result["leader_following"]["shoulder_lift"]
    assert following["mean_absolute_error"] == 30
    assert following["longest_stationary_demand_seconds"] == pytest.approx(179 / 30)
    assert result["leader_following"]["elbow_flex"]["longest_stationary_demand_seconds"] == 0


def test_long_sample_gap_breaks_stationary_interval_instead_of_inventing_stall_time():
    values = fixture_data()
    values[1][:] = values[1][:3]
    for tick in values[1]:
        tick["observation"]["shoulder_lift.pos"] = 0.0
        tick["desired_action"]["shoulder_lift.pos"] = 30.0
    values[1][2]["timestamp_ns"] = 1_000_000_000
    result = review.assess(*values)
    assert result["leader_following"]["shoulder_lift"][
        "stationary_demand_seconds"
    ] == pytest.approx(1 / 30)

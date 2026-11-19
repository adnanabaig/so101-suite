"""Preview predicts play (B16), one sample_hz range (B17), seated scaling (B18).

These are engine-agnostic request semantics. The physics engine reuses the
same planner, see ``test_sim_physics_engine.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.services.app import create_app
from so101.services.schemas import SimulationTrajectoryPreviewRequest
from so101.simulation import SimulationEngine, SimulationValidationError
from so101.simulation.engine import (
    MAX_PREVIEW_SAMPLE_HZ,
    MIN_PREVIEW_SAMPLE_HZ,
    plan_preview,
    steps_for_duration,
)
from so101.simulation.scenarios import (
    DEFAULT_SCENARIO_PATH,
    ScenarioError,
    ScenarioLibrary,
    vertical_half_extent,
)


def _engine() -> SimulationEngine:
    engine = SimulationEngine(ScenarioLibrary(), clock=lambda: 0.0)
    # Start mid-motion with a pending target, so "unspecified joints keep
    # their commanded target" is exercised too.
    engine.step(7, targets={"shoulder_pan": -30.0, "elbow_flex": 40.0})
    return engine


def _assert_point_matches_state(point: dict, state: dict) -> None:
    telemetry = state["telemetry"]
    assert point["measured"] == pytest.approx(telemetry["measured"], abs=1e-9)
    assert point["target"] == pytest.approx(telemetry["target"], abs=1e-9)
    assert point["end_effector"]["position_m"] == pytest.approx(
        state["kinematics"]["position_m"], abs=1e-9
    )


# -- B16: preview predicts play ------------------------------------------------
def test_target_preview_equals_set_targets_then_fixed_steps():
    previewer, player = _engine(), _engine()
    target = {"shoulder_pan": 60.0, "gripper": 10.0}
    preview = previewer.preview_trajectory(target=target, duration_s=1.3, sample_hz=12.0)

    player.set_targets(target)
    done = 0
    for point in preview["points"][1:]:
        state = player.step(point["step"] - done)
        done = point["step"]
        _assert_point_matches_state(point, state)
    assert done == steps_for_duration(1.3, previewer.fixed_dt_s) == 78
    assert preview["quantized_duration_s"] == pytest.approx(1.3)
    assert preview["semantics"] == "step_targets_like_play"
    # The pending elbow target from before the preview stays commanded.
    assert preview["points"][-1]["target"]["elbow_flex"] == 40.0
    assert preview["final_reached"] is True
    assert previewer.state()["step_count"] == 7  # preview did not advance the engine


def test_waypoint_preview_equals_commanding_each_waypoint_at_its_segment_start():
    previewer, player = _engine(), _engine()
    waypoints = [
        {"time_s": 0.25, "targets": {"shoulder_pan": 20.0}},
        {"time_s": 0.6, "targets": {"elbow_flex": -30.0, "gripper": 0.0}},
        {"time_s": 1.0, "targets": {"shoulder_pan": -15.0}},
    ]
    preview = previewer.preview_trajectory(waypoints=waypoints, sample_hz=60.0)
    points = preview["points"]
    assert [point["step"] for point in points] == list(range(61))

    done = 0
    for waypoint in waypoints:
        end_step = steps_for_duration(waypoint["time_s"], player.fixed_dt_s)
        player.set_targets(waypoint["targets"])
        while done < end_step:
            state = player.step(1)
            done += 1
            _assert_point_matches_state(points[done], state)
    assert points[-1]["target"] == pytest.approx(player.state()["telemetry"]["target"])


def test_preview_plan_quantises_waypoint_times_onto_engine_ticks():
    plan = plan_preview(
        fixed_hz=60.0,
        current_target={joint: 0.0 for joint in JOINT_NAMES},
        target=None,
        duration_s=None,
        waypoints=[
            {"time_s": 4.15, "targets": {"shoulder_pan": 10.0}},
            {"time_s": 4.16, "targets": {"shoulder_pan": 11.0}},
        ],
        sample_hz=60.0,
    )
    # 4.15 s at 60 Hz is exactly 249 ticks, not 250 (float noise).
    assert [end for end, _ in plan.segments] == [249, 250]
    assert plan.target_at(249)["shoulder_pan"] == 10.0
    assert plan.target_at(250)["shoulder_pan"] == 11.0


# -- B17: one sample_hz range ---------------------------------------------------
def test_engine_and_schema_accept_the_same_sample_hz_range():
    schema = SimulationTrajectoryPreviewRequest.model_json_schema()["properties"]["sample_hz"]
    assert schema["minimum"] == MIN_PREVIEW_SAMPLE_HZ == 1.0
    assert schema["maximum"] == MAX_PREVIEW_SAMPLE_HZ == 240.0

    engine = _engine()
    fast = engine.preview_trajectory(target={"shoulder_pan": 5.0}, duration_s=0.5, sample_hz=240.0)
    assert fast["requested_sample_hz"] == 240.0
    assert fast["sample_hz"] == 60.0  # quantised to the engine tick, not rejected
    assert [point["step"] for point in fast["points"]] == list(range(31))
    slow = engine.preview_trajectory(target={"shoulder_pan": 5.0}, duration_s=2.0, sample_hz=1.0)
    assert [point["step"] for point in slow["points"]] == [0, 60, 120]
    for bad in (0.5, 240.5, math.nan, math.inf):
        with pytest.raises(SimulationValidationError, match="sample_hz"):
            engine.preview_trajectory(target={"shoulder_pan": 5.0}, duration_s=0.5, sample_hz=bad)


def test_api_sample_hz_boundaries():
    with TestClient(create_app()) as client:
        client.post("/api/simulation/reset", json={"scenario_id": "tabletop_pick", "seed": 1})
        ok = client.post(
            "/api/simulation/trajectory/preview",
            json={"target": {"shoulder_pan": 5.0}, "duration_s": 0.25, "sample_hz": 240.0},
        )
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert (body["requested_sample_hz"], body["sample_hz"]) == (240.0, 60.0)
        assert body["semantics"] == "step_targets_like_play"
        for bad in (0.5, 241.0):
            rejected = client.post(
                "/api/simulation/trajectory/preview",
                json={"target": {"shoulder_pan": 5.0}, "duration_s": 0.25, "sample_hz": bad},
            )
            assert rejected.status_code == 422


# -- B18: scaled objects stay seated ---------------------------------------------
def _bottom(obj: dict, pose_key: str, size_key: str) -> float:
    pose = obj[pose_key]
    return pose[2] - vertical_half_extent(obj["shape"], obj[size_key], pose[3], pose[4])


def test_randomised_scale_keeps_every_object_on_its_nominal_support():
    library = ScenarioLibrary()
    scales = []
    for scenario in ("tabletop_pick", "bin_sort"):
        for seed in range(25):
            scene = library.instantiate(scenario, seed=seed)
            for obj in scene["objects"]:
                scales.append(scene["randomization"]["samples"][obj["id"]]["scale"])
                assert _bottom(obj, "pose", "size_m") == pytest.approx(
                    _bottom(obj, "nominal_pose", "nominal_size_m"), abs=1e-8
                )
                if obj["id"] == "cube":
                    # Tabletop at z = 0: a seated cube's centre is at half its height.
                    assert obj["pose"][2] == pytest.approx(obj["size_m"][2] / 2, abs=1e-8)
    assert min(scales) < 0.97 and max(scales) > 1.03


def test_reseat_handles_tilted_objects(tmp_path):
    raw = yaml.safe_load(DEFAULT_SCENARIO_PATH.read_text(encoding="utf-8"))
    raw["scenarios"] = [
        {
            "id": "tilted",
            "surface": {"size_m": [0.8, 0.6, 0.04], "height_m": 0.72},
            "randomization": {"object_scale_range": [1.2, 1.2]},
            "objects": [
                {
                    "id": "log",
                    "shape": "cylinder",
                    "size_m": [0.02, 0.1],
                    "pose": [0.2, 0.0, 0.02, 90.0, 0.0, 0.0],
                },
                {
                    "id": "wedge",
                    "shape": "box",
                    "size_m": [0.04, 0.02, 0.06],
                    "pose": [0.2, 0.1, 0.0, 0.0, 30.0, 0.0],
                },
            ],
        }
    ]
    path = tmp_path / "tilted.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    scene = ScenarioLibrary(path).instantiate("tilted", seed=3)
    log, wedge = scene["objects"]
    assert log["pose"][2] == pytest.approx(0.02 * 1.2)  # a lying cylinder rests on its radius
    for obj in (log, wedge):
        assert _bottom(obj, "pose", "size_m") == pytest.approx(
            _bottom(obj, "nominal_pose", "nominal_size_m"), abs=1e-9
        )


def _rotation(roll_deg: float, pitch_deg: float) -> np.ndarray:
    roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
    rx = np.array(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    )
    ry = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )
    return ry @ rx


@pytest.mark.parametrize(("roll", "pitch"), [(0, 0), (90, 0), (0, 90), (30, -50), (-120, 15)])
def test_vertical_half_extent_matches_brute_force(roll, pitch):
    rot = _rotation(roll, pitch)
    corners = np.array(
        [[sx * 0.02, sy * 0.03, sz * 0.05] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    box = float(np.max((corners @ rot.T)[:, 2]))
    assert vertical_half_extent("box", [0.04, 0.06, 0.1], roll, pitch) == pytest.approx(box)

    angles = np.linspace(0.0, 2.0 * math.pi, 7201)
    rim = np.stack([0.025 * np.cos(angles), 0.025 * np.sin(angles), np.zeros_like(angles)], axis=1)
    caps = np.concatenate([rim + [0, 0, 0.04], rim - [0, 0, 0.04]])
    cylinder = float(np.max((caps @ rot.T)[:, 2]))
    assert vertical_half_extent("cylinder", [0.025, 0.08], roll, pitch) == pytest.approx(
        cylinder, abs=1e-6
    )
    assert vertical_half_extent("sphere", [0.03], roll, pitch) == 0.03


def test_stale_engine_kinematics_block_is_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_SCENARIO_PATH.read_text(encoding="utf-8"))
    raw["engine"]["kinematics"] = {"base_height_m": 0.1, "upper_arm_m": 0.116}
    path = tmp_path / "stale.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ScenarioError, match="no longer supported"):
        ScenarioLibrary(path)

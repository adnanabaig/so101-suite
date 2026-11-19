"""Determinism, safety-envelope, scene, and API tests for simulation v0.1."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from so101.hardware.safety import DEFAULT_LIMITS
from so101.services.app import create_app
from so101.services.store import get_store
from so101.simulation import SimulationConflict, SimulationEngine, SimulationValidationError
from so101.simulation.capabilities import simulation_capabilities
from so101.simulation.kinematics import JOINT_LIMITS
from so101.simulation.scenarios import ScenarioLibrary


@pytest.fixture(autouse=True)
def _kinematic_engine(monkeypatch):
    """Pin these v0.1 contract tests to the kinematic engine.

    The MuJoCo engine serves the same contract; tests/test_sim_physics_api.py
    runs the contract against both engines.
    """

    from so101.config import get_settings
    from so101.services.routers import simulation as simulation_router

    monkeypatch.setenv("SO101_SIM_ENGINE", "kinematic")
    get_settings.cache_clear()
    simulation_router.reset_simulator()
    yield
    simulation_router.reset_simulator()
    get_settings.cache_clear()


def _engine(*, clock=lambda: 0.0) -> SimulationEngine:
    return SimulationEngine(ScenarioLibrary(), clock=clock)


def test_engine_uses_shared_so101_limits_and_fixed_rate_limited_steps():
    engine = _engine()

    # Effective limits = SafetyGuard envelope intersected with the MJCF ranges.
    for joint, (low, high) in JOINT_LIMITS.items():
        assert DEFAULT_LIMITS[joint][0] <= low < high <= DEFAULT_LIMITS[joint][1]
    assert JOINT_LIMITS["elbow_flex"] == (-96.83, 99.0)
    assert JOINT_LIMITS["shoulder_lift"] == (-101.0, 100.0)
    assert JOINT_LIMITS["wrist_flex"] == (-95.0, 95.0)
    assert JOINT_LIMITS["wrist_roll"] == (-157.21, 160.0)
    assert JOINT_LIMITS["shoulder_pan"] == DEFAULT_LIMITS["shoulder_pan"]
    state = engine.step(1, targets={"shoulder_pan": 90.0, "gripper": 100.0})

    assert state["backend"] == "kinematic"
    assert state["physics"] is False
    assert state["hardware_isolated"] is True
    assert state["writes_motor_commands"] is False
    assert state["step_count"] == 1
    assert state["sim_time_s"] == pytest.approx(1 / 60)
    assert state["telemetry"]["target"]["shoulder_pan"] == 90.0
    assert state["telemetry"]["measured"]["shoulder_pan"] == pytest.approx(1.5)
    assert state["telemetry"]["velocity"]["shoulder_pan"] == 90.0
    assert state["telemetry"]["measured"]["gripper"] == pytest.approx(53.333333333)
    assert state["kinematics"]["model"] == "mjcf:robotstudio_so101@4d038b3"


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ({"shoulder_pan": 110.1}, "outside SO-101 limits"),
        ({"elbow_flex": 99.1}, "outside SO-101 limits"),
        ({"gripper": -0.1}, "outside SO-101 limits"),
        ({"not_a_joint": 0.0}, "unknown SO-101 joints"),
    ],
)
def test_invalid_simulation_targets_are_rejected_not_clamped(targets, message):
    with pytest.raises(SimulationValidationError, match=message):
        _engine().set_targets(targets)


def test_seeded_scene_randomization_is_reproducible_and_keeps_nominal_manifest():
    library = ScenarioLibrary()

    first = library.instantiate("tabletop_pick", seed=4242, randomize=True)
    second = library.instantiate("tabletop_pick", seed=4242, randomize=True)
    different = library.instantiate("tabletop_pick", seed=4243, randomize=True)
    nominal = library.instantiate("tabletop_pick", seed=4242, randomize=False)

    assert first == second
    assert first["objects"] != different["objects"]
    assert nominal["objects"][0]["pose"] == nominal["objects"][0]["nominal_pose"]
    assert nominal["objects"][0]["size_m"] == nominal["objects"][0]["nominal_size_m"]
    assert first["randomization"]["seed"] == 4242
    # Instantiation must not mutate the versioned catalog.
    assert library.catalog()[0]["objects"][0]["pose"] == [0.22, 0.0, 0.02, 0.0, 0.0, 0.0]


def test_packaged_scenario_manifest_matches_operator_config():
    project_manifest = Path(__file__).resolve().parents[1] / "configs/simulation/scenarios.yaml"
    packaged_manifest = Path(__file__).resolve().parents[1] / "src/so101/simulation/scenarios.yaml"

    project = ScenarioLibrary(project_manifest)
    packaged = ScenarioLibrary(packaged_manifest)
    assert packaged.engine == project.engine
    assert packaged.catalog() == project.catalog()
    assert packaged.instantiate("bin_sort", seed=91) == project.instantiate("bin_sort", seed=91)


def test_domain_randomization_can_keep_target_zones_fixed():
    library = ScenarioLibrary()
    first = library.instantiate("bin_sort", seed=10)
    second = library.instantiate("bin_sort", seed=11)

    first_by_id = {obj["id"]: obj for obj in first["objects"]}
    second_by_id = {obj["id"]: obj for obj in second["objects"]}
    assert first_by_id["blue_cube"]["pose"] != second_by_id["blue_cube"]["pose"]
    assert first_by_id["left_target"]["pose"] == second_by_id["left_target"]["pose"]
    assert first["randomization"]["samples"]["left_target"]["applied"] is False


def test_play_pause_converts_elapsed_time_to_integer_fixed_steps():
    now = [10.0]
    engine = _engine(clock=lambda: now[0])
    engine.set_targets({"shoulder_pan": 90.0})
    assert engine.play(rate=2.0)["status"] == "playing"

    now[0] += engine.fixed_dt_s * 5.25
    state = engine.state()
    assert state["step_count"] == 10
    assert state["telemetry"]["measured"]["shoulder_pan"] == 15.0
    with pytest.raises(SimulationConflict, match="pause"):
        engine.step()

    paused = engine.pause()
    now[0] += 5.0
    assert engine.state()["step_count"] == paused["step_count"]


def test_trajectory_preview_is_fixed_step_rate_limited_and_side_effect_free():
    engine = _engine()
    before = copy.deepcopy(engine.state())

    preview = engine.preview_trajectory(
        target={"shoulder_pan": 110.0, "gripper": 0.0},
        duration_s=0.1,
        sample_hz=60.0,
    )

    assert preview["physics"] is False
    assert preview["mutates_state"] is False
    assert preview["fixed_dt_s"] == pytest.approx(1 / 60)
    assert len(preview["points"]) == 7
    # Preview predicts play: the target is a step command from t=0.
    assert all(point["target"]["shoulder_pan"] == 110.0 for point in preview["points"])
    assert preview["points"][-1]["measured"]["shoulder_pan"] == pytest.approx(9.0)
    assert preview["final_reached"] is False
    after = engine.state()
    assert after["step_count"] == before["step_count"]
    assert after["telemetry"] == before["telemetry"]


def test_waypoint_preview_is_deterministic_and_validates_monotonic_time():
    engine = _engine()
    waypoints = [
        {"time_s": 0.5, "targets": {"shoulder_pan": 20.0}},
        {"time_s": 1.0, "targets": {"shoulder_pan": -10.0, "gripper": 25.0}},
    ]
    first = engine.preview_trajectory(waypoints=waypoints, sample_hz=10.0)
    second = engine.preview_trajectory(waypoints=waypoints, sample_hz=10.0)
    assert first == second
    assert first["points"][-1]["target"]["shoulder_pan"] == -10.0

    quantized_rate = engine.preview_trajectory(waypoints=waypoints, sample_hz=17.0)
    assert quantized_rate["requested_sample_hz"] == 17.0
    assert quantized_rate["sample_hz"] == 15.0

    with pytest.raises(SimulationValidationError, match="increasing"):
        engine.preview_trajectory(
            waypoints=[waypoints[1], waypoints[0]],
            sample_hz=10.0,
        )


def test_capability_probe_never_mislabels_package_detection_as_physics():
    capabilities = simulation_capabilities()
    assert capabilities["active_backend"] == "kinematic"
    assert capabilities["physics_active"] is False
    assert capabilities["adapters"]["kinematic"]["active"] is True
    assert capabilities["adapters"]["mujoco"]["active"] is False
    # The physics adapter exists (v0.2), but this process is pinned to kinematic,
    # so it is neither active nor probed.
    assert capabilities["adapters"]["mujoco"]["adapter_implemented"] is True
    assert capabilities["adapters"]["mujoco"]["available"] is None


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        response = test_client.post(
            "/api/simulation/reset",
            json={"scenario_id": "tabletop_pick", "seed": 101, "randomize": True},
        )
        assert response.status_code == 200
        yield test_client
        test_client.post("/api/simulation/pause")


def test_state_step_play_pause_reset_and_scene_api(client: TestClient):
    scenarios = client.get("/api/simulation/scenarios").json()
    assert {scenario["id"] for scenario in scenarios} == {
        "tabletop_pick",
        "bin_sort",
        "calibration_grid",
    }
    assert all(scenario["backend"] == "kinematic" for scenario in scenarios)
    assert all(scenario["physics"] is False for scenario in scenarios)
    assert scenarios[0]["objects"]

    initial = client.get("/api/simulation/state").json()
    assert initial["status"] == "paused"
    assert initial["scene"]["seed"] == 101
    stepped = client.post(
        "/api/simulation/step",
        json={"steps": 2, "targets": {"shoulder_lift": 30.0}},
    )
    assert stepped.status_code == 200, stepped.text
    assert stepped.json()["step_count"] == 2
    assert stepped.json()["telemetry"]["target"]["shoulder_lift"] == 30.0
    assert client.get("/api/simulation/telemetry").json()["source"] == "simulation"

    assert client.post("/api/simulation/play", json={"rate": 1.0}).json()["status"] == "playing"
    conflict = client.post("/api/simulation/step", json={"steps": 1})
    assert conflict.status_code == 409
    assert client.post("/api/simulation/pause").json()["status"] == "paused"

    reset = client.post(
        "/api/simulation/reset",
        json={"scenario_id": "bin_sort", "seed": 77, "randomize": True},
    )
    assert reset.status_code == 200, reset.text
    payload = reset.json()
    assert payload["ok"] is True
    assert "seed 77" in payload["detail"]
    assert payload["state"]["scene"]["scenario_id"] == "bin_sort"
    assert payload["state"]["step_count"] == 0
    assert len(payload["state"]["scene"]["objects"]) == 4


def test_api_trajectory_preview_and_strict_limit_errors(client: TestClient):
    before = client.get("/api/simulation/state").json()
    response = client.post(
        "/api/simulation/trajectory/preview",
        json={
            "target": {"shoulder_pan": 45.0, "gripper": 75.0},
            "duration_s": 1.0,
            "sample_hz": 20.0,
        },
    )
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["points"][0]["time_s"] == 0.0
    assert preview["points"][-1]["time_s"] == 1.0
    assert preview["mutates_state"] is False
    after = client.get("/api/simulation/state").json()
    assert after["step_count"] == before["step_count"]
    assert after["telemetry"] == before["telemetry"]

    outside = client.post("/api/simulation/targets", json={"targets": {"shoulder_pan": 111.0}})
    assert outside.status_code == 422
    assert "SO-101 limits" in outside.json()["detail"]
    unknown = client.post("/api/simulation/step", json={"targets": {"fake_joint": 0.0}})
    assert unknown.status_code == 422
    missing = client.post("/api/simulation/reset", json={"scenario_id": "does_not_exist"})
    assert missing.status_code == 404


def test_simulation_targets_and_steps_are_not_persisted_as_motor_commands(client: TestClient):
    before = len(get_store().list_events(limit=500))
    assert (
        client.post("/api/simulation/targets", json={"targets": {"shoulder_pan": 10.0}}).status_code
        == 200
    )
    assert client.post("/api/simulation/step", json={"steps": 3}).status_code == 200
    after = len(get_store().list_events(limit=500))
    assert after == before


def test_simulation_capability_api_is_truthful(client: TestClient):
    response = client.get("/api/simulation/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend_kind"] == "kinematic"
    assert payload["physics_active"] is False
    assert payload["hardware_isolated"] is True
    assert payload["writes_motor_commands"] is False

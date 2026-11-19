"""PhysicsSimulationEngine: the ``/api/simulation`` contract on MuJoCo physics.

Fake-clock unit tests of the engine, engine selection and the capability
report. ``test_sim_physics_api.py`` runs the HTTP contract on both engines.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from so101 import JOINT_NAMES  # noqa: E402
from so101.simulation import (  # noqa: E402
    ScenarioNotFound,
    SimulationConflict,
    SimulationEngine,
    SimulationValidationError,
)
from so101.simulation import capabilities as capabilities_module  # noqa: E402
from so101.simulation import selection as selection_module  # noqa: E402
from so101.simulation.capabilities import simulation_capabilities  # noqa: E402
from so101.simulation.physics_engine import (  # noqa: E402
    MARKER_ALPHA,
    PhysicsSimulationEngine,
    scene_spec_for,
    servo_speed_limits,
)
from so101.simulation.render import shared_renderer  # noqa: E402
from so101.simulation.scenarios import ScenarioLibrary  # noqa: E402
from so101.simulation.selection import (  # noqa: E402
    EngineUnavailable,
    MujocoStatus,
    create_engine,
    resolve_selection,
)
from so101.simulation.servo import (  # noqa: E402
    STS3215_ACCELERATION_DEG_S2,
    STS3215_MAX_SPEED_DEG_S,
)

LIBRARY = ScenarioLibrary()
TICK = 1 / 60


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def physics(clock: Callable[[], float] | None = None, **kwargs: Any) -> PhysicsSimulationEngine:
    return PhysicsSimulationEngine(LIBRARY, clock=clock or FakeClock(), **kwargs)


def kinematic(clock: Callable[[], float] | None = None) -> SimulationEngine:
    return SimulationEngine(LIBRARY, clock=clock or FakeClock())


def comparable(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "updated_at"}


def validation_error(call: Callable[[], object]) -> str:
    with pytest.raises(SimulationValidationError) as info:
        call()
    return str(info.value)


# -- the contract ----------------------------------------------------------------------
def test_state_is_the_kinematic_contract_plus_physics_state():
    reference = kinematic().state()
    engine = physics()
    state = engine.state()

    assert set(state) == set(reference) | {"physics_state"}
    for key in ("telemetry", "kinematics", "scene"):
        assert set(state[key]) == set(reference[key]), key
    for key in ("measured", "target", "error", "velocity"):
        assert list(state["telemetry"][key]) == list(JOINT_NAMES)
    assert (state["backend"], state["backend_kind"], state["physics"]) == (
        "mujoco",
        "physics",
        True,
    )
    assert state["hardware_isolated"] is True and state["writes_motor_commands"] is False
    for key in ("scheduler", "joint_order", "joint_limits_deg", "fixed_dt_s", "fixed_hz"):
        assert state[key] == reference[key], key
    assert (state["step_count"], state["sim_time_s"], state["status"]) == (0, 0.0, "paused")
    assert state["scene"]["scenario_id"] == reference["scene"]["scenario_id"] == "tabletop_pick"
    assert state["telemetry"]["target"] == reference["telemetry"]["target"]
    for joint in JOINT_NAMES:  # the initial pose holds under gravity
        assert abs(state["telemetry"]["error"][joint]) < 0.5, joint

    assert engine.fixed_dt_s == pytest.approx(TICK)
    assert engine.substeps_per_tick == 4
    assert engine.timestep_s == pytest.approx(1 / 240)
    assert engine.servo_mode == "sts3215"
    assert engine.cameras == ("front", "wrist_cam")


def test_physics_state_reports_the_simulation_truth():
    state = physics().state()
    physics_state = state["physics_state"]

    assert physics_state["engine"] == "mujoco"
    assert physics_state["mujoco_version"] == mujoco.__version__
    assert physics_state["model"] == state["kinematics"]["model"]
    assert physics_state["timestep_s"] == pytest.approx(1 / 240)
    assert physics_state["substeps_per_tick"] == 4
    assert (physics_state["physics_steps"], physics_state["physics_time_s"]) == (0, 0.0)
    assert physics_state["dropped_ticks"] == 0
    assert physics_state["torque_enabled"] is True

    servo = physics_state["servo"]
    assert servo["mode"] == "sts3215"
    assert servo["acceleration_deg_s2"] == pytest.approx(STS3215_ACCELERATION_DEG_S2, abs=1e-3)
    assert servo["speed_units"] == "suite_units_per_second"
    # The manifest's slew limits, all below the STS3215's 270 deg/s.
    assert servo["max_speed"] == pytest.approx(LIBRARY.engine.max_velocity_deg_s, abs=1e-3)
    assert servo["reference"] == pytest.approx(state["telemetry"]["target"], abs=1e-6)
    assert list(physics_state["effort_pct"]) == list(JOINT_NAMES)

    contacts = physics_state["contacts"]
    assert contacts["arm_self"] == 0 and contacts["arm_table"] == 0
    assert contacts["total"] >= 1 and 0.0 <= contacts["max_penetration_m"] < 2e-3
    assert set(physics_state["objects"]) == {"cube"} and physics_state["markers"] == {}
    cube = physics_state["objects"]["cube"]
    assert cube["speed_m_s"] < 1e-3 and cube["grasped"] is False
    assert cube["contacts"]["table"] >= 1
    assert cube["contacts"]["fixed_jaw"] == cube["contacts"]["moving_jaw"] == 0
    assert np.linalg.norm(cube["quat_wxyz"]) == pytest.approx(1.0)
    (scene_cube,) = state["scene"]["objects"]
    assert scene_cube["pose"][:3] == pytest.approx(cube["position_m"], abs=1e-6)


@pytest.mark.parametrize(
    "targets",
    [
        {"shoulder_pan": 110.1},
        {"elbow_flex": 99.1},
        {"gripper": -0.1},
        {"not_a_joint": 0.0},
        {"shoulder_pan": math.nan},
        {"wrist_roll": math.inf},
        {"shoulder_pan": 10.0, "elbow_flex": 99.1},  # nothing is half-applied
    ],
)
def test_targets_are_validated_exactly_like_the_kinematic_engine(targets):
    engine, reference = physics(), kinematic()
    before = comparable(engine.state())

    assert validation_error(lambda: engine.set_targets(targets)) == validation_error(
        lambda: reference.set_targets(targets)
    )
    assert validation_error(lambda: engine.step(2, targets=targets)) == validation_error(
        lambda: reference.step(2, targets=targets)
    )
    assert comparable(engine.state()) == before  # rejected, never clamped


@pytest.mark.parametrize("make", [physics, kinematic], ids=["mujoco", "kinematic"])
def test_step_play_and_reset_rules_are_shared(make):
    engine = make()
    assert engine.MAX_MANUAL_STEPS == SimulationEngine.MAX_MANUAL_STEPS
    for steps in (0, engine.MAX_MANUAL_STEPS + 1):
        with pytest.raises(SimulationValidationError, match="steps must be between"):
            engine.step(steps)
    for rate in (0.05, 4.5, math.nan, math.inf):
        with pytest.raises(SimulationValidationError, match="playback rate"):
            engine.play(rate)
    engine.play(1.0)
    with pytest.raises(SimulationConflict, match="pause simulation"):
        engine.step(1)
    engine.pause()

    before = engine.step(3, targets={"shoulder_pan": 10.0})
    with pytest.raises(ScenarioNotFound):
        engine.reset("no_such_scenario")
    after = engine.state()  # a failed reset leaves the current run intact
    assert comparable(after) == comparable(before)


# -- scheduling --------------------------------------------------------------------------
def test_play_follows_the_monotonic_clock_and_caps_catch_up():
    clock = FakeClock()
    engine = physics(clock)

    engine.play(1.0)
    clock.now = 0.5
    assert engine.state()["step_count"] == 30
    clock.now = 0.5 + 0.5 * TICK
    assert engine.state()["step_count"] == 30  # a partial tick waits
    clock.now = 0.5 + TICK
    state = engine.state()
    assert (state["step_count"], state["status"]) == (31, "playing")
    assert state["sim_time_s"] == pytest.approx(31 * TICK)

    assert engine.pause()["status"] == "paused"
    clock.now = 5.0
    assert engine.state()["step_count"] == 31  # paused: the clock is ignored

    engine.play(2.0)
    clock.now = 5.25
    assert engine.state()["step_count"] == 61  # 0.25 s at 2x

    # Unobserved for 10 s at 2x: 1200 ticks are due, at most 2 s x 2 x 60 run.
    clock.now = 15.25
    state = engine.state()
    physics_state = state["physics_state"]
    assert state["step_count"] == 61 + 240
    assert physics_state["dropped_ticks"] == 1200 - 240
    assert state["sim_time_s"] == pytest.approx(301 * TICK)  # simulated ticks only
    assert physics_state["physics_steps"] == 301 * 4
    assert physics_state["physics_time_s"] == pytest.approx(301 * TICK, abs=1e-9)

    clock.now = 15.25 + TICK  # the skipped backlog is gone: one tick at 2x is 2 steps
    assert engine.state()["step_count"] == 303


def test_play_equals_manual_steps():
    target = {"shoulder_pan": 35.0, "elbow_flex": -25.0, "gripper": 80.0}
    clock = FakeClock(100.0)
    played = physics(clock)
    played.set_targets(target)
    played.play(1.0)
    clock.now = 100.5
    stepped = physics().step(30, targets=target)
    assert played.state()["telemetry"] == stepped["telemetry"]


def test_reset_stops_playback_and_restores_the_initial_target():
    clock = FakeClock()
    engine = physics(clock)
    engine.set_targets({"shoulder_pan": 30.0})
    engine.play(4.0)
    clock.now = 3.0
    assert engine.state()["physics_state"]["dropped_ticks"] > 0

    state = engine.reset("calibration_grid", seed=3)
    assert (state["status"], state["playback_rate"]) == ("paused", 1.0)
    assert (state["step_count"], state["sim_time_s"], state["generation"]) == (0, 0.0, 1)
    assert state["physics_state"]["dropped_ticks"] == 0
    assert state["physics_state"]["objects"] == {}
    assert state["telemetry"]["target"] == LIBRARY.engine.initial_joint_positions
    clock.now = 60.0
    assert engine.state()["step_count"] == 0


# -- preview -----------------------------------------------------------------------------
def _mid_motion() -> PhysicsSimulationEngine:
    engine = physics()
    # A pending target and a moving servo profile: the preview must fork both.
    engine.step(7, targets={"shoulder_pan": -30.0, "elbow_flex": 40.0})
    return engine


def test_preview_runs_on_a_fork_and_never_mutates_live_state():
    engine = _mid_motion()
    before = engine.state()
    data = engine._world.data
    raw = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))

    preview = engine.preview_trajectory(
        target={"shoulder_pan": 45.0, "gripper": 90.0}, duration_s=1.0
    )
    engine.preview_trajectory(
        waypoints=[
            {"time_s": 0.3, "targets": {"wrist_flex": 30.0}},
            {"time_s": 0.6, "targets": {"elbow_flex": -20.0}},
        ]
    )

    assert engine.state() == before  # including updated_at
    assert np.array_equal(data.qpos, raw[0]) and np.array_equal(data.qvel, raw[1])
    assert np.array_equal(data.ctrl, raw[2]) and float(data.time) == raw[3]
    assert preview["mutates_state"] is False and preview["source_step_count"] == 7
    assert preview["starting_target"] == before["telemetry"]["target"]


def test_preview_predicts_manual_steps_exactly():
    engine = _mid_motion()
    target = {"shoulder_pan": 45.0, "wrist_flex": -20.0, "gripper": 90.0}
    preview = engine.preview_trajectory(target=target, duration_s=1.0, sample_hz=60.0)

    assert (preview["backend"], preview["physics"]) == ("mujoco", True)
    assert preview["semantics"] == "step_targets_like_play"
    assert preview["final_reached"] is True and preview["reached_tolerance"] == 0.5
    points = preview["points"]
    assert [point["step"] for point in points] == list(range(61))
    assert points[0]["measured"] == engine.state()["telemetry"]["measured"]

    engine.set_targets(target)
    for point in points[1:]:
        state = engine.step(1)
        assert state["telemetry"]["measured"] == point["measured"], point["step"]
        assert state["telemetry"]["target"] == point["target"]
        assert state["kinematics"]["position_m"] == point["end_effector"]["position_m"]


def test_waypoint_preview_predicts_the_step_sequence():
    engine = _mid_motion()
    waypoints = [
        {"time_s": 0.25, "targets": {"shoulder_pan": 20.0}},
        {"time_s": 0.5, "targets": {"elbow_flex": 30.0, "gripper": 10.0}},
        {"time_s": 0.75, "targets": {"shoulder_pan": -10.0}},
    ]
    preview = engine.preview_trajectory(waypoints=waypoints, sample_hz=60.0)
    by_step = {point["step"]: point for point in preview["points"]}

    tick = 0
    for waypoint in waypoints:
        end = round(waypoint["time_s"] * 60)
        engine.set_targets(waypoint["targets"])
        state = engine.step(end - tick)
        tick = end
        assert state["telemetry"]["measured"] == by_step[end]["measured"], end


def test_an_unreachable_preview_is_not_final_reached():
    engine = physics()
    engine.reset("calibration_grid", seed=1)
    # The table stops the arm well before shoulder_lift reaches 60 degrees.
    preview = engine.preview_trajectory(
        target={"shoulder_lift": 60.0, "elbow_flex": 0.0}, duration_s=2.0
    )
    assert preview["final_reached"] is False
    assert preview["points"][-1]["error"]["shoulder_lift"] > 10.0


# -- the physics ------------------------------------------------------------------------
def test_joints_lag_their_targets_like_a_servo():
    reference = kinematic().step(1, targets={"shoulder_pan": 90.0})
    engine = physics()
    state = engine.step(1, targets={"shoulder_pan": 90.0})

    pan = state["telemetry"]["measured"]["shoulder_pan"]
    assert reference["telemetry"]["measured"]["shoulder_pan"] == pytest.approx(1.5)
    assert 0.0 < pan < 1.5  # acceleration-limited, unlike the kinematic slew
    assert state["telemetry"]["target"]["shoulder_pan"] == 90.0

    state = engine.step(120)
    assert abs(state["telemetry"]["error"]["shoulder_pan"]) < 0.5
    assert abs(state["telemetry"]["velocity"]["shoulder_pan"]) < 1.0


def test_the_table_stops_the_arm_and_contacts_are_reported():
    engine = physics()
    engine.reset("calibration_grid", seed=1)
    state = engine.step(180, targets={"shoulder_lift": 60.0, "elbow_flex": 0.0})
    physics_state = state["physics_state"]

    assert physics_state["contacts"]["arm_table"] >= 1
    assert physics_state["contacts"]["max_penetration_m"] < 2e-3
    assert state["telemetry"]["error"]["shoulder_lift"] > 10.0
    assert physics_state["effort_pct"]["shoulder_lift"] == pytest.approx(100.0, abs=0.5)


def test_scene_poses_are_live_rigid_body_poses():
    engine = physics()
    engine._world.set_object_pose("cube", (0.15, -0.15, 0.08), (1.0, 0.0, 0.0, 0.0))

    falling = engine.step(4)["physics_state"]["objects"]["cube"]
    assert falling["speed_m_s"] > 0.3 and falling["contacts"]["table"] == 0

    state = engine.step(60)
    (cube,) = state["scene"]["objects"]
    resting = state["physics_state"]["objects"]["cube"]
    assert cube["pose"][:2] == pytest.approx([0.15, -0.15], abs=1e-3)
    assert cube["pose"][2] == pytest.approx(cube["size_m"][-1] / 2, abs=2e-3)
    assert cube["pose"][:3] == pytest.approx(resting["position_m"], abs=1e-6)
    assert resting["speed_m_s"] < 1e-3 and resting["contacts"]["table"] >= 1


def test_reset_is_seeded_and_settles_bodies_where_the_scenario_put_them():
    first, second = physics(), physics()
    state = first.reset("bin_sort", seed=7)
    assert comparable(second.reset("bin_sort", seed=7)) == comparable(state)
    assert (state["generation"], state["step_count"]) == (1, 0)

    physics_state = state["physics_state"]
    assert set(physics_state["objects"]) == {"blue_cube", "orange_cylinder"}
    assert set(physics_state["markers"]) == {"left_target", "right_target"}
    nominal = {
        item["id"]: item["pose"] for item in LIBRARY.instantiate("bin_sort", seed=7)["objects"]
    }
    live = {item["id"]: item["pose"] for item in state["scene"]["objects"]}
    for object_id, body in physics_state["objects"].items():
        assert body["speed_m_s"] < 1e-3 and body["grasped"] is False, object_id
        assert body["contacts"]["table"] >= 1, object_id
        assert live[object_id][:3] == pytest.approx(nominal[object_id][:3], abs=2e-3)
    for marker_id, position in physics_state["markers"].items():
        assert live[marker_id] == nominal[marker_id]  # visual-only: never simulated
        assert position == pytest.approx(nominal[marker_id][:3], abs=1e-9)

    other = first.reset("bin_sort", seed=8)
    assert other["generation"] == 2
    assert other["physics_state"]["objects"] != physics_state["objects"]


def test_step_servo_mode_and_bad_modes():
    servo = physics(servo_mode="step").state()["physics_state"]["servo"]
    assert servo["mode"] == "step" and servo["acceleration_deg_s2"] is None
    with pytest.raises(ValueError, match="servo mode"):
        physics(servo_mode="pwm")


def test_scene_spec_maps_target_zones_to_markers():
    scene = LIBRARY.instantiate("bin_sort", seed=7)
    spec, placements = scene_spec_for(scene, timestep_s=1 / 240)

    assert [item.id for item in spec.objects] == ["blue_cube", "orange_cylinder"]
    assert [item.id for item in spec.markers] == ["left_target", "right_target"]
    assert all(item.rgba[3] == MARKER_ALPHA for item in spec.markers)
    assert spec.table_size_m == tuple(scene["surface"]["size_m"])
    assert spec.table_height_m == scene["surface"]["height_m"]
    assert spec.timestep_s == pytest.approx(1 / 240)
    assert [item.object_id for item in placements] == [item["id"] for item in scene["objects"]]
    assert [item.marker for item in placements] == [False, False, True, True]
    for placement, item in zip(placements, scene["objects"], strict=True):
        assert placement.position_m == tuple(item["pose"][:3])
        assert np.linalg.norm(placement.quat_wxyz) == pytest.approx(1.0)


def test_servo_speed_limits_follow_the_manifest_under_the_sts3215_cap():
    limits = servo_speed_limits(LIBRARY.engine.max_velocity_deg_s)
    assert limits["shoulder_pan"] == pytest.approx(90.0)
    assert all(0.0 < value <= STS3215_MAX_SPEED_DEG_S for value in limits.values())
    capped = servo_speed_limits(dict.fromkeys(JOINT_NAMES, 10_000.0))
    assert all(value == STS3215_MAX_SPEED_DEG_S for value in capped.values())


# -- concurrency and rendering ----------------------------------------------------------
def test_concurrent_requests_are_serialized():
    engine = physics()
    errors: list[BaseException] = []

    def stepper() -> None:
        try:
            for _ in range(25):
                engine.step(1, targets={"shoulder_pan": 20.0})
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    def observer() -> None:
        try:
            for _ in range(10):
                engine.state()
                engine.preview_trajectory(target={"elbow_flex": 10.0}, duration_s=0.1)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=stepper) for _ in range(4)]
    threads += [threading.Thread(target=observer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    state = engine.state()
    assert state["step_count"] == 100
    assert state["physics_state"]["physics_steps"] == 400


def test_render_frame_validates_then_draws_the_live_scene():
    engine = physics()
    with pytest.raises(SimulationValidationError, match="unknown camera"):
        engine.render_frame("ceiling", 64, 48)
    with pytest.raises(SimulationValidationError, match="outside"):
        engine.render_frame("front", 8, 48)
    if not shared_renderer().available:
        pytest.skip(f"no OpenGL here: {shared_renderer().error}")

    front, meta = engine.render_frame("front", 64, 48)
    assert front.shape == (48, 64, 3) and front.dtype == np.uint8 and front.std() > 5.0
    assert meta == {
        "camera": "front",
        "width": 64,
        "height": 48,
        "step_count": 0,
        "generation": 0,
        "sim_time_s": 0.0,
    }
    wrist, _ = engine.render_frame("wrist_cam", 64, 48)
    assert (np.abs(front.astype(int) - wrist.astype(int)) > 16).mean() > 0.01
    engine.step(1)
    assert engine.render_frame("front", 64, 48)[1]["step_count"] == 1


# -- engine selection and capabilities ---------------------------------------------------
def test_kinematic_selection_needs_no_physics():
    chosen = resolve_selection("kinematic")
    assert (chosen.requested, chosen.active, chosen.physics) == ("kinematic", "kinematic", False)
    engine, selected = create_engine("kinematic", LIBRARY)
    assert type(engine) is SimulationEngine and selected == chosen


@pytest.mark.parametrize("requested", ["auto", None, " MuJoCo "])
def test_auto_and_mujoco_select_physics_when_it_runs(requested):
    chosen = resolve_selection(requested)
    assert chosen.active == "mujoco" and chosen.physics and chosen.error is None
    engine, selected = create_engine(requested, LIBRARY)
    assert isinstance(engine, PhysicsSimulationEngine) and selected == chosen
    assert engine.library is LIBRARY


def test_unknown_engines_are_rejected():
    with pytest.raises(EngineUnavailable, match="unknown simulation engine"):
        resolve_selection("bullet")


def test_auto_falls_back_when_mujoco_cannot_run_and_mujoco_refuses(monkeypatch):
    broken = MujocoStatus(False, True, "3.0.0", "RuntimeError: gladLoadGL error")
    monkeypatch.setattr(selection_module, "mujoco_status", lambda: broken)
    monkeypatch.setattr(capabilities_module, "mujoco_status", lambda: broken)

    auto = resolve_selection("auto")
    assert auto.active == "kinematic" and auto.error == broken.error
    assert "gladLoadGL" in auto.reason
    engine, selected = create_engine("auto", LIBRARY)
    assert type(engine) is SimulationEngine and selected == auto

    explicit = resolve_selection("mujoco")
    assert explicit.active is None and "cannot run here" in explicit.reason
    with pytest.raises(EngineUnavailable, match="gladLoadGL"):
        create_engine("mujoco", LIBRARY)

    report = simulation_capabilities(auto)
    adapter = report["adapters"]["mujoco"]
    assert report["active_backend"] == "kinematic" and report["physics_active"] is False
    assert adapter["available"] is False and adapter["active"] is False
    assert adapter["error"] == broken.error and adapter["selection_reason"] == auto.reason
    none = simulation_capabilities(explicit)
    assert (none["active_backend"], none["backend_kind"]) == ("none", "none")


def test_auto_falls_back_when_the_physics_engine_fails_to_build(monkeypatch):
    def broken_init(self: PhysicsSimulationEngine, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("model failed to compile")

    monkeypatch.setattr(PhysicsSimulationEngine, "__init__", broken_init)
    engine, selected = create_engine("auto", LIBRARY)
    assert type(engine) is SimulationEngine
    assert selected.active == "kinematic" and "model failed to compile" in selected.reason
    with pytest.raises(EngineUnavailable, match="model failed to compile"):
        create_engine("mujoco", LIBRARY)


def test_capabilities_report_the_engine_that_runs():
    engine = physics()
    report = simulation_capabilities(resolve_selection("auto"), engine=engine, probe_render=False)
    assert (report["active_backend"], report["backend_kind"]) == ("mujoco", "physics")
    assert report["physics_active"] is True and report["writes_motor_commands"] is False
    adapter = report["adapters"]["mujoco"]
    assert adapter["package_installed"] is True
    assert adapter["package_version"] == mujoco.__version__
    assert adapter["adapter_implemented"] is True
    assert adapter["available"] is True and adapter["active"] is True
    assert adapter["engine_setting"] == "auto" and adapter["error"] is None
    assert adapter["fixed_hz"] == pytest.approx(60.0)
    assert adapter["timestep_s"] == pytest.approx(1 / 240)
    assert adapter["substeps_per_tick"] == 4 and adapter["servo_mode"] == "sts3215"
    assert adapter["cameras"] == ["front", "wrist_cam"]
    assert adapter["model"]["vendored"] is True and len(adapter["model"]["commit"]) == 40
    assert adapter["render_available"] is None and adapter["render_status"] is None  # not probed
    assert report["adapters"]["kinematic"]["active"] is False
    for name, item in report["adapters"].items():  # the UI reads these from every adapter
        assert {"active", "kind", "detail"} <= set(item), name

    # The report never waits for the GL probe, so let it finish first.
    available = shared_renderer().available
    probed = simulation_capabilities(resolve_selection("auto"), engine=engine)["adapters"]["mujoco"]
    assert probed["render_available"] is available
    assert probed["render_status"] == (
        "available" if available else f"unavailable: {probed['render_error']}"
    )

    pinned = simulation_capabilities(resolve_selection("kinematic"))
    adapter = pinned["adapters"]["mujoco"]
    assert pinned["active_backend"] == "kinematic" and pinned["physics_active"] is False
    assert adapter["available"] is None and adapter["active"] is False  # never probed
    assert adapter["fixed_hz"] is None and adapter["render_available"] is None
    assert "Not probed" in adapter["detail"]

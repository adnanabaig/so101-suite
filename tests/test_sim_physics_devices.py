"""Sim backend devices: SimRuntime, MujocoBus, SimExpertLeader and SimCamera.

Most tests use a lockstep runtime (``realtime=False``) and are exact. The
real-time thread has its own tests, paced by a fake clock where the numbers
matter. The devices are put into the real follower, leader and camera
objects by hand. The wave-2 ``make_bus``/``CameraManager`` integration will
do it the same way, and this stream does not touch ``so101.hardware``.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101 import JOINT_NAMES  # noqa: E402
from so101 import config as config_module  # noqa: E402
from so101.hardware.cameras import CameraManager, CameraUnavailableError  # noqa: E402
from so101.hardware.follower import SO101Follower  # noqa: E402
from so101.hardware.leader import SO101Leader  # noqa: E402
from so101.hardware.motors import MotorBus  # noqa: E402
from so101.hardware.safety import SafetyGuard  # noqa: E402
from so101.simulation import devices as devices_module  # noqa: E402
from so101.simulation.devices import (  # noqa: E402
    MAX_CATCHUP_S,
    MujocoBus,
    SimCamera,
    SimExpertLeader,
    SimRuntime,
    get_sim_runtime,
    make_sim_bus,
    make_sim_camera,
    on_episode_end,
    on_episode_start,
    reset_sim_runtime,
    stop_sim_runtime,
)
from so101.simulation.render import FrameRenderer, shared_renderer  # noqa: E402
from so101.simulation.rest_pose import MODEL_REST_POSE  # noqa: E402
from so101.teleop.controller import TeleopController  # noqa: E402

PAN_CTRL_HIGH = 109.99987526  # the actuator range ends a hair inside the 110 deg limit


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture(autouse=True)
def _fresh_singleton() -> Iterator[None]:
    reset_sim_runtime()
    yield
    reset_sim_runtime()


@pytest.fixture
def runtime() -> SimRuntime:
    return SimRuntime("pick_lift", realtime=False, seed=0)


@pytest.fixture
def bus(runtime: SimRuntime) -> Iterator[MujocoBus]:
    bus = MujocoBus(runtime=runtime)
    bus.connect()
    yield bus
    bus.disconnect()


def enable(bus: MujocoBus) -> dict[str, float]:
    """What ``SO101Follower.connect`` does: prime the goal register, then torque on."""

    present = bus.read_positions()
    bus.write_positions(present)
    bus.set_torque(True)
    return present


# -- MotorBus protocol --------------------------------------------------------------------
def test_buses_implement_the_motor_bus_protocol(runtime: SimRuntime) -> None:
    follower = make_sim_bus("follower", runtime=runtime)
    leader = make_sim_bus("SO101 Leader arm", runtime=runtime)
    assert isinstance(follower, MujocoBus) and isinstance(leader, SimExpertLeader)
    assert isinstance(follower, MotorBus) and isinstance(leader, MotorBus)
    assert type(make_sim_bus("arm", runtime=runtime)) is MujocoBus


@pytest.mark.parametrize("kind", [MujocoBus, SimExpertLeader])
def test_bus_calls_need_a_connection_and_connect_is_idempotent(
    runtime: SimRuntime, kind: type[MujocoBus] | type[SimExpertLeader]
) -> None:
    bus = kind(runtime=runtime)
    assert not bus.connected
    with pytest.raises(RuntimeError, match="bus not connected"):
        bus.read_positions()
    with pytest.raises(RuntimeError, match="bus not connected"):
        bus.write_positions({"gripper": 10.0})
    with pytest.raises(RuntimeError, match="bus not connected"):
        bus.set_torque(True)
    bus.disconnect()  # nothing to do

    bus.connect()
    bus.connect()
    assert bus.connected and sum(runtime.status()["devices"].values()) == 1
    assert list(bus.read_positions()) == list(JOINT_NAMES)
    bus.disconnect()
    bus.disconnect()
    assert not bus.connected and runtime.status()["devices"] == {}
    with pytest.raises(RuntimeError, match="bus not connected"):
        bus.read_positions()


def test_power_on_is_folded_at_rest_with_torque_off(runtime: SimRuntime, bus: MujocoBus) -> None:
    status = runtime.status()
    assert status["torque_enabled"] is False and status["sim_time_s"] == 0.0
    present = bus.read_positions()
    for joint, rest in MODEL_REST_POSE.items():
        assert present[joint] == pytest.approx(rest, abs=3.0)  # settled onto its stops
    # Goal_Position holds the present pose at boot: enabling torque moves nothing.
    assert runtime.goal_positions() == pytest.approx(present, abs=1e-9)
    bus.set_torque(True)
    runtime.advance(1.0)
    assert bus.read_positions() == pytest.approx(present, abs=0.05)


def test_writes_are_partial_ordered_clipped_and_validated(
    runtime: SimRuntime, bus: MujocoBus
) -> None:
    applied = bus.write_positions({"gripper": 40.0, "bogus": 1.0, "shoulder_pan": 110.0})
    assert list(applied) == ["shoulder_pan", "gripper"]  # JOINT_NAMES order; unknown ignored
    assert applied["shoulder_pan"] == pytest.approx(PAN_CTRL_HIGH, abs=1e-6)
    assert applied["shoulder_pan"] < 110.0
    assert applied["gripper"] == 40.0
    goal = runtime.goal_positions()
    assert goal["gripper"] == 40.0 and goal["shoulder_pan"] == applied["shoulder_pan"]

    # Unclipped values come back exactly, so a SafetyGuard-clamped command
    # never reads back as just outside the guard's range.
    rng = np.random.default_rng(0)
    for value in rng.uniform(-99.0, 99.0, 50):
        assert bus.write_positions({"shoulder_lift": float(value)}) == {
            "shoulder_lift": float(value)
        }
    assert bus.write_positions({"gripper": 0.0, "wrist_roll": 160.0}) == {
        "wrist_roll": 160.0,
        "gripper": 0.0,
    }

    before = runtime.goal_positions()
    with pytest.raises(ValueError, match="non-finite"):
        bus.write_positions({"gripper": 10.0, "elbow_flex": math.nan})
    assert runtime.goal_positions() == before  # nothing was applied
    assert bus.write_positions({}) == {}
    assert bus.write_positions({"bogus": 1.0}) == {}


# -- torque ----------------------------------------------------------------------------------
def test_a_de_energized_arm_sags_and_a_primed_torque_on_holds_it(
    runtime: SimRuntime, bus: MujocoBus
) -> None:
    info = runtime.on_episode_start()  # the start pose; the torque stays off
    assert info["torque_enabled"] is False
    start = bus.read_positions()
    runtime.advance(1.0)
    sagged = bus.read_positions()
    assert max(abs(sagged[j] - start[j]) for j in ("shoulder_lift", "elbow_flex")) > 2.0

    held = enable(bus)
    runtime.advance(2.0)
    assert bus.read_positions() == pytest.approx(held, abs=0.1)


def test_enabling_torque_drives_a_stale_goal_along_the_profile(
    runtime: SimRuntime, bus: MujocoBus
) -> None:
    start = bus.read_positions()["shoulder_pan"]
    bus.write_positions({"shoulder_pan": start + 30.0})  # torque off: only the register changes
    runtime.advance(0.5)
    assert bus.read_positions()["shoulder_pan"] == pytest.approx(start, abs=0.01)

    bus.set_torque(True)  # as on the servo: goal RAM is live the moment torque comes on
    pan = []
    for _ in range(30):
        runtime.advance(1.0 / 30.0)
        pan.append(bus.read_positions()["shoulder_pan"] - start)
    assert 0.0 < pan[0] < 3.0  # 2232 deg/s^2 for 1/30 s, not a jump
    assert pan[-1] == pytest.approx(30.0, abs=0.1)


def test_follower_disconnect_disables_torque(runtime: SimRuntime) -> None:
    follower = SO101Follower(guard=SafetyGuard())
    follower.bus = MujocoBus(runtime=runtime)
    follower.connect()
    assert runtime.torque_enabled and follower.connected
    follower.disconnect()
    assert not runtime.torque_enabled and not follower.connected
    assert runtime.status()["devices"] == {}


# -- time -------------------------------------------------------------------------------------
def test_lockstep_writes_advance_one_period_and_reads_see_the_pre_command_state(
    runtime: SimRuntime, bus: MujocoBus
) -> None:
    present = enable(bus)
    before = runtime.status()
    target = present["shoulder_pan"] + 20.0
    bus.write_positions({"shoulder_pan": target})
    after = runtime.status()
    assert after["physics_steps"] - before["physics_steps"] == runtime.control_substeps == 8
    assert after["sim_time_s"] - before["sim_time_s"] == pytest.approx(1.0 / 30.0, abs=1e-6)
    # The period ran under the previous goal: the observation read right after
    # the write is the state the new command acts on, as on hardware.
    assert bus.read_positions()["shoulder_pan"] == pytest.approx(present["shoulder_pan"], abs=1e-3)
    for _ in range(5):
        bus.read_positions()
    assert runtime.status()["physics_steps"] == after["physics_steps"]  # reads never advance

    bus.write_positions({"shoulder_pan": target})
    assert bus.read_positions()["shoulder_pan"] > present["shoulder_pan"] + 0.5

    bus.set_torque(False)
    steps = runtime.status()["physics_steps"]
    bus.write_positions({"shoulder_pan": present["shoulder_pan"]})
    assert runtime.status()["physics_steps"] == steps  # a torque-off write is a register write


def _trajectory(seed: int) -> np.ndarray:
    runtime = SimRuntime("pick_lift", realtime=False, seed=seed)
    bus = MujocoBus(runtime=runtime)
    bus.connect()
    runtime.on_episode_start()
    enable(bus)
    rows = []
    for k in range(20):
        bus.write_positions({"shoulder_pan": 2.0 * k, "gripper": 50.0 - k})
        rows.append(list(bus.read_positions().values()))
    bus.disconnect()
    return np.array(rows)


def test_lockstep_is_deterministic() -> None:
    first = _trajectory(5)
    assert np.array_equal(first, _trajectory(5))
    assert not np.array_equal(first, _trajectory(6))


def test_realtime_thread_follows_its_clock_and_drops_long_stalls() -> None:
    clock = FakeClock()
    runtime = SimRuntime("reach", realtime=True, seed=0, clock=clock)
    dt = runtime.world.timestep_s
    anchor = clock.t
    catch_up = runtime.world.steps_for(MAX_CATCHUP_S)

    def owed() -> int:  # the thread's own arithmetic
        return math.floor((clock.t - anchor) / dt)

    try:
        runtime.start()
        runtime.start()
        assert runtime.running
        clock.t += 0.2
        first = owed()
        assert 0 < first <= catch_up
        assert wait_for(lambda: runtime.status()["physics_steps"] == first)
        time.sleep(0.05)
        assert runtime.status()["physics_steps"] == first  # no clock time, no physics

        clock.t += 10.0  # a stall: catch up by at most 0.25 s and drop the rest
        assert wait_for(lambda: runtime.status()["physics_steps"] == first + catch_up)
        dropped = owed() - first - catch_up
        assert runtime.status()["dropped_steps"] == dropped > 0
        with pytest.raises(RuntimeError, match="running"):
            runtime.advance(0.1)
    finally:
        runtime.stop()
        runtime.stop()
    assert not runtime.running
    runtime.advance(0.1)  # a stopped runtime advances by hand
    assert runtime.status()["physics_steps"] == first + catch_up + 24

    clock.t += 100.0  # stopped time is not owed: a restart neither replays nor drops it
    runtime.start()
    try:
        time.sleep(0.05)
        status = runtime.status()
        assert status["physics_steps"] == first + catch_up + 24
        assert status["dropped_steps"] == dropped
    finally:
        runtime.stop()


def test_realtime_runtime_keeps_pace_with_the_wall_clock() -> None:
    runtime = SimRuntime("reach", realtime=True, seed=0)
    bus = MujocoBus(runtime=runtime)
    bus.connect()  # starts the physics thread
    try:
        assert runtime.running
        sim0, wall0 = runtime.status()["sim_time_s"], time.monotonic()
        time.sleep(0.6)
        sim = runtime.status()["sim_time_s"] - sim0
        wall = time.monotonic() - wall0
        assert wall - 0.25 <= sim <= wall + 0.05
    finally:
        bus.disconnect()
    assert runtime.running  # physics outlives the devices: a de-energized arm keeps falling
    runtime.stop()
    assert not runtime.running


# -- thread safety ---------------------------------------------------------------------------
def test_concurrent_reads_writes_and_torque_toggles_are_safe() -> None:
    runtime = SimRuntime("reach", realtime=True, seed=0)
    buses = [MujocoBus(name=f"follower{i}", runtime=runtime) for i in range(4)]
    for bus in buses:
        bus.connect()
    enable(buses[0])
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker(index: int, bus: MujocoBus) -> None:
        rng = np.random.default_rng(index)
        try:
            while not stop.is_set():
                present = bus.read_positions()
                assert list(present) == list(JOINT_NAMES)
                assert all(math.isfinite(value) for value in present.values())
                goal = {
                    "shoulder_pan": float(rng.uniform(-30.0, 30.0)),
                    "gripper": float(rng.uniform(10.0, 90.0)),
                }
                assert bus.write_positions(goal) == goal
        except BaseException as exc:  # reported by the main thread
            errors.append(exc)

    def toggler(bus: MujocoBus) -> None:
        try:
            while not stop.is_set():
                bus.set_torque(False)
                bus.set_torque(True)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i, bus)) for i, bus in enumerate(buses)]
    threads.append(threading.Thread(target=toggler, args=(buses[1],)))
    for thread in threads:
        thread.start()
    time.sleep(0.8)
    # Physics must keep stepping while every thread contends for the runtime.
    # Wait for the steps against a deadline: a loaded machine paces slower.
    deadline = time.monotonic() + 10.0
    while runtime.status()["physics_steps"] <= 60 and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    for thread in threads:
        thread.join(5.0)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    status = runtime.status()
    assert status["fault"] is None and status["physics_steps"] > 60
    for bus in buses:
        bus.disconnect()
    assert runtime.status()["devices"] == {} and not runtime.torque_enabled
    runtime.stop()


def test_concurrent_lockstep_writes_each_advance_exactly_one_period(runtime: SimRuntime) -> None:
    buses = [MujocoBus(name=f"follower{i}", runtime=runtime) for i in range(4)]
    for bus in buses:
        bus.connect()
    enable(buses[0])
    steps = runtime.status()["physics_steps"]
    writes = 25
    errors: list[BaseException] = []

    def worker(index: int, bus: MujocoBus) -> None:
        try:
            for k in range(writes):
                bus.write_positions({"shoulder_pan": float(index * 5 + k % 5)})
                bus.read_positions()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i, bus)) for i, bus in enumerate(buses)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
    assert not errors, errors
    expected = len(buses) * writes * runtime.control_substeps
    assert runtime.status()["physics_steps"] - steps == expected
    for bus in buses:
        bus.disconnect()


# -- the expert leader ---------------------------------------------------------------------------
def test_expert_leader_reports_targets_and_ignores_feedback(runtime: SimRuntime) -> None:
    leader = SimExpertLeader(runtime=runtime)
    leader.connect()
    status = runtime.status()
    assert status["expert_active"] and status["devices"] == {"leader": 1}
    first = leader.read_positions()
    assert list(first) == list(JOINT_NAMES)
    assert first == pytest.approx(runtime.read_positions(), abs=0.05)  # starts from the arm
    assert leader.write_positions({"gripper": 3.0, "bogus": 1.0}) == {"gripper": 3.0}
    with pytest.raises(ValueError, match="non-finite"):
        leader.write_positions({"gripper": math.inf})
    leader.set_torque(True)  # accepted; a passive arm has nothing to energize
    assert not runtime.torque_enabled  # and the follower's torque is untouched
    assert leader.read_positions() == first  # no control period has passed
    runtime.advance(1.0 / 30.0)
    assert leader.read_positions() != first
    leader.disconnect()
    assert not runtime.status()["expert_active"]


def test_an_expert_failure_reaches_the_leader_not_the_physics(
    runtime: SimRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    leader = SimExpertLeader(runtime=runtime)
    leader.connect()
    expert = runtime._expert
    assert expert is not None
    real = expert.select_action
    failing = {"now": True}

    def flaky(observation: Mapping[str, Any]) -> np.ndarray:
        if failing["now"]:
            raise ArithmeticError("IK blew up")
        return real(observation)

    monkeypatch.setattr(expert, "select_action", flaky)
    runtime.advance(1.0 / 30.0)
    with pytest.raises(RuntimeError, match="IK blew up"):
        leader.read_positions()
    assert runtime.status()["fault"] is None and "IK blew up" in runtime.status()["expert_error"]
    runtime.advance(0.1)  # the physics carries on
    failing["now"] = False
    runtime.on_episode_start()  # a new episode restarts the expert
    assert runtime.status()["expert_error"] is None
    assert list(leader.read_positions()) == list(JOINT_NAMES)


@pytest.mark.parametrize("task", ["reach", "pick_lift", "pick_place"])
def test_teleop_with_the_expert_leader_succeeds_in_lockstep(task: str) -> None:
    """The real TeleopController, SO101Follower/SafetyGuard and SO101Leader, end to end."""

    runtime = SimRuntime(task, realtime=False, seed=1)
    clock = FakeClock()  # lockstep: the guard sees one nominal period per step
    follower = SO101Follower(guard=SafetyGuard(_clock=clock))
    follower.bus = MujocoBus(runtime=runtime)
    leader = SO101Leader()
    leader.bus = SimExpertLeader(runtime=runtime)
    controller = TeleopController(leader, follower)
    controller.connect()
    assert runtime.torque_enabled

    info = runtime.on_episode_start()  # what RobotManager.begin_episode will call
    follower.guard.seed(follower.read_positions())  # mandatory after the start-pose move
    assert info["seed"] == 1 and info["instruction"] == runtime.task.instruction
    for _ in range(runtime.task.episode_steps()):
        clock.t += 1.0 / 30.0
        step = controller.step()
        if runtime.task.first_success_s is not None:
            break
    assert set(step.action) == {f"{joint}.pos" for joint in JOINT_NAMES}
    assert set(step.observation) == set(step.action)

    outcome = runtime.on_episode_end()
    assert outcome["success"] is True, outcome
    json.dumps(outcome, allow_nan=False)  # what cli_sim.label_episode stores
    metrics = outcome["metrics"]
    assert metrics["task"] == task and metrics["seed"] == 1 and metrics["fault"] is None
    assert 0.0 < metrics["first_success_s"] <= runtime.task.episode_s

    controller.disconnect()
    assert not runtime.torque_enabled and runtime.status()["devices"] == {}


# -- episode hooks ------------------------------------------------------------------------------
def test_episode_hooks_reseed_report_and_keep_the_torque_state(
    runtime: SimRuntime, bus: MujocoBus
) -> None:
    info = runtime.on_episode_start()
    assert (info["seed"], info["episode"], info["torque_enabled"]) == (0, 1, False)
    assert info["task"] == "pick_lift" and info["instruction"] == runtime.task.instruction
    assert bus.read_positions() == pytest.approx(info["start_pose"], abs=0.5)
    json.dumps(info, allow_nan=False)

    second = runtime.on_episode_start()
    assert (second["seed"], second["episode"]) == (1, 2)
    assert second["layout"] != info["layout"]
    again = runtime.on_episode_start(0)
    assert again["layout"] == info["layout"] and again["start_pose"] == info["start_pose"]

    enable(bus)
    assert runtime.on_episode_start(3)["torque_enabled"] is True and runtime.torque_enabled
    runtime.advance(0.5)
    outcome = runtime.on_episode_end()
    assert outcome["success"] is False
    json.dumps(outcome, allow_nan=False)
    metrics = outcome["metrics"]
    assert (metrics["seed"], metrics["episode"], metrics["first_success_s"]) == (3, 4, None)
    assert metrics["sim_time_s"] == pytest.approx(0.5, abs=1e-3)
    assert metrics["final"]["success"] == 0.0 and metrics["fault"] is None


def test_recording_snapshot_is_one_realtime_physics_barrier(monkeypatch):
    clock = FakeClock()
    runtime = SimRuntime("reach", realtime=True, seed=2, clock=clock)
    runtime.on_episode_start()
    entered = threading.Event()
    release = threading.Event()
    original = runtime.task.env_state

    def blocked_environment(world):
        entered.set()
        assert release.wait(5), "test did not release the runtime snapshot"
        return original(world)

    monkeypatch.setattr(runtime.task, "env_state", blocked_environment)
    result: list[tuple[np.ndarray, dict | None, Exception | None]] = []
    runtime.start()
    thread = threading.Thread(target=lambda: result.append(runtime.recording_snapshot()))
    thread.start()
    try:
        assert entered.wait(5), "recording snapshot did not acquire the runtime lock"
        frozen_time = runtime.world.time_s
        clock.t += 0.5
        time.sleep(0.03)
        assert runtime.world.time_s == frozen_time  # the physics thread waits on the barrier
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        environment, grade, error = result[0]
        assert environment.shape == (3,) and grade is not None and error is None
        assert grade["metrics"]["sim_time_s"] == round(frozen_time, 4)
        assert wait_for(lambda: runtime.world.time_s > frozen_time)
    finally:
        release.set()
        runtime.stop()


def test_module_hooks_use_the_singleton_built_from_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SO101_SIM_TASK", "reach")
    monkeypatch.setenv("SO101_SIM_SEED", "7")
    monkeypatch.setenv("SO101_SIM_REALTIME", "false")
    config_module.get_settings.cache_clear()
    stop_sim_runtime()  # no runtime yet: does not build one
    assert devices_module._runtime is None
    runtime = get_sim_runtime()
    assert get_sim_runtime() is runtime
    assert (runtime.task.name, runtime.base_seed, runtime.realtime) == ("reach", 7, False)
    assert on_episode_start()["seed"] == 7
    assert on_episode_start()["seed"] == 8
    assert on_episode_end()["metrics"]["seed"] == 8

    bus = make_sim_bus("follower")
    bus.connect()
    assert runtime.status()["devices"] == {"follower": 1}
    bus.disconnect()
    reset_sim_runtime()
    assert get_sim_runtime() is not runtime


# -- faults -----------------------------------------------------------------------------------
def test_physics_divergence_latches_a_fault(
    runtime: SimRuntime, bus: MujocoBus, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)  # MuJoCo appends its warning to MUJOCO_LOG.TXT in the cwd
    enable(bus)
    runtime.world.data.qvel[runtime.world.index.dof[1]] = math.nan
    with pytest.raises(RuntimeError, match="diverged"):
        bus.write_positions({"gripper": 20.0})
    assert "bad qvel" in runtime.status()["fault"]
    with pytest.raises(RuntimeError, match="simulation fault"):
        bus.read_positions()
    with pytest.raises(RuntimeError, match="simulation fault"):
        bus.write_positions({"gripper": 20.0})
    with pytest.raises(RuntimeError, match="simulation fault"):
        runtime.observation()
    with pytest.raises(RuntimeError, match="simulation fault"):
        runtime.on_episode_start()
    bus.set_torque(False)  # de-energizing always works
    assert not runtime.torque_enabled
    with pytest.raises(RuntimeError, match="simulation fault"):
        bus.set_torque(True)
    outcome = runtime.on_episode_end()
    assert outcome["success"] is False and "diverged" in outcome["metrics"]["fault"]
    json.dumps(outcome, allow_nan=False)
    with pytest.raises(RuntimeError, match="simulation fault"):
        MujocoBus(runtime=runtime).connect()


def test_a_fault_in_the_realtime_thread_stops_it(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    clock = FakeClock()
    runtime = SimRuntime("reach", realtime=True, seed=0, clock=clock)
    runtime.start()
    try:
        with runtime._lock:
            runtime.world.data.qvel[runtime.world.index.dof[0]] = math.inf
        clock.t += 0.1
        assert wait_for(lambda: not runtime.running)
        assert "diverged" in (runtime.fault or "")
        with pytest.raises(RuntimeError, match="simulation fault"):
            runtime.start()
    finally:
        runtime.stop()


# -- cameras ------------------------------------------------------------------------------------
def test_sim_cameras_render_the_shared_world(runtime: SimRuntime, bus: MujocoBus) -> None:
    if not shared_renderer().available:
        pytest.skip(f"no OpenGL context here: {shared_renderer().error}")
    width, height = config_module.get_settings().sim_image_wh
    camera = make_sim_camera("front", runtime=runtime)
    assert isinstance(camera, SimCamera)
    assert (camera.cam_id, camera.width, camera.height) == ("front", width, height)
    frame = camera.read()
    assert frame.shape == (height, width, 3) and frame.dtype == np.uint8 and frame.std() > 0

    enable(bus)
    for _ in range(30):
        bus.write_positions({"shoulder_pan": 45.0})
    assert not np.array_equal(camera.read(), frame)  # the arm moved in the shared world

    wrist = make_sim_camera("wrist_cam", runtime=runtime, width=64, height=48)
    assert wrist.read().shape == (48, 64, 3)
    assert runtime.status()["devices"] == {"follower": 1, "camera": 2}
    camera.release()
    camera.release()
    with pytest.raises(RuntimeError, match="released"):
        camera.read()
    wrist.release()
    assert runtime.status()["devices"] == {"follower": 1}


def test_camera_manager_reads_sim_cameras(
    runtime: SimRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not shared_renderer().available:
        pytest.skip(f"no OpenGL context here: {shared_renderer().error}")

    def open_sim(self: CameraManager, cid: str) -> SimCamera:
        return make_sim_camera(cid, runtime=runtime, width=32, height=24)

    monkeypatch.setattr(CameraManager, "_open_one", open_sim)  # the wave-2 hook, by hand
    cameras = CameraManager(cam_ids=["front", "wrist_cam"])
    frames = cameras.read()
    assert {cid: frame.shape for cid, frame in frames.items()} == {
        "front": (24, 32, 3),
        "wrist_cam": (24, 32, 3),
    }
    cameras.close()
    assert runtime.status()["devices"] == {}


def test_sim_cameras_fail_like_real_ones(runtime: SimRuntime) -> None:
    with pytest.raises(CameraUnavailableError, match="unknown sim camera"):
        make_sim_camera("0", runtime=runtime)  # a real camera id is not a sim camera
    no_gl = SimRuntime("reach", realtime=False, renderer=FrameRenderer(enabled=False))
    with pytest.raises(CameraUnavailableError, match="rendering disabled"):
        make_sim_camera("front", runtime=no_gl)
    with pytest.raises(CameraUnavailableError, match="outside"):
        make_sim_camera("front", runtime=runtime, width=4, height=4)
    assert runtime.status()["devices"] == {} and no_gl.status()["devices"] == {}


def test_a_hung_gl_probe_holds_a_sim_camera_no_longer_than_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    monkeypatch.setattr(FrameRenderer, "_probe", staticmethod(lambda: gate.wait(30)))
    monkeypatch.setattr(devices_module, "FRAME_TIMEOUT_S", 0.3)
    hung = FrameRenderer()
    stuck = SimRuntime("reach", realtime=False, renderer=hung)
    started = time.perf_counter()
    try:
        with pytest.raises(CameraUnavailableError, match="did not start within 0.3 s"):
            make_sim_camera("front", runtime=stuck)
        assert time.perf_counter() - started < 0.3 + 0.5
    finally:
        gate.set()
        hung.close()

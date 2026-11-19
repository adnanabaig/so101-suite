"""Scripted expert, IK, the closed-loop runner and the evaluation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101 import JOINT_NAMES  # noqa: E402
from so101.simulation.expert import (  # noqa: E402
    DlsIk,
    ExpertConfig,
    ScriptedExpert,
    copy_model,
    grasp_rotation,
    wrap_quarter_turn,
)
from so101.simulation.policy import (  # noqa: E402
    HoldPolicy,
    UnsupportedPolicyError,
    evaluate,
    make_policy,
    register_policy_loader,
    wilson_interval,
)
from so101.simulation.rollout import EpisodeRunner, action_array  # noqa: E402
from so101.simulation.tasks import make_task  # noqa: E402


def test_quarter_turn_wrap_and_grasp_frame():
    assert wrap_quarter_turn(math.radians(50)) == pytest.approx(math.radians(-40))
    assert wrap_quarter_turn(math.radians(-44)) == pytest.approx(math.radians(-44))
    frame = grasp_rotation(0.3)
    assert frame @ frame.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    assert frame[:, 0] == pytest.approx([0.0, 0.0, -1.0])  # approach straight down


def test_ik_recovers_a_reachable_pose():
    task = make_task("pick_lift")
    ik = DlsIk(copy_model(task))
    pose = np.array([15.0, -10.0, 20.0, 70.0, -20.0, 30.0])
    position, rotation = ik.forward(pose)
    seed = np.array([0.0, -40.0, 30.0, 60.0, 0.0, 30.0])
    result = ik.solve(seed, position, rotation, max_iterations=300)
    assert result.position_error_m < 5e-4 and result.rotation_error_rad < 5e-3
    reached, _ = ik.forward(result.joints)
    assert np.linalg.norm(reached - position) < 5e-4
    assert result.joints[5] == seed[5]  # the gripper is never touched


@pytest.mark.parametrize(("name", "seeds"), [("reach", (0, 1)), ("pick_lift", (0, 1))])
def test_expert_succeeds(name, seeds):
    report = evaluate(name, "expert", episodes=len(seeds), seed=seeds[0])
    assert report.successes == len(seeds), report.outcomes
    assert report.physics_steps > 0 and report.real_time_factor > 1.0


def test_expert_places_the_cube():
    task = make_task("pick_place")
    expert = ScriptedExpert(task)
    result = EpisodeRunner(task).run(expert, 4)
    assert result.success, (result.summary, expert.history)
    assert expert.history[:5] == ["approach", "pre_grasp", "descend", "close", "lift"]
    assert expert.history[5:8] == ["transport", "place", "release"]
    assert result.final_metrics["in_zone"] == 1.0


def test_hold_policy_never_picks_anything_up():
    report = evaluate("pick_lift", "hold", episodes=1, seed=0)
    assert report.successes == 0
    assert report.outcomes[0]["summary"]["max_lift_m"] < 0.005


def test_noisy_expert_is_deterministic_per_seed():
    task = make_task("pick_lift")
    runner = EpisodeRunner(task)
    expert = ScriptedExpert(task, ExpertConfig(noise_deg=1.0, waypoint_noise_m=0.003))

    def actions(seed: int) -> np.ndarray:
        trace: list[np.ndarray] = []
        runner.run(
            expert,
            seed,
            on_step=lambda record: trace.append(record.action),
            policy_reset=lambda: expert.reset(seed),
            max_steps=60,
        )
        return np.array(trace)

    first = actions(5)
    assert np.array_equal(first, actions(5))
    assert not np.array_equal(first, actions(6))


def test_runner_sends_every_action_through_the_safety_guard():
    task = make_task("reach")
    runner = EpisodeRunner(task)

    class Jump:
        def reset(self) -> None:
            pass

        def select_action(self, observation):
            target = np.array(observation["observation.state"], dtype=float)
            target[0] += 90.0
            return target

    records = []
    runner.run(Jump(), 0, on_step=records.append, max_steps=3, stop_on_success=False)
    first = records[0]
    step = abs(float(first.command[0] - first.observation["observation.state"][0]))
    assert step <= 450.0 / 30.0 + 0.5  # the guard's 450 deg/s at 30 Hz
    assert abs(float(first.action[0] - first.command[0])) > 70.0


def test_action_array_accepts_dicts_and_rejects_bad_shapes():
    values = {joint: float(index) for index, joint in enumerate(JOINT_NAMES)}
    assert action_array(values).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="missing"):
        action_array({"shoulder_pan": 0.0})
    with pytest.raises(ValueError, match="shape"):
        action_array(np.zeros(5))
    with pytest.raises(ValueError, match="non-finite"):
        action_array([0, 0, 0, 0, 0, math.nan])


def test_policy_specs_and_the_loader_registry(tmp_path):
    task = make_task("reach")
    assert isinstance(make_policy("expert", task), ScriptedExpert)
    assert isinstance(make_policy("hold", task), HoldPolicy)
    with pytest.raises(UnsupportedPolicyError, match="unsupported model path"):
        make_policy(str(tmp_path / "missing"), task)
    checkpoint = tmp_path / "model.npz"
    checkpoint.write_bytes(b"")
    with pytest.raises(UnsupportedPolicyError, match="no registered policy loader"):
        make_policy(str(checkpoint), task)
    register_policy_loader(lambda path: path.suffix == ".npz", lambda path: HoldPolicy())
    try:
        assert isinstance(make_policy(str(checkpoint), task), HoldPolicy)
    finally:
        from so101.simulation import policy as module

        module._LOADERS.pop()


def test_wilson_interval():
    low, high = wilson_interval(50, 50)
    assert high == 1.0 and low == pytest.approx(0.9287, abs=1e-4)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)

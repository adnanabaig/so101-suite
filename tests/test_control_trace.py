"""Control tick capture is passive, bounded and independent of dashboard cadence."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from so101.services.control_trace import ControlTrace
from so101.services.robot_manager import get_manager
from so101.teleop.controller import TeleopStep


def test_trace_preserves_all_commands_and_reports_overrun_without_mutable_aliases():
    trace = ControlTrace(capacity=2)
    step = TeleopStep(1.5, {"wrist_roll.pos": 1}, {"wrist_roll.pos": 2}, {"wrist_roll.pos": 5})
    for _ in range(3):
        trace.append(step, backend="real", started=10, completed=10.01, fps=30)
    step.action["wrist_roll.pos"] = 99
    batch = trace.snapshot(0)
    assert batch["missing_before_buffer"] == 1
    assert [tick["seq"] for tick in batch["ticks"]] == [2, 3]
    assert batch["ticks"][0]["timestamp_ns"] == 1500000000
    assert batch["ticks"][0]["action"] == {"wrist_roll.pos": 2}
    assert batch["ticks"][0]["desired_action"] == {"wrist_roll.pos": 5}
    batch["ticks"][0]["observation"]["wrist_roll.pos"] = 999
    assert trace.snapshot(1)["ticks"][0]["observation"] == {"wrist_roll.pos": 1}
    assert trace.snapshot(3)["ticks"] == []
    assert trace.snapshot(0)["stream_id"] != ControlTrace().stream_id


def test_recorded_command_profile_has_no_mutable_configuration_alias():
    trace = ControlTrace()
    profile = {"max_velocity": {"shoulder_lift": 60}, "driver_present_position_window": None}
    trace.append(
        TeleopStep(1, {}, {}, {}),
        backend="real",
        started=0,
        completed=0.01,
        fps=30,
        profile=profile,
    )
    profile["max_velocity"]["shoulder_lift"] = 999
    assert trace.snapshot()["ticks"][0]["control_profile"]["max_velocity"]["shoulder_lift"] == 60


def test_timing_preserves_previous_cycle_provenance_without_aliases():
    trace = ControlTrace()
    timing = {"previous_pacing": {"deadline_lateness_seconds": 0.1}, "motor_step_seconds": 0.003}
    trace.append(
        TeleopStep(1, {}, {}, {}),
        backend="real",
        started=0,
        completed=0.01,
        fps=30,
        timing=timing,
    )
    timing["previous_pacing"]["deadline_lateness_seconds"] = 99
    saved = trace.snapshot()["ticks"][0]["timing"]
    assert saved["previous_pacing"]["deadline_lateness_seconds"] == 0.1
    assert saved["motor_step_seconds"] == 0.003


def test_shared_real_teleop_never_opens_direct_camera_devices(monkeypatch):
    from so101.services import robot_manager as module

    manager = get_manager()
    manager.settings.camera_capture_mode = "shared"
    leader = SimpleNamespace(connected=False, bus=object(), disconnect=lambda: None)
    follower = SimpleNamespace(
        leader_teleop_profile=nullcontext,
        command_profile=lambda: {"kind": "lerobot_leader_mirror"},
    )

    def step():
        manager._stop.set()
        return TeleopStep(
            1, {"shoulder_pan.pos": 1}, {"shoulder_pan.pos": 2}, {"shoulder_pan.pos": 3}
        )

    monkeypatch.setattr(module, "current_backend", lambda settings: "real")
    monkeypatch.setattr(module, "bus_torque_state", lambda bus: "off")
    monkeypatch.setattr(
        module,
        "TeleopController",
        lambda *args, **kw: SimpleNamespace(
            leader=leader,
            follower=follower,
            connect=lambda: None,
            step=step,
        ),
    )
    monkeypatch.setattr(manager, "leader", lambda: leader)
    monkeypatch.setattr(manager, "follower", lambda: follower)
    monkeypatch.setattr(manager, "cameras", lambda: pytest.fail("Direct camera opened"))
    monkeypatch.setattr(manager, "_prepare_global_energized_intent", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_retire_confirmed_recovery_devices", lambda *a: None)
    monkeypatch.setattr(manager, "_sleep", lambda seconds: None)
    manager._control_loop(30, record=False)
    ticks = manager.control_trace.snapshot()["ticks"]
    assert len(ticks) == 1 and ticks[0]["desired_action"]["shoulder_pan.pos"] == 3
    assert ticks[0]["timing"]["cycle_interval_seconds"] is None
    assert ticks[0]["timing"]["previous_pacing"] is None
    assert ticks[0]["timing"]["io_lock_wait_seconds"] >= 0
    assert ticks[0]["timing"]["motor_step_seconds"] >= 0
    with pytest.raises(RuntimeError, match="shared relays"):
        manager.start_recording(SimpleNamespace(fps=30), operator_id="operator")
    with pytest.raises(RuntimeError, match="shared relays"):
        manager.start_camera_preview()

"""Manual arm CLI orchestrates independent joint checks without enabling motion."""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101 import JOINT_NAMES
from so101.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "manual_arm_calibration", ROOT / "scripts/calibrate_arm_manual.py"
)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def args(tmp_path, **changes):
    values = {
        "arm": "follower",
        "joint": None,
        "validations": 3,
        "max_seconds": 180,
        "output": tmp_path / "arm-session",
        "resume": None,
        "dry_run": False,
    }
    return SimpleNamespace(**(values | changes))


def settings():
    return get_settings().model_copy(update={"backend": "real", "mock": False})


def worker(calls, fail=None):
    def calibrate(options, config):
        calls.append(options.joint)
        assert options.fresh_reference and options.baseline is None
        assert options.validations == 3
        if options.joint == fail:
            raise RuntimeError("injected joint failure; prior registers restored")
        folder = options.output
        folder.mkdir()
        recipe = {
            "arm": options.arm,
            "joint": options.joint,
            "backend": "lerobot",
            "device_id": getattr(config, f"{options.arm}_id"),
            "port": getattr(config, f"{options.arm}_port"),
            "joints": {options.joint: {}},
            "diagnostics": {
                "validated": True,
                "restored": True,
                "calibration_applied": False,
                "validation_sweeps": options.validations,
            },
        }
        (folder / "validated_calibration.json").write_text(json.dumps(recipe))
        return folder

    return calibrate


def test_all_six_run_in_gripper_to_base_order_and_bundle_only_after_completion(tmp_path):
    options, config, calls = args(tmp_path), settings(), []
    folder = cli.run_session(options, config, calibrate_fn=worker(calls))
    assert calls == list(reversed(JOINT_NAMES))
    manifest = json.loads((folder / "arm_session.json").read_text())
    bundle = json.loads((folder / "validated_joint_bundle.json").read_text())
    assert manifest["state"] == "validated"
    assert set(bundle["joint_recipes"]) == set(JOINT_NAMES)
    assert bundle["all_six_joints_validated"] is True
    assert not bundle["calibration_applied"] and not bundle["powered_motion_ready"]
    assert not (config.data_dir / "calibration_follower.json").exists()


def test_failure_stops_remaining_joints_and_resume_reuses_only_verified_successes(tmp_path):
    options, config, calls = args(tmp_path), settings(), []
    with pytest.raises(RuntimeError, match="injected joint failure"):
        cli.run_session(options, config, calibrate_fn=worker(calls, fail="wrist_flex"))
    assert calls == ["gripper", "wrist_roll", "wrist_flex"]
    manifest = json.loads((options.output / "arm_session.json").read_text())
    assert manifest["state"] == "partial"
    assert not (options.output / "validated_joint_bundle.json").exists()
    resume = args(tmp_path, output=None, resume=options.output)
    second_calls = []
    folder = cli.run_session(resume, config, calibrate_fn=worker(second_calls))
    assert second_calls == ["wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
    final = json.loads((folder / "arm_session.json").read_text())
    assert final["state"] == "validated" and "error" not in final
    assert len(final["attempts"]) == 7
    assert final["attempts"][2]["state"] == "failed"


def test_resuming_completed_session_preserves_bundle_and_manifest_bytes(tmp_path):
    options, config, calls = args(tmp_path), settings(), []
    folder = cli.run_session(options, config, calibrate_fn=worker(calls))
    before = {
        name: (folder / name).read_bytes()
        for name in (
            "arm_session.json",
            "validated_joint_bundle.json",
        )
    }
    resume = args(tmp_path, output=None, resume=folder)
    cli.run_session(resume, config, calibrate_fn=lambda *args: pytest.fail("Must reuse recipes"))
    assert {name: (folder / name).read_bytes() for name in before} == before


def test_resuming_completed_session_rejects_changed_bundle_contents(tmp_path):
    options, config, calls = args(tmp_path), settings(), []
    folder = cli.run_session(options, config, calibrate_fn=worker(calls))
    path = folder / "validated_joint_bundle.json"
    bundle = json.loads(path.read_text())
    bundle["method"] = "changed"
    path.write_text(json.dumps(bundle))
    with pytest.raises(RuntimeError, match="bundle differs"):
        cli.run_session(args(tmp_path, output=None, resume=folder), config)


@pytest.mark.parametrize("change", ["device", "port", "validation_count", "recipe"])
def test_resume_rejects_changed_profile_or_evidence_before_hardware(tmp_path, change):
    config = settings()
    options = args(tmp_path, joint=["gripper"])
    cli.run_session(options, config, calibrate_fn=worker([]))
    resume = args(tmp_path, output=None, resume=options.output, joint=["gripper"])
    if change == "device":
        config = config.model_copy(update={"follower_id": "other"})
    elif change == "port":
        config = config.model_copy(update={"follower_port": "/dev/another"})
    elif change == "validation_count":
        resume.validations = 5
    else:
        manifest = json.loads((options.output / "arm_session.json").read_text())
        recipe = options.output / manifest["results"]["gripper"]["recipe"]
        recipe.write_text("{}")
    with pytest.raises((RuntimeError, ValueError)):
        cli.run_session(
            resume, config, calibrate_fn=lambda *a: pytest.fail("invalid resume opened hardware")
        )


def test_single_joint_and_dry_run_never_imply_complete_arm(tmp_path):
    options = args(tmp_path, joint=["wrist_flex"])
    folder = cli.run_session(options, settings(), calibrate_fn=worker([]))
    bundle = json.loads((folder / "validated_joint_bundle.json").read_text())
    assert not bundle["all_six_joints_validated"]
    assert set(bundle["joint_recipes"]) == {"wrist_flex"}
    dry = args(tmp_path, output=tmp_path / "untouched", dry_run=True)
    cli.run_session(dry, settings(), calibrate_fn=lambda *a: pytest.fail("dry run opened hardware"))
    assert not dry.output.exists()


def test_cancelled_joint_is_retained_as_partial_session(tmp_path):
    options = args(tmp_path)

    def cancelled(*a):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.run_session(options, settings(), calibrate_fn=cancelled)
    manifest = json.loads((options.output / "arm_session.json").read_text())
    assert manifest["state"] == "partial"
    assert manifest["results"]["gripper"]["state"] == "failed"


class Native:
    def __init__(self):
        self.motors = {
            name: SimpleNamespace(id=index + 1) for index, name in enumerate(JOINT_NAMES)
        }
        self.regs = {
            name: {
                "Homing_Offset": 0,
                "Operating_Mode": 0,
                "Min_Position_Limit": 100,
                "Max_Position_Limit": 4000,
                "Lock": 1,
            }
            for name in JOINT_NAMES
        }
        self.regs["wrist_roll"].update(Min_Position_Limit=0, Max_Position_Limit=4095)
        self.original = {name: dict(values) for name, values in self.regs.items()}
        self.writes = []

    def raw_center(self, joint):
        return 1400 + self.motors[joint].id * 100

    def read(self, register, joint, **kwargs):
        if register == "Present_Position":
            return self.raw_center(joint) - self.regs[joint]["Homing_Offset"]
        return self.regs[joint][register]

    def sync_read(self, register, names, **kwargs):
        if register == "Torque_Enable":
            return dict.fromkeys(JOINT_NAMES, 0)
        assert register == "Present_Position" and len(names) == 1
        return {names[0]: self.read(register, names[0])}

    def write(self, register, joint, value, **kwargs):
        assert register in self.regs[joint]  # no goal or torque-on register writes
        self.writes.append((joint, register, value))
        self.regs[joint][register] = value


def fake_capture(native, folder, number, stop_key, *, joint="wrist_roll", **kwargs):
    low, high = native.raw_center(joint) - 700, native.raw_center(joint) + 700
    offset = native.regs[joint]["Homing_Offset"]
    values = [low] * 12 + [high] * 12
    events = [
        {"type": "sample", "raw_ticks": value, "elapsed_seconds": index * 0.1}
        for index, value in enumerate(values)
    ]
    filename = f"trial_{number:02d}.jsonl"
    (folder / filename).write_text("\n".join(map(json.dumps, events)) + "\n")
    return {
        "trial": number,
        "samples_file": filename,
        "usable_for_comparison": True,
        "status": "complete",
        "quality_warnings": [],
        "raw_min": low,
        "raw_max": high,
        "reported_min": low - offset,
        "reported_max": high - offset,
        "registers_before": dict(native.regs[joint]),
    }


@pytest.mark.parametrize("joint", JOINT_NAMES)
def test_shared_worker_homes_only_selected_joint_and_preserves_other_joints(
    tmp_path, monkeypatch, joint
):
    worker_module = cli.manual
    native = Native()

    class Arm:
        _bus = native
        closed = False

        def connect(self):
            pass

        def set_torque(self, enabled):
            assert enabled is False

        def disconnect(self):
            self.closed = True

    arm = Arm()

    class Access:
        def disconnect_device(self, device, **kwargs):
            device.disconnect()

    @contextmanager
    def gate(*a, **k):
        yield Access()

    monkeypatch.setattr(worker_module, "standalone_access", gate)
    config = settings()
    options = SimpleNamespace(
        arm="follower",
        joint=joint,
        baseline=None,
        fresh_reference=True,
        validations=3,
        max_seconds=180,
        output=tmp_path / joint,
        dry_run=False,
    )
    answers = iter(["ready", "", "test", "", "", ""])
    folder = worker_module.workflow(
        options,
        config,
        ask=lambda _: next(answers),
        bus_factory=lambda *a: arm,
        capture_fn=fake_capture,
        position_fn=lambda *a, **k: {"raw": native.raw_center(joint)},
    )
    assert arm.closed and native.regs == native.original
    assert all(name == joint for name, _, _ in native.writes)
    recipe = json.loads((folder / "validated_calibration.json").read_text())
    assert set(recipe["joints"]) == {joint}
    assert recipe["diagnostics"]["motor_ids"] == {joint: JOINT_NAMES.index(joint) + 1}
    if joint == "gripper":
        assert recipe["joints"][joint] == {"min": 0, "max": 100, "offset": 50}


def test_generic_capture_reads_selected_joint_only(tmp_path):
    native = Native()
    clock = SimpleNamespace(value=0.0)

    def now():
        return clock.value

    def sleep(seconds):
        clock.value += seconds

    keys = iter([None, None, "complete"])
    capture = cli.manual.capture
    result = capture.capture_trial(
        native,
        tmp_path,
        1,
        lambda: next(keys),
        joint="gripper",
        clock=now,
        sleep=sleep,
    )
    assert result["joint"] == "gripper"
    assert result["registers_before"]["Min_Position_Limit"] == 100
    events = list(map(json.loads, (tmp_path / "trial_01.jsonl").read_text().splitlines()))
    values = [event["raw_ticks"] for event in events if event["type"] == "sample"]
    assert values == [native.raw_center("gripper")] * 2

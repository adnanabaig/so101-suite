"""Endpoint-first CLI validates raw travel, restores registers and isolates joint recipes."""

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
    "manual_wrist_calibration", ROOT / "scripts/calibrate_wrist_roll.py"
)
assert spec and spec.loader
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class Native:
    def __init__(self):
        self.motors = {
            name: SimpleNamespace(id=index + 1) for index, name in enumerate(JOINT_NAMES)
        }
        self.reg = {
            "Homing_Offset": 0,
            "Operating_Mode": 0,
            "Min_Position_Limit": 0,
            "Max_Position_Limit": 4095,
            "Lock": 1,
        }
        self.original = dict(self.reg)
        self.writes = []
        self.ignore_offset = False
        self.restore_fails = False

    def read(self, register, joint, **kwargs):
        assert joint == "wrist_roll"
        if register == "Present_Position":
            return 2611 - self.reg["Homing_Offset"]
        return self.reg[register]

    def sync_read(self, register, names, **kwargs):
        if register == "Torque_Enable":
            return dict.fromkeys(JOINT_NAMES, 0)
        assert register == "Present_Position" and names == ["wrist_roll"]
        return {"wrist_roll": 2611 - self.reg["Homing_Offset"]}

    def write(self, register, joint, value, **kwargs):
        assert joint == "wrist_roll"  # never writes another joint
        assert register in self.original  # never Goal_Position or torque on
        self.writes.append((register, value))
        if register == "Homing_Offset":
            if self.ignore_offset or (self.restore_fails and value == 0):
                return
        self.reg[register] = value


class Arm:
    def __init__(self, native):
        self._bus = native
        self.torque_state = "off"
        self.closed = False

    def connect(self):
        pass

    def set_torque(self, enabled):
        assert enabled is False
        self.torque_state = "off"

    def disconnect(self):
        self.closed = True


@pytest.fixture
def rig(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    session = {
        "backend": "real",
        "synthetic": False,
        "state": "complete",
        "completed_trials": 10,
        "requested_trials": 10,
        "joint": "wrist_roll",
        "expected_motor_id": 5,
        "arm": "follower",
        "device_id": settings.follower_id,
        "port": settings.follower_port,
        "session_id": "ten-recorded-trials",
    }
    (baseline / "session.json").write_text(json.dumps(session))
    for number in range(1, 11):
        summary = {
            "usable_for_comparison": True,
            "sample_count": 2,
            "raw_min": 666,
            "raw_max": 4556,
            "registers_before": {"Homing_Offset": 0},
            "registers_after": {"Homing_Offset": 0},
        }
        events = [
            {
                "type": "sample",
                "raw_ticks": raw,
                "reported_ticks": raw,
                "homing_offset": 0,
                "elapsed_seconds": index,
            }
            for index, raw in enumerate((666, 4556))
        ]
        events.append({"type": "trial_end"})
        (baseline / f"trial_{number:02d}.json").write_text(json.dumps(summary))
        (baseline / f"trial_{number:02d}.jsonl").write_text(
            "\n".join(map(json.dumps, events)) + "\n"
        )
    native = Native()
    arm = Arm(native)
    state = {"locked": False, "opened": False, "close_error": None}

    class Access:
        def disconnect_device(self, bus, *, device_name):
            assert state["locked"] and device_name == "follower"
            bus.disconnect()
            if state["close_error"]:
                raise RuntimeError(state["close_error"])

    @contextmanager
    def gate(*a, **k):
        state["locked"] = True
        try:
            yield Access()
        finally:
            state["locked"] = False

    monkeypatch.setattr(cal, "standalone_access", gate)

    def factory(*a):
        assert state["locked"]
        state["opened"] = True
        return arm

    args = SimpleNamespace(
        baseline=baseline,
        arm="follower",
        validations=5,
        max_seconds=180,
        output=tmp_path / "workflow",
        dry_run=False,
    )
    return SimpleNamespace(
        args=args,
        settings=settings,
        native=native,
        arm=arm,
        state=state,
        factory=factory,
    )


def capture_good(native, folder, number, stop_key, **kwargs):
    kwargs["ready"]()
    offset = native.reg["Homing_Offset"]
    raw = [666] * 12 + [2611] + [4556] * 12
    events = [
        {
            "type": "sample",
            "raw_ticks": value,
            "reported_ticks": value - offset,
            "homing_offset": offset,
            "elapsed_seconds": index * 0.1,
        }
        for index, value in enumerate(raw)
    ]
    name = f"trial_{number:02d}.jsonl"
    (folder / name).write_text("\n".join(map(json.dumps, events)) + "\n")
    result = {
        "trial": number,
        "samples_file": name,
        "usable_for_comparison": True,
        "status": "complete",
        "quality_warnings": [],
        "raw_min": 666,
        "raw_max": 4556,
        "reported_min": 666 - offset,
        "reported_max": 4556 - offset,
        "registers_before": dict(native.reg),
        "registers_after": dict(native.reg),
    }
    (folder / f"trial_{number:02d}.json").write_text(json.dumps(result))
    return result


def run(rig, *, answers=None, capture_fn=capture_good):
    replies = iter(answers or ["ready", "", "test"] + [""] * 5)
    return cal.workflow(
        rig.args,
        rig.settings,
        ask=lambda _: next(replies),
        bus_factory=rig.factory,
        capture_fn=capture_fn,
        position_fn=lambda native, target: {"raw": 2611, "target_raw": target},
    )


def test_success_restores_motor_and_saves_only_separate_validated_joint_recipe(rig):
    full = rig.settings.data_dir / "calibration_follower.json"
    full.write_text('{"existing": "full-arm file"}')
    latest = rig.settings.data_dir / "calibration_joint_follower_wrist_roll.json"
    latest.write_text('{"previous": "baseline"}')
    folder = run(rig)
    assert rig.arm.closed and not rig.state["locked"]
    assert rig.native.reg == rig.native.original
    assert rig.native.writes == [
        ("Lock", 0),
        ("Homing_Offset", 564),
        ("Lock", 0),
        ("Homing_Offset", 0),
        ("Lock", 1),
    ]
    assert full.read_text() == '{"existing": "full-arm file"}'
    result = json.loads(latest.read_text())
    assert set(result["joints"]) == {"wrist_roll"}
    assert result["diagnostics"]["validated"] and result["diagnostics"]["restored"]
    assert result["diagnostics"]["homing_offsets_requested"] == {"wrist_roll": 564}
    assert result["diagnostics"]["range_mins"] == {"wrist_roll": 102}
    assert result["diagnostics"]["range_maxes"] == {"wrist_roll": 3992}
    assert result["diagnostics"]["calibration_applied"] is False
    assert json.loads((folder / "previous_joint_baseline.json").read_text()) == {
        "previous": "baseline"
    }
    session = json.loads((folder / "session.json").read_text())
    assert session["state"] == "validated" and len(session["trials"]) == 6
    assert session["original_registers_restored"]


def test_dry_run_never_takes_lock_or_opens_bus(rig, monkeypatch):
    rig.args.dry_run = True
    monkeypatch.setattr(cal, "standalone_access", lambda *a, **k: pytest.fail("dry-run took lock"))
    run(rig)
    assert not rig.state["opened"] and not rig.args.output.exists()


def test_reference_coverage_requires_a_settled_endpoint_not_a_transient(tmp_path):
    values = [1000] + [1050] * 50 + [2500] * 50
    events = [
        {"type": "sample", "raw_ticks": value, "elapsed_seconds": index * 0.02}
        for index, value in enumerate(values)
    ]
    (tmp_path / "trial.jsonl").write_text("\n".join(map(json.dumps, events)))
    trial = {
        "trial": 1,
        "usable_for_comparison": True,
        "samples_file": "trial.jsonl",
        "raw_min": 1000,
        "raw_max": 2500,
    }
    # A transient reaches the old reference, but the stable pause misses it.
    with pytest.raises(RuntimeError, match="covers_reference_low.*False"):
        cal.check_trial(
            tmp_path, trial, {"raw_min": 970, "raw_max": 2500}, offset=None, dwell_seconds=0.5
        )
    checks = cal.check_trial(
        tmp_path, trial, {"raw_min": 1000, "raw_max": 2500}, offset=None, dwell_seconds=0.5
    )
    assert checks["low_settled_ticks"] == 1050
    assert checks["low_dwell_seconds"] >= 0.5


@pytest.mark.parametrize(
    "failure", ["range", "dwell", "coverage", "cancel", "offset", "read_error"]
)
def test_failed_validation_restores_and_preserves_previous_recipe(rig, failure):
    latest = rig.settings.data_dir / "calibration_joint_follower_wrist_roll.json"
    latest.write_text('{"existing": true}')
    if failure == "offset":
        rig.native.ignore_offset = True

    def broken(native, folder, number, stop_key, **kwargs):
        result = capture_good(native, folder, number, stop_key, **kwargs)
        if number > 1:
            if failure == "read_error":
                raise ConnectionError("injected serial error")
            if failure == "range":
                result["reported_max"] = 4100
            elif failure == "coverage":
                result["raw_max"] = 4400
            elif failure == "cancel":
                result.update(status="interrupted", usable_for_comparison=False)
            elif failure == "dwell":
                path = folder / result["samples_file"]
                events = list(map(json.loads, path.read_text().splitlines()))
                for event in events:
                    event["elapsed_seconds"] *= 0.01
                path.write_text("\n".join(map(json.dumps, events)) + "\n")
        return result

    with pytest.raises(RuntimeError, match="Calibration not saved"):
        run(rig, capture_fn=broken)
    assert rig.arm.closed and rig.native.reg == rig.native.original
    assert latest.read_text() == '{"existing": true}'
    assert not (rig.args.output / "validated_calibration.json").exists()


@pytest.mark.parametrize("answer", ["q", "no", ""])
def test_declining_temporary_write_restores_without_homing_write(rig, answer):
    with pytest.raises(RuntimeError, match="not confirmed"):
        run(rig, answers=["ready", "", answer])
    assert rig.arm.closed and rig.native.reg == rig.native.original
    assert all(name != "Homing_Offset" for name, value in rig.native.writes)


def test_unconfirmed_setup_never_opens_bus(rig):
    with pytest.raises(RuntimeError, match="Setup not confirmed"):
        run(rig, answers=["no"])
    assert not rig.state["opened"]


def test_identity_mismatch_rejected_before_open(rig):
    rig.settings = rig.settings.model_copy(update={"follower_id": "different-arm"})
    with pytest.raises(ValueError, match="exact arm"):
        run(rig)
    assert not rig.state["opened"]


def test_corrupt_baseline_rejected_before_open(rig):
    path = rig.args.baseline / "trial_01.jsonl"
    path.write_text(path.read_text().replace('"raw_ticks": 666', '"raw_ticks": 999'))
    with pytest.raises(ValueError, match="integrity"):
        run(rig)
    assert not rig.state["opened"]


def test_restore_failure_is_latched_and_no_validated_recipe_is_published(rig, monkeypatch):
    saved = []
    monkeypatch.setattr(cal, "persist_standalone_fault", lambda *a, **k: saved.append((a, k)))
    rig.native.restore_fails = True
    with pytest.raises(RuntimeError, match="restore read-back"):
        run(rig)
    assert saved and rig.arm.closed
    assert not (rig.args.output / "validated_calibration.json").exists()
    session = json.loads((rig.args.output / "session.json").read_text())
    assert session["state"] == "failed" and "restore_error" in session


def test_disconnect_failure_prevents_recipe_publication(rig):
    rig.state["close_error"] = "torque-off unconfirmed"
    with pytest.raises(RuntimeError, match="torque-off unconfirmed"):
        run(rig)
    assert rig.native.reg == rig.native.original
    assert not (rig.args.output / "validated_calibration.json").exists()


def test_keyboard_interrupt_during_validation_still_restores(rig):
    def interrupted(native, folder, number, stop_key, **kwargs):
        if number > 1:
            raise KeyboardInterrupt
        return capture_good(native, folder, number, stop_key, **kwargs)

    with pytest.raises(RuntimeError, match="KeyboardInterrupt"):
        run(rig, capture_fn=interrupted)
    assert rig.native.reg == rig.native.original and rig.arm.closed


def test_nonstandard_existing_limits_are_preserved_and_refused(rig):
    rig.native.reg["Min_Position_Limit"] = 100
    rig.native.original = dict(rig.native.reg)
    with pytest.raises(RuntimeError, match="Expected wrist limits"):
        run(rig)
    assert rig.native.reg == rig.native.original and rig.arm.closed
    assert not (rig.args.output / "validated_calibration.json").exists()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("delay", [0, 0.1])
def test_probe_waits_for_three_direct_and_sync_matches_without_relaxing_tolerance(delay):
    clock = Clock()

    class Delayed(Native):
        def read(self, register, joint, **kwargs):
            if register == "Present_Position" and clock.now < delay:
                return 2611
            return super().read(register, joint, **kwargs)

        def sync_read(self, register, names, **kwargs):
            if register == "Present_Position" and clock.now < delay:
                return {"wrist_roll": 2611}
            return super().sync_read(register, names, **kwargs)

    native = Delayed()
    probes = []
    result = cal.write_offset(
        native,
        564,
        clock=clock,
        sleep=clock.sleep,
        record=lambda data: probes.append(json.loads(json.dumps(data))),
    )
    assert result["status"] == "verified"
    assert clock.now >= delay
    assert len(result["observations"]) >= 3
    assert result["expected_after"] == 2047
    assert all(
        entry["direct_position"] == entry["sync_position"] == 2047
        for entry in result["observations"][-3:]
    )
    if delay:
        assert result["observations"][0]["sync_position"] == 2611
    assert probes[-1]["status"] == "verified"
    assert native.writes == [("Lock", 0), ("Homing_Offset", 564)]


@pytest.mark.parametrize("behavior", ["never_shift", "direct_sync_disagree"])
def test_probe_never_accepts_unchanged_or_disagreeing_feedback(behavior):
    clock = Clock()

    class WrongFeedback(Native):
        def read(self, register, joint, **kwargs):
            if register == "Present_Position":
                return 2611
            return super().read(register, joint, **kwargs)

        def sync_read(self, register, names, **kwargs):
            if register == "Present_Position" and behavior == "never_shift":
                return {"wrist_roll": 2611}
            return super().sync_read(register, names, **kwargs)

    probes = []
    with pytest.raises(RuntimeError, match="feedback did not converge"):
        cal.write_offset(
            WrongFeedback(),
            564,
            clock=clock,
            sleep=clock.sleep,
            record=lambda data: probes.append(json.loads(json.dumps(data))),
        )
    assert 1 <= clock.now <= 1.03
    assert probes[-1]["status"] == "failed"
    assert len(probes[-1]["observations"]) > 3
    assert probes[-1]["read_back"] == 564
    assert probes[-1]["after"] == (2611 if behavior == "never_shift" else 2047)


def test_probe_failure_evidence_is_saved_even_when_readback_is_wrong():
    native = Native()
    native.ignore_offset = True
    evidence = []
    with pytest.raises(RuntimeError, match="homing read-back mismatch"):
        cal.write_offset(native, 564, record=lambda data: evidence.append(dict(data)))
    assert evidence[-1]["status"] == "failed"
    assert evidence[-1]["read_back"] == 0
    assert evidence[-1]["before"] == 2611


def test_midpoint_guide_requires_operator_enter_and_settled_reference():
    native = Native()
    clock = Clock()
    keys = iter([None, None, "complete"])
    result = cal.position_at_midpoint(
        native,
        2611,
        stop_key=lambda: next(keys),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result["raw"] == 2611
    assert result["recent_raw"] == [2611] * 3
    assert not native.writes


def test_midpoint_guide_does_not_accept_enter_outside_target_window():
    clock = Clock()
    native = Native()
    values = iter([4500, 2611, 2611, 2611])
    original = native.sync_read

    def positions(register, names, **kwargs):
        if register == "Present_Position":
            return {"wrist_roll": next(values)}
        return original(register, names, **kwargs)

    native.sync_read = positions
    keys = iter(["complete", None, None, "complete"])
    result = cal.position_at_midpoint(
        native,
        2611,
        stop_key=lambda: next(keys),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result["recent_raw"] == [2611] * 3
    assert not native.writes


def test_midpoint_abort_cannot_write_any_offset():
    native = Native()
    clock = Clock()
    with pytest.raises(RuntimeError, match="cancelled; no offset written"):
        cal.position_at_midpoint(
            native,
            2611,
            stop_key=lambda: "abort",
            clock=clock,
            sleep=clock.sleep,
        )
    assert not native.writes

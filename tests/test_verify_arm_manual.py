"""Endpoint verification preserves installed registers and resumes accepted sweeps."""

import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101 import JOINT_NAMES

spec = importlib.util.spec_from_file_location(
    "verify_arm", Path(__file__).resolve().parents[1] / "scripts/verify_arm_manual.py"
)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    source_file = tmp_path / "installed.json"
    source_file.write_text("unchanged")
    reference = {
        "plan": {
            "port": "fake-port",
            "native": {
                name: {
                    "id": index + 1,
                    "homing_offset": 100,
                    "range_min": 1000,
                    "range_max": 3000,
                }
                for index, name in enumerate(JOINT_NAMES)
            },
            "observed": {name: {"min": 1000, "max": 3000} for name in JOINT_NAMES},
        },
        "source_hashes": {str(source_file): cli.digest(source_file)},
        "seeds": {
            name: [{"raw_min": 1100, "raw_max": 3100, "sample_count": 80}] * 6
            for name in JOINT_NAMES
        },
    }
    monkeypatch.setattr(cli, "load_reference", lambda *args: reference)

    @contextmanager
    def access(*args, **kwargs):
        yield

    monkeypatch.setattr(cli, "standalone_access", access)

    class ReadOnlyBus:
        torque = 0
        offset = 100
        closed = False
        fail_cleanup = False

        def sync_read(self, register, joints, **kwargs):
            assert register == "Torque_Enable"
            return dict.fromkeys(joints, self.torque)

        def read(self, register, joint, **kwargs):
            return {
                "Homing_Offset": self.offset,
                "Operating_Mode": 0,
                "Min_Position_Limit": 1000,
                "Max_Position_Limit": 3000,
            }[register]

        def disconnect(self, *, disable_torque):
            assert disable_torque is False
            self.closed = True
            if self.fail_cleanup:
                raise OSError("cleanup failed")

    bus = ReadOnlyBus()
    calls = []

    def capture(bus, folder, number, stop, *, joint, **kwargs):
        calls.append(joint)
        samples = [
            {"type": "sample", "elapsed_seconds": index * 0.02, "raw_ticks": ticks}
            for index, ticks in enumerate([1100] * 40 + [3100] * 40)
        ]
        filename = f"trial_{number:02d}.jsonl"
        (folder / filename).write_text("".join(json.dumps(item) + "\n" for item in samples))
        return {
            "trial": number,
            "sample_count": len(samples),
            "samples_file": filename,
            "usable_for_comparison": True,
            "raw_min": 1100,
            "raw_max": 3100,
            "reported_min": 1000,
            "reported_max": 3000,
            "registers_before": {"Homing_Offset": 100},
        }

    args = SimpleNamespace(
        source=tmp_path / "source",
        arm="follower",
        joint=["gripper"],
        resume=None,
        dry_run=False,
    )
    settings = SimpleNamespace(data_dir=tmp_path, baudrate=1000000)
    return SimpleNamespace(
        args=args,
        settings=settings,
        bus=bus,
        capture=capture,
        calls=calls,
        opener=lambda *args, **kwargs: bus,
        ask=lambda prompt: "READY" if "READY" in prompt else "",
    )


def run(rig, **kwargs):
    return cli.collect(
        rig.args,
        rig.settings,
        ask=kwargs.pop("ask", rig.ask),
        opener=rig.opener,
        capture_fn=kwargs.pop("capture_fn", rig.capture),
        **kwargs,
    )


def test_four_fresh_sweeps_make_ten_without_motor_writes(rig):
    folder = run(rig)
    report = json.loads((folder / "verification.json").read_text())
    assert rig.calls == ["gripper"] * 4
    assert report["state"] == "ten_sweeps_verified"
    assert report["comparison"]["gripper"]["successful_sweeps"] == 10
    assert not report["calibration_applied"]
    assert not report["powered_motion_verified"]
    assert rig.bus.closed


def test_all_six_joints_each_collect_four_sweeps(rig):
    rig.args.joint = None
    folder = run(rig)
    report = json.loads((folder / "verification.json").read_text())
    assert len(rig.calls) == 24
    assert set(report["comparison"]) == set(JOINT_NAMES)
    assert all(item["successful_sweeps"] == 10 for item in report["comparison"].values())


def test_new_endpoint_outside_reference_is_retained_as_failure(rig):
    def drifted(*args, **kwargs):
        trial = rig.capture(*args, **kwargs)
        trial["raw_min"] = 1000
        trial["reported_min"] = 900
        path = args[1] / trial["samples_file"]
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        for sample in samples:
            if sample["raw_ticks"] == 1100:
                sample["raw_ticks"] = 1000
        path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
        return trial

    with pytest.raises(RuntimeError, match="drifted"):
        run(rig, capture_fn=drifted)
    path = next(rig.settings.data_dir.glob("manual_arm_verifications/follower/*/verification.json"))
    report = json.loads(path.read_text())
    assert report["state"] == "partial"
    assert report["attempts"]["gripper"][0]["state"] == "failed"


@pytest.mark.parametrize(("attribute", "value"), [("torque", 1), ("offset", 101)])
def test_live_reference_mismatch_refuses_capture(rig, attribute, value):
    setattr(rig.bus, attribute, value)
    with pytest.raises(RuntimeError):
        run(rig)
    assert not rig.calls
    assert rig.bus.closed


def test_resume_keeps_passed_sweeps_and_rejects_modified_trace(rig):
    def interrupted(*args, **kwargs):
        if len(rig.calls) == 1:
            raise RuntimeError("operator stopped")
        return rig.capture(*args, **kwargs)

    with pytest.raises(RuntimeError, match="operator stopped"):
        run(rig, capture_fn=interrupted)
    folder = next((rig.settings.data_dir / "manual_arm_verifications/follower").iterdir())
    rig.args.resume = folder
    rig.args.source = None
    run(rig)
    assert rig.calls == ["gripper"] * 4
    (folder / "gripper/trial_01.jsonl").write_text("tampered")
    with pytest.raises(RuntimeError, match="trace changed"):
        run(rig)


def test_cleanup_failure_cannot_leave_success_report(rig):
    rig.bus.fail_cleanup = True
    with pytest.raises(OSError, match="cleanup failed"):
        run(rig)
    path = next(rig.settings.data_dir.glob("manual_arm_verifications/follower/*/verification.json"))
    report = json.loads(path.read_text())
    assert report["state"] == "partial"
    assert "cleanup failed" in report["cleanup_error"]


def test_plain_command_resumes_most_progress_despite_newer_empty_run(rig, capsys):
    rig.args.joint = None

    def interrupted(*args, **kwargs):
        if len(rig.calls) == 12:
            raise RuntimeError("failed elbow sweep")
        return rig.capture(*args, **kwargs)

    with pytest.raises(RuntimeError, match="failed elbow"):
        run(rig, capture_fn=interrupted)
    root = rig.settings.data_dir / "manual_arm_verifications/follower"
    original = next(root.iterdir())
    empty = root / "newer-empty"
    empty.mkdir()
    report = json.loads((original / "verification.json").read_text())
    report["attempts"] = {name: [] for name in report["joints"]}
    path = empty / "verification.json"
    path.write_text(json.dumps(report))
    original_time = (original / "verification.json").stat().st_mtime
    os.utime(path, (original_time + 60, original_time + 60))
    rig.args.resume = None
    assert run(rig) == original
    assert len(rig.calls) == 24
    assert rig.calls[12] == "elbow_flex"
    output = capsys.readouterr().out
    assert "wrist_flex: 10/10 accepted · 0 remaining" in output
    assert "elbow_flex: 6/10 accepted · 4 remaining" in output


def test_auto_resume_validates_trace_before_opening_bus(rig):
    folder = run(rig)
    (folder / "gripper/trial_01.jsonl").write_text("tampered")
    rig.args.resume = None
    rig.opener = lambda *args, **kwargs: pytest.fail("Must not open bus")
    with pytest.raises(RuntimeError, match="trace changed"):
        run(rig)


def test_complete_run_is_acknowledged_and_new_is_explicit(rig, capsys):
    folder = run(rig)
    rig.args.resume = None
    rig.opener = lambda *args, **kwargs: pytest.fail("Must not open bus")
    assert run(rig) == folder
    assert "Already complete" in capsys.readouterr().out
    rig.args.resume = None
    rig.args.new = True
    rig.args.dry_run = True
    assert run(rig) != folder


def prepare_bundle_repair(rig, monkeypatch):
    reference = cli.load_reference(rig.args.source, rig.settings, "follower")
    reference["plan"]["bundle_sha256"] = "old-bundle"
    folder = run(rig)
    source = rig.args.source
    source.mkdir()
    manifest = {
        "arm": "follower",
        "device_id": "fixture",
        "port": "fixture",
        "method": "joint-by-joint endpoint-first manual",
        "validations": 5,
    }
    recipes = {name: {"joint": name} for name in JOINT_NAMES}
    (source / "arm_session.json").write_text(json.dumps(manifest))
    bundle = cli.installer.arm_manual.make_bundle(manifest, recipes, "new timestamp")
    (source / "validated_joint_bundle.json").write_text(json.dumps(bundle))
    monkeypatch.setattr(
        cli.installer.arm_manual, "verify_completed", lambda source, manifest, name: recipes[name]
    )
    reference["plan"]["bundle_sha256"] = "new-bundle"
    return folder, reference


def test_bundle_only_repair_preserves_sweeps_and_original_journal(rig, monkeypatch):
    folder, reference = prepare_bundle_repair(rig, monkeypatch)
    original = (folder / "verification.json").read_bytes()
    cli.repair_bundle_metadata(folder, rig.settings)
    report = json.loads((folder / "verification.json").read_text())
    assert report["reference"] == reference
    assert report["attempts"] == json.loads(original)["attempts"]
    repair = report["reference_repairs"][0]
    assert (folder / repair["previous_journal"]).read_bytes() == original
    rig.args.resume = folder
    rig.opener = lambda *args, **kwargs: pytest.fail("Completed replay cannot open bus")
    run(rig)


@pytest.mark.parametrize("changed", ["offset", "trace"])
def test_bundle_repair_refuses_calibration_or_trace_changes(rig, monkeypatch, changed):
    folder, reference = prepare_bundle_repair(rig, monkeypatch)
    original = (folder / "verification.json").read_bytes()
    if changed == "offset":
        reference["plan"]["native"]["gripper"]["homing_offset"] = 101
    else:
        (folder / "gripper/trial_01.jsonl").write_text("{}")
    with pytest.raises(RuntimeError, match="Repair refused"):
        cli.repair_bundle_metadata(folder, rig.settings)
    assert (folder / "verification.json").read_bytes() == original
    assert not list(folder.glob("verification.before-bundle-repair-*.json"))

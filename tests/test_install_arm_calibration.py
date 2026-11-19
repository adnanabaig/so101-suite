"""Installation is torque-off, identity-bound, transactional and separately evidenced."""

from __future__ import annotations

import copy
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.hardware.calibration import CalibrationResult

spec = importlib.util.spec_from_file_location(
    "installer", Path(__file__).resolve().parents[1] / "scripts/install_arm_calibration.py"
)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Native:
    def __init__(self):
        self.motors = {
            name: SimpleNamespace(id=index + 1) for index, name in enumerate(JOINT_NAMES)
        }
        self.reg = {
            name: {
                "Homing_Offset": 0,
                "Min_Position_Limit": 0,
                "Max_Position_Limit": 4095,
                "Operating_Mode": 0,
                "Lock": 1,
            }
            for name in JOINT_NAMES
        }
        self.original = copy.deepcopy(self.reg)
        self.raw = dict.fromkeys(JOINT_NAMES, 2047)
        self.writes = []
        self.fail_once = False

    def read(self, register, joint, **kwargs):
        if register == "Present_Position":
            return self.raw[joint] - self.reg[joint]["Homing_Offset"]
        return self.reg[joint][register]

    def sync_read(self, register, joints, **kwargs):
        if register == "Torque_Enable":
            return dict.fromkeys(joints, 0)
        return {joint: self.read(register, joint) for joint in joints}

    def write(self, register, joint, value, **kwargs):
        assert register in self.reg[joint]  # Never a goal or torque-on register.
        self.writes.append((register, joint, value))
        if self.fail_once and register == "Min_Position_Limit":
            self.fail_once = False
            raise OSError("injected register write failure")
        self.reg[joint][register] = value


@pytest.fixture
def rig(tmp_path, monkeypatch):
    config = get_settings().model_copy(
        update={"backend": "real", "mock": False, "data_dir": tmp_path / "data"}
    )
    config.data_dir.mkdir(exist_ok=True)
    folder = tmp_path / "manual"
    folder.mkdir()
    manifest = {
        "schema_version": 1,
        "state": "validated",
        "arm": "follower",
        "backend": "real",
        "device_id": config.follower_id,
        "port": config.follower_port,
        "joints": list(JOINT_NAMES),
        "validations": 5,
        "results": {},
    }
    recipes = {}
    for index, name in enumerate(JOINT_NAMES):
        diag = {
            "validated": True,
            "restored": True,
            "calibration_applied": False,
            "validation_sweeps": 5,
            "homing_offsets_requested": {name: 100 + index},
            "range_mins": {name: 700},
            "range_maxes": {name: 3400},
            "motor_ids": {name: index + 1},
            "homing_readback": {name: {"homing_offset": 100 + index}},
        }
        recipe = {
            "arm": "follower",
            "joint": name,
            "backend": "lerobot",
            "device_id": config.follower_id,
            "port": config.follower_port,
            "joints": {name: {}},
            "diagnostics": diag,
        }
        path = folder / f"{name}.json"
        path.write_text(json.dumps(recipe))
        manifest["results"][name] = {
            "state": "validated",
            "recipe": path.name,
            "sha256": cli.arm_manual.sha256(path),
        }
        recipes[name] = recipe
    (folder / "arm_session.json").write_text(json.dumps(manifest))
    (folder / "validated_joint_bundle.json").write_text(
        json.dumps({"all_six_joints_validated": True, "joint_recipes": recipes})
    )
    native = Native()
    device = SimpleNamespace(calibration_fpath=tmp_path / "native.json")
    device.calibration_fpath.write_bytes(b'{"original_native": true}\n')
    suite = config.data_dir / "calibration_follower.json"
    suite.write_bytes(b'{"original_suite": true}\n')
    state = {"opened": False, "closed": False, "faults": []}

    class Bus:
        _bus = native
        _device = device

        def connect(self):
            state["opened"] = True

        def set_torque(self, enabled):
            assert enabled is False

    @contextmanager
    def access(*args, **kwargs):
        def close(bus, **kwargs):
            state["closed"] = True

        yield SimpleNamespace(disconnect_device=close)

    monkeypatch.setattr(cli, "standalone_access", access)
    monkeypatch.setattr(
        cli,
        "persist_standalone_fault",
        lambda config, exc, **kwargs: state["faults"].append(str(exc)),
    )

    def position(bus, target, *, joint):
        native.raw[joint] = round(target)

    return SimpleNamespace(
        config=config,
        folder=folder,
        native=native,
        state=state,
        bus=lambda *args: Bus(),
        position=position,
        options=SimpleNamespace(arm="follower", session=folder, dry_run=False),
        suite=suite,
        native_path=device.calibration_fpath,
    )


def run(rig, **kwargs):
    return cli.install(
        rig.options,
        rig.config,
        ask=lambda _: "READY" if not rig.state["opened"] else "INSTALL",
        bus_factory=rig.bus,
        position_fn=rig.position,
        **kwargs,
    )


def test_install_readbacks_and_retains_torque_off(rig):
    evidence = run(rig)
    state = json.loads((evidence / "install.json").read_text())
    assert state["state"] == "installed" and state["torque_off_verified"]
    assert rig.state["closed"] and not rig.state["faults"]
    assert len(state["probes"]) == 6
    suite = CalibrationResult.load("follower", rig.suite)
    assert suite and set(suite.joints) == set(JOINT_NAMES)
    native = json.loads(rig.native_path.read_text())
    assert (native["wrist_roll"]["range_min"], native["wrist_roll"]["range_max"]) == (0, 4095)
    assert suite.joints["wrist_roll"].min == pytest.approx((700 - 2047.5) * 360 / 4095)
    assert suite.joints["gripper"].max == 100
    assert (evidence / "previous_file_0.bin").read_bytes() == b'{"original_suite": true}\n'
    assert (evidence / "previous_file_1.bin").read_bytes() == b'{"original_native": true}\n'


def test_dry_run_never_opens_or_changes_files(rig, monkeypatch):
    rig.options.dry_run = True
    monkeypatch.setattr(cli, "standalone_access", lambda *a, **k: pytest.fail("lock acquired"))
    run(rig)
    assert not rig.state["opened"] and not rig.native.writes
    assert not (rig.config.data_dir / "arm_calibration_installs").exists()


@pytest.mark.parametrize("bad", ["hash", "profile", "count", "joint_state", "bundle"])
def test_invalid_session_refused_before_open(rig, bad):
    path = rig.folder / "arm_session.json"
    manifest = json.loads(path.read_text())
    if bad == "hash":
        (rig.folder / "gripper.json").write_text("{}")
    elif bad == "profile":
        manifest["device_id"] = "different_arm"
    elif bad == "count":
        manifest["validations"] = 3
    elif bad == "joint_state":
        manifest["results"]["gripper"]["state"] = "failed"
    else:
        bundle = rig.folder / "validated_joint_bundle.json"
        data = json.loads(bundle.read_text())
        data["joint_recipes"]["gripper"]["diagnostics"]["validated"] = False
        bundle.write_text(json.dumps(data))
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError)):
        run(rig)
    assert not rig.state["opened"]


@pytest.mark.parametrize("failure", ["register", "probe", "publish", "interrupt"])
def test_failure_rolls_back_all_registers_and_exact_files(rig, monkeypatch, failure):
    if failure == "register":
        rig.native.fail_once = True
    if failure == "publish":
        original = cli.atomic_write_json

        def write(path, value):
            if path == rig.suite:
                raise OSError("injected suite publication failure")
            original(path, value)

        monkeypatch.setattr(cli, "atomic_write_json", write)

    def probe(native, offset, **kwargs):
        cli.manual.write_offset(native, offset, **kwargs)
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure == "probe":
            raise RuntimeError("injected feedback failure")

    with pytest.raises((RuntimeError, OSError, KeyboardInterrupt)):
        run(rig, offset_fn=probe)
    assert rig.native.reg == rig.native.original
    assert rig.suite.read_bytes() == b'{"original_suite": true}\n'
    assert rig.native_path.read_bytes() == b'{"original_native": true}\n'
    evidence = next((rig.config.data_dir / "arm_calibration_installs/follower").iterdir())
    state = json.loads((evidence / "install.json").read_text())
    assert state["state"] == "failed" and state["rollback_verified"]
    assert rig.state["closed"] and not rig.state["faults"]


def test_rollback_failure_persists_fault(rig, monkeypatch):
    rig.native.fail_once = True

    def broken(*args, **kwargs):
        raise OSError("injected restore failure")

    monkeypatch.setattr(cli.manual, "restore", broken)
    with pytest.raises(OSError):
        run(rig)
    assert rig.state["faults"] and "rollback failed" in rig.state["faults"][0]


def test_cleanup_failure_keeps_consistent_install_and_latches_fault(rig, monkeypatch):
    @contextmanager
    def access(*args, **kwargs):
        def close(*args, **kwargs):
            raise OSError("injected teardown failure")

        yield SimpleNamespace(disconnect_device=close)

    monkeypatch.setattr(cli, "standalone_access", access)
    with pytest.raises(OSError):
        run(rig)
    assert rig.state["faults"]
    assert set(json.loads(rig.native_path.read_text())) == set(JOINT_NAMES)
    evidence = next((rig.config.data_dir / "arm_calibration_installs/follower").iterdir())
    state = json.loads((evidence / "install.json").read_text())
    assert state["state"] == "installed_cleanup_failed"


def test_positioning_cancel_before_write_leaves_originals(rig):
    def stop(*args, **kwargs):
        raise RuntimeError("operator cancelled")

    rig.position = stop
    with pytest.raises(RuntimeError, match="operator cancelled"):
        run(rig)
    assert not rig.native.writes and rig.state["closed"]

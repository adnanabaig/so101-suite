"""Reference measurements never change motor state or infer a model correction."""

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101 import JOINT_NAMES
from so101.config import get_settings

spec = importlib.util.spec_from_file_location(
    "reference", Path(__file__).resolve().parents[1] / "scripts/check_shoulder_reference.py"
)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Bus:
    def __init__(self):
        self.reg = {
            "Homing_Offset": 527,
            "Min_Position_Limit": 823,
            "Max_Position_Limit": 3267,
            "Operating_Mode": 0,
        }
        self.position = 2045
        self.torque = 0
        self.closed = False
        self.reads = 0

    def read(self, register, joint, **kwargs):
        assert joint == "shoulder_lift"
        if register == "Present_Position":
            self.reads += 1
            return self.position
        return self.reg[register]

    def sync_read(self, register, joints, **kwargs):
        assert register == "Torque_Enable" and set(joints) == set(JOINT_NAMES)
        return dict.fromkeys(joints, self.torque)

    def disconnect(self, *, disable_torque):
        assert disable_torque is False
        self.closed = True


CAL = {"homing_offset": 527, "range_min": 823, "range_max": 3267}


def test_model_reference_is_derived_from_pivot_geometry():
    assert cli.model_angle(0) == pytest.approx(13.96796008)
    assert cli.model_angle(20) - cli.model_angle(0) == pytest.approx(20)
    assert cli.model_angle(-13.96796008) == pytest.approx(0, abs=1e-7)


def test_measure_retains_raw_ticks_and_degree_scale():
    bus = Bus()
    result = cli.measure(bus, CAL, sleep=lambda _: None)
    assert bus.reads == 30 and result["suite_degrees"] == 0
    assert result["spread_degrees"] == 0
    assert result["samples"][0]["raw_ticks"] == 2572


@pytest.mark.parametrize("failure", ["torque", "offset", "range", "motion"])
def test_bad_measurements_refused(failure):
    bus = Bus()
    if failure == "torque":
        bus.torque = 1
    elif failure == "offset":
        bus.reg["Homing_Offset"] = 1
    elif failure == "range":
        bus.position = 4000

    def sleep(_):
        if failure == "motion":
            bus.position += 2

    with pytest.raises(RuntimeError):
        cli.measure(bus, CAL, sleep=sleep)


@pytest.mark.parametrize("invalid", [[], [""], ["enter", "nan", "14", "46"]])
def test_collect_two_poses_keeps_direction_unverified(tmp_path, monkeypatch, invalid):
    settings = get_settings().model_copy(
        update={"backend": "real", "mock": False, "data_dir": tmp_path}
    )
    path = tmp_path / "native.json"
    path.write_text(json.dumps({"shoulder_lift": CAL}))
    original = path.read_bytes()
    monkeypatch.setattr(cli, "lerobot_calibration_file", lambda *args: path)
    monkeypatch.setattr(cli, "calibration_digest", lambda *args: "same-digest")

    @contextmanager
    def access(*args, **kwargs):
        yield

    monkeypatch.setattr(cli, "standalone_access", access)
    answers = iter(["READY", "", *invalid, "20", ""])
    bus = Bus()
    results = iter([-14.0, 6.0])

    def measure(*args):
        return {"suite_degrees": next(results), "spread_degrees": 0.0, "samples": []}

    evidence = cli.collect(
        SimpleNamespace(dry_run=False),
        settings,
        ask=lambda _: next(answers),
        opener=lambda *a, **k: bus,
        measure_fn=measure,
    )
    report = json.loads((evidence / "reference.json").read_text())
    assert report["state"] == "captured" and bus.closed
    assert report["encoder_to_measured_angle_ratio"] == 1
    assert len(report["poses"][1]["expected_model_alternatives_degrees"]) == 2
    assert report["calibration_applied"] is False
    assert path.read_bytes() == original


def test_dry_run_never_takes_lock_or_opens_serial(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    monkeypatch.setattr(cli, "standalone_access", lambda *a, **k: pytest.fail("took lock"))
    assert cli.collect(SimpleNamespace(dry_run=True), settings) is None

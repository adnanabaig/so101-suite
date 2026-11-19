"""Manual baseline recorder retains raw evidence and never writes motor state."""

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
    "wrist_recorder", ROOT / "scripts/record_wrist_roll.py"
)
assert spec and spec.loader
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


@pytest.mark.parametrize(
    ("excursion", "gap", "low_ready"),
    [(1155, 0.02, True), (1100, 0.02, False), (1155, 0.2, False)],
)
def test_settled_endpoints_preserve_excursions_and_require_continuity(excursion, gap, low_ready):
    values = [excursion] + [1198 + index % 3 for index in range(50)] + [2670] * 50
    samples = [
        {"raw_ticks": value, "elapsed_seconds": index * gap} for index, value in enumerate(values)
    ]
    checks = recorder.settled_endpoints(samples, min(values), max(values))
    assert (checks["low_settled_ticks"] is not None) is low_ready
    assert min(sample["raw_ticks"] for sample in samples) == excursion
    assert (checks["high_settled_ticks"] is not None) is (gap == 0.02)


def test_moving_through_endpoint_band_is_not_a_settled_pause():
    samples = [
        {"raw_ticks": value, "elapsed_seconds": index * 0.02}
        for index, value in enumerate(range(1200, 1350))
    ]
    checks = recorder.settled_endpoints(samples, 1200, 1349)
    assert checks["low_settled_ticks"] is None
    assert checks["high_settled_ticks"] is None


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ReadOnlyBus:
    def __init__(self, *, offset=688, positions=(-16, 3856, 2047)):
        self.offset = offset
        self.positions = iter(positions)
        self.torque = dict.fromkeys(JOINT_NAMES, 0)
        self.closed = False
        self.calls = []

    def read(self, register, joint, **kwargs):
        assert joint == "wrist_roll" and kwargs["normalize"] is False
        return {
            "Homing_Offset": self.offset,
            "Operating_Mode": 0,
            "Min_Position_Limit": 0,
            "Max_Position_Limit": 4095,
        }[register]

    def sync_read(self, register, joints, **kwargs):
        self.calls.append((register, list(joints)))
        assert kwargs["normalize"] is False
        if register == "Torque_Enable":
            return dict(self.torque)
        assert register == "Present_Position" and joints == ["wrist_roll"]
        value = next(self.positions)
        if isinstance(value, Exception):
            raise value
        return {"wrist_roll": value}

    def disconnect(self, *, disable_torque):
        assert disable_torque is False  # no driver torque-off side effect
        self.closed = True


def finish_after(samples):
    count = 0

    def key():
        nonlocal count
        count += 1
        return "complete" if count > samples else None

    return key


def capture(tmp_path, bus=None, *, stop=None, **kwargs):
    clock = Clock()
    return recorder.capture_trial(
        bus or ReadOnlyBus(),
        tmp_path,
        1,
        stop or finish_after(3),
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_records_unclamped_raw_trace_and_candidate_without_writes(tmp_path):
    positions = (-16, 1000, 2000, 3000, 3856, 2047)
    result = capture(tmp_path, ReadOnlyBus(positions=positions), stop=finish_after(6))
    assert result["status"] == "complete" and result["usable_for_comparison"]
    assert result["reported_min"] == -16 and result["reported_max"] == 3856
    assert result["raw_min"] == 672 and result["raw_max"] == 4544
    assert result["registers_before"] == result["registers_after"]
    proposed = result["candidate_not_applied"]
    assert proposed["homing_offset"] == 561
    assert proposed["range_min"] == 111 and proposed["range_max"] == 3983
    assert proposed["applied"] is False
    events = [json.loads(line) for line in (tmp_path / "trial_01.jsonl").read_text().splitlines()]
    samples = [event for event in events if event["type"] == "sample"]
    assert [event["raw_ticks"] for event in samples] == [value + 688 for value in positions]
    assert [event["reported_ticks"] for event in samples] == list(positions)
    assert [event["elapsed_seconds"] for event in samples] == pytest.approx(
        [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    )
    assert json.loads((tmp_path / "trial_01.json").read_text()) == result


@pytest.mark.parametrize("kind", ["key", "ctrl_c", "read_error"])
def test_interruption_retains_partial_samples(tmp_path, kind):
    bus = ReadOnlyBus()
    calls = 0

    def key():
        nonlocal calls
        calls += 1
        if calls > 2:
            if kind == "ctrl_c":
                raise KeyboardInterrupt
            return "abort"
        return None

    if kind == "read_error":
        bus.positions = iter((1000, 1200, ConnectionError("serial read failed")))

        def key():
            return None

    result = capture(tmp_path, bus, stop=key)
    assert result["sample_count"] == 2
    assert result["status"] == ("error" if kind == "read_error" else "interrupted")
    assert not result["usable_for_comparison"]
    assert (tmp_path / "trial_01.json").exists()
    assert (
        len(
            [
                event
                for event in map(json.loads, (tmp_path / "trial_01.jsonl").read_text().splitlines())
                if event["type"] == "sample"
            ]
        )
        == 2
    )


@pytest.mark.parametrize("unsafe", ["on", "missing", "unreadable"])
def test_torque_must_be_confirmed_off_before_recording(tmp_path, unsafe):
    bus = ReadOnlyBus()
    if unsafe == "on":
        bus.torque["gripper"] = 1
    elif unsafe == "missing":
        bus.torque.pop("elbow_flex")
    else:
        bus.sync_read = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("unreadable"))
    ready = []
    result = capture(tmp_path, bus, ready=lambda: ready.append(True))
    assert result["status"] == "error" and result["sample_count"] == 0
    assert not ready  # never tells the operator to begin moving


def test_mid_capture_torque_change_stops_and_retains_evidence(tmp_path):
    bus = ReadOnlyBus(positions=range(1000, 1100))
    clock = Clock()

    def key():
        if clock.now > 0.5:
            bus.torque["wrist_roll"] = 1
        return None

    result = recorder.capture_trial(bus, tmp_path, 1, key, clock=clock, sleep=clock.sleep)
    assert result["status"] == "error"
    assert "torque OFF" in result["error"]
    assert result["sample_count"] > 0
    assert not result["usable_for_comparison"]


def test_register_change_invalidates_trial(tmp_path):
    bus = ReadOnlyBus()
    calls = 0

    def key():
        nonlocal calls
        calls += 1
        if calls > 3:
            bus.offset = 1000
            return "complete"
        return None

    result = capture(tmp_path, bus, stop=key)
    assert result["status"] == "error"
    assert "registers changed" in result["error"]
    assert not result["usable_for_comparison"]


@pytest.mark.parametrize(
    "positions,warning",
    [
        ((1000, 1001), "observed span below 100 ticks"),
        ((0, 5000), "observed span exceeds one encoder revolution"),
        ((1000, 4000), "large adjacent jump"),
    ],
)
def test_quality_flags_do_not_hide_original_samples(tmp_path, positions, warning):
    result = capture(tmp_path, ReadOnlyBus(positions=positions), stop=finish_after(2))
    assert any(warning in value for value in result["quality_warnings"])
    assert not result["usable_for_comparison"]
    assert result["sample_count"] == 2


def test_time_limit_is_incomplete_not_a_success(tmp_path):
    result = capture(
        tmp_path,
        ReadOnlyBus(positions=range(1000, 2000)),
        stop=lambda: None,
        max_seconds=0.05,
    )
    assert result["status"] == "timed_out"
    assert not result["operator_confirmed_safe_endpoints"]
    assert not result["usable_for_comparison"]


@pytest.mark.parametrize("confirmation", ["READY", "ready", "  Ready  "])
def test_ten_trials_persist_individually_and_analyze(tmp_path, monkeypatch, confirmation):
    args = SimpleNamespace(
        mock=False,
        arm="follower",
        trials=10,
        max_seconds=180,
        output=tmp_path / "ten-trials",
    )
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    gate_held = False

    @contextmanager
    def gate(*a, **kwargs):
        nonlocal gate_held
        assert kwargs["settings"] is settings
        gate_held = True
        yield
        gate_held = False

    bus = ReadOnlyBus(positions=tuple(range(1000, 3001, 100)) * 10)

    def opener(*a, **k):
        assert gate_held
        return bus

    original = recorder.capture_trial

    def short_trial(bus, folder, number, stop_key, **kwargs):
        clock = Clock()
        return original(bus, folder, number, finish_after(21), clock=clock, sleep=clock.sleep)

    monkeypatch.setattr(recorder, "standalone_access", gate)
    monkeypatch.setattr(recorder, "capture_trial", short_trial)
    answers = iter([confirmation] + [""] * 10)
    folder = recorder.collect(args, settings, ask=lambda _: next(answers), opener=opener)
    assert bus.closed and not gate_held
    assert len(list(folder.glob("trial_*.json"))) == 10
    assert len(list(folder.glob("trial_*.jsonl"))) == 10
    session = json.loads((folder / "session.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    assert session["state"] == "complete" and session["completed_trials"] == 10
    assert report["usable_trials"] == 10 and report["all_requested_trials_usable"]
    assert report["comparison"]["low_spread_ticks"] == 0
    assert report["comparison"]["high_spread_ticks"] == 0
    assert report["calibration_applied"] is False
    assert report["hardware_safety_qualified"] is False


def test_server_or_latch_refusal_never_opens_bus(tmp_path, monkeypatch):
    @contextmanager
    def refuse(*a, **k):
        raise RuntimeError("server owns bus")
        yield

    monkeypatch.setattr(recorder, "standalone_access", refuse)
    args = SimpleNamespace(mock=False)
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    with pytest.raises(RuntimeError, match="server owns bus"):
        recorder.collect(
            args,
            settings,
            ask=lambda _: "READY",
            opener=lambda *a, **k: pytest.fail("opened competing bus"),
        )


def test_existing_output_is_preserved_and_bus_is_closed(tmp_path, monkeypatch):
    @contextmanager
    def gate(*a, **k):
        yield

    monkeypatch.setattr(recorder, "standalone_access", gate)
    bus = ReadOnlyBus()
    args = SimpleNamespace(mock=False, arm="follower", output=tmp_path)
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    before = set(tmp_path.iterdir())
    with pytest.raises(FileExistsError):
        recorder.collect(args, settings, ask=lambda _: "READY", opener=lambda *a, **k: bus)
    assert bus.closed
    assert set(tmp_path.iterdir()) == before


def test_read_failure_keeps_session_manifest_and_closes_bus(tmp_path, monkeypatch):
    @contextmanager
    def gate(*a, **k):
        yield

    monkeypatch.setattr(recorder, "standalone_access", gate)
    bus = ReadOnlyBus(positions=(ConnectionError("encoder unavailable"),))
    args = SimpleNamespace(
        mock=False, arm="follower", output=tmp_path / "failed", trials=10, max_seconds=180
    )
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    replies = iter(["READY", ""])
    folder = recorder.collect(
        args, settings, ask=lambda _: next(replies), opener=lambda *a, **k: bus
    )
    assert bus.closed
    assert json.loads((folder / "session.json").read_text())["state"] == "error"
    assert json.loads((folder / "report.json").read_text())["usable_trials"] == 0


@pytest.mark.parametrize("confirmation", ["", "yes", "not ready", "q"])
def test_unconfirmed_setup_never_reaches_gate_or_bus(monkeypatch, confirmation):
    monkeypatch.setattr(
        recorder, "standalone_access", lambda *a, **k: pytest.fail("unconfirmed setup took lock")
    )
    settings = get_settings().model_copy(update={"backend": "real", "mock": False})
    with pytest.raises(RuntimeError, match="Setup not confirmed"):
        recorder.collect(
            SimpleNamespace(mock=False),
            settings,
            ask=lambda _: confirmation,
            opener=lambda *a, **k: pytest.fail("unconfirmed setup opened motor bus"),
        )

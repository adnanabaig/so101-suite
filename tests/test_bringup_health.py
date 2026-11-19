"""Servo health: unit conversion, threshold evaluation and bus read paths."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from so101 import JOINT_NAMES
from so101.config import Settings, get_settings
from so101.hardware import health as h
from so101.hardware.motors import FeetechBus, MockBus

BASE_RAW = {
    "Present_Temperature": 30,
    "Present_Voltage": 50,
    "Present_Load": 0,
    "Present_Current": 10,
    "Status": 0,
}


def _sample(limits: h.HealthLimits | None = None, **raw: int) -> h.ServoHealthSample:
    values = {**BASE_RAW, **raw}
    return h.sample_from_raw(
        "elbow_flex", values, source="real", timestamp=1.0, motor_id=3, limits=limits
    )


def _limits(max_temperature: int = 70) -> h.HealthLimits:
    return h.limits_from_raw(
        {"Max_Temperature_Limit": max_temperature, "Max_Voltage_Limit": 80, "Min_Voltage_Limit": 40}
    )


def _level(sample: h.ServoHealthSample, thresholds: h.HealthThresholds | None = None) -> str:
    return h.evaluate_sample(sample, thresholds or h.HealthThresholds()).level


# --- conversion -----------------------------------------------------------------


def test_conversions_use_the_documented_scales() -> None:
    sample = _sample(
        Present_Temperature=41,
        Present_Voltage=123,
        Present_Load=-512,
        Present_Current=200,
        Status=0b100101,
    )
    assert sample.temperature_c == 41.0
    assert sample.voltage_v == 12.3
    assert sample.load_pct == -51.2  # sign already decoded by LeRobot's sync_read
    assert sample.current_ma == 1300.0
    assert sample.status_flags == ("voltage", "temperature", "overload")
    assert sample.raw["Present_Load"] == -512
    assert sample.motor_id == 3 and sample.source == "real"


def test_unknown_status_bits_are_named_by_position() -> None:
    assert h.decode_status(0) == ()
    assert h.decode_status(0b1100_0000) == ("bit6", "bit7")


def test_a_sample_needs_every_health_register() -> None:
    with pytest.raises(ValueError, match="Status"):
        h.sample_from_raw(
            "gripper",
            {k: v for k, v in BASE_RAW.items() if k != "Status"},
            source="mock",
            timestamp=0.0,
        )


def test_limits_convert_with_the_same_scales() -> None:
    limits = _limits(72)
    assert limits.max_temperature_c == 72.0
    assert limits.max_voltage_v == 8.0
    assert limits.min_voltage_v == 4.0


def test_physical_samples_round_trip_through_raw_registers() -> None:
    sample = h.sample_from_physical(
        "wrist_flex",
        source="sim",
        timestamp=5.0,
        temperature_c=33.4,
        voltage_v=11.96,
        load_pct=-12.34,
        current_ma=130.0,
        status=0,
    )
    assert sample.raw == {
        "Present_Temperature": 33,
        "Present_Voltage": 120,
        "Present_Load": -123,
        "Present_Current": 20,
        "Status": 0,
    }
    assert sample.source == "sim"
    assert sample.voltage_v == 12.0


def test_every_physical_scale_is_labeled_unverified() -> None:
    for register in h.HEALTH_REGISTERS + h.LIMIT_REGISTERS:
        provenance = h.UNIT_PROVENANCE[register]
        assert provenance.status == "unverified"
        assert "LeRobot 0.6.1" in provenance.source
    assert "Sign bit 10 is verified" in h.UNIT_PROVENANCE["Present_Load"].source


# --- thresholds -----------------------------------------------------------------


def test_thresholds_follow_the_supply_profile_per_arm() -> None:
    settings = Settings(health_follower_supply="12v")
    follower = h.HealthThresholds.from_settings(settings, "follower")
    leader = h.HealthThresholds.from_settings(settings, "leader")
    assert follower.supply_profile == "12v"
    assert follower.voltage_band == settings.health_supply_band_12v
    assert leader.supply_profile == "7.4v"
    assert leader.voltage_band == settings.health_supply_band_7v4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voltage_band": (5.0, 4.0, 8.0, 9.0)},
        {"temp_warn_margin_c": 5.0, "temp_critical_margin_c": 5.0},
        {"temp_fallback_warn_c": 70.0, "temp_fallback_critical_c": 60.0},
        {"load_warn_pct": 90.0, "load_critical_pct": 90.0},
        {"current_warn_ma": 900.0, "current_critical_ma": 100.0},
        {"max_consecutive_failures": 0},
    ],
)
def test_inconsistent_thresholds_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        h.HealthThresholds(**kwargs)


def test_health_settings_validate_and_parse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        Settings(health_poll_hz=3.0)
    with pytest.raises(ValidationError):
        Settings(health_supply_band_7v4=(4.0, 4.0, 8.0, 8.5))
    with pytest.raises(ValidationError):
        Settings(health_max_consecutive_failures=0)
    monkeypatch.setenv("SO101_HEALTH_LOAD_WARN_PCT", "none")
    monkeypatch.setenv("SO101_HEALTH_CURRENT_CRITICAL_MA", "2500")
    monkeypatch.setenv("SO101_HEALTH_SUPPLY_BAND_12V", "[9, 10, 13, 14]")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.health_load_warn_pct is None
    assert settings.health_current_critical_ma == 2500.0
    assert settings.health_supply_band_12v == (9.0, 10.0, 13.0, 14.0)
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(54, "ok"), (55, "warn"), (64, "warn"), (65, "critical"), (90, "critical")],
)
def test_temperature_boundaries_come_from_the_servo_limit(raw: int, expected: str) -> None:
    # Max_Temperature_Limit 70 -> warn at 70 - 15, critical at 70 - 5 (inclusive).
    assert _level(_sample(limits=_limits(70), Present_Temperature=raw)) == expected


def test_temperature_falls_back_when_the_limit_is_unknown_or_implausible() -> None:
    thresholds = h.HealthThresholds()
    assert thresholds.temperature_thresholds(None).warn_c == 55.0
    implausible = thresholds.temperature_thresholds(200.0)
    assert (implausible.warn_c, implausible.critical_c) == (55.0, 65.0)
    assert "implausible" in implausible.basis
    lower = thresholds.temperature_thresholds(60.0)
    assert (lower.warn_c, lower.critical_c) == (45.0, 55.0)
    assert _level(_sample(Present_Temperature=65)) == "critical"
    assert _level(_sample(limits=_limits(60), Present_Temperature=45)) == "warn"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (39, "critical"),
        (40, "critical"),
        (41, "warn"),
        (45, "warn"),
        (46, "ok"),
        (50, "ok"),
        (79, "ok"),
        (80, "warn"),
        (84, "warn"),
        (85, "critical"),
        (120, "critical"),
    ],
)
def test_voltage_band_boundaries_for_the_7v4_profile(raw: int, expected: str) -> None:
    assert _level(_sample(Present_Voltage=raw)) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (50, "critical"),
        (95, "critical"),
        (105, "warn"),
        (106, "ok"),
        (130, "warn"),
        (140, "critical"),
    ],
)
def test_voltage_band_boundaries_for_the_12v_profile(raw: int, expected: str) -> None:
    thresholds = h.HealthThresholds.from_settings(Settings(health_follower_supply="12v"))
    assert _level(_sample(Present_Voltage=raw), thresholds) == expected


@pytest.mark.parametrize("supply", ["7.4v", "12v"])
def test_shared_grading_helpers_match_the_monitor_at_every_reading(supply: str) -> None:
    """Preflight grades with grade_voltage/grade_temperature; the monitor must agree exactly."""
    thresholds = h.HealthThresholds.from_settings(Settings(health_follower_supply=supply))

    def monitor_level(sample: h.ServoHealthSample, category: str) -> str:
        reasons = h.evaluate_sample(sample, thresholds).reasons
        return next((r.level for r in reasons if r.category == category), "ok")

    for raw in range(0, 160):
        sample = _sample(Present_Voltage=raw)
        level, threshold, relation = h.grade_voltage(sample.voltage_v, thresholds)
        assert monitor_level(sample, "voltage") == level, raw
        assert (threshold is None) == (level == "ok") == (relation == "within")
    for limit in (None, 60, 70, 200):
        limits = _limits(limit) if limit is not None else None
        limit_c = limits.max_temperature_c if limits is not None else None
        for raw in range(20, 100):
            sample = _sample(limits=limits, Present_Temperature=raw)
            level, _, basis = h.grade_temperature(sample.temperature_c, limit_c, thresholds)
            assert monitor_level(sample, "temperature") == level, (limit, raw)
            assert basis == thresholds.temperature_thresholds(limit_c)


def test_load_is_graded_on_magnitude_and_critical_is_optional() -> None:
    assert _level(_sample(Present_Load=799)) == "ok"
    assert _level(_sample(Present_Load=800)) == "warn"
    assert _level(_sample(Present_Load=-800)) == "warn"
    assert _level(_sample(Present_Load=1000)) == "warn"  # no critical by default
    strict = h.HealthThresholds(load_warn_pct=50.0, load_critical_pct=95.0)
    assert _level(_sample(Present_Load=-950), strict) == "critical"
    disabled = h.HealthThresholds(load_warn_pct=None)
    assert _level(_sample(Present_Load=1000), disabled) == "ok"


def test_current_boundaries() -> None:
    thresholds = h.HealthThresholds(current_warn_ma=1300.0, current_critical_ma=1950.0)
    assert _level(_sample(Present_Current=199), thresholds) == "ok"
    assert _level(_sample(Present_Current=200), thresholds) == "warn"  # 1300.0 mA
    assert _level(_sample(Present_Current=300), thresholds) == "critical"  # 1950.0 mA


def test_status_level_is_configurable() -> None:
    assert _level(_sample(Status=0b1)) == "warn"
    assert _level(_sample(Status=0b1), h.HealthThresholds(status_nonzero_level="critical")) == (
        "critical"
    )
    assert _level(_sample(Status=0b1), h.HealthThresholds(status_nonzero_level="ok")) == "ok"


def test_evaluate_rolls_up_the_worst_motor_with_reasons() -> None:
    hot = _sample(limits=_limits(70), Present_Temperature=66)
    sagging = h.sample_from_raw(
        "gripper", {**BASE_RAW, "Present_Voltage": 44}, source="real", timestamp=1.0
    )
    evaluation = h.evaluate({"elbow_flex": hot, "gripper": sagging}, h.HealthThresholds(), now=9.0)
    assert evaluation.level == "critical"
    assert evaluation.motors["gripper"].level == "warn"
    categories = {(reason.motor, reason.category) for reason in evaluation.reasons}
    assert categories == {("elbow_flex", "temperature"), ("gripper", "voltage")}
    assert evaluation.evaluated_at == 9.0
    assert h.evaluate({}, h.HealthThresholds()).level == "warn"


# --- read paths ------------------------------------------------------------------


class FakeMotorsBus:
    """Mimics LeRobot 0.6.1 ``sync_read(data_name, motors, *, normalize, num_retry)``."""

    def __init__(self, values: dict[str, dict[str, int]], fail_on: str | None = None) -> None:
        self.motors = {
            name: SimpleNamespace(id=index + 1) for index, name in enumerate(JOINT_NAMES)
        }
        self.values = values
        self.fail_on = fail_on
        self.calls: list[tuple[str, tuple[str, ...], bool, int]] = []

    def sync_read(
        self, data_name: str, motors: Any = None, *, normalize: bool = True, num_retry: int = 0
    ) -> dict[str, int]:
        self.calls.append((data_name, tuple(motors or ()), normalize, num_retry))
        if data_name == self.fail_on:
            raise ConnectionError(f"no status packet for {data_name}")
        return {name: self.values.get(data_name, {}).get(name, 0) for name in motors}


def _bus_values() -> dict[str, dict[str, int]]:
    registers = {**BASE_RAW, "Max_Temperature_Limit": 70, "Max_Voltage_Limit": 80}
    values = {register: dict.fromkeys(JOINT_NAMES, value) for register, value in registers.items()}
    values["Min_Voltage_Limit"] = dict.fromkeys(JOINT_NAMES, 40)
    values["Present_Load"]["shoulder_lift"] = -321
    return values


def test_motors_bus_health_uses_raw_public_sync_reads() -> None:
    bus = FakeMotorsBus(_bus_values())
    samples = h.read_motors_bus_health(bus, source="real", clock=lambda: 42.0)
    assert [call[0] for call in bus.calls] == list(h.HEALTH_REGISTERS)
    assert all(call[2] is False and call[3] == 0 for call in bus.calls)
    assert set(samples) == set(JOINT_NAMES)
    assert samples["shoulder_lift"].load_pct == -32.1
    assert samples["gripper"].motor_id == 6
    assert samples["gripper"].timestamp == 42.0
    limits = h.read_motors_bus_limits(bus)
    assert [call[0] for call in bus.calls[-3:]] == list(h.LIMIT_REGISTERS)
    assert limits["wrist_roll"].max_temperature_c == 70.0


def test_a_failed_register_aborts_the_tick() -> None:
    bus = FakeMotorsBus(_bus_values(), fail_on="Present_Load")
    with pytest.raises(ConnectionError):
        h.read_motors_bus_health(bus)
    assert [call[0] for call in bus.calls] == [
        "Present_Temperature",
        "Present_Voltage",
        "Present_Load",
    ]


def test_mock_bus_health_is_synthetic_plausible_and_side_effect_free() -> None:
    bus = MockBus(name="follower")
    with pytest.raises(RuntimeError, match="not connected"):
        bus.read_health()
    bus.connect()
    before = dict(bus._pos)
    samples = bus.read_health()
    assert bus._pos == before  # reading health never integrates the simulated arm
    assert set(samples) == set(JOINT_NAMES)
    for joint, sample in samples.items():
        assert sample.source == "mock"
        assert sample.motor_id == JOINT_NAMES.index(joint) + 1
        assert 20.0 <= sample.temperature_c <= 40.0
        assert sample.voltage_v == 5.0  # 7.4 V servo variant on the kit's 5 V supply
        assert sample.load_pct == 0.0  # torque off
        assert sample.limits is not None and sample.limits.max_temperature_c == 70.0
    bus.set_torque(True)
    bus.write_positions({"shoulder_lift": 40.0})
    loaded = bus.read_health()
    assert loaded["shoulder_lift"].load_pct > 0
    assert h.evaluate(loaded, h.HealthThresholds()).level == "ok"
    assert isinstance(bus, h.HealthReadable)


def test_mock_bus_uses_the_configured_supply_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SO101_HEALTH_LEADER_SUPPLY", "12v")
    get_settings.cache_clear()
    leader = MockBus(name="leader")
    leader.connect()
    assert leader.read_health()["gripper"].voltage_v == 12.0
    get_settings.cache_clear()


def test_mock_temperature_drifts_up_with_motion_and_cools_at_rest() -> None:
    now = [0.0]
    model = h.MockHealthModel(clock=lambda: now[0], wall_clock=lambda: 1000.0)
    rest = dict.fromkeys(JOINT_NAMES, 0.0)
    first = model.sample(rest, rest, torque=True)["elbow_flex"].temperature_c
    goal = dict(rest)
    for step in range(1, 11):
        now[0] = float(step)
        goal = {joint: (100.0 if step % 2 else -100.0) for joint in JOINT_NAMES}
        model.sample(goal, goal, torque=True)
    warmed = model._temperature["elbow_flex"]
    # Nine 200-degree swings at HEAT_PER_DEG_C (0.004 degC per degree): about 7 degC.
    assert warmed > first + 6.0
    now[0] = 2000.0
    model.sample(goal, goal, torque=False)
    assert model._temperature["elbow_flex"] < warmed - 5.0
    assert model._temperature["elbow_flex"] == pytest.approx(model.AMBIENT_C, abs=0.1)


def test_mock_overrides_and_injected_failures() -> None:
    bus = MockBus(name="follower")
    bus.connect()
    bus.health_model.set_override("wrist_roll", temperature_c=66.0, status=0b100)
    samples = bus.read_health()
    assert samples["wrist_roll"].temperature_c == 66.0
    assert samples["wrist_roll"].status_flags == ("temperature",)
    assert h.evaluate(samples, h.HealthThresholds()).level == "critical"
    bus.health_model.fail_next_reads(1)
    with pytest.raises(RuntimeError, match="injected"):
        bus.read_health()
    bus.health_model.clear_overrides()
    assert bus.read_health()["wrist_roll"].temperature_c < 40.0
    with pytest.raises(ValueError):
        bus.health_model.set_override("tail", temperature_c=1.0)


def test_feetech_bus_health_reads_limits_once_per_connection() -> None:
    bus = FeetechBus("/dev/null", 1_000_000, name="follower")
    with pytest.raises(RuntimeError, match="not connected"):
        bus.read_health()
    fake = FakeMotorsBus(_bus_values())
    bus._device = SimpleNamespace(is_connected=True)
    bus._bus = fake
    first = bus.read_health()
    second = bus.read_health()
    assert first["gripper"].source == "real"
    assert second["gripper"].limits is not None
    limit_reads = [call for call in fake.calls if call[0] in h.LIMIT_REGISTERS]
    assert len(limit_reads) == len(h.LIMIT_REGISTERS)
    assert isinstance(bus, h.HealthReadable)


def test_feetech_bus_health_survives_unreadable_limits() -> None:
    bus = FeetechBus("/dev/null", 1_000_000, name="follower")
    fake = FakeMotorsBus(_bus_values(), fail_on="Max_Temperature_Limit")
    bus._device = SimpleNamespace(is_connected=True)
    bus._bus = fake
    for _ in range(5):
        samples = bus.read_health()
    assert samples["elbow_flex"].limits is None  # fallback temperature thresholds apply
    attempts = [call for call in fake.calls if call[0] == "Max_Temperature_Limit"]
    assert len(attempts) == 3

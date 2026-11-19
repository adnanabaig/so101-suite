"""Servo health: register sampling, unit conversion, thresholds and a monitor.

What is verified and what is not
--------------------------------
Register names, addresses and sizes are pinned from LeRobot 0.6.1
(``lerobot/motors/feetech/tables.py``, ``STS_SMS_SERIES_CONTROL_TABLE``); see
``so101.bringup.lerobot_surface`` and ``tests/test_bringup_lerobot061_pin.py``.
LeRobot's public ``sync_read`` decodes the sign bit of ``Present_Load`` (bit 10
in ``STS_SMS_SERIES_ENCODINGS_TABLE``), so raw load values arrive signed.

LeRobot does not define physical units for temperature, voltage, load, current
or the meaning of the ``Status`` bits. The scales below are the commonly quoted
STS-series values. They are labeled ``unverified`` in :data:`UNIT_PROVENANCE`,
in API payloads and in docs/bring-up.md until someone checks them against a
Feetech STS3215 manual. Every threshold here is a suite default, not a Feetech
specification.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import JOINT_NAMES
from ..logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Settings

log = get_logger(__name__)

HealthLevel = Literal["ok", "warn", "critical"]
HealthSource = Literal["real", "mock", "sim"]
SupplyProfile = Literal["7.4v", "12v"]

LEVEL_ORDER: dict[str, int] = {"ok": 0, "warn": 1, "critical": 2}

#: Registers sampled on every health tick (names verified in LeRobot 0.6.1).
HEALTH_REGISTERS: tuple[str, ...] = (
    "Present_Temperature",
    "Present_Voltage",
    "Present_Load",
    "Present_Current",
    "Status",
)
#: EEPROM limit registers read once per connection (names verified in 0.6.1).
LIMIT_REGISTERS: tuple[str, ...] = (
    "Max_Temperature_Limit",
    "Max_Voltage_Limit",
    "Min_Voltage_Limit",
)

#: SO-101 motor IDs (LeRobot 0.6.1 so_follower.py:54-59, so_leader.py:46-51).
JOINT_IDS: dict[str, int] = {name: index + 1 for index, name in enumerate(JOINT_NAMES)}

# Unit scales. UNVERIFIED: not stated by LeRobot 0.6.1; see UNIT_PROVENANCE.
TEMPERATURE_C_PER_LSB = 1.0
VOLTAGE_LSB_PER_V = 10  # 0.1 V per LSB
LOAD_LSB_PER_PCT = 10  # 0.1 % of maximum torque per LSB
CURRENT_MA_PER_LSB = 6.5

#: Status register bit names. UNVERIFIED (commonly quoted STS-series meaning).
STATUS_BITS: dict[int, str] = {
    0: "voltage",
    1: "sensor",
    2: "temperature",
    3: "current",
    4: "angle",
    5: "overload",
}

#: A Max_Temperature_Limit read-back outside this band is treated as unknown.
PLAUSIBLE_TEMPERATURE_LIMIT_C = (40.0, 100.0)

_LEROBOT_TABLE = "LeRobot 0.6.1 lerobot/motors/feetech/tables.py STS_SMS_SERIES_CONTROL_TABLE"
_UNVERIFIED_SCALE = (
    "Scale is the commonly quoted STS-series value; LeRobot 0.6.1 does not state "
    "it and it has not been checked against a Feetech STS3215 manual."
)


class UnitProvenance(BaseModel):
    """Where a register's address and unit come from."""

    model_config = ConfigDict(frozen=True)

    register_name: str
    address: int
    size: int
    unit: str
    scale: str
    status: Literal["verified", "unverified"]
    source: str


UNIT_PROVENANCE: dict[str, UnitProvenance] = {
    "Present_Temperature": UnitProvenance(
        register_name="Present_Temperature",
        address=63,
        size=1,
        unit="degC",
        scale="1 degC per LSB",
        status="unverified",
        source=f"Address/size: {_LEROBOT_TABLE}. {_UNVERIFIED_SCALE}",
    ),
    "Present_Voltage": UnitProvenance(
        register_name="Present_Voltage",
        address=62,
        size=1,
        unit="V",
        scale="0.1 V per LSB",
        status="unverified",
        source=f"Address/size: {_LEROBOT_TABLE}. {_UNVERIFIED_SCALE}",
    ),
    "Present_Load": UnitProvenance(
        register_name="Present_Load",
        address=60,
        size=2,
        unit="% of max torque",
        scale="0.1 % per LSB, signed",
        status="unverified",
        source=(
            f"Address/size: {_LEROBOT_TABLE}. Sign bit 10 is verified "
            "(STS_SMS_SERIES_ENCODINGS_TABLE; LeRobot decodes it in sync_read). "
            "Which rotation direction is positive is not documented by LeRobot. "
            f"{_UNVERIFIED_SCALE}"
        ),
    ),
    "Present_Current": UnitProvenance(
        register_name="Present_Current",
        address=69,
        size=2,
        unit="mA",
        scale="6.5 mA per LSB, magnitude",
        status="unverified",
        source=(
            f"Address/size: {_LEROBOT_TABLE}. LeRobot 0.6.1 does not sign-decode this "
            f"register, so it is treated as a magnitude. {_UNVERIFIED_SCALE}"
        ),
    ),
    "Status": UnitProvenance(
        register_name="Status",
        address=65,
        size=1,
        unit="bit flags",
        scale="bit0 voltage, bit1 sensor, bit2 temperature, bit3 current, bit4 angle, "
        "bit5 overload",
        status="unverified",
        source=(
            f"Address/size: {_LEROBOT_TABLE}. Bit meanings are the commonly quoted "
            "STS-series ones and are not stated by LeRobot 0.6.1."
        ),
    ),
    "Max_Temperature_Limit": UnitProvenance(
        register_name="Max_Temperature_Limit",
        address=13,
        size=1,
        unit="degC",
        scale="1 degC per LSB",
        status="unverified",
        source=f"Address/size: {_LEROBOT_TABLE}. {_UNVERIFIED_SCALE}",
    ),
    "Max_Voltage_Limit": UnitProvenance(
        register_name="Max_Voltage_Limit",
        address=14,
        size=1,
        unit="V",
        scale="0.1 V per LSB",
        status="unverified",
        source=f"Address/size: {_LEROBOT_TABLE}. {_UNVERIFIED_SCALE}",
    ),
    "Min_Voltage_Limit": UnitProvenance(
        register_name="Min_Voltage_Limit",
        address=15,
        size=1,
        unit="V",
        scale="0.1 V per LSB",
        status="unverified",
        source=f"Address/size: {_LEROBOT_TABLE}. {_UNVERIFIED_SCALE}",
    ),
}


# --- conversions -----------------------------------------------------------


def convert_temperature(raw: int) -> float:
    """Present_Temperature / Max_Temperature_Limit LSBs to degC (unverified)."""
    return float(raw) * TEMPERATURE_C_PER_LSB


def convert_voltage(raw: int) -> float:
    """Present_Voltage LSBs to volts (unverified 0.1 V/LSB).

    Division keeps exact decimal boundaries (45 -> 4.5) for threshold tests.
    """
    return raw / VOLTAGE_LSB_PER_V


def convert_load(raw: int) -> float:
    """Signed Present_Load LSBs to % of max torque (unverified 0.1 %/LSB)."""
    return raw / LOAD_LSB_PER_PCT


def convert_current(raw: int) -> float:
    """Present_Current LSBs to mA (unverified 6.5 mA/LSB)."""
    return raw * CURRENT_MA_PER_LSB


def decode_status(raw: int) -> tuple[str, ...]:
    """Name the set bits of the Status register (bit meanings unverified)."""
    return tuple(STATUS_BITS.get(bit, f"bit{bit}") for bit in range(8) if raw & (1 << bit))


# --- samples -----------------------------------------------------------------


class HealthLimits(BaseModel):
    """EEPROM protection limits read back from one servo."""

    model_config = ConfigDict(frozen=True)

    raw: dict[str, int]
    max_temperature_c: float | None = None
    max_voltage_v: float | None = None
    min_voltage_v: float | None = None


def limits_from_raw(raw: Mapping[str, int]) -> HealthLimits:
    def _get(name: str) -> int | None:
        value = raw.get(name)
        return None if value is None else int(value)

    max_t, max_v, min_v = (_get(name) for name in LIMIT_REGISTERS)
    return HealthLimits(
        raw={name: int(value) for name, value in raw.items()},
        max_temperature_c=None if max_t is None else convert_temperature(max_t),
        max_voltage_v=None if max_v is None else convert_voltage(max_v),
        min_voltage_v=None if min_v is None else convert_voltage(min_v),
    )


class ServoHealthSample(BaseModel):
    """One servo's health registers, raw and converted."""

    model_config = ConfigDict(frozen=True)

    motor: str
    motor_id: int | None = None
    source: HealthSource
    timestamp: float = Field(description="Unix time (s) at which the registers were read")
    raw: dict[str, int] = Field(description="Register values as LeRobot returns them")
    temperature_c: float
    voltage_v: float
    load_pct: float = Field(description="Signed % of maximum torque (unverified scale)")
    current_ma: float
    status_raw: int
    status_flags: tuple[str, ...] = ()
    limits: HealthLimits | None = None


def sample_from_raw(
    motor: str,
    raw: Mapping[str, int],
    *,
    source: HealthSource,
    timestamp: float,
    motor_id: int | None = None,
    limits: HealthLimits | None = None,
) -> ServoHealthSample:
    """Convert raw register values (already sign-decoded) into a sample."""
    missing = [name for name in HEALTH_REGISTERS if name not in raw]
    if missing:
        raise ValueError(f"health sample for {motor!r} is missing registers: {missing}")
    values = {name: int(raw[name]) for name in HEALTH_REGISTERS}
    return ServoHealthSample(
        motor=motor,
        motor_id=motor_id,
        source=source,
        timestamp=float(timestamp),
        raw=values,
        temperature_c=convert_temperature(values["Present_Temperature"]),
        voltage_v=convert_voltage(values["Present_Voltage"]),
        load_pct=convert_load(values["Present_Load"]),
        current_ma=convert_current(values["Present_Current"]),
        status_raw=values["Status"],
        status_flags=decode_status(values["Status"]),
        limits=limits,
    )


def sample_from_physical(
    motor: str,
    *,
    source: HealthSource,
    timestamp: float,
    temperature_c: float,
    voltage_v: float,
    load_pct: float,
    current_ma: float,
    status: int = 0,
    motor_id: int | None = None,
    limits: HealthLimits | None = None,
) -> ServoHealthSample:
    """Build a synthetic sample (mock or sim) through the same raw path.

    Values are quantized to register resolution so synthetic and real samples
    look identical to every consumer.
    """
    raw = {
        "Present_Temperature": int(round(temperature_c / TEMPERATURE_C_PER_LSB)),
        "Present_Voltage": int(round(voltage_v * VOLTAGE_LSB_PER_V)),
        "Present_Load": int(round(max(-100.0, min(100.0, load_pct)) * LOAD_LSB_PER_PCT)),
        "Present_Current": int(round(max(0.0, current_ma) / CURRENT_MA_PER_LSB)),
        "Status": int(status) & 0xFF,
    }
    return sample_from_raw(
        motor, raw, source=source, timestamp=timestamp, motor_id=motor_id, limits=limits
    )


def read_motors_bus_health(
    bus: Any,
    *,
    source: HealthSource = "real",
    limits: Mapping[str, HealthLimits] | None = None,
    motors: Sequence[str] | None = None,
    num_retry: int = 0,
    clock: Callable[[], float] = time.time,
) -> dict[str, ServoHealthSample]:
    """Sync-read the health registers from a LeRobot 0.6.1 ``MotorsBus``.

    Uses only the public ``sync_read(data_name, motors, *, normalize, num_retry)``
    API (motors_bus.py:1127-1171). The first failing register aborts the tick so
    a dead bus holds the caller's I/O lock for at most one packet timeout.
    """
    names = list(motors) if motors is not None else list(bus.motors)
    values: dict[str, Mapping[str, Any]] = {}
    for register in HEALTH_REGISTERS:
        values[register] = bus.sync_read(register, names, normalize=False, num_retry=num_retry)
    timestamp = clock()
    known_limits = limits or {}
    samples: dict[str, ServoHealthSample] = {}
    for name in names:
        motor = bus.motors.get(name) if hasattr(bus, "motors") else None
        samples[name] = sample_from_raw(
            name,
            {register: int(values[register][name]) for register in HEALTH_REGISTERS},
            source=source,
            timestamp=timestamp,
            motor_id=getattr(motor, "id", None),
            limits=known_limits.get(name),
        )
    return samples


def read_motors_bus_limits(
    bus: Any, *, motors: Sequence[str] | None = None, num_retry: int = 1
) -> dict[str, HealthLimits]:
    """Read the EEPROM protection limits through the public ``sync_read``."""
    names = list(motors) if motors is not None else list(bus.motors)
    values = {
        register: bus.sync_read(register, names, normalize=False, num_retry=num_retry)
        for register in LIMIT_REGISTERS
    }
    return {
        name: limits_from_raw({register: int(values[register][name]) for register in values})
        for name in names
    }


@runtime_checkable
class HealthReadable(Protocol):
    """A bus that can report per-servo health (real, mock or sim)."""

    def read_health(self) -> dict[str, ServoHealthSample]: ...


# --- thresholds and evaluation ---------------------------------------------


class TemperatureThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    warn_c: float
    critical_c: float
    basis: str


class HealthThresholds(BaseModel):
    """Warn/critical thresholds. All values are suite defaults (configurable)."""

    model_config = ConfigDict(frozen=True)

    arm: str = "follower"
    supply_profile: SupplyProfile = "7.4v"
    voltage_band: tuple[float, float, float, float] = (4.0, 4.5, 8.0, 8.5)
    temp_warn_margin_c: float = 15.0
    temp_critical_margin_c: float = 5.0
    temp_fallback_warn_c: float = 55.0
    temp_fallback_critical_c: float = 65.0
    load_warn_pct: float | None = 80.0
    load_critical_pct: float | None = None
    current_warn_ma: float | None = 2000.0
    current_critical_ma: float | None = None
    status_nonzero_level: HealthLevel = "warn"
    max_consecutive_failures: int = 3

    @model_validator(mode="after")
    def _check_order(self) -> HealthThresholds:
        band = self.voltage_band
        if any(not math.isfinite(v) for v in band) or not band[0] < band[1] < band[2] < band[3]:
            raise ValueError("voltage band must be strictly increasing")
        if not 0 <= self.temp_critical_margin_c < self.temp_warn_margin_c:
            raise ValueError("temperature margins need 0 <= critical margin < warn margin")
        if not self.temp_fallback_warn_c < self.temp_fallback_critical_c:
            raise ValueError("fallback temperature warn must be below critical")
        for warn, critical, label in (
            (self.load_warn_pct, self.load_critical_pct, "load"),
            (self.current_warn_ma, self.current_critical_ma, "current"),
        ):
            if warn is not None and critical is not None and not warn < critical:
                raise ValueError(f"{label} warn threshold must be below critical")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")
        return self

    @classmethod
    def from_settings(cls, settings: Settings, arm: str = "follower") -> HealthThresholds:
        profile: SupplyProfile = (
            settings.health_leader_supply if arm == "leader" else settings.health_follower_supply
        )
        band = (
            settings.health_supply_band_12v if profile == "12v" else settings.health_supply_band_7v4
        )
        return cls(
            arm=arm,
            supply_profile=profile,
            voltage_band=band,
            temp_warn_margin_c=settings.health_temp_warn_margin_c,
            temp_critical_margin_c=settings.health_temp_critical_margin_c,
            temp_fallback_warn_c=settings.health_temp_fallback_warn_c,
            temp_fallback_critical_c=settings.health_temp_fallback_critical_c,
            load_warn_pct=settings.health_load_warn_pct,
            load_critical_pct=settings.health_load_critical_pct,
            current_warn_ma=settings.health_current_warn_ma,
            current_critical_ma=settings.health_current_critical_ma,
            status_nonzero_level=settings.health_status_nonzero_level,
            max_consecutive_failures=settings.health_max_consecutive_failures,
        )

    def temperature_thresholds(self, limit_c: float | None) -> TemperatureThresholds:
        low, high = PLAUSIBLE_TEMPERATURE_LIMIT_C
        if limit_c is not None and math.isfinite(limit_c) and low <= limit_c <= high:
            return TemperatureThresholds(
                warn_c=limit_c - self.temp_warn_margin_c,
                critical_c=limit_c - self.temp_critical_margin_c,
                basis=f"Max_Temperature_Limit {limit_c:g} degC (unit unverified)",
            )
        basis = "fallback (limit unknown)"
        if limit_c is not None:
            basis = f"fallback (implausible Max_Temperature_Limit {limit_c:g})"
        return TemperatureThresholds(
            warn_c=self.temp_fallback_warn_c,
            critical_c=self.temp_fallback_critical_c,
            basis=basis,
        )


ReasonCategory = Literal["temperature", "voltage", "load", "current", "status", "read", "samples"]


class HealthReason(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: ReasonCategory
    level: HealthLevel
    message: str
    motor: str | None = None
    value: float | None = None
    threshold: float | None = None


class MotorHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    motor: str
    level: HealthLevel
    reasons: tuple[HealthReason, ...] = ()


class HealthEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: HealthLevel
    motors: dict[str, MotorHealth] = Field(default_factory=dict)
    reasons: tuple[HealthReason, ...] = ()
    evaluated_at: float


def worst_level(levels: Sequence[str]) -> HealthLevel:
    worst: HealthLevel = "ok"
    for level in levels:
        if LEVEL_ORDER[level] > LEVEL_ORDER[worst]:
            worst = level  # type: ignore[assignment]
    return worst


def _graded(
    value: float, warn: float | None, critical: float | None
) -> tuple[HealthLevel, float | None]:
    if critical is not None and value >= critical:
        return "critical", critical
    if warn is not None and value >= warn:
        return "warn", warn
    return "ok", None


def grade_temperature(
    temperature_c: float, limit_c: float | None, thresholds: HealthThresholds
) -> tuple[HealthLevel, float | None, TemperatureThresholds]:
    """Grade one temperature reading against the limit-derived thresholds."""
    temp = thresholds.temperature_thresholds(limit_c)
    level, threshold = _graded(temperature_c, temp.warn_c, temp.critical_c)
    return level, threshold, temp


def grade_voltage(
    volts: float, thresholds: HealthThresholds
) -> tuple[HealthLevel, float | None, str]:
    """Grade one supply reading: (level, threshold crossed, relation). Inclusive."""
    critical_low, warn_low, warn_high, critical_high = thresholds.voltage_band
    if volts <= critical_low:
        return "critical", critical_low, "at or below"
    if volts >= critical_high:
        return "critical", critical_high, "at or above"
    if volts <= warn_low:
        return "warn", warn_low, "at or below"
    if volts >= warn_high:
        return "warn", warn_high, "at or above"
    return "ok", None, "within"


def evaluate_sample(sample: ServoHealthSample, thresholds: HealthThresholds) -> MotorHealth:
    """Grade one servo. Boundaries are inclusive: reaching a threshold trips it."""
    motor = sample.motor
    reasons: list[HealthReason] = []

    limit_c = sample.limits.max_temperature_c if sample.limits is not None else None
    level, threshold, temp = grade_temperature(sample.temperature_c, limit_c, thresholds)
    if level != "ok":
        reasons.append(
            HealthReason(
                category="temperature",
                level=level,
                motor=motor,
                value=sample.temperature_c,
                threshold=threshold,
                message=(
                    f"{motor}: temperature {sample.temperature_c:g} degC reached the {level} "
                    f"threshold {threshold:g} degC ({temp.basis})"
                ),
            )
        )

    volts = sample.voltage_v
    v_level, v_threshold, relation = grade_voltage(volts, thresholds)
    if v_level != "ok" and v_threshold is not None:
        reasons.append(
            HealthReason(
                category="voltage",
                level=v_level,
                motor=motor,
                value=volts,
                threshold=v_threshold,
                message=(
                    f"{motor}: supply {volts:g} V is {relation} the {v_level} threshold "
                    f"{v_threshold:g} V for the {thresholds.supply_profile} profile "
                    "(0.1 V/LSB unverified)"
                ),
            )
        )

    load = abs(sample.load_pct)
    level, threshold = _graded(load, thresholds.load_warn_pct, thresholds.load_critical_pct)
    if level != "ok":
        reasons.append(
            HealthReason(
                category="load",
                level=level,
                motor=motor,
                value=sample.load_pct,
                threshold=threshold,
                message=(
                    f"{motor}: load {sample.load_pct:g}% of max torque reached the {level} "
                    f"threshold {threshold:g}% (scale unverified)"
                ),
            )
        )

    level, threshold = _graded(
        sample.current_ma, thresholds.current_warn_ma, thresholds.current_critical_ma
    )
    if level != "ok":
        reasons.append(
            HealthReason(
                category="current",
                level=level,
                motor=motor,
                value=sample.current_ma,
                threshold=threshold,
                message=(
                    f"{motor}: current {sample.current_ma:g} mA reached the {level} threshold "
                    f"{threshold:g} mA (6.5 mA/LSB unverified)"
                ),
            )
        )

    if sample.status_raw and thresholds.status_nonzero_level != "ok":
        flags = ", ".join(sample.status_flags) or "unknown"
        reasons.append(
            HealthReason(
                category="status",
                level=thresholds.status_nonzero_level,
                motor=motor,
                value=float(sample.status_raw),
                message=(
                    f"{motor}: servo status register 0x{sample.status_raw:02X} ({flags}; "
                    "bit meanings unverified)"
                ),
            )
        )

    return MotorHealth(
        motor=motor,
        level=worst_level([reason.level for reason in reasons]),
        reasons=tuple(reasons),
    )


def evaluate(
    samples: Mapping[str, ServoHealthSample],
    thresholds: HealthThresholds,
    *,
    now: float | None = None,
) -> HealthEvaluation:
    """Grade every servo and roll up the worst level."""
    evaluated_at = time.time() if now is None else now
    if not samples:
        reason = HealthReason(category="samples", level="warn", message="no health samples")
        return HealthEvaluation(level="warn", reasons=(reason,), evaluated_at=evaluated_at)
    motors = {name: evaluate_sample(sample, thresholds) for name, sample in samples.items()}
    reasons = tuple(reason for health in motors.values() for reason in health.reasons)
    return HealthEvaluation(
        level=worst_level([health.level for health in motors.values()]),
        motors=motors,
        reasons=reasons,
        evaluated_at=evaluated_at,
    )


# --- monitor -----------------------------------------------------------------


class HealthSnapshot(BaseModel):
    """The latest monitor result: samples, their grade and read status."""

    model_config = ConfigDict(frozen=True)

    arm: str
    source: HealthSource | None = None
    taken_at: float
    samples: dict[str, ServoHealthSample] = Field(default_factory=dict)
    evaluation: HealthEvaluation
    error: str | None = None
    consecutive_failures: int = 0
    previous_level: HealthLevel | None = None


class LockLike(Protocol):
    def acquire(self, blocking: bool = ..., timeout: float = ...) -> bool: ...
    def release(self) -> None: ...


SnapshotHandler = Callable[[HealthSnapshot], None]


class ServoHealthMonitor:
    """Poll ``read_fn`` at 1-2 Hz without ever blocking the control loop.

    * ``lock`` is the bus I/O lock shared with control traffic. Each tick tries
      to acquire it for at most ``lock_timeout_s`` and skips the tick when the
      bus is busy. The whole tick (all register reads) runs under the lock.
    * Escalation hooks are generic: ``on_critical`` handlers run on every
      critical tick (level-triggered, so a re-armed but still-overheated arm is
      stopped again; handlers must be idempotent). ``on_warn`` handlers run when
      the level changes to warn. Handlers run after the lock is released, on the
      monitor thread; exceptions are logged and counted, never propagated.
    * One failed read grades the tick ``warn``; ``max_consecutive_failures`` in
      a row grade it ``critical``.
    """

    def __init__(
        self,
        read_fn: Callable[[], Mapping[str, ServoHealthSample]],
        *,
        thresholds: HealthThresholds | None = None,
        lock: LockLike | None = None,
        poll_hz: float = 1.0,
        lock_timeout_s: float = 0.05,
        on_warn: SnapshotHandler | None = None,
        on_critical: SnapshotHandler | None = None,
        clock: Callable[[], float] = time.time,
        name: str = "follower",
    ) -> None:
        if not math.isfinite(poll_hz) or not 1.0 <= poll_hz <= 2.0:
            raise ValueError("poll_hz must be between 1 and 2")
        if not math.isfinite(lock_timeout_s) or lock_timeout_s < 0:
            raise ValueError("lock_timeout_s must be finite and non-negative")
        self._read_fn = read_fn
        self._thresholds = thresholds or HealthThresholds(arm=name)
        self._lock = lock
        self._poll_hz = poll_hz
        self._lock_timeout_s = lock_timeout_s
        self._clock = clock
        self.name = name
        self._warn_handlers: list[SnapshotHandler] = [on_warn] if on_warn else []
        self._critical_handlers: list[SnapshotHandler] = [on_critical] if on_critical else []
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: HealthSnapshot | None = None
        self._level: HealthLevel | None = None
        self._consecutive_failures = 0
        self._counters = {"polls": 0, "skipped_busy": 0, "read_failures": 0, "callback_errors": 0}

    # --- escalation hooks -------------------------------------------------
    def on_warn(self, callback: SnapshotHandler) -> SnapshotHandler:
        """Register another warn handler (usable as a decorator)."""
        with self._state_lock:
            self._warn_handlers.append(callback)
        return callback

    def on_critical(self, callback: SnapshotHandler) -> SnapshotHandler:
        """Register another critical handler (usable as a decorator)."""
        with self._state_lock:
            self._critical_handlers.append(callback)
        return callback

    # --- state ------------------------------------------------------------
    @property
    def thresholds(self) -> HealthThresholds:
        return self._thresholds

    @property
    def poll_period_s(self) -> float:
        return 1.0 / self._poll_hz

    @property
    def latest(self) -> HealthSnapshot | None:
        with self._state_lock:
            return self._latest

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop_event.is_set()

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                **self._counters,
                "consecutive_failures": self._consecutive_failures,
                "level": self._level,
                "running": self.running,
                "poll_hz": self._poll_hz,
            }

    # --- polling ----------------------------------------------------------
    def poll_once(self) -> HealthSnapshot | None:
        """Run one tick. Returns ``None`` when the bus lock was busy."""
        lock = self._lock
        if lock is not None and not lock.acquire(timeout=self._lock_timeout_s):
            with self._state_lock:
                self._counters["skipped_busy"] += 1
            return None
        samples: dict[str, ServoHealthSample] = {}
        error: str | None = None
        try:
            try:
                samples = dict(self._read_fn())
            except Exception as exc:  # hardware errors are data here
                error = f"{type(exc).__name__}: {exc}"
        finally:
            if lock is not None:
                lock.release()

        now = self._clock()
        with self._state_lock:
            self._counters["polls"] += 1
            previous = self._level
            if error is None:
                self._consecutive_failures = 0
                evaluation = evaluate(samples, self._thresholds, now=now)
            else:
                self._counters["read_failures"] += 1
                self._consecutive_failures += 1
                failures = self._consecutive_failures
                level: HealthLevel = (
                    "critical" if failures >= self._thresholds.max_consecutive_failures else "warn"
                )
                reason = HealthReason(
                    category="read",
                    level=level,
                    message=f"health read failed ({failures} consecutive): {error}",
                )
                evaluation = HealthEvaluation(level=level, reasons=(reason,), evaluated_at=now)
            sources = {sample.source for sample in samples.values()}
            snapshot = HealthSnapshot(
                arm=self.name,
                source=sources.pop() if len(sources) == 1 else None,
                taken_at=now,
                samples=samples,
                evaluation=evaluation,
                error=error,
                consecutive_failures=self._consecutive_failures,
                previous_level=previous,
            )
            self._latest = snapshot
            self._level = evaluation.level
            if evaluation.level == "critical":
                handlers = list(self._critical_handlers)
            elif evaluation.level == "warn" and previous != "warn":
                handlers = list(self._warn_handlers)
            else:
                handlers = []

        for handler in handlers:
            try:
                handler(snapshot)
            except Exception:
                with self._state_lock:
                    self._counters["callback_errors"] += 1
                log.exception("servo health %s handler failed", evaluation.level)
        return snapshot

    def _run(self, stop_event: threading.Event) -> None:
        period = self.poll_period_s
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                self.poll_once()
            except Exception:  # pragma: no cover - poll_once already guards
                log.exception("servo health tick failed")
            # No burst catch-up after a slow tick: wait out the remainder only.
            if stop_event.wait(max(0.0, period - (time.monotonic() - started))):
                break

    # --- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        """Start polling. Idempotent: returns False when already running."""
        with self._lifecycle_lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                if not self._stop_event.is_set():
                    return False
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError("previous health monitor thread is still stopping")
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(self._stop_event,),
                name=f"so101-health-{self.name}",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop polling. Idempotent and safe to call from a handler."""
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread
            if thread is None:
                return True
            if thread is threading.current_thread():
                return True  # called from a handler: the loop exits after it returns
            thread.join(timeout=timeout)
            return not thread.is_alive()


# --- synthetic model (mock backend) -----------------------------------------


class MockHealthModel:
    """Plausible synthetic health for ``MockBus``. Every sample is ``source="mock"``.

    Temperature drifts up slowly with commanded motion and holding load and
    cools toward ambient; load follows tracking error and a gravity term while
    torque is on. None of these numbers describe a real STS3215.
    """

    AMBIENT_C = 28.0
    # Per degree of commanded goal travel. 0.004 keeps hours of mock teleop
    # (the synthetic leader sweeps the gripper ~29 units/s) near 42 degC, well
    # under the 55 degC warn, so the always-on server monitor never raises a
    # false warning in a rehearsal (tests/test_robot_health.py).
    HEAT_PER_DEG_C = 0.004
    HEAT_PER_LOAD_PCT_S = 0.0005
    COOL_TAU_S = 120.0
    GRAVITY_PCT = {"shoulder_lift": 18.0, "elbow_flex": 10.0, "wrist_flex": 3.0}

    def __init__(
        self,
        *,
        supply_profile: SupplyProfile = "7.4v",
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.supply_profile = supply_profile
        self._clock = clock
        self._wall_clock = wall_clock
        self._last_t: float | None = None
        self._last_goal: dict[str, float] = {}
        self._temperature = {joint: self.AMBIENT_C for joint in JOINT_NAMES}
        self._overrides: dict[str, dict[str, float]] = {}
        self._fail_reads = 0
        self._lock = threading.Lock()

    @property
    def nominal_voltage_v(self) -> float:
        # The standard SO-101 kit powers the 7.4 V servo variant from 5 V
        # (SO-ARM100 README); the 12 V variant runs from 12 V.
        return 12.0 if self.supply_profile == "12v" else 5.0

    def limits(self) -> dict[str, HealthLimits]:
        max_v, min_v = (140, 95) if self.supply_profile == "12v" else (80, 40)
        raw = {"Max_Temperature_Limit": 70, "Max_Voltage_Limit": max_v, "Min_Voltage_Limit": min_v}
        return {joint: limits_from_raw(raw) for joint in JOINT_NAMES}

    def set_override(
        self,
        motor: str,
        *,
        temperature_c: float | None = None,
        voltage_v: float | None = None,
        load_pct: float | None = None,
        current_ma: float | None = None,
        status: int | None = None,
    ) -> None:
        """Pin synthetic values for fault-injection tests and rehearsals."""
        if motor not in JOINT_NAMES:
            raise ValueError(f"unknown motor {motor!r}")
        values = {
            "temperature_c": temperature_c,
            "voltage_v": voltage_v,
            "load_pct": load_pct,
            "current_ma": current_ma,
            "status": None if status is None else float(status),
        }
        with self._lock:
            entry = self._overrides.setdefault(motor, {})
            entry.update({key: float(value) for key, value in values.items() if value is not None})

    def fail_next_reads(self, count: int) -> None:
        with self._lock:
            self._fail_reads = max(0, int(count))

    def clear_overrides(self) -> None:
        with self._lock:
            self._overrides.clear()
            self._fail_reads = 0

    def sample(
        self,
        positions: Mapping[str, float],
        goals: Mapping[str, float],
        *,
        torque: bool,
    ) -> dict[str, ServoHealthSample]:
        with self._lock:
            if self._fail_reads > 0:
                self._fail_reads -= 1
                raise RuntimeError("injected mock health read failure")
            now = self._clock()
            dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
            self._last_t = now
            decay = math.exp(-dt / self.COOL_TAU_S)
            loads: dict[str, float] = {}
            for joint in JOINT_NAMES:
                goal = float(goals.get(joint, 0.0))
                position = float(positions.get(joint, 0.0))
                motion = abs(goal - self._last_goal.get(joint, goal))
                self._last_goal[joint] = goal
                load = 0.0
                if torque:
                    load = 0.8 * (goal - position) + self.GRAVITY_PCT.get(joint, 0.0) * math.cos(
                        math.radians(position)
                    )
                load = max(-100.0, min(100.0, load))
                loads[joint] = load
                # Motion adds an impulse; holding load and cooling follow the exact
                # solution of dT/dt = q - (T - ambient) / tau, stable for any gap.
                temperature = self._temperature[joint] + motion * self.HEAT_PER_DEG_C
                equilibrium = (
                    self.AMBIENT_C + abs(load) * self.HEAT_PER_LOAD_PCT_S * self.COOL_TAU_S
                )
                temperature = equilibrium + (temperature - equilibrium) * decay
                self._temperature[joint] = temperature
            sag = 0.002 * sum(abs(load) for load in loads.values())
            voltage = max(0.0, self.nominal_voltage_v - sag)
            timestamp = self._wall_clock()
            limits = self.limits()
            samples: dict[str, ServoHealthSample] = {}
            for joint in JOINT_NAMES:
                override = self._overrides.get(joint, {})
                load = override.get("load_pct", loads[joint])
                idle_ma = 20.0 if torque else 5.0
                samples[joint] = sample_from_physical(
                    joint,
                    source="mock",
                    timestamp=timestamp,
                    temperature_c=override.get("temperature_c", self._temperature[joint]),
                    voltage_v=override.get("voltage_v", voltage),
                    load_pct=load,
                    current_ma=override.get("current_ma", idle_ma + 10.0 * abs(load)),
                    status=int(override.get("status", 0)),
                    motor_id=JOINT_IDS[joint],
                    limits=limits[joint],
                )
            return samples

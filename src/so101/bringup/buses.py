"""Read-only access to an SO-101 Feetech bus for diagnostics.

The real path opens a bare LeRobot 0.6.1 ``FeetechMotorsBus`` with
``connect(handshake=False)`` so a bus with a missing motor or mixed firmware
can still be diagnosed (the handshake raises on both: motors_bus.py:465,
feetech.py:145-157). It only pings and reads; it never writes a register,
never changes torque and closes with ``disconnect(disable_torque=False)``
(motors_bus.py:547-563), which only closes the port.

:class:`FakeFeetechBus` is the rehearsal stand-in. It has the same read API,
no write methods at all, values that are plainly synthetic (firmware "0.0"),
and fault-injection knobs for tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Protocol

from .. import JOINT_NAMES
from ..hardware.health import JOINT_IDS
from .lerobot_surface import (
    DEFAULT_BAUDRATE,
    STS3215_MODEL,
    STS3215_MODEL_NUMBER,
    STS3215_RESOLUTION,
)
from .ownership import require_device_lock

#: Registers read once per diagnostic (all exist in tables.py:41-101).
SNAPSHOT_REGISTERS: tuple[str, ...] = (
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Homing_Offset",
    "Max_Temperature_Limit",
    "Max_Voltage_Limit",
    "Min_Voltage_Limit",
    "Torque_Enable",
    "Present_Voltage",
    "Present_Temperature",
    "Present_Load",
    "Present_Current",
    "Status",
)
POSITION_SAMPLES = 10
POSITION_INTERVAL_S = 0.02
READ_RETRIES = 1


class RegisterBus(Protocol):
    """The subset of ``FeetechMotorsBus`` the diagnostics use (all reads)."""

    @property
    def motors(self) -> Mapping[str, Any]: ...

    @property
    def is_connected(self) -> bool: ...

    def disconnect(self, disable_torque: bool = True) -> None: ...

    def ping(self, motor: Any, num_retry: int = 0, raise_on_error: bool = False) -> int | None: ...

    def read(
        self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0
    ) -> Any: ...

    def sync_read(
        self,
        data_name: str,
        motors: Any = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Any]: ...


def open_feetech_bus(port: str, *, baudrate: int = DEFAULT_BAUDRATE) -> RegisterBus:
    """Open the real bus read-only. Raises ImportError/ConnectionError on failure."""
    locked_port = require_device_lock(port)
    from lerobot.motors import Motor, MotorNormMode  # type: ignore[import-not-found]
    from lerobot.motors.feetech import FeetechMotorsBus  # type: ignore[import-not-found]

    motors = {
        name: Motor(
            JOINT_IDS[name],
            STS3215_MODEL,
            MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES,
        )
        for name in JOINT_NAMES
    }
    bus = FeetechMotorsBus(port=locked_port, motors=motors)
    if require_device_lock(port) != locked_port:
        raise RuntimeError("motor endpoint changed between diagnostic build and open")
    bus.connect(handshake=False)
    try:
        if require_device_lock(port) != locked_port:
            raise RuntimeError("motor endpoint changed during diagnostic open")
    except BaseException:
        bus.disconnect(disable_torque=False)
        raise
    try:
        # Host UART only (motors_bus.py:711-722); no packet goes to a motor.
        bus.set_baudrate(baudrate)
    except BaseException:
        bus.disconnect(disable_torque=False)
        raise
    return bus  # type: ignore[no-any-return]


# --- facts -------------------------------------------------------------------


@dataclass
class BusFacts:
    """Everything one diagnostic pass read from one arm's bus."""

    arm: str
    port: str
    source: str  # "real" | "mock"
    attempted: bool = False
    opened: bool = False
    open_error: str | None = None
    models: dict[str, int | None] = field(default_factory=dict)
    firmware: dict[str, str] = field(default_factory=dict)
    firmware_errors: dict[str, str] = field(default_factory=dict)
    registers: dict[str, dict[str, int]] = field(default_factory=dict)
    register_errors: dict[str, dict[str, str]] = field(default_factory=dict)
    position_samples: list[dict[str, int]] = field(default_factory=list)
    position_errors: list[str] = field(default_factory=list)
    close_error: str | None = None
    #: In-memory LeRobot calibration (rehearsal only; real runs read the file).
    synthetic_calibration: dict[str, dict[str, int]] | None = None
    duration_ms: float = 0.0

    @property
    def responding(self) -> list[str]:
        return [name for name in JOINT_NAMES if self.models.get(name) is not None]

    @property
    def missing(self) -> list[str]:
        return [name for name in JOINT_NAMES if self.models.get(name) is None]

    def register(self, name: str) -> dict[str, int]:
        return self.registers.get(name, {})

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "port": self.port,
            "source": self.source,
            "opened": self.opened,
            "open_error": self.open_error,
            "responding": self.responding,
            "missing": self.missing,
            "duration_ms": round(self.duration_ms, 1),
        }


def bus_facts_to_json(facts: BusFacts) -> dict[str, Any]:
    """JSON-safe form of ``facts`` (``POST /api/preflight/bus``)."""
    return asdict(facts)


def bus_facts_from_json(payload: Mapping[str, Any]) -> BusFacts:
    """Rebuild :class:`BusFacts` from :func:`bus_facts_to_json`; unknown keys are ignored."""
    known = {item.name for item in fields(BusFacts)}
    return BusFacts(**{key: value for key, value in payload.items() if key in known})


class BusDiagnosticOpenError(RuntimeError):
    """A lent diagnostic could not open the port; nothing was read.

    ``RobotManager.run_idle_bus_diagnostic`` raises it. A diagnostic result
    rather than a refusal: preflight reports it as the bus that did not open.
    """


def _as_int(value: Any) -> int:
    return int(value)


def _read_register(
    bus: RegisterBus, name: str, motors: Sequence[str]
) -> tuple[dict[str, int], dict[str, str]]:
    """Sync-read one register; fall back to per-motor reads to localize errors."""
    if not motors:
        return {}, {}
    try:
        values = bus.sync_read(name, list(motors), normalize=False, num_retry=READ_RETRIES)
        return {motor: _as_int(values[motor]) for motor in motors}, {}
    except Exception:
        pass
    values_by_motor: dict[str, int] = {}
    errors: dict[str, str] = {}
    for motor in motors:
        try:
            values_by_motor[motor] = _as_int(
                bus.read(name, motor, normalize=False, num_retry=READ_RETRIES)
            )
        except Exception as exc:
            errors[motor] = f"{type(exc).__name__}: {exc}"
    return values_by_motor, errors


def collect_bus_facts(
    bus: RegisterBus,
    *,
    arm: str,
    port: str,
    source: str,
    samples: int = POSITION_SAMPLES,
    interval_s: float = POSITION_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BusFacts:
    """Ping, then read firmware, snapshot registers and position samples."""
    started = clock()
    facts = BusFacts(arm=arm, port=port, source=source, attempted=True, opened=True)
    for name in JOINT_NAMES:
        try:
            model = bus.ping(name, num_retry=READ_RETRIES)
        except Exception:
            model = None
        facts.models[name] = None if model is None else int(model)
    responding = facts.responding

    for motor in responding:
        try:
            major = _as_int(
                bus.read("Firmware_Major_Version", motor, normalize=False, num_retry=READ_RETRIES)
            )
            minor = _as_int(
                bus.read("Firmware_Minor_Version", motor, normalize=False, num_retry=READ_RETRIES)
            )
            # Same formatting as feetech.py:_read_firmware_version (line 446).
            facts.firmware[motor] = f"{major}.{minor}"
        except Exception as exc:
            facts.firmware_errors[motor] = f"{type(exc).__name__}: {exc}"

    for register in SNAPSHOT_REGISTERS:
        values, errors = _read_register(bus, register, responding)
        facts.registers[register] = values
        if errors:
            facts.register_errors[register] = errors

    if responding:
        for index in range(max(0, samples)):
            if index:
                sleep(interval_s)
            try:
                values = bus.sync_read(
                    "Present_Position", responding, normalize=False, num_retry=READ_RETRIES
                )
                facts.position_samples.append({m: _as_int(values[m]) for m in responding})
            except Exception as exc:
                facts.position_errors.append(f"{type(exc).__name__}: {exc}")
    synthetic = getattr(bus, "synthetic_calibration", None)
    if callable(synthetic):
        facts.synthetic_calibration = synthetic()
    facts.duration_ms = (clock() - started) * 1000.0
    return facts


BusOpener = Callable[[str], RegisterBus]


def diagnose_port(
    port: str,
    *,
    arm: str,
    source: str,
    opener: BusOpener,
    **collect_kwargs: Any,
) -> BusFacts:
    """Open ``port``, collect facts, always close without touching torque."""
    try:
        bus = opener(port)
    except Exception as exc:
        return BusFacts(
            arm=arm,
            port=port,
            source=source,
            attempted=True,
            opened=False,
            open_error=f"{type(exc).__name__}: {exc}",
        )
    try:
        return collect_bus_facts(bus, arm=arm, port=port, source=source, **collect_kwargs)
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


# --- rehearsal bus -------------------------------------------------------------


def synthetic_calibration_for(arm: str) -> dict[str, dict[str, int]]:
    """A plausible LeRobot calibration payload (``draccus`` JSON shape)."""
    del arm  # both arms use the same synthetic ranges
    payload: dict[str, dict[str, int]] = {}
    for name in JOINT_NAMES:
        if name == "wrist_roll":
            low, high = 0, STS3215_RESOLUTION - 1  # so_follower.py:142-143
        elif name == "gripper":
            low, high = 2000, 3300
        else:
            low, high = 1024, 3072
        payload[name] = {
            "id": JOINT_IDS[name],
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": low,
            "range_max": high,
        }
    return payload


class FakeFeetechBus:
    """Synthetic STS3215 bus for rehearsals and tests. Read-only by design."""

    def __init__(
        self,
        *,
        arm: str = "follower",
        supply_profile: str = "7.4v",
        missing_ids: Sequence[int] = (),
        model_numbers: Mapping[str, int] | None = None,
        firmware: Mapping[str, tuple[int, int]] | None = None,
        registers: Mapping[str, Mapping[str, int]] | None = None,
        register_errors: Mapping[str, Sequence[str]] | None = None,
        position_noise: Mapping[str, Sequence[int]] | None = None,
        position_read_errors: int = 0,
        sync_read_fails: bool = False,
    ) -> None:
        self.arm = arm
        self.motors = {name: JOINT_IDS[name] for name in JOINT_NAMES}
        self.default_baudrate = DEFAULT_BAUDRATE
        self._connected = True
        self.calls: list[tuple[str, Any]] = []
        self.missing_ids = set(missing_ids)
        self.model_numbers = dict(model_numbers or {})
        self.firmware = {name: (0, 0) for name in JOINT_NAMES}
        self.firmware.update(firmware or {})
        self._calibration = synthetic_calibration_for(arm)
        volts_raw = 120 if supply_profile == "12v" else 50
        limit_max, limit_min = (140, 95) if supply_profile == "12v" else (80, 40)
        base: dict[str, dict[str, int]] = {
            "Min_Position_Limit": {n: c["range_min"] for n, c in self._calibration.items()},
            "Max_Position_Limit": {n: c["range_max"] for n, c in self._calibration.items()},
            "Homing_Offset": {n: c["homing_offset"] for n, c in self._calibration.items()},
            "Max_Temperature_Limit": dict.fromkeys(JOINT_NAMES, 70),
            "Max_Voltage_Limit": dict.fromkeys(JOINT_NAMES, limit_max),
            "Min_Voltage_Limit": dict.fromkeys(JOINT_NAMES, limit_min),
            "Torque_Enable": dict.fromkeys(JOINT_NAMES, 0),
            "Present_Voltage": dict.fromkeys(JOINT_NAMES, volts_raw),
            "Present_Temperature": dict.fromkeys(JOINT_NAMES, 30),
            "Present_Load": dict.fromkeys(JOINT_NAMES, 0),
            "Present_Current": dict.fromkeys(JOINT_NAMES, 1),
            "Status": dict.fromkeys(JOINT_NAMES, 0),
        }
        for register, values in (registers or {}).items():
            base.setdefault(register, {}).update(values)
        self._registers = base
        self.register_errors = {k: set(v) for k, v in (register_errors or {}).items()}
        self.position_noise = {k: list(v) for k, v in (position_noise or {}).items()}
        self.position_read_errors = position_read_errors
        self.sync_read_fails = sync_read_fails
        self._position_reads = 0

    # LeRobot-compatible surface ------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))
        self._connected = False

    def synthetic_calibration(self) -> dict[str, dict[str, int]]:
        return {name: dict(values) for name, values in self._calibration.items()}

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("FakeFeetechBus is not connected")

    def ping(self, motor: Any, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        self._require_connected()
        self.calls.append(("ping", motor))
        if JOINT_IDS[motor] in self.missing_ids:
            if raise_on_error:
                raise ConnectionError(f"[TxRxResult] There is no status packet (id {motor})")
            return None
        return self.model_numbers.get(motor, STS3215_MODEL_NUMBER)

    def _position(self, motor: str) -> int:
        calibration = self._calibration[motor]
        middle = (calibration["range_min"] + calibration["range_max"]) // 2
        noise = self.position_noise.get(motor) or [0, 1]
        return middle + noise[self._position_reads % len(noise)]

    def _value(self, data_name: str, motor: str) -> int:
        if JOINT_IDS[motor] in self.missing_ids:
            raise ConnectionError(f"[TxRxResult] There is no status packet (id {motor})")
        if motor in self.register_errors.get(data_name, set()):
            raise ConnectionError(f"Failed to read '{data_name}' on {motor} (injected)")
        if data_name == "Firmware_Major_Version":
            return self.firmware[motor][0]
        if data_name == "Firmware_Minor_Version":
            return self.firmware[motor][1]
        if data_name == "Present_Position":
            return self._position(motor)
        return int(self._registers[data_name][motor])

    def read(
        self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0
    ) -> Any:
        self._require_connected()
        self.calls.append(("read", (data_name, motor, normalize)))
        if normalize and data_name in ("Present_Position", "Goal_Position"):
            raise ValueError("FakeFeetechBus only serves raw (normalize=False) reads")
        return self._value(data_name, motor)

    def sync_read(
        self,
        data_name: str,
        motors: Any = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Any]:
        self._require_connected()
        names = list(JOINT_NAMES) if motors is None else list(motors)
        self.calls.append(("sync_read", (data_name, tuple(names), normalize)))
        if normalize and data_name in ("Present_Position", "Goal_Position"):
            raise ValueError("FakeFeetechBus only serves raw (normalize=False) reads")
        if data_name == "Present_Position":
            self._position_reads += 1
            if self._position_reads <= self.position_read_errors:
                raise ConnectionError("Failed to sync read 'Present_Position' (injected)")
        if self.sync_read_fails:
            raise ConnectionError(f"Failed to sync read '{data_name}' (injected)")
        return {name: self._value(data_name, name) for name in names}

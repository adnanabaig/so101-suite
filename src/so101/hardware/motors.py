"""Feetech bus servo abstraction.

``MotorBus`` is the protocol the rest of the platform depends on.
``FeetechBus`` drives real STS3215 servos through the pinned LeRobot SO-101
device objects.
``MockBus`` simulates a critically-damped arm only in explicit mock mode.
"""

from __future__ import annotations

import copy
import math
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from .. import JOINT_NAMES
from ..config import get_settings
from ..logging import get_logger
from .backend import current_backend
from .health import (
    HealthLimits,
    MockHealthModel,
    ServoHealthSample,
    read_motors_bus_health,
    read_motors_bus_limits,
)

log = get_logger(__name__)


@runtime_checkable
class MotorBus(Protocol):
    """Minimal interface every joint backend implements.

    ``disconnect()`` turns torque off first and always closes the port. It
    raises when that torque-off was not confirmed, so a closed port never
    means "torque off" by itself. Real and mock buses also report
    ``torque_state`` ("on", "off" or "unconfirmed"), which survives the close
    (safety-B2, docs/audit/v0.1/safety.md).
    """

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def disconnect_holding(self) -> None: ...
    @property
    def connected(self) -> bool: ...
    def read_positions(self) -> dict[str, float]: ...
    def write_positions(self, goal: dict[str, float]) -> dict[str, float]: ...
    def set_torque(self, enabled: bool) -> None: ...


class HardwareUnavailableError(RuntimeError):
    """Raised when real hardware was requested but its driver is unavailable."""


TorqueState = Literal["on", "off", "unconfirmed"]

# Retries for each acknowledged per-motor torque-off write. LeRobot 0.6.1
# waits about 50 ms per try for a status packet.
TORQUE_OFF_RETRIES = 2


class TorqueOffUnconfirmed(RuntimeError):
    """Torque-off was not confirmed on some motors (safety-B2).

    Raised only after every motor was tried. ``motors`` names the ones that
    may still hold torque. From ``disconnect()`` the port is already closed.
    """

    def __init__(self, bus_name: str, motors: list[str]) -> None:
        self.bus_name = bus_name
        self.motors = list(motors)
        super().__init__(
            f"{bus_name}: torque-off unconfirmed on {', '.join(self.motors)}; "
            "cut physical power if they still hold"
        )


class MockBus:
    """Physics-free arm sim: positions ease toward the last goal.

    It keeps the bus semantics the safety code depends on, so tests exercise
    real failure behavior (docs/audit/v0.1/safety.md §2 item 7):

    - I/O while disconnected raises, as on a real port.
    - Goal RAM accepts writes with torque off, but the arm only moves while
      torque is on.
    - ``disconnect()`` disables torque first and always closes the port, like
      ``FeetechBus``. When that torque-off fails it raises, the servos keep
      their torque and ``torque_state`` stays "unconfirmed".

    ``fail_torque_off``, ``fail_io`` and ``fail_connect`` are fault-injection
    knobs for tests; ``torque_off_attempts`` counts torque-off passes.
    """

    def __init__(self, name: str = "mock", smoothing: float = 0.35) -> None:
        self.name = name
        self.smoothing = smoothing
        self._pos = {j: 0.0 for j in JOINT_NAMES}
        self._goal = dict(self._pos)
        self._connected = False
        self._torque = False
        self._t0 = time.monotonic()
        self.fail_torque_off = False  # torque-off writes get no status packet
        self.fail_io = False  # the whole bus stops answering
        self.fail_connect = False  # the port does not open (adapter unplugged)
        self.torque_off_attempts = 0
        self._torque_state: TorqueState = "off"

    def connect(self) -> None:
        if self.fail_connect:
            raise ConnectionError(f"MockBus[{self.name}]: could not open the port")
        # Opening a port proves nothing about torque: torque_state is kept.
        self._connected = True
        log.info("MockBus[%s] connected", self.name)

    def disconnect(self) -> None:
        """Torque off with read-back, then always close the port.

        Raises ``TorqueOffUnconfirmed`` with the port closed when the
        torque-off fails; the servos keep their torque.
        """
        if not self._connected:
            return
        confirmed = self.torque_off_all()
        self._connected = False
        if not confirmed:
            raise TorqueOffUnconfirmed(self.name, list(JOINT_NAMES))

    def disconnect_holding(self) -> None:
        """Close the mock host connection without changing an energized hold."""

        if not self._connected:
            raise RuntimeError(f"MockBus[{self.name}] is not connected")
        if self._torque_state != "on" or not self._torque:
            raise RuntimeError(f"MockBus[{self.name}] is not confirmed holding")
        self._connected = False

    def release_port(self) -> None:
        """Drop the host-side connection with no torque write. Never raises.

        Only for use after physical power was cut, so the host treats torque
        as off from here on. The simulated servos are left as they are.
        """
        self._connected = False
        self._torque_state = "off"

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def torque_enabled(self) -> bool:
        """Servo-side Torque_Enable. It survives a host-side disconnect."""
        return self._torque

    @property
    def torque_state(self) -> TorqueState:
        """What the host knows about torque: "on", "off" or "unconfirmed"."""
        return self._torque_state

    def torque_off_all(self) -> bool:
        """Broadcast torque-off, then read back. Never raises."""
        self.torque_off_attempts += 1
        if not self._connected:
            return False
        if self.fail_io or self.fail_torque_off:
            self._torque_state = "unconfirmed"
            return False
        self._torque = False
        self._torque_state = "off"
        return True

    def _check_io(self) -> None:
        if not self._connected:
            raise RuntimeError(f"MockBus[{self.name}] is not connected")
        if self.fail_io:
            raise ConnectionError(f"MockBus[{self.name}]: no status packet")

    def _integrate(self) -> None:
        if not self._torque:
            return  # a de-energized arm does not track its goal
        for j in JOINT_NAMES:
            self._pos[j] += (self._goal[j] - self._pos[j]) * self.smoothing

    def read_positions(self) -> dict[str, float]:
        self._check_io()
        self._integrate()
        # A touch of idle drift on the gripper so live plots aren't flat.
        wobble = 0.2 * math.sin(time.monotonic() - self._t0)
        out = dict(self._pos)
        out["gripper"] = max(0.0, min(100.0, out["gripper"] + wobble))
        return out

    def write_positions(self, goal: dict[str, float]) -> dict[str, float]:
        self._check_io()
        commanded: dict[str, float] = {}
        for j, v in goal.items():
            if j in self._goal:
                self._goal[j] = float(v)
                commanded[j] = float(v)
        return commanded

    def set_torque(self, enabled: bool) -> None:
        if not enabled:
            if not self._connected:
                raise RuntimeError(f"MockBus[{self.name}] is not connected")
            if not self.torque_off_all():
                raise TorqueOffUnconfirmed(self.name, list(JOINT_NAMES))
            return
        self._check_io()
        self._torque = True
        self._torque_state = "on"

    # --- servo health (bring-up kit; synthetic, source="mock") ------------
    _health_model: MockHealthModel | None = None

    @property
    def health_model(self) -> MockHealthModel:
        """Synthetic health model, created on first use (fault injection hooks)."""
        if self._health_model is None:
            settings = get_settings()
            profile = (
                settings.health_leader_supply
                if "leader" in self.name.lower()
                else settings.health_follower_supply
            )
            self._health_model = MockHealthModel(supply_profile=profile)
        return self._health_model

    def read_health(self) -> dict[str, ServoHealthSample]:
        """Synthetic, plausible health samples. Never moves the simulated arm.

        A dead bus (``fail_io``) fails the read like a real one. Pin values or
        inject read failures through ``health_model`` (``set_override``,
        ``fail_next_reads``).
        """
        self._check_io()
        return self.health_model.sample(dict(self._pos), dict(self._goal), torque=self._torque)

    def read_health_limits(self) -> dict[str, HealthLimits]:
        return self.health_model.limits()


class FeetechBus:
    """Real SO-101 hardware through LeRobot's supported device API.

    Current LeRobot releases keep motor normalization and calibration in the
    official ``SO101Follower`` / ``SO101Leader`` objects.  Going through those
    objects is important: constructing a raw ``FeetechMotorsBus`` without its
    calibration would either fail or expose encoder ticks instead of the
    canonical degree / 0..100 units used by datasets and policies.

    Runtime hardware errors are never converted into a mock arm.
    """

    def __init__(self, port: str, baudrate: int, name: str = "feetech") -> None:
        self._configured_port = port
        self.port = port
        self.baudrate = baudrate
        self.name = name
        self.calibration_diagnostics: dict[str, Any] = {}
        self._device: Any | None = None
        self._bus: Any | None = None
        self._backend: str | None = None
        self._calibrated = False
        self._relative_target_limit: float | None = 5.0
        self.driver_version: str | None = None
        # What the host knows about servo torque. It outlives the port: a port
        # closed after an unconfirmed torque-off stays "unconfirmed" until a
        # later connection confirms torque off (safety-B2,
        # docs/audit/v0.1/safety.md).
        self.torque_state: TorqueState = "off"

    @property
    def _is_leader(self) -> bool:
        return "leader" in self.name.lower()

    @property
    def device_id(self) -> str:
        settings = get_settings()
        return settings.leader_id if self._is_leader else settings.follower_id

    def _build_modern_device(self) -> Any:
        """Build a current (LeRobot >= 0.6) SO-101 device lazily."""
        try:
            self.driver_version = version("lerobot")
        except PackageNotFoundError:
            self.driver_version = None
        if self._is_leader:
            from lerobot.teleoperators.so_leader import (  # type: ignore[import-not-found]
                SO101Leader,
                SO101LeaderConfig,
            )

            return SO101Leader(
                SO101LeaderConfig(port=self.port, id=self.device_id, use_degrees=True)
            )

        from lerobot.robots.so_follower import (  # type: ignore[import-not-found]
            SO101Follower,
            SO101FollowerConfig,
        )

        return SO101Follower(
            SO101FollowerConfig(
                port=self.port,
                id=self.device_id,
                use_degrees=True,
                max_relative_target=self._relative_target_limit,
            )
        )

    @contextmanager
    def leader_teleop_targets(self) -> Iterator[None]:
        if self._is_leader:
            raise RuntimeError("leader teleop targets apply only to the follower")
        previous = self._relative_target_limit
        config = getattr(self._device, "config", None)
        previous_config = getattr(config, "max_relative_target", previous)
        self._relative_target_limit = None
        if config is not None:
            config.max_relative_target = None
        try:
            yield
        finally:
            self._relative_target_limit = previous
            current_config = getattr(self._device, "config", None)
            if current_config is not None:
                current_config.max_relative_target = previous_config

    def _refuse_sim_backend(self) -> None:
        """Fail closed when SO101_BACKEND=sim reaches a real Feetech bus.

        ``make_bus`` used to key off ``settings.mock`` only, which the sim
        backend leaves False, so under sim every arm path opened the real
        follower and leader ports while labelled a rehearsal. ``make_bus`` now
        builds the simulated bus under sim; this check stays as a permanent
        invariant for any other route to a FeetechBus. It runs before LeRobot
        is imported or a port is opened: a FeetechBus never connects under the
        sim backend.
        """
        if current_backend(get_settings()) == "sim":
            raise HardwareUnavailableError(
                f"SO101_BACKEND=sim: the real Feetech bus on {self.port} is refused (it "
                "would move the real arm); under sim the suite drives the simulated arm "
                "that make_bus builds. Use SO101_BACKEND=mock, or `so101 sim` for physics."
            )

    def _build(self) -> tuple[str, Any]:
        # SO-101 motors are configured by LeRobot at 1 Mbps.  Accepting a
        # different value here would make the setting appear honored while the
        # modern device class still opens at its fixed default.
        if self.baudrate != 1_000_000:
            raise ValueError(
                "SO-101 LeRobot devices require 1,000,000 baud; "
                f"got {self.baudrate}. Reconfigure the motors with lerobot-setup-motors."
            )
        # LeRobot 0.6.1 raises a bare ValueError at import time when this is
        # set (lerobot/utils/constants.py). Name the fix before importing it.
        if "LEROBOT_HOME" in os.environ:
            raise HardwareUnavailableError(
                "LEROBOT_HOME is no longer supported by LeRobot 0.6.1; set HF_LEROBOT_HOME"
            )

        try:
            return "modern", self._build_modern_device()
        except (ImportError, ModuleNotFoundError) as exc:
            # Double quotes: cmd.exe keeps single quotes literally.
            raise HardwareUnavailableError(
                "Real SO-101 hardware requires the pinned LeRobot 0.6.1 adapter. "
                'Install with `pip install -e ".[robot]"`; no legacy or mock '
                "fallback was started."
            ) from exc

    def _connect_modern_safely(self, implementation: Any) -> bool:
        """Open and configure the pinned LeRobot device with torque kept off.

        Cleanup after a failure is ``connect``'s job (``_abandon``): it turns
        off what still answers and always closes the port.
        """

        from ..bringup.ownership import require_device_lock

        pinned = require_device_lock(self._configured_port)
        if pinned != self.port:
            raise RuntimeError(
                f"motor endpoint changed between build and open ({self.port} -> {pinned})"
            )
        motor_bus = implementation.bus
        try:
            motor_bus.connect()
        except Exception as exc:
            # LeRobot opens the port before the motor handshake, and a failed
            # handshake raises with the port still open.
            detail = " ".join(str(exc).split()) or type(exc).__name__
            raise HardwareUnavailableError(
                f"{self.name} arm on {self.port} did not connect: {detail}; "
                "run `so101 preflight` (motors/firmware checks)"
            ) from exc
        try:
            if require_device_lock(self._configured_port) != self.port:
                raise RuntimeError("motor endpoint changed during driver open")
        except BaseException as endpoint_error:
            self.torque_state = "unconfirmed"
            try:
                motor_bus.disconnect(disable_torque=False)
            except Exception:
                pass
            raise endpoint_error
        self._require_torque_off(motor_bus)
        calibrated = bool(implementation.is_calibrated)
        for camera in getattr(implementation, "cameras", {}).values():
            camera.connect()
        motor_bus.configure_motors()
        for motor in motor_bus.motors:
            motor_bus.write("Operating_Mode", motor, self._position_mode_value())
            if not self._is_leader:
                config = implementation.config
                motor_bus.write("P_Coefficient", motor, config.position_p_coefficient)
                motor_bus.write("I_Coefficient", motor, config.position_i_coefficient)
                motor_bus.write("D_Coefficient", motor, config.position_d_coefficient)
                if motor == "gripper":
                    motor_bus.write("Max_Torque_Limit", motor, 500)
                    motor_bus.write("Protection_Current", motor, 250)
                    motor_bus.write("Overload_Torque", motor, 25)
        # The suite enables follower torque only after calibration, a
        # measured-state seed, and an E-stop check in SO101Follower.connect.
        self._require_torque_off(motor_bus)
        return calibrated

    @staticmethod
    def _position_mode_value() -> int:
        from lerobot.motors.feetech import (  # type: ignore[import-not-found]
            OperatingMode,
        )

        return int(OperatingMode.POSITION.value)

    def connect(self) -> None:
        self._refuse_sim_backend()
        if self.connected:
            return
        # PolicyRunner may construct this wrapper before its standalone gate
        # acquires the device lock. Resolve again at the actual open boundary:
        # a symlink retargeted after acquisition must not redirect this bus to
        # an endpoint the process never locked.
        from ..bringup.ownership import require_device_lock

        self.port = require_device_lock(self._configured_port)
        backend, implementation = self._build()
        try:
            calibrated = self._connect_modern_safely(implementation)
        except BaseException:
            # Never keep a half-connected object, never leak its port, and,
            # critically, never replace it with a mock in real mode.
            self._abandon(implementation)
            raise

        self._backend = backend
        self._device = implementation
        self._bus = implementation.bus
        self._calibrated = calibrated
        # Every motor's Torque_Enable read back off on this connection.
        self.torque_state = "off"
        log.info("FeetechBus[%s] connected on %s", self.name, self.port)

    def disconnect(self) -> None:
        """Torque off on every motor, then always close the port.

        LeRobot 0.6.1's ``disconnect(disable_torque=True)`` skips
        ``closePort()`` when its torque-off raises, which leaks the handle (on
        Windows the COM port stays locked until the process exits). Here the
        torque-off is the read-back pass of ``set_torque(False)`` and the close
        always follows, with ``disable_torque=False``. When torque-off was not
        confirmed this raises ``TorqueOffUnconfirmed`` after the close, and
        ``torque_state`` stays "unconfirmed": a closed port never reads as
        torque off (safety-B2, docs/audit/v0.1/safety.md).
        """
        implementation = self._device
        bus = self._bus
        pending: list[str] = []
        try:
            if implementation is not None and bus is not None:
                if self.connected:
                    pending = self._turn_torque_off(bus)
                elif self.torque_state != "off":
                    # The port closed under us while torque was not known off.
                    self.torque_state = "unconfirmed"
                    pending = list(getattr(bus, "motors", None) or []) or ["all motors"]
        finally:
            if implementation is not None:
                self._close_port(implementation)
            self._drop_handle()
        if pending:
            raise TorqueOffUnconfirmed(self.name, pending)

    def disconnect_holding(self) -> None:
        """Close only the host port while confirmed torque-on holds the goal.

        This is exclusively for a successful standalone category-2 Halt. It
        never falls back to a driver ``disconnect()`` whose default could
        disable torque. The caller has already durably recorded energized
        intent and will retain it after the port closes.
        """

        implementation = self._device
        motor_bus = self._bus
        if implementation is None or motor_bus is None or not self.connected:
            raise RuntimeError(f"{self.name}: cannot Halt-disconnect a closed port")
        if self.torque_state != "on":
            raise RuntimeError(
                f"{self.name}: cannot Halt-disconnect with torque {self.torque_state}"
            )
        disconnect = getattr(motor_bus, "disconnect", None)
        if not callable(disconnect):
            raise RuntimeError(f"{self.name}: motor bus cannot close its host port")
        try:
            disconnect(disable_torque=False)
        except TypeError as exc:
            raise RuntimeError(
                f"{self.name}: motor driver cannot close without a torque write"
            ) from exc
        if self._port_open(motor_bus):
            raise RuntimeError(f"{self.name}: motor port stayed open after Halt")
        for camera in (getattr(implementation, "cameras", None) or {}).values():
            try:
                if bool(getattr(camera, "is_connected", False)):
                    camera.disconnect()
            except Exception as exc:
                log.warning("%s camera disconnect raised after Halt: %s", self.name, exc)
        self._drop_handle()

    def release_port(self) -> None:
        """Close the port with no torque write and drop the handle. Never raises.

        Only for use after physical power was cut, so torque counts as off
        from here on. A dead adapter's handle cannot carry a torque-off, and an
        unplugged adapter stays dead even after a replug (safety-B2,
        docs/audit/v0.1/safety.md).
        """
        implementation = self._device
        if implementation is not None:
            self._close_port(implementation)
        self._drop_handle()
        self.torque_state = "off"

    def _abandon(self, implementation: Any) -> None:
        """Clean up after a failed connect. Never raises.

        Turns off torque on every motor that answers, then closes the port. It
        leaves ``torque_state`` unconfirmed when any motor does not answer: a
        fresh wrapper cannot know whether an earlier session left it energized.
        """
        motor_bus = getattr(implementation, "bus", None)
        if motor_bus is not None and self._port_open(motor_bus):
            try:
                self._turn_torque_off(motor_bus)
            except Exception:
                self.torque_state = "unconfirmed"
                log.exception("%s torque-off after a failed connect raised", self.name)
        self._close_port(implementation)

    @staticmethod
    def _port_open(motor_bus: Any) -> bool:
        for attribute in ("is_connected", "connected"):
            state = getattr(motor_bus, attribute, None)
            if state is not None:
                return bool(state)
        return True

    def _close_port(self, implementation: Any) -> None:
        """Close the motor port with no torque write, then the cameras. Never raises."""
        motor_bus = getattr(implementation, "bus", None)
        try:
            if motor_bus is not None and callable(getattr(motor_bus, "disconnect", None)):
                if self._port_open(motor_bus):
                    try:
                        motor_bus.disconnect(disable_torque=False)
                    except TypeError:  # older or simplified bus signature
                        motor_bus.disconnect()
            elif bool(getattr(implementation, "is_connected", False)):
                implementation.disconnect()
        except Exception as exc:
            log.warning("%s port close raised: %s", self.name, exc)
        for camera in (getattr(implementation, "cameras", None) or {}).values():
            try:
                if bool(getattr(camera, "is_connected", False)):
                    camera.disconnect()
            except Exception as exc:
                log.warning("%s camera disconnect raised: %s", self.name, exc)

    def _drop_handle(self) -> None:
        self._device = None
        self._bus = None
        self._backend = None
        self._calibrated = False

    def torque_off_all(self) -> bool:
        """Torque off on every motor, confirmed by read-back. Never raises.

        The same pass as ``set_torque(False)``, reported instead of raised.
        """
        bus = self._bus
        if bus is None or not self.connected:
            return False
        return not self._turn_torque_off(bus)

    def _turn_torque_off(self, bus: Any) -> list[str]:
        pending = self._torque_off_every_motor(bus)
        self.torque_state = "unconfirmed" if pending else "off"
        return pending

    def _require_torque_off(self, bus: Any) -> None:
        """Run the confirmed torque-off pass, raising if any motor may still hold."""
        pending = self._turn_torque_off(bus)
        if pending:
            raise TorqueOffUnconfirmed(self.name, pending)

    def _torque_off_every_motor(self, bus: Any) -> list[str]:
        """Turn torque off on every motor; return the motors not confirmed off.

        LeRobot's ``disable_torque()`` writes motor by motor and stops at the
        first servo that does not answer, leaving every later motor energized.
        This pass never stops early:

        1. One ``sync_write("Torque_Enable", 0)`` broadcast. It has no status
           packets, so a silent servo cannot delay the others.
        2. Each motor's Torque_Enable is read back.
        3. Each motor that did not read 0 gets its own acknowledged
           ``disable_torque`` (Torque_Enable, then Lock) with
           ``TORQUE_OFF_RETRIES`` retries, and is read back again.

        Never raises. Worst case, six silent motors at LeRobot 0.6.1's ~50 ms
        packet timeout, is about 1.2 s.
        """
        try:
            motors = list(getattr(bus, "motors", None) or [])
        except Exception as exc:
            log.error("%s motor metadata could not be read: %s", self.name, exc)
            motors = []
        try:
            bus.sync_write("Torque_Enable", 0, normalize=False)
        except Exception as exc:
            log.warning("%s broadcast torque-off failed: %s", self.name, exc)
        try:
            read = getattr(bus, "read", None)
        except Exception as exc:
            log.error("%s torque read-back is unavailable: %s", self.name, exc)
            read = None
        if not motors:
            # Make one best-effort bulk write, but without the motor roster no
            # servo can be read back or named as confirmed off.
            self._disable_torque_acknowledged(bus, None)
            log.error("%s torque-off unconfirmed: motor metadata is empty", self.name)
            return ["all motors"]
        if not callable(read):
            # Still try every known motor individually. An acknowledgement is
            # not read-back, so none can count as confirmed off.
            for motor in motors:
                self._disable_torque_acknowledged(bus, motor)
            log.error("%s torque-off unconfirmed: no per-motor read-back", self.name)
            return motors
        pending = self._torque_not_confirmed_off(bus, motors, num_retry=0)
        acknowledged = [motor for motor in pending if self._disable_torque_acknowledged(bus, motor)]
        still_on = set(self._torque_not_confirmed_off(bus, acknowledged, num_retry=1))
        unconfirmed = [motor for motor in pending if motor not in acknowledged or motor in still_on]
        if unconfirmed:
            log.error("%s torque-off unconfirmed on %s", self.name, ", ".join(unconfirmed))
        return unconfirmed

    def _disable_torque_acknowledged(self, bus: Any, motors: str | list[str] | None) -> bool:
        try:
            try:
                bus.disable_torque(motors, num_retry=TORQUE_OFF_RETRIES)
            except TypeError:  # older or simplified bus signature
                bus.disable_torque()
        except Exception as exc:
            log.error("%s torque-off write failed for %s: %s", self.name, motors or "all", exc)
            return False
        return True

    def _torque_not_confirmed_off(
        self, bus: Any, motors: list[str], *, num_retry: int
    ) -> list[str]:
        pending: list[str] = []
        for motor in motors:
            try:
                value = bus.read("Torque_Enable", motor, normalize=False, num_retry=num_retry)
                confirmed = int(value) == 0
            except Exception as exc:
                log.warning("%s torque read-back failed for %s: %s", self.name, motor, exc)
                confirmed = False
            if not confirmed:
                pending.append(motor)
        return pending

    @property
    def register_bus(self) -> Any | None:
        """The open LeRobot motor bus, or None while the port is closed.

        Only for read-only diagnostics that the manager runs under its bus
        lock (``RobotManager.run_idle_bus_diagnostic``, which lends it through
        a read-only view).
        """
        return self._bus if self.connected else None

    @property
    def connected(self) -> bool:
        implementation = self._device
        if implementation is None:
            return False
        state = getattr(implementation, "is_connected", None)
        if state is not None:
            return bool(state)
        return bool(getattr(implementation, "connected", False))

    @staticmethod
    def _canonical_positions(values: Any) -> dict[str, float]:
        if isinstance(values, dict):
            normalized = {
                joint: values.get(f"{joint}.pos", values.get(joint)) for joint in JOINT_NAMES
            }
        else:
            normalized = dict(zip(JOINT_NAMES, values, strict=False))

        missing = [joint for joint in JOINT_NAMES if normalized.get(joint) is None]
        if missing:
            raise RuntimeError(f"motor read omitted canonical SO-101 joints: {missing}")

        result: dict[str, float] = {}
        for joint in JOINT_NAMES:
            value = normalized[joint]
            assert value is not None
            result[joint] = float(value)
        non_finite = [joint for joint, value in result.items() if not math.isfinite(value)]
        if non_finite:
            raise RuntimeError(f"motor read returned non-finite positions for: {non_finite}")
        return result

    def read_positions(self) -> dict[str, float]:
        if not self.connected or self._bus is None:
            raise RuntimeError("bus not connected")
        if not self._calibrated:
            raise RuntimeError(
                f"{self.name} arm is not calibrated; run the SO-101 calibration flow first"
            )
        if self._backend == "modern":
            assert self._device is not None
            values = (
                self._device.get_action() if self._is_leader else self._device.get_observation()
            )
            return self._canonical_positions(values)
        raise RuntimeError("unsupported non-modern LeRobot backend")

    def write_positions(self, goal: dict[str, float]) -> dict[str, float]:
        if not self.connected or self._bus is None:
            raise RuntimeError("bus not connected")
        if not self._calibrated:
            raise RuntimeError(f"{self.name} arm is not calibrated; refusing a position command")
        ordered_goal = {joint: float(goal[joint]) for joint in JOINT_NAMES if joint in goal}
        if any(not math.isfinite(value) for value in ordered_goal.values()):
            raise ValueError("goal contains a non-finite motor position")

        if self._backend == "modern":
            assert self._device is not None
            action = {f"{joint}.pos": value for joint, value in ordered_goal.items()}
            if self._is_leader:
                self._device.send_feedback(action)
                return ordered_goal
            else:
                actual = self._device.send_action(action)
                commanded: dict[str, float] = {}
                for joint in ordered_goal:
                    raw = actual.get(f"{joint}.pos", actual.get(joint))
                    if raw is None or not math.isfinite(float(raw)):
                        raise RuntimeError(f"driver returned no finite command for {joint!r}")
                    commanded[joint] = float(raw)
                return commanded

        raise RuntimeError("unsupported non-modern LeRobot backend")

    def set_torque(self, enabled: bool) -> None:
        """Torque on, or torque off on every motor with read-back.

        Torque-off tries every motor even when one does not answer, never
        closes the port (Halt keeps it open), and then raises
        ``TorqueOffUnconfirmed`` naming the motors that may still hold torque.
        The real ``True`` path itself requires an adopted DeviceBusLock and a
        fresh durable global-intent capability; a raw caller cannot bypass the
        ownership/write-ahead contract.
        """
        if not self.connected or self._bus is None:
            raise RuntimeError("bus not connected")
        if self._backend != "modern":
            raise RuntimeError("unsupported non-modern LeRobot backend")
        if enabled:
            from ..bringup.ownership import require_enable_intent

            require_enable_intent(self._configured_port)
            # Marked first: a partial enable may already have energized motors.
            self.torque_state = "on"
            self._bus.enable_torque()
            return
        pending = self._turn_torque_off(self._bus)
        if pending:
            raise TorqueOffUnconfirmed(self.name, pending)

    def calibrate_timed(
        self,
        sample_seconds: float,
        progress=None,
        cancel_event: threading.Event | None = None,
        telemetry=None,
        joint: str | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Run LeRobot-compatible midpoint/range calibration without stdin.

        The operator must place the arm at the midpoint before starting, then
        sweep every joint (except continuous wrist roll) through its range while
        this method samples raw encoder ticks. Calibration is written both to
        the motors and LeRobot's standard per-device JSON file.

        With ``joint`` set, capture an independent baseline: home and sample
        that joint only, restore and verify its prior registers, and return
        measured ranges without changing the native file or calibrated flag.
        The suite persists this baseline separately from full-arm calibration.
        """
        if self._backend != "modern" or self._device is None or self._bus is None:
            raise RuntimeError("timed calibration requires the current LeRobot SO-101 API")
        if not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError("sample_seconds must be finite and positive")

        if joint is not None and joint not in JOINT_NAMES:
            raise ValueError("unknown calibration joint")
        selected = [joint] if joint else list(self._bus.motors)
        self.calibration_diagnostics = {
            "joint": joint,
            "baseline_only": joint is not None,
            "motor_ids": {name: self._bus.motors[name].id for name in selected},
        }

        from lerobot.motors import MotorCalibration  # type: ignore[import-not-found]
        from lerobot.motors.feetech import OperatingMode  # type: ignore[import-not-found]

        motor_bus = self._bus
        device = self._device
        # The suite's torque-off pass (broadcast, then per-motor retries, then
        # read-back), not LeRobot's disable_torque(), which stops at the first
        # silent servo and leaves every later motor energized. The operator
        # moves the arm by hand next, so an unconfirmed torque-off stops here,
        # before anything is written.
        pending = self._turn_torque_off(motor_bus)
        if pending:
            raise TorqueOffUnconfirmed(self.name, pending)
        previous_motor_calibration = motor_bus.read_calibration()
        previous_device_calibration = dict(getattr(self._device, "calibration", {}) or {})
        calibration_path = Path(self._device.calibration_fpath)
        previous_file = calibration_path.read_bytes() if calibration_path.exists() else None
        previous_calibrated = self._calibrated
        # Joint baselines are measured independently, then the selected motor's
        # original registers are restored. Other joints' calibration is untouched.
        restore_calibration = (
            {joint: previous_motor_calibration[joint]} if joint else previous_motor_calibration
        )

        def restore() -> None:
            unconfirmed = self._turn_torque_off(motor_bus)
            if joint and unconfirmed:
                raise TorqueOffUnconfirmed(self.name, unconfirmed)
            motor_bus.write_calibration(restore_calibration)
            motor_bus.calibration = previous_motor_calibration
            if joint:
                actual = motor_bus.read_calibration().get(joint)
                expected = previous_motor_calibration[joint]
                if any(
                    getattr(actual, name, None) != getattr(expected, name, None)
                    for name in ("homing_offset", "range_min", "range_max")
                ):
                    raise RuntimeError(f"prior calibration restore read-back failed for {joint}")
            device.calibration = previous_device_calibration or previous_motor_calibration
            if joint is None:
                if previous_file is None:
                    if calibration_path.exists():
                        calibration_path.unlink()
                else:
                    calibration_path.parent.mkdir(parents=True, exist_ok=True)
                    calibration_path.write_bytes(previous_file)
            self._calibrated = previous_calibrated

        sequence = 0
        sample_count = 0

        def publish(phase: str, *, elapsed: float = 0.0, error: str | None = None) -> None:
            nonlocal sequence
            sequence += 1
            snapshot = {
                key: copy.deepcopy(value)
                for key, value in self.calibration_diagnostics.items()
                if key != "live"
            }
            snapshot.update(
                phase=phase,
                sequence=sequence,
                sample_count=sample_count,
                elapsed_seconds=elapsed,
                error=error,
            )
            self.calibration_diagnostics["live"] = snapshot
            if telemetry:
                telemetry(copy.deepcopy(snapshot))

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("calibration cancelled before commit")

        try:
            publish("homing")
            check_cancelled()
            for motor in selected:
                motor_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

            # Forward the pinned upstream algorithm unchanged. Attribute errors
            # from its individual register writes to the motor being written,
            # including sign-encoding errors raised before any serial transfer.
            original_write = motor_bus.write
            had_override = "write" in vars(motor_bus)
            previous_override = vars(motor_bus).get("write")

            def homing_write(data_name, motor, value, **kwargs):
                if data_name == "Homing_Offset":
                    self.calibration_diagnostics.setdefault("homing_offsets_requested", {})[
                        motor
                    ] = value
                try:
                    return original_write(data_name, motor, value, **kwargs)
                except Exception as exc:
                    self.calibration_diagnostics["affected_motors"] = [motor]
                    self.calibration_diagnostics["failed_write"] = {
                        "motor": motor,
                        "register": data_name,
                        "value": value,
                    }
                    raise RuntimeError(
                        f"calibration homing failed for {motor} ({data_name}={value}): {exc}"
                    ) from exc

            motor_bus.write = homing_write
            try:
                homing_offsets = (
                    motor_bus.set_half_turn_homings([joint])
                    if joint
                    else motor_bus.set_half_turn_homings()
                )
            finally:
                if had_override:
                    motor_bus.write = previous_override
                else:
                    del motor_bus.write
            # Confirm the new offsets and reset limits on the actual registers
            # before announcing a sweep. Requested writes are not read-back evidence.
            self.calibration_diagnostics["homing_offsets_requested"] = dict(homing_offsets)
            homing_readback = motor_bus.read_calibration()
            self.calibration_diagnostics["homing_readback"] = {
                motor: {
                    field: getattr(homing_readback.get(motor), field, None)
                    for field in ("homing_offset", "range_min", "range_max")
                }
                for motor in selected
            }
            mismatches = {}
            for motor in selected:
                actual = homing_readback.get(motor)
                expected = {
                    "homing_offset": int(homing_offsets[motor]),
                    "range_min": 0,
                    "range_max": 4095,
                }
                differences = {
                    field: {"expected": value, "actual": getattr(actual, field, None)}
                    for field, value in expected.items()
                    if getattr(actual, field, None) != value
                }
                if differences:
                    mismatches[motor] = differences
            if mismatches:
                self.calibration_diagnostics["affected_motors"] = list(mismatches)
                raise ValueError(
                    f"calibration homing read-back mismatch: {mismatches}; "
                    "the sweep was not started; prior calibration will be restored"
                )
            sampled_motors = (
                selected if joint else [motor for motor in selected if motor != "wrist_roll"]
            )
            positions = motor_bus.sync_read(
                "Present_Position", sampled_motors, normalize=False, num_retry=5
            )
            homed_positions = dict(positions)
            self.calibration_diagnostics["starting_positions"] = homed_positions
            range_mins = dict(positions)
            range_maxes = dict(positions)
            sample_count = 1
            self.calibration_diagnostics.update(
                positions=dict(positions),
                range_mins=dict(range_mins),
                range_maxes=dict(range_maxes),
            )
            publish("sampling")
            log.info(
                "%s calibration: homing offsets verified %s; post-homing starting readings %s",
                self.name,
                homing_offsets,
                homed_positions,
            )
            sweep_instruction = (
                f"sweep only {joint} through its safe range"
                if joint
                else "sweep every joint through its range"
            )
            if progress:
                progress(
                    0.02,
                    f"midpoint captured (homing read-back verified); now {sweep_instruction}",
                )

            start = time.monotonic()
            deadline = start + sample_seconds
            while (now := time.monotonic()) < deadline:
                check_cancelled()
                positions = motor_bus.sync_read(
                    "Present_Position", sampled_motors, normalize=False, num_retry=5
                )
                for motor in sampled_motors:
                    range_mins[motor] = min(range_mins[motor], positions[motor])
                    range_maxes[motor] = max(range_maxes[motor], positions[motor])
                sample_count += 1
                self.calibration_diagnostics.update(
                    positions=dict(positions),
                    range_mins=dict(range_mins),
                    range_maxes=dict(range_maxes),
                )
                publish("sampling", elapsed=now - start)
                if progress:
                    fraction = (now - start) / sample_seconds
                    progress(
                        min(0.98, max(0.02, fraction)),
                        sweep_instruction,
                    )
                time.sleep(0.02)

            publish("validating", elapsed=sample_seconds)
            insufficient = [
                motor for motor in sampled_motors if range_maxes[motor] - range_mins[motor] < 100
            ]
            self.calibration_diagnostics["range_mins"] = dict(range_mins)
            self.calibration_diagnostics["range_maxes"] = dict(range_maxes)
            if insufficient:
                self.calibration_diagnostics["affected_motors"] = list(insufficient)
                raise ValueError(
                    "calibration observed less than 100 encoder ticks of range for: "
                    f"{insufficient}; prior calibration will be restored"
                )

            invalid_ranges = {
                motor: (range_mins[motor], range_maxes[motor])
                for motor in sampled_motors
                if not 0 <= range_mins[motor] < range_maxes[motor] <= 4095
            }
            if invalid_ranges:
                self.calibration_diagnostics["affected_motors"] = list(invalid_ranges)
                raise ValueError(
                    f"calibration encoder ranges outside 0..4095: {invalid_ranges}; "
                    f"post-homing starting readings: {homed_positions}; "
                    f"verified homing offsets: {homing_offsets}; "
                    "restart with every joint in the middle of its physical range before "
                    "midpoint capture, then sweep gently without forcing stops; "
                    "prior calibration will be restored"
                )

            if joint is not None:
                check_cancelled()
                publish("restoring", elapsed=sample_seconds)
                restore()
                check_cancelled()
                self.calibration_diagnostics["restored"] = True
                lo = range_mins[joint]
                hi = range_maxes[joint]
                half_degrees = (hi - lo) * 180.0 / 4095.0
                ranges = {
                    joint: (0.0, 100.0) if joint == "gripper" else (-half_degrees, half_degrees)
                }
                if progress:
                    progress(1.0, f"{joint} baseline captured; prior calibration restored")
                publish("complete", elapsed=sample_seconds)
                return ranges

            range_mins["wrist_roll"] = 0
            range_maxes["wrist_roll"] = 4095
            calibration = {
                motor: MotorCalibration(
                    id=definition.id,
                    drive_mode=0,
                    homing_offset=int(homing_offsets[motor]),
                    range_min=int(range_mins[motor]),
                    range_max=int(range_maxes[motor]),
                )
                for motor, definition in motor_bus.motors.items()
            }
            check_cancelled()
            publish("saving", elapsed=sample_seconds)
            motor_bus.write_calibration(calibration)
            self._device.calibration = calibration
            self._device._save_calibration()
            check_cancelled()
            self._calibrated = True

            normalized_ranges: dict[str, tuple[float, float]] = {}
            for motor in JOINT_NAMES:
                if motor == "gripper":
                    normalized_ranges[motor] = (0.0, 100.0)
                    continue
                half_span_ticks = (range_maxes[motor] - range_mins[motor]) / 2
                half_span_degrees = half_span_ticks * 360.0 / 4095.0
                normalized_ranges[motor] = (-half_span_degrees, half_span_degrees)
            if progress:
                progress(1.0, "calibration written to motors and LeRobot calibration file")
            publish("complete", elapsed=sample_seconds)
            return normalized_ranges
        except BaseException as exc:
            publish(
                "failed",
                elapsed=self.calibration_diagnostics.get("live", {}).get("elapsed_seconds", 0.0),
                error=str(exc),
            )
            try:
                # Same pass on the way out; an unconfirmed result stays in
                # torque_state, and the caller's disconnect() reports it.
                restore()
            except Exception as rollback_exc:
                raise RuntimeError(
                    f"calibration failed ({exc}) and rollback failed ({rollback_exc})"
                ) from rollback_exc
            raise

    # --- servo health (bring-up kit) ----------------------------------------
    # (bus object, limits or None, read attempts) for the current connection.
    _health_limits_cache: tuple[Any, dict[str, HealthLimits] | None, int] | None = None

    def read_health(self) -> dict[str, ServoHealthSample]:
        """Sync-read Present_Temperature/Voltage/Load/Current and Status.

        Read-only. The caller must hold the same I/O lock that serializes
        control traffic on this bus (RobotManager's ``_io_lock``).
        """
        if not self.connected or self._bus is None:
            raise RuntimeError("bus not connected")
        bus = self._bus
        return read_motors_bus_health(bus, source="real", limits=self._health_limits_for(bus))

    def _health_limits_for(self, bus: Any) -> dict[str, HealthLimits] | None:
        """EEPROM limits, read once per connection (up to 3 attempts)."""
        cached = self._health_limits_cache
        attempts = 0
        if cached is not None and cached[0] is bus:
            if cached[1] is not None or cached[2] >= 3:
                return cached[1]
            attempts = cached[2]
        try:
            limits: dict[str, HealthLimits] | None = read_motors_bus_limits(bus)
        except Exception as exc:
            log.warning("could not read %s servo limit registers: %s", self.name, exc)
            limits = None
        self._health_limits_cache = (bus, limits, attempts + 1)
        return limits


class SimBusUnavailable:
    """Stand-in bus for ``SO101_BACKEND=sim`` when the ``[sim]`` extra is missing.

    Construction never raises, so the manager and ``/api/robot/status`` still
    work; ``connect`` fails closed with the install hint. It never falls back to
    a real or mock bus.
    """

    torque_state: TorqueState = "off"

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def _refuse(self) -> HardwareUnavailableError:
        return HardwareUnavailableError(
            f"SO101_BACKEND=sim needs the [sim] extra for the simulated {self.name} arm "
            f'({self.reason}); install it with `pip install -e ".[sim]"`, or use '
            "SO101_BACKEND=mock"
        )

    def connect(self) -> None:
        raise self._refuse()

    def disconnect(self) -> None:
        return None

    def disconnect_holding(self) -> None:
        raise self._refuse()

    @property
    def connected(self) -> bool:
        return False

    def read_positions(self) -> dict[str, float]:
        raise self._refuse()

    def write_positions(self, goal: dict[str, float]) -> dict[str, float]:
        raise self._refuse()

    def set_torque(self, enabled: bool) -> None:
        raise self._refuse()


def make_bus(port: str, *, name: str, force_mock: bool | None = None) -> MotorBus:
    """Build the bus the backend selects for ``port``.

    - mock: a ``MockBus``, only when mock mode is explicitly enabled.
    - sim: the simulated bus (``so101.simulation.devices.make_sim_bus``): the
      MuJoCo follower, or the scripted-expert leader for a name containing
      "leader". The ``port`` is ignored; a real port is never opened.
    - real: a ``FeetechBus`` that fails loudly on connect if its dependency or
      device is missing; silently moving a simulated arm would be unsafe and
      could also produce mislabeled datasets.
    """
    settings = get_settings()
    use_mock = settings.mock if force_mock is None else force_mock
    if use_mock:
        return MockBus(name=name)
    if force_mock is None and current_backend(settings) == "sim":
        try:
            from ..simulation.devices import make_sim_bus  # the [sim] extra (MuJoCo)
        except Exception as exc:  # not only ImportError: a broken MuJoCo fails its import too
            return SimBusUnavailable(name, f"{type(exc).__name__}: {exc}")
        return make_sim_bus(name)
    return FeetechBus(port=port, baudrate=settings.baudrate, name=name)


def bus_torque_state(bus: Any) -> TorqueState:
    """What the host knows about ``bus``'s torque: "on", "off" or "unconfirmed".

    Buses report it as ``torque_state`` (C2 three-state torque truth, which
    survives a port close). A bus that only exposes ``torque_enabled`` maps it
    to "on"/"off". Anything else, or a bus that raises, is "unconfirmed":
    never assume torque is off. Never raises.
    """
    try:
        state = getattr(bus, "torque_state", None)
        if state in ("on", "off", "unconfirmed"):
            return state  # type: ignore[no-any-return]
        enabled = getattr(bus, "torque_enabled", None)
        if isinstance(enabled, bool):
            return "on" if enabled else "off"
    except Exception as exc:
        log.error("could not read the torque state of %s: %s", type(bus).__name__, exc)
    return "unconfirmed"


def torque_off_all(bus: MotorBus) -> bool:
    """Disable torque on every motor of ``bus`` and confirm it. Never raises.

    Uses the bus's own ``torque_off_all`` (sync write, then per-motor
    read-back) when it has one. Otherwise it falls back to ``set_torque(False)``
    plus a ``torque_enabled`` read-back when the bus exposes one. Returns True
    only when torque-off is confirmed.
    """
    method = getattr(bus, "torque_off_all", None)
    if callable(method):
        try:
            return bool(method())
        except Exception as exc:
            log.error("torque_off_all failed on %s: %s", type(bus).__name__, exc)
            return False
    try:
        bus.set_torque(False)
    except Exception as exc:
        log.error("torque-off failed on %s: %s", type(bus).__name__, exc)
        return False
    return getattr(bus, "torque_enabled", False) is not True


def release_port(bus: MotorBus) -> None:
    """Close ``bus``'s port with no torque write when it can. Never raises.

    Only for use after physical power was cut (safety-B2,
    docs/audit/v0.1/safety.md). A bus without ``release_port`` gets a plain
    ``disconnect()``. Its torque write is harmless once power is gone.
    """
    method = getattr(bus, "release_port", None)
    try:
        if callable(method):
            method()
        else:
            bus.disconnect()
    except Exception as exc:
        log.error("port release failed on %s: %s", type(bus).__name__, exc)

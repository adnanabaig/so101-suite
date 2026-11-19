"""SO-101 follower arm: the actuated arm that executes commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from .. import JOINT_NAMES
from ..config import get_settings
from ..logging import get_logger
from .backend import current_backend
from .calibration import CalibrationResult, counts_as_missing
from .motors import FeetechBus, MotorBus, TorqueState, bus_torque_state, make_bus
from .safety import DEFAULT_LIMITS, SafetyGuard, SafetyViolation

log = get_logger(__name__)


class SO101Follower:
    """Actuated arm. All goal commands pass through a :class:`SafetyGuard`."""

    def __init__(self, port: str | None = None, guard: SafetyGuard | None = None):
        settings = get_settings()
        self.port = port or settings.follower_port
        self.bus: MotorBus = make_bus(self.port, name="follower")
        self.guard = guard or self.guard_from_saved_calibration()
        # The last goal written to each servo's Goal_Position (what the driver
        # accepted). The Halt grip rule reads the gripper's (design doc §5).
        self.last_goal: dict[str, float] = {}

    @contextmanager
    def leader_teleop_profile(self) -> Iterator[None]:
        """Use native LeRobot targets with a bounded Suite command slew."""
        previous = self.guard.max_velocity_deg_s
        ceiling = get_settings().teleop_max_velocity_deg_s
        default = self.guard.max_step_deg / self.guard.nominal_dt_s
        rates = {
            joint: min(
                ceiling,
                previous.get(joint, default)
                if isinstance(previous, dict)
                else default
                if previous is None
                else previous,
            )
            for joint in JOINT_NAMES
        }
        context = (
            self.bus.leader_teleop_targets() if isinstance(self.bus, FeetechBus) else nullcontext()
        )
        with context:
            self.guard.max_velocity_deg_s = rates
            try:
                yield
            finally:
                self.guard.max_velocity_deg_s = previous

    def command_profile(self) -> dict[str, Any]:
        return {
            "kind": "lerobot_leader_mirror",
            "driver_version": self.bus.driver_version if isinstance(self.bus, FeetechBus) else None,
            "follower_id": get_settings().follower_id,
            "leader_id": get_settings().leader_id,
            "limits": dict(self.guard.limits),
            "max_step": self.guard.max_step_deg,
            "max_velocity": self.guard.max_velocity_deg_s,
            "driver_present_position_window": (
                self.bus._relative_target_limit if isinstance(self.bus, FeetechBus) else None
            ),
            "angular_unit": "degrees",
            "gripper_unit": "percent",
        }

    @staticmethod
    def guard_from_saved_calibration() -> SafetyGuard:
        """Intersect conservative defaults with the measured physical ranges."""
        try:
            calibration = CalibrationResult.load("follower")
        except Exception as exc:
            log.warning("could not load follower safety calibration: %s", exc)
            calibration = None
        if calibration is None:
            return SafetyGuard()
        if counts_as_missing(calibration, current_backend(get_settings())):
            log.warning(
                "ignoring the follower calibration made on the mock backend; SafetyGuard keeps "
                "its conservative default limits. Recalibrate on the real arm"
            )
            return SafetyGuard()
        expected_device = get_settings().follower_id
        if calibration.device_id and calibration.device_id not in {"mock", expected_device}:
            log.error(
                "ignoring follower safety calibration for device %s; expected %s",
                calibration.device_id,
                expected_device,
            )
            return SafetyGuard()

        limits = dict(DEFAULT_LIMITS)
        for joint, measured in calibration.joints.items():
            if joint not in limits:
                continue
            default_lo, default_hi = limits[joint]
            lo = max(default_lo, float(measured.min))
            hi = min(default_hi, float(measured.max))
            if lo < hi:
                limits[joint] = (lo, hi)
            else:
                log.warning(
                    "ignoring invalid measured safety range for %s: (%s, %s)",
                    joint,
                    measured.min,
                    measured.max,
                )
        return SafetyGuard(limits=limits)

    # --- lifecycle ------------------------------------------------------
    def connect(self, *, enable_torque: bool = True) -> None:
        try:
            self.bus.connect()
            # Seed the limiter from the measured arm pose before accepting a
            # command. This prevents the first teleop/policy frame from jumping
            # from an implicit zero reference.
            measured = self.bus.read_positions()
            self.guard.seed(reference=measured)
            self._warn_if_outside_envelope()
            if self.guard.estopped:
                raise SafetyViolation("E-STOP engaged; refusing to enable follower torque")
            # Prime Goal_Position to the measured pose while torque is still
            # disabled. Feetech servos retain goal RAM across sessions; enabling
            # against a stale goal can otherwise make a backdriven arm jump.
            commanded = self.bus.write_positions(measured)
            self.last_goal.update(commanded)
            self.guard.record_actual_command(commanded)
            if enable_torque:
                if self.guard.estopped:
                    raise SafetyViolation("E-STOP engaged during follower hold priming")
                self.bus.set_torque(True)
        except BaseException:
            try:
                self.bus.disconnect()
            except BaseException:
                pass
            raise

    def enable_torque(self) -> dict[str, float]:
        if self.guard.estopped:
            raise SafetyViolation("E-STOP engaged; refusing to enable follower torque")
        measured = self.bus.read_positions()
        self.guard.seed(measured)
        self._warn_if_outside_envelope()
        commanded = self.bus.write_positions(measured)
        self.last_goal.update(commanded)
        self.guard.record_actual_command(commanded)
        if self.guard.estopped:
            raise SafetyViolation("E-STOP engaged during follower hold priming")
        self.bus.set_torque(True)
        return measured

    def resume_hold(self) -> dict[str, float]:
        """Take over an arm that is already energized and holding (after a Halt).

        No torque write and no goal write: the servos keep holding the goal the
        halt left, so re-arming neither toggles torque nor makes the arm jump.
        The guard is seeded from the measured pose, so the first command is
        rate-limited from where the arm really is. Returns the measured pose.
        """
        if self.guard.estopped:
            raise SafetyViolation("E-STOP engaged; refusing to take over the follower")
        if self.torque_state != "on":
            raise RuntimeError(f"follower torque is {self.torque_state}, not holding")
        measured = self.bus.read_positions()
        self.guard.seed(measured)
        self._warn_if_outside_envelope()
        return measured

    def _warn_if_outside_envelope(self) -> None:
        outside = self.guard.outside_envelope
        if outside:
            log.warning(
                "follower rests outside the safety envelope (%s); it holds there and "
                "only moves back toward the envelope (safety-B10)",
                ", ".join(f"{joint}={value:.1f}" for joint, value in outside.items()),
            )

    def disconnect(self) -> None:
        """Turn torque off, then close the bus, in one pass.

        The bus's ``disconnect()`` runs the torque-off with read-back on every
        motor and always closes the port. When that torque-off is not
        confirmed it raises ``TorqueOffUnconfirmed`` with the port already
        closed, and the bus's ``torque_state`` stays "unconfirmed". A closed
        port never means "torque off" by itself (safety-B2,
        docs/audit/v0.1/safety.md).
        """
        if not self.bus.connected:
            return
        self.bus.disconnect()
        if self.bus.connected:
            raise RuntimeError(
                "follower bus is still open after disconnect; torque-off is unconfirmed"
            )

    def halt_and_disconnect(self) -> dict[str, float]:
        """Hold fresh Present with torque on, then close only the host port.

        Standalone commands use this on successful completion after writing a
        durable energized intent. Failure callers must fall back to the normal
        category-0 :meth:`disconnect` path while the port is still available.
        """

        if not self.bus.connected:
            raise RuntimeError("follower bus is closed; position cannot be held")
        if self.torque_state != "on":
            raise RuntimeError(f"follower torque is {self.torque_state}, not holding")
        present = self.bus.read_positions()
        goal = {
            joint: float(value)
            for joint, value in present.items()
            if joint in JOINT_NAMES and joint != "gripper"
        }
        if "gripper" in present:
            value = float(present["gripper"])
            last = float(self.last_goal.get("gripper", value))
            margin = get_settings().grip_hold_margin
            grip = min(max(last, value - margin), value)
            low, high = self.guard.limits.get("gripper", (0.0, 100.0))
            goal["gripper"] = min(min(max(grip, float(low)), float(high)), value)
        commanded = self.hold(goal)
        if self.torque_state != "on":
            raise RuntimeError(f"follower torque became {self.torque_state} during Halt")
        self.bus.disconnect_holding()
        if self.bus.connected:
            raise RuntimeError("follower bus is still open after Halt")
        if self.torque_state != "on":
            raise RuntimeError(
                f"follower torque truth became {self.torque_state} after Halt port close"
            )
        return commanded

    @property
    def connected(self) -> bool:
        return self.bus.connected

    @property
    def torque_state(self) -> TorqueState:
        """The bus's three-state torque truth (C2): "on", "off" or "unconfirmed"."""
        return bus_torque_state(self.bus)

    # --- io -------------------------------------------------------------
    def read_positions(self) -> dict[str, float]:
        return self.bus.read_positions()

    def send_goal(self, goal: dict[str, float], *, dt: float | None = None) -> dict[str, float]:
        """Clamp ``goal`` through the safety guard, write it, return what was
        actually commanded. ``dt`` is the control period for the guard's rate
        limit; without it the guard measures elapsed time itself."""
        safe = self.guard.clamp(goal, dt=dt)
        commanded = self.bus.write_positions(safe)
        self.last_goal.update(commanded)
        self.guard.record_actual_command(commanded)
        return commanded

    def hold(self, goal: dict[str, float]) -> dict[str, float]:
        """Write a Halt hold goal straight to the servos; return what was accepted.

        Not a motion command, so it skips the guard's rate limit: the caller
        builds ``goal`` from a fresh read of the present position (and the grip
        rule), so the arm stays where it is. Only stop paths call this.
        """
        commanded = self.bus.write_positions(goal)
        self.last_goal.update(commanded)
        return commanded

    def observation(self) -> dict[str, float]:
        """Flat joint observation keyed ``<joint>.pos`` (LeRobot convention)."""
        pos = self.read_positions()
        return {f"{j}.pos": pos[j] for j in JOINT_NAMES if j in pos}

    def environment_state(self) -> Any:
        """The simulator's object state (``observation.environment_state``), or None.

        Only the sim backend's bus (``MujocoBus``) knows it: the true object
        poses a privileged-state baseline reads, laid out as the task's
        ``env_state_names``. A real or mock arm returns None. The bus's class is
        checked, not the instance, so a stand-in bus never fakes the hook.
        """
        if not callable(getattr(type(self.bus), "environment_state", None)):
            return None
        return self.bus.environment_state()  # type: ignore[attr-defined]

    def __enter__(self) -> SO101Follower:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

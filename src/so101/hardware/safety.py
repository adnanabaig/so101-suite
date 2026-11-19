"""Software safety: per-joint limits, velocity clamping, and an e-stop.

Applied on *every* command path (teleop, replay, policy rollout) so no
single caller can drive the arm out of range or too fast.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import JOINT_NAMES

# Conservative default position limits in degrees, per joint. Tune per arm
# after calibration. Gripper is 0 (closed) .. 100 (open).
DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-101.0, 100.0),
    "elbow_flex": (-100.0, 99.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (0.0, 100.0),
}

# Max change per joint per control step (deg). Caps teleop/jump commands.
DEFAULT_MAX_STEP_DEG = 15.0
DEFAULT_NOMINAL_DT_S = 1.0 / 30.0
DEFAULT_MAX_DT_S = 0.25
# How far outside the envelope a measured rest pose may sit and still be held
# in place (safety-B10). An arm folded on its mechanical stop can read a few
# degrees past +-100 deg; a reading much further out is refused as implausible.
DEFAULT_REST_TOLERANCE_DEG = 30.0


class SafetyViolation(RuntimeError):
    """Raised when a command is rejected outright (not just clamped)."""


@dataclass
class SafetyGuard:
    limits: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    # Kept as a hard per-command ceiling for configuration compatibility.
    max_step_deg: float = DEFAULT_MAX_STEP_DEG
    # When omitted, derive the historical 15 deg/step at 30 Hz as 450 deg/s.
    max_velocity_deg_s: float | dict[str, float] | None = None
    nominal_dt_s: float = DEFAULT_NOMINAL_DT_S
    # A long debugger pause or process stall must not authorize an arbitrarily
    # large catch-up move on the next command.
    max_dt_s: float = DEFAULT_MAX_DT_S
    rest_tolerance_deg: float = DEFAULT_REST_TOLERANCE_DEG
    estopped: bool = False
    _last: dict[str, float] | None = None
    _last_time: float | None = None
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    # Seeded joints that rest outside the envelope, with the measured value
    # (safety-B10). Nothing is ever accepted further out than this.
    _rest: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        for joint, bounds in self.limits.items():
            if len(bounds) != 2:
                raise ValueError(f"limits for {joint!r} must be a (min, max) pair")
            lo, hi = (float(bounds[0]), float(bounds[1]))
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError(f"invalid finite limits for {joint!r}: {bounds}")
        for name, value in (
            ("max_step_deg", self.max_step_deg),
            ("nominal_dt_s", self.nominal_dt_s),
            ("max_dt_s", self.max_dt_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.rest_tolerance_deg) or self.rest_tolerance_deg < 0:
            raise ValueError("rest_tolerance_deg must be finite and non-negative")
        if isinstance(self.max_velocity_deg_s, dict):
            invalid = {
                joint: value
                for joint, value in self.max_velocity_deg_s.items()
                if not math.isfinite(float(value)) or float(value) <= 0
            }
            if invalid:
                raise ValueError(f"max_velocity_deg_s contains invalid values: {invalid}")
        elif self.max_velocity_deg_s is not None and (
            not math.isfinite(float(self.max_velocity_deg_s)) or float(self.max_velocity_deg_s) <= 0
        ):
            raise ValueError("max_velocity_deg_s must be finite and positive")

    def estop(self) -> None:
        """Engage the software e-stop. All subsequent commands are rejected."""
        self.estopped = True

    def reset(self, reference: dict[str, float] | None = None) -> None:
        """Clear e-stop and optionally seed the rate limiter from measured state."""
        self.estopped = False
        self.seed(reference)

    def seed(self, reference: dict[str, float] | None = None) -> None:
        """Seed rate-limit state without changing the latched e-stop state.

        ``reference`` is the measured pose. A joint resting outside the
        envelope, by at most ``rest_tolerance_deg``, is held where it is and
        may then only move back toward the envelope (safety-B10,
        docs/audit/v0.1/safety.md). Further out fails closed.
        """
        if reference is None:
            self._last = None
            self._rest = {}
        else:
            seeded = {joint: float(reference[joint]) for joint in JOINT_NAMES if joint in reference}
            non_finite = [joint for joint, value in seeded.items() if not math.isfinite(value)]
            if non_finite:
                raise SafetyViolation(
                    f"cannot seed safety state from non-finite joints: {non_finite}"
                )
            rest: dict[str, float] = {}
            too_far: dict[str, float] = {}
            for joint, value in seeded.items():
                low, high = self._limits_for(joint)
                overshoot = max(low - value, value - high)
                if overshoot > self.rest_tolerance_deg:
                    too_far[joint] = value
                elif overshoot > 0:
                    rest[joint] = value
            if too_far:
                raise SafetyViolation(
                    "measured pose is more than "
                    f"{self.rest_tolerance_deg:g} deg outside the safety envelope: "
                    + ", ".join(f"{joint}={value:.1f}" for joint, value in too_far.items())
                    + "; move the arm back into range by hand"
                )
            self._last = seeded
            self._rest = rest
        # The first command after reset gets one nominal control period rather
        # than a near-zero allowance caused by connect/setup latency.
        self._last_time = None

    def _limits_for(self, joint: str) -> tuple[float, float]:
        low, high = self.limits.get(joint, (-360.0, 360.0))
        return float(low), float(high)

    @property
    def outside_envelope(self) -> dict[str, float]:
        """Joints seeded outside the envelope, with their rest value (safety-B10)."""
        return dict(self._rest)

    def effective_limits(self, joint: str) -> tuple[float, float]:
        """The range a command for ``joint`` can currently reach.

        The envelope, widened toward the last accepted value of a joint that
        is still outside it, so a target there holds instead of being pulled
        to the envelope edge (safety-B10).
        """
        low, high = self._limits_for(joint)
        previous = None if self._last is None else self._last.get(joint)
        if previous is None:
            return low, high
        return min(low, previous), max(high, previous)

    def _velocity_limit(self, joint: str) -> float:
        configured = self.max_velocity_deg_s
        if configured is None:
            return self.max_step_deg / self.nominal_dt_s
        if isinstance(configured, dict):
            return float(configured.get(joint, self.max_step_deg / self.nominal_dt_s))
        return float(configured)

    def record_actual_command(self, commanded: dict[str, float]) -> None:
        """Keep the limiter anchored to what the downstream driver accepted."""

        actual: dict[str, float] = {}
        for joint, raw in commanded.items():
            if joint not in JOINT_NAMES:
                continue
            value = float(raw)
            if not math.isfinite(value):
                raise SafetyViolation(f"driver accepted non-finite command for {joint}")
            low, high = self._limits_for(joint)
            rest = self._rest.get(joint)
            if rest is not None:
                # A hold at the rest pose, or a driver clip that still lags
                # between it and the goal (safety-B10).
                low, high = min(low, rest), max(high, rest)
            if value < low or value > high:
                raise SafetyViolation(f"driver accepted out-of-range command for {joint}: {value}")
            actual[joint] = value
        self._last = {**(self._last or {}), **actual}

    def clamp(self, goal: dict[str, float], *, dt: float | None = None) -> dict[str, float]:
        """Return a safe version of ``goal``: e-stop -> raise; else clamp to
        limits and rate-limit relative to the last accepted command.

        Rate limiting is expressed in degrees/second and uses monotonic elapsed
        time. ``max_step_deg`` remains a hard ceiling for compatibility with
        existing configuration. Tests and deterministic callers may provide
        ``dt`` explicitly.
        """
        if self.estopped:
            raise SafetyViolation("E-STOP engaged; command rejected")

        requested: dict[str, float] = {}
        for joint in JOINT_NAMES:
            if joint not in goal:
                continue
            try:
                value = float(goal[joint])
            except (TypeError, ValueError) as exc:
                raise SafetyViolation(f"non-numeric command for {joint}: {goal[joint]!r}") from exc
            if not math.isfinite(value):
                raise SafetyViolation(f"non-finite command for {joint}: {value!r}")
            requested[joint] = value

        if dt is not None:
            dt_s = float(dt)
            if not math.isfinite(dt_s) or dt_s < 0:
                raise SafetyViolation(f"dt must be finite and non-negative, got {dt!r}")
        else:
            now = self._clock()
            dt_s = self.nominal_dt_s if self._last_time is None else max(0.0, now - self._last_time)
            self._last_time = now
        dt_s = min(dt_s, self.max_dt_s)

        safe: dict[str, float] = {}
        for joint in JOINT_NAMES:
            if joint not in requested:
                continue
            val = requested[joint]
            lo, hi = self._limits_for(joint)
            prev = None if self._last is None else self._last.get(joint)
            if prev is not None and prev < lo:
                # Outside the envelope: hold, or move back toward it (safety-B10).
                val = min(max(val, prev), hi)
            elif prev is not None and prev > hi:
                val = max(min(val, prev), lo)
            else:
                val = min(max(val, lo), hi)
            if prev is not None:
                delta = val - prev
                allowed_delta = min(self.max_step_deg, self._velocity_limit(joint) * dt_s)
                if abs(delta) > allowed_delta:
                    val = prev + allowed_delta * (1 if delta > 0 else -1)
            safe[joint] = val

        self._last = {**(self._last or {}), **safe}
        return safe

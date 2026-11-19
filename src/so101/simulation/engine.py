"""Thread-safe deterministic fixed-step SO-101 kinematic runtime.

The module also holds the request semantics that every simulation engine
shares, so the kinematic and physics engines cannot drift apart:

- target validation against the effective SO-101 limits;
- preview planning: preview predicts play;
- ``sample_hz`` handling, validated in one place.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .. import JOINT_NAMES
from .kinematics import JOINT_LIMITS, forward_kinematics
from .scenarios import ScenarioLibrary

# Mirrors ``SimulationTrajectoryPreviewRequest.sample_hz`` (ge=1, le=240). The
# API schema and the engines accept the same range. A rate above the engine
# tick is quantised to the tick and reported, never rejected.
MIN_PREVIEW_SAMPLE_HZ = 1.0
MAX_PREVIEW_SAMPLE_HZ = 240.0
MAX_PREVIEW_DURATION_S = 30.0


class SimulationError(RuntimeError):
    """Base simulation runtime error."""


class SimulationValidationError(SimulationError):
    """Input cannot be represented safely by the simulation contract."""


class SimulationConflict(SimulationError):
    """An operation conflicts with the current play/pause state."""


def validate_targets(targets: Mapping[str, Any]) -> dict[str, float]:
    """Check suite-unit targets against the effective limits. Never clamps."""

    if not targets:
        raise SimulationValidationError("targets must contain at least one SO-101 joint")
    unknown = sorted(set(targets) - set(JOINT_NAMES))
    if unknown:
        raise SimulationValidationError(f"unknown SO-101 joints: {', '.join(unknown)}")
    validated: dict[str, float] = {}
    for joint, raw in targets.items():
        if isinstance(raw, bool):
            raise SimulationValidationError(f"target for {joint} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SimulationValidationError(f"target for {joint} must be numeric") from exc
        if not math.isfinite(value):
            raise SimulationValidationError(f"target for {joint} must be finite")
        low, high = JOINT_LIMITS[joint]
        if not low <= value <= high:
            raise SimulationValidationError(
                f"target for {joint}={value} is outside SO-101 limits [{low}, {high}]"
            )
        validated[joint] = value
    return validated


def steps_for_duration(duration_s: float, fixed_dt_s: float) -> int:
    """Whole fixed steps covering ``duration_s``, robust to float noise.

    Without the epsilon, 4.15 s at 60 Hz rounds up to 250 steps instead of 249.
    """

    return max(0, math.ceil(duration_s / fixed_dt_s - 1e-9))


def advance_rate_limited(
    measured: Mapping[str, float],
    target: Mapping[str, float],
    max_velocity: Mapping[str, float],
    dt_s: float,
    steps: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Exact pose and velocity after ``steps`` rate-limited steps to a fixed target.

    Targets are constant over an advance, so the fixed-step result has a
    closed form. A workstation resuming after a long sleep therefore does not
    run millions of Python loops. Play, manual step and preview all use this
    function, so preview equals play by construction.
    """

    pose: dict[str, float] = {}
    velocity: dict[str, float] = {}
    for joint in JOINT_NAMES:
        previous = measured[joint]
        if steps <= 0:
            pose[joint] = previous
            velocity[joint] = 0.0
            continue
        delta = target[joint] - previous
        allowance = max_velocity[joint] * dt_s
        if abs(delta) <= allowance * steps:
            steps_to_target = max(1, math.ceil((abs(delta) / allowance) - 1e-12))
            pose[joint] = target[joint]
            if steps == steps_to_target:
                prior_motion = allowance * (steps - 1)
                last_delta = max(0.0, abs(delta) - prior_motion)
                velocity[joint] = math.copysign(last_delta / dt_s, delta)
            else:
                velocity[joint] = 0.0
        else:
            pose[joint] = previous + math.copysign(allowance * steps, delta)
            velocity[joint] = math.copysign(max_velocity[joint], delta)
    return pose, velocity


@dataclass(frozen=True)
class PreviewPlan:
    """A validated preview request, in fixed steps.

    Preview predicts play. A target is a step command at t=0, identical to
    ``set_targets`` followed by fixed steps. Waypoint ``i`` is commanded as a
    step at the start of its segment ``(t[i-1], t[i]]`` (``t[0] = 0``) and held
    until the next waypoint. Unspecified joints keep the engine's current
    commanded target, as ``set_targets`` would.
    """

    duration_s: float
    total_steps: int
    segments: tuple[tuple[int, dict[str, float]], ...]
    sample_stride: int
    requested_sample_hz: float
    sample_hz: float

    def target_at(self, step: int) -> dict[str, float]:
        """Commanded target in force during fixed step ``step`` (``0`` = at t=0)."""

        for end_step, commanded in self.segments:
            if step <= end_step:
                return commanded
        return self.segments[-1][1]

    def sampled(self, step: int) -> bool:
        return step == 0 or step % self.sample_stride == 0 or step == self.total_steps


def plan_preview(
    *,
    fixed_hz: float,
    current_target: Mapping[str, float],
    target: Mapping[str, Any] | None,
    duration_s: float | None,
    waypoints: list[dict[str, Any]] | None,
    sample_hz: float,
) -> PreviewPlan:
    """Validate a preview request once, for every engine."""

    if (target is None) == (not waypoints):
        raise SimulationValidationError("provide exactly one of target or waypoints")
    try:
        requested_hz = float(sample_hz)
    except (TypeError, ValueError) as exc:
        raise SimulationValidationError("sample_hz must be numeric") from exc
    if not math.isfinite(requested_hz) or not (
        MIN_PREVIEW_SAMPLE_HZ <= requested_hz <= MAX_PREVIEW_SAMPLE_HZ
    ):
        raise SimulationValidationError(
            f"sample_hz must be between {MIN_PREVIEW_SAMPLE_HZ:g} and {MAX_PREVIEW_SAMPLE_HZ:g}"
        )
    stride = max(1, round(fixed_hz / requested_hz))
    fixed_dt_s = 1.0 / fixed_hz

    commanded = dict(current_target)
    segments: list[tuple[int, dict[str, float]]] = []
    if target is not None:
        if duration_s is None:
            raise SimulationValidationError("duration_s is required for a target preview")
        duration = float(duration_s)
        if not math.isfinite(duration) or not 0.01 <= duration <= MAX_PREVIEW_DURATION_S:
            raise SimulationValidationError("duration_s must be between 0.01 and 30")
        commanded.update(validate_targets(target))
        total_steps = steps_for_duration(duration, fixed_dt_s)
        segments.append((total_steps, commanded))
    else:
        previous_time = 0.0
        for index, waypoint in enumerate(waypoints or []):
            try:
                waypoint_time = float(waypoint["time_s"])
                waypoint_targets = waypoint["targets"]
            except (KeyError, TypeError, ValueError) as exc:
                raise SimulationValidationError(
                    f"waypoint {index} needs numeric time_s and targets"
                ) from exc
            if not math.isfinite(waypoint_time) or not (
                previous_time < waypoint_time <= MAX_PREVIEW_DURATION_S
            ):
                raise SimulationValidationError(
                    f"waypoint {index} time_s must be increasing and at most 30"
                )
            commanded = {**commanded, **validate_targets(waypoint_targets)}
            segments.append((steps_for_duration(waypoint_time, fixed_dt_s), commanded))
            previous_time = waypoint_time
        duration = previous_time
        total_steps = segments[-1][0]
    return PreviewPlan(
        duration_s=duration,
        total_steps=total_steps,
        segments=tuple(segments),
        sample_stride=stride,
        requested_sample_hz=requested_hz,
        sample_hz=fixed_hz / stride,
    )


class SimulationEngine:
    """In-memory kinematic runtime with no hardware or persistence hooks.

    ``play`` uses a monotonic request-synchronized scheduler: whenever state is
    observed or mutated, elapsed wall time is converted to an integer number
    of fixed steps.  This avoids a background motor-like process while keeping
    playback deterministic for the same target and number of ticks.
    """

    MAX_MANUAL_STEPS = 10_000
    # ``final_reached`` tolerance, suite units. Rate-limited motion lands exactly.
    REACHED_TOLERANCE = 1e-6

    def __init__(
        self,
        library: ScenarioLibrary | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.library = library or ScenarioLibrary()
        self._clock = clock
        self._lock = threading.RLock()
        self._fixed_dt_s = 1.0 / self.library.engine.fixed_hz
        self._measured = dict(self.library.engine.initial_joint_positions)
        self._target = dict(self._measured)
        self._velocity = {joint: 0.0 for joint in JOINT_NAMES}
        self._step_count = 0
        self._playing = False
        self._playback_rate = 1.0
        self._clock_anchor = self._clock()
        self._generation = 0
        self._updated_at = datetime.now(timezone.utc)
        self._scene = self.library.instantiate(
            next(iter(item["id"] for item in self.library.catalog())),
            seed=self.library.engine.default_seed,
            randomize=True,
        )

    @property
    def fixed_dt_s(self) -> float:
        return self._fixed_dt_s

    def catalog(self) -> list[dict[str, Any]]:
        """Scenario entries labelled with this engine (``/api/simulation/scenarios``)."""

        return self.library.catalog()

    def _touch_locked(self) -> None:
        self._updated_at = datetime.now(timezone.utc)

    def _validate_targets(self, targets: Mapping[str, Any]) -> dict[str, float]:
        return validate_targets(targets)

    def _set_targets_locked(self, targets: dict[str, float]) -> None:
        validated = self._validate_targets(targets)
        self._target.update(validated)
        self._touch_locked()

    def _advance_locked(self, steps: int) -> None:
        if not steps:
            return
        self._measured, self._velocity = advance_rate_limited(
            self._measured,
            self._target,
            self.library.engine.max_velocity_deg_s,
            self._fixed_dt_s,
            steps,
        )
        self._step_count += steps
        self._touch_locked()

    def _sync_playback_locked(self) -> None:
        if not self._playing:
            return
        now = self._clock()
        elapsed = max(0.0, now - self._clock_anchor) * self._playback_rate
        steps = int((elapsed + 1e-12) / self._fixed_dt_s)
        if steps <= 0:
            return
        self._advance_locked(steps)
        # Retain sub-step elapsed time rather than resetting to ``now``.
        self._clock_anchor += (steps * self._fixed_dt_s) / self._playback_rate

    def _snapshot_locked(self) -> dict[str, Any]:
        measured = {joint: round(self._measured[joint], 9) for joint in JOINT_NAMES}
        target = {joint: round(self._target[joint], 9) for joint in JOINT_NAMES}
        velocity = {joint: round(self._velocity[joint], 9) for joint in JOINT_NAMES}
        error = {joint: round(target[joint] - measured[joint], 9) for joint in JOINT_NAMES}
        return {
            "backend": "kinematic",
            "backend_kind": "kinematic",
            "physics": False,
            "truth_model": (
                "Deterministic rate-limited joint-space motion. Forward kinematics come from "
                "the vendored Menagerie SO-101 MJCF. No contacts, collision, torque, gravity, "
                "or motor dynamics."
            ),
            "hardware_isolated": True,
            "writes_motor_commands": False,
            "status": "playing" if self._playing else "paused",
            "scheduler": "request_synchronized_monotonic_fixed_step",
            "playback_rate": self._playback_rate,
            "fixed_dt_s": self._fixed_dt_s,
            "fixed_hz": self.library.engine.fixed_hz,
            "step_count": self._step_count,
            "sim_time_s": round(self._step_count * self._fixed_dt_s, 12),
            "generation": self._generation,
            "updated_at": self._updated_at,
            "joint_order": list(JOINT_NAMES),
            "joint_limits_deg": {
                joint: [JOINT_LIMITS[joint][0], JOINT_LIMITS[joint][1]] for joint in JOINT_NAMES
            },
            "telemetry": {
                "source": "simulation",
                "position_units": "degrees",
                "velocity_units": "degrees_per_second",
                "measured": measured,
                "target": target,
                "error": error,
                "velocity": velocity,
            },
            "kinematics": forward_kinematics(measured),
            "scene": copy.deepcopy(self._scene),
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._sync_playback_locked()
            return self._snapshot_locked()

    def telemetry(self) -> dict[str, Any]:
        return self.state()["telemetry"]

    def set_targets(self, targets: dict[str, float]) -> dict[str, Any]:
        with self._lock:
            self._sync_playback_locked()
            self._set_targets_locked(targets)
            return self._snapshot_locked()

    def step(
        self,
        steps: int = 1,
        *,
        targets: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if not 1 <= steps <= self.MAX_MANUAL_STEPS:
            raise SimulationValidationError(f"steps must be between 1 and {self.MAX_MANUAL_STEPS}")
        with self._lock:
            if self._playing:
                raise SimulationConflict("pause simulation before using manual step")
            if targets is not None:
                self._set_targets_locked(targets)
            self._advance_locked(steps)
            return self._snapshot_locked()

    def play(self, rate: float = 1.0) -> dict[str, Any]:
        rate = float(rate)
        if not math.isfinite(rate) or not 0.1 <= rate <= 4.0:
            raise SimulationValidationError("playback rate must be between 0.1 and 4.0")
        with self._lock:
            self._sync_playback_locked()
            self._playback_rate = rate
            self._clock_anchor = self._clock()
            self._playing = True
            self._touch_locked()
            return self._snapshot_locked()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._sync_playback_locked()
            self._playing = False
            self._clock_anchor = self._clock()
            self._touch_locked()
            return self._snapshot_locked()

    def reset(
        self,
        scenario_id: str,
        *,
        seed: int | None = None,
        randomize: bool = True,
    ) -> dict[str, Any]:
        # Instantiate first so a bad scenario/manifest leaves current state intact.
        scene = self.library.instantiate(scenario_id, seed=seed, randomize=randomize)
        with self._lock:
            self._playing = False
            self._playback_rate = 1.0
            self._clock_anchor = self._clock()
            self._measured = dict(self.library.engine.initial_joint_positions)
            self._target = dict(self._measured)
            self._velocity = {joint: 0.0 for joint in JOINT_NAMES}
            self._step_count = 0
            self._generation += 1
            self._scene = scene
            self._touch_locked()
            return self._snapshot_locked()

    def preview_trajectory(
        self,
        *,
        target: dict[str, float] | None = None,
        duration_s: float | None = None,
        waypoints: list[dict[str, Any]] | None = None,
        sample_hz: float = 20.0,
    ) -> dict[str, Any]:
        """Predict exactly what play would do, without mutating state.

        See :class:`PreviewPlan` for the shared semantics. Each sample is the
        state that ``set_targets`` plus ``step`` would produce at that step.
        """

        with self._lock:
            self._sync_playback_locked()
            start = dict(self._measured)
            state_target = dict(self._target)
            source_step = self._step_count
        plan = plan_preview(
            fixed_hz=self.library.engine.fixed_hz,
            current_target=state_target,
            target=target,
            duration_s=duration_s,
            waypoints=waypoints,
            sample_hz=sample_hz,
        )

        def point(step_number: int, measured: dict[str, float]) -> dict[str, Any]:
            commanded = plan.target_at(step_number)
            measured_out = {joint: round(measured[joint], 9) for joint in JOINT_NAMES}
            target_out = {joint: round(commanded[joint], 9) for joint in JOINT_NAMES}
            return {
                "step": step_number,
                "time_s": round(step_number * self._fixed_dt_s, 12),
                "measured": measured_out,
                "target": target_out,
                "error": {
                    joint: round(target_out[joint] - measured_out[joint], 9)
                    for joint in JOINT_NAMES
                },
                "end_effector": forward_kinematics(measured_out),
            }

        points = [point(0, start)]
        segment_start_step = 0
        segment_start_pose = start
        measured = start
        for end_step, commanded in plan.segments:
            # Sample inside the segment from its start pose with the closed
            # form, which is exactly what play computes between commands.
            for step_number in range(segment_start_step + 1, end_step + 1):
                if plan.sampled(step_number):
                    measured, _ = advance_rate_limited(
                        segment_start_pose,
                        commanded,
                        self.library.engine.max_velocity_deg_s,
                        self._fixed_dt_s,
                        step_number - segment_start_step,
                    )
                    points.append(point(step_number, measured))
            segment_start_pose, _ = advance_rate_limited(
                segment_start_pose,
                commanded,
                self.library.engine.max_velocity_deg_s,
                self._fixed_dt_s,
                end_step - segment_start_step,
            )
            segment_start_step = end_step
        measured = segment_start_pose
        final_target = plan.segments[-1][1]
        final_error = {joint: final_target[joint] - measured[joint] for joint in JOINT_NAMES}
        return {
            "backend": "kinematic",
            "physics": False,
            "mutates_state": False,
            "semantics": "step_targets_like_play",
            "source_step_count": source_step,
            "fixed_dt_s": self._fixed_dt_s,
            "requested_duration_s": plan.duration_s,
            "quantized_duration_s": round(plan.total_steps * self._fixed_dt_s, 12),
            "requested_sample_hz": plan.requested_sample_hz,
            "sample_hz": plan.sample_hz,
            "points": points,
            "final_reached": all(
                abs(value) <= self.REACHED_TOLERANCE for value in final_error.values()
            ),
            "reached_tolerance": self.REACHED_TOLERANCE,
            "joint_limits_deg": {
                joint: [JOINT_LIMITS[joint][0], JOINT_LIMITS[joint][1]] for joint in JOINT_NAMES
            },
            "starting_target": state_target,
        }

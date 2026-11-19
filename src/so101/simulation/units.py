"""Suite units, MJCF units and the effective SO-101 joint limits, in one place.

Suite units are used by the motor, safety and API layers:

- The five arm joints are in degrees. Zero is the middle of the range (the
  "new calibration" zero that LeRobot uses with ``use_degrees=True``), which is
  also the zero of the vendored MJCF.
- The gripper runs from 0 (closed) to 100 (open).

In MJCF units every joint is a hinge in radians. The gripper maps linearly
onto its position actuator's ``ctrlrange``, roughly -0.1745 to 1.7453 rad.
Commanding 0 therefore squeezes slightly past jaw contact, which is what
produces grip force.

The mapping is ``mjcf_rad = scale * suite + offset``:

- arm joints: ``scale = pi/180`` and ``offset = 0``;
- gripper: ``scale = (hi - lo) / 100`` and ``offset = lo``.

Effective limits are the safety envelope intersected with what the model can
reach. Each MJCF bound uses the tighter of the joint ``range`` and the
actuator ``ctrlrange``, is converted to suite units and rounded to 0.01. It is
then intersected with ``SafetyGuard.DEFAULT_LIMITS``. Rounding can widen a
bound by at most 0.005 suite units. MuJoCo still clamps ``ctrl`` to the
``ctrlrange``, so the physics never leaves the model's range.

Sign agreement with a calibrated real arm is an HIL check, not an assumption.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from .. import JOINT_NAMES
from ..hardware.safety import DEFAULT_LIMITS
from .mjcf import MjcfModel, so101_model

GRIPPER_JOINT = "gripper"
ARM_JOINTS: tuple[str, ...] = tuple(joint for joint in JOINT_NAMES if joint != GRIPPER_JOINT)
LIMIT_DECIMALS = 2


class UnitsError(RuntimeError):
    """The vendored model cannot be mapped onto suite units."""


class JointOrderError(UnitsError):
    """The MJCF joint or actuator order differs from ``so101.JOINT_NAMES``."""


@dataclass(frozen=True)
class JointMap:
    """``mjcf_rad = scale * suite + offset`` for one joint."""

    suite_joint: str
    mjcf_joint: str
    kind: Literal["revolute_deg", "gripper_linear"]
    scale: float
    offset: float

    def to_mjcf(self, suite: float) -> float:
        return self.scale * float(suite) + self.offset

    def to_suite(self, mjcf_rad: float) -> float:
        return (float(mjcf_rad) - self.offset) / self.scale

    def as_dict(self) -> dict[str, object]:
        return {
            "mjcf_joint": self.mjcf_joint,
            "kind": self.kind,
            "scale": self.scale,
            "offset": self.offset,
        }


def assert_joint_order(
    joint_names: Sequence[str],
    actuator_names: Sequence[str],
    actuator_joints: Sequence[str] | None = None,
) -> None:
    """Fail loudly unless the model's joints and actuators are ``JOINT_NAMES``, in order.

    Every array conversion in the simulation relies on this ordering, so it is
    checked when a model is loaded, not assumed.
    """

    expected = list(JOINT_NAMES)
    if list(joint_names) != expected:
        raise JointOrderError(f"MJCF joints {list(joint_names)} != JOINT_NAMES {expected}")
    if list(actuator_names) != expected:
        raise JointOrderError(f"MJCF actuators {list(actuator_names)} != JOINT_NAMES {expected}")
    if actuator_joints is not None and list(actuator_joints) != expected:
        raise JointOrderError(
            f"MJCF actuators drive {list(actuator_joints)}, expected {expected} in order"
        )


def _check_model(model: MjcfModel) -> None:
    assert_joint_order(
        [joint.name for joint in model.joints],
        [actuator.name for actuator in model.actuators],
        [actuator.joint for actuator in model.actuators],
    )


@lru_cache(maxsize=1)
def joint_maps() -> dict[str, JointMap]:
    """Suite joint -> MJCF mapping, derived from the vendored model."""

    model = so101_model()
    _check_model(model)
    maps: dict[str, JointMap] = {}
    for joint in JOINT_NAMES:
        if joint == GRIPPER_JOINT:
            ctrlrange = model.actuator(joint).ctrlrange
            if ctrlrange is None:
                raise UnitsError("the gripper actuator needs a ctrlrange to map 0..100")
            low, high = ctrlrange
            maps[joint] = JointMap(joint, joint, "gripper_linear", (high - low) / 100.0, low)
        else:
            maps[joint] = JointMap(joint, joint, "revolute_deg", math.pi / 180.0, 0.0)
    return maps


def suite_to_mjcf(joints: Mapping[str, float]) -> dict[str, float]:
    """Convert suite units (degrees, gripper 0..100) to MJCF radians, by name."""

    maps = joint_maps()
    return {joint: maps[joint].to_mjcf(value) for joint, value in joints.items()}


def mjcf_to_suite(joints: Mapping[str, float]) -> dict[str, float]:
    """Convert MJCF radians to suite units, by name."""

    maps = joint_maps()
    return {joint: maps[joint].to_suite(value) for joint, value in joints.items()}


@lru_cache(maxsize=1)
def _vector_coefficients() -> tuple[np.ndarray, np.ndarray]:
    maps = joint_maps()
    scale = np.array([maps[joint].scale for joint in JOINT_NAMES], dtype=np.float64)
    offset = np.array([maps[joint].offset for joint in JOINT_NAMES], dtype=np.float64)
    scale.setflags(write=False)
    offset.setflags(write=False)
    return scale, offset


def suite_to_mjcf_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Vectorised :func:`suite_to_mjcf` for arrays in ``JOINT_NAMES`` order."""

    scale, offset = _vector_coefficients()
    return np.asarray(values, dtype=np.float64) * scale + offset


def mjcf_to_suite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Vectorised :func:`mjcf_to_suite` for arrays in ``JOINT_NAMES`` order."""

    scale, offset = _vector_coefficients()
    return (np.asarray(values, dtype=np.float64) - offset) / scale


def velocity_suite_to_mjcf(joint: str, suite_per_s: float) -> float:
    """Convert a rate (suite units per second) to rad/s."""

    return joint_maps()[joint].scale * float(suite_per_s)


def velocity_mjcf_to_suite(joint: str, rad_per_s: float) -> float:
    """Convert a rate in rad/s to suite units per second."""

    return float(rad_per_s) / joint_maps()[joint].scale


def _rounded(value: float) -> float:
    return round(value, LIMIT_DECIMALS) + 0.0  # ``+ 0.0`` folds -0.0 into 0.0


@lru_cache(maxsize=1)
def mjcf_limits() -> dict[str, tuple[float, float]]:
    """What the model can reach, in suite units, before the safety intersection."""

    model = so101_model()
    _check_model(model)
    maps = joint_maps()
    limits: dict[str, tuple[float, float]] = {}
    for joint in JOINT_NAMES:
        bounds: list[tuple[float, float]] = []
        joint_range = model.joint(joint).range
        if joint_range is not None:
            bounds.append(joint_range)
        ctrlrange = model.actuator(joint).ctrlrange
        if ctrlrange is not None:
            bounds.append(ctrlrange)
        if not bounds:
            limits[joint] = (-math.inf, math.inf)
            continue
        low_rad = max(low for low, _ in bounds)
        high_rad = min(high for _, high in bounds)
        limit_map = maps[joint]
        limits[joint] = (
            _rounded(limit_map.to_suite(low_rad)),
            _rounded(limit_map.to_suite(high_rad)),
        )
    return limits


@lru_cache(maxsize=1)
def effective_limits() -> dict[str, tuple[float, float]]:
    """``SafetyGuard.DEFAULT_LIMITS`` intersected with :func:`mjcf_limits`."""

    model_limits = mjcf_limits()
    limits: dict[str, tuple[float, float]] = {}
    for joint in JOINT_NAMES:
        safe_low, safe_high = DEFAULT_LIMITS[joint]
        model_low, model_high = model_limits[joint]
        low, high = max(float(safe_low), model_low), min(float(safe_high), model_high)
        if low >= high:
            raise UnitsError(f"{joint}: safety and model ranges do not overlap")
        limits[joint] = (low, high)
    return limits


@lru_cache(maxsize=1)
def effective_limit_arrays() -> tuple[np.ndarray, np.ndarray]:
    """:func:`effective_limits` as read-only ``(low, high)`` arrays, ``JOINT_NAMES`` order."""

    limits = effective_limits()
    low = np.array([limits[joint][0] for joint in JOINT_NAMES], dtype=float)
    high = np.array([limits[joint][1] for joint in JOINT_NAMES], dtype=float)
    low.setflags(write=False)
    high.setflags(write=False)
    return low, high

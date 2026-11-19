"""Convert read-only encoder/register observations using the installed reference."""

from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any

from .. import JOINT_NAMES

_pose_lock = threading.Lock()
_poses: dict[str, dict[str, Any]] = {}


def remember_pose(arm: str, pose: dict[str, Any]) -> None:
    with _pose_lock:
        _poses[arm] = copy.deepcopy(pose)


def cached_pose(arm: str) -> dict[str, Any] | None:
    with _pose_lock:
        return copy.deepcopy(_poses.get(arm))


def read_calibration_pose(bus: Any, calibration: dict[str, Any]) -> dict[str, Any]:
    observation: dict[str, float] = {}
    joints = {}
    for motor_id, joint in enumerate(JOINT_NAMES, 1):
        reference = calibration[joint]
        if reference["id"] != motor_id:
            raise ValueError(f"{joint}: installed motor ID differs from expected ID {motor_id}")
        if bus.read("Torque_Enable", joint, normalize=False) != 0:
            raise ValueError(f"{joint}: torque must already be OFF; no torque change was made")
        registers = {
            field: bus.read(register, joint, normalize=False)
            for register, field in (
                ("Homing_Offset", "homing_offset"),
                ("Min_Position_Limit", "range_min"),
                ("Max_Position_Limit", "range_max"),
            )
        }
        value = bus.read("Present_Position", joint, normalize=False)
        issues = []
        for field, actual in registers.items():
            if actual != reference[field]:
                issues.append(f"{field} differs from installed calibration")
        low, high = reference["range_min"], reference["range_max"]
        valid_value = (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
        if not (0 <= low < high <= 4095 and high - low >= 100):
            issues.append("Installed encoder range is invalid")
        elif not valid_value or not low <= value <= high:
            issues.append("Current encoder reading is outside the installed range")
        if not issues:
            observation[f"{joint}.pos"] = (
                (value - low) * 100 / (high - low)
                if joint == "gripper"
                else (value - (low + high) / 2) * 360 / 4095
            )
        joints[joint] = {
            "motor_id": motor_id,
            "ticks": value if valid_value else None,
            "range_min": low,
            "range_max": high,
            "issues": issues,
        }
    for joint in JOINT_NAMES:
        if bus.read("Torque_Enable", joint, normalize=False) != 0:
            raise ValueError(f"{joint}: torque changed during pose read; result discarded")
    return {"observation": observation, "joints": joints, "captured_at": time.time()}

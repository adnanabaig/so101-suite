"""Evaluate the vendored model from cached telemetry; never reads a motor bus."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .. import JOINT_NAMES
from ..simulation.mjcf import mat_to_quat, so101_model
from ..simulation.units import suite_to_mjcf


def encoder_model(telemetry: dict[str, Any]) -> dict[str, Any]:
    observation = telemetry.get("observation", {})
    joints = {
        joint: observation.get(f"{joint}.pos", observation.get(joint)) for joint in JOINT_NAMES
    }
    complete = all(
        isinstance(value, (float, int)) and math.isfinite(value) for value in joints.values()
    )
    live = (
        complete
        and telemetry.get("observation_live") is True
        and 0 <= telemetry.get("sample_age_ms", math.inf) <= 500
    )
    result: dict[str, Any] = {
        "source": "motor_encoders",
        "live": live,
        "complete": complete,
        "sample_age_ms": telemetry.get("sample_age_ms"),
        "sample_seq": telemetry.get("sample_seq"),
        "measured": joints,
        "physical_alignment_verified": False,
        "models": [],
        "motors": [],
    }
    if not complete:
        return result
    model = so101_model()
    poses = model.body_poses(suite_to_mjcf(joints))
    for body in model.bodies:
        if body.joint is not None:
            name = body.joint.name
            result["motors"].append(
                {
                    "id": JOINT_NAMES.index(name) + 1,
                    "name": name,
                    "position_m": [float(value) for value in poses[body.name].pos],
                }
            )
        for index, geom in enumerate(body.geoms):
            if not geom.visual or geom.kind != "mesh" or geom.mesh is None:
                continue
            pose = poses[body.name].apply(geom.pos, geom.quat)
            result["models"].append(
                {
                    "id": f"{body.name}-{index}",
                    "mesh": Path(model.meshes[geom.mesh]).stem,
                    "position": [float(value) for value in pose.pos],
                    "quaternion_wxyz": [float(value) for value in mat_to_quat(pose.rot)],
                    "rgba": [0.1, 0.8, 1, 1 if live else 0.25],
                }
            )
    return result

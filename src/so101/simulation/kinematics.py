"""SO-101 forward kinematics and model description, in suite units.

:func:`forward_kinematics` is the single FK used by the kinematic engine,
scenario telemetry, the trajectory preview and the physics engine's
``kinematics`` field. It evaluates the vendored Menagerie SO-101 body tree
with numpy (see :mod:`so101.simulation.mjcf`). The result matches MuJoCo's
``mj_kinematics`` to floating-point precision and needs no MuJoCo install.

Angles are degrees at the API boundary, matching the SO-101 motor and safety
APIs. Cartesian outputs are in metres, in the MJCF world frame: x forward,
y left, z up, origin at the robot base.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .. import JOINT_NAMES
from .mjcf import (
    MODEL_COMMIT,
    MODEL_LICENSE,
    MODEL_REPO,
    MODEL_REPO_PATH,
    GeomSpec,
    mat_to_quat,
    mat_to_rpy,
    so101_model,
)
from .units import effective_limits, joint_maps, suite_to_mjcf

TOOL_SITE = "gripperframe"
MODEL_ID = f"mjcf:{MODEL_REPO_PATH}@{MODEL_COMMIT[:7]}"
# The jaw-tip contact spheres. The aperture is the surface gap between them.
_FIXED_TIP_GEOM = "fixed_jaw_sph_tip1"
_MOVING_TIP_GEOM = "moving_jaw_sph_tip1"
_PRIMITIVE_SIZE_ENTRIES = {"box": 3, "capsule": 2, "cylinder": 2, "sphere": 1, "ellipsoid": 3}

JOINT_LIMITS: dict[str, tuple[float, float]] = dict(effective_limits())


def default_joint_positions() -> dict[str, float]:
    """Return a neutral, in-range SO-101 pose."""

    return {joint: 50.0 if joint == "gripper" else 0.0 for joint in JOINT_NAMES}


def _rounded(values: Any, digits: int = 9) -> list[float]:
    return [round(float(value), digits) + 0.0 for value in values]


def forward_kinematics(joints_deg: Mapping[str, float]) -> dict[str, Any]:
    """FK for a full joint map in suite units (degrees, gripper 0..100).

    The keys are:

    - ``position_m``: the ``gripperframe`` tool site.
    - ``orientation_rpy_deg``: the tool site's extrinsic X-Y-Z (URDF ``rpy``)
      angles.
    - ``orientation_quat_wxyz``: the same orientation as a quaternion.
    - ``gripper_aperture_m``: the surface gap between the two jaw-tip contact
      spheres.
    - ``frames``: every body origin, plus the tool site.
    """

    model = so101_model()
    qpos = suite_to_mjcf({joint: float(joints_deg[joint]) for joint in JOINT_NAMES})
    poses = model.body_poses(qpos)
    site = model.site(TOOL_SITE)
    tool = poses[site.body].apply(site.pos, site.quat)

    fixed_body, fixed_tip = model.geom(_FIXED_TIP_GEOM)
    moving_body, moving_tip = model.geom(_MOVING_TIP_GEOM)
    fixed_center = poses[fixed_body.name].apply(fixed_tip.pos).pos
    moving_center = poses[moving_body.name].apply(moving_tip.pos).pos
    gap = float(np.linalg.norm(fixed_center - moving_center))
    gap -= fixed_tip.size[0] + moving_tip.size[0]

    frames = [
        {"name": body.name, "position_m": _rounded(poses[body.name].pos)} for body in model.bodies
    ]
    frames.append({"name": TOOL_SITE, "position_m": _rounded(tool.pos)})
    return {
        "model": MODEL_ID,
        "frame": "base",
        "tool_frame": TOOL_SITE,
        "position_m": _rounded(tool.pos),
        "orientation_rpy_deg": _rounded(math.degrees(angle) for angle in mat_to_rpy(tool.rot)),
        "orientation_quat_wxyz": _rounded(mat_to_quat(tool.rot)),
        "gripper_aperture_m": round(max(gap, 0.0), 9),
        "frames": frames,
    }


def _geom_json(geom: GeomSpec) -> dict[str, Any]:
    common = {
        "pos": list(geom.pos),
        "quat": list(geom.quat),
        "rgba": list(geom.rgba),
    }
    if geom.kind == "mesh":
        return {"kind": "mesh", "mesh": geom.mesh, **common}
    entries = _PRIMITIVE_SIZE_ENTRIES.get(geom.kind)
    if entries is None:
        raise ValueError(f"visual geom kind {geom.kind!r} is not described by the model API")
    return {"kind": geom.kind, "size": list(geom.size[:entries]), **common}


def model_description(asset_url_prefix: str = "/api/simulation/assets") -> dict[str, Any]:
    """The SO-101 kinematic tree served by ``GET /api/simulation/model``.

    The description is derived from the vendored MJCF and does not need
    MuJoCo. Only visual geoms (``contype == conaffinity == 0``) are listed.
    Mesh geoms keep their raw MJCF ``pos``/``quat``, which place the raw STL
    vertices. MuJoCo's internal mesh recentring is deliberately not applied.
    """

    model = so101_model()
    visual_meshes = model.visual_mesh_files()
    bodies = []
    for body in model.bodies:
        joint = body.joint
        if joint is not None and (any(joint.pos) or joint.ref):
            # The published transform convention has no joint anchor or ref.
            raise ValueError(f"joint {joint.name} has an anchor/ref the model API cannot express")
        bodies.append(
            {
                "name": body.name,
                "parent": body.parent,
                "pos": list(body.pos),
                "quat": list(body.quat),
                "joint": None
                if joint is None
                else {
                    "name": joint.name,
                    "axis": list(joint.axis),
                    "range": None if joint.range is None else list(joint.range),
                },
                "geoms": [_geom_json(geom) for geom in body.geoms if geom.visual],
            }
        )
    return {
        "name": model.name,
        "source": {
            "repo": MODEL_REPO,
            "path": MODEL_REPO_PATH,
            "commit": MODEL_COMMIT,
            "license": MODEL_LICENSE,
        },
        "units": {"length": "m", "angle": "rad"},
        "up_axis": "z",
        "bodies": bodies,
        "meshes": {
            name: f"{asset_url_prefix}/{file}" for name, file in sorted(visual_meshes.items())
        },
        "sites": {
            site.name: {"body": site.body, "pos": list(site.pos), "quat": list(site.quat)}
            for site in model.sites
        },
        "cameras": {
            camera.name: {
                "body": camera.body,
                "pos": list(camera.pos),
                "quat": list(camera.quat),
                "fovy_deg": camera.fovy_deg,
            }
            for camera in model.cameras
        },
        "joint_map": {joint: joint_maps()[joint].as_dict() for joint in JOINT_NAMES},
        "joint_limits_suite": {joint: list(JOINT_LIMITS[joint]) for joint in JOINT_NAMES},
        "conventions": {
            "frame": "MJCF world = robot base frame; x forward, y left, z up",
            "quat": "unit quaternion [w, x, y, z]",
            "body_transform": (
                "world_T_body = world_T_parent * T(pos) * R(quat) * R(joint.axis, mjcf_rad)"
            ),
            "joint_map": "mjcf_rad = scale * suite + offset",
            "suite_units": "degrees for arm joints; gripper 0 (closed) to 100 (open)",
            "geom_size": (
                "MJCF half-sizes: box [hx, hy, hz], cylinder/capsule [radius, half_length], "
                "sphere [radius]"
            ),
            "mesh": (
                "STL vertices in metres, placed by the geom pos/quat in the body frame "
                "(raw MJCF frames, no recentring)"
            ),
            "tool_site": TOOL_SITE,
        },
    }

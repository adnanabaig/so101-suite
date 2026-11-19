#!/usr/bin/env python3
"""Regenerate ``tests/fixtures/so101_fk_golden.json`` from MuJoCo itself.

This is a dev-only script and needs the ``[sim]`` extra (MuJoCo). It sets
``qpos`` for a deterministic table of SO-101 poses, runs ``mj_kinematics`` on
the vendored model, and records:

- the ``gripperframe`` site pose;
- every body origin;
- both jaw-tip sphere centres.

``tests/test_sim_physics_kinematics.py`` then checks the pure-numpy FK in
``so101.simulation.mjcf`` against this table, with or without MuJoCo
installed.

    PYTHONPATH=src python scripts/generate_fk_golden.py          # rewrite
    PYTHONPATH=src python scripts/generate_fk_golden.py --check  # verify only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from so101 import JOINT_NAMES
from so101.simulation.mjcf import MODEL_COMMIT, MODEL_XML, mat_to_quat
from so101.simulation.units import effective_limits, suite_to_mjcf

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "so101_fk_golden.json"
SEED = 20261001
RANDOM_POSES = 40
TIP_GEOMS = ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")
KINEMATIC_ROUNDOFF_ATOL = 1e-12


def pose_table() -> list[dict[str, float]]:
    limits = effective_limits()
    neutral = {joint: 0.0 for joint in JOINT_NAMES} | {"gripper": 50.0}
    poses = [dict(neutral), {joint: 0.0 for joint in JOINT_NAMES}]
    for joint in JOINT_NAMES:
        for bound in limits[joint]:
            poses.append(neutral | {joint: bound})
    rng = np.random.default_rng(SEED)
    for _ in range(RANDOM_POSES):
        poses.append({joint: round(float(rng.uniform(*limits[joint])), 4) for joint in JOINT_NAMES})
    return poses


def build() -> dict[str, object]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    site_id = model.site("gripperframe").id
    joint_qpos = [int(model.jnt_qposadr[model.joint(joint).id]) for joint in JOINT_NAMES]
    body_names = [model.body(index).name for index in range(1, model.nbody)]
    entries = []
    for suite in pose_table():
        qpos = suite_to_mjcf(suite)
        data.qpos[:] = 0.0
        for joint, address in zip(JOINT_NAMES, joint_qpos, strict=True):
            data.qpos[address] = qpos[joint]
        mujoco.mj_kinematics(model, data)
        entries.append(
            {
                "suite": suite,
                "qpos_rad": [qpos[joint] for joint in JOINT_NAMES],
                "gripperframe": {
                    "pos": data.site_xpos[site_id].tolist(),
                    "quat_wxyz": list(mat_to_quat(data.site_xmat[site_id].reshape(3, 3))),
                },
                "bodies": {name: data.xpos[model.body(name).id].tolist() for name in body_names},
                "jaw_tips": {
                    name: data.geom_xpos[model.geom(name).id].tolist() for name in TIP_GEOMS
                },
            }
        )
    return {
        "description": (
            "SO-101 FK golden table: MuJoCo mj_kinematics on the vendored Menagerie model. "
            "Regenerate with scripts/generate_fk_golden.py."
        ),
        "model_commit": MODEL_COMMIT,
        "mujoco_version": mujoco.__version__,
        "joint_order": list(JOINT_NAMES),
        "suite_units": "degrees; gripper 0 (closed) to 100 (open)",
        "frame": "MJCF world (robot base), metres",
        "tolerance_m": 1e-4,
        "poses": entries,
    }


def _dump(payload: dict[str, object]) -> str:
    """Readable diffs: metadata pretty-printed, one compact line per pose."""

    head = {key: value for key, value in payload.items() if key != "poses"}
    lines = json.dumps(head, indent=2)[:-2] + ',\n  "poses": [\n'
    poses = payload["poses"]
    assert isinstance(poses, list)
    lines += ",\n".join("    " + json.dumps(pose, separators=(",", ":")) for pose in poses)
    return lines + "\n  ]\n}\n"


def _comparable(payload: dict[str, object]) -> dict[str, object]:
    # The MuJoCo version is provenance, not part of the kinematic truth.
    return {key: value for key, value in payload.items() if key != "mujoco_version"}


def _kinematics_close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _kinematics_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _kinematics_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=0, abs_tol=KINEMATIC_ROUNDOFF_ATOL)
        )
    return False


def equivalent(left: dict[str, object], right: dict[str, object]) -> bool:
    """Keep metadata/pose inputs exact; allow only 1e-12 FK output roundoff."""
    left, right = _comparable(left), _comparable(right)
    if {key: value for key, value in left.items() if key != "poses"} != {
        key: value for key, value in right.items() if key != "poses"
    }:
        return False
    left_poses, right_poses = left.get("poses"), right.get("poses")
    if not isinstance(left_poses, list) or not isinstance(right_poses, list):
        return False
    if len(left_poses) != len(right_poses):
        return False
    for original, generated in zip(left_poses, right_poses, strict=True):
        if not isinstance(original, dict) or not isinstance(generated, dict):
            return False
        if original.keys() != generated.keys():
            return False
        for key in ("suite", "qpos_rad"):
            if original.get(key) != generated.get(key):
                return False
        for key in original.keys() - {"suite", "qpos_rad"}:
            if not _kinematics_close(original[key], generated[key]):
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the fixture is stale")
    args = parser.parse_args(argv)
    payload = build()
    if args.check:
        current = json.loads(FIXTURE.read_text(encoding="utf-8"))
        stale = not equivalent(current, json.loads(json.dumps(payload)))
        print(f"{FIXTURE}: {'STALE' if stale else 'up to date'}")
        return 1 if stale else 0
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(_dump(payload), encoding="utf-8")
    print(f"wrote {FIXTURE} ({len(payload['poses'])} poses)")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Model-derived SO-101 rest pose, and a physics check for any rest pose.

``MODEL_REST_POSE`` is plain data. Importing this module never imports
MuJoCo, so ``hardware/rest_pose.py`` can use it as its default rest pose.
:func:`check_rest_pose` needs the ``[sim]`` extra.

Provenance
----------
Derived by ``scripts/derive_rest_pose.py`` on the vendored Menagerie SO-101
(``so101.simulation.mjcf.MODEL_COMMIT``) with MuJoCo 3.14 at the 1/240 s
scene timestep. The criteria are those of design section 5:

- folded and compact;
- collision-free at the commanded pose;
- with actuation off, every joint settles by less than 3 deg within 2 s;
- every target at least 5 deg (gripper: 5 units) inside the effective limits,
  or at least 2 deg short of the mechanical stop the joint settles onto;
- gripper closed or nearly closed.

Why the elbow rests on a stop: in every folded pose gravity loads the elbow
with 0.33-0.48 N m, 6-9 times the 0.052 N m friction loss of the STS3215
model, so with actuation off the forearm always falls until something holds
it. Here the forearm comes to rest against the shoulder bracket
(``lower_arm|shoulder`` contact) at elbow 95.3 deg, 1.5 deg before the joint
range ends. The pose commands the elbow 2.5 deg short of that contact. Lift
(|tau_g| <= 0.06 N m, partly carried by the same contact), wrist (gripper
pointing up, |tau_g| ~ 0.002 N m) and the vertical pan axis stay put on
friction. Nothing is near the table, so a +-2 deg error on lift, elbow or
wrist never commands a collision.

Measured by ``check_rest_pose(MODEL_REST_POSE)`` (MuJoCo 3.14, 2 s window):

==============  ======  ===========  =======  ======================
joint           target  limit margin settled  settle (actuation off)
==============  ======  ===========  =======  ======================
shoulder_pan       0.0       110.00     0.00    0.00
shoulder_lift    -95.0         5.00   -95.37    0.37
elbow_flex        92.8         4.03    95.30    2.65 (onto its stop)
wrist_flex       -85.0        10.00   -84.96    0.04
wrist_roll         0.0       157.21     0.00    0.00
gripper            5.0         5.00     5.00    0.00
==============  ======  ===========  =======  ======================

Commanded pose: no contacts. Tool at (0.085, -0.001, 0.305) m; the whole arm
within 0.166 m of the base axis (0.401 m fully extended) and 0.313 m high.

A narrow window, by construction: an elbow that settles onto a stop must be
commanded at least 2 deg short of it (no pushing into the stop) and at most
3 deg short (settle < 3 deg), so its target has a 1 deg window, 92.3-93.3 deg
here. Every ``-2 deg`` elbow error in the robustness sweep therefore settles
4.7 deg onto the stop (9 of 27 cases); no other joint exceeds 1.4 deg. A scan
of 20,720 folded poses (``scripts/derive_rest_pose.py``) found 178 that pass,
and none with fewer perturbed settle failures.

``0,-95,90,70,0,0``, the example the bring-up docs used to give, presses the
gripper into the shoulder bracket and the table at the commanded pose in this
model, and gripper 0 sits on the closed stop, so it fails the criteria. The
docs now give ``MODEL_REST_POSE`` (``tests/test_rest_pose_examples.py``).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .. import JOINT_NAMES

MODEL_REST_POSE: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -95.0,
    "elbow_flex": 92.8,
    "wrist_flex": -85.0,
    "wrist_roll": 0.0,
    "gripper": 5.0,
}

LIMIT_MARGIN = 5.0  # suite units (degrees; gripper percent) inside the effective limits
STOP_CLEARANCE = 2.0  # or this far short of the stop the joint settles onto
SETTLE_LIMIT = 3.0  # largest drift with actuation off ...
SETTLE_WINDOW_S = 2.0  # ... within this window
GRIPPER_CLOSED_MAX = 10.0  # "closed or nearly closed", suite units
STOP_SUPPORT_NM = 0.02  # contact or joint-limit torque that means "resting on a stop"
PERTURBATION_DEG = 2.0  # calibration error the pose should tolerate
PERTURBED_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
COMPACT_REACH_M = 0.25  # whole arm within this radius of the base axis ...
COMPACT_HEIGHT_M = 0.35  # ... and below this height above the table


@dataclass(frozen=True)
class JointRestCheck:
    """One joint of a rest-pose check. Angles in suite units."""

    target: float
    limits: tuple[float, float]
    limit_margin: float  # distance from the target to the nearer effective limit
    settled: float  # position after the settle window, actuation off
    settle: float  # largest |position - target| during the window
    support_nm: float  # contact + joint-limit torque holding the joint at the end
    on_stop: bool  # the passive equilibrium is a mechanical stop
    margin_ok: bool


@dataclass(frozen=True)
class RestPoseReport:
    pose: dict[str, float]
    joints: dict[str, JointRestCheck]
    commanded_contacts: tuple[str, ...]  # "body_a|body_b" pairs touching at the target
    settled_contacts: tuple[str, ...]  # ... and after the settle window
    tool_position_m: tuple[float, float, float]  # gripperframe at the target
    reach_m: float  # furthest arm point from the base axis
    height_m: float  # highest arm point above the table
    robustness: dict[str, Any]
    checks: dict[str, bool]

    @property
    def ok(self) -> bool:
        return all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def check_rest_pose(
    pose: Mapping[str, float] | None = None,
    *,
    settle_window_s: float = SETTLE_WINDOW_S,
    perturbation_deg: float = PERTURBATION_DEG,
) -> RestPoseReport:
    """Check ``pose`` (default :data:`MODEL_REST_POSE`) against the rest-pose criteria.

    Simulates the vendored model on an empty table: the arm holds the pose,
    actuation is switched off, and the joints are watched for
    ``settle_window_s``. ``robustness`` repeats the collision and settle checks
    for every combination of ``-perturbation_deg``, ``0`` and
    ``+perturbation_deg`` on lift, elbow and wrist (0 skips it).

    Needs the ``[sim]`` extra.
    """

    target = dict(MODEL_REST_POSE if pose is None else pose)
    if set(target) != set(JOINT_NAMES):
        raise ValueError(f"a rest pose needs exactly the joints {list(JOINT_NAMES)}")

    from .units import effective_limits
    from .world import PhysicsWorld

    world = PhysicsWorld()
    limits = effective_limits()
    trial = _settle(world, target, settle_window_s)

    joints: dict[str, JointRestCheck] = {}
    for slot, joint in enumerate(JOINT_NAMES):
        low, high = limits[joint]
        value = float(target[joint])
        margin = min(value - low, high - value)
        settled = float(trial["settled"][slot])
        on_stop = abs(trial["support_nm"][slot]) >= STOP_SUPPORT_NM
        clearance = abs(settled - value)
        margin_ok = margin >= LIMIT_MARGIN or (
            on_stop and margin >= STOP_CLEARANCE and clearance >= STOP_CLEARANCE
        )
        joints[joint] = JointRestCheck(
            target=value,
            limits=(low, high),
            limit_margin=round(margin, 3),
            settled=round(settled, 3),
            settle=round(float(trial["settle"][slot]), 3),
            support_nm=round(float(trial["support_nm"][slot]), 4),
            on_stop=bool(on_stop),
            margin_ok=bool(margin_ok),
        )

    robustness = _robustness(world, target, settle_window_s, perturbation_deg)
    checks = {
        "collision_free": not trial["commanded_contacts"],
        "settles": all(check.settle < SETTLE_LIMIT for check in joints.values()),
        "margins": all(check.margin_ok for check in joints.values()),
        "gripper_closed": target["gripper"] <= GRIPPER_CLOSED_MAX,
        "compact": trial["reach_m"] <= COMPACT_REACH_M and trial["height_m"] <= COMPACT_HEIGHT_M,
    }
    return RestPoseReport(
        pose={joint: float(target[joint]) for joint in JOINT_NAMES},
        joints=joints,
        commanded_contacts=trial["commanded_contacts"],
        settled_contacts=trial["settled_contacts"],
        tool_position_m=trial["tool_position_m"],
        reach_m=round(trial["reach_m"], 4),
        height_m=round(trial["height_m"], 4),
        robustness=robustness,
        checks=checks,
    )


# -- simulation helpers (need MuJoCo) -------------------------------------------
def _settle(world: Any, pose: Mapping[str, float], window_s: float) -> dict[str, Any]:
    import numpy as np

    world.reset(joints=pose)
    commanded = _arm_contacts(world)
    reach, height = _extent(world)
    tool = tuple(round(float(v), 4) for v in world.site_pose()[0])
    start = world.joint_array()
    world.set_torque(False)
    worst = np.zeros(len(JOINT_NAMES))
    for _ in range(max(1, world.steps_for(window_s))):
        world.step(1)
        worst = np.maximum(worst, np.abs(world.joint_array() - start))
    world.forward()
    support = _support_torque(world)
    return {
        "commanded_contacts": commanded,
        "settled_contacts": _arm_contacts(world),
        "settled": world.joint_array(),
        "settle": worst,
        "support_nm": support,
        "reach_m": reach,
        "height_m": height,
        "tool_position_m": tool,
    }


def _arm_contacts(world: Any) -> tuple[str, ...]:
    from .world import ROLE_ARM, ROLE_FIXED_JAW, ROLE_MOVING_JAW

    model, data = world.model, world.data
    arm_roles = (ROLE_ARM, ROLE_FIXED_JAW, ROLE_MOVING_JAW)
    role = world.index.geom_role
    pairs = set()
    ncon = int(data.ncon)
    for g1, g2, excluded in zip(
        data.contact.geom1[:ncon].tolist(),
        data.contact.geom2[:ncon].tolist(),
        data.contact.exclude[:ncon].tolist(),
        strict=True,
    ):
        if excluded or (role[g1] not in arm_roles and role[g2] not in arm_roles):
            continue
        names = sorted(
            model.body(int(model.geom_bodyid[g])).name or model.geom(g).name for g in (g1, g2)
        )
        pairs.add("|".join(names))
    return tuple(sorted(pairs))


def _support_torque(world: Any) -> Any:
    """Generalized force from contacts and joint limits on each arm joint."""

    import mujoco
    import numpy as np

    model, data = world.model, world.data
    nefc = int(data.nefc)
    result = np.zeros(model.nv)
    if nefc:
        kinds = {
            int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
            int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
        }
        keep = np.isin(data.efc_type[:nefc], list(kinds))
        force = np.where(keep, data.efc_force[:nefc], 0.0)
        mujoco.mj_mulJacTVec(model, data, result, force)
    return result[world.index.dof]


def _extent(world: Any) -> tuple[float, float]:
    """Reach from the base axis and height above the table of every arm geom box."""

    import numpy as np

    from .world import ROLE_ARM, ROLE_FIXED_JAW, ROLE_MOVING_JAW

    model, data = world.model, world.data
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    reach = height = 0.0
    for geom in range(model.ngeom):
        if world.index.geom_role[geom] not in (ROLE_ARM, ROLE_FIXED_JAW, ROLE_MOVING_JAW):
            continue
        centre, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        corners = (centre + signs * half) @ data.geom_xmat[geom].reshape(3, 3).T
        corners += data.geom_xpos[geom]
        reach = max(reach, float(np.max(np.hypot(corners[:, 0], corners[:, 1]))))
        height = max(height, float(np.max(corners[:, 2])))
    return reach, height


def _robustness(
    world: Any, pose: Mapping[str, float], window_s: float, delta: float
) -> dict[str, Any]:
    """Re-run the checks for every -delta/0/+delta combination on lift, elbow, wrist."""

    if delta <= 0.0:
        return {"perturbation_deg": 0.0, "cases": 0}
    worst = dict.fromkeys(JOINT_NAMES, 0.0)
    collisions = failures = cases = 0
    for offsets in itertools.product((-delta, 0.0, delta), repeat=len(PERTURBED_JOINTS)):
        trial_pose = dict(pose)
        for joint, offset in zip(PERTURBED_JOINTS, offsets, strict=True):
            trial_pose[joint] = float(pose[joint]) + offset
        trial = _settle(world, trial_pose, window_s)
        cases += 1
        collisions += bool(trial["commanded_contacts"])
        failures += bool(max(trial["settle"]) >= SETTLE_LIMIT)
        for joint, settle in zip(JOINT_NAMES, trial["settle"], strict=True):
            worst[joint] = max(worst[joint], round(float(settle), 3))
    return {
        "perturbation_deg": delta,
        "joints": list(PERTURBED_JOINTS),
        "cases": cases,
        "commanded_collisions": collisions,
        "settle_failures": failures,
        "worst_settle": worst,
    }


def format_report(report: RestPoseReport) -> str:
    """A short human-readable table (used by the derivation script and docs)."""

    lines = [f"rest pose {report.pose}  ok={report.ok}  checks={report.checks}"]
    for joint, check in report.joints.items():
        lines.append(
            f"  {joint:13s} target {check.target:8.2f}  margin {check.limit_margin:6.2f}  "
            f"settled {check.settled:8.2f}  settle {check.settle:5.2f}  "
            f"support {check.support_nm:+.3f} N m  stop={check.on_stop}  ok={check.margin_ok}"
        )
    lines.append(f"  commanded contacts: {list(report.commanded_contacts) or 'none'}")
    lines.append(f"  settled contacts:   {list(report.settled_contacts) or 'none'}")
    lines.append(
        f"  tool {report.tool_position_m} m  reach {report.reach_m} m  height {report.height_m} m"
    )
    lines.append(f"  robustness {report.robustness}")
    return "\n".join(lines)


__all__ = [
    "MODEL_REST_POSE",
    "JointRestCheck",
    "RestPoseReport",
    "check_rest_pose",
    "format_report",
]

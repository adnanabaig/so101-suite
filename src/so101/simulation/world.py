"""MuJoCo physics world: the vendored SO-101, a table and task objects.

Needs the ``[sim]`` extra (``pip install -e '.[sim]'``). Import this module
only after checking that MuJoCo is available; ``so101.simulation`` itself
never imports it.

Design:

- **One compiled template per scene.** ``compiled_scene(SceneSpec)`` builds the
  arm plus table, floor, lights, the ``front`` camera, free-body objects and
  visual-only mocap markers through ``mujoco.MjSpec``. The result is cached and
  treated as immutable.
- **Each world copies the template.** Torque-off toggles
  ``model.opt.disableflags``, which must not leak into other worlds or into the
  renderer, so every :class:`PhysicsWorld` owns a ``copy.copy`` of the model.
- **Suite units at the edge.** Joint targets and readings are suite units
  (degrees; gripper 0..100) in ``JOINT_NAMES`` order. Conversion lives in
  :mod:`so101.simulation.units`.
- **Torque-off is disabled actuation.** The arm then sags under gravity
  against the joint damping and friction of the STS3215 model, as real
  unpowered servos do. Re-enabling first sets every target to the present
  position, so the arm does not jump.

A world is not thread-safe. Its owner (``SimRuntime``, the engine, a batch
runner) serializes access.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import mujoco
import numpy as np

from .. import JOINT_NAMES
from .mjcf import MODEL_XML, mat_to_quat
from .scenarios import object_extents
from .units import (
    GRIPPER_JOINT,
    UnitsError,
    assert_joint_order,
    effective_limit_arrays,
    joint_maps,
    mjcf_to_suite_array,
    suite_to_mjcf_array,
)

# 240 Hz: 4 substeps per 60 Hz engine tick and 8 per 30 Hz control step. The
# vendored MJCF uses 5 ms; ``docs/simulation.md`` records why 1/240 is used.
DEFAULT_TIMESTEP_S = 1.0 / 240.0
TABLE_SIZE_M = (0.8, 0.6, 0.04)  # matches the scenario surface and the 3D twin
TABLE_HEIGHT_M = 0.72
FRONT_CAMERA = "front"
WRIST_CAMERA = "wrist_cam"
CAMERAS = (FRONT_CAMERA, WRIST_CAMERA)
OFFSCREEN_SIZE = (1280, 960)  # largest render the offscreen buffer allows
FRONT_CAMERA_EYE_M = (0.62, 0.0, 0.42)
FRONT_CAMERA_TARGET_M = (0.16, 0.0, 0.06)
FRONT_CAMERA_FOVY_DEG = 50.0
DEFAULT_OBJECT_DENSITY = 500.0  # kg/m^3: dense foam or light wood
# The follower writes Max_Torque_Limit=500 (50 %) to its gripper servo
# (hardware/motors.py, the follower branch of configure), so the jaw stalls
# at half the sts3215 class's 2.94 N m. The arm joints keep the full range.
GRIPPER_FORCE_LIMIT_NM = 1.47
OBJECT_FRICTION = (1.0, 0.03, 0.003)  # the Menagerie scene_box.xml values
ARM_ROOT_BODY = "base"
FIXED_JAW_BODY = "gripper"
MOVING_JAW_BODY = "moving_jaw_so101_v1"
TOOL_SITE = "gripperframe"

Shape = Literal["box", "cylinder", "sphere", "mesh"]
RGBA = tuple[float, float, float, float]


class WorldError(RuntimeError):
    """The physics world was asked for something it does not contain."""


# -- scene description ----------------------------------------------------------
@dataclass(frozen=True)
class ObjectSpec:
    """A free rigid body. ``size_m`` follows the scenario manifest convention.

    Box: full extents ``[x, y, z]`` (one entry is a cube). Cylinder:
    ``[radius, height]``, upright along z. Sphere: ``[radius]``. ``mesh`` has no
    mesh data in the manifest and is simulated as its bounding box, as the twin
    draws it.
    """

    id: str
    shape: Shape
    size_m: tuple[float, ...]
    rgba: RGBA = (1.0, 0.72, 0.3, 1.0)
    mass_kg: float | None = None
    friction: tuple[float, float, float] = OBJECT_FRICTION


@dataclass(frozen=True)
class MarkerSpec:
    """A visual-only mocap body (reach targets, place zones). Never collides."""

    id: str
    shape: Shape
    size_m: tuple[float, ...]
    rgba: RGBA


@dataclass(frozen=True)
class SceneSpec:
    """Everything that changes the compiled model. Poses are runtime state."""

    objects: tuple[ObjectSpec, ...] = ()
    markers: tuple[MarkerSpec, ...] = ()
    table_size_m: tuple[float, float, float] = TABLE_SIZE_M
    table_height_m: float = TABLE_HEIGHT_M
    timestep_s: float = DEFAULT_TIMESTEP_S

    def __post_init__(self) -> None:
        ids = [obj.id for obj in self.objects] + [marker.id for marker in self.markers]
        if len(ids) != len(set(ids)):
            raise WorldError(f"object and marker ids must be unique, got {ids}")
        if not 1e-4 <= self.timestep_s <= 0.01:
            raise WorldError("timestep_s must be between 0.1 ms and 10 ms")


@dataclass(frozen=True)
class SceneIndex:
    """Name -> address lookups for one compiled template."""

    qpos: np.ndarray  # arm qpos addresses, JOINT_NAMES order
    dof: np.ndarray  # arm dof addresses
    actuators: np.ndarray  # actuator ids, JOINT_NAMES order
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray
    force_limit: np.ndarray  # |actuator force| limit per joint, N*m
    site: int
    object_qpos: dict[str, int]
    object_dof: dict[str, int]
    object_body: dict[str, int]
    marker_mocap: dict[str, int]
    geom_role: np.ndarray  # see ROLE_* below
    geom_object: np.ndarray  # object index per geom, -1 if not an object
    object_ids: tuple[str, ...]
    cameras: tuple[str, ...]


ROLE_OTHER, ROLE_ARM, ROLE_FIXED_JAW, ROLE_MOVING_JAW, ROLE_TABLE, ROLE_OBJECT = range(6)


@dataclass(frozen=True)
class CompiledScene:
    """An immutable compiled template. Never mutate ``model``; copy it."""

    spec: SceneSpec
    model: Any
    index: SceneIndex


@dataclass(frozen=True)
class WorldSnapshot:
    """Copied state for rendering on another thread."""

    scene: SceneSpec
    time_s: float
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray


# -- small geometry helpers -------------------------------------------------------
def rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """``w, x, y, z`` for the manifest pose convention ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""

    roll, pitch, yaw = (math.radians(value) / 2.0 for value in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_rpy_deg(quat: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_deg_to_quat`, in degrees."""

    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])


def hex_to_rgba(color: str | None, alpha: float = 1.0) -> RGBA:
    """``#rrggbb`` -> RGBA in 0..1. Anything unparseable becomes neutral grey."""

    text = (color or "").strip().lstrip("#")
    if len(text) != 6:
        return (0.6, 0.6, 0.6, alpha)
    try:
        red, green, blue = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.6, 0.6, 0.6, alpha)
    return (red, green, blue, alpha)


def look_at_quat(eye: Sequence[float], target: Sequence[float]) -> tuple[float, ...]:
    """Camera quaternion: MuJoCo cameras look down their local -z with +y up."""

    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return mat_to_quat(np.column_stack([right, up, -forward]))


def _primitive(shape: str, size_m: Sequence[float]) -> tuple[int, list[float]]:
    """MuJoCo geom type and ``size`` (half extents) for a manifest shape."""

    kind, values = object_extents(shape, list(size_m))
    if kind == "sphere":
        return int(mujoco.mjtGeom.mjGEOM_SPHERE), [values[0], 0.0, 0.0]
    if kind == "cylinder":
        return int(mujoco.mjtGeom.mjGEOM_CYLINDER), [values[0], values[1] / 2.0, 0.0]
    return int(mujoco.mjtGeom.mjGEOM_BOX), [value / 2.0 for value in values]


def resting_half_height(shape: str, size_m: Sequence[float]) -> float:
    """Centre height above the table for an upright object."""

    kind, values = object_extents(shape, list(size_m))
    if kind == "sphere":
        return values[0]
    if kind == "cylinder":
        return values[1] / 2.0
    return values[2] / 2.0


def object_volume_m3(shape: str, size_m: Sequence[float]) -> float:
    kind, values = object_extents(shape, list(size_m))
    if kind == "sphere":
        return 4.0 / 3.0 * math.pi * values[0] ** 3
    if kind == "cylinder":
        return math.pi * values[0] ** 2 * values[1]
    return values[0] * values[1] * values[2]


# -- compilation -------------------------------------------------------------------
def _decorate(spec: Any, scene: SceneSpec) -> None:
    spec.option.timestep = scene.timestep_s
    spec.actuator(GRIPPER_JOINT).forcerange = [-GRIPPER_FORCE_LIMIT_NM, GRIPPER_FORCE_LIMIT_NM]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = OFFSCREEN_SIZE
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 2048
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.42, 0.52, 0.62],
        rgb2=[0.08, 0.1, 0.13],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="floor",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    floor = spec.add_material(name="floor", texrepeat=[5, 5], texuniform=True, reflectance=0.1)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"

    world = spec.worldbody
    world.add_light(
        name="key",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0.3, -0.4, 1.6],
        dir=[-0.25, 0.3, -1.0],
        diffuse=[0.65, 0.65, 0.65],
        specular=[0.15, 0.15, 0.15],
        castshadow=True,
    )
    sx, sy, sz = scene.table_size_m
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[sx / 2.0, sy / 2.0, sz / 2.0],
        pos=[0.0, 0.0, -sz / 2.0],
        rgba=[0.55, 0.47, 0.38, 1.0],
        friction=[1.0, 0.005, 0.0001],
        condim=3,
        contype=1,
        conaffinity=1,
    )
    leg = 0.025
    leg_height = max(0.0, scene.table_height_m - sz)
    for index, (lx, ly) in enumerate(
        (sign_x * (sx / 2.0 - leg), sign_y * (sy / 2.0 - leg))
        for sign_x in (-1.0, 1.0)
        for sign_y in (-1.0, 1.0)
    ):
        world.add_geom(
            name=f"table_leg_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[leg / 2.0, leg / 2.0, leg_height / 2.0],
            pos=[lx, ly, -sz - leg_height / 2.0],
            rgba=[0.32, 0.28, 0.24, 1.0],
            contype=0,
            conaffinity=0,
        )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, -scene.table_height_m],
        material="floor",
        contype=1,
        conaffinity=1,
    )
    world.add_camera(
        name=FRONT_CAMERA,
        pos=list(FRONT_CAMERA_EYE_M),
        quat=list(look_at_quat(FRONT_CAMERA_EYE_M, FRONT_CAMERA_TARGET_M)),
        fovy=FRONT_CAMERA_FOVY_DEG,
    )

    for slot, obj in enumerate(scene.objects):
        geom_type, size = _primitive(obj.shape, obj.size_m)
        half_height = resting_half_height(obj.shape, obj.size_m)
        # Parked in a row on the table; tasks and engines set real poses.
        body = world.add_body(
            name=f"obj_{obj.id}", pos=[0.3, -0.2 + 0.08 * slot, half_height + 0.001]
        )
        body.add_freejoint(name=f"obj_{obj.id}")
        mass = obj.mass_kg
        if mass is None:
            mass = DEFAULT_OBJECT_DENSITY * object_volume_m3(obj.shape, obj.size_m)
        body.add_geom(
            name=f"obj_{obj.id}",
            type=geom_type,
            size=size,
            rgba=list(obj.rgba),
            mass=mass,
            friction=list(obj.friction),
            condim=3,
            contype=1,
            conaffinity=1,
            solref=[0.01, 1.0],
        )
    for marker in scene.markers:
        geom_type, size = _primitive(marker.shape, marker.size_m)
        body = world.add_body(name=f"marker_{marker.id}", mocap=True, pos=[0.0, 0.0, -0.5])
        body.add_geom(
            name=f"marker_{marker.id}",
            type=geom_type,
            size=size,
            rgba=list(marker.rgba),
            contype=0,
            conaffinity=0,
            mass=0.0,
        )


def _subtree(model: Any, root: int) -> set[int]:
    members = {root}
    for body in range(model.nbody):
        parent = body
        while parent > 0:
            if parent in members:
                members.add(body)
                break
            parent = int(model.body_parentid[parent])
    return members


def _index(model: Any, scene: SceneSpec) -> SceneIndex:
    arm_joints = [model.joint(i).name for i in range(min(model.njnt, len(JOINT_NAMES)))]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    actuator_joints = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
    assert_joint_order(arm_joints, actuator_names, actuator_joints)
    joint_ids = [model.joint(name).id for name in JOINT_NAMES]
    actuator_ids = np.array([model.actuator(name).id for name in JOINT_NAMES])
    force = np.array(
        [
            max(abs(model.actuator_forcerange[i, 0]), abs(model.actuator_forcerange[i, 1]))
            if model.actuator_forcelimited[i]
            else math.inf
            for i in actuator_ids
        ]
    )
    arm_bodies = _subtree(model, model.body(ARM_ROOT_BODY).id)
    fixed_jaw = model.body(FIXED_JAW_BODY).id
    moving_jaw = model.body(MOVING_JAW_BODY).id
    object_ids = tuple(obj.id for obj in scene.objects)
    object_body = {obj_id: model.body(f"obj_{obj_id}").id for obj_id in object_ids}
    body_object = {body: slot for slot, body in enumerate(object_body.values())}
    role = np.full(model.ngeom, ROLE_OTHER, dtype=np.int8)
    geom_object = np.full(model.ngeom, -1, dtype=np.int16)
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body == fixed_jaw:
            role[geom] = ROLE_FIXED_JAW
        elif body == moving_jaw:
            role[geom] = ROLE_MOVING_JAW
        elif body in arm_bodies:
            role[geom] = ROLE_ARM
        elif body in body_object:
            role[geom] = ROLE_OBJECT
            geom_object[geom] = body_object[body]
    for name in ("table", "floor"):
        role[model.geom(name).id] = ROLE_TABLE
    object_joint = {obj_id: model.joint(f"obj_{obj_id}").id for obj_id in object_ids}
    return SceneIndex(
        qpos=np.array([model.jnt_qposadr[j] for j in joint_ids]),
        dof=np.array([model.jnt_dofadr[j] for j in joint_ids]),
        actuators=actuator_ids,
        ctrl_low=np.array([model.actuator_ctrlrange[i, 0] for i in actuator_ids]),
        ctrl_high=np.array([model.actuator_ctrlrange[i, 1] for i in actuator_ids]),
        force_limit=force,
        site=model.site(TOOL_SITE).id,
        object_qpos={key: int(model.jnt_qposadr[j]) for key, j in object_joint.items()},
        object_dof={key: int(model.jnt_dofadr[j]) for key, j in object_joint.items()},
        object_body=object_body,
        marker_mocap={
            marker.id: int(model.body_mocapid[model.body(f"marker_{marker.id}").id])
            for marker in scene.markers
        },
        geom_role=role,
        geom_object=geom_object,
        object_ids=object_ids,
        cameras=tuple(model.camera(i).name for i in range(model.ncam)),
    )


@lru_cache(maxsize=8)
def compiled_scene(scene: SceneSpec) -> CompiledScene:
    """Compile (and cache) the template for ``scene``. About 20-35 ms on a laptop.

    A compiled model is about 15 MB, and every randomized seed that rescales
    an object is a new ``SceneSpec``, so the cache stays small: a miss only
    costs a recompile.
    """

    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    _decorate(spec, scene)
    model = spec.compile()
    return CompiledScene(scene, model, _index(model, scene))


def clip_to_limits(values: np.ndarray) -> np.ndarray:
    low, high = effective_limit_arrays()
    return np.clip(values, low, high)


# -- the world ------------------------------------------------------------------------
class PhysicsWorld:
    """One simulated SO-101 workcell. Joint I/O is in suite units."""

    def __init__(self, scene: SceneSpec | None = None) -> None:
        self.template = compiled_scene(scene or SceneSpec())
        self.scene = self.template.spec
        self.index = self.template.index
        self.model = copy.copy(self.template.model)
        self.data = mujoco.MjData(self.model)
        self.seed: int | None = None
        self.rng = np.random.default_rng()
        self._velocity_scale = np.array([joint_maps()[joint].scale for joint in JOINT_NAMES])
        self.reset()

    # -- lifecycle ------------------------------------------------------------
    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    def reset(
        self,
        seed: int | None = None,
        *,
        joints: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Deterministic reset. Arm at ``joints`` (default all zero, gripper 50).

        Actuation is enabled and holding the arm pose. Objects return to their
        parked poses and markers are hidden below the table; tasks place them.
        ``self.rng`` is reseeded from ``seed`` for whoever randomizes the scene.
        """

        mujoco.mj_resetData(self.model, self.data)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if joints is None:
            pose = np.zeros(len(JOINT_NAMES))
            pose[JOINT_NAMES.index(GRIPPER_JOINT)] = 50.0
        else:
            pose = self.joint_vector(joints, np.zeros(len(JOINT_NAMES)))
        pose = clip_to_limits(pose)
        self.data.qpos[self.index.qpos] = suite_to_mjcf_array(pose)
        self.data.qvel[self.index.dof] = 0.0
        self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self._write_ctrl(pose)
        mujoco.mj_forward(self.model, self.data)

    def fork(self) -> PhysicsWorld:
        """An independent copy (model flags and full state) for what-if rollouts."""

        twin = PhysicsWorld.__new__(PhysicsWorld)
        twin.template = self.template
        twin.scene = self.scene
        twin.index = self.index
        twin.model = copy.copy(self.model)
        twin.data = copy.copy(self.data)
        twin.seed = self.seed
        twin.rng = copy.deepcopy(self.rng)
        twin._velocity_scale = self._velocity_scale
        return twin

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            scene=self.scene,
            time_s=self.time_s,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
        )

    # -- stepping --------------------------------------------------------------
    def step(self, n: int = 1) -> None:
        if n > 0:
            mujoco.mj_step(self.model, self.data, nstep=int(n))

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def steps_for(self, duration_s: float) -> int:
        return max(0, round(duration_s / self.timestep_s))

    def settle(self, duration_s: float, *, rewind_clock: bool = True) -> None:
        """Let objects come to rest (arm holding). Optionally restart the clock at 0."""

        self.step(self.steps_for(duration_s))
        if rewind_clock:
            self.data.time = 0.0

    # -- actuation ----------------------------------------------------------------
    def joint_vector(
        self, values: Mapping[str, float] | Sequence[float] | np.ndarray, base: np.ndarray
    ) -> np.ndarray:
        """Suite-unit vector from a (partial) mapping over ``base``, or a full 6-vector."""

        if isinstance(values, Mapping):
            unknown = sorted(set(values) - set(JOINT_NAMES))
            if unknown:
                raise UnitsError(f"unknown SO-101 joints: {unknown}")
            vector = np.array(base, dtype=float)
            for joint, value in values.items():
                vector[JOINT_NAMES.index(joint)] = float(value)
        else:
            vector = np.asarray(values, dtype=float).reshape(-1)
            if vector.shape != (len(JOINT_NAMES),):
                raise UnitsError(f"expected {len(JOINT_NAMES)} joint values, got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise UnitsError("joint values must be finite")
        return vector

    def _write_ctrl(self, suite: np.ndarray) -> np.ndarray:
        radians = np.clip(suite_to_mjcf_array(suite), self.index.ctrl_low, self.index.ctrl_high)
        self.data.ctrl[self.index.actuators] = radians
        return mjcf_to_suite_array(radians)

    def set_targets(
        self, targets: Mapping[str, float] | Sequence[float] | np.ndarray
    ) -> dict[str, float]:
        """Position targets in suite units. A mapping may be partial.

        Values are clipped to the effective limits (SafetyGuard defaults
        intersected with the MJCF ranges). Returns the applied targets.
        """

        vector = clip_to_limits(self.joint_vector(targets, self.target_array()))
        applied = self._write_ctrl(vector)
        return dict(zip(JOINT_NAMES, (float(value) for value in applied), strict=True))

    def target_array(self) -> np.ndarray:
        return mjcf_to_suite_array(self.data.ctrl[self.index.actuators])

    def targets(self) -> dict[str, float]:
        return dict(zip(JOINT_NAMES, (float(v) for v in self.target_array()), strict=True))

    def hold(self) -> dict[str, float]:
        """Hold in place: every target := the present position (clipped)."""

        return self.set_targets(self.joint_array())

    @property
    def torque_enabled(self) -> bool:
        return not bool(self.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION))

    def set_torque(self, enabled: bool) -> None:
        """Explicit torque switch. Off disables actuation; on holds first, so no jump."""

        flag = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        if enabled:
            if not self.torque_enabled:
                self.hold()
            self.model.opt.disableflags &= ~flag
        else:
            self.model.opt.disableflags |= flag
        mujoco.mj_forward(self.model, self.data)

    # -- arm readings ----------------------------------------------------------------
    def joint_array(self) -> np.ndarray:
        return mjcf_to_suite_array(self.data.qpos[self.index.qpos])

    def joints(self) -> dict[str, float]:
        return dict(zip(JOINT_NAMES, (float(v) for v in self.joint_array()), strict=True))

    def velocity_array(self) -> np.ndarray:
        """Joint velocities in suite units per second (deg/s; gripper units/s)."""

        return self.data.qvel[self.index.dof] / self._velocity_scale

    def velocities(self) -> dict[str, float]:
        return dict(zip(JOINT_NAMES, (float(v) for v in self.velocity_array()), strict=True))

    def effort_nm(self) -> np.ndarray:
        return np.array(self.data.actuator_force[self.index.actuators], dtype=float)

    def effort_pct(self) -> np.ndarray:
        """Signed actuator force as a percentage of the force limit (servo "load")."""

        return 100.0 * self.effort_nm() / self.index.force_limit

    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        site = self.index.site
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3, 3).copy()

    # -- objects and markers -----------------------------------------------------------
    def _object(self, object_id: str) -> tuple[int, int]:
        try:
            return self.index.object_qpos[object_id], self.index.object_dof[object_id]
        except KeyError as exc:
            raise WorldError(f"no object {object_id!r} in this scene") from exc

    def object_pose(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        address, _ = self._object(object_id)
        return self.data.qpos[address : address + 3].copy(), self.data.qpos[
            address + 3 : address + 7
        ].copy()

    def object_velocity(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Linear (world, m/s) and angular (body frame, rad/s) velocity."""

        _, dof = self._object(object_id)
        return self.data.qvel[dof : dof + 3].copy(), self.data.qvel[dof + 3 : dof + 6].copy()

    def set_object_pose(
        self,
        object_id: str,
        position_m: Sequence[float] | np.ndarray,
        quat_wxyz: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        address, dof = self._object(object_id)
        quat = np.asarray(quat_wxyz, dtype=float)
        self.data.qpos[address : address + 3] = np.asarray(position_m, dtype=float)
        self.data.qpos[address + 3 : address + 7] = quat / np.linalg.norm(quat)
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_marker(
        self,
        marker_id: str,
        position_m: Sequence[float] | np.ndarray,
        quat_wxyz: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        try:
            mocap = self.index.marker_mocap[marker_id]
        except KeyError as exc:
            raise WorldError(f"no marker {marker_id!r} in this scene") from exc
        self.data.mocap_pos[mocap] = np.asarray(position_m, dtype=float)
        self.data.mocap_quat[mocap] = np.asarray(quat_wxyz, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def marker_position(self, marker_id: str) -> np.ndarray:
        try:
            return self.data.mocap_pos[self.index.marker_mocap[marker_id]].copy()
        except KeyError as exc:
            raise WorldError(f"no marker {marker_id!r} in this scene") from exc

    # -- contacts --------------------------------------------------------------------------
    def contacts(self) -> dict[str, Any]:
        """Active contacts, grouped by what touches what."""

        ncon = int(self.data.ncon)
        objects: dict[str, dict[str, Any]] = {
            obj_id: {
                "table": 0,
                "fixed_jaw": 0,
                "moving_jaw": 0,
                "arm": 0,
                "objects": 0,
                "grasped": False,
            }
            for obj_id in self.index.object_ids
        }
        summary: dict[str, Any] = {
            "total": 0,
            "arm_self": 0,
            "arm_table": 0,
            "max_penetration_m": 0.0,
            "objects": objects,
        }
        if ncon == 0:
            return summary
        contact = self.data.contact
        active = np.asarray(contact.exclude[:ncon]) == 0
        geom1 = np.asarray(contact.geom1[:ncon])[active]
        geom2 = np.asarray(contact.geom2[:ncon])[active]
        dist = np.asarray(contact.dist[:ncon])[active]
        summary["total"] = int(active.sum())
        if dist.size:
            summary["max_penetration_m"] = float(max(0.0, -float(dist.min())))
        role = self.index.geom_role
        owner = self.index.geom_object
        arm_roles = (ROLE_ARM, ROLE_FIXED_JAW, ROLE_MOVING_JAW)
        names = {ROLE_TABLE: "table", ROLE_FIXED_JAW: "fixed_jaw", ROLE_MOVING_JAW: "moving_jaw"}
        for g1, g2 in zip(geom1.tolist(), geom2.tolist(), strict=True):
            r1, r2 = int(role[g1]), int(role[g2])
            if r1 in arm_roles and r2 in arm_roles:
                summary["arm_self"] += 1
            elif (r1 in arm_roles and r2 == ROLE_TABLE) or (r2 in arm_roles and r1 == ROLE_TABLE):
                summary["arm_table"] += 1
            for this, this_role, other_role in ((g1, r1, r2), (g2, r2, r1)):
                if this_role != ROLE_OBJECT:
                    continue
                entry = objects[self.index.object_ids[int(owner[this])]]
                if other_role == ROLE_OBJECT:
                    entry["objects"] += 1
                elif other_role == ROLE_ARM:
                    entry["arm"] += 1
                elif other_role in names:
                    entry[names[other_role]] += 1
        for entry in objects.values():
            entry["grasped"] = bool(entry["fixed_jaw"] and entry["moving_jaw"])
        return summary

    # -- observation ------------------------------------------------------------------------
    def observe(self) -> dict[str, Any]:
        """Everything a test, task or engine reads, in suite units and metres."""

        site_pos, site_rot = self.site_pose()
        objects = {}
        for obj_id in self.index.object_ids:
            position, quat = self.object_pose(obj_id)
            linear, angular = self.object_velocity(obj_id)
            objects[obj_id] = {
                "position_m": position.tolist(),
                "quat_wxyz": quat.tolist(),
                "linear_velocity_m_s": linear.tolist(),
                "angular_velocity_rad_s": angular.tolist(),
            }
        return {
            "time_s": self.time_s,
            "torque_enabled": self.torque_enabled,
            "joints": self.joints(),
            "velocities": self.velocities(),
            "targets": self.targets(),
            "effort_pct": dict(zip(JOINT_NAMES, self.effort_pct().tolist(), strict=True)),
            "effort_nm": dict(zip(JOINT_NAMES, self.effort_nm().tolist(), strict=True)),
            "gripper_site": {
                "position_m": site_pos.tolist(),
                "quat_wxyz": list(mat_to_quat(site_rot)),
            },
            "objects": objects,
            "markers": {
                marker_id: {"position_m": self.marker_position(marker_id).tolist()}
                for marker_id in self.index.marker_mocap
            },
            "contacts": self.contacts(),
        }

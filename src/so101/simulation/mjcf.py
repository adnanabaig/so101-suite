"""Pure-Python reader for the vendored SO-101 MJCF.

This module gives every engine one kinematic truth. The kinematic preview
engine, scenario telemetry and the browser twin all read the body tree that
MuJoCo compiles, so their forward kinematics agree with ``mj_kinematics`` to
floating-point precision without importing MuJoCo. The agreement is pinned by
``tests/fixtures/so101_fk_golden.json``.

Only the MJCF subset the vendored model uses is supported:

- nested default classes and ``childclass``;
- bodies with at most one hinge joint;
- geoms, sites, cameras, materials and file meshes;
- joint-transmission actuators.

Anything outside that subset raises :class:`MjcfError` instead of being
silently mis-modelled.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).with_name("assets") / "robotstudio_so101"
MODEL_XML = MODEL_DIR / "so101.xml"
MODEL_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
MODEL_REPO_PATH = "robotstudio_so101"
MODEL_COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
MODEL_LICENSE = "Apache-2.0"

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

_IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
_ORIENTATION_KEYS = ("quat", "euler", "axisangle", "xyaxes", "zaxis")
_ACTUATOR_TAGS = {
    "general",
    "motor",
    "position",
    "velocity",
    "intvelocity",
    "damper",
    "cylinder",
    "muscle",
}
# MuJoCo's internal geom rgba default. A geom rgba that differs from it takes
# precedence over the geom's material.
_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)
_UNSUPPORTED_BODY_CHILDREN = {"frame", "replicate", "include", "attach", "freejoint", "composite"}


class MjcfError(ValueError):
    """The MJCF uses a feature this reader does not model."""


@dataclass(frozen=True)
class JointSpec:
    name: str
    axis: Vec3
    pos: Vec3
    range: tuple[float, float] | None
    ref: float


@dataclass(frozen=True)
class GeomSpec:
    name: str | None
    kind: str
    pos: Vec3
    quat: Quat
    size: tuple[float, ...]
    mesh: str | None
    material: str | None
    rgba: tuple[float, float, float, float]
    contype: int
    conaffinity: int
    group: int

    @property
    def visual(self) -> bool:
        """True for geoms that never collide (MuJoCo's visual-only convention)."""

        return self.contype == 0 and self.conaffinity == 0


@dataclass(frozen=True)
class SiteSpec:
    name: str
    body: str
    pos: Vec3
    quat: Quat


@dataclass(frozen=True)
class CameraSpec:
    name: str
    body: str
    pos: Vec3
    quat: Quat
    fovy_deg: float


@dataclass(frozen=True)
class BodySpec:
    name: str
    parent: str | None
    pos: Vec3
    quat: Quat
    joint: JointSpec | None
    geoms: tuple[GeomSpec, ...]


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    joint: str
    ctrlrange: tuple[float, float] | None


@dataclass(frozen=True)
class Pose:
    """A world-frame pose: position in metres and a 3x3 rotation matrix."""

    pos: np.ndarray
    rot: np.ndarray

    def apply(self, pos: Vec3, quat: Quat = _IDENTITY_QUAT) -> Pose:
        """Return this pose composed with a local offset."""

        return Pose(self.pos + self.rot @ np.asarray(pos), self.rot @ quat_to_mat(quat))


@dataclass(frozen=True)
class MjcfModel:
    name: str
    path: Path
    meshdir: Path
    bodies: tuple[BodySpec, ...]
    sites: tuple[SiteSpec, ...]
    cameras: tuple[CameraSpec, ...]
    actuators: tuple[ActuatorSpec, ...]
    meshes: dict[str, str]
    materials: dict[str, tuple[float, float, float, float]]

    # -- lookups -----------------------------------------------------------
    def body(self, name: str) -> BodySpec:
        for body in self.bodies:
            if body.name == name:
                return body
        raise KeyError(f"unknown MJCF body {name!r}")

    def site(self, name: str) -> SiteSpec:
        for site in self.sites:
            if site.name == name:
                return site
        raise KeyError(f"unknown MJCF site {name!r}")

    def geom(self, name: str) -> tuple[BodySpec, GeomSpec]:
        for body in self.bodies:
            for geom in body.geoms:
                if geom.name == name:
                    return body, geom
        raise KeyError(f"unknown MJCF geom {name!r}")

    @property
    def joints(self) -> tuple[JointSpec, ...]:
        """Hinge joints in MuJoCo's qpos order (depth-first body order)."""

        return tuple(body.joint for body in self.bodies if body.joint is not None)

    def joint(self, name: str) -> JointSpec:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(f"unknown MJCF joint {name!r}")

    def actuator(self, name: str) -> ActuatorSpec:
        for actuator in self.actuators:
            if actuator.name == name:
                return actuator
        raise KeyError(f"unknown MJCF actuator {name!r}")

    def visual_mesh_files(self) -> dict[str, str]:
        """Mesh name -> file name for meshes drawn by at least one visual geom."""

        used = {
            geom.mesh
            for body in self.bodies
            for geom in body.geoms
            if geom.visual and geom.kind == "mesh" and geom.mesh
        }
        return {name: self.meshes[name] for name in sorted(used)}

    # -- forward kinematics -------------------------------------------------
    def body_poses(self, qpos: Mapping[str, float]) -> dict[str, Pose]:
        """World pose of every body for hinge angles in radians.

        This mirrors ``mj_kinematics``: a body frame is its parent's frame,
        then the fixed ``pos``/``quat`` offset, then a rotation of
        ``qpos - ref`` about the joint axis, anchored at the joint ``pos``.
        Every joint must be given.
        """

        poses: dict[str, Pose] = {}
        for body in self.bodies:
            parent = poses[body.parent] if body.parent else _WORLD
            pose = parent.apply(body.pos, body.quat)
            if body.joint is not None:
                joint = body.joint
                try:
                    angle = float(qpos[joint.name]) - joint.ref
                except KeyError as exc:
                    raise KeyError(f"missing MJCF joint angle {joint.name!r}") from exc
                local = _axis_angle_mat(joint.axis, angle)
                rot = pose.rot @ local
                anchor = pose.pos + pose.rot @ np.asarray(joint.pos)
                pose = Pose(anchor - rot @ np.asarray(joint.pos), rot)
            poses[body.name] = pose
        return poses

    def site_pose(self, name: str, qpos: Mapping[str, float]) -> Pose:
        site = self.site(name)
        return self.body_poses(qpos)[site.body].apply(site.pos, site.quat)


_WORLD = Pose(np.zeros(3), np.eye(3))


# -- math helpers -------------------------------------------------------------
def quat_to_mat(quat: Quat | np.ndarray) -> np.ndarray:
    """Rotation matrix for a unit quaternion in MuJoCo's ``w, x, y, z`` order."""

    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(rot: np.ndarray) -> Quat:
    """Unit quaternion ``w, x, y, z`` (``w >= 0``) for a rotation matrix."""

    m = np.asarray(rot, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    norm = math.sqrt(sum(value * value for value in quat))
    sign = -1.0 if quat[0] < 0 else 1.0
    w, x, y, z = (float(sign * value / norm) for value in quat)
    return (w, x, y, z)


def mat_to_rpy(rot: np.ndarray) -> Vec3:
    """Extrinsic X-Y-Z (URDF ``rpy``) angles in radians: ``R = Rz(y) Ry(p) Rx(r)``."""

    m = np.asarray(rot, dtype=float)
    pitch = math.atan2(-m[2, 0], math.hypot(m[0, 0], m[1, 0]))
    if math.hypot(m[0, 0], m[1, 0]) < 1e-12:  # gimbal lock: fold yaw into roll
        return (math.atan2(-m[1, 2], m[1, 1]), pitch, 0.0)
    return (math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0]))


def _axis_angle_mat(axis: Vec3, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize(values: tuple[float, ...], context: str) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise MjcfError(f"{context} has zero length")
    return tuple(value / norm for value in values)


# -- parsing ------------------------------------------------------------------
def _floats(text: str, count: int | None, context: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in text.split())
    except ValueError as exc:
        raise MjcfError(f"{context} must be numeric: {text!r}") from exc
    if count is not None and len(values) != count:
        raise MjcfError(f"{context} needs {count} numbers, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise MjcfError(f"{context} must be finite")
    return values


def _vec3(attrs: Mapping[str, str], key: str, default: str, context: str) -> Vec3:
    x, y, z = _floats(attrs.get(key, default), 3, f"{context} {key}")
    return (x, y, z)


@dataclass(frozen=True)
class _Compiler:
    angle_scale: float
    eulerseq: str
    meshdir: Path
    autolimits: bool


class _Reader:
    def __init__(self, root: ET.Element, path: Path) -> None:
        self.root = root
        self.path = path
        self.compiler = self._compiler()
        self.classes: dict[str, dict[str, dict[str, str]]] = {}
        for default in root.findall("default"):
            self._defaults(default, {}, top_level=True)
        if "main" not in self.classes:
            self.classes["main"] = {}

    def _compiler(self) -> _Compiler:
        attrs: dict[str, str] = {}
        for element in self.root.findall("compiler"):
            attrs.update(element.attrib)
        angle = attrs.get("angle", "degree")
        if angle not in {"degree", "radian"}:
            raise MjcfError(f"unsupported compiler angle {angle!r}")
        eulerseq = attrs.get("eulerseq", "xyz")
        if len(eulerseq) != 3 or any(axis not in "xyzXYZ" for axis in eulerseq):
            raise MjcfError(f"invalid compiler eulerseq {eulerseq!r}")
        meshdir = attrs.get("meshdir", attrs.get("assetdir", ""))
        return _Compiler(
            angle_scale=math.pi / 180.0 if angle == "degree" else 1.0,
            eulerseq=eulerseq,
            meshdir=(self.path.parent / meshdir).resolve(),
            autolimits=attrs.get("autolimits", "true") == "true",
        )

    def _defaults(
        self,
        element: ET.Element,
        inherited: dict[str, dict[str, str]],
        *,
        top_level: bool,
    ) -> None:
        name = element.get("class", "main" if top_level else "")
        if not name:
            raise MjcfError("nested <default> elements need a class name")
        resolved = {tag: dict(attrs) for tag, attrs in inherited.items()}
        for child in element:
            if child.tag == "default":
                continue
            tag = "actuator" if child.tag in _ACTUATOR_TAGS else child.tag
            resolved.setdefault(tag, {}).update(child.attrib)
        self.classes[name] = resolved
        for child in element.findall("default"):
            self._defaults(child, resolved, top_level=False)

    def attrs(self, element: ET.Element, tag: str, childclass: str | None) -> dict[str, str]:
        cls = element.get("class") or childclass or "main"
        if cls not in self.classes:
            raise MjcfError(f"unknown default class {cls!r}")
        merged = dict(self.classes[cls].get(tag, {}))
        merged.update(element.attrib)
        return merged

    def orientation(self, element: ET.Element, context: str) -> Quat:
        present = [key for key in _ORIENTATION_KEYS if key in element.attrib]
        if len(present) > 1:
            raise MjcfError(f"{context} specifies more than one orientation: {present}")
        if "fromto" in element.attrib:
            raise MjcfError(f"{context} uses fromto, which this reader does not model")
        if not present:
            return _IDENTITY_QUAT
        key = present[0]
        text = element.attrib[key]
        if key == "quat":
            w, x, y, z = _normalize(_floats(text, 4, f"{context} quat"), f"{context} quat")
            return (w, x, y, z)
        if key == "axisangle":
            ax, ay, az, angle = _floats(text, 4, f"{context} axisangle")
            ux, uy, uz = _normalize((ax, ay, az), f"{context} axisangle")
            half = 0.5 * angle * self.compiler.angle_scale
            s = math.sin(half)
            return (math.cos(half), ux * s, uy * s, uz * s)
        if key == "euler":
            quat = _IDENTITY_QUAT
            for axis, angle in zip(
                self.compiler.eulerseq, _floats(text, 3, f"{context} euler"), strict=True
            ):
                half = 0.5 * angle * self.compiler.angle_scale
                s = math.sin(half)
                step: Quat = (
                    math.cos(half),
                    s if axis in "xX" else 0.0,
                    s if axis in "yY" else 0.0,
                    s if axis in "zZ" else 0.0,
                )
                quat = _quat_mul(quat, step) if axis.islower() else _quat_mul(step, quat)
            w, x, y, z = _normalize(quat, f"{context} euler")
            return (w, x, y, z)
        if key == "xyaxes":
            values = _floats(text, 6, f"{context} xyaxes")
            x_axis = np.asarray(_normalize(values[:3], f"{context} xyaxes x"))
            y_raw = np.asarray(values[3:])
            y_axis = y_raw - x_axis * float(x_axis @ y_raw)
            y_axis = np.asarray(_normalize(tuple(y_axis), f"{context} xyaxes y"))
            z_axis = np.cross(x_axis, y_axis)
            return mat_to_quat(np.column_stack([x_axis, y_axis, z_axis]))
        # zaxis: the minimal rotation taking +z onto the given direction.
        target = np.asarray(_normalize(_floats(text, 3, f"{context} zaxis"), f"{context} zaxis"))
        up = np.array([0.0, 0.0, 1.0])
        cross = np.cross(up, target)
        sin_angle = float(np.linalg.norm(cross))
        cos_angle = float(up @ target)
        if sin_angle < 1e-12:
            return _IDENTITY_QUAT if cos_angle > 0 else (0.0, 1.0, 0.0, 0.0)
        rot_axis = cross / sin_angle
        half_angle = 0.5 * math.atan2(sin_angle, cos_angle)
        sin_half = math.sin(half_angle)
        return (
            math.cos(half_angle),
            float(rot_axis[0] * sin_half),
            float(rot_axis[1] * sin_half),
            float(rot_axis[2] * sin_half),
        )

    def model(self) -> MjcfModel:
        materials = self._materials()
        meshes = self._meshes()
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            raise MjcfError("MJCF has no <worldbody>")
        bodies: list[BodySpec] = []
        sites: list[SiteSpec] = []
        cameras: list[CameraSpec] = []
        for element in worldbody:
            if element.tag in _UNSUPPORTED_BODY_CHILDREN:
                raise MjcfError(f"<{element.tag}> is not supported by this reader")
        for element in worldbody.findall("body"):
            self._body(element, None, None, materials, meshes, bodies, sites, cameras)
        if worldbody.findall("geom") or worldbody.findall("site"):
            raise MjcfError("world-attached geoms/sites are not part of the arm model")
        actuators = self._actuators({body.joint.name for body in bodies if body.joint})
        return MjcfModel(
            name=self.root.get("model", "mjcf"),
            path=self.path,
            meshdir=self.compiler.meshdir,
            bodies=tuple(bodies),
            sites=tuple(sites),
            cameras=tuple(cameras),
            actuators=actuators,
            meshes=meshes,
            materials=materials,
        )

    def _materials(self) -> dict[str, tuple[float, float, float, float]]:
        materials: dict[str, tuple[float, float, float, float]] = {}
        for element in self._asset_children("material"):
            attrs = self.attrs(element, "material", None)
            name = attrs.get("name", "")
            if not name:
                raise MjcfError("materials need a name")
            r, g, b, a = _floats(attrs.get("rgba", "1 1 1 1"), 4, f"material {name} rgba")
            materials[name] = (r, g, b, a)
        return materials

    def _meshes(self) -> dict[str, str]:
        meshes: dict[str, str] = {}
        for element in self._asset_children("mesh"):
            attrs = self.attrs(element, "mesh", None)
            file = attrs.get("file", "")
            if not file:
                raise MjcfError("only file meshes are supported")
            if _floats(attrs.get("scale", "1 1 1"), 3, f"mesh {file} scale") != (1.0, 1.0, 1.0):
                raise MjcfError(f"mesh {file} is scaled, which this reader does not model")
            name = attrs.get("name") or Path(file).stem
            if name in meshes:
                raise MjcfError(f"duplicate mesh name {name!r}")
            meshes[name] = file
        return meshes

    def _asset_children(self, tag: str) -> Iterator[ET.Element]:
        for asset in self.root.findall("asset"):
            yield from asset.findall(tag)

    def _body(
        self,
        element: ET.Element,
        parent: str | None,
        childclass: str | None,
        materials: Mapping[str, tuple[float, float, float, float]],
        meshes: Mapping[str, str],
        bodies: list[BodySpec],
        sites: list[SiteSpec],
        cameras: list[CameraSpec],
    ) -> None:
        name = element.get("name", "")
        if not name:
            raise MjcfError("every body needs a name")
        context = f"body {name}"
        childclass = element.get("childclass") or childclass
        if childclass is not None and childclass not in self.classes:
            raise MjcfError(f"unknown childclass {childclass!r}")
        for child in element:
            if child.tag in _UNSUPPORTED_BODY_CHILDREN:
                raise MjcfError(f"<{child.tag}> in {context} is not supported by this reader")
        joints = element.findall("joint")
        if len(joints) > 1:
            raise MjcfError(f"{context} has {len(joints)} joints; at most one is supported")
        joint = self._joint(joints[0], childclass) if joints else None
        geoms = tuple(
            self._geom(geom, childclass, materials, meshes, f"{context} geom {index}")
            for index, geom in enumerate(element.findall("geom"))
        )
        bodies.append(
            BodySpec(
                name=name,
                parent=parent,
                pos=_vec3(element.attrib, "pos", "0 0 0", context),
                quat=self.orientation(element, context),
                joint=joint,
                geoms=geoms,
            )
        )
        for site in element.findall("site"):
            attrs = self.attrs(site, "site", childclass)
            site_name = attrs.get("name", "")
            if site_name:
                sites.append(
                    SiteSpec(
                        name=site_name,
                        body=name,
                        pos=_vec3(attrs, "pos", "0 0 0", f"site {site_name}"),
                        quat=self.orientation(site, f"site {site_name}"),
                    )
                )
        for camera in element.findall("camera"):
            attrs = self.attrs(camera, "camera", childclass)
            camera_name = attrs.get("name", "")
            if attrs.get("mode", "fixed") != "fixed":
                raise MjcfError(f"camera {camera_name} must use mode='fixed'")
            if camera_name:
                cameras.append(
                    CameraSpec(
                        name=camera_name,
                        body=name,
                        pos=_vec3(attrs, "pos", "0 0 0", f"camera {camera_name}"),
                        quat=self.orientation(camera, f"camera {camera_name}"),
                        fovy_deg=_floats(attrs.get("fovy", "45"), 1, "camera fovy")[0],
                    )
                )
        for child in element.findall("body"):
            self._body(child, name, childclass, materials, meshes, bodies, sites, cameras)

    def _joint(self, element: ET.Element, childclass: str | None) -> JointSpec:
        attrs = self.attrs(element, "joint", childclass)
        name = attrs.get("name", "")
        if not name:
            raise MjcfError("every joint needs a name")
        if attrs.get("type", "hinge") != "hinge":
            raise MjcfError(f"joint {name} is {attrs['type']!r}; only hinge joints are supported")
        ax, ay, az = _normalize(_floats(attrs.get("axis", "0 0 1"), 3, f"joint {name} axis"), name)
        limited = attrs.get("limited", "auto")
        has_range = "range" in attrs
        if limited == "auto":
            is_limited = has_range if self.compiler.autolimits else False
        else:
            is_limited = limited == "true"
        joint_range: tuple[float, float] | None = None
        if is_limited:
            low, high = _floats(attrs.get("range", "0 0"), 2, f"joint {name} range")
            scale = self.compiler.angle_scale
            joint_range = (low * scale, high * scale)
        return JointSpec(
            name=name,
            axis=(ax, ay, az),
            pos=_vec3(attrs, "pos", "0 0 0", f"joint {name}"),
            range=joint_range,
            ref=_floats(attrs.get("ref", "0"), 1, f"joint {name} ref")[0]
            * self.compiler.angle_scale,
        )

    def _geom(
        self,
        element: ET.Element,
        childclass: str | None,
        materials: Mapping[str, tuple[float, float, float, float]],
        meshes: Mapping[str, str],
        context: str,
    ) -> GeomSpec:
        attrs = self.attrs(element, "geom", childclass)
        name = attrs.get("name") or None
        kind = attrs.get("type", "sphere")
        mesh = attrs.get("mesh") if kind == "mesh" else None
        if kind == "mesh" and (mesh is None or mesh not in meshes):
            raise MjcfError(f"{context} references unknown mesh {mesh!r}")
        material = attrs.get("material") or None
        if material is not None and material not in materials:
            raise MjcfError(f"{context} references unknown material {material!r}")
        r, g, b, a = _floats(attrs.get("rgba", "0.5 0.5 0.5 1"), 4, f"{context} rgba")
        rgba = (r, g, b, a)
        if rgba == _DEFAULT_GEOM_RGBA and material is not None:
            rgba = materials[material]
        return GeomSpec(
            name=name,
            kind=kind,
            pos=_vec3(attrs, "pos", "0 0 0", context),
            quat=self.orientation(element, context),
            size=_floats(attrs.get("size", "0 0 0"), None, f"{context} size"),
            mesh=mesh,
            material=material,
            rgba=rgba,
            contype=int(attrs.get("contype", "1")),
            conaffinity=int(attrs.get("conaffinity", "1")),
            group=int(attrs.get("group", "0")),
        )

    def _actuators(self, joint_names: set[str]) -> tuple[ActuatorSpec, ...]:
        actuators: list[ActuatorSpec] = []
        for block in self.root.findall("actuator"):
            for element in block:
                if element.tag not in _ACTUATOR_TAGS:
                    raise MjcfError(f"unsupported actuator <{element.tag}>")
                attrs = self.attrs(element, "actuator", None)
                joint = attrs.get("joint", "")
                if joint not in joint_names:
                    raise MjcfError(f"actuator {attrs.get('name')!r} must drive a known joint")
                name = attrs.get("name") or joint
                ctrlrange: tuple[float, float] | None = None
                limited = attrs.get("ctrllimited", "auto")
                if "ctrlrange" in attrs and (
                    limited == "true" or (limited == "auto" and self.compiler.autolimits)
                ):
                    low, high = _floats(attrs["ctrlrange"], 2, f"actuator {name} ctrlrange")
                    ctrlrange = (low, high)
                actuators.append(ActuatorSpec(name=name, joint=joint, ctrlrange=ctrlrange))
        return tuple(actuators)


def parse_mjcf(path: Path | str) -> MjcfModel:
    """Parse an MJCF file into an immutable kinematic description."""

    resolved = Path(path).resolve()
    try:
        root = ET.parse(resolved).getroot()
    except (OSError, ET.ParseError) as exc:
        raise MjcfError(f"cannot read MJCF {resolved}: {exc}") from exc
    if root.tag != "mujoco":
        raise MjcfError(f"{resolved} is not an MJCF <mujoco> document")
    if root.find("include") is not None:
        raise MjcfError("<include> is not supported by this reader")
    return _Reader(root, resolved).model()


@lru_cache(maxsize=1)
def so101_model() -> MjcfModel:
    """The vendored Menagerie SO-101 description, parsed once per process."""

    return parse_mjcf(MODEL_XML)

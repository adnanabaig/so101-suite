"""Versioned scenario loading and deterministic scene randomization."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .. import JOINT_NAMES
from .kinematics import JOINT_LIMITS, default_joint_positions

_PROJECT_SCENARIO_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "simulation" / "scenarios.yaml"
)
_PACKAGED_SCENARIO_PATH = Path(__file__).with_name("scenarios.yaml")
# Editable/source checkouts use the operator-facing config. Wheels and the
# slim API container do not copy repository-root configs, so ship the same
# manifest as package data rather than making service import depend on CWD.
DEFAULT_SCENARIO_PATH = (
    _PROJECT_SCENARIO_PATH if _PROJECT_SCENARIO_PATH.is_file() else _PACKAGED_SCENARIO_PATH
)
SUPPORTED_SHAPES = {"box", "cylinder", "sphere", "mesh"}


class ScenarioError(ValueError):
    """A scenario manifest is malformed."""


class ScenarioNotFound(ScenarioError):
    """The requested scenario id is not in the loaded library."""


@dataclass(frozen=True)
class EngineConfig:
    fixed_hz: float
    default_seed: int
    initial_joint_positions: dict[str, float]
    max_velocity_deg_s: dict[str, float]


@dataclass(frozen=True)
class ScenarioDefinition:
    id: str
    label: str
    description: str
    surface: dict[str, Any]
    objects: tuple[dict[str, Any], ...]
    randomization: dict[str, Any]


def _finite_number(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"{context} must be numeric") from exc
    if not math.isfinite(number):
        raise ScenarioError(f"{context} must be finite")
    return number


def _positive_vector(value: Any, context: str, lengths: set[int]) -> list[float]:
    if not isinstance(value, list) or len(value) not in lengths:
        allowed = ", ".join(str(length) for length in sorted(lengths))
        raise ScenarioError(f"{context} must be a list with length {allowed}")
    result = [_finite_number(item, context) for item in value]
    if any(item <= 0 for item in result):
        raise ScenarioError(f"{context} entries must be positive")
    return result


def object_extents(shape: str, size_m: list[float]) -> tuple[str, list[float]]:
    """Normalise a manifest ``size_m`` into an explicit primitive description.

    This matches how the twin draws objects:

    - box and mesh: full extents ``[x, y, z]``. One entry is a cube; two
      entries reuse the first for z.
    - cylinder: ``[radius, height]``, upright along z.
    - sphere: ``[radius]``.

    Returns ``(kind, values)``, where ``values`` are full box extents, a
    cylinder ``[radius, height]``, or a sphere ``[radius]``.
    """

    if shape == "cylinder":
        radius = size_m[0]
        return "cylinder", [radius, size_m[1] if len(size_m) >= 2 else radius * 2.0]
    if shape == "sphere":
        return "sphere", [size_m[0]]
    x = size_m[0]
    y = size_m[1] if len(size_m) >= 2 else x
    z = size_m[2] if len(size_m) >= 3 else x
    return "box", [x, y, z]


def vertical_half_extent(
    shape: str, size_m: list[float], roll_deg: float = 0.0, pitch_deg: float = 0.0
) -> float:
    """Half of an object's height above its centre for a ``[roll, pitch]`` tilt.

    Yaw never changes height. Roll and pitch follow the pose convention
    ``R = Rz(yaw) Ry(pitch) Rx(roll)`` (extrinsic X-Y-Z, as in URDF ``rpy``).
    """

    roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
    # Third row of Ry(pitch) @ Rx(roll): world z expressed in object axes.
    up = (-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll))
    kind, values = object_extents(shape, size_m)
    if kind == "sphere":
        return values[0]
    if kind == "cylinder":
        radius, height = values
        tilt = abs(up[2])
        return 0.5 * height * tilt + radius * math.sqrt(max(0.0, 1.0 - tilt * tilt))
    return 0.5 * sum(abs(component) * extent for component, extent in zip(up, values, strict=True))


class ScenarioLibrary:
    """Immutable scenario definitions loaded from the versioned YAML asset."""

    def __init__(self, path: Path | str = DEFAULT_SCENARIO_PATH) -> None:
        self.path = Path(path)
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ScenarioError(f"cannot read simulation scenarios: {self.path}") from exc
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ScenarioError("simulation scenarios require manifest version 1")
        self.engine = self._parse_engine(raw.get("engine", {}))
        entries = raw.get("scenarios")
        if not isinstance(entries, list) or not entries:
            raise ScenarioError("simulation manifest must contain scenarios")
        parsed = [self._parse_scenario(entry) for entry in entries]
        ids = [scenario.id for scenario in parsed]
        if len(ids) != len(set(ids)):
            raise ScenarioError("simulation scenario ids must be unique")
        self._scenarios = {scenario.id: scenario for scenario in parsed}

    @staticmethod
    def _parse_engine(raw: Any) -> EngineConfig:
        if not isinstance(raw, dict):
            raise ScenarioError("engine config must be an object")
        fixed_hz = _finite_number(raw.get("fixed_hz", 60.0), "engine.fixed_hz")
        if not 1.0 <= fixed_hz <= 240.0:
            raise ScenarioError("engine.fixed_hz must be between 1 and 240")
        default_seed = int(raw.get("default_seed", 101))
        if not 0 <= default_seed <= 2**32 - 1:
            raise ScenarioError("engine.default_seed must be an unsigned 32-bit integer")

        initial_raw = raw.get("initial_joint_positions", default_joint_positions())
        velocity_raw = raw.get("max_velocity_deg_s", {})
        if not isinstance(initial_raw, dict) or not isinstance(velocity_raw, dict):
            raise ScenarioError("engine joint settings must be objects")
        initial: dict[str, float] = {}
        velocity: dict[str, float] = {}
        for joint in JOINT_NAMES:
            position = _finite_number(initial_raw.get(joint, 0.0), f"initial {joint}")
            low, high = JOINT_LIMITS[joint]
            if not low <= position <= high:
                raise ScenarioError(f"initial {joint} is outside SO-101 limits")
            speed = _finite_number(velocity_raw.get(joint, 90.0), f"velocity {joint}")
            if speed <= 0:
                raise ScenarioError(f"velocity {joint} must be positive")
            initial[joint] = position
            velocity[joint] = speed

        if "kinematics" in raw:
            # v0.1 shipped hand-measured link lengths here. Geometry now comes
            # only from the vendored MJCF, so a stale block must not look live.
            raise ScenarioError(
                "engine.kinematics is no longer supported: SO-101 geometry comes from the "
                "vendored MJCF (so101.simulation.mjcf)"
            )
        return EngineConfig(fixed_hz, default_seed, initial, velocity)

    @staticmethod
    def _parse_scenario(raw: Any) -> ScenarioDefinition:
        if not isinstance(raw, dict):
            raise ScenarioError("each scenario must be an object")
        scenario_id = str(raw.get("id", "")).strip()
        if not scenario_id or not scenario_id.replace("_", "").isalnum():
            raise ScenarioError(f"invalid scenario id {scenario_id!r}")
        label = str(raw.get("label", scenario_id)).strip()
        description = str(raw.get("description", "")).strip()
        surface = copy.deepcopy(raw.get("surface", {}))
        if not isinstance(surface, dict):
            raise ScenarioError(f"surface for {scenario_id} must be an object")
        surface["size_m"] = _positive_vector(
            surface.get("size_m"), f"{scenario_id}.surface.size_m", {3}
        )
        surface["height_m"] = _finite_number(
            surface.get("height_m"), f"{scenario_id}.surface.height_m"
        )
        if surface["height_m"] <= 0:
            raise ScenarioError(f"{scenario_id}.surface.height_m must be positive")

        objects_raw = raw.get("objects", [])
        if not isinstance(objects_raw, list):
            raise ScenarioError(f"objects for {scenario_id} must be a list")
        objects: list[dict[str, Any]] = []
        object_ids: set[str] = set()
        for item in objects_raw:
            if not isinstance(item, dict):
                raise ScenarioError(f"objects for {scenario_id} must be objects")
            obj = copy.deepcopy(item)
            object_id = str(obj.get("id", "")).strip()
            if not object_id or object_id in object_ids:
                raise ScenarioError(f"duplicate or empty object id in {scenario_id}")
            object_ids.add(object_id)
            shape = str(obj.get("shape", ""))
            if shape not in SUPPORTED_SHAPES:
                raise ScenarioError(f"unsupported shape {shape!r} for {object_id}")
            obj["size_m"] = _positive_vector(
                obj.get("size_m"), f"{scenario_id}.{object_id}.size_m", {1, 2, 3}
            )
            pose = obj.get("pose")
            if not isinstance(pose, list) or len(pose) != 6:
                raise ScenarioError(f"pose for {scenario_id}.{object_id} must have 6 entries")
            obj["pose"] = [
                _finite_number(value, f"{scenario_id}.{object_id}.pose") for value in pose
            ]
            if "randomize" in obj and not isinstance(obj["randomize"], bool):
                raise ScenarioError(f"randomize for {scenario_id}.{object_id} must be boolean")
            objects.append(obj)

        randomization = copy.deepcopy(raw.get("randomization", {}))
        if not isinstance(randomization, dict):
            raise ScenarioError(f"randomization for {scenario_id} must be an object")
        return ScenarioDefinition(
            scenario_id,
            label,
            description,
            surface,
            tuple(objects),
            randomization,
        )

    def get(self, scenario_id: str) -> ScenarioDefinition:
        try:
            return self._scenarios[scenario_id]
        except KeyError as exc:
            raise ScenarioNotFound(f"unknown simulation scenario {scenario_id!r}") from exc

    def catalog(
        self,
        *,
        backend: str = "kinematic",
        physics: bool = False,
        detail: str = "Deterministic fixed-step scene for the isolated kinematic backend.",
    ) -> list[dict[str, Any]]:
        """Scenario entries labelled with the backend that will actually run them."""

        return [
            {
                "id": scenario.id,
                "name": scenario.label,
                "label": scenario.label,
                "description": scenario.description,
                "detail": detail,
                "available": True,
                "backend": backend,
                "physics": physics,
                "surface": copy.deepcopy(scenario.surface),
                "objects": copy.deepcopy(list(scenario.objects)),
                "supports_domain_randomization": bool(scenario.randomization),
            }
            for scenario in self._scenarios.values()
        ]

    def instantiate(
        self,
        scenario_id: str,
        *,
        seed: int | None = None,
        randomize: bool = True,
    ) -> dict[str, Any]:
        """Create a reproducible scene without mutating its definition."""

        scenario = self.get(scenario_id)
        user_seed = self.engine.default_seed if seed is None else int(seed)
        if not 0 <= user_seed <= 2**32 - 1:
            raise ScenarioError("seed must be an unsigned 32-bit integer")
        digest = hashlib.sha256(f"{scenario.id}:{user_seed}".encode()).digest()
        derived_seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(derived_seed)
        settings = scenario.randomization
        position_jitter = settings.get("object_position_jitter_m", [0.0, 0.0, 0.0])
        if not isinstance(position_jitter, list) or len(position_jitter) != 3:
            raise ScenarioError("object_position_jitter_m must contain x, y, z")
        jitter = [abs(_finite_number(value, "object position jitter")) for value in position_jitter]
        yaw_jitter = abs(
            _finite_number(settings.get("object_yaw_jitter_deg", 0.0), "object yaw jitter")
        )
        scale_range = settings.get("object_scale_range", [1.0, 1.0])
        if not isinstance(scale_range, list) or len(scale_range) != 2:
            raise ScenarioError("object_scale_range must contain min and max")
        scale_low, scale_high = [
            _finite_number(value, "object scale range") for value in scale_range
        ]
        if scale_low <= 0 or scale_low > scale_high:
            raise ScenarioError("object_scale_range must be positive and ordered")

        objects: list[dict[str, Any]] = []
        samples: dict[str, dict[str, Any]] = {}
        for definition in scenario.objects:
            obj = copy.deepcopy(definition)
            nominal_pose = list(obj["pose"])
            object_randomized = randomize and bool(obj.get("randomize", True))
            if object_randomized:
                offsets = [rng.uniform(-amount, amount) for amount in jitter]
                yaw_offset = rng.uniform(-yaw_jitter, yaw_jitter)
                scale = rng.uniform(scale_low, scale_high)
            else:
                offsets = [0.0, 0.0, 0.0]
                yaw_offset = 0.0
                scale = 1.0
            obj["nominal_pose"] = nominal_pose
            # Scaling about the object centre would sink or float it. Re-seat
            # it so its lowest point stays where the manifest put it.
            reseat = (scale - 1.0) * vertical_half_extent(
                obj["shape"], obj["size_m"], nominal_pose[3], nominal_pose[4]
            )
            obj["pose"] = [
                round(nominal_pose[0] + offsets[0], 9),
                round(nominal_pose[1] + offsets[1], 9),
                round(nominal_pose[2] + offsets[2] + reseat, 9),
                nominal_pose[3],
                nominal_pose[4],
                round(nominal_pose[5] + yaw_offset, 9),
            ]
            obj["nominal_size_m"] = list(obj["size_m"])
            obj["size_m"] = [round(dimension * scale, 9) for dimension in obj["size_m"]]
            objects.append(obj)
            samples[obj["id"]] = {
                "position_offset_m": [round(value, 9) for value in offsets],
                "yaw_offset_deg": round(yaw_offset, 9),
                "scale": round(scale, 9),
                "applied": object_randomized,
            }

        return {
            "scenario_id": scenario.id,
            "label": scenario.label,
            "description": scenario.description,
            "seed": user_seed,
            "randomized": bool(randomize),
            "surface": copy.deepcopy(scenario.surface),
            "objects": objects,
            "randomization": {
                "applied": bool(randomize),
                "seed": user_seed,
                "settings": copy.deepcopy(settings),
                "samples": samples,
            },
        }

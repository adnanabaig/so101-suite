"""Rest pose for the Collapse stop (v0.2 design doc, section 5 "Rest pose").

Collapse lowers the arm to the rest pose at a capped speed and then turns the
motors off, so the rest pose must be one the arm can stay in without torque.

Resolution order (:func:`resolve_rest_pose`):

1. The pose recorded during bring-up (``so101 rest-pose record``). It is bound to
   the follower id and to ``calibration_digest``, the sha256 of the follower's
   LeRobot calibration file, so recalibrating invalidates it. A recorded pose
   that does not match is ignored with a warning.
2. ``SO101_REST_POSE``: six numbers in ``JOINT_NAMES`` order or a JSON object.
3. The model-derived default,
   :data:`so101.simulation.rest_pose.MODEL_REST_POSE`: folded,
   collision-free, at least 5 degrees inside the limits and gravity-stable in
   MuJoCo (plain data; no MuJoCo import). It is not verified on a real arm.

Joints use suite units: degrees for the five arm joints and 0..100 for the
gripper.

Collapse (``RobotManager``), preflight, the HIL runner and ``so101 rest-pose``
all resolve through :func:`resolve_collapse_rest_pose` (also reached as
:func:`resolve_for_settings`), so every view shows the pose Collapse would
use: the same order, clamped to the active SafetyGuard limits intersected with
the MJCF limits (design doc section 5 "Rest pose").
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .. import JOINT_NAMES, __version__
from ..atomic_io import atomic_write_json
from .backend import Backend, current_backend
from .lerobot_paths import calibration_digest
from .safety import DEFAULT_LIMITS

REST_POSE_SCHEMA = 1
REST_POSE_FILENAME = "rest_pose.json"

#: ``so101 rest-pose record`` sampling: about 10 reads, spread under 1 degree
#: (1 unit for the 0..100 gripper).
RECORD_SAMPLES = 10
RECORD_INTERVAL_S = 0.1
RECORD_MAX_SPREAD = 1.0

RestPoseSource = Literal["recorded", "setting", "default"]


class RestPoseError(ValueError):
    """A rest pose file or value is malformed."""


class RestPoseUnstableError(RuntimeError):
    """The arm moved too much while the rest pose was being sampled."""

    def __init__(self, message: str, spread: dict[str, float]) -> None:
        super().__init__(message)
        self.spread = spread


def _finite_joint_map(values: Mapping[str, Any], *, what: str) -> dict[str, float]:
    unknown = sorted(set(values) - set(JOINT_NAMES))
    missing = [joint for joint in JOINT_NAMES if joint not in values]
    if unknown or missing:
        raise RestPoseError(
            f"{what} needs exactly {JOINT_NAMES}; missing={missing} unknown={unknown}"
        )
    joints: dict[str, float] = {}
    for joint in JOINT_NAMES:
        value = values[joint]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RestPoseError(f"{what}: {joint} must be a number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise RestPoseError(f"{what}: {joint} must be finite")
        joints[joint] = number
    return joints


def parse_rest_pose_setting(text: str) -> dict[str, float] | None:
    """Parse ``SO101_REST_POSE``. Empty means unset.

    Accepts six numbers in ``JOINT_NAMES`` order (comma or whitespace separated,
    or a JSON array) or a JSON object keyed by joint name with all six joints.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    what = "SO101_REST_POSE"
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RestPoseError(f"{what} is not valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return _finite_joint_map(parsed, what=what)
        values = parsed
    else:
        values = [part for part in stripped.replace(",", " ").split() if part]
    if not isinstance(values, list) or len(values) != len(JOINT_NAMES):
        raise RestPoseError(f"{what} needs {len(JOINT_NAMES)} numbers in order {JOINT_NAMES}")
    numbers: list[float] = []
    for value in values:
        try:
            if isinstance(value, bool):
                raise ValueError
            numbers.append(float(value))
        except (TypeError, ValueError) as exc:
            raise RestPoseError(f"{what}: {value!r} is not a number") from exc
    return _finite_joint_map(dict(zip(JOINT_NAMES, numbers, strict=True)), what=what)


@dataclass(frozen=True)
class RestPoseRecord:
    """Contents of ``<data_dir>/rest_pose.json``."""

    joints: dict[str, float]
    follower_id: str
    calibration_digest: str | None
    recorded_at: str
    so101_version: str
    backend: Backend = "real"
    spread: dict[str, float] = field(default_factory=dict)
    samples: int | None = None
    schema: int = REST_POSE_SCHEMA

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "joints": dict(self.joints),
            "follower_id": self.follower_id,
            "calibration_digest": self.calibration_digest,
            "recorded_at": self.recorded_at,
            "so101_version": self.so101_version,
            # Additive fields: where and how well the pose was measured.
            "backend": self.backend,
            "spread": dict(self.spread),
            "samples": self.samples,
        }

    @classmethod
    def from_json(cls, payload: Any) -> RestPoseRecord:
        if not isinstance(payload, dict):
            raise RestPoseError("rest pose file must contain a JSON object")
        if payload.get("schema") != REST_POSE_SCHEMA:
            raise RestPoseError(f"unsupported rest pose schema {payload.get('schema')!r}")
        joints = payload.get("joints")
        if not isinstance(joints, dict):
            raise RestPoseError("rest pose file has no joints object")
        follower_id = payload.get("follower_id")
        if not isinstance(follower_id, str) or not follower_id:
            raise RestPoseError("rest pose file has no follower_id")
        digest = payload.get("calibration_digest")
        if digest is not None and not isinstance(digest, str):
            raise RestPoseError("calibration_digest must be a string or null")
        backend = payload.get("backend", "real")
        if backend not in ("real", "mock", "sim"):
            raise RestPoseError(f"unknown backend {backend!r}")
        spread = payload.get("spread") or {}
        return cls(
            joints=_finite_joint_map(joints, what="rest pose file"),
            follower_id=follower_id,
            calibration_digest=digest,
            recorded_at=str(payload.get("recorded_at", "")),
            so101_version=str(payload.get("so101_version", "")),
            backend=backend,
            spread={str(k): float(v) for k, v in spread.items()}
            if isinstance(spread, dict)
            else {},
            samples=payload.get("samples") if isinstance(payload.get("samples"), int) else None,
        )


def rest_pose_path(settings: Any) -> Path:
    return Path(settings.data_dir) / REST_POSE_FILENAME


def load_recorded_rest_pose(settings: Any) -> RestPoseRecord | None:
    """Load the recorded pose; None when absent, RestPoseError when corrupt."""
    path = rest_pose_path(settings)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RestPoseError(f"{path} is not valid JSON: {exc}") from exc
    return RestPoseRecord.from_json(payload)


def save_rest_pose(
    settings: Any,
    joints: Mapping[str, float],
    *,
    follower_id: str,
    calibration_digest: str | None,
    backend: Backend,
    spread: Mapping[str, float] | None = None,
    samples: int | None = None,
    now: datetime | None = None,
) -> RestPoseRecord:
    """Atomically write ``<data_dir>/rest_pose.json``."""
    record = RestPoseRecord(
        joints=_finite_joint_map(joints, what="rest pose"),
        follower_id=follower_id,
        calibration_digest=calibration_digest,
        recorded_at=(now or datetime.now(timezone.utc)).isoformat(),
        so101_version=__version__,
        backend=backend,
        spread={joint: round(float(value), 4) for joint, value in (spread or {}).items()},
        samples=samples,
    )
    atomic_write_json(rest_pose_path(settings), record.to_json())
    return record


def clear_rest_pose(settings: Any) -> bool:
    """Delete the recorded pose. Returns False when there was none."""
    try:
        rest_pose_path(settings).unlink()
    except FileNotFoundError:
        return False
    return True


def follower_calibration_digest(settings: Any, *, follower_id: str | None = None) -> str | None:
    """sha256 of the follower's LeRobot calibration file (None when missing)."""
    return calibration_digest("follower", follower_id or settings.follower_id)


def clamp_to_limits(
    joints: Mapping[str, float], limits: Mapping[str, tuple[float, float]]
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Clamp to ``limits``; returns the pose and ``{joint: (original, clamped)}``."""
    clamped: dict[str, float] = {}
    changes: dict[str, tuple[float, float]] = {}
    for joint in JOINT_NAMES:
        value = float(joints[joint])
        low, high = limits.get(joint, DEFAULT_LIMITS[joint])
        bounded = min(max(value, float(low)), float(high))
        clamped[joint] = bounded
        if bounded != value:
            changes[joint] = (value, bounded)
    return clamped, changes


class RestPoseResolution(BaseModel):
    joints: dict[str, float]
    source: RestPoseSource
    warnings: list[str]
    clamped: dict[str, tuple[float, float]] = {}
    #: Always False: step 3 is the model-derived pose. Kept so saved reports
    #: and ``--json`` consumers from wave 1 still parse.
    placeholder: bool = False
    follower_id: str
    calibration_digest: str | None = None
    backend: Backend
    recorded: dict[str, Any] | None = None


def resolve_rest_pose(
    settings: Any,
    *,
    follower_id: str,
    calibration_digest: str | None,
    limits: Mapping[str, tuple[float, float]] | None = None,
    backend: Backend | None = None,
    default_pose: Mapping[str, float] | None = None,
    limits_label: str = "SafetyGuard limits",
) -> RestPoseResolution:
    """Pick the rest pose: recorded, then ``SO101_REST_POSE``, then the default.

    ``limits`` is the clamp; the conservative ``DEFAULT_LIMITS`` are used
    without it. :func:`resolve_collapse_rest_pose` passes SafetyGuard ∩ MJCF.
    A recorded pose counts only when its follower id, backend and calibration
    digest all match; on the real backend it also needs a digest at all.
    ``default_pose`` is step 3; without it, the model-derived pose
    (:func:`model_rest_pose`). ``limits_label`` names the clamp in warnings.
    """
    active_backend = backend or current_backend(settings)
    warnings: list[str] = []
    record: RestPoseRecord | None = None
    try:
        record = load_recorded_rest_pose(settings)
    except RestPoseError as exc:
        warnings.append(f"ignored the recorded rest pose: {exc}")

    joints: dict[str, float] | None = None
    source: RestPoseSource = "default"
    if record is not None:
        problems: list[str] = []
        if record.follower_id != follower_id:
            problems.append(
                f"it was recorded for follower {record.follower_id!r}, not {follower_id!r}"
            )
        if record.backend != active_backend:
            problems.append(
                f"it was recorded on the {record.backend} backend, not {active_backend}"
            )
        if active_backend == "real" and calibration_digest is None:
            problems.append("the follower has no LeRobot calibration file to bind it to")
        elif record.calibration_digest != calibration_digest:
            problems.append("the follower calibration changed after it was recorded")
        if problems:
            warnings.append("ignored the recorded rest pose: " + "; ".join(problems))
        else:
            joints, source = dict(record.joints), "recorded"

    if joints is None:
        try:
            from_setting = parse_rest_pose_setting(getattr(settings, "rest_pose", "") or "")
        except RestPoseError as exc:  # settings validation normally catches this
            warnings.append(f"ignored SO101_REST_POSE: {exc}")
            from_setting = None
        if from_setting is not None:
            joints, source = from_setting, "setting"

    if joints is None:
        default = model_rest_pose() if default_pose is None else default_pose
        joints, source = _finite_joint_map(default, what="default rest pose"), "default"
        warnings.append(
            "no usable recorded pose and no SO101_REST_POSE, so Collapse uses the "
            "model-derived default (folded and gravity-stable in MuJoCo, not verified on "
            "this arm); record one with `so101 rest-pose record`"
        )

    bounded, changes = clamp_to_limits(joints, limits or DEFAULT_LIMITS)
    for joint, (original, value) in changes.items():
        warnings.append(f"clamped {joint} from {original:g} to {value:g} ({limits_label})")
    return RestPoseResolution(
        joints=bounded,
        source=source,
        warnings=warnings,
        clamped=changes,
        follower_id=follower_id,
        calibration_digest=calibration_digest,
        backend=active_backend,
        recorded=record.to_json() if record is not None else None,
    )


def resolve_for_settings(
    settings: Any, *, limits: Mapping[str, tuple[float, float]] | None = None
) -> RestPoseResolution:
    """The pose Collapse would use: :func:`resolve_collapse_rest_pose`.

    ``limits`` is the active SafetyGuard (``so101 rest-pose`` passes the one
    the saved calibration builds; the HIL runner the server's).
    """
    return resolve_collapse_rest_pose(settings, limits=limits)


def model_rest_pose() -> dict[str, float]:
    """The pose derived from the official model (plain data; no MuJoCo import)."""
    from ..simulation.rest_pose import MODEL_REST_POSE

    return dict(MODEL_REST_POSE)


def model_joint_limits() -> dict[str, tuple[float, float]]:
    """What the vendored MJCF can reach, in suite units (parsed XML; no MuJoCo)."""
    from ..simulation.units import mjcf_limits

    return dict(mjcf_limits())


def intersect_limits(
    first: Mapping[str, tuple[float, float]], second: Mapping[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Per-joint intersection of two limit maps. Raises when a joint's ranges do not meet."""
    limits: dict[str, tuple[float, float]] = {}
    for joint in JOINT_NAMES:
        low_a, high_a = first.get(joint, DEFAULT_LIMITS[joint])
        low_b, high_b = second.get(joint, (low_a, high_a))
        low, high = max(float(low_a), float(low_b)), min(float(high_a), float(high_b))
        if low > high:
            raise RestPoseError(
                f"{joint}: the SafetyGuard range ({low_a:g}, {high_a:g}) and the model range "
                f"({low_b:g}, {high_b:g}) do not overlap"
            )
        limits[joint] = (low, high)
    return limits


def resolve_collapse_rest_pose(
    settings: Any,
    *,
    limits: Mapping[str, tuple[float, float]] | None = None,
    backend: Backend | None = None,
) -> RestPoseResolution:
    """The pose Collapse lowers the arm to (design doc section 5 "Rest pose").

    Every bring-up view uses it too (:func:`resolve_for_settings`). Through
    :func:`resolve_rest_pose`: the recorded pose (bound to the follower id and
    calibration), then ``SO101_REST_POSE``, then the model-derived default.
    Every source is clamped to ``limits`` (the active SafetyGuard limits)
    intersected with the MJCF limits. When the model limits cannot be read,
    the clamp falls back to the SafetyGuard limits alone and says so in
    ``warnings``. ``backend`` defaults to the settings' backend. Raises
    :class:`RestPoseError` when no pose can be resolved.
    """
    guard_limits = dict(limits or DEFAULT_LIMITS)
    warnings: list[str] = []
    label = "SafetyGuard ∩ MJCF limits"
    try:
        effective = intersect_limits(guard_limits, model_joint_limits())
    except RestPoseError:
        raise
    except Exception as exc:  # a missing or corrupt vendored model
        warnings.append(f"model joint limits unavailable ({exc}); clamped to SafetyGuard only")
        effective, label = guard_limits, "SafetyGuard limits"
    resolution = resolve_rest_pose(
        settings,
        follower_id=settings.follower_id,
        calibration_digest=follower_calibration_digest(settings),
        limits=effective,
        backend=backend,
        default_pose=model_rest_pose(),
        limits_label=label,
    )
    if warnings:
        resolution = resolution.model_copy(update={"warnings": [*warnings, *resolution.warnings]})
    return resolution


def measure_rest_pose(
    read_positions: Callable[[], Mapping[str, float]],
    *,
    samples: int = RECORD_SAMPLES,
    interval_s: float = RECORD_INTERVAL_S,
    max_spread: float = RECORD_MAX_SPREAD,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, float], dict[str, float]]:
    """Sample positions and return (mean pose, per-joint spread).

    Raises :class:`RestPoseUnstableError` when any joint spreads by
    ``max_spread`` or more (the arm has not settled).
    """
    if samples < 2:
        raise ValueError("need at least two samples to measure spread")
    readings: list[dict[str, float]] = []
    for index in range(samples):
        if index:
            sleep(interval_s)
        reading = read_positions()
        readings.append(_finite_joint_map(reading, what="position read"))
    spread = {
        joint: max(r[joint] for r in readings) - min(r[joint] for r in readings)
        for joint in JOINT_NAMES
    }
    unstable = {joint: value for joint, value in spread.items() if value >= max_spread}
    if unstable:
        detail = ", ".join(f"{joint} {value:.2f}" for joint, value in unstable.items())
        raise RestPoseUnstableError(
            f"the arm has not settled (spread >= {max_spread:g}: {detail}); let go, wait, retry",
            spread,
        )
    mean = {joint: sum(r[joint] for r in readings) / len(readings) for joint in JOINT_NAMES}
    return mean, spread

"""Calibration: capture each joint's range of motion and a homing offset.

Real calibration asks the operator to move each arm through its full range;
the recorded min/max define the usable range and the midpoint the offset.
In mock mode it returns sensible defaults instantly so the flow is testable.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import JOINT_NAMES
from ..config import get_settings
from ..logging import get_logger
from .motors import FeetechBus, MockBus, MotorBus
from .safety import DEFAULT_LIMITS

log = get_logger(__name__)

# Under SO101_BACKEND=sim make_bus builds the simulated arm, which has no
# encoder ticks: calibrating it would sample a motionless arm and overwrite the
# real calibration file (simulation/devices.py "Integration points"). The
# calibration route and `so101 calibrate` refuse with this.
SIM_CALIBRATION_REFUSED = (
    "calibration has no meaning on the sim backend (the simulated arm has no encoders) "
    "and would overwrite the real calibration; use SO101_BACKEND=real"
)


# On the real backend a calibration a mock run saved counts as missing: the
# data folder is shared, and the mock path writes DEFAULT_LIMITS, not
# anything measured on an arm. Diagnostics, the calibration status route,
# preflight and the follower's SafetyGuard all apply it.
MOCK_CALIBRATION_ON_REAL = (
    "the saved calibration was made on the mock backend, not on this arm, so it counts as "
    "missing; recalibrate on the real arm"
)


@dataclass
class JointCalibration:
    min: float
    max: float
    offset: float  # midpoint, subtracted to centre the joint


@dataclass
class CalibrationResult:
    arm: str  # "follower" | "leader"
    joints: dict[str, JointCalibration] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    schema_version: int = 2
    device_id: str = ""
    port: str = ""
    backend: str = ""
    joint: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        if self.joint is not None and self.joint not in JOINT_NAMES:
            raise ValueError("unknown calibration joint")
        filename = (
            f"calibration_joint_{self.arm}_{self.joint}.json"
            if self.joint
            else f"calibration_{self.arm}.json"
        )
        path = path or (get_settings().data_dir / filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "arm": self.arm,
            "created_at": self.created_at,
            "device_id": self.device_id,
            "port": self.port,
            "backend": self.backend,
            "joints": {k: asdict(v) for k, v in self.joints.items()},
        }
        if self.joint:
            payload.update(joint=self.joint, diagnostics=self.diagnostics, baseline_only=True)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        log.info("saved calibration for %s -> %s", self.arm, path)
        return path

    @classmethod
    def load(cls, arm: str, path: Path | None = None) -> CalibrationResult | None:
        path = path or (get_settings().data_dir / f"calibration_{arm}.json")
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        joints = {k: JointCalibration(**v) for k, v in raw["joints"].items()}
        if raw.get("arm") != arm:
            raise ValueError(f"calibration arm mismatch: expected {arm!r}, got {raw.get('arm')!r}")
        return cls(
            arm=raw["arm"],
            joints=joints,
            created_at=raw["created_at"],
            schema_version=int(raw.get("schema_version", 1)),
            device_id=str(raw.get("device_id", "")),
            port=str(raw.get("port", "")),
            backend=str(raw.get("backend", "legacy")),
            joint=raw.get("joint"),
            diagnostics=raw.get("diagnostics", {}),
        )

    @classmethod
    def load_joint(cls, arm: str, joint: str) -> CalibrationResult | None:
        if arm not in {"follower", "leader"} or joint not in JOINT_NAMES:
            raise ValueError("invalid arm or joint")
        path = get_settings().data_dir / f"calibration_joint_{arm}_{joint}.json"
        result = cls.load(arm, path)
        if result is not None and (result.joint != joint or set(result.joints) != {joint}):
            raise ValueError("joint baseline identity mismatch")
        return result


def is_mock_calibration(calibration: CalibrationResult) -> bool:
    """Whether a saved calibration came from a mock run rather than a real arm.

    A mock run saves ``device_id`` and ``backend`` "mock". A file from before
    those fields carries neither; it is a mock run's when every joint holds
    exactly DEFAULT_LIMITS with the midpoint offset, which is what the mock
    path writes and what no measured sweep produces.
    """
    if calibration.device_id == "mock" or calibration.backend == "mock":
        return True
    if calibration.device_id or calibration.backend not in {"", "legacy"}:
        return False
    if set(calibration.joints) != set(JOINT_NAMES):
        return False
    for joint in JOINT_NAMES:
        lo, hi = DEFAULT_LIMITS[joint]
        measured = calibration.joints[joint]
        if (measured.min, measured.max, measured.offset) != (lo, hi, (lo + hi) / 2):
            return False
    return True


def counts_as_missing(calibration: CalibrationResult | None, backend: str) -> bool:
    """On the real backend, a mock run's calibration is no calibration at all."""
    return calibration is not None and backend == "real" and is_mock_calibration(calibration)


def calibrate(
    bus: MotorBus,
    arm: str,
    *,
    sample_seconds: float = 8.0,
    progress=None,
    cancel_event: threading.Event | None = None,
    telemetry=None,
    joint: str | None = None,
) -> CalibrationResult:
    """Sample joint positions while the operator sweeps each joint.

    ``progress`` is an optional callable ``(fraction: float, msg: str)`` for
    UI/CLI feedback. With a mock bus this completes instantly with defaults.
    """
    if joint is not None and joint not in JOINT_NAMES:
        raise ValueError("unknown calibration joint")
    selected = [joint] if joint else list(JOINT_NAMES)
    result = CalibrationResult(
        arm=arm,
        joint=joint,
        device_id=(bus.device_id if isinstance(bus, FeetechBus) else "mock"),
        port=str(getattr(bus, "port", "")),
        backend=("lerobot" if isinstance(bus, FeetechBus) else "mock"),
    )

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("calibration cancelled")

    if isinstance(bus, MockBus):
        for j in selected:
            lo, hi = DEFAULT_LIMITS[j]
            result.joints[j] = JointCalibration(min=lo, max=hi, offset=(lo + hi) / 2)
        if progress:
            progress(1.0, "mock calibration complete")
        return result

    if isinstance(bus, FeetechBus):
        ranges = bus.calibrate_timed(
            sample_seconds,
            progress=progress,
            cancel_event=cancel_event,
            telemetry=telemetry,
            joint=joint,
        )
        if joint:
            result.diagnostics = dict(bus.calibration_diagnostics)
        for name in selected:
            lo, hi = ranges[name]
            result.joints[name] = JointCalibration(
                min=lo,
                max=hi,
                offset=(lo + hi) / 2,
            )
        return result

    if joint:
        raise RuntimeError("joint baselines require the pinned Feetech backend or mock")
    mins = {j: float("inf") for j in JOINT_NAMES}
    maxs = {j: float("-inf") for j in JOINT_NAMES}
    t_end = time.monotonic() + sample_seconds
    while time.monotonic() < t_end:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("calibration cancelled")
        pos = bus.read_positions()
        for j in JOINT_NAMES:
            if j in pos:
                mins[j] = min(mins[j], pos[j])
                maxs[j] = max(maxs[j], pos[j])
        if progress:
            frac = 1 - (t_end - time.monotonic()) / sample_seconds
            progress(max(0.0, min(1.0, frac)), "move each joint through its range")
        time.sleep(0.02)

    for j in JOINT_NAMES:
        lo, hi = mins[j], maxs[j]
        result.joints[j] = JointCalibration(min=lo, max=hi, offset=(lo + hi) / 2)
    return result


def affected_motors(error: str | None) -> list[str]:
    # New errors may include every motor's starting readings after the first
    # semicolon. Only the error clause identifies which motors actually failed.
    clause = (error or "").split(";", 1)[0]
    return [name for name in JOINT_NAMES if re.search(rf"\b{name}\b", clause)]

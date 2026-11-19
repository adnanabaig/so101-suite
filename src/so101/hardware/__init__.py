"""Hardware abstraction for the SO-101 arm (motors, cameras, safety).

Mock hardware is available only when ``Settings.mock`` is explicitly set.
Real motor-driver and connection failures remain visible and fail closed so a
session cannot unknowingly move a simulated arm or record mislabeled data.
"""

from .calibration import CalibrationResult, calibrate
from .cameras import CameraManager, CameraUnavailableError
from .follower import SO101Follower
from .leader import SO101Leader
from .motors import FeetechBus, HardwareUnavailableError, MockBus, MotorBus, make_bus
from .safety import SafetyGuard, SafetyViolation

__all__ = [
    "CameraManager",
    "CameraUnavailableError",
    "CalibrationResult",
    "calibrate",
    "SO101Follower",
    "SO101Leader",
    "FeetechBus",
    "HardwareUnavailableError",
    "MockBus",
    "MotorBus",
    "make_bus",
    "SafetyGuard",
    "SafetyViolation",
]

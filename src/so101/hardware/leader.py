"""SO-101 leader arm: the passive arm a human moves to teleoperate.

Torque is left off so it back-drives freely; we only read its joints.
"""

from __future__ import annotations

import math
import time

from .. import JOINT_NAMES
from ..config import get_settings
from ..logging import get_logger
from .motors import MockBus, MotorBus, make_bus

log = get_logger(__name__)


class SO101Leader:
    def __init__(self, port: str | None = None):
        settings = get_settings()
        self.port = port or settings.leader_port
        self.bus: MotorBus = make_bus(self.port, name="leader")
        self._t0 = time.monotonic()

    def connect(self) -> None:
        try:
            self.bus.connect()
            # Leader is back-driven by hand: no torque.
            self.bus.set_torque(False)
        except BaseException:
            try:
                self.bus.disconnect()
            except BaseException:
                pass
            raise

    def disconnect(self) -> None:
        if self.bus.connected:
            self.bus.disconnect()

    @property
    def connected(self) -> bool:
        return self.bus.connected

    def read_positions(self) -> dict[str, float]:
        pos = self.bus.read_positions()
        # In mock mode the leader has no human driving it, so synthesize a
        # gentle demo trajectory that the follower can mirror.
        if isinstance(self.bus, MockBus):
            t = time.monotonic() - self._t0
            pos = {
                "shoulder_pan": 40 * math.sin(t * 0.5),
                "shoulder_lift": 30 * math.sin(t * 0.4 + 1),
                "elbow_flex": 35 * math.sin(t * 0.6 + 2),
                "wrist_flex": 25 * math.sin(t * 0.7),
                "wrist_roll": 60 * math.sin(t * 0.3),
                "gripper": 50 + 50 * math.sin(t * 0.9),
            }
        return {j: pos.get(j, 0.0) for j in JOINT_NAMES}

    def action(self) -> dict[str, float]:
        """Action dict keyed ``<joint>.pos`` (LeRobot convention)."""
        pos = self.read_positions()
        return {f"{j}.pos": pos[j] for j in JOINT_NAMES}

    def __enter__(self) -> SO101Leader:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

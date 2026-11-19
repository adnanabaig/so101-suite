"""The teleoperation loop: leader action -> safety -> follower.

Designed to be driven step-by-step so the same loop powers the CLI, the
WebSocket telemetry stream, and the dataset recorder without duplication.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import JOINT_NAMES
from ..hardware.follower import SO101Follower
from ..hardware.leader import SO101Leader
from ..logging import get_logger

if TYPE_CHECKING:
    from ..bringup.standalone import StandaloneAccess

log = get_logger(__name__)


@dataclass
class TeleopStep:
    """One control cycle's worth of data (also the per-frame record schema)."""

    t: float
    observation: dict[str, float]  # follower joints, <joint>.pos
    # Actual command accepted by SafetyGuard and sent to the follower.
    action: dict[str, float]
    # Raw leader intent before position/velocity clamping. Kept separate so a
    # dataset never trains on an action the robot did not execute.
    desired_action: dict[str, float] = field(default_factory=dict)


class TeleopController:
    def __init__(
        self,
        leader: SO101Leader,
        follower: SO101Follower,
        fps: int = 30,
        standalone_access: StandaloneAccess | None = None,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 240:
            raise ValueError("fps must be an integer between 1 and 240")
        self.leader = leader
        self.follower = follower
        self.fps = fps
        self.period = 1.0 / fps
        self._standalone_access = standalone_access

    def _disconnect_follower(self) -> None:
        if self._standalone_access is None:
            self.follower.disconnect()
        else:
            self._standalone_access.disconnect_follower(self.follower)

    def _disconnect_leader(self) -> None:
        if self._standalone_access is None:
            self.leader.disconnect()
        else:
            self._standalone_access.disconnect_device(self.leader, device_name="leader")

    def connect(self) -> None:
        if self._standalone_access is not None:
            # Teleop/recording opens the passive leader before the follower.
            # Write intent for the whole configured pair first so a crash or
            # uncertain leader setup cannot escape cross-data-dir recovery.
            self._standalone_access.prepare_to_energize(device_name="")
        try:
            self.leader.connect()
        except BaseException:
            self._disconnect_leader()
            raise
        try:
            if self._standalone_access is not None:
                self._standalone_access.connect_follower(self.follower)
            else:
                self.follower.connect()
        except BaseException:
            try:
                if self._standalone_access is None:
                    self._disconnect_follower()
            finally:
                self._disconnect_leader()
            raise

    def disconnect(self) -> None:
        if self._standalone_access is not None:
            try:
                self._disconnect_leader()
            finally:
                # The all-device energized intent is cleared only after the
                # passive leader and actuated follower both confirmed off.
                self._disconnect_follower()
            return
        try:
            self._disconnect_follower()
        finally:
            self._disconnect_leader()

    def halt(self) -> None:
        """Successful standalone end: release leader, hold follower energized."""

        if self._standalone_access is None:
            self.disconnect()
            return
        try:
            self._disconnect_leader()
        except BaseException:
            self._disconnect_follower()
            raise
        self._standalone_access.halt_follower(self.follower)

    def step(self) -> TeleopStep:
        desired_action = self.leader.action()
        goal = {
            joint: desired_action[f"{joint}.pos"]
            for joint in JOINT_NAMES
            if f"{joint}.pos" in desired_action
        }
        commanded = self.follower.send_goal(goal)
        action = {f"{joint}.pos": commanded[joint] for joint in JOINT_NAMES if joint in commanded}
        obs = self.follower.observation()
        return TeleopStep(
            t=time.time(),
            observation=obs,
            action=action,
            desired_action=desired_action,
        )

    def run(self, duration: float | None = None) -> Iterator[TeleopStep]:
        """Yield :class:`TeleopStep`s at ``fps`` until ``duration`` elapses
        (or forever if ``None``). The caller controls stopping."""
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            raise ValueError("duration must be finite and non-negative")
        t_end = None if duration is None else time.monotonic() + duration
        next_t = time.monotonic()
        while t_end is None or time.monotonic() < t_end:
            yield self.step()
            next_t += self.period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # we fell behind; resync

    def __enter__(self) -> TeleopController:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.halt()
        else:
            self.disconnect()

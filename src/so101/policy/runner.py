"""Closed-loop autonomous rollout: policy -> safety -> follower."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .. import JOINT_NAMES
from ..bringup.standalone import StandaloneAccess, refuse_real_motion, standalone_access
from ..hardware.cameras import CameraManager, stream_feature_names
from ..hardware.follower import SO101Follower
from ..logging import get_logger
from .server import Policy

log = get_logger(__name__)


@dataclass
class RolloutResult:
    steps: int
    duration: float
    success: bool | None = None  # set by the operator/eval harness


def _environment_state(follower: object) -> np.ndarray | None:
    """Simulator object state, when the follower provides it.

    ``SO101Follower.environment_state()`` returns the simulated arm's object
    state under ``SO101_BACKEND=sim`` (``MujocoBus``) and adds
    ``observation.environment_state``, the true object state a
    privileged-state baseline reads. On a real or mock arm it returns None,
    which means "not available". The class is checked, not the instance, so a
    stand-in follower never fakes the hook.
    """

    if not callable(getattr(type(follower), "environment_state", None)):
        return None
    value = follower.environment_state()  # type: ignore[attr-defined]
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _declared_inputs(policy: object) -> frozenset[str] | None:
    """The observation keys a policy declares (LeRobot ``config.input_features``), if any."""
    keys = getattr(policy, "input_feature_keys", None)
    return frozenset(keys) if keys else None


def _build_observation(
    follower: SO101Follower,
    frames: dict[str, np.ndarray],
    *,
    policy: object = None,
    feature_names: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """The policy observation for one control step.

    Camera frames are keyed ``observation.images.<name>``: ``feature_names``
    (``CameraManager.feature_names``) when given, else
    :func:`stream_feature_names` of the frame keys, which gives a depth stream
    its color camera's index. When ``policy`` declares its inputs, only those
    are passed, so a color-only policy never sees a depth frame.
    """
    pos = follower.read_positions()
    state = np.array([pos.get(j, 0.0) for j in JOINT_NAMES], dtype=np.float32)
    obs: dict[str, np.ndarray] = {"observation.state": state}
    env_state = _environment_state(follower)
    if env_state is not None:
        obs["observation.environment_state"] = env_state
    names = feature_names if feature_names is not None else stream_feature_names(list(frames))
    for cam, img in frames.items():
        obs[f"observation.images.{names[cam]}"] = img
    declared = _declared_inputs(policy)
    if declared is not None:
        obs = {key: value for key, value in obs.items() if key in declared}
    return obs


class PolicyRunner:
    """Drives the follower from a policy at a fixed control rate.

    :meth:`connect` opens the follower itself, so it first passes the
    standalone gate (:mod:`so101.bringup.standalone`): ``StandaloneRefused``
    before anything opens while a safety latch is set, the latch store is
    unreadable, or (real backend) a server owns the bus. :meth:`disconnect`
    gives the access back. A follower already connected when :meth:`rollout`
    starts belongs to the caller; it avoids device ownership setup, but not
    the real-motion refusal below.

    Real-hardware rollout is refused even with a caller-owned follower: it
    must run through RobotManager, which owns the watchdog and timed stop
    lifecycle. Mock and simulation callers may still let this class own or
    receive their follower.
    """

    def __init__(
        self,
        policy: Policy,
        follower: SO101Follower | None = None,
        cameras: CameraManager | None = None,
        fps: int = 30,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 240:
            raise ValueError("fps must be an integer between 1 and 240")
        self.policy = policy
        self.follower = follower or SO101Follower()
        self.cameras = cameras or CameraManager()
        self.fps = fps
        self.period = 1.0 / fps
        self._access: StandaloneAccess | None = None

    def connect(self) -> None:
        if self._access is None:
            self._access = standalone_access("PolicyRunner (policy rollout)", motion=True)
        access = self._access
        assert access is not None
        follower_may_have_opened = False
        try:
            access.connect_follower(self.follower)
            follower_may_have_opened = True
            self.cameras.open()
            self.policy.reset()
        except BaseException:
            try:
                if follower_may_have_opened:
                    self._disconnect_follower()
            finally:
                try:
                    self.cameras.close()
                finally:
                    self.release_access()
            raise

    def disconnect(self) -> None:
        try:
            self._disconnect_follower()
        finally:
            try:
                self.cameras.close()
            finally:
                self.release_access()

    def _disconnect_follower(self) -> None:
        access = self._access
        if access is None:
            self.follower.disconnect()
        else:
            access.disconnect_follower(self.follower)

    def release_access(self) -> None:
        """Give back the standalone device-lock access taken by :meth:`connect`."""
        access, self._access = self._access, None
        if access is not None:
            access.release()

    def disconnect_owned(self) -> None:
        """Close only a connection for which this runner still holds standalone access."""

        if self._access is not None:
            self.disconnect()

    def halt_owned(self) -> None:
        """Halt and close a standalone-owned follower without turning torque off."""

        access = self._access
        if access is None:
            return
        try:
            access.halt_follower(self.follower)
        finally:
            try:
                self.cameras.close()
            finally:
                self.release_access()

    def step(self) -> dict[str, float]:
        if self._access is None and self.follower.connected:
            refuse_real_motion("PolicyRunner.step with a caller-owned follower")
        frames = self.cameras.read()
        obs = _build_observation(
            self.follower,
            frames,
            policy=self.policy,
            # Camera stand-ins without feature_names get the default naming.
            feature_names=getattr(self.cameras, "feature_names", None),
        )
        action = self.policy.select_action(obs)
        return self.follower.send_goal(action)

    def run(self, duration: float = 30.0) -> Iterator[dict[str, float]]:
        """Yield the commanded action each control step for ``duration`` s."""
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")
        if self._access is None and self.follower.connected:
            refuse_real_motion("PolicyRunner with a caller-owned follower")
        next_t = time.monotonic()
        t_end = next_t + duration
        while time.monotonic() < t_end:
            yield self.step()
            next_t += self.period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    def rollout(self, duration: float = 30.0) -> RolloutResult:
        """Run one episode. The policy always starts it from a clean state.

        ``connect()`` resets the policy when this call owns the connection;
        otherwise (an eval running several episodes on one connection) it is
        reset here, so ACT does not open the episode with up to
        ``n_action_steps - 1`` stale actions planned for the previous scene
        (audit B11).

        Real followers are rejected whether owned here or passed by a caller;
        supported real rollout uses RobotManager's own policy loop.
        """
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")
        if self._access is None and self.follower.connected:
            refuse_real_motion("PolicyRunner with a caller-owned follower")
        own = not self.follower.connected
        t0 = time.monotonic()
        n = 0
        completed = False
        try:
            if own:
                self.connect()
            else:
                self.policy.reset()
            for _ in self.run(duration):
                n += 1
            completed = True
        finally:
            if own:
                if completed:
                    self.halt_owned()
                else:
                    self.disconnect_owned()
        return RolloutResult(steps=n, duration=time.monotonic() - t0)

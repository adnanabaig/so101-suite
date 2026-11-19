"""Simulated robot devices: one MuJoCo SO-101 behind the hardware interfaces.

With ``SO101_BACKEND=sim``, these objects replace the Feetech buses and the
OpenCV cameras. Every device shares one :class:`SimRuntime`. The runtime holds
a single :class:`~so101.simulation.world.PhysicsWorld` built from the
``SO101_SIM_TASK`` scene. A :class:`~so101.simulation.servo.ServoDriver` in
``sts3215`` mode drives the arm, so it follows Goal_Position along the STS3215
trapezoid exactly as in ``so101 sim demos`` and ``so101 sim eval``.

Devices
-------
:class:`MujocoBus` is the follower arm. It implements ``MotorBus`` with the
``FeetechBus`` semantics:

- It uses the suite units (degrees, gripper 0..100) and returns all six joints.
- Read, write and ``set_torque`` raise ``RuntimeError("bus not connected")``
  until ``connect``. ``MockBus`` is lenient here; this bus is not.
- ``write_positions`` takes a partial goal and ignores unknown joints. A
  non-finite value raises ``ValueError`` and nothing is applied. It returns the
  goal the servo actually tracks, for the written joints only: clipped to the
  effective limits and to the actuator range.
- The goal is a register, like Goal_Position RAM. It is written even while
  torque is off, and enabling torque drives the arm to it along the profile.
  ``SO101Follower`` primes it with the measured pose before enabling torque, as
  it must on hardware.
- ``disconnect`` disables torque first, as LeRobot's ``disconnect`` does.

:class:`SimExpertLeader` is the leader arm. It reports the targets of the
scripted expert (:class:`~so101.simulation.expert.ScriptedExpert`), which
observes the shared world once per 1/30 s control period. Writes are accepted
as feedback and ignored.

:class:`SimCamera` follows the camera object interface ``CameraManager``
uses: ``read() -> HxWx3 uint8 RGB``, ``release()``, and ``cam_id``, ``width``
and ``height``. Frames are rendered on the shared render thread
(:func:`so101.simulation.render.shared_renderer`) from a snapshot taken under
the runtime lock, so a frame never shows a half-finished step.

Power-on
--------
The runtime starts with the task's first layout (seed ``SO101_SIM_SEED``) and
the arm folded at :data:`~so101.simulation.rest_pose.MODEL_REST_POSE` with
torque off. That is how a real arm is found before anyone powers it. Nothing
moves until the control plane enables torque.

Time
----
``SO101_SIM_REALTIME=true`` (the default) runs a daemon thread,
``so101-sim-physics``, that advances the physics to the wall clock. It wakes
every 1/120 s and catches up by at most 0.25 s; any further backlog is dropped
and counted in ``status()["dropped_steps"]``. The arm tracks the latest goal
continuously, and physics keeps running when every device has disconnected, so
a de-energized arm sags under gravity.

``SO101_SIM_REALTIME=false`` (lockstep) runs no thread. A follower write with
torque on first simulates one control period (1/30 s: 8 substeps, then the
task update and the expert's next target) under the previous goal, and then
latches the new goal. Torque-off writes and all reads leave time untouched.
The observation read right after a write is therefore the state the new
command acts on, as with hardware and in real time. Teleop recordings pair
each observation with the action that followed it. A read-decide-write policy
loop does see its action one control period later than in real time. For
zero-latency policy evaluation use ``so101 sim eval``. In lockstep an
E-stopped (torque-off) arm does not sag unless someone calls
:meth:`SimRuntime.advance`. Each torque-on write is 1/30 s of simulated time
whatever the caller's rate, while the SafetyGuard rate limit runs on wall
time. A lockstep loop must therefore run at 30 Hz. ``RobotManager`` paces its
loops at their ``fps``, so lockstep needs ``fps=30``; use real time for any
other rate.

Episodes
--------
:func:`on_episode_start` reseeds the task. By default the seed is
``SO101_SIM_SEED`` plus the number of episodes started, so the first recorded
episode uses the power-on layout. It places the arm at the task's start pose,
as ``EpisodeRunner`` does, and keeps the torque state: a de-energized arm stays
de-energized. :func:`on_episode_end` returns ``{"success", "metrics"}``.
``success`` uses the rule of the runner and ``so101 sim``: the success
condition held at some point in the episode. It is never true after a physics
fault.

Integration points
------------------
This module needs the ``[sim]`` extra, so import it lazily inside the
``backend == "sim"`` branches.

- ``hardware/motors.py:make_bus``: when ``settings.backend == "sim"``, return
  ``make_sim_bus(name)``. ``SO101Follower`` and ``SO101Leader`` already pass
  ``name="follower"`` and ``name="leader"``.
- ``hardware/cameras.py:CameraManager.__init__``: with the sim backend,
  default ``cam_ids`` to ``settings.sim_camera_names``.
  ``CameraManager._open_one``: return ``make_sim_camera(cid)``. It raises
  ``CameraUnavailableError`` for an unknown camera or when rendering is
  unavailable, the same way a real camera fails. ``CameraManager._cams`` is
  annotated ``dict[str, _CvCamera | _MockCamera]``. Add ``SimCamera`` to it (or
  a ``Protocol``) if the sim camera is stored anywhere but through the
  unannotated ``_open_one``.
- ``services/robot_manager.py:RobotManager.begin_episode`` (wired): under
  ``self._io_lock``, so no teleop step interleaves, it calls
  ``info = on_episode_start()`` and then
  ``follower.guard.seed(follower.read_positions())``. The re-seed is
  mandatory: the arm has just been moved to the start pose, and a guard still
  anchored to the old pose would rate-limit the next command from there.
  ``info["instruction"]`` is the task text.
- ``RobotManager.start_recording`` (wired): records
  ``observation.environment_state`` with the task's ``env_state_names``; the
  recording loop reads it through :meth:`recording_snapshot`.
- ``RobotManager`` recording (wired): every selected frame gets its environment
  state and predicate grade from one :meth:`recording_snapshot` runtime-lock
  barrier. Save closes the take, waits for its pending final write, then uses
  that frame's grade while retaining the dataset-operation lock through label
  publication. The outcome source is ``auto:sim_predicate``.
- ``RobotManager._rollout_loop`` (optional): call ``on_episode_start()``
  before ``follower.connect()``, so the connect seeds the guard, and
  ``on_episode_end()`` when the loop stops. A policy trained with
  ``observation.environment_state`` already gets it:
  ``SO101Follower.environment_state()`` returns
  :meth:`MujocoBus.environment_state`, and the policy runner adds it to
  every observation.
- ``RobotManager.shutdown``: call ``stop_sim_runtime()``. ``reset_manager``
  (tests) should call ``reset_sim_runtime()``.
- Calibration has no meaning in sim, because there are no encoder ticks, so
  ``services/routers/calibration.py:start`` and ``cli.py:calibrate`` must
  refuse the sim backend before calling ``make_bus``. Otherwise
  ``hardware/calibration.py:calibrate`` takes its generic path for a bus that
  is neither ``MockBus`` nor ``FeetechBus``. It samples a motionless arm, so
  min equals max for every joint, and ``save()`` overwrites the real
  calibration file. Also, the extra ``MujocoBus`` that ``make_bus`` returns
  shares the one world, and its ``disconnect`` turns the arm's torque off.

The ``/api/simulation`` engine is a separate sandbox world and never shares
state with these devices.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import mujoco
import numpy as np

from .. import JOINT_NAMES
from ..config import get_settings
from ..hardware.cameras import CameraUnavailableError
from ..hardware.motors import MotorBus
from ..logging import get_logger
from .expert import ExpertConfig, ScriptedExpert
from .render import FrameRenderer, check_size, shared_renderer
from .rest_pose import MODEL_REST_POSE
from .rollout import ENV_STATE_KEY, STATE_KEY, action_array
from .servo import ServoDriver, ServoMode
from .tasks import CONTROL_HZ, Task, make_task
from .units import suite_to_mjcf_array
from .world import CAMERAS

log = get_logger(__name__)

CONTROL_DT_S = 1.0 / CONTROL_HZ
PHYSICS_WAKE_S = 1.0 / 120.0  # how often the real-time thread catches up with the clock
MAX_CATCHUP_S = 0.25  # a longer stall is dropped rather than replayed at full speed
POWER_ON_SETTLE_S = 0.5  # the folded, de-energized arm comes to rest on its stops
FRAME_TIMEOUT_S = 5.0
DEVICE_KINDS = ("follower", "leader", "camera")
_DIVERGENCE = {
    int(mujoco.mjtWarning.mjWARN_BADQPOS): "bad qpos",
    int(mujoco.mjtWarning.mjWARN_BADQVEL): "bad qvel",
    int(mujoco.mjtWarning.mjWARN_BADQACC): "bad qacc",
}


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy values become Python values; NaN and inf become ``None``."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SimRuntime:
    """One simulated SO-101 shared by every sim device. Thread-safe.

    ``task``, ``realtime`` and ``seed`` default to ``SO101_SIM_TASK``,
    ``SO101_SIM_REALTIME`` and ``SO101_SIM_SEED``. ``clock`` paces the
    real-time thread. ``renderer`` defaults to the shared render thread, which
    the runtime never closes.
    """

    def __init__(
        self,
        task: str | None = None,
        *,
        realtime: bool | None = None,
        seed: int | None = None,
        servo_mode: ServoMode = "sts3215",
        expert_config: ExpertConfig | None = None,
        renderer: FrameRenderer | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        settings = get_settings()
        self.task: Task = make_task(task or settings.sim_task)
        self.realtime = settings.sim_realtime if realtime is None else bool(realtime)
        self.base_seed = settings.sim_seed if seed is None else int(seed)
        self.world = self.task.make_world()
        self.driver = ServoDriver(self.world, mode=servo_mode)
        self.control_substeps = self.world.steps_for(CONTROL_DT_S)
        self._expert_config = expert_config or ExpertConfig()
        self._renderer = renderer
        self._clock = clock
        self._lock = threading.RLock()  # world, task, expert and register state
        self._lifecycle = threading.Lock()  # the physics thread handle; never held by it
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._devices: dict[int, str] = {}
        self._expert: ScriptedExpert | None = None
        self._expert_action: np.ndarray | None = None
        self._expert_error: str | None = None
        self._fault: str | None = None
        self._phase = 0  # substeps since the last control tick
        self._steps = 0
        self._dropped = 0
        self._episodes = 0
        self._episode_seed = self.base_seed
        self._last_metrics: dict[str, float] = {}
        self._goal = np.zeros(len(JOINT_NAMES))
        with self._lock:
            self.task.reset(self.world, self.base_seed)
            self._power_on()

    def _power_on(self) -> None:
        """Fold the arm at the model rest pose with torque off and let it settle."""

        world = self.world
        rest = world.joint_vector(MODEL_REST_POSE, world.joint_array())
        world.data.qpos[world.index.qpos] = suite_to_mjcf_array(rest)
        world.data.qvel[world.index.dof] = 0.0
        self.driver.set_torque(False)
        self.driver.step(world.steps_for(POWER_ON_SETTLE_S))  # torque off: tracks the arm
        world.data.time = 0.0
        # Goal_Position RAM holds the present position at boot.
        self._goal = self.driver.clip_goal(world.joint_array())

    # -- lifecycle -------------------------------------------------------------------
    @property
    def running(self) -> bool:
        """Whether the real-time physics thread is alive."""

        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def renderer(self) -> FrameRenderer:
        return self._renderer or shared_renderer()

    @property
    def fault(self) -> str | None:
        return self._fault

    def start(self) -> None:
        """Start the real-time physics thread. Idempotent; a lockstep runtime has none."""

        if not self.realtime:
            return
        with self._lifecycle:
            if self.running:
                return
            self._check_fault()
            stop = threading.Event()
            # Simulated time accrues from now, not from whenever the thread first runs.
            thread = threading.Thread(
                target=self._run,
                args=(stop, self._clock()),
                name="so101-sim-physics",
                daemon=True,
            )
            self._thread, self._stop_event = thread, stop
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the physics thread. Idempotent. The world keeps its state; ``start`` resumes."""

        with self._lifecycle:
            thread, stop = self._thread, self._stop_event
            self._thread, self._stop_event = None, None
            if thread is None or stop is None:
                return
            stop.set()
            if thread is not threading.current_thread():
                thread.join(timeout)

    def advance(self, duration_s: float) -> None:
        """Advance time by hand (lockstep, or a stopped real-time runtime)."""

        if self.running:
            raise RuntimeError("the real-time physics thread is running; stop() it first")
        with self._lock:
            self._check_fault()
            self._advance(self.world.steps_for(duration_s))

    def _run(self, stop: threading.Event, anchor: float) -> None:
        dt = self.world.timestep_s
        max_steps = max(1, self.world.steps_for(MAX_CATCHUP_S))
        done = 0
        while not stop.wait(PHYSICS_WAKE_S):
            due = math.floor((self._clock() - anchor) / dt) - done
            if due <= 0:
                continue
            with self._lock:
                if due > max_steps:
                    self._dropped += due - max_steps
                    done += due - max_steps
                    due = max_steps
                if self._fault is not None:
                    return
                try:
                    self._advance(due)
                except Exception as exc:  # recorded as the fault; devices raise it
                    self._set_fault(f"{type(exc).__name__}: {exc}")
                    return
            done += due

    # -- stepping (call with self._lock held) -------------------------------------------
    def _advance(self, steps: int) -> None:
        """Run ``steps`` substeps, with a control tick at every 1/30 s boundary.

        A divergence latches the fault and raises. MuJoCo silently resets a
        diverged state, and only its warning counters remember it, so they are
        checked after every chunk, before the task or the expert sees the state.
        """

        remaining = int(steps)
        try:
            while remaining > 0 and self._fault is None:
                chunk = min(remaining, self.control_substeps - self._phase)
                self.driver.step(chunk)
                self._steps += chunk
                self._phase += chunk
                remaining -= chunk
                self._check_divergence()
                if self._fault is None and self._phase >= self.control_substeps:
                    self._phase = 0
                    self._control_tick()
        except Exception as exc:
            self._set_fault(f"{type(exc).__name__}: {exc}")
        self._check_fault()

    def _check_divergence(self) -> None:
        warnings = self.world.data.warning
        for kind, name in _DIVERGENCE.items():
            if warnings[kind].number > 0:
                self._set_fault(f"physics diverged ({name}); MuJoCo reset the state")
                return

    def _control_tick(self) -> None:
        self._last_metrics = self.task.update(self.world, CONTROL_DT_S)
        if self._expert is not None and self._leader_count() > 0:
            self._next_expert_action()

    def _observe(self) -> dict[str, np.ndarray]:
        return {
            STATE_KEY: self.world.joint_array().astype(np.float32),
            ENV_STATE_KEY: self.task.env_state(self.world),
        }

    def _next_expert_action(self) -> None:
        if self._expert is None or self._expert_error is not None:
            return
        try:
            self._expert_action = action_array(self._expert.select_action(self._observe()))
        except Exception as exc:  # the leader reports it; the physics carries on
            self._expert_action = None
            self._expert_error = f"{type(exc).__name__}: {exc}"
            log.error("sim expert failed: %s", self._expert_error)

    def _restart_expert(self) -> None:
        if self._expert is None:
            self._expert = ScriptedExpert(self.task, self._expert_config)
        self._expert.reset(self._episode_seed)
        self._expert_action = None
        self._expert_error = None
        self._next_expert_action()

    def _leader_count(self) -> int:
        return sum(1 for kind in self._devices.values() if kind == "leader")

    def _set_fault(self, message: str) -> None:
        if self._fault is None:
            self._fault = message
            log.error("simulation fault: %s", message)

    def _check_fault(self) -> None:
        if self._fault is not None:
            raise RuntimeError(f"simulation fault: {self._fault}")

    # -- devices ------------------------------------------------------------------------------
    def attach(self, device: object, kind: str) -> None:
        """Register a device. The first leader starts the expert from the present state."""

        if kind not in DEVICE_KINDS:
            raise ValueError(f"device kind must be one of {DEVICE_KINDS}, got {kind!r}")
        with self._lock:
            self._check_fault()
            first_leader = kind == "leader" and self._leader_count() == 0
            self._devices[id(device)] = kind
            if first_leader:
                self._restart_expert()

    def detach(self, device: object) -> None:
        with self._lock:
            self._devices.pop(id(device), None)

    def read_positions(self) -> dict[str, float]:
        """Present joint positions, suite units, all six joints."""

        with self._lock:
            self._check_fault()
            values = self.world.joint_array()
            if not np.all(np.isfinite(values)):
                self._set_fault("non-finite joint positions")
                self._check_fault()
        return dict(zip(JOINT_NAMES, (float(value) for value in values), strict=True))

    def write_goal(self, goal: Mapping[str, float]) -> dict[str, float]:
        """Write Goal_Position for the given joints. Returns what the servo tracks."""

        ordered = {joint: float(goal[joint]) for joint in JOINT_NAMES if joint in goal}
        if any(not math.isfinite(value) for value in ordered.values()):
            raise ValueError("goal contains a non-finite motor position")
        with self._lock:
            self._check_fault()
            if not ordered:
                return {}
            vector = self.world.joint_vector(ordered, self._goal)
            if self.world.torque_enabled:
                if not self.realtime:
                    self._advance(self.control_substeps)  # the period under the previous goal
                applied = self.driver.command(vector)
            else:
                applied = self.driver.clip_goal(vector)
            self._goal = applied
            return {joint: float(applied[JOINT_NAMES.index(joint)]) for joint in ordered}

    def set_torque(self, enabled: bool) -> None:
        """Torque on holds the present pose, then tracks the goal register. Off always works."""

        with self._lock:
            if not enabled:
                self.driver.set_torque(False)
                return
            self._check_fault()
            if not self.world.torque_enabled:
                self.driver.set_torque(True)
                self.driver.command(self._goal)

    @property
    def torque_enabled(self) -> bool:
        with self._lock:
            return self.world.torque_enabled

    def goal_positions(self) -> dict[str, float]:
        """The Goal_Position register, suite units."""

        with self._lock:
            return dict(zip(JOINT_NAMES, (float(value) for value in self._goal), strict=True))

    def expert_positions(self) -> dict[str, float]:
        """The expert's latest target, what the simulated leader arm reports."""

        with self._lock:
            self._check_fault()
            if self._expert_error is not None:
                raise RuntimeError(f"sim expert leader failed: {self._expert_error}")
            if self._expert_action is None:
                raise RuntimeError("sim expert leader has no target; is a leader connected?")
            action = self._expert_action.copy()
        return dict(zip(JOINT_NAMES, (float(value) for value in action), strict=True))

    def observation(self) -> dict[str, np.ndarray]:
        """The pinned policy observation: ``observation.state`` and ``.environment_state``."""

        with self._lock:
            self._check_fault()
            return self._observe()

    def render(
        self, camera: str, width: int, height: int, *, timeout: float = FRAME_TIMEOUT_S
    ) -> np.ndarray:
        """HxWx3 uint8 RGB of ``camera`` now (snapshot under the lock, render outside it)."""

        if camera not in CAMERAS:
            raise ValueError(f"unknown sim camera {camera!r}; available: {', '.join(CAMERAS)}")
        renderer = self.renderer
        with self._lock:
            self._check_fault()
            snapshot = self.world.snapshot()
        return renderer.render(snapshot, camera, width, height, timeout=timeout)

    # -- episodes --------------------------------------------------------------------------
    def on_episode_start(self, seed: int | None = None) -> dict[str, Any]:
        """Reseed the task and put the arm at the task's start pose. Returns the episode info.

        The caller must re-seed the follower's SafetyGuard from the new measured pose.
        """

        with self._lock:
            self._check_fault()
            seed = self.base_seed + self._episodes if seed is None else int(seed)
            torque = self.world.torque_enabled
            info = self.task.reset(self.world, seed)  # re-enables actuation
            self.driver.sync()
            self._goal = self.driver.goal_array()
            if not torque:
                self.driver.set_torque(False)
            self._phase = 0
            self._episodes += 1
            self._episode_seed = seed
            self._last_metrics = {}
            if self._leader_count() > 0:
                self._restart_expert()
            return _plain(
                {**info, "episode": self._episodes, "torque_enabled": self.world.torque_enabled}
            )

    def on_episode_end(self) -> dict[str, Any]:
        """``{"success": bool, "metrics": {...}}`` for the current episode (JSON-safe)."""

        with self._lock:
            return self._episode_end_locked()

    def recording_snapshot(
        self,
    ) -> tuple[np.ndarray, dict[str, Any] | None, Exception | None]:
        """Environment state and predicate grade from one frozen runtime state.

        The real-time physics thread uses the same lock. A recording loop keeps
        this result with the frame it selected, so a later save never grades a
        world that continued advancing during camera encoding or filesystem I/O.
        A predicate failure is returned beside the still-valid environment
        state; the frame can be saved and reported as recoverably unlabeled.
        """

        with self._lock:
            self._check_fault()
            environment_state = self.task.env_state(self.world).copy()
            error: Exception | None
            try:
                grade = self._episode_end_locked()
            except Exception as exc:
                grade, error = None, exc
            else:
                error = None
            return environment_state, grade, error

    def _episode_end_locked(self) -> dict[str, Any]:
        task = self.task
        metrics = {
            "task": task.name,
            "instruction": task.instruction,
            "seed": self._episode_seed,
            "episode": self._episodes,
            "sim_time_s": round(self.world.time_s, 4),
            "success_at_end": task.success(),
            **task.summary(),
            "final": {key: round(float(value), 6) for key, value in self._last_metrics.items()},
            "fault": self._fault,
        }
        success = task.first_success_s is not None and self._fault is None
        return {"success": bool(success), "metrics": _plain(metrics)}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task": self.task.name,
                "realtime": self.realtime,
                "running": self.running,
                "servo_mode": self.driver.mode,
                "timestep_s": self.world.timestep_s,
                "control_hz": CONTROL_HZ,
                "sim_time_s": round(self.world.time_s, 6),
                "physics_steps": self._steps,
                "dropped_steps": self._dropped,
                "torque_enabled": self.world.torque_enabled,
                "episode": self._episodes,
                "seed": self._episode_seed,
                "devices": dict(Counter(self._devices.values())),
                "expert_active": self._expert is not None and self._leader_count() > 0,
                "expert_error": self._expert_error,
                "fault": self._fault,
            }


class _SimBus:
    """Connection bookkeeping shared by the follower and leader buses."""

    kind = "follower"

    def __init__(self, name: str, runtime: SimRuntime | None = None) -> None:
        self.name = name
        self._bound = runtime
        self._runtime: SimRuntime | None = None
        self._detached_holding = False
        self._state = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._runtime is not None

    @property
    def runtime(self) -> SimRuntime:
        """The runtime this bus uses (the singleton unless one was given)."""

        return self._runtime or self._bound or get_sim_runtime()

    def connect(self) -> None:
        with self._state:
            if self._runtime is not None:
                return
            runtime = self._bound or get_sim_runtime()
            runtime.attach(self, self.kind)
            try:
                runtime.start()
            except BaseException:
                runtime.detach(self)
                raise
            self._runtime = runtime
            self._detached_holding = False
            log.info("sim %s bus %r connected", self.kind, self.name)

    def _require(self) -> SimRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("bus not connected")
        return runtime


class MujocoBus(_SimBus):
    """The simulated follower arm: a ``MotorBus`` over the shared runtime.

    Several instances may connect at once. They all drive the same arm, as two
    handles on one serial bus would.
    """

    kind = "follower"

    def __init__(self, name: str = "follower", runtime: SimRuntime | None = None) -> None:
        super().__init__(name, runtime)

    def disconnect(self) -> None:
        with self._state:
            runtime, self._runtime = self._runtime, None
            self._detached_holding = False
        if runtime is None:
            return
        try:
            runtime.set_torque(False)  # LeRobot disables torque on disconnect
        finally:
            runtime.detach(self)

    def disconnect_holding(self) -> None:
        with self._state:
            runtime, self._runtime = self._runtime, None
            if runtime is None or not runtime.torque_enabled:
                self._detached_holding = False
                raise RuntimeError("sim follower is not confirmed holding")
            self._detached_holding = True
        runtime.detach(self)

    def read_positions(self) -> dict[str, float]:
        return self._require().read_positions()

    def write_positions(self, goal: dict[str, float]) -> dict[str, float]:
        return self._require().write_goal(goal)

    def set_torque(self, enabled: bool) -> None:
        self._require().set_torque(enabled)

    def environment_state(self) -> np.ndarray:
        """``observation.environment_state`` now: the task's object state, float32.

        Its layout is the task's ``env_state_names``. ``SO101Follower`` hands
        it to policies and recordings; a hardware bus has no such state.
        """

        return np.asarray(self._require().observation()[ENV_STATE_KEY], dtype=np.float32)

    # C2 three-state torque truth for the control plane (stops wave 2). A
    # simulated actuator always answers, so the state is never "unconfirmed".
    @property
    def torque_enabled(self) -> bool:
        """Actuation as the shared world has it; False while this bus is disconnected."""

        runtime = self._runtime
        return bool((runtime is not None and runtime.torque_enabled) or self._detached_holding)

    @property
    def torque_state(self) -> str:
        return "on" if self.torque_enabled else "off"


class SimExpertLeader(_SimBus):
    """The simulated leader arm: reports the scripted expert's targets.

    The expert observes the shared world once per control period and plans
    from whatever state it finds, so it can take over at any pose.
    """

    kind = "leader"

    def __init__(self, name: str = "leader", runtime: SimRuntime | None = None) -> None:
        super().__init__(name, runtime)
        self._torque = False

    def disconnect(self) -> None:
        with self._state:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            runtime.detach(self)

    def disconnect_holding(self) -> None:
        self.disconnect()

    def read_positions(self) -> dict[str, float]:
        return self._require().expert_positions()

    def write_positions(self, goal: dict[str, float]) -> dict[str, float]:
        """Leader feedback: validated and ignored, like ``send_feedback`` on a passive arm."""

        self._require()
        ordered = {joint: float(goal[joint]) for joint in JOINT_NAMES if joint in goal}
        if any(not math.isfinite(value) for value in ordered.values()):
            raise ValueError("goal contains a non-finite motor position")
        return ordered

    def set_torque(self, enabled: bool) -> None:
        self._require()
        self._torque = bool(enabled)

    @property
    def torque_state(self) -> str:
        """The expert leader has no actuators: torque is always off (stops wave 2)."""

        return "off"


class SimCamera:
    """One simulated camera, rendered from the shared runtime's world."""

    def __init__(
        self,
        camera_id: str,
        width: int | None = None,
        height: int | None = None,
        *,
        runtime: SimRuntime | None = None,
        timeout_s: float = FRAME_TIMEOUT_S,
    ) -> None:
        if camera_id not in CAMERAS:
            raise ValueError(f"unknown sim camera {camera_id!r}; available: {', '.join(CAMERAS)}")
        default_width, default_height = get_settings().sim_image_wh
        self.cam_id = camera_id
        self.width, self.height = check_size(
            default_width if width is None else width,
            default_height if height is None else height,
        )
        self.timeout_s = float(timeout_s)
        runtime = runtime or get_sim_runtime()
        runtime.attach(self, "camera")
        try:
            runtime.start()
        except BaseException:
            runtime.detach(self)
            raise
        self._runtime = runtime
        self._released = False

    def read(self) -> np.ndarray:
        if self._released:
            raise RuntimeError(f"sim camera {self.cam_id!r} was released")
        try:
            return self._runtime.render(
                self.cam_id, self.width, self.height, timeout=self.timeout_s
            )
        except Exception as exc:
            raise CameraUnavailableError(
                f"sim camera {self.cam_id!r} failed to render: {type(exc).__name__}: {exc}"
            ) from exc

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._runtime.detach(self)


# -- factories and the process singleton ------------------------------------------------------
def make_sim_bus(name: str, *, runtime: SimRuntime | None = None) -> MotorBus:
    """The sim bus for ``name``. Like ``FeetechBus``, a name containing "leader" is the leader."""

    if "leader" in name.lower():
        return SimExpertLeader(name=name, runtime=runtime)
    return MujocoBus(name=name, runtime=runtime)


def make_sim_camera(
    camera_id: str,
    *,
    runtime: SimRuntime | None = None,
    width: int | None = None,
    height: int | None = None,
) -> SimCamera:
    """Open a simulated camera. Size defaults to ``SO101_SIM_IMAGE_SIZE``.

    Raises :class:`~so101.hardware.cameras.CameraUnavailableError` for an
    unknown camera or when this process cannot render. Nothing substitutes
    synthetic frames.
    """

    if camera_id not in CAMERAS:
        raise CameraUnavailableError(
            f"unknown sim camera {camera_id!r}; available: {', '.join(CAMERAS)}"
        )
    default_width, default_height = get_settings().sim_image_wh
    try:
        width, height = check_size(
            default_width if width is None else width,
            default_height if height is None else height,
        )
    except ValueError as exc:
        raise CameraUnavailableError(f"sim camera {camera_id!r}: {exc}") from exc
    runtime = runtime or get_sim_runtime()
    renderer = runtime.renderer
    if not renderer.start(FRAME_TIMEOUT_S):  # waits for the GL probe no longer than a frame
        raise CameraUnavailableError(
            f"sim camera {camera_id!r} cannot render here: {renderer.error or 'no OpenGL context'}"
        )
    return SimCamera(camera_id, width, height, runtime=runtime)


_runtime: SimRuntime | None = None
_runtime_lock = threading.Lock()


def get_sim_runtime() -> SimRuntime:
    """The process-wide runtime, built from the settings on first use."""

    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = SimRuntime()
        return _runtime


def stop_sim_runtime() -> None:
    """Stop the physics thread, if a runtime exists. Never creates one."""

    with _runtime_lock:
        runtime = _runtime
    if runtime is not None:
        runtime.stop()


def reset_sim_runtime() -> None:
    """Stop and drop the singleton; the next use builds a fresh one (settings re-read)."""

    global _runtime
    with _runtime_lock:
        runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.stop()


def on_episode_start(seed: int | None = None) -> dict[str, Any]:
    """:meth:`SimRuntime.on_episode_start` on the singleton."""

    return get_sim_runtime().on_episode_start(seed)


def on_episode_end() -> dict[str, Any]:
    """:meth:`SimRuntime.on_episode_end` on the singleton."""

    return get_sim_runtime().on_episode_end()


def recording_snapshot() -> tuple[np.ndarray, dict[str, Any] | None, Exception | None]:
    """:meth:`SimRuntime.recording_snapshot` on the singleton."""

    return get_sim_runtime().recording_snapshot()


__all__ = [
    "CONTROL_DT_S",
    "DEVICE_KINDS",
    "MAX_CATCHUP_S",
    "PHYSICS_WAKE_S",
    "MujocoBus",
    "SimCamera",
    "SimExpertLeader",
    "SimRuntime",
    "get_sim_runtime",
    "make_sim_bus",
    "make_sim_camera",
    "on_episode_end",
    "on_episode_start",
    "recording_snapshot",
    "reset_sim_runtime",
    "stop_sim_runtime",
]

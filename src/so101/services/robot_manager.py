"""Single-owner robot runtime and fail-closed control plane.

All hardware access in the web service converges here. Legacy leader teleop,
recording, policy rollout, and external command sources are mutually exclusive.
External sources additionally require a short-lived lease, explicit arming,
monotonic sequence numbers, a deadman signal, and fresh timestamps.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

from .. import JOINT_NAMES
from ..atomic_io import atomic_write_json
from ..bringup import ownership as bus_ownership
from ..bringup.buses import BusDiagnosticOpenError
from ..config import COLLAPSE_TIMEOUT_MARGIN_S, COLLAPSE_TIMEOUT_MAX_S, get_settings
from ..data.inspection import SIM_OUTCOME_SOURCE as _SIM_OUTCOME_SOURCE
from ..data.inspection import (
    LabelNotStoredError,
    persist_sim_outcome,
)
from ..data.recorder import EpisodeRecorder, RecordingConfig
from ..hardware.backend import current_backend
from ..hardware.calibration import (
    SIM_CALIBRATION_REFUSED,
    CalibrationResult,
    affected_motors,
    calibrate,
)
from ..hardware.cameras import CameraManager, configured_streams
from ..hardware.follower import SO101Follower
from ..hardware.health import (
    UNIT_PROVENANCE,
    HealthSnapshot,
    HealthThresholds,
    ServoHealthMonitor,
    ServoHealthSample,
    worst_level,
)
from ..hardware.leader import SO101Leader
from ..hardware.motors import (
    FeetechBus,
    TorqueOffUnconfirmed,
    TorqueState,
    bus_torque_state,
    release_port,
    torque_off_all,
)
from ..hardware.rest_pose import resolve_collapse_rest_pose
from ..hardware.safety import SafetyGuard, SafetyViolation
from ..logging import get_logger
from ..policy.runner import _build_observation
from ..policy.server import load_policy
from ..teleop.controller import TeleopController
from .control_trace import ControlTrace
from .store import (
    LATCH_MARKER_KIND as _LATCH_MARKER_KIND,
)
from .store import (
    LATCH_MARKER_SCHEMA_VERSION,
    ControlStore,
    SafetyLatchIdentity,
    combine_identity_latch_markers,
    get_store,
    latch_marker_path,
    read_identity_latch_markers,
    safety_latch_identity,
)

log = get_logger(__name__)
# Public compatibility name shared with ``so101.cli_sim`` and pinned by tests.
SIM_OUTCOME_SOURCE = _SIM_OUTCOME_SOURCE


class Mode(str, Enum):
    IDLE = "idle"
    TELEOP = "teleop"
    RECORDING = "recording"
    ROLLOUT = "rollout"
    MANUAL = "manual"


class ControlError(RuntimeError):
    """Base class for an external control request rejection."""


class ControlConflict(ControlError):
    """The requested resource is already owned or in another state."""


class ControlRejected(ControlError):
    """The request violates a safety or freshness invariant."""


class ControlBusy(ControlRejected):
    """The motor bus did not come free in time, so nothing was sent.

    Routers answer 503 (safety-B9, docs/audit/v0.1/safety.md). It subclasses
    ``ControlRejected`` so every existing error path, including owner-scoped
    cleanup, still handles it.
    """


class BusDiagnosticRefused(ControlConflict):
    """The manager will not lend a motor bus for diagnostics now (HTTP 409).

    :meth:`RobotManager.run_idle_bus_diagnostic` raises it unless the robot is
    idle: not armed or arming, no Halt holding, no Collapse, no loop, no
    calibration and no E-stop. It also refuses a bus it cannot read registers
    from, and an E-stop that lands during a diagnostic ends it with this.
    """


class CalibrationTorqueOffUnconfirmed(RuntimeError):
    """Calibration ended, but the torque-off after it was not confirmed (C2).

    The manager has latched a fault and ``torque_unconfirmed``. ``saved`` says
    whether the new calibration reached disk before that.
    """

    def __init__(self, message: str, *, saved: bool) -> None:
        super().__init__(message)
        self.saved = saved


_T = TypeVar("_T")
DiagnosticTarget = Literal["follower", "leader"]
_DIAGNOSTIC_TARGETS: tuple[DiagnosticTarget, ...] = ("follower", "leader")


class _LentBus:
    """A register bus lent for diagnostics: reads only, preempted by an E-stop.

    It exposes ``motors``, ``is_connected``, ``ping``, ``read`` and
    ``sync_read`` (the reads ``so101.bringup.buses.collect_bus_facts`` uses)
    and an in-memory ``synthetic_calibration`` when the bus has one. Every
    other attribute, including every write and torque method, raises
    ``AttributeError``: a diagnostic cannot write a register. ``check`` runs
    before every transfer and raises once an E-stop has latched, so the
    diagnostic does no bus I/O after it.
    """

    def __init__(self, bus: Any, check: Callable[[], None]) -> None:
        self._bus = bus
        self._check = check

    @property
    def motors(self) -> Any:
        return self._bus.motors

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._bus, "is_connected", True))

    def ping(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return self._bus.ping(*args, **kwargs)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return self._bus.read(*args, **kwargs)

    def sync_read(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return self._bus.sync_read(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "synthetic_calibration" and hasattr(self._bus, name):
            return getattr(self._bus, name)  # in-memory data of the rehearsal bus
        raise AttributeError(
            f"a lent diagnostic bus is read-only: {name!r} is not available "
            "(only ping, read and sync_read)"
        )


def _open_diagnostic_bus(port: str) -> Any:
    """The real backend's diagnostic opener: a bare LeRobot bus, read-only.

    ``connect(handshake=False)`` and a host-side baud rate; no packet goes to
    a motor until the diagnostic reads (``so101.bringup.buses``).
    """

    from ..bringup.buses import open_feetech_bus
    from ..bringup.lerobot_surface import DEFAULT_BAUDRATE

    return open_feetech_bus(port, baudrate=DEFAULT_BAUDRATE)


class StopKind(str, Enum):
    """Stop categories (docs/plans/2026-10-01-suite-v02-design.md §5).

    - ``ESTOP``: category 0. Torque off on every bus at once, then latch.
    - ``HALT``: category 2. Freeze in place: motors stay on and hold the
      present position, under a hold timer that ends in a Collapse.
    - ``COLLAPSE``: category 1. A bounded, speed-capped descent to the rest
      pose, then torque off with read-back.

    docs/control-plane-safety.md describes all three as built.
    """

    ESTOP = "estop"
    HALT = "halt"
    COLLAPSE = "collapse"


# A latch write that fails fast is retried briefly: on Windows an antivirus
# scanner can hold a freshly written database file for a moment. A busy or
# locked database is not retried; SQLite already waited its busy_timeout.
_LATCH_WRITE_RETRY_DELAYS_S = (0.05, 0.1)


# The fallback for a latch the store refused (safety-B8): an identity-bearing
# marker next to the store. The historical unscoped path is reserved for the
# real backend, so a mock/sim rehearsal can never replace or remove it.
# Whether a restart comes back at least as latched as this process is now:
# "saved" (the store has the newest latching snapshot), "marker" (the store
# refused it, the fallback marker makes a restart come back faulted) or
# "unsaved" (neither could be written: the latch lives only in memory).
LatchPersistence = Literal["saved", "marker", "unsaved"]


@dataclass(frozen=True)
class _LatchMarker:
    """A fallback marker found on disk. ``latches`` is None when it is unreadable."""

    path: Path
    version: int | None
    latches: dict[str, Any] | None
    error: str | None  # why the store refused the snapshot, or why the marker is unreadable
    identity: SafetyLatchIdentity | None = None


def _marker_identity(
    payload: dict[str, Any],
    *,
    expected: SafetyLatchIdentity | None,
    allow_legacy_real: bool,
) -> SafetyLatchIdentity | None:
    """Validate a v2 identity, or conservatively bind a legacy marker to real."""

    raw_schema = payload.get("schema_version")
    raw_identity = payload.get("identity")
    if raw_schema is None and raw_identity is None:
        if not allow_legacy_real or (expected is not None and expected.backend != "real"):
            raise ValueError("legacy unscoped marker is not valid for this backend")
        return expected
    if raw_schema != LATCH_MARKER_SCHEMA_VERSION or not isinstance(raw_identity, dict):
        raise ValueError("unsupported latch marker schema or identity")
    identity = SafetyLatchIdentity(
        str(raw_identity.get("backend") or ""),
        str(raw_identity.get("follower_id") or ""),
        str(raw_identity.get("leader_id") or ""),
    )
    if expected is not None and identity != expected:
        raise ValueError(
            "latch marker identity mismatch: "
            f"stored {identity.as_dict()}, expected {expected.as_dict()}"
        )
    return identity


def _latches_latch_something(latches: dict[str, Any]) -> bool:
    return bool(
        latches.get("estopped")
        or latches.get("fault")
        or latches.get("torque_unconfirmed")
        or latches.get("energized")
    )


def _read_latch_marker(
    path: Path,
    *,
    expected_identity: SafetyLatchIdentity | None = None,
    allow_legacy_real: bool = False,
) -> _LatchMarker | None:
    """The marker at ``path``, None when there is none. Never raises."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:  # there, but unreadable: fail closed
        return _LatchMarker(path, None, None, f"{type(exc).__name__}: {exc}")
    try:
        payload = json.loads(raw.decode("utf-8"))
        version = payload["version"]
        latches = payload["latches"]
        if (
            payload.get("kind") != _LATCH_MARKER_KIND
            or isinstance(version, bool)
            or not isinstance(version, int)
            or not isinstance(latches, dict)
        ):
            raise ValueError("not a latch marker")
        marker_identity = _marker_identity(
            payload,
            expected=expected_identity,
            allow_legacy_real=allow_legacy_real,
        )
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        return _LatchMarker(path, None, None, f"unreadable marker: {exc}")
    error = payload.get("error")
    return _LatchMarker(
        path,
        version,
        latches,
        error if isinstance(error, str) else None,
        marker_identity,
    )


def _transient_store_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, sqlite3.OperationalError):
        return getattr(exc, "sqlite_errorname", None) not in {"SQLITE_BUSY", "SQLITE_LOCKED"}
    return False


def lease_tag(lease_id: str) -> str:
    """A short, non-reversible name for a lease that is safe to publish.

    The lease id is the only credential for arm, observe and command, so
    events carry this tag instead (C4, safety-B5 in docs/audit/v0.1/safety.md).
    It is the first 12 hex digits of SHA-256 over ``"so101-lease:" + id``; the
    owner, who knows the id, can compute it to find its own events.
    """

    return hashlib.sha256(f"so101-lease:{lease_id}".encode()).hexdigest()[:12]


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (param.name == name and param.kind is not inspect.Parameter.POSITIONAL_ONLY)
        or param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters
    )


# Kinds whose real implementation sits behind the dispatcher. Status
# advertises exactly these (C11 item 5).
IMPLEMENTED_STOP_KINDS: tuple[StopKind, ...] = (StopKind.ESTOP, StopKind.HALT, StopKind.COLLAPSE)

StopState = Literal["none", "halted", "collapsing"]

# Halt telemetry: while holding, the watchdog samples the follower this often
# (manager clock) so the UI and the HIL drift probe see live positions.
_HOLD_SAMPLE_PERIOD_S = 0.1
# Collapse progress rules (design doc §5): a stall is no progress for this
# long with more than _COLLAPSE_STALL_ERROR_DEG still to go. Within that error
# the descent counts as settled when it stops moving; within
# _COLLAPSE_ARRIVED_DEG it has arrived.
_COLLAPSE_STALL_WINDOW_S = 0.5
_COLLAPSE_STALL_ERROR_DEG = 3.0
_COLLAPSE_ARRIVED_DEG = 1.0
_COLLAPSE_PROGRESS_EPS_DEG = 0.1
# The watchdog's backstop for a wedged descent thread: this long past the
# planned bound, it cuts torque itself.
_COLLAPSE_POLICE_MARGIN_S = 1.0
# Servo health (docs/bring-up.md "Servo health monitor"). The so101-health
# thread wakes this often (like the watchdog's pass) and polls an arm when its
# period, 1 / health_poll_hz on the manager clock, is due.
_HEALTH_TICK_S = 0.05
# A poll skipped because the bus stayed busy is retried this soon (manager
# clock) rather than a whole period later.
_HEALTH_RETRY_S = 0.2
# A sample older than this many poll periods is flagged stale.
_HEALTH_STALE_PERIODS = 3.0
# Status and telemetry carry at most this many warn/critical reasons.
_HEALTH_SUMMARY_REASONS = 3
HealthArmName = Literal["follower", "leader"]
_HEALTH_ARMS: tuple[HealthArmName, ...] = ("follower", "leader")


# Safety states that only their own reset path clears (C5). No stop overwrites
# a latched state; an owner-scoped stop also leaves a tripped state visible.
class _Pacer:
    """Deadline scheduling for a fixed-rate loop (data-B4, docs/audit/v0.1/data.md).

    Call :meth:`wait` at the end of every iteration. An iteration then takes
    max(work, period) rather than work + period, so a 30 fps recording really
    holds 30 frames per second of motion. A late iteration is caught up by
    the next ones; a loop more than one period behind (a stall) resyncs
    instead of bursting frames back to back.
    """

    def __init__(
        self,
        period: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._deadline = clock() + period

    def wait(self) -> dict[str, float | bool]:
        now = self._clock()
        delay = self._deadline - now
        if delay < -self.period:
            self._deadline = now + self.period
            return {
                "requested_sleep_seconds": 0.0,
                "deadline_lateness_seconds": -delay,
                "resynchronized": True,
            }
        if delay > 0:
            self._sleep(delay)
        lateness = max(0.0, self._clock() - self._deadline)
        self._deadline += self.period
        return {
            "requested_sleep_seconds": max(0.0, delay),
            "deadline_lateness_seconds": lateness,
            "resynchronized": False,
        }


_LOOP_MODES = frozenset({Mode.TELEOP, Mode.RECORDING, Mode.ROLLOUT})
# C6: a sample older than this is not live, whatever produced it.
_LIVE_SAMPLE_MAX_AGE_S = 0.5

_LATCHED_STATES = frozenset({"estopped", "fault"})
_TRIPPED_STATES = frozenset({"watchdog_timeout", "lease_expired", "operator_timeout"})


@dataclass(frozen=True)
class StopResult:
    """What one :meth:`RobotManager.request_stop` call did.

    ``applied`` is False when the request named a lease or authority epoch that
    is no longer current; nothing changed then. ``torque_off`` is the confirmed
    torque truth afterwards. ``loop_stopped`` is False when a legacy loop thread
    did not exit in time, which latches a fault.

    ``holding`` is True when a HALT holds position with torque on, and
    ``collapsing`` when a COLLAPSE descent is running (it ends with its own
    torque-off). ``fallback`` names why a HALT or COLLAPSE fell back to an
    immediate torque cut; a fault is latched then.
    """

    kind: StopKind
    reason: str
    applied: bool = True
    loop_stopped: bool = True
    torque_off: bool = True
    holding: bool = False
    collapsing: bool = False
    fallback: str | None = None

    @property
    def safe(self) -> bool:
        """The arm is accounted for: holding, descending, or confirmed off."""
        return self.holding or self.collapsing or self.torque_off

    @property
    def ok(self) -> bool:
        return self.applied and self.loop_stopped and self.fallback is None and self.safe


@dataclass
class _Halt:
    """An active Halt (category 2). Times are the manager clock unless ``_wall``."""

    since: float
    since_wall: float
    reason: str
    deadline: float  # hold expiry while an operator supervises
    renewals: int = 0
    last_sample: float = float("-inf")


@dataclass
class _CollapsePlan:
    """A planned descent: present -> rest in joint space, capped per joint."""

    start: dict[str, float]
    target: dict[str, float]  # arm joints only; the gripper follows the grip rule
    gripper_goal: float | None
    distance: float  # the largest arm-joint distance, degrees
    duration: float  # distance / speed: the slowest joint at the cap
    timeout: float
    rest_source: str
    rest_warnings: list[str]


@dataclass
class _Collapse:
    """An active Collapse (category 1)."""

    since: float
    since_wall: float
    reason: str
    fault: str | None = None  # the fault that triggered it; it stays latched
    cancel: threading.Event = field(default_factory=threading.Event)
    ended: bool = False  # claimed by whoever finishes it (done or fallback)
    plan: _CollapsePlan | None = None
    started: float | None = None  # descent start, manager clock
    deadline: float | None = None
    progress: float = 0.0
    eta_s: float = 0.0  # 0 until the plan is made (milliseconds)
    thread: threading.Thread | None = None


@dataclass
class _HealthArm:
    """Servo-health polling state for one arm. Guarded by the manager's ``_lock``.

    ``monitor`` exists only while the arm is polled; each activation (the
    manager opening that bus) starts a fresh one, so a read-failure streak
    never carries over from an earlier session. ``latest`` is the last poll's
    verdict, ``last_good`` the last poll that read every servo. Times ending
    in ``_mono`` are the manager clock.
    """

    arm: HealthArmName
    thresholds: HealthThresholds | None
    config_error: str | None = None
    monitor: ServoHealthMonitor | None = None
    active: bool = False
    inactive_reason: str | None = "not polled yet"
    next_due: float = float("-inf")
    pending: dict[str, ServoHealthSample] | BaseException | None = None
    latest: HealthSnapshot | None = None
    last_good: HealthSnapshot | None = None
    last_good_mono: float | None = None
    polled_mono: float | None = None
    polls: int = 0
    read_failures: int = 0
    skipped_busy: int = 0
    last_skip: str | None = None
    last_read_ms: float | None = None
    max_read_ms: float = 0.0
    activations: int = 0
    critical_dispatches: int = 0


@dataclass
class ControlLease:
    id: str
    source: str
    ttl_seconds: float
    expires_at: datetime
    deadline: float
    last_sequence: int = -1

    @property
    def tag(self) -> str:
        return lease_tag(self.id)

    def public(self) -> dict:
        """What any reader may see: no id (C4, safety-B5)."""

        return {
            "source": self.source,
            "expires_at": self.expires_at,
            "ttl_seconds": self.ttl_seconds,
            "server_time": datetime.now(timezone.utc),
        }

    def owner_view(self) -> dict:
        """The public view plus the id, only for the create and heartbeat answers."""

        return {"id": self.id, **self.public()}


class RobotManager:
    # Longest a request handler waits for the motor bus before it answers 503
    # (safety-B9, docs/audit/v0.1/safety.md). Loops and teardown keep their own
    # bounds; the E-stop never waits on a request at all.
    request_io_timeout_s: float = 0.25
    # I/O bound for a stop's own bus transfers (the halt's read and hold
    # write, each collapse tick). Past it a stop falls back to torque-off with
    # a fault (design doc §5).
    stop_io_timeout_s: float = 0.5
    # Collapse goal rate. Lockstep sim (SO101_SIM_REALTIME=0) simulates 1/30 s
    # per torque-on write, so the descent runs at the sim's control rate.
    collapse_rate_hz: float = 30.0
    # Longest a servo-health poll waits for the motor bus. A control loop holds
    # the bus only for its own short transfers each period, so the poll slips
    # in between them; when the bus stays busy longer, the poll is skipped and
    # counted (docs/bring-up.md "Servo health monitor").
    health_io_wait_s: float = 0.05

    def __init__(self, *, store: ControlStore | None = None) -> None:
        self.settings = get_settings()
        self.store = store or get_store()
        self._latch_identity = safety_latch_identity(self.settings)
        store_identity = getattr(self.store, "latch_identity", self._latch_identity)
        if store_identity != self._latch_identity:
            raise RuntimeError(
                "control store safety identity does not match RobotManager: "
                f"{store_identity.as_dict()} != {self._latch_identity.as_dict()}"
            )
        self.mode = Mode.IDLE

        # Constructing the wrapper does not open hardware, but it does load the
        # saved calibration-derived safety envelope. Keep that exact guard as
        # the shared authority for every command path.
        self._follower: SO101Follower | None = SO101Follower()
        self.guard: SafetyGuard = self._follower.guard
        self._leader: SO101Leader | None = None
        self._cameras: CameraManager | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # The one monotonic clock for leases, operator heartbeats, the command
        # deadman, the watchdog, sample and frame ages, and loop pacing; the
        # sleep paces the loops. Tests inject both, so timers run on frozen
        # time that the test advances (devex-B4, docs/audit/v0.1/devex.md).
        self._clock: Callable[[], float] = time.monotonic
        self._sleep: Callable[[float], None] = time.sleep
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        # Held across one lease expiry (check and Halt), always before _io_lock
        # and _lock, so an owner-scoped stop can wait for an expiry in flight.
        self._expiry_lock = threading.Lock()
        self._fault: str | None = None
        # Monotonic identity of the currently latched runtime fault. A reset
        # snapshots this before any bus I/O and only clears the same fault;
        # a fault that lands while the reset is in flight always wins.
        self._fault_epoch = 0
        # Three-state torque truth: True after a torque-off that could not be
        # confirmed. Only a confirmed torque-off clears it.
        self._torque_unconfirmed = False
        # A standalone process wrote this before it could energize the arm and
        # exited without a confirmed torque-off. Recovery treats it exactly as
        # unknown torque until a read-back pass (or acknowledged power cut).
        self._energized_recovery = False
        self._energized_at: str | None = None
        self._energized_by: str | None = None
        # Devices whose off state a restart/recovery must actively confirm.
        # Pre-scope local rows are conservatively both arms; host-global v1
        # intent retains its exact follower/leader set when no local unknown
        # state broadens it.
        self._recovery_devices: set[str] = set()
        self._recovery_epoch = 0
        self._global_intent_lock = threading.Lock()
        self._global_intent_unsaved_devices: set[str] = set()
        # Set only when app lifespan, while holding the host-global device
        # locks, adopts a marker left by another process/data directory.
        self._global_energized_recovery = False
        # A reset compare-and-clears the E-stop generation it observed when
        # it began. The separate in-flight count also makes an E-stop that
        # started just before the reset (but is waiting for ``_lock``) win.
        # This lock protects only three integer operations and is never held
        # while taking ``_lock`` or touching hardware/storage.
        self._estop_generation_lock = threading.Lock()
        self._estop_generation = 0
        self._estop_inflight = 0
        self._authority_epoch = 0
        # Halt and Collapse (design doc §5). E-stop and fault keep their own
        # state; these describe an energized hold or a running descent.
        self._stop_state: StopState = "none"
        self._halt: _Halt | None = None
        self._collapse: _Collapse | None = None
        # A Halt/Collapse request reserves its transition before any join or
        # bus wait and releases it only after the result is final. New arm and
        # loop starts cannot slip between the stop's I/O and published state.
        # Multiple stop requests may overlap (E-stop never waits on these), so
        # each owns a token rather than a process-wide mutex.
        self._stop_transition_serial = 0
        self._stop_transitions: dict[int, tuple[StopKind, bool, bool]] = {}

        self._snapshot: dict = {
            "t": time.time(),
            "observation": {},
            "action": {},
            "fps": 0.0,
        }
        self._snapshot_monotonic = self._clock()
        self._sample_seq = 0
        self.control_trace = ControlTrace()
        self._frames: dict[str, np.ndarray] = {}
        self._frame_times: dict[str, float] = {}
        self._camera_preview_thread: threading.Thread | None = None
        self._camera_preview_stop = threading.Event()
        self._camera_preview_error: str | None = None
        self._camera_preview_devices: dict[str, Any] = {}

        self._recorder: EpisodeRecorder | None = None
        self._rec_cfg: RecordingConfig | None = None
        self._episode_active = False
        self._episode_saving = False
        self._episode_starting = False
        self._episode_frames = 0
        # A loop step selects its recorder and captures the sim grade while it
        # still owns _io_lock, then writes camera data outside that lock. Save
        # closes under _io_lock and waits here for every selected frame.
        self._episode_write_condition = threading.Condition(self._lock)
        self._episode_writes = 0
        self._episode_write_error: Exception | None = None
        self._episode_grade: dict[str, Any] | None = None
        self._episode_grade_error: Exception | None = None
        # Sim backend: bumped (under _io_lock) whenever begin_episode resets
        # the scene. A loop step taken before the reset is never recorded into
        # the episode that starts after it.
        self._scene_epoch = 0
        self._recorder_epoch = 0

        self.cal_in_progress = False
        self.cal_fraction = 0.0
        self.cal_message = ""
        self.cal_arm = ""
        self._calibration_attempt: dict[str, Any] = {}
        self._calibration_diagnostics: dict[str, Any] = {}
        self._calibration_live: dict[str, Any] | None = None
        self._calibration_live_at = 0.0
        self._calibration_cancel = threading.Event()
        self._calibration_thread: threading.Thread | None = None

        self.watchdog_ms = 500
        self.operator_timeout_ms = 2_000
        self._operator_heartbeats: dict[str, float] = {}
        self._operation_owner: str | None = None
        # C7: the checkpoint path the current rollout runs (eval trials too).
        self._active_model: str | None = None
        self._lease: ControlLease | None = None
        self._armed = False
        # Pose measured when the current armed session engaged torque, keyed by
        # its authority epoch (C9: a null tracking depth holds the elbow here).
        self._engaged_pose: tuple[int, dict[str, float]] | None = None
        self._safety_state = "disarmed"
        self._last_command_at: datetime | None = None
        self._last_command_monotonic: float | None = None
        self._last_command_audit_monotonic = 0.0
        self._commands_since_audit = 0
        self._watchdog_shutdown = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        # Servo health (docs/bring-up.md "Servo health monitor"): polled on
        # its own thread, so a slow or wedged health read never delays the
        # watchdog's supervision.
        self._health: dict[HealthArmName, _HealthArm] = {
            arm: self._new_health_arm(arm) for arm in _HEALTH_ARMS
        }
        self._health_shutdown = threading.Event()
        self._health_thread: threading.Thread | None = None
        # safety-B8: every latch snapshot written to the store carries a
        # version; the store keeps the highest. The floor is wall-clock
        # microseconds, so a process that could not read the stored version
        # still writes above an earlier process's snapshots.
        self._latch_version = time.time_ns() // 1_000
        # The fallback marker for a latch the store refused, and what status
        # reports about it (``latch_persistence``). Guarded by
        # ``_latch_file_lock``, which is never held while taking another lock.
        self._latch_file_lock = threading.Lock()
        self._latch_saved_version = 0  # the newest snapshot known to be in the store
        self._latch_unsaved_version: int | None = None  # the newest latching one that is not
        self._latch_marker_version: int | None = None  # what the marker on disk records
        self._latch_persistence: LatchPersistence = "saved"
        self._restore_latches()
        self._report_health_config_errors()
        self._ensure_watchdog()
        self._ensure_health_thread()

    # --- internal utilities ---------------------------------------------
    def _event(
        self,
        event_type: str,
        message: str,
        *,
        level: str = "info",
        source: str = "robot",
        detail: dict | None = None,
    ) -> None:
        try:
            self.store.append_event(event_type, message, level=level, source=source, detail=detail)
        except Exception as exc:  # audit failure must not bypass safety cleanup
            log.error("could not persist audit event: %s", exc)

    def _latched_state_locked(self) -> str:
        """The safety state a latch shows; the E-stop outranks a fault (C5)."""

        return "estopped" if self.guard.estopped else "fault"

    def _add_recovery_devices_locked(self, devices: set[str]) -> None:
        """Record new recovery truth and stale every cut that began earlier."""

        self._recovery_devices.update(devices)
        self._recovery_epoch += 1

    def _latch_fault_in_memory(
        self,
        exc: BaseException | str,
        *,
        expected_fault_epoch: int | None = None,
        torque_unconfirmed: bool = False,
    ) -> tuple[str, int]:
        """Latch a fault without touching disk or the event store.

        Runtime I/O failure paths use this while they still own the bus: the
        in-memory latch immediately rejects queued commands, then they release
        the bus and perform their category-0 torque cut before either SQLite
        operation can block.
        """

        message = str(exc)
        with self._lock:
            if expected_fault_epoch is not None and self._fault_epoch != expected_fault_epoch:
                raise ControlConflict(
                    "a newer runtime fault was latched during the fault reset; it was not cleared"
                )
            self._fault = message
            self._fault_epoch += 1
            if torque_unconfirmed:
                self._torque_unconfirmed = True
                self._add_recovery_devices_locked({"follower", "leader"})
            self._armed = False
            self._authority_epoch += 1
            self._safety_state = self._latched_state_locked()
            epoch = self._fault_epoch
        return message, epoch

    def _record_fault(self, message: str, *, source: str) -> None:
        """Persist and audit a fault that is already latched in memory."""

        self._retain_global_safety_latch(f"runtime fault retained: {message}")
        self._persist_latches()
        self._event("fault", message, level="error", source=source)

    def _retain_global_safety_latch(self, reason: str) -> None:
        """Keep a real E-stop/fault host-global until its deliberate reset.

        The energized journal is conservative here: torque may already be
        confirmed off, but another data directory must still adopt a blocking
        fault instead of bypassing this process's safety latch.
        """

        if current_backend(self.settings) != "real":
            return
        # One marker on the actuated follower is the host-global safety
        # sentinel for this configured arm pair. Keeping it to the follower
        # also lets a Collapse confirm and clear actual torque before the
        # fault marker is retained again at the end.
        devices = {"follower"}
        with self._lock:
            already_durable = (
                self._global_energized_recovery
                and devices <= self._recovery_devices
                and not (devices & self._global_intent_unsaved_devices)
            )
        if already_durable:
            return
        if not bus_ownership.device_lock_is_held(self.settings, device_name="follower"):
            self._global_intent_unsaved_devices.update(devices)
            log.critical(
                "real safety latch has no host-global device lock; it cannot be published: %s",
                reason,
            )
            return
        failure = self._write_global_recovery_intent(reason, devices)
        if failure is not None:
            log.critical("host-global safety latch could not be retained: %s", failure)

    def _set_fault(self, exc: BaseException | str, *, source: str = "runtime") -> None:
        message, _epoch = self._latch_fault_in_memory(exc)
        self._record_fault(message, source=source)

    def _fail_closed_runtime_io(
        self,
        exc: BaseException | str,
        *,
        source: str,
        cut_reason: str,
        include_leader: bool = False,
        torque_unconfirmed: bool = False,
    ) -> bool:
        """Latch, cut torque, then persist a runtime I/O failure.

        The order is safety-significant: storage and audit writes may wait on
        SQLite or the filesystem, while category-0 torque removal may not.
        """

        message, fault_epoch = self._latch_fault_in_memory(
            exc, torque_unconfirmed=torque_unconfirmed
        )
        if torque_unconfirmed:
            scope = {"follower", "leader"} if include_leader else {"follower"}
            global_error = self._write_global_recovery_intent(message, scope)
            if global_error is not None:
                message += (
                    "; host-global energized intent is unsaved "
                    f"({type(global_error).__name__}: {global_error})"
                )
                with self._lock:
                    if self._fault_epoch == fault_epoch:
                        self._fault = message
        torque_off = self._cut_torque(cut_reason, include_leader=include_leader)
        self._record_fault(message, source=source)
        return torque_off

    # --- latch persistence (safety-B8, docs/audit/v0.1/safety.md) --------
    def _persist_latches(self) -> None:
        """Write the safety latches through to SQLite. Never raises.

        The snapshot and its version are taken together under ``_lock``. The
        write happens outside it, so a slow disk never blocks a status read or
        a stop. The store keeps the highest version, so a writer that was
        overtaken cannot roll a newer snapshot back. Callers on the E-stop path
        persist after the torque cut, never before it.

        When the store refuses a snapshot that latches something (E-stop,
        fault or torque unconfirmed), a fallback marker next to the store makes
        the next start come back faulted; a later snapshot that reaches the
        store removes it (``_latches_unsaved``, ``_latches_saved``).
        """

        with self._lock:
            self._latch_version += 1
            version = self._latch_version
            latches = {
                "estopped": self.guard.estopped,
                "fault": self._fault,
                "safety_state": self._safety_state,
                "torque_unconfirmed": self._torque_unconfirmed,
                "energized": self._energized_recovery,
                "energized_at": self._energized_at,
                "energized_by": self._energized_by,
            }
        failure: Exception | None = None
        for delay in (*_LATCH_WRITE_RETRY_DELAYS_S, None):
            try:
                self.store.save_latches(latches, version=version)
            except Exception as exc:  # the in-memory latch still holds
                if delay is None or not _transient_store_error(exc):
                    log.error("could not persist the safety latches (version %d): %s", version, exc)
                    failure = exc
                    break
                log.warning("safety latch write failed (%s); retrying in %.2f s", exc, delay)
                time.sleep(delay)
                continue
            self._latches_saved(version)
            return
        if failure is not None and _latches_latch_something(latches):
            self._latches_unsaved(latches, version, failure)

    def _latch_marker_path(self) -> Path:
        store_path = getattr(self.store, "path", None)
        if store_path is None:
            store_path = Path(self.settings.data_dir) / "control_plane.sqlite3"
        return latch_marker_path(Path(store_path), self._latch_identity)

    def _latches_saved(self, version: int) -> None:
        """A snapshot reached the store: a marker for an older one is obsolete."""

        with self._latch_file_lock:
            self._latch_saved_version = max(self._latch_saved_version, version)
            unsaved = self._latch_unsaved_version
            if unsaved is not None and version < unsaved:
                return  # an older writer finished late; the newer latch is still unsaved
            self._latch_unsaved_version = None
            self._latch_persistence = "saved"
            marker = self._latch_marker_version
            if marker is None or marker >= version:
                return
            path = self._latch_marker_path()
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:  # fail closed: the next start comes back faulted
                log.warning("could not remove the latch marker %s: %s", path, exc)
                return
            self._latch_marker_version = None
        log.info("safety latches saved (version %d); removed the fallback marker", version)

    def _latches_unsaved(self, latches: dict[str, Any], version: int, failure: Exception) -> None:
        """The store refused a latching snapshot: write the fallback marker.

        The marker makes the next start come back faulted. If even the marker
        cannot be written, the latch lives only in memory: status reports
        ``latch_persistence: "unsaved"``. Both emit a critical event (which
        may not reach a store that is failing; the log always has it).
        """

        path = self._latch_marker_path()
        marker_error: str | None = None
        with self._latch_file_lock:
            if self._latch_saved_version > version:
                return  # a newer snapshot already reached the store
            self._latch_unsaved_version = max(self._latch_unsaved_version or 0, version)
            marker = self._latch_marker_version
            if marker is None or marker < version:
                payload = {
                    "kind": _LATCH_MARKER_KIND,
                    "schema_version": LATCH_MARKER_SCHEMA_VERSION,
                    "identity": self._latch_identity.as_dict(),
                    "version": version,
                    "latches": dict(latches),
                    "error": f"{type(failure).__name__}: {failure}",
                    "written_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    atomic_write_json(path, payload)
                    self._latch_marker_version = version
                except Exception as exc:  # the in-memory latch still holds
                    marker_error = f"{type(exc).__name__}: {exc}"
            written = self._latch_marker_version is not None
            self._latch_persistence = "marker" if written else "unsaved"
        saved = f"the safety latches (version {version}) could not be saved ({failure})"
        if written and marker_error is None:
            message = f"{saved}; wrote {path}, so a restart comes back faulted"
        elif written:
            message = (
                f"{saved}, and the fallback marker {path} could not be updated "
                f"({marker_error}); the older marker still makes a restart come back faulted"
            )
        else:
            message = (
                f"{saved}, and neither could the fallback marker {path} ({marker_error}); the "
                "latch holds only in memory: do not restart the server before checking the arm"
            )
        log.critical(message)
        self._event(
            "safety.latches_unsaved",
            message,
            level="critical",
            source="robot.latches",
            detail={
                "version": version,
                "error": f"{type(failure).__name__}: {failure}",
                "latches": dict(latches),
                "marker": str(path),
                "marker_written": written,
                "marker_error": marker_error,
            },
        )

    def _restore_latches(self) -> None:
        """Come back latched after any restart (safety-B8, devex-B2).

        Runs in ``__init__``, before the manager serves anything. A crash, a
        clean restart and a ``--reload`` all restore the E-stop latch, the
        fault and the unconfirmed-torque flag; only the deliberate resets clear
        them. The safety state is derived from those latches: trips such as
        ``watchdog_timeout`` belonged to a lease that died with the old process.
        A store that cannot be read fails closed and starts faulted.

        So does a fallback marker (``<data dir>/control_plane.latch-unsaved``):
        the last process could not save a latch to the store. Its latches are
        restored too, and the fault says so. A marker older than the stored
        snapshot is obsolete (a newer snapshot reached the store) and is
        removed; an unreadable one counts as torque unconfirmed.
        """

        path = Path(
            getattr(
                self.store,
                "path",
                Path(self.settings.data_dir) / "control_plane.sqlite3",
            )
        )
        marker_records = read_identity_latch_markers(path, self._latch_identity)
        marker = combine_identity_latch_markers(marker_records)
        try:
            saved = self.store.load_latches()
        except Exception as exc:
            message = (
                f"could not read the persisted safety latches ({exc}); "
                "check the arm and its power, then reset the fault"
            )
            log.error(message)
            self._adopt_latch_marker(marker)
            with self._lock:
                if marker is not None:
                    self._apply_marker_latches_locked(marker)
                self._fault = message
                self._fault_epoch += 1
                self._safety_state = "estopped" if self.guard.estopped else "fault"
            self._event(
                "safety.latches_unreadable",
                message,
                level="critical",
                source="robot.restore",
                detail=None if marker is None else self._latch_marker_detail(marker),
            )
            return
        saved_version = int(saved.get("version") or 0) if saved is not None else 0
        for record in marker_records:
            if (
                record.version is not None
                and record.version <= saved_version
                and record.path == self._latch_marker_path()
            ):
                try:
                    record.path.unlink(missing_ok=True)
                except OSError as exc:  # it stays; harmless, the store is newer
                    log.warning(
                        "could not remove the obsolete latch marker %s: %s",
                        record.path,
                        exc,
                    )
        marker = combine_identity_latch_markers(
            marker_records,
            saved_version=saved_version,
        )
        if saved is None and marker is None:
            return
        estopped = bool(saved.get("estopped")) if saved is not None else False
        torque_unconfirmed = bool(saved.get("torque_unconfirmed")) if saved is not None else False
        energized = bool(saved.get("energized")) if saved is not None else False
        fault = (saved.get("fault") or None) if saved is not None else None
        if energized and saved is not None:
            torque_unconfirmed = True
            if fault is None:
                who = saved.get("energized_by") or "a standalone command"
                when = saved.get("energized_at") or saved.get("updated_at") or "an unknown time"
                fault = (
                    f"{who} may have left the arm energized at {when}; reset the fault to "
                    "perform a confirmed torque-off, or cut physical power"
                )
        if torque_unconfirmed and fault is None:
            fault = (
                "torque-off was unconfirmed before the restart; "
                "reset the fault to retry it, or cut physical power"
            )
        if marker is not None:
            fault = self._latch_marker_fault(marker, fault)
        self._adopt_latch_marker(marker)
        with self._lock:
            self._latch_version = max(self._latch_version, saved_version)
            if estopped:
                self.guard.estop()
            self._fault = fault
            if fault is not None:
                self._fault_epoch += 1
            self._torque_unconfirmed = torque_unconfirmed
            self._energized_recovery = energized
            if torque_unconfirmed or energized:
                self._add_recovery_devices_locked({"follower", "leader"})
            self._energized_at = saved.get("energized_at") if saved is not None else None
            self._energized_by = saved.get("energized_by") if saved is not None else None
            if marker is not None:
                self._apply_marker_latches_locked(marker)
            if current_backend(self.settings) == "real" and (estopped or fault is not None):
                # App lifespan republishes this under the host-global device
                # lock before serving, even when torque was previously off.
                self._add_recovery_devices_locked({"follower"})
            estopped = self.guard.estopped
            torque_unconfirmed = self._torque_unconfirmed
            if estopped:
                self._safety_state = "estopped"
            elif fault is not None:
                self._safety_state = "fault"
        if estopped or fault is not None:
            log.warning(
                "safety latches restored: estopped=%s fault=%r torque_unconfirmed=%s",
                estopped,
                fault,
                torque_unconfirmed,
            )
            detail: dict[str, Any] = {
                "estopped": estopped,
                "fault": fault,
                "torque_unconfirmed": torque_unconfirmed,
                "energized": self._energized_recovery,
                "saved_safety_state": saved.get("safety_state") if saved is not None else None,
                "saved_at": saved.get("updated_at") if saved is not None else None,
            }
            if marker is not None:
                detail["unsaved_marker"] = self._latch_marker_detail(marker)
            self._event(
                "safety.latches_restored",
                "safety latches restored after a restart"
                + ("; the last latch had not reached the store" if marker is not None else ""),
                level="critical" if marker is not None else "warning",
                source="robot.restore",
                detail=detail,
            )

    def _adopt_latch_marker(self, marker: _LatchMarker | None) -> None:
        """At start: the marker stays until a snapshot of this process reaches the store."""

        if marker is None:
            return
        version = marker.version or 0
        with self._latch_file_lock:
            self._latch_marker_version = version
            self._latch_unsaved_version = version
            self._latch_persistence = "marker"
        with self._lock:
            self._latch_version = max(self._latch_version, version)

    def _apply_marker_latches_locked(self, marker: _LatchMarker) -> None:
        """Merge an unsaved snapshot's latches into the restored ones. Holds ``_lock``."""

        latches = marker.latches
        if latches is None:
            # Unreadable: which latch it held is unknown, so assume the worst
            # one a fault reset can still clear (its retry runs first).
            self._torque_unconfirmed = True
            return
        if latches.get("estopped"):
            self.guard.estop()
        if latches.get("torque_unconfirmed"):
            self._torque_unconfirmed = True
            self._add_recovery_devices_locked({"follower", "leader"})
        if latches.get("energized"):
            self._energized_recovery = True
            self._torque_unconfirmed = True
            self._add_recovery_devices_locked({"follower", "leader"})
            self._energized_at = latches.get("energized_at") or self._energized_at
            self._energized_by = latches.get("energized_by") or self._energized_by

    @staticmethod
    def _latch_marker_fault(marker: _LatchMarker, saved_fault: str | None) -> str:
        if marker.latches is None:
            cause = f"its fallback marker {marker.path} is unreadable: {marker.error}"
        else:
            cause = marker.error or "the store refused it"
            unsaved_fault = marker.latches.get("fault")
            if unsaved_fault:
                cause += f"; the unsaved fault was: {unsaved_fault}"
        message = (
            f"the last safety latch could not be saved before the restart ({cause}); "
            "check the arm and its power, then reset the fault"
        )
        if saved_fault:
            message += f" (last saved fault: {saved_fault})"
        return message

    @staticmethod
    def _latch_marker_detail(marker: _LatchMarker) -> dict[str, Any]:
        return {
            "path": str(marker.path),
            "version": marker.version,
            "latches": marker.latches,
            "error": marker.error,
            "identity": marker.identity.as_dict() if marker.identity is not None else None,
        }

    @property
    def fault(self) -> str | None:
        with self._lock:
            return self._fault

    def _global_intent_devices(self, intent: dict[str, Any] | None) -> set[str]:
        if intent is None:
            return set()
        devices: set[str] = set()
        records = intent.get("markers")
        if not isinstance(records, list):
            records = [intent]
        for record in records:
            name = record.get("device_name") if isinstance(record, dict) else None
            if name in {"follower", "leader"}:
                devices.add(str(name))
            elif name == "" and isinstance(record, dict):
                aliases = record.get("alias_paths")
                configured_paths = (
                    {str(item) for item in aliases}
                    if isinstance(aliases, list)
                    else {str(record.get("configured_port") or "")}
                )
                matched = {
                    arm
                    for arm in ("follower", "leader")
                    if os.path.abspath(
                        os.path.expanduser(str(getattr(self.settings, f"{arm}_port", "")))
                    )
                    in configured_paths
                }
                devices.update(matched or {"follower", "leader"})
            else:
                devices.update({"follower", "leader"})
        return devices

    def adopt_global_energized_intent(
        self,
        intent: dict[str, Any] | None,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Mirror host-global standalone truth into the manager's visible fault.

        App lifespan calls this only while it owns the configured real-device
        locks. A marker from another data directory is therefore adopted
        before the server can accept motion.
        """

        if intent is None and error is None:
            return
        devices = self._global_intent_devices(intent)
        if not devices:
            devices.update({"follower", "leader"})
        if error is not None:
            message = (
                "the host-global energized marker could not be trusted "
                f"({type(error).__name__}: {error}); treat the arm as energized"
            )
        else:
            assert intent is not None
            purpose = str(intent.get("purpose") or "a standalone command")
            written_at = str(intent.get("written_at") or "an unknown time")
            message = (
                f"{purpose} may have left the arm energized at {written_at}; reset the fault "
                "to perform a confirmed torque-off, or cut physical power"
            )
        with self._lock:
            previous = self._fault
            if previous and previous not in message:
                message += f" (previous fault: {previous})"
            self._fault = message
            self._fault_epoch += 1
            self._torque_unconfirmed = True
            self._energized_recovery = True
            self._global_energized_recovery = True
            self._add_recovery_devices_locked(devices)
            self._energized_at = (
                str(intent.get("written_at"))
                if intent is not None
                else datetime.now(timezone.utc).isoformat()
            )
            self._energized_by = (
                str(intent.get("purpose")) if intent is not None else "unreadable global marker"
            )
            self._armed = False
            self._authority_epoch += 1
            self._safety_state = self._latched_state_locked()
        if error is not None:
            self._global_intent_unsaved_devices.update(devices)
        self._persist_latches()
        self._event(
            "safety.global_energized_restored",
            message,
            level="critical",
            source="robot.restore",
            detail={"intent": intent, "error": None if error is None else str(error)},
        )

    def _clear_global_energized_recovery(
        self,
        reason: str,
        *,
        devices: set[str] | None = None,
        expected_recovery_epoch: int | None = None,
        force_current_aliases: bool = False,
    ) -> bool:
        selected = set(devices or {"follower"})
        with self._global_intent_lock:
            with self._lock:
                active = self._global_energized_recovery
                if (
                    expected_recovery_epoch is not None
                    and self._recovery_epoch != expected_recovery_epoch
                ):
                    return False
            if not active:
                return True
            if selected == {"follower", "leader"}:
                device_name = ""
            elif selected == {"leader"}:
                device_name = "leader"
            else:
                device_name = "follower"
            try:
                bus_ownership.clear_global_energized_intent(
                    self.settings,
                    purpose=reason,
                    device_name=device_name,
                    force_current_aliases=force_current_aliases,
                    recover_corrupt_current=True,
                )
                remaining = bus_ownership.read_global_energized_intent(self.settings)
            except Exception as exc:
                self._global_intent_unsaved_devices.update(selected)
                message = (
                    "torque-off was confirmed, but the host-global energized marker could not "
                    f"be cleared ({exc}); recovery remains latched"
                )
                log.critical(message)
                self._event(
                    "safety.global_energized_clear_failed",
                    message,
                    level="critical",
                    source=reason,
                )
                return False
            remaining_devices = self._global_intent_devices(remaining)
            if remaining_devices & selected:
                message = (
                    "confirmed torque-off did not clear every selected host-global energized "
                    f"marker ({', '.join(sorted(remaining_devices & selected))}); "
                    "recovery remains latched"
                )
                log.critical(message)
                self._event(
                    "safety.global_energized_clear_incomplete",
                    message,
                    level="critical",
                    source=reason,
                    detail={"remaining": remaining},
                )
                return False
            with self._lock:
                self._global_energized_recovery = bool(remaining_devices)
            self._global_intent_unsaved_devices.difference_update(selected)
            return True

    def _prepare_global_energized_intent(
        self, reason: str, *, include_leader: bool = False
    ) -> None:
        """Write ahead before a real manager-owned path can enable torque."""

        if current_backend(self.settings) != "real":
            return
        written_at = datetime.now(timezone.utc).isoformat()
        devices = {"follower", "leader"} if include_leader else {"follower"}
        device_name = "" if include_leader else "follower"
        failure: Exception | None = None
        adopted = False
        with self._global_intent_lock:
            try:
                adopted = bus_ownership.enable_intent_is_adopted(
                    self.settings,
                    device_name=device_name,
                )
                if not adopted:
                    bus_ownership.write_global_energized_intent(
                        self.settings,
                        purpose=reason,
                        device_name=device_name,
                    )
            except Exception as exc:
                failure = exc
                self._global_intent_unsaved_devices.update(devices)
            else:
                self._global_intent_unsaved_devices.difference_update(devices)
            with self._lock:
                self._energized_recovery = True
                self._global_energized_recovery = True
                self._energized_at = written_at
                self._energized_by = reason
                self._add_recovery_devices_locked(devices)
        if failure is not None:
            message = (
                "refusing to enable follower torque because the host-global energized "
                f"intent could not be saved ({failure})"
            )
            fault, _epoch = self._latch_fault_in_memory(message)
            self._record_fault(fault, source="robot.energized_intent")
            raise RuntimeError(message) from failure
        if adopted:
            return  # Halt retained both durable marker and local mirror.
        # The host-global marker is already durable, so a local-store failure
        # cannot open a cross-data-directory safety hole. It remains visible
        # through latch_persistence and the store's fallback marker when able.
        self._persist_latches()

    def _write_global_recovery_intent(self, reason: str, devices: set[str]) -> Exception | None:
        """Best-effort durable cross-data truth for an already-unknown bus."""

        if current_backend(self.settings) != "real":
            return None
        if devices == {"follower", "leader"}:
            device_name = ""
        elif devices == {"leader"}:
            device_name = "leader"
        else:
            device_name = "follower"
        failure: Exception | None = None
        with self._global_intent_lock:
            try:
                bus_ownership.write_global_energized_intent(
                    self.settings,
                    purpose=reason,
                    device_name=device_name,
                )
            except Exception as exc:
                failure = exc
                self._global_intent_unsaved_devices.update(devices)
            else:
                self._global_intent_unsaved_devices.difference_update(devices)
            with self._lock:
                self._global_energized_recovery = True
                self._energized_recovery = True
                self._energized_at = datetime.now(timezone.utc).isoformat()
                self._energized_by = reason
                self._add_recovery_devices_locked(devices)
        return failure

    def publish_restored_recovery_intent(self) -> None:
        """Publish pre-global local recovery truth before a real server serves."""

        with self._lock:
            devices = set(self._recovery_devices)
        if not devices or current_backend(self.settings) != "real":
            return
        reason = "restored local torque/energized recovery state"
        failure = self._write_global_recovery_intent(reason, devices)
        if failure is None:
            self._persist_latches()
            return
        message = (
            f"{reason}; host-global energized intent is unsaved "
            f"({type(failure).__name__}: {failure})"
        )
        with self._lock:
            previous = self._fault
            self._fault = f"{message} (previous fault: {previous})" if previous else message
            self._fault_epoch += 1
            self._safety_state = self._latched_state_locked()
        self._persist_latches()
        self._event(
            "safety.global_energized_unsaved",
            message,
            level="critical",
            source="robot.restore",
        )

    def _retire_failed_enable_intent_if_safe(self, follower: SO101Follower, reason: str) -> None:
        """Retire a write-ahead marker when setup proved it never left torque on."""

        if current_backend(self.settings) != "real":
            return
        try:
            confirmed_off = not follower.connected and follower.torque_state == "off"
        except Exception:
            confirmed_off = False
        if not confirmed_off:
            return
        self._retire_confirmed_recovery_devices({"follower"}, reason)

    def _retire_confirmed_recovery_devices(self, devices: set[str], reason: str) -> bool:
        """Retire only devices just confirmed off, preserving newer/other intent."""

        with self._lock:
            recovery_epoch = self._recovery_epoch
        if not self._clear_global_energized_recovery(
            reason,
            devices=devices,
            expected_recovery_epoch=recovery_epoch,
        ):
            return False
        with self._lock:
            if self._recovery_epoch != recovery_epoch:
                return False
            self._recovery_devices.difference_update(devices)
            if not self._recovery_devices and not self._global_energized_recovery:
                self._energized_recovery = False
                self._energized_at = None
                self._energized_by = None
            self._recovery_epoch += 1
        self._persist_latches()
        return True

    def clear_fault(self, *, operator_id: str, power_cut_acknowledged: bool = False) -> None:
        """Deliberately clear a runtime fault (C2).

        While torque-off is unconfirmed, the follower port is still open, or a
        bus closed with its torque-off unconfirmed, this first retries a real
        torque-off with read-back on every bus, reopening closed ports through
        a fresh handle (safety-B2, docs/audit/v0.1/safety.md). If the retry
        cannot confirm it, the fault stays latched (ControlConflict, HTTP 409)
        unless the operator acknowledges that physical power was cut. The
        acknowledgement never skips the retry; it only overrides the retry's
        failure. It then releases the buses the retry could not confirm: open
        ports close with no torque write, and closed ones stop reporting
        torque unconfirmed (ControlBusy, HTTP 503, with the fault still
        latched, while the bus is held).
        """

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ControlConflict("cannot clear a fault while a control loop is active")
            if self._stop_transitions:
                raise ControlConflict("cannot clear a fault while a stop is being established")
            if self._stop_state == "collapsing":
                raise ControlConflict(
                    "a Collapse is lowering the arm; wait for it to finish, then reset the fault"
                )
            if self.guard.estopped:
                raise ControlConflict(
                    "reset the E-stop after physical isolation before clearing the fault"
                )
            follower = self._follower
            leader = self._leader
            retry = (
                self._torque_unconfirmed
                or self._energized_recovery
                or self._global_energized_recovery
                or bool(self._recovery_devices)
                or bool(follower is not None and follower.connected)
                or bool(leader is not None and leader.connected)
            )
            fault_epoch = self._fault_epoch
            recovery_epoch_before_retry = self._recovery_epoch
        retry = retry or bool(self._closed_unconfirmed_buses(include_leader=True))
        confirmed = self._retry_torque_off("fault reset") if retry else True
        expected_fault_epoch = fault_epoch
        expected_recovery_epoch = recovery_epoch_before_retry
        if not confirmed:
            # The retry deliberately suppresses its normal fault mutation so
            # this reset can compare against the fault it started with. Keep
            # torque truth fail-closed even if that comparison finds a newer
            # fault: the newer message wins, but the failed cut is persisted.
            with self._lock:
                self._torque_unconfirmed = True
            suffix = (
                "physical power cut was acknowledged"
                if power_cut_acknowledged
                else "cut physical power before recovery"
            )
            try:
                message, expected_fault_epoch = self._latch_fault_in_memory(
                    f"torque-off is still unconfirmed after the fault-reset retry; {suffix}",
                    expected_fault_epoch=fault_epoch,
                    torque_unconfirmed=True,
                )
            except ControlConflict:
                self._persist_latches()
                raise
            global_error = self._write_global_recovery_intent(
                "fault-reset retry left torque unconfirmed",
                {"follower", "leader"},
            )
            if global_error is not None:
                message += (
                    "; host-global energized intent is unsaved "
                    f"({type(global_error).__name__}: {global_error})"
                )
                with self._lock:
                    if self._fault_epoch == expected_fault_epoch:
                        self._fault = message
            self._record_fault(message, source="fault reset")
            # _latch_fault_in_memory above is this reset's sole recovery-scope
            # mutation. Derive its generation rather than sampling later, so a
            # concurrent newer fault can never be incorporated then force-cleared.
            expected_recovery_epoch = recovery_epoch_before_retry + (
                2 if current_backend(self.settings) == "real" else 1
            )
        else:
            with self._lock:
                expected_recovery_epoch = self._recovery_epoch
        if not confirmed and not power_cut_acknowledged:
            raise ControlConflict(
                "torque-off is still unconfirmed after a retry; cut physical power to the "
                "arm, then reset the fault with power_cut_acknowledged: true"
            )
        released = [] if confirmed else self._release_ports()
        if (
            not confirmed
            and power_cut_acknowledged
            and not self._clear_global_energized_recovery(
                "operator acknowledged physical power cut",
                devices={"follower", "leader"},
                expected_recovery_epoch=expected_recovery_epoch,
                force_current_aliases=True,
            )
        ):
            self._persist_latches()
            raise ControlConflict(
                "physical power cut was acknowledged, but the host-global energized marker "
                "could not be cleared; the fault remains latched"
            )
        reset_conflict: str | None = None
        with self._lock:
            if self.guard.estopped:
                reset_conflict = "the E-stop was latched during the fault reset"
            elif self._fault_epoch != expected_fault_epoch:
                reset_conflict = (
                    "a newer runtime fault was latched during the fault reset; it was not cleared"
                )
            elif self._recovery_epoch != expected_recovery_epoch:
                reset_conflict = (
                    "newer torque recovery state was recorded during the fault reset; "
                    "it was not cleared"
                )
            if reset_conflict is not None:
                # A concurrent category-0 failure may have set this just after
                # the reset's successful pass released the bus. Never let the
                # older reset clear that newer torque truth.
                self._torque_unconfirmed = True
            else:
                self._fault = None
                # Only reached unconfirmed when the operator acknowledged the cut.
                self._torque_unconfirmed = False
                self._energized_recovery = False
                self._energized_at = None
                self._energized_by = None
                self._recovery_devices.clear()
                self._recovery_epoch += 1
                self._safety_state = "disarmed"
        if reset_conflict is not None:
            self._retain_global_safety_latch("newer safety fault retained after superseded reset")
        self._persist_latches()
        if reset_conflict is not None:
            raise ControlConflict(reset_conflict)
        if not confirmed:
            self._event(
                "safety.power_cut_acknowledged",
                "fault cleared on the operator's acknowledgement that physical power was cut; "
                "the torque-off retry was not confirmed",
                level="critical",
                detail={"operator_id": operator_id, "ports_released": released},
            )
        self._event(
            "fault.cleared",
            "runtime fault cleared",
            detail={
                "operator_id": operator_id,
                "torque_off_retried": retry,
                "torque_off_confirmed": confirmed,
            },
        )

    def claim_calibration(
        self, arm: str, *, sample_seconds: float | None = None, joint: str | None = None
    ) -> None:
        """Atomically reserve hardware for the separate calibration worker."""

        if arm not in ("follower", "leader"):
            raise ValueError("arm must be follower or leader")
        if joint is not None and joint not in JOINT_NAMES:
            raise ValueError("unknown calibration joint")
        self._expire_lease_if_needed()
        with self._lock:
            if self.cal_in_progress:
                raise ControlConflict("calibration already running")
            if self._camera_preview_thread is not None:
                raise ControlConflict("stop camera preview before calibrating")
            if self._stop_transitions:
                raise ControlConflict("a Halt or Collapse is being established")
            if self.mode is not Mode.IDLE or (self._thread and self._thread.is_alive()):
                raise ControlConflict(f"robot is busy ({self.mode.value})")
            follower = self._follower
            if self._stop_state != "none" or (follower is not None and follower.connected):
                # The calibration worker opens its own bus on the same port, and
                # the halted follower is energized (design doc §5).
                raise ControlConflict(
                    "the follower is energized and holding (halted); Collapse it to the rest "
                    "pose before calibrating"
                )
            if self._lease is not None or self._armed:
                raise ControlConflict("external control lease is active")
            if self.guard.estopped:
                raise ControlRejected("E-STOP is latched; explicitly reset it first")
            if self._fault:
                raise ControlRejected(f"runtime fault: {self._fault}")
            self.cal_in_progress = True
            self.cal_arm = arm
            self.cal_fraction = 0.0
            self.cal_message = "starting"
            self._calibration_cancel.clear()
            attempt = {
                "attempt_id": uuid.uuid4().hex,
                "arm": arm,
                "backend": current_backend(self.settings),
                "device_id": getattr(self.settings, f"{arm}_id"),
                "sample_seconds": sample_seconds,
                "joint": joint,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self._calibration_attempt = attempt
            self._calibration_diagnostics = {}
            self._calibration_live = None
        try:
            self.store.append_event(
                "calibration.started",
                f"{arm} calibration started",
                source="calibration",
                detail=attempt,
                durable=True,
            )
        except Exception as exc:
            with self._lock:
                self.cal_in_progress = False
                self.cal_message = "calibration history could not be saved; no hardware opened"
            raise ControlRejected(self.cal_message) from exc

    @property
    def calibration_attempt_id(self) -> str | None:
        with self._lock:
            return self._calibration_attempt.get("attempt_id") if self.cal_in_progress else None

    def calibration_live_status(self, arm: str) -> dict[str, Any] | None:
        """Cached worker readings only; never acquire the bus lock or poll hardware."""
        with self._lock:
            if arm != self.cal_arm or self._calibration_live is None:
                return None
            return {
                **copy.deepcopy(self._calibration_live),
                "attempt_id": self._calibration_attempt.get("attempt_id"),
                "age_seconds": max(0.0, self._clock() - self._calibration_live_at),
                "frozen": not self.cal_in_progress,
            }

    def finish_calibration(
        self, arm: str, error: str | None = None, *, saved: bool | None = None
    ) -> None:
        """End the calibration reservation. ``saved`` (default: no error) reloads
        the follower's safety limits from the file the calibration wrote, also
        when an error came after the save (an unconfirmed torque-off)."""

        calibrated_limits: dict[str, tuple[float, float]] | None = None
        if saved is None:
            saved = error is None
        joint = self._calibration_attempt.get("joint")
        if saved and arm == "follower" and joint is None:
            try:
                calibrated_limits = dict(SO101Follower.guard_from_saved_calibration().limits)
            except Exception as exc:
                reload_error = f"saved, but safety limits could not be reloaded: {exc}"
                error = f"{error}; {reload_error}" if error else reload_error
        with self._lock:
            self.cal_in_progress = False
            self.cal_arm = arm
            if error:
                self.cal_message = f"failed: {error}"
            else:
                if calibrated_limits is not None:
                    self.guard.limits = calibrated_limits
                    self.guard.seed()
                self.cal_fraction = 1.0
                self.cal_message = (
                    f"{joint} baseline saved; arm calibration unchanged" if joint else "saved"
                )
            if self._calibration_live is not None:
                self._calibration_live.update(
                    phase="failed" if error else "complete",
                    error=error,
                    affected_motors=[
                        motor
                        for motor in JOINT_NAMES
                        if motor in self._calibration_live.get("affected_motors", [])
                        or motor in self._calibration_diagnostics.get("affected_motors", [])
                        or motor in affected_motors(error)
                    ],
                )
                self._calibration_diagnostics["live"] = copy.deepcopy(self._calibration_live)
            self._calibration_thread = None
            detail = {
                **self._calibration_attempt,
                "arm": arm,
                "saved": saved,
                "error": error,
                "affected_motors": [
                    motor
                    for motor in JOINT_NAMES
                    if motor in self._calibration_diagnostics.get("affected_motors", [])
                    or motor in affected_motors(error)
                ],
                "diagnostics": dict(self._calibration_diagnostics),
            }
        try:
            self.store.append_event(
                "calibration.failed" if error else "calibration.saved",
                f"{arm} calibration {'failed: ' + error if error else 'saved'}",
                level="error" if error else "info",
                source="calibration",
                detail=detail,
                durable=True,
            )
        except Exception as exc:
            log.error("could not persist calibration result: %s", exc)
            with self._lock:
                if self._calibration_attempt.get("attempt_id") == detail.get("attempt_id"):
                    self.cal_message += "; calibration history persistence failed"

    @property
    def calibration_cancel_event(self) -> threading.Event:
        return self._calibration_cancel

    def set_calibration_worker(self, worker: threading.Thread) -> None:
        with self._lock:
            if not self.cal_in_progress:
                raise ControlConflict("calibration reservation is not active")
            self._calibration_thread = worker

    def cancel_calibration(self, timeout: float = 3.0) -> None:
        with self._lock:
            if not self.cal_in_progress:
                return
            self._calibration_cancel.set()
            worker = self._calibration_thread
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        if worker and worker.is_alive():
            self._event(
                "calibration.cancel_timeout",
                "calibration worker did not stop before timeout; torque remains disabled",
                level="critical",
                source="calibration",
            )

    def run_calibration(
        self, arm: str, sample_seconds: float, *, joint: str | None = None
    ) -> CalibrationResult:
        """The calibration worker's bus access, through the manager (the only bus owner).

        Needs the reservation from :meth:`claim_calibration`. Uses the
        manager's own bus for ``arm`` under ``_io_lock`` for the whole sweep,
        so the manager's three-state torque truth covers it: the sweep starts
        with the suite's torque-off pass and ends with ``disconnect()`` (the
        same pass, then the port always closes). The lock is taken with the
        request bound (``ControlBusy``, HTTP 503). An E-stop cancels the sweep
        through the calibration cancel event; the sweep checks it every
        sample. Refused under the sim backend.

        A torque-off that cannot be confirmed afterwards latches a fault and
        ``torque_unconfirmed`` and raises
        :class:`CalibrationTorqueOffUnconfirmed` with a "cut power" message;
        any calibration error is named in it too.
        """

        if arm not in ("follower", "leader"):
            raise ValueError("arm must be follower or leader")
        if current_backend(self.settings) == "sim":
            raise ControlRejected(SIM_CALIBRATION_REFUSED)
        with self._lock:
            if not self.cal_in_progress or self.cal_arm != arm:
                raise ControlConflict(f"no {arm} calibration reservation is active")
            if self._calibration_attempt.get("joint") != joint:
                raise ControlConflict("calibration joint does not match the reservation")
        device: SO101Follower | SO101Leader = (
            self.follower() if arm == "follower" else self.leader()
        )
        bus = device.bus
        if isinstance(bus, FeetechBus):
            bus.calibration_diagnostics = {}

        def progress(fraction: float, message: str) -> None:
            with self._lock:
                self.cal_fraction, self.cal_message = fraction, message

        def telemetry(snapshot: dict[str, Any]) -> None:
            with self._lock:
                self._calibration_live = copy.deepcopy(snapshot)
                # Native calibration is only one part of success: the suite
                # file and confirmed disconnect must finish before UI PASS.
                if self._calibration_live.get("phase") == "complete":
                    self._calibration_live["phase"] = "saving"
                self._calibration_live_at = self._clock()

        self._acquire_io_for_request("calibration")
        result: CalibrationResult | None = None
        saved = False
        failure: Exception | None = None
        unconfirmed: Exception | None = None
        try:
            try:
                if self._calibration_cancel.is_set() or self.guard.estopped:
                    raise RuntimeError("calibration cancelled before it started")
                bus.connect()
                result = calibrate(
                    bus,
                    arm,
                    sample_seconds=sample_seconds,
                    progress=progress,
                    cancel_event=self._calibration_cancel,
                    telemetry=telemetry,
                    **({"joint": joint} if joint is not None else {}),
                )
                result.save()
                saved = True
            except Exception as exc:  # reported after the torque-off below
                failure = exc
            try:
                # Torque-off with read-back on every motor, then the port
                # always closes; raises when torque-off is not confirmed.
                bus.disconnect()
            except Exception as exc:
                unconfirmed = exc
            if unconfirmed is None and bus_torque_state(bus) != "off":
                # A failed connect may already have closed and dropped its
                # handle. disconnect() is then an intentional no-op, but the
                # bus's surviving three-state truth must still latch the fault.
                unconfirmed = (
                    failure
                    if isinstance(failure, TorqueOffUnconfirmed)
                    else TorqueOffUnconfirmed(str(getattr(bus, "name", arm)), ["all motors"])
                )
        finally:
            with self._lock:
                self._calibration_diagnostics = dict(getattr(bus, "calibration_diagnostics", {}))
            self._io_lock.release()
        if unconfirmed is not None:
            what = f"calibration of the {arm} arm" + (" saved, but" if saved else " ended and")
            message = (
                f"{what} torque-off was not confirmed ({unconfirmed}); cut physical power to "
                f"the {arm} arm, then reset the fault"
            )
            if failure is not None:
                message += f" (the calibration itself failed: {failure})"
            self._mark_torque_unconfirmed(
                message,
                event="safety.force_deenergize_failed",
                source="calibration",
                devices={arm},
            )
            raise CalibrationTorqueOffUnconfirmed(message, saved=saved) from unconfirmed
        if failure is not None:
            raise failure
        assert result is not None
        return result

    # --- camera-only preview --------------------------------------------
    def camera_preview_status(self) -> dict[str, Any]:
        """Cached preview facts only; polling never reads a device."""
        with self._lock:
            return {
                "active": self._camera_preview_thread is not None,
                "error": self._camera_preview_error,
                "cameras": configured_streams(self.settings),
                "devices": copy.deepcopy(self._camera_preview_devices),
            }

    def start_camera_preview(self) -> None:
        """Reserve camera capture while idle, without opening a motor bus."""
        self._require_direct_camera_capture()
        with self._lock:
            if self._camera_preview_thread is not None:
                raise RuntimeError("camera preview is already running or stopping")
            if self.cal_in_progress or self.mode is not Mode.IDLE or self._thread is not None:
                raise RuntimeError("robot is busy; stop its operation before camera preview")
            if self._lease is not None or self._armed:
                raise RuntimeError("release external control before camera preview")
            if not configured_streams(self.settings):
                raise RuntimeError("no cameras are configured")
            self._camera_preview_stop.clear()
            self._camera_preview_error = None
            self._camera_preview_devices = {}
            thread = threading.Thread(
                target=self._camera_preview_loop, name="camera-preview", daemon=True
            )
            self._camera_preview_thread = thread
            try:
                thread.start()
            except BaseException:
                self._camera_preview_thread = None
                raise

    def _camera_preview_loop(self) -> None:
        cameras: CameraManager | None = None
        try:
            cameras = self.cameras()
            cameras.open()
            with self._lock:
                self._camera_preview_devices = cameras.describe()
            while not self._camera_preview_stop.is_set():
                frames = cameras.read()
                captured_at = self._clock()
                with self._lock:
                    self._frames = frames
                    self._frame_times = {camera: captured_at for camera in frames}
                self._camera_preview_stop.wait(1 / 20)
        except Exception as exc:
            # A camera error is visible here, without invoking motor fault or
            # torque paths. Real CameraManager never substitutes mock frames.
            with self._lock:
                self._camera_preview_error = str(exc)
            log.warning("camera preview failed: %s", exc)
        finally:
            cleanup_error: Exception | None = None
            if cameras is not None:
                try:
                    cameras.close()
                except Exception as exc:
                    cleanup_error = exc
            with self._lock:
                self._frames = {}
                self._frame_times = {}
                if cleanup_error is None:
                    self._camera_preview_thread = None
                else:
                    # Unknown camera ownership must not race a second capture.
                    self._camera_preview_error = (
                        f"camera cleanup failed: {cleanup_error}; capture remains reserved; "
                        "restart the service before using cameras"
                    )

    def stop_camera_preview(self) -> None:
        """Request camera-only cleanup; a blocked driver retains its reservation."""
        with self._lock:
            thread = self._camera_preview_thread
            self._camera_preview_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                with self._lock:
                    self._camera_preview_error = (
                        "camera driver is still stopping; capture remains reserved"
                    )

    # --- lazy hardware --------------------------------------------------
    def follower(self) -> SO101Follower:
        with self._lock:
            if self._follower is None:
                # Let the follower construct its guard from saved calibration,
                # then adopt that same instance as the manager-wide guard.
                follower = SO101Follower()
                if self.guard.estopped:
                    follower.guard.estop()
                self.guard = follower.guard
                self._follower = follower
            return self._follower

    def leader(self) -> SO101Leader:
        with self._lock:
            if self._leader is None:
                self._leader = SO101Leader()
            return self._leader

    def cameras(self) -> CameraManager:
        with self._lock:
            if self._cameras is None:
                self._cameras = CameraManager()
            return self._cameras

    # --- stop visibility (design doc §5 "Visibility") ------------------
    def _torque_state(self) -> TorqueState:
        """Torque truth over every bus the manager owns: unconfirmed > on > off.

        The manager's own ``torque_unconfirmed`` latch counts as unconfirmed,
        and so does any bus that cannot say (``bus_torque_state``).
        """

        with self._lock:
            buses = [device.bus for device in (self._follower, self._leader) if device is not None]
            flagged = self._torque_unconfirmed
        states = {bus_torque_state(bus) for bus in buses}
        if flagged or "unconfirmed" in states:
            return "unconfirmed"
        return "on" if "on" in states else "off"

    @staticmethod
    def _iso(wall: float) -> str:
        return datetime.fromtimestamp(wall, timezone.utc).isoformat()

    def _supervision_locked(self, now: float) -> tuple[bool, float | None]:
        """(supervised, stale_at). Supervised while any operator heartbeat is fresh.

        ``stale_at`` is when the newest heartbeat goes (or went) stale, on the
        manager clock; None when no operator ever sent one.
        """

        latest = max(self._operator_heartbeats.values(), default=None)
        if latest is None:
            return False, None
        stale_at = latest + self.operator_timeout_ms / 1000
        return now <= stale_at, stale_at

    def _operator_fresh_locked(self, operator_id: str | None, now: float) -> bool:
        if not operator_id:
            return False
        seen = self._operator_heartbeats.get(operator_id)
        return seen is not None and (now - seen) * 1000 <= self.operator_timeout_ms

    def _halt_expiry_locked(self, halt: _Halt, now: float) -> float:
        """When this halt becomes a Collapse, given supervision right now.

        Supervised: the hold timer. Unsupervised: the grace period, counted
        from when supervision was lost (or from the halt, if later), never
        past the hold timer.
        """

        supervised, stale_at = self._supervision_locked(now)
        if supervised:
            return halt.deadline
        lost_from = halt.since if stale_at is None else max(stale_at, halt.since)
        return min(halt.deadline, lost_from + self.settings.supervision_grace_seconds)

    def _stop_views_locked(self) -> dict[str, Any]:
        """``stop_state``, ``halt``, ``collapse`` and ``supervision`` (JSON-safe)."""

        now, wall = self._clock(), time.time()
        halt_view: dict[str, Any] | None = None
        halt = self._halt if self._stop_state == "halted" else None
        if halt is not None:
            expiry = self._halt_expiry_locked(halt, now)
            halt_view = {
                "since": self._iso(halt.since_wall),
                "reason": halt.reason,
                "hold_expires_at": self._iso(wall + max(0.0, expiry - now)),
                "server_time": self._iso(wall),
            }
        collapse_view: dict[str, Any] | None = None
        collapse = self._collapse if self._stop_state == "collapsing" else None
        if collapse is not None:
            collapse_view = {
                "since": self._iso(collapse.since_wall),
                "reason": collapse.reason,
                "progress": round(collapse.progress, 3),
                "eta_s": round(collapse.eta_s, 2),
            }
        supervised, stale_at = self._supervision_locked(now)
        lost_at = None
        if not supervised and stale_at is not None:
            lost_at = self._iso(wall - (now - stale_at))
        return {
            "stop_state": self._stop_state,
            "halt": halt_view,
            "collapse": collapse_view,
            "supervision": {
                "supervised": supervised,
                "lost_at": lost_at,
                "grace_s": self.settings.supervision_grace_seconds,
            },
        }

    # --- status ---------------------------------------------------------
    def status(self) -> dict:
        """Return the current robot view without changing state or touching a bus.

        The watchdog and mutating command paths expire leases. Status is also
        served by an unguarded GET, so it must never turn a read into a Halt.
        """

        torque_state = self._torque_state()
        with self._lock:
            follower, leader = self._follower, self._leader
            now = self._clock()
            frame_ages = {
                camera: round((now - captured_at) * 1000, 1)
                for camera, captured_at in self._frame_times.items()
            }
            live_cameras = [camera for camera, age_ms in frame_ages.items() if age_ms <= 1_000]
            return {
                "mode": self.mode.value,
                "follower_connected": bool(follower and follower.connected),
                "leader_connected": bool(leader and leader.connected),
                "estopped": self.guard.estopped,
                "mock": self.settings.mock,
                "fault": self._fault,
                # Stream keys (credentials stripped; "<entry>#depth" for depth).
                "cameras": configured_streams(self.settings),
                "live_cameras": live_cameras,
                "camera_frame_age_ms": frame_ages,
                "operation_owner": ("active browser session" if self._operation_owner else None),
                "fps": float(self._snapshot.get("fps", 0.0)),
                "stop_kinds": [kind.value for kind in IMPLEMENTED_STOP_KINDS],
                "torque_unconfirmed": self._torque_unconfirmed,
                # safety-B8: whether a restart comes back at least this latched
                # (an attribute read; _latch_file_lock guards its writes).
                "latch_persistence": (
                    "unsaved" if self._global_intent_unsaved_devices else self._latch_persistence
                ),
                # C7 (frontend F10, docs/audit/v0.1/frontend.md).
                "active_model": self._active_model if self.mode is Mode.ROLLOUT else None,
                # Design doc §5 "Visibility".
                "torque_state": torque_state,
                **self._stop_views_locked(),
                # Servo health in brief; GET /api/robot/health has the detail.
                "health": self._health_summary_locked(now),
                # mock | sim | real: "mock" above is false under sim too.
                "backend": current_backend(self.settings),
            }

    def telemetry(self) -> dict:
        torque_state = self._torque_state()
        with self._lock:
            snapshot = dict(self._snapshot)
            now = self._clock()
            sample_age_s = now - self._snapshot_monotonic
            loop_active = (
                self.mode in _LOOP_MODES and self._thread is not None and self._thread.is_alive()
            )
            stops = self._stop_views_locked()
            snapshot.update(
                mode=self.mode.value,
                estopped=self.guard.estopped,
                recording=self._episode_active,
                recording_saving=self._episode_saving,
                episode_frames=self._episode_frames,
                sample_age_ms=round(sample_age_s * 1000, 1),
                sample_seq=self._sample_seq,
                # C6 (frontend F8, docs/audit/v0.1/frontend.md): only an active
                # loop, an armed session, a halt's hold or a collapse descent
                # samples continuously. An idle robot or a one-off observe read
                # is not live, however recent.
                observation_live=bool(snapshot.get("observation"))
                and (loop_active or self._armed or self._stop_state != "none")
                and sample_age_s <= _LIVE_SAMPLE_MAX_AGE_S,
                # Design doc §5 "Visibility": every frame carries the stop state.
                torque_state=torque_state,
                stop_state=stops["stop_state"],
                halt=stops["halt"],
                collapse=stops["collapse"],
                # Servo health in brief (docs/bring-up.md "Servo health monitor").
                health=self._health_summary_locked(now),
            )
            return snapshot

    def get_frame(self, cam: str) -> np.ndarray | None:
        with self._lock:
            captured_at = self._frame_times.get(cam)
            if captured_at is None or self._clock() - captured_at > 1.0:
                return None
            return self._frames.get(cam)

    def operator_heartbeat(self, client_id: str) -> None:
        """Record a heartbeat sent *by* a live operator browser."""

        if not client_id or len(client_id) > 128:
            return
        with self._lock:
            now = self._clock()
            self._operator_heartbeats[client_id] = now
            if len(self._operator_heartbeats) > 256:
                cutoff = now - 60
                self._operator_heartbeats = {
                    key: seen for key, seen in self._operator_heartbeats.items() if seen >= cutoff
                }

    def owns_operation(self, operator_id: str) -> bool:
        with self._lock:
            return bool(operator_id and self._operation_owner == operator_id)

    def require_operator_heartbeat(self, operator_id: str) -> None:
        with self._lock:
            last_heartbeat = self._operator_heartbeats.get(operator_id)
            if (
                not operator_id
                or last_heartbeat is None
                or (self._clock() - last_heartbeat) * 1000 > self.operator_timeout_ms
            ):
                raise ControlRejected("a fresh owning operator heartbeat is required")

    # --- e-stop ---------------------------------------------------------
    def estop(self, reason: str = "operator E-stop") -> None:
        """Category 0 stop. Raises ControlRejected when it cannot confirm it."""

        self.request_stop(StopKind.ESTOP, reason=reason, strict=True)

    def latch_estop(self, reason: str, *, lock_timeout: float = -1) -> bool:
        """Set the E-stop latch and revoke all control authority. No bus I/O.

        Safe to call on the event loop (safety-B9, docs/audit/v0.1/safety.md).
        The guard latch is a lock-free flag set first, so every motion path
        rejects from here on. The state lock only ever guards short
        bookkeeping, never I/O, joins or sleeps. With ``lock_timeout`` >= 0
        the bookkeeping is left to the teardown if that lock does not come
        free in time; the latch itself is already set. Returns True when the
        bookkeeping ran. Idempotent.
        """

        # Keep the safety guard's lock-free flag first. Generation bookkeeping
        # comes second and never holds its tiny lock while waiting for state.
        self.guard.estop()
        with self._estop_generation_lock:
            self._estop_generation += 1
            self._estop_inflight += 1
        self.guard.estop()
        acquired = False
        try:
            # A running descent stops at its next tick, before any further
            # write (lock-free: an attribute read, so it lands even when
            # ``_lock`` is busy).
            collapse = self._collapse
            if collapse is not None:
                collapse.cancel.set()
            acquired = self._lock.acquire(timeout=lock_timeout)
            self._stop.set()
            self._calibration_cancel.set()
            if acquired:
                # A reset may have cleared the guard while this E-stop waited
                # for state. Reassert it as part of the same state transition.
                self.guard.estop()
                self._armed = False
                self._authority_epoch += 1
                # E-stop permanently supersedes every pending non-E-stop
                # transition, including one that has not published its
                # Collapse object yet. The lock-free guard already stopped
                # writes before this bookkeeping became available.
                self._stop_transition_serial += 1
                self._safety_state = "estopped"
                self._lease = None
                self._last_command_monotonic = None
        finally:
            if acquired:
                self._lock.release()
            # Publish completion only after the final guard write. A reset
            # that began while this latch was in flight rejects even if this
            # thread finished before the reset obtained ``_lock``.
            self.guard.estop()
            with self._estop_generation_lock:
                self._estop_inflight -= 1
        return acquired

    def _estop_reset_snapshot(self) -> tuple[int, bool]:
        """(generation, latch_in_flight) at reset invocation."""

        with self._estop_generation_lock:
            return self._estop_generation, self._estop_inflight > 0

    def _estop_category0(self, reason: str) -> StopResult:
        # Latch first: every command path and the guard reject from here on.
        self.latch_estop(reason)
        with self._lock:
            preempted = self._stop_state
        # Category 0 (docs/plans/2026-10-01-suite-v02-design.md §5): torque off
        # on every bus the manager owns before anything that can block, such as
        # joining the loop thread or waiting for the calibration worker. The cut
        # holds the I/O lock, so a late enable or write cannot resurrect motion.
        deenergized = self._cut_torque("estop", include_leader=True)
        # safety-B8: the latch reaches the store only after the cut, so the
        # write never delays torque-off.
        self._persist_latches()
        log.warning("E-STOP engaged (%s); torque_off=%s", reason, deenergized)
        stopped = self._stop_loop(join_timeout=0.5)
        if not deenergized and stopped:
            deenergized = self._retry_cut("estop-retry")
        self.cancel_calibration(timeout=0.5)
        self._persist_latches()
        detail: dict[str, Any] = {
            "reason": reason,
            "torque_off": deenergized,
            "loop_stopped": stopped,
        }
        if preempted != "none":
            detail["preempted"] = preempted  # E-stop preempts a Halt or a Collapse (§5)
        self._event("safety.estop", "E-STOP engaged", level="critical", detail=detail)
        return StopResult(StopKind.ESTOP, reason, loop_stopped=stopped, torque_off=deenergized)

    def reset_estop(self, *, operator_id: str) -> None:
        """Deliberately clear the E-stop latch (C2).

        A fault or an unconfirmed torque-off stays latched; clearing those is
        the separate fault reset, which retries the torque-off.
        """

        # Snapshot before waiting for ``_lock``. Both an E-stop requested
        # afterwards and one already in flight must win over this reset.
        generation, latch_in_flight = self._estop_reset_snapshot()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ControlConflict("cannot reset E-stop while a control loop is active")
            fault, torque_unconfirmed = self._fault, self._torque_unconfirmed
            recovery_epoch = self._recovery_epoch
            clear_global_latch = (
                fault is None and not torque_unconfirmed and self._global_energized_recovery
            )
        if clear_global_latch and not self._clear_global_energized_recovery(
            "E-stop explicitly reset after confirmed torque-off",
            devices={"follower", "leader"},
            expected_recovery_epoch=recovery_epoch,
        ):
            raise ControlConflict(
                "the host-global E-stop latch could not be cleared; E-stop remains latched"
            )
        with self._lock:
            with self._estop_generation_lock:
                newer_estop = (
                    latch_in_flight
                    or self._estop_generation != generation
                    or self._estop_inflight > 0
                )
            if not newer_estop:
                self.guard.reset()
                with self._estop_generation_lock:
                    newer_estop = (
                        self._estop_generation != generation
                        or self._estop_inflight > 0
                        or self.guard.estopped
                    )
            if newer_estop:
                self.guard.estop()
                self._safety_state = "estopped"
            else:
                self._armed = False
                self._safety_state = "fault" if self._fault else "disarmed"
                fault, torque_unconfirmed = self._fault, self._torque_unconfirmed
                if fault is None and not torque_unconfirmed and not self._global_energized_recovery:
                    self._energized_recovery = False
                    self._energized_at = None
                    self._energized_by = None
                    self._recovery_devices.clear()
                    self._recovery_epoch += 1
        if newer_estop:
            self._retain_global_safety_latch("newer E-stop retained after superseded reset")
            self._persist_latches()
            raise ControlConflict("a newer E-stop was latched during the reset")
        self._persist_latches()
        self._event(
            "safety.estop_reset",
            "E-stop explicitly reset",
            detail={
                "operator_id": operator_id,
                "fault": fault,
                "torque_unconfirmed": torque_unconfirmed,
            },
        )

    # --- stop dispatcher (C11, design doc §5) ---------------------------
    def request_stop(
        self,
        kind: StopKind,
        *,
        reason: str,
        lease_id: str | None = None,
        strict: bool = False,
        operator_id: str | None = None,
        origin_verified: bool = False,
    ) -> StopResult:
        """The single entry point for every stop.

        ``kind`` is the category from docs/plans/2026-10-01-suite-v02-design.md
        §5, as a ``StopKind`` or its string value. ``ESTOP`` is always global
        and ignores ``lease_id``. For ``HALT`` and ``COLLAPSE``:

        - With ``lease_id`` the stop is owner-scoped. If the id does not name
          the current lease, the call is a no-op and returns ``applied=False``.
        - Without it the stop is global and also ends legacy loops and
          calibration. Only a global HALT freezes a running Collapse; a scoped
          one only revokes authority.
        - ``operator_id`` and ``origin_verified`` matter only to a repeat
          HALT: it renews the hold timer when that operator has a fresh
          supervision heartbeat and the request passed the Origin guard.

        ``strict=False`` never raises, so error paths can always call it.
        ``strict=True`` raises ``ControlRejected`` when the stop was not
        applied, fell back to a torque cut, did not leave the arm accounted for
        (holding, descending or confirmed off), or a loop did not exit.
        """

        kind = StopKind(kind)
        if kind is not StopKind.ESTOP and lease_id is not None:
            # A lapsed lease is handled as expiry, itself a HALT, so a late
            # owner-scoped request becomes a no-op instead of a second stop.
            # It waits for an expiry already in flight on another thread, so
            # the caller never reads the arm mid-Halt.
            self._expire_lease_if_needed(wait=True)
        return self._dispatch_stop(
            kind,
            reason=reason,
            lease_id=lease_id,
            strict=strict,
            operator_id=operator_id,
            origin_verified=origin_verified,
        )

    def servo_health_critical(
        self, detail: str, *, health: dict[str, Any] | None = None
    ) -> StopResult:
        """Servo health critical (design doc §5): a Collapse that latches a fault.

        The fault latches after any loop has stopped, so the loop's teardown
        does not cut torque before the descent, and it stays latched when the
        Collapse ends: only the deliberate fault reset clears it. With the
        motors already off, or while E-stopped, it only latches the fault.
        Never raises. The servo-health poll (``so101-health``) calls this on a
        critical verdict; ``health`` (arm, reasons) goes into the event.
        """

        message = f"servo health critical: {detail}"
        self._event(
            "safety.servo_health_critical",
            message,
            level="critical",
            source="robot.health",
            detail=health,
        )
        return self._dispatch_stop(StopKind.COLLAPSE, reason=message, fault=message)

    def _dispatch_stop(
        self,
        kind: StopKind,
        *,
        reason: str,
        lease_id: str | None = None,
        strict: bool = False,
        epoch: int | None = None,
        trip_state: str | None = None,
        end_lease: bool = False,
        loops_only: bool = False,
        join_timeout: float = 3.0,
        operator_id: str | None = None,
        origin_verified: bool = False,
        fault: str | None = None,
        expected_halt: _Halt | None = None,
    ) -> StopResult:
        """Route one tagged stop to its implementation.

        Internal triggers pass their scope:

        - ``epoch``: apply only while the authority epoch is unchanged.
        - ``trip_state``: the safety state to show afterwards, for example
          ``watchdog_timeout``. A latched state always wins.
        - ``end_lease``: also revoke the lease (release, expiry).
        - ``loops_only``: end the legacy loop and leave any external session
          alone (the loop's own end, the eval runner, lost supervision).
        - ``fault``: a COLLAPSE that a fault triggered latches it (servo health).
        - ``expected_halt``: a COLLAPSE from the hold timer applies only while
          that halt is still the active one.
        """

        transition: int | None = None
        protected_pending_collapse = False
        protected_was_armed = False
        if kind is not StopKind.ESTOP:
            # Validate before publishing an in-progress stop. The concrete
            # implementation validates again after any join/I/O, because the
            # lease, epoch or expected Halt may change while it works.
            with self._lock:
                scope_valid = self._scope_valid_locked(lease_id, epoch)
                expected_valid = expected_halt is None or (
                    self._halt is expected_halt and self._stop_state == "halted"
                )
                if scope_valid and expected_valid:
                    current_pending = self._stop_transitions.get(self._stop_transition_serial)
                    pending_collapses: list[bool] = []
                    if current_pending is not None and not self.guard.estopped:
                        pending_kind, _pending_loops_only, fault_reaction = current_pending
                        if pending_kind is StopKind.COLLAPSE and (
                            fault_reaction or self._fault is None
                        ):
                            pending_collapses.append(fault_reaction)
                    operator_global_halt = (
                        kind is StopKind.HALT
                        and lease_id is None
                        and epoch is None
                        and not loops_only
                    )
                    protected_pending_collapse = bool(pending_collapses) and (
                        any(pending_collapses)
                        or (kind is StopKind.HALT and not operator_global_halt)
                    )
                    if protected_pending_collapse and not loops_only:
                        protected_was_armed = self._revoke_authority_locked(
                            global_stop=lease_id is None and epoch is None,
                            end_lease=end_lease,
                            trip_state=trip_state,
                        )
                    elif not protected_pending_collapse:
                        self._stop_transition_serial += 1
                        transition = self._stop_transition_serial
                        self._stop_transitions[transition] = (
                            kind,
                            loops_only,
                            fault is not None,
                        )
        if protected_pending_collapse:
            if protected_was_armed:
                self._disarmed_event(kind, reason)
            result = StopResult(kind, reason, torque_off=False, collapsing=True)
            if strict:
                self._raise_unless_confirmed(result)
            return result
        try:
            if kind is StopKind.ESTOP:
                result = self._estop_category0(reason)
            elif kind is StopKind.HALT:
                result = self._halt_stop(
                    reason,
                    transition=transition,
                    lease_id=lease_id,
                    epoch=epoch,
                    trip_state=trip_state,
                    end_lease=end_lease,
                    loops_only=loops_only,
                    join_timeout=join_timeout,
                    operator_id=operator_id,
                    origin_verified=origin_verified,
                )
            else:
                result = self._collapse_stop(
                    reason,
                    transition=transition,
                    lease_id=lease_id,
                    epoch=epoch,
                    trip_state=trip_state,
                    end_lease=end_lease,
                    join_timeout=join_timeout,
                    fault=fault,
                    expected_halt=expected_halt,
                )
        finally:
            if transition is not None:
                with self._lock:
                    self._stop_transitions.pop(transition, None)
        if strict:
            self._raise_unless_confirmed(result)
        return result

    def _external_stop_in_progress_locked(self) -> bool:
        """Whether a non-loops-only Halt/Collapse currently owns transition authority."""

        return any(
            not loops_only for _kind, loops_only, _fault_reaction in self._stop_transitions.values()
        )

    def _any_stop_in_progress_locked(self) -> bool:
        """Whether any Halt/Collapse request is between validation and its final result."""

        return bool(self._stop_transitions)

    def _stop_transition_is_current_locked(self, transition: int | None) -> bool:
        """Whether ``transition`` is still the newest valid non-E-stop request."""

        return (
            transition is not None
            and transition == self._stop_transition_serial
            and transition in self._stop_transitions
        )

    def _arm_cleanup_requires_torque_cut_locked(
        self, *, holding: bool, authority_epoch: int
    ) -> bool:
        """Whether a revoked arm must cut torque instead of deferring to its stop.

        E-stop always wins. A Halt/Collapse that is being established, or
        whose state is already visible, owns energized hardware and must not
        be converted into an uncommanded torque cut by stale arming cleanup;
        its own fault/fallback path remains responsible for a required cut.
        """

        if self.guard.estopped:
            return True
        if self._external_stop_in_progress_locked() or self._stop_state != "none":
            return False
        if self._fault is not None:
            return True
        if self._armed and self._authority_epoch != authority_epoch:
            return False  # a newer armed session owns the follower
        return not holding

    def _scope_valid_locked(self, lease_id: str | None, epoch: int | None) -> bool:
        if lease_id is not None and (self._lease is None or self._lease.id != lease_id):
            return False
        return epoch is None or epoch == self._authority_epoch

    def _revoke_authority_locked(
        self, *, global_stop: bool, end_lease: bool, trip_state: str | None
    ) -> bool:
        """Disarm external control; returns whether it was armed. Holds ``_lock``."""

        was_armed = self._armed
        self._armed = False
        self._authority_epoch += 1
        self._last_command_monotonic = None
        if self.mode is Mode.MANUAL:
            self.mode = Mode.IDLE
        if end_lease:
            self._lease = None
        latched = (
            self.guard.estopped or self._fault is not None or self._safety_state in _LATCHED_STATES
        )
        if latched:
            pass
        elif trip_state is not None:
            self._safety_state = trip_state
        elif global_stop or end_lease or self._safety_state not in _TRIPPED_STATES:
            # An owner-scoped stop leaves a trip visible (C5). A global stop or
            # the end of the lease is the deliberate clear, as in v0.1.
            self._safety_state = "disarmed"
        return was_armed

    def _disarmed_event(self, kind: StopKind, reason: str) -> None:
        self._event(
            "control.disarmed",
            f"manual control disarmed ({reason})",
            detail={"reason": reason, "stop_kind": kind.value},
        )

    # --- Halt: category 2, freeze in place -------------------------------
    def _halt_stop(
        self,
        reason: str,
        *,
        transition: int | None,
        lease_id: str | None,
        epoch: int | None,
        trip_state: str | None,
        end_lease: bool,
        loops_only: bool,
        join_timeout: float,
        operator_id: str | None,
        origin_verified: bool,
    ) -> StopResult:
        """Stop every motion source, disarm, and hold Goal := Present.

        Motors that are off stay off (an ok no-op). A repeat halt keeps the
        hold it finds and renews the timer only for a supervising operator
        whose request passed the Origin guard. A global halt freezes a running
        Collapse (operator override); a scoped one leaves it running.
        """

        kind = StopKind.HALT
        owner_scoped = lease_id is not None
        global_stop = not owner_scoped and epoch is None and not loops_only
        loop_stopped = True
        froze: str | None = None
        with self._lock:
            if not self._stop_transition_is_current_locked(
                transition
            ) or not self._scope_valid_locked(lease_id, epoch):
                return StopResult(kind, reason, applied=False)
        if global_stop:
            self.cancel_calibration()
            froze = self._freeze_collapse()
        if global_stop or loops_only:
            loop_stopped = self._stop_loop(join_timeout=join_timeout)
        renewed = False
        with self._lock:
            if not self._stop_transition_is_current_locked(
                transition
            ) or not self._scope_valid_locked(lease_id, epoch):
                return StopResult(kind, reason, applied=False)
            was_armed = False
            if not loops_only:
                was_armed = self._revoke_authority_locked(
                    global_stop=global_stop, end_lease=end_lease, trip_state=trip_state
                )
            state = self._stop_state
            halt = self._halt if state == "halted" else None
            # A loops-only halt leaves an external session alone (the eval
            # runner between episodes); that session owns the arm.
            session_owns_arm = loops_only and (self._armed or self._safety_state == "arming")
            now = self._clock()
            if (
                halt is not None
                and origin_verified
                and self._operator_fresh_locked(operator_id, now)
            ):
                halt.deadline = now + self.settings.halt_hold_seconds
                halt.renewals += 1
                renewed = True
        if was_armed:
            self._disarmed_event(kind, reason)
        if not loop_stopped:
            # _stop_loop latched a fault and cut torque (category 0 fallback).
            return StopResult(
                kind, reason, loop_stopped=False, torque_off=self._torque_state() == "off"
            )
        if self.guard.estopped:
            # The E-stop owns the arm; its torque-off is the truth.
            return StopResult(kind, reason, torque_off=self._torque_state() == "off")
        if state == "collapsing":
            return StopResult(kind, reason, torque_off=False, collapsing=True)
        if session_owns_arm:
            return StopResult(kind, reason, torque_off=self._torque_state() == "off")
        if halt is not None:
            with self._lock:
                active_fault = self._fault
            if active_fault is not None:
                torque_off = self._cut_torque(
                    f"halt under a latched fault ({active_fault})", include_leader=True
                )
                return StopResult(kind, reason, torque_off=torque_off)
            if self.guard.estopped:
                return StopResult(kind, reason, torque_off=self._torque_state() == "off")
            follower = self._follower
            hold_confirmed = (
                follower is not None
                and follower.connected
                and follower.torque_state == "on"
                and self._torque_state() == "on"
            )
            if not hold_confirmed:
                # Do not let a repeat Halt return the cached ``holding=True``
                # after the port or torque truth changed between watchdog
                # samples. The sampler performs the same fail-closed cut and
                # fault path used for a live read failure.
                self._sample_hold(halt)
                with self._lock:
                    still_holding = self._halt is halt and self._stop_state == "halted"
                    fault = self._fault
                if not still_holding:
                    torque_off = self._torque_state() == "off"
                    if torque_off and fault is None:
                        return StopResult(kind, reason, torque_off=True)
                    return StopResult(
                        kind,
                        reason,
                        torque_off=torque_off,
                        fallback="the active halt lost torque confirmation",
                    )
            if renewed:
                self._event(
                    "safety.halt",
                    f"halt hold renewed by operator {operator_id}",
                    detail={
                        "reason": reason,
                        "holding": True,
                        "renewed": True,
                        "renewals": halt.renewals,
                        "hold_s": self.settings.halt_hold_seconds,
                    },
                )
            return StopResult(kind, reason, torque_off=False, holding=True)
        return self._hold_in_place(
            reason,
            froze_collapse=froze,
            transition=transition,
            loops_only=loops_only,
        )

    def _hold_in_place(
        self,
        reason: str,
        *,
        froze_collapse: str | None,
        transition: int | None,
        loops_only: bool,
    ) -> StopResult:
        """Goal := freshly read Present on the arm joints, the grip rule on the gripper.

        Falls back to torque-off with a latched fault when the position cannot
        be read or the goal cannot be written within ``stop_io_timeout_s``.
        """

        kind = StopKind.HALT
        with self._lock:
            current = self._stop_transition_is_current_locked(transition)
            session_owns_arm = loops_only and (self._armed or self._safety_state == "arming")
        if not current:
            return StopResult(
                kind,
                reason,
                applied=False,
                torque_off=self._torque_state() == "off",
            )
        if session_owns_arm:
            return StopResult(kind, reason, torque_off=self._torque_state() == "off")
        follower = self._follower
        if follower is None:
            return StopResult(kind, reason, torque_off=self._torque_state() == "off")
        if not self._io_lock.acquire(timeout=self.stop_io_timeout_s):
            return self._halt_fallback(
                reason, f"the motor bus did not come free within {self.stop_io_timeout_s:g} s"
            )
        fallback: str | None = None
        active_fault: str | None = None
        result: StopResult | None = None
        present: dict[str, float] = {}
        commanded: dict[str, float] = {}
        halt: _Halt | None = None
        expires_in = 0.0
        try:
            with self._lock:
                current = self._stop_transition_is_current_locked(transition)
                session_owns_arm = loops_only and (self._armed or self._safety_state == "arming")
            if not current:
                result = StopResult(
                    kind,
                    reason,
                    applied=False,
                    torque_off=self._torque_state() == "off",
                )
            elif session_owns_arm:
                result = StopResult(kind, reason, torque_off=self._torque_state() == "off")
            elif self.guard.estopped:
                result = StopResult(kind, reason, torque_off=self._torque_state() == "off")
            else:
                # Read connection and torque truth only after taking the bus:
                # an arm may be completing torque-enable while Halt waits.
                state = follower.torque_state
                if not follower.connected or state != "on":
                    if state == "off":
                        # Halt never energizes motors that are off (§5).
                        result = StopResult(kind, reason, torque_off=self._torque_state() == "off")
                    else:
                        fallback = (
                            f"follower torque is {state}, so there is no confirmed hold to keep"
                        )
                else:
                    with self._lock:
                        active_fault = self._fault
                    if active_fault is None:
                        started = self._clock()
                        try:
                            present = follower.read_positions()
                            commanded = follower.hold(self._hold_goal(follower, present))
                        except Exception as exc:
                            fallback = f"could not hold position: {exc}"
                        else:
                            elapsed = self._clock() - started
                            if elapsed > self.stop_io_timeout_s:
                                fallback = (
                                    f"the hold took {elapsed:.2f} s "
                                    f"(limit {self.stop_io_timeout_s:g} s)"
                                )
                    if result is None and fallback is None and active_fault is None:
                        now = self._clock()
                        # Publish while the bus is still ours. A re-arm cannot
                        # observe "no stop" after the hold write and take it over
                        # before Halt becomes visible.
                        with self._lock:
                            if self.guard.estopped:
                                result = StopResult(
                                    kind,
                                    reason,
                                    torque_off=self._torque_state() == "off",
                                )
                            elif not self._stop_transition_is_current_locked(transition):
                                result = StopResult(
                                    kind,
                                    reason,
                                    applied=False,
                                    torque_off=self._torque_state() == "off",
                                )
                            elif loops_only and (self._armed or self._safety_state == "arming"):
                                result = StopResult(
                                    kind,
                                    reason,
                                    torque_off=self._torque_state() == "off",
                                )
                            elif self._fault is not None:
                                active_fault = self._fault
                            elif self._stop_state == "collapsing":
                                result = StopResult(kind, reason, torque_off=False, collapsing=True)
                            elif self._stop_state == "halted" and self._halt is not None:
                                result = StopResult(kind, reason, torque_off=False, holding=True)
                            else:
                                halt = _Halt(
                                    since=now,
                                    since_wall=time.time(),
                                    reason=reason,
                                    deadline=now + self.settings.halt_hold_seconds,
                                    last_sample=now,
                                )
                                self._halt = halt
                                self._stop_state = "halted"
                                self._publish_sample_locked(present, commanded)
                                expires_in = self._halt_expiry_locked(halt, now) - now
        except Exception as exc:
            fallback = fallback or f"could not hold position: {exc}"
        finally:
            self._io_lock.release()
        if result is not None:
            return result
        if active_fault is not None:
            # A latched fault is category 0: a Halt never holds under it. The
            # path that latched it may not have cut torque yet; this does.
            torque_off = self._cut_torque(
                f"halt under a latched fault ({active_fault})", include_leader=True
            )
            return StopResult(kind, reason, torque_off=torque_off)
        if fallback is not None:
            return self._halt_fallback(reason, fallback)
        assert halt is not None
        detail: dict[str, Any] = {
            "reason": reason,
            "holding": True,
            "hold_s": self.settings.halt_hold_seconds,
            "expires_in_s": round(expires_in, 2),
            "goal": {joint: round(value, 3) for joint, value in commanded.items()},
            "present": {joint: round(value, 3) for joint, value in present.items()},
        }
        if froze_collapse is not None:
            detail["froze_collapse"] = froze_collapse
        self._event("safety.halt", f"Halt: holding position ({reason})", detail=detail)
        return StopResult(kind, reason, torque_off=False, holding=True)

    def _hold_goal(self, follower: SO101Follower, present: dict[str, float]) -> dict[str, float]:
        goal = {
            joint: float(value)
            for joint, value in present.items()
            if joint in JOINT_NAMES and joint != "gripper"
        }
        if "gripper" in present:
            goal["gripper"] = self._grip_goal(follower, float(present["gripper"]))
        return goal

    def _grip_goal(self, follower: SO101Follower, present: float) -> float:
        """``clamp(last_goal, present - grip_hold_margin, present)`` (design doc §5).

        The gripper never opens past where it is, so a held object is not
        dropped, and keeps at most a small squeeze so an existing grasp holds.
        It is also kept inside the guard's gripper range, but never above
        ``present``.
        """

        last = float(follower.last_goal.get("gripper", present))
        goal = min(max(last, present - self.settings.grip_hold_margin), present)
        low, high = follower.guard.limits.get("gripper", (0.0, 100.0))
        return min(min(max(goal, float(low)), float(high)), present)

    def _halt_fallback(self, reason: str, cause: str) -> StopResult:
        """Category 0 fallback: torque off on every bus, and a latched fault."""

        # Latch in memory before the bus cut or either persistence write. A
        # live lease can therefore never re-arm in the gap after a confirmed
        # cut and leave the faulted arm energized again.
        message = f"Halt could not hold position ({cause}); category-0 torque cut required"
        torque_off = self._fail_closed_runtime_io(
            message,
            source="robot.halt",
            cut_reason=f"halt fallback: {cause}",
            include_leader=True,
        )
        self._event(
            "safety.halt",
            f"Halt fell back to torque-off: {cause}",
            level="critical",
            detail={
                "reason": reason,
                "holding": False,
                "fallback": cause,
                "torque_off": torque_off,
            },
        )
        return StopResult(StopKind.HALT, reason, torque_off=torque_off, fallback=cause)

    def _publish_sample_locked(
        self, observation: dict[str, float], commanded: dict[str, float] | None = None
    ) -> None:
        self._snapshot = {
            "t": time.time(),
            "observation": {
                f"{joint}.pos": value
                for joint, value in observation.items()
                if joint in JOINT_NAMES
            },
            "action": (
                {f"{joint}.pos": value for joint, value in commanded.items()}
                if commanded is not None
                else dict(self._snapshot.get("action", {}))
            ),
            "fps": 0.0,
        }
        self._snapshot_monotonic = self._clock()
        self._sample_seq += 1

    def _clear_halt(self) -> None:
        with self._lock:
            if self._stop_state == "halted":
                self._halt = None
                self._stop_state = "none"

    # --- Collapse: category 1, slow fall to the rest pose -----------------
    def _freeze_collapse(self) -> str | None:
        """An explicit Halt overrides a running Collapse (§5); returns its reason.

        The descent checks ``cancel`` under the bus lock before every write, so
        once this returns it never writes again, and the halt's hold that
        follows is the last goal the servos see.
        """

        with self._lock:
            run = self._collapse if self._stop_state == "collapsing" else None
            if run is None or self._fault is not None:
                # A Collapse under a latched fault (servo health critical) is
                # the fault's reaction: a Halt cannot hold under a fault, so it
                # leaves the descent running. E-stop still preempts it.
                return None
            run.cancel.set()
            run.ended = True
            self._collapse = None
            self._stop_state = "none"
        return run.reason

    def _collapse_stop(
        self,
        reason: str,
        *,
        transition: int | None,
        lease_id: str | None,
        epoch: int | None,
        trip_state: str | None,
        end_lease: bool,
        join_timeout: float,
        fault: str | None,
        expected_halt: _Halt | None,
    ) -> StopResult:
        """Stop every motion source, disarm, then lower the arm to the rest pose.

        The plan (a fresh read of Present and the resolved rest pose) is made
        here, so a bus that is already dead fails this call. The descent then
        runs on its own thread, ``so101-collapse``; a Halt or an E-stop
        preempts it at its next tick.
        """

        kind = StopKind.COLLAPSE
        owner_scoped = lease_id is not None
        global_stop = not owner_scoped and epoch is None
        loop_stopped = True
        with self._lock:
            if not self._stop_transition_is_current_locked(transition) or (
                expected_halt is not None
                and (self._halt is not expected_halt or self._stop_state != "halted")
            ):
                return StopResult(kind, reason, applied=False)
            # _dispatch_stop validated lease/epoch before reserving this token.
            # That reservation survives a later release/expiry: those revoke
            # future authority but cannot cancel an already-authorized Collapse.
        if global_stop and expected_halt is None:
            # The hold timer's Collapse skips this: a halted arm has no loop,
            # and a loop started since took the halt over (checked below).
            self.cancel_calibration()
            loop_stopped = self._stop_loop(join_timeout=join_timeout)
        with self._lock:
            if not self._stop_transition_is_current_locked(transition):
                return StopResult(kind, reason, applied=False)
            if expected_halt is not None and (
                self._halt is not expected_halt or self._stop_state != "halted"
            ):
                return StopResult(kind, reason, applied=False)
            was_armed = self._revoke_authority_locked(
                global_stop=global_stop, end_lease=end_lease, trip_state=trip_state
            )
            if fault is not None:
                self._fault = fault
                self._fault_epoch += 1
                self._safety_state = self._latched_state_locked()
            running = self._stop_state == "collapsing"
        if fault is not None:
            self._record_fault(fault, source="robot.collapse")
        if was_armed:
            self._disarmed_event(kind, reason)
        if not loop_stopped:
            return StopResult(
                kind, reason, loop_stopped=False, torque_off=self._torque_state() == "off"
            )
        with self._lock:
            if not self._stop_transition_is_current_locked(transition):
                return StopResult(
                    kind, reason, applied=False, torque_off=self._torque_state() == "off"
                )
        if self.guard.estopped:
            return StopResult(kind, reason, torque_off=self._torque_state() == "off")
        if running:
            return StopResult(kind, reason, torque_off=False, collapsing=True)
        follower = self._follower
        state: TorqueState = follower.torque_state if follower is not None else "off"
        run = _Collapse(since=self._clock(), since_wall=time.time(), reason=reason, fault=fault)
        if follower is None or not follower.connected or state != "on":
            if state == "off":
                self._clear_halt()
                return StopResult(
                    kind, reason, torque_off=self._torque_state() == "off"
                )  # motors already off: a no-op (§5)
            torque_off = self._collapse_fallback(
                run, "torque_unconfirmed", f"follower torque is {state}; nothing to lower"
            )
            return StopResult(kind, reason, torque_off=torque_off, fallback="torque_unconfirmed")
        with self._lock:
            if self.guard.estopped:
                return StopResult(kind, reason, torque_off=False)
            if not self._stop_transition_is_current_locked(transition):
                return StopResult(
                    kind,
                    reason,
                    applied=False,
                    torque_off=self._torque_state() == "off",
                )
            if self._stop_state == "collapsing":
                return StopResult(kind, reason, torque_off=False, collapsing=True)
            self._halt = None
            self._stop_state = "collapsing"
            self._collapse = run
        plan, cause, detail = self._plan_collapse(run, follower)
        if plan is None:
            if cause is None:  # preempted by an E-stop or a Halt while planning
                return StopResult(kind, reason, torque_off=self._torque_state() == "off")
            torque_off = self._collapse_fallback(run, cause, detail)
            return StopResult(kind, reason, torque_off=torque_off, fallback=cause)
        now = self._clock()
        with self._lock:
            run.plan = plan
            run.started = now
            run.deadline = now + plan.timeout
            run.eta_s = plan.duration
            still_current = self._collapse is run and not run.ended
        if not still_current:
            return StopResult(kind, reason, torque_off=self._torque_state() == "off")
        self._event(
            "safety.collapse.start",
            f"Collapse: lowering to the rest pose ({reason})",
            level="warning",
            detail={
                "reason": reason,
                "rest_source": plan.rest_source,
                "rest_pose": {joint: round(value, 3) for joint, value in plan.target.items()},
                "gripper_goal": None if plan.gripper_goal is None else round(plan.gripper_goal, 3),
                "distance_deg": round(plan.distance, 3),
                "speed_deg_s": self.settings.collapse_speed_deg_s,
                "duration_s": round(plan.duration, 3),
                "timeout_s": round(plan.timeout, 3),
                "fault": fault,
                "warnings": plan.rest_warnings,
            },
        )
        thread = threading.Thread(
            target=self._collapse_descent, args=(run,), name="so101-collapse", daemon=True
        )
        run.thread = thread
        try:
            thread.start()
        except Exception as exc:
            detail = f"could not start the Collapse descent thread: {type(exc).__name__}: {exc}"
            torque_off = self._collapse_fallback(run, "error", detail)
            return StopResult(kind, reason, torque_off=torque_off, fallback="error")
        return StopResult(kind, reason, torque_off=False, collapsing=True)

    def _plan_collapse(
        self, run: _Collapse, follower: SO101Follower
    ) -> tuple[_CollapsePlan | None, str | None, str]:
        """Read Present, seed the guard and resolve the rest pose.

        Returns ``(plan, None, "")``, or ``(None, cause, detail)`` for a
        fallback, or ``(None, None, ...)`` when an E-stop or Halt preempted it.
        Without a resolvable rest pose there is nowhere safe to lower to: the
        cause is ``no_rest_pose`` and the arm is de-energized where it stands
        (a zero-distance descent) with a latched fault.
        """

        rest_error: str | None = None
        try:
            rest = resolve_collapse_rest_pose(self.settings, limits=follower.guard.limits)
        except Exception as exc:
            rest, rest_error = None, f"{type(exc).__name__}: {exc}"
        if not self._io_lock.acquire(timeout=self.stop_io_timeout_s):
            return (
                None,
                "bus_error",
                (f"the motor bus did not come free within {self.stop_io_timeout_s:g} s"),
            )
        try:
            if run.cancel.is_set() or self.guard.estopped:
                return None, None, "preempted"
            present = follower.read_positions()
            follower.guard.seed(present)
        except SafetyViolation as exc:
            return None, "guard_rejection", str(exc)
        except Exception as exc:
            return None, "bus_error", f"could not read the follower: {exc}"
        finally:
            self._io_lock.release()
        if rest is None:
            return (
                None,
                "no_rest_pose",
                (
                    f"no rest pose could be resolved ({rest_error}); the arm was de-energized "
                    "where it stood"
                ),
            )
        arm = [joint for joint in JOINT_NAMES if joint != "gripper" and joint in present]
        target = {joint: float(rest.joints[joint]) for joint in arm}
        distance = max((abs(target[joint] - present[joint]) for joint in arm), default=0.0)
        duration = distance / self.settings.collapse_speed_deg_s
        timeout = self.settings.collapse_timeout_seconds or min(
            duration + COLLAPSE_TIMEOUT_MARGIN_S, COLLAPSE_TIMEOUT_MAX_S
        )
        gripper_goal = (
            self._grip_goal(follower, float(present["gripper"])) if "gripper" in present else None
        )
        return (
            _CollapsePlan(
                start={joint: float(present[joint]) for joint in arm},
                target=target,
                gripper_goal=gripper_goal,
                distance=distance,
                duration=duration,
                timeout=timeout,
                rest_source=rest.source,
                rest_warnings=list(rest.warnings),
            ),
            None,
            "",
        )

    def _collapse_descent(self, run: _Collapse) -> None:
        """The ``so101-collapse`` thread body. Any bug ends in the fallback."""

        try:
            self._run_descent(run)
        except Exception as exc:  # never leave a descent energized and unsupervised
            log.exception("collapse descent failed")
            self._collapse_fallback(run, "error", f"{type(exc).__name__}: {exc}")

    def _run_descent(self, run: _Collapse) -> None:
        """Joint-space interpolation from Present to rest, every joint capped.

        All joints move together, so the joint with the farthest to go runs at
        ``collapse_speed_deg_s`` and the others slower. Each tick goes through
        the SafetyGuard (limits and rate) but not the lease. Paced by deadline
        scheduling on the injected clock (``_Pacer``).

        The cap holds on every tick, also a late one. The trajectory runs on
        its own time, which advances by the real interval since the last tick
        but by at most one period per tick: a tick delayed by a bus wait (up
        to ``stop_io_timeout_s``) or a scheduler stall moves the setpoint one
        period's worth, never the time it lost, so the arm never catches up.
        Each joint's step is also clamped to its share of the cap (speed x
        period, scaled per joint like the plan) from the goal the servos last
        accepted, so a goal the driver clipped is not made up in one jump
        either. Delays therefore make the descent longer, not faster. The
        bound (``run.deadline``) stays on the real clock: a descent delayed
        past it ends in the timeout fallback, torque off with a fault.
        """

        plan, follower = run.plan, self._follower
        if plan is None or run.started is None or run.deadline is None or follower is None:
            self._collapse_fallback(run, "error", "the descent started without a plan")
            return
        period = 1.0 / self.collapse_rate_hz
        speed = self.settings.collapse_speed_deg_s
        eps = min(_COLLAPSE_PROGRESS_EPS_DEG, speed * _COLLAPSE_STALL_WINDOW_S / 4)
        arm = list(plan.target)
        # One tick's largest step per joint: the farthest joint at the cap,
        # the others proportionally slower, as in the plan.
        step_cap = {
            joint: (
                speed * period * abs(plan.target[joint] - plan.start[joint]) / plan.distance
                if plan.distance > 0
                else 0.0
            )
            for joint in arm
        }
        best = plan.distance
        last_progress = run.started
        trajectory_t = 0.0  # descent time along the plan; at most one period per tick
        last_tick = run.started
        last_goal = dict(plan.start)  # what the servos last accepted, per arm joint
        pacer = _Pacer(period, clock=self._clock, sleep=self._sleep)
        while True:
            now = self._clock()
            if run.cancel.is_set() or self.guard.estopped:
                return
            if now >= run.deadline:
                self._collapse_fallback(
                    run,
                    "timeout",
                    f"not at the rest pose within {plan.timeout:.1f} s ({best:.1f} deg to go)",
                )
                return
            trajectory_t += min(max(0.0, now - last_tick), period)
            last_tick = now
            fraction = 1.0 if plan.duration <= 0 else min(1.0, trajectory_t / plan.duration)
            setpoint: dict[str, float] = {}
            for joint in arm:
                planned = plan.start[joint] + (plan.target[joint] - plan.start[joint]) * fraction
                previous, cap = last_goal[joint], step_cap[joint]
                setpoint[joint] = min(max(planned, previous - cap), previous + cap)
            if plan.gripper_goal is not None:
                setpoint["gripper"] = plan.gripper_goal
            if not self._io_lock.acquire(timeout=self.stop_io_timeout_s):
                self._collapse_fallback(
                    run,
                    "bus_error",
                    f"the motor bus did not come free within {self.stop_io_timeout_s:g} s",
                )
                return
            failure: Exception | None = None
            cause = "bus_error"
            try:
                if run.cancel.is_set() or self.guard.estopped:
                    return
                commanded = follower.send_goal(setpoint, dt=period)
                measured = follower.read_positions()
            except SafetyViolation as exc:
                failure, cause = exc, "guard_rejection"
            except Exception as exc:
                failure = exc
            finally:
                self._io_lock.release()
            if failure is not None:
                if not (run.cancel.is_set() or self.guard.estopped):
                    self._collapse_fallback(run, cause, str(failure))
                return
            for joint in arm:
                if joint in commanded:
                    last_goal[joint] = float(commanded[joint])
            error = max((abs(measured[joint] - plan.target[joint]) for joint in arm), default=0.0)
            now = self._clock()
            if error < best - eps:
                best, last_progress = error, now
            stalled = now - last_progress >= _COLLAPSE_STALL_WINDOW_S
            with self._lock:
                if self._collapse is run:
                    run.progress = (
                        max(0.0, min(1.0, 1.0 - error / plan.distance))
                        if plan.distance > 0
                        else 1.0
                    )
                    run.eta_s = (
                        max(0.0, plan.duration - trajectory_t) if fraction < 1.0 else error / speed
                    )
                    self._publish_sample_locked(measured, commanded)
            if fraction >= 1.0 and (
                error <= _COLLAPSE_ARRIVED_DEG or (stalled and error <= _COLLAPSE_STALL_ERROR_DEG)
            ):
                self._collapse_done(run, error, now - run.started)
                return
            if stalled and error > _COLLAPSE_STALL_ERROR_DEG:
                self._collapse_fallback(
                    run,
                    "stall",
                    f"no progress for {_COLLAPSE_STALL_WINDOW_S:g} s with {error:.1f} deg to go",
                )
                return
            pacer.wait()

    def _claim_collapse_end(self, run: _Collapse) -> bool:
        """Exactly one finisher (done, fallback, or the watchdog's backstop)."""

        with self._lock:
            if run.ended:
                return False
            run.ended = True
        run.cancel.set()  # the descent, if it is not the caller, writes no more
        return True

    def _end_collapse_state(self, run: _Collapse) -> None:
        """``stop_state`` leaves "collapsing" only after the finisher's event is written."""

        with self._lock:
            if self._collapse is run:
                self._collapse = None
                self._stop_state = "none"

    def _collapse_done(self, run: _Collapse, final_error: float, duration: float) -> None:
        if not self._claim_collapse_end(run):
            return
        with self._lock:
            run.progress, run.eta_s = 1.0, 0.0
        torque_off = self._cut_torque_pass(f"collapse done ({run.reason})", include_leader=True)
        if run.fault is not None:
            self._retain_global_safety_latch(f"runtime fault retained: {run.fault}")
        plan = run.plan
        self._event(
            "safety.collapse.done",
            (
                "Collapse finished at the rest pose; torque off"
                if torque_off
                else "Collapse reached the rest pose but torque-off is unconfirmed; "
                "cut physical power"
            ),
            level="info" if torque_off else "critical",
            detail={
                "reason": run.reason,
                "duration_s": round(duration, 3),
                "final_error_deg": round(final_error, 3),
                "torque_off": torque_off,
                "rest_source": plan.rest_source if plan is not None else None,
                "fault": run.fault,
            },
        )
        self._end_collapse_state(run)

    def _collapse_fallback(self, run: _Collapse, cause: str, detail: str) -> bool:
        """Immediate torque-off with ``fault`` or ``torque_unconfirmed`` per C2."""

        if not self._claim_collapse_end(run):
            return self._torque_state() == "off"
        torque_off = self._cut_torque_pass(f"collapse fallback ({cause})", include_leader=True)
        message = f"Collapse fell back to torque-off ({cause}): {detail}"
        if torque_off:
            # An unconfirmed cut has latched its own fault (C2).
            self._set_fault(
                f"{run.fault}; {message}" if run.fault else message, source="robot.collapse"
            )
        self._event(
            "safety.collapse.fallback",
            message,
            level="critical",
            detail={
                "reason": run.reason,
                "cause": cause,
                "detail": detail,
                "torque_off": torque_off,
            },
        )
        self._end_collapse_state(run)
        return torque_off

    def _await_collapse(self) -> None:
        """Wait for a running descent, bounded; past the bound cut torque (shutdown)."""

        with self._lock:
            run = self._collapse if self._stop_state == "collapsing" else None
        if run is None:
            return
        timeout = run.plan.timeout if run.plan is not None else COLLAPSE_TIMEOUT_MAX_S
        thread = run.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout + _COLLAPSE_POLICE_MARGIN_S + 1.0)
        with self._lock:
            unfinished = not run.ended
        if unfinished:
            run.cancel.set()
            self._collapse_fallback(run, "timeout", "the descent did not finish within its bound")

    @staticmethod
    def _raise_unless_confirmed(result: StopResult) -> None:
        if not result.applied:
            raise ControlRejected("valid exclusive control lease required")
        if result.kind is StopKind.ESTOP:
            if not result.torque_off:
                raise ControlRejected(
                    "E-STOP is latched but software torque-off failed; cut physical power now"
                )
            if not result.loop_stopped:
                raise ControlRejected(
                    "E-STOP torque-off was confirmed, but the control thread is hung; "
                    "keep physical power isolated and restart the service"
                )
            return
        if not result.safe:
            raise ControlRejected(
                "control disarmed logically but torque-off is unconfirmed; cut physical power"
            )
        if not result.loop_stopped:
            raise ControlRejected(
                "control loop hung; forced torque-off attempted and fault latched"
            )
        if result.fallback is not None:
            raise ControlRejected(
                f"{result.kind.value} fell back to torque-off ({result.fallback}); "
                "a fault is latched"
            )

    # --- exclusive lease + watchdog -----------------------------------
    def _ensure_watchdog(self) -> None:
        with self._lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            self._watchdog_shutdown.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="so101-watchdog", daemon=True
            )
            self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_shutdown.wait(0.05):
            try:
                self._watchdog_tick()
            except Exception:  # supervision must outlive any single failure
                log.exception("watchdog tick failed")

    def _watchdog_tick(self) -> None:
        """One supervision pass.

        Every trip is a HALT through the dispatcher; the hold timer and lost
        supervision turn a halt into a COLLAPSE (design doc §5). All deadlines
        are on the injected clock.
        """

        self._expire_lease_if_needed()
        stale_epoch: int | None = None
        operator_timed_out = False
        elapsed_ms = 0.0
        with self._lock:
            now = self._clock()
            if (
                self.mode in _LOOP_MODES
                and not self._stop.is_set()
                and (
                    self._operation_owner is None
                    or (now - self._operator_heartbeats.get(self._operation_owner, float("-inf")))
                    * 1000
                    > self.operator_timeout_ms
                )
            ):
                self._stop.set()
                self._safety_state = "operator_timeout"
                operator_timed_out = True
            if self._armed and self._last_command_monotonic is not None:
                elapsed_ms = (now - self._last_command_monotonic) * 1000
                if elapsed_ms > self.watchdog_ms:
                    stale_epoch = self._authority_epoch
        if stale_epoch is not None:
            # Epoch-scoped: a re-arm or another stop in between wins.
            result = self._dispatch_stop(
                StopKind.HALT,
                reason="command watchdog",
                epoch=stale_epoch,
                trip_state="watchdog_timeout",
            )
            if result.applied:
                self._event(
                    "control.watchdog_timeout",
                    f"manual command stream became stale; {self._stop_outcome(result)}",
                    level="warning" if result.ok else "critical",
                    detail={"elapsed_ms": round(elapsed_ms, 1)},
                )
        if operator_timed_out:
            # Lost supervision (§5): a Halt. The hold then lasts only the
            # supervision grace period unless an operator comes back, and
            # becomes a Collapse.
            result = self._dispatch_stop(
                StopKind.HALT, reason="operator heartbeat lost", loops_only=True
            )
            self._event(
                "runtime.operator_timeout",
                f"operator heartbeat lost; loop stopped; {self._stop_outcome(result)}",
                level="warning" if result.ok else "critical",
                source="watchdog",
                detail={"timeout_ms": self.operator_timeout_ms},
            )
        self._supervise_stops()

    @staticmethod
    def _stop_outcome(result: StopResult) -> str:
        if result.fallback is not None:
            if result.torque_off:
                return f"the hold failed ({result.fallback}); torque cut and fault latched"
            return f"the hold failed ({result.fallback}); torque-off unconfirmed—cut physical power"
        if result.holding:
            return "holding position (halt)"
        if result.collapsing:
            return "a Collapse is lowering the arm"
        if result.torque_off:
            return "motors are off"
        return "torque-off unconfirmed—cut physical power"

    def _supervise_stops(self) -> None:
        """The hold timer, the hold's live samples and the descent backstop."""

        sample = False
        expired: str | None = None
        with self._lock:
            now = self._clock()
            halt = self._halt if self._stop_state == "halted" else None
            collapse = self._collapse if self._stop_state == "collapsing" else None
            if halt is not None:
                supervised, _stale_at = self._supervision_locked(now)
                expiry = self._halt_expiry_locked(halt, now)
                if now >= expiry:
                    if not supervised and expiry < halt.deadline:
                        expired = (
                            "supervision lost: no fresh operator heartbeat for "
                            f"{self.settings.supervision_grace_seconds:g} s while halted"
                        )
                    else:
                        expired = f"halt hold expired after {self.settings.halt_hold_seconds:g} s"
                elif now - halt.last_sample >= _HOLD_SAMPLE_PERIOD_S:
                    halt.last_sample = now
                    sample = True
        if halt is not None and expired is not None:
            self._dispatch_stop(StopKind.COLLAPSE, reason=expired, expected_halt=halt)
        elif halt is not None and sample:
            self._sample_hold(halt)
        if collapse is not None:
            self._police_collapse(collapse, now)

    def _sample_hold(self, halt: _Halt) -> None:
        """Read the held arm for telemetry; a dead bus is an E-stop-equivalent fault.

        Skips (never waits) while another transfer holds the bus.
        """

        follower = self._follower
        if follower is None:
            self._fail_closed_runtime_io(
                "lost the follower object while holding a halt",
                source="robot.halt",
                cut_reason="halt hold lost the follower",
                include_leader=True,
                torque_unconfirmed=True,
            )
            return
        connected = follower.connected
        torque_state = follower.torque_state
        aggregate_torque = self._torque_state()
        if not connected or torque_state != "on" or aggregate_torque != "on":
            if aggregate_torque == "off":
                # The hold ended under a confirmed torque cut.
                with self._lock:
                    if self._halt is halt:
                        self._halt, self._stop_state = None, "none"
                return
            connection = "connected" if connected else "disconnected"
            self._fail_closed_runtime_io(
                "lost the follower while holding a halt: "
                f"the port is {connection} and torque is {aggregate_torque}",
                source="robot.halt",
                cut_reason="halt hold lost torque confirmation",
                include_leader=True,
            )
            return
        if not self._io_lock.acquire(blocking=False):
            return
        failure: Exception | None = None
        measured: dict[str, float] = {}
        try:
            with self._lock:
                current = self._halt is halt
            if not current or self.guard.estopped:
                return
            measured = follower.read_positions()
        except Exception as exc:
            failure = exc
        finally:
            self._io_lock.release()
        if failure is not None:
            self._fail_closed_runtime_io(
                f"lost the follower while holding a halt: {failure}",
                source="robot.halt",
                cut_reason="halt hold lost the bus",
                include_leader=True,
            )
            return
        with self._lock:
            if self._halt is halt:
                self._publish_sample_locked(measured)

    def _police_collapse(self, run: _Collapse, now: float) -> None:
        """Backstop for a descent thread stuck in the driver: cut torque past its bound."""

        if run.deadline is None or now <= run.deadline + _COLLAPSE_POLICE_MARGIN_S:
            return
        run.cancel.set()
        self._collapse_fallback(
            run, "timeout", "the descent thread did not finish within its bound (stuck in I/O)"
        )

    # --- servo health (docs/bring-up.md "Servo health monitor") ------------
    def _new_health_arm(self, arm: HealthArmName) -> _HealthArm:
        try:
            thresholds: HealthThresholds | None = HealthThresholds.from_settings(self.settings, arm)
            error = None
        except Exception as exc:  # cross-field settings errors (e.g. margins out of order)
            thresholds, error = None, f"invalid health thresholds for the {arm}: {exc}"
        state = _HealthArm(arm=arm, thresholds=thresholds, config_error=error)
        if error is not None:
            state.inactive_reason = error
        return state

    def _report_health_config_errors(self) -> None:
        for state in self._health.values():
            if state.config_error is not None:
                log.error("servo health monitoring is off: %s", state.config_error)
                self._event(
                    "robot.health.config_error",
                    f"servo health monitoring is off: {state.config_error}",
                    level="error",
                    source="robot.health",
                    detail={"arm": state.arm},
                )

    def _ensure_health_thread(self) -> None:
        with self._lock:
            if self._health_thread is not None and self._health_thread.is_alive():
                return
            self._health_shutdown.clear()
            self._health_thread = threading.Thread(
                target=self._health_loop, name="so101-health", daemon=True
            )
            self._health_thread.start()

    def _health_loop(self) -> None:
        # The wait is only the pass cadence, like the watchdog's; when a poll
        # is due is decided on the injected clock in _health_tick.
        while not self._health_shutdown.wait(_HEALTH_TICK_S):
            try:
                self._health_tick()
            except Exception:  # monitoring must outlive any single failure
                log.exception("servo health tick failed")

    def _health_inactive_reason(
        self, arm: HealthArmName, device: SO101Follower | SO101Leader | None
    ) -> str | None:
        """Why ``arm`` is not polled right now, or None when it should be."""

        state = self._health[arm]
        if state.config_error is not None:
            return state.config_error
        if self._health_shutdown.is_set():
            return "the server is shutting down"
        if current_backend(self.settings) == "sim":
            return (
                f"SO101_BACKEND=sim: the simulated {arm} has no servo health registers, so "
                "monitoring is off (no synthetic values, no false alarms)"
            )
        if device is None:
            return f"the {arm} bus has not been opened in this server"
        if not callable(getattr(device.bus, "read_health", None)):
            return f"the {arm} bus has no servo health registers"
        if self.guard.estopped:
            return "the E-stop is latched: the motors are off and the E-stop owns the bus"
        if self.cal_in_progress and self.cal_arm == arm:
            return f"calibration owns the {arm} bus"
        if not device.connected:
            if arm == "follower":
                return (
                    "the follower bus is closed (motors off); polling starts when the server "
                    "opens it: arming, a loop, a Halt or a Collapse"
                )
            return "the leader bus is closed; teleop and recording open it"
        return None

    def _health_tick(self) -> None:
        """One servo-health pass on ``so101-health``: poll the most overdue arm.

        An arm is polled while the manager has its bus open (see
        :meth:`_health_inactive_reason`). The first poll of an activation is
        due at once, so a servo that is already unhealthy shows at arm time;
        then every ``1 / health_poll_hz`` seconds of the injected clock, with
        no burst catch-up after a stall. At most one arm is read per pass, so
        a control loop never waits behind two health reads back to back; the
        other due arm is read on the next pass, 50 ms later.
        """

        now = self._clock()
        period = 1.0 / self.settings.health_poll_hz
        due: list[tuple[float, HealthArmName, SO101Follower | SO101Leader, _HealthArm]] = []
        for arm in _HEALTH_ARMS:
            with self._lock:
                device: SO101Follower | SO101Leader | None = (
                    self._follower if arm == "follower" else self._leader
                )
            reason = self._health_inactive_reason(arm, device)
            state = self._health[arm]
            with self._lock:
                if reason is not None or device is None:
                    state.active, state.inactive_reason, state.monitor = False, reason, None
                    continue
                if state.monitor is None:
                    assert state.thresholds is not None  # no config error, checked above
                    state.monitor = ServoHealthMonitor(
                        partial(self._take_health_read, state),
                        thresholds=state.thresholds,
                        poll_hz=self.settings.health_poll_hz,
                        on_warn=partial(self._on_health_warn, arm),
                        on_critical=partial(self._on_health_critical, arm),
                        name=arm,
                    )
                    state.active, state.inactive_reason = True, None
                    state.next_due = now
                    state.activations += 1
                if now >= state.next_due:
                    due.append((state.next_due, arm, device, state))
        if not due:
            return
        _, arm, device, state = min(due, key=lambda entry: entry[0])
        with self._lock:
            state.next_due = now + period
        self._health_poll(arm, device, state)

    @staticmethod
    def _take_health_read(state: _HealthArm) -> dict[str, ServoHealthSample]:
        """The monitor's read_fn: hand over what :meth:`_health_poll` read."""

        read, state.pending = state.pending, None
        if isinstance(read, BaseException):
            raise read
        if read is None:
            raise RuntimeError("no servo health read is pending")
        return read

    def _health_poll(
        self, arm: HealthArmName, device: SO101Follower | SO101Leader, state: _HealthArm
    ) -> None:
        """One health read under the bus lock, graded after the lock is released.

        Interleaves with a running control loop: the read waits at most
        ``health_io_wait_s`` for the bus, which the loop holds only for its
        own transfers, and is skipped and counted when the bus stays busy.
        The monitor's warn and critical handlers run after the release, so a
        critical verdict's Collapse can take the bus at once.
        """

        wait = self.health_io_wait_s
        if not self._io_lock.acquire(timeout=wait):
            with self._lock:
                state.skipped_busy += 1
                state.last_skip = (
                    f"skipped a poll: the motor bus stayed busy for {wait * 1000:.0f} ms"
                )
                state.next_due = min(state.next_due, self._clock() + _HEALTH_RETRY_S)
            return
        read: dict[str, ServoHealthSample] | BaseException
        try:
            # The bus may have closed, calibration started or an E-stop
            # latched while this waited; the next pass records why.
            if self.guard.estopped or not device.connected or self.cal_in_progress:
                return
            started = self._clock()
            try:
                read = dict(device.bus.read_health())  # type: ignore[attr-defined]
            except Exception as exc:  # a failed read is data for the monitor
                read = exc
            read_ms = (self._clock() - started) * 1000.0
        finally:
            self._io_lock.release()
        with self._lock:
            monitor = state.monitor
            if monitor is None:
                return
            state.pending = read
            state.last_read_ms = round(read_ms, 3)
            state.max_read_ms = max(state.max_read_ms, state.last_read_ms)
        snapshot = monitor.poll_once()
        if snapshot is None:
            return
        self._record_health(state, snapshot)
        previous = snapshot.previous_level
        if snapshot.evaluation.level == "ok" and previous in ("warn", "critical"):
            self._event(
                "robot.health.ok",
                f"{arm} servo health is back to ok (was {previous})",
                source="robot.health",
                detail={"arm": arm, "previous_level": previous},
            )

    def _record_health(self, state: _HealthArm, snapshot: HealthSnapshot) -> None:
        """Keep ``snapshot`` for status and the API. Idempotent per snapshot.

        The handlers record it before they act, so a critical verdict is in
        the status while its Collapse runs.
        """

        with self._lock:
            if state.latest is snapshot:
                return
            now = self._clock()
            state.latest = snapshot
            state.polled_mono = now
            state.polls += 1
            if snapshot.error is None:
                state.last_good, state.last_good_mono = snapshot, now
            else:
                state.read_failures += 1

    @staticmethod
    def _health_reasons_json(snapshot: HealthSnapshot) -> list[dict[str, Any]]:
        return [reason.model_dump(mode="json") for reason in snapshot.evaluation.reasons]

    @staticmethod
    def _health_messages(snapshot: HealthSnapshot, *, levels: tuple[str, ...]) -> list[str]:
        return [reason.message for reason in snapshot.evaluation.reasons if reason.level in levels]

    def _on_health_warn(self, arm: HealthArmName, snapshot: HealthSnapshot) -> None:
        """WARN: an event and the status summary. Never a stop."""

        self._record_health(self._health[arm], snapshot)
        messages = self._health_messages(snapshot, levels=("warn",))
        self._event(
            "robot.health.warn",
            f"{arm} servo health warning: " + ("; ".join(messages) or "see /api/robot/health"),
            level="warning",
            source="robot.health",
            detail={
                "arm": arm,
                "reasons": self._health_reasons_json(snapshot),
                "consecutive_failures": snapshot.consecutive_failures,
            },
        )

    def _on_health_critical(self, arm: HealthArmName, snapshot: HealthSnapshot) -> None:
        """CRITICAL: Collapse + fault through :meth:`servo_health_critical`.

        The monitor calls this on every critical poll. It dispatches unless
        the stop would change nothing: an E-stop owns the arm, or this health
        fault is already latched and its Collapse is running or the follower
        is already off.
        """

        state = self._health[arm]
        self._record_health(state, snapshot)
        messages = self._health_messages(snapshot, levels=("critical",))
        detail = f"{arm}: " + ("; ".join(messages) or "critical servo health")
        with self._lock:
            follower = self._follower
            fault = self._fault or ""
            redundant = self.guard.estopped or (
                fault.startswith("servo health critical")
                and (
                    self._stop_state == "collapsing"
                    or follower is None
                    or bus_torque_state(follower.bus) != "on"
                )
            )
            if not redundant:
                state.critical_dispatches += 1
        if redundant:
            return
        self.servo_health_critical(
            detail,
            health={
                "arm": arm,
                "reasons": self._health_reasons_json(snapshot),
                "consecutive_failures": snapshot.consecutive_failures,
                "source": snapshot.source,
            },
        )

    def _health_summary_locked(self, now: float) -> dict[str, Any]:
        """The compact health view in status and telemetry. Holds ``_lock``.

        ``level`` is the worst verdict over the arms being polled (null before
        their first poll or when none is); ``age_s`` is the oldest of their
        last full samples; ``stale`` flags one older than three poll periods.
        """

        period = 1.0 / self.settings.health_poll_hz
        arms: dict[str, str | None] = {}
        levels: list[str] = []
        ages: list[float] = []
        reasons: list[str] = []
        for arm, state in self._health.items():
            latest = state.latest if state.active else None
            arms[arm] = latest.evaluation.level if latest is not None else None
            if latest is None:
                continue
            levels.append(latest.evaluation.level)
            if state.last_good_mono is not None:
                ages.append(max(0.0, now - state.last_good_mono))
            reasons.extend(
                f"{arm}: {message}"
                for message in self._health_messages(latest, levels=("warn", "critical"))
            )
        age = max(ages) if ages else None
        return {
            "level": worst_level(levels) if levels else None,
            "active": any(state.active for state in self._health.values()),
            "arms": arms,
            "age_s": None if age is None else round(age, 2),
            "stale": age is not None and age > _HEALTH_STALE_PERIODS * period,
            "reasons": reasons[:_HEALTH_SUMMARY_REASONS],
        }

    def _health_arm_view(self, state: _HealthArm, now: float) -> dict[str, Any]:
        latest, good = state.latest, state.last_good
        motors: dict[str, Any] = {}
        if good is not None:
            for motor, sample in good.samples.items():
                verdict = good.evaluation.motors.get(motor)
                motors[motor] = {
                    "level": verdict.level if verdict is not None else None,
                    "reasons": (
                        [reason.model_dump(mode="json") for reason in verdict.reasons]
                        if verdict is not None
                        else []
                    ),
                    "sample": sample.model_dump(mode="json"),
                }
        basis = good if good is not None else latest
        source = basis.source if basis is not None else None
        return {
            "arm": state.arm,
            "active": state.active,
            "inactive_reason": state.inactive_reason,
            "level": latest.evaluation.level if latest is not None else None,
            "source": source,
            "polled_at": self._iso(latest.taken_at) if latest is not None else None,
            "sampled_at": self._iso(good.taken_at) if good is not None else None,
            "sample_age_s": (
                round(max(0.0, now - state.last_good_mono), 3)
                if state.last_good_mono is not None
                else None
            ),
            "consecutive_failures": latest.consecutive_failures if latest is not None else 0,
            "error": latest.error if latest is not None else None,
            "reasons": (
                [reason.model_dump(mode="json") for reason in latest.evaluation.reasons]
                if latest is not None
                else []
            ),
            "motors": motors,
            "thresholds": (
                state.thresholds.model_dump(mode="json") if state.thresholds is not None else None
            ),
            "polls": state.polls,
            "read_failures": state.read_failures,
            "skipped_busy": state.skipped_busy,
            "last_skip": state.last_skip,
            "last_read_ms": state.last_read_ms,
            "max_read_ms": round(state.max_read_ms, 3),
            "activations": state.activations,
            "critical_dispatches": state.critical_dispatches,
        }

    def health_snapshot(self) -> dict[str, Any]:
        """``GET /api/robot/health``: the last sample per servo, verdicts and thresholds.

        JSON-safe. Per arm: whether it is polled (or why not), the last
        poll's verdict and reasons, the last full sample of every servo with
        its own verdict, the thresholds, the sample age, and the poll, failure
        and skip counters. The HIL runner stores it in every probe bundle.
        """

        with self._lock:
            now, wall = self._clock(), time.time()
            summary = self._health_summary_locked(now)
            arms = {arm: self._health_arm_view(state, now) for arm, state in self._health.items()}
            poller = self._health_thread
        return {
            "active": summary["active"],
            "level": summary["level"],
            "stale": summary["stale"],
            "poller_running": bool(poller is not None and poller.is_alive()),
            "poll_hz": self.settings.health_poll_hz,
            "io_wait_ms": round(self.health_io_wait_s * 1000.0, 3),
            "server_time": self._iso(wall),
            "arms": arms,
            "unit_provenance": {
                name: provenance.model_dump(mode="json")
                for name, provenance in UNIT_PROVENANCE.items()
            },
        }

    def _expire_lease_if_needed(self, *, wait: bool = False) -> None:
        """Expire a lapsed lease with a HALT.

        One expiry runs at a time. A caller that finds another in flight
        returns at once, unless ``wait`` is set: then it returns only after
        that expiry's Halt has settled, so what it reads next is the held arm.
        """

        if not self._expiry_lock.acquire(blocking=wait):
            return
        try:
            with self._lock:
                lease = self._lease
                if lease is None or self._clock() < lease.deadline:
                    return
                expired_id, source, tag = lease.id, lease.source, lease.tag
            result = self._dispatch_stop(
                StopKind.HALT,
                reason="lease expired",
                lease_id=expired_id,
                end_lease=True,
                trip_state="lease_expired",
            )
        finally:
            self._expiry_lock.release()
        if result.applied:
            self._event(
                "control.lease_expired",
                f"control lease expired for {source}",
                level="warning",
                source=source,
                detail={"lease_tag": tag},
            )

    def acquire_lease(self, source: str, ttl_seconds: float) -> dict:
        self._ensure_watchdog()
        self._expire_lease_if_needed()
        with self._lock:
            if self._any_stop_in_progress_locked():
                raise ControlConflict("a Halt or Collapse is being established")
            if self._lease is not None:
                raise ControlConflict(
                    f"control is leased to {self._lease.source!r} until "
                    f"{self._lease.expires_at.isoformat()}"
                )
            if self.mode is not Mode.IDLE or self.cal_in_progress:
                raise ControlConflict(f"robot is busy ({self.mode.value})")
            now = datetime.now(timezone.utc)
            lease = ControlLease(
                id=uuid.uuid4().hex,
                source=source,
                ttl_seconds=ttl_seconds,
                expires_at=now + timedelta(seconds=ttl_seconds),
                deadline=self._clock() + ttl_seconds,
            )
            self._lease = lease
            self._last_command_audit_monotonic = 0.0
            self._commands_since_audit = 0
        self._event(
            "control.lease_acquired",
            f"exclusive control acquired by {source}",
            source=source,
            detail={"lease_tag": lease.tag, "ttl_seconds": ttl_seconds},
        )
        return lease.owner_view()

    def holds_lease(self, lease_id: str) -> bool:
        """True when ``lease_id`` names the live lease. Never raises."""

        with self._lock:
            lease = self._lease
            return lease is not None and lease.id == lease_id and self._clock() < lease.deadline

    def engaged_position(self, joint: str) -> float | None:
        """Where ``joint`` was when the current armed session engaged torque.

        ``None`` when nothing is armed. A re-arm or any stop starts over.
        """

        with self._lock:
            if not self._armed or self._engaged_pose is None:
                return None
            epoch, pose = self._engaged_pose
            if epoch != self._authority_epoch:
                return None
            return pose.get(joint)

    def heartbeat_lease(self, lease_id: str) -> dict:
        self._ensure_watchdog()
        self._expire_lease_if_needed()
        with self._lock:
            lease = self._require_lease_locked(lease_id)
            now = datetime.now(timezone.utc)
            lease.expires_at = now + timedelta(seconds=lease.ttl_seconds)
            lease.deadline = self._clock() + lease.ttl_seconds
            answer = lease.owner_view()
            follower = self._follower if self._armed else None
        if follower is not None and follower.connected:
            # safety-B9: the renewal above is what the heartbeat is for. A busy
            # bus skips this refresh instead of failing the lease.
            if not self._io_lock.acquire(timeout=self.request_io_timeout_s):
                log.debug("lease heartbeat: motor bus busy; observation refresh skipped")
                return answer
            try:
                try:
                    observation = follower.observation()
                finally:
                    self._io_lock.release()
            except Exception as exc:
                self._fail_closed_runtime_io(
                    exc,
                    source="control.heartbeat",
                    cut_reason="heartbeat observation failed",
                )
                raise ControlRejected(f"could not refresh armed observation: {exc}") from exc
            with self._lock:
                self._snapshot = {
                    "t": time.time(),
                    "observation": observation,
                    "action": dict(self._snapshot.get("action", {})),
                    "fps": 0.0,
                }
                self._snapshot_monotonic = self._clock()
                self._sample_seq += 1
        return answer

    def release_lease(self, lease_id: str) -> None:
        self._expire_lease_if_needed()
        with self._lock:
            source = self._require_lease_locked(lease_id).source
        result = self._dispatch_stop(
            StopKind.HALT, reason="lease released", lease_id=lease_id, end_lease=True
        )
        if not result.applied:
            raise ControlRejected("valid exclusive control lease required")
        self._event(
            "control.lease_released",
            f"exclusive control released by {source}",
            source=source,
            detail={"lease_tag": lease_tag(lease_id)},
        )
        if not result.safe:
            raise ControlRejected("lease revoked but torque-off is unconfirmed; cut physical power")

    def _require_lease_locked(self, lease_id: str) -> ControlLease:
        lease = self._lease
        # A lapsed lease is invalid even before the watchdog has expired it (C5).
        if lease is None or lease.id != lease_id or self._clock() >= lease.deadline:
            raise ControlRejected("valid exclusive control lease required")
        return lease

    def control_state(self) -> dict:
        """Return the current control view without expiring leases or doing I/O."""

        with self._lock:
            if self.guard.estopped:
                safety = "estopped"
            elif self._fault:
                safety = "fault"
            else:
                safety = self._safety_state
            return {
                "armed": self._armed,
                "safety_state": safety,
                "lease": self._lease.public() if self._lease else None,
                "watchdog_ms": self.watchdog_ms,
                "last_command_at": self._last_command_at,
            }

    def safety_authority_generation(self) -> tuple[int, int, int]:
        """Monotonic (E-stop, fault, authority) generations for durable gates.

        A caller compares snapshots around a blocking disk or audit write. A
        transient E-stop/fault still changes the tuple after its latch is reset,
        so it cannot disappear between two ordinary status reads.
        """

        with self._estop_generation_lock:
            estop_generation = self._estop_generation
        with self._lock:
            return estop_generation, self._fault_epoch, self._authority_epoch

    def observe_with_lease(self, lease_id: str) -> dict:
        """Read a fresh pose with torque disabled so a UI can seed targets."""

        self._expire_lease_if_needed()
        with self._lock:
            self._require_lease_locked(lease_id)
            if self.guard.estopped:
                raise ControlRejected("E-STOP is latched")
            if self._fault:
                raise ControlRejected(f"runtime fault: {self._fault}")
            if self.mode is not Mode.IDLE or self.cal_in_progress:
                raise ControlConflict("robot must be idle to observe")
            self._refuse_observe_while_armed_locked()
            follower = self.follower()
        self._acquire_io_for_request("observe")
        try:
            with self._lock:
                # An arm may have landed while this request waited for the bus.
                self._refuse_observe_while_armed_locked()
        except ControlConflict:
            self._io_lock.release()
            raise
        try:
            try:
                # A halted arm is energized and holding: read it and keep the
                # hold. Disconnecting would drop it (design doc §5).
                holding = follower.connected and follower.torque_state == "on"
                if not follower.connected:
                    follower.connect(enable_torque=False)
                measured = follower.read_positions()
                if not holding:
                    follower.disconnect()
            finally:
                self._io_lock.release()
        except Exception as exc:
            # Locked and read back, as in arm_control (safety-B2). Torque is
            # cut before either fault-persistence write can block.
            self._fail_closed_runtime_io(exc, source="control.observe", cut_reason="observe failed")
            raise ControlRejected(f"could not read follower pose: {exc}") from exc
        with self._lock:
            self._require_lease_locked(lease_id)
            observation = {f"{joint}.pos": value for joint, value in measured.items()}
            captured_at = datetime.now(timezone.utc)
            self._snapshot = {
                "t": captured_at.timestamp(),
                "observation": observation,
                "action": {},
                "fps": 0.0,
            }
            self._snapshot_monotonic = self._clock()
            self._sample_seq += 1
        return {"observation": observation, "captured_at": captured_at}

    def _refuse_observe_while_armed_locked(self) -> None:
        # Observe reads with torque off, which would drop an armed arm while
        # the state still says armed (safety-B12, docs/audit/v0.1/safety.md).
        if self._armed or self._safety_state == "arming":
            raise ControlConflict(
                "control is armed or arming; disarm first to observe with torque off"
            )

    def _arm_authority_lost_locked(
        self,
        lease_id: str,
        authority_epoch: int,
        *,
        torque_phase: bool,
        holding: bool = False,
    ) -> str | None:
        """Why an arm in progress may no longer write, or None. Holds ``_lock``.

        ``arm_control`` checks this each time it takes the bus lock, before any
        write: the lease (named and not lapsed), the authority epoch, the
        E-stop, a fault, calibration, the mode and the stop state. In the
        torque phase the arm has set ``_armed`` and a command of its own lease
        may have made the mode MANUAL; ``holding`` is whether it found a
        Halt's hold to take over.
        """

        lease = self._lease
        if lease is None or lease.id != lease_id:
            return "the lease was released or replaced"
        if self._clock() >= lease.deadline:
            return "the lease lapsed"
        if self._authority_epoch != authority_epoch:
            return "a stop revoked control authority"
        if self.guard.estopped:
            return "E-STOP is latched"
        if self._fault is not None:
            return f"runtime fault: {self._fault}"
        if self._external_stop_in_progress_locked():
            return "a Halt or Collapse is being established"
        if self.cal_in_progress:
            return "calibration is in progress"
        if self._stop_state == "collapsing":
            return "a Collapse is lowering the arm"
        if torque_phase:
            if not self._armed:
                return "control was disarmed"
            if self.mode not in (Mode.IDLE, Mode.MANUAL):
                return f"the robot is busy ({self.mode.value})"
            if self._stop_state == "halted" and not holding:
                return "a Halt took the arm"
        elif self.mode is not Mode.IDLE:
            return f"the robot is busy ({self.mode.value})"
        return None

    def arm_control(self, lease_id: str) -> dict:
        self._expire_lease_if_needed()
        with self._lock:
            lease = self._require_lease_locked(lease_id)
            if self.guard.estopped:
                raise ControlRejected("E-STOP is latched; explicitly reset it first")
            if self._fault:
                raise ControlRejected(f"runtime fault must be cleared first: {self._fault}")
            if self._any_stop_in_progress_locked():
                raise ControlConflict("a Halt or Collapse is being established")
            if self.mode is not Mode.IDLE:
                raise ControlConflict(f"robot is busy ({self.mode.value})")
            if self.cal_in_progress:
                raise ControlConflict("calibration is in progress")
            if self._stop_state == "collapsing":
                raise ControlConflict(
                    "a Collapse is lowering the arm to rest; wait for it to finish (or E-stop)"
                )
            follower = self.follower()
            source = lease.source
            authority_epoch = self._authority_epoch
            state_before_arming = self._safety_state
            self._safety_state = "arming"
        try:
            # Write ahead before even the torque-off connect/open: a symlink
            # target that changes at open must retain the pre-open identity's
            # durable marker, and disk I/O must not hold the motor lock.
            self._prepare_global_energized_intent(f"RobotManager manual arm ({source})")
        except Exception as exc:
            self._fail_closed_runtime_io(
                exc,
                source="control.arm_intent",
                cut_reason="arm energized-intent failed",
            )
            raise ControlRejected(f"could not prepare follower torque-enable: {exc}") from exc
        try:
            self._acquire_io_for_request("arm")
        except ControlBusy:
            with self._lock:
                if self._safety_state == "arming":
                    self._safety_state = state_before_arming
            self._retire_failed_enable_intent_if_safe(
                follower,
                "manual arm bus-busy setup confirmed torque off",
            )
            raise
        # A stop may have landed while this waited for the bus: check again
        # before any bus I/O (connect primes Goal_Position), with the bus held.
        with self._lock:
            lost = self._arm_authority_lost_locked(lease_id, authority_epoch, torque_phase=False)
            if lost is not None and self._safety_state == "arming":
                self._safety_state = state_before_arming
        if lost is not None:
            self._io_lock.release()
            raise ControlRejected(f"control authority changed while arming ({lost})")
        try:
            try:
                # A halted arm is already energized and holding. Re-arming takes
                # it over without a torque write or a goal write (design doc §5:
                # no toggle, no jump).
                holding = follower.connected and follower.torque_state == "on"
                if not follower.connected:
                    follower.connect(enable_torque=False)
                measured = follower.read_positions()
            finally:
                self._io_lock.release()
        except Exception as exc:
            # A bounded torque-off with read-back, under the bus lock like every
            # stop. The old cleanup disconnected without the lock, racing any
            # stop on the same serial port, and an unanswered servo still read
            # as torque-off confirmed (safety-B2, docs/audit/v0.1/safety.md).
            self._fail_closed_runtime_io(exc, source="control.arm", cut_reason="arm failed")
            raise ControlRejected(f"could not arm follower: {exc}") from exc

        authority_revoked = False
        with self._lock:
            current_lease = self._lease
            authority_revoked = (
                current_lease is None
                or current_lease.id != lease_id
                or self.guard.estopped
                or self._fault is not None
                or self._external_stop_in_progress_locked()
                or self.mode is not Mode.IDLE
                or self.cal_in_progress
                or self._authority_epoch != authority_epoch
            )
            if authority_revoked and self._safety_state == "arming":
                # Nothing armed, so a trip shown before arming stays visible (C5).
                self._safety_state = state_before_arming
            if not authority_revoked:
                self.guard.seed(measured)
                self._armed = True
                self._safety_state = "armed"
                # The deadman starts once torque is on, below (devex-B4).
                self._last_command_monotonic = None
                self._snapshot = {
                    "t": time.time(),
                    "observation": {f"{joint}.pos": value for joint, value in measured.items()},
                    "action": {},
                    "fps": 0.0,
                }
                self._snapshot_monotonic = self._clock()
                self._sample_seq += 1
        if authority_revoked:
            # Bounded and never raises; an unconfirmed torque-off latches. A
            # Halt/Collapse that owns this revocation keeps the arm; category
            # 0 or an unowned fresh connection is cut.
            with self._lock:
                cut_revoked_arm = self._arm_cleanup_requires_torque_cut_locked(
                    holding=holding, authority_epoch=authority_epoch
                )
            if cut_revoked_arm:
                self._cut_torque("arm revoked")
            raise ControlRejected("control authority changed while arming")
        try:
            self._acquire_io_for_request("arm")
        except ControlBusy:
            self._dispatch_stop(StopKind.HALT, reason="arm: motor bus busy", epoch=authority_epoch)
            raise
        # The gap between the check above and this bus lock is where a Halt,
        # Collapse, global stop or fault can complete. Check again with the bus
        # held, before any torque or goal write: a revoked arm enables nothing,
        # and the stop that revoked it owns the arm.
        with self._lock:
            lost = self._arm_authority_lost_locked(
                lease_id, authority_epoch, torque_phase=True, holding=holding
            )
            unrevoked = lost is not None and self._authority_epoch == authority_epoch
        if lost is not None:
            self._io_lock.release()
            with self._lock:
                stop_owns_arm = not self._arm_cleanup_requires_torque_cut_locked(
                    holding=holding, authority_epoch=authority_epoch
                )
            if unrevoked and not stop_owns_arm:
                # Nothing revoked this arm yet (a lease that lapsed while it
                # waited, or an E-stop latch whose bookkeeping is pending).
                self._dispatch_stop(StopKind.HALT, reason=f"arm: {lost}", epoch=authority_epoch)
            with self._lock:
                cut_revoked_arm = self._arm_cleanup_requires_torque_cut_locked(
                    holding=holding, authority_epoch=authority_epoch
                )
            if cut_revoked_arm:
                # The port this arm opened to read the pose, torque never
                # enabled: torque-off with read-back, then the port closes, as
                # for any revoked arm. No torque-enable and no goal write.
                self._cut_torque("arm revoked")
            raise ControlRejected(
                f"control authority changed before torque-enable ({lost}); nothing was enabled"
            )
        try:
            try:
                # The durable write above can block. Revalidate under the bus
                # lock before any torque write; a stop that landed meanwhile wins.
                with self._lock:
                    lost_after_intent = self._arm_authority_lost_locked(
                        lease_id,
                        authority_epoch,
                        torque_phase=True,
                        holding=holding,
                    )
                if lost_after_intent is not None:
                    raise ControlRejected(
                        "control authority changed while recording energized intent "
                        f"({lost_after_intent})"
                    )
                if holding:
                    measured_at_enable = follower.resume_hold()
                else:
                    measured_at_enable = follower.enable_torque()
            finally:
                self._io_lock.release()
        except Exception as exc:
            with self._lock:
                cancelled = self._authority_epoch != authority_epoch
            if cancelled:
                with self._lock:
                    cut_revoked_arm = self._arm_cleanup_requires_torque_cut_locked(
                        holding=holding, authority_epoch=authority_epoch
                    )
                if cut_revoked_arm:
                    self._deenergize_external()
                raise ControlRejected("control authority changed while enabling torque") from exc
            self._fail_closed_runtime_io(
                exc, source="control.arm_torque", cut_reason="arm torque-enable failed"
            )
            raise ControlRejected(f"could not enable follower torque: {exc}") from exc
        with self._lock:
            current_lease = self._lease
            authority_revoked = (
                current_lease is None
                or current_lease.id != lease_id
                or not self._armed
                or self.guard.estopped
                or self._fault is not None
                or self._external_stop_in_progress_locked()
                or self._authority_epoch != authority_epoch
            )
            if not authority_revoked:
                # The armed session takes over a halt's hold (design doc §5).
                if self._stop_state == "halted":
                    self._halt, self._stop_state = None, "none"
                # Arming itself starts the local deadman timer, once torque is
                # on, so a slow bus write cannot eat the operator's window
                # (devex-B4, docs/audit/v0.1/devex.md). A client that never
                # sends its first command is halted within 500 ms.
                self._last_command_monotonic = self._clock()
                self._engaged_pose = (authority_epoch, dict(measured_at_enable))
                self._snapshot = {
                    "t": time.time(),
                    "observation": {
                        f"{joint}.pos": value for joint, value in measured_at_enable.items()
                    },
                    "action": {},
                    "fps": 0.0,
                }
                self._snapshot_monotonic = self._clock()
                self._sample_seq += 1
        if authority_revoked:
            with self._lock:
                cut_revoked_arm = self._arm_cleanup_requires_torque_cut_locked(
                    holding=holding, authority_epoch=authority_epoch
                )
            if cut_revoked_arm:
                self._deenergize_external()
            raise ControlRejected("control authority changed while enabling torque")
        self._event(
            "control.armed",
            f"manual control armed by {source}",
            source=source,
            detail={"lease_tag": lease_tag(lease_id)},
        )
        return self.control_state()

    def preview_targets(self, targets: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(targets) - set(JOINT_NAMES))
        if unknown:
            raise ControlRejected(f"unknown joints: {', '.join(unknown)}")
        bounded: dict[str, float] = {}
        for joint, raw_value in targets.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ControlRejected(f"non-finite target for {joint}")
            # A joint resting outside the envelope can hold there (safety-B10).
            low, high = self.guard.effective_limits(joint)
            bounded[joint] = min(max(value, low), high)
        if not bounded:
            raise ControlRejected("at least one recognized joint target is required")
        return bounded

    def validate_command_freshness(self, sent_at: datetime, ttl_ms: int) -> None:
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        age_ms = (
            datetime.now(timezone.utc) - sent_at.astimezone(timezone.utc)
        ).total_seconds() * 1000
        max_age_ms = min(ttl_ms, self.watchdog_ms)
        if age_ms < -1000:
            raise ControlRejected("command timestamp is too far in the future")
        if age_ms > max_age_ms:
            raise ControlRejected(f"stale command ({age_ms:.0f} ms old; limit {max_age_ms} ms)")

    def _acquire_io_for_request(self, what: str) -> None:
        """Take the bus lock for a request, or raise ControlBusy (safety-B9)."""

        if not self._io_lock.acquire(timeout=self.request_io_timeout_s):
            raise ControlBusy(
                f"motor bus is busy; {what} not sent "
                f"(waited {self.request_io_timeout_s * 1000:.0f} ms); retry"
            )

    def _check_command_authority_locked(
        self, lease_id: str, source: str, sequence: int
    ) -> tuple[ControlLease, SO101Follower]:
        lease = self._require_lease_locked(lease_id)
        if source != lease.source:
            raise ControlRejected("command source does not own the lease")
        if not self._armed:
            raise ControlRejected("control is disarmed")
        if self.guard.estopped:
            raise ControlRejected("E-STOP is engaged")
        if self._fault:
            raise ControlRejected(f"runtime fault: {self._fault}")
        if self._external_stop_in_progress_locked():
            raise ControlRejected("a Halt or Collapse is being established")
        if sequence <= lease.last_sequence:
            raise ControlRejected(
                f"out-of-order command sequence {sequence}; last was {lease.last_sequence}"
            )
        follower = self._follower
        if follower is None or not follower.connected:
            raise ControlRejected("follower is not connected; arm control again")
        return lease, follower

    def send_external_command(
        self,
        *,
        lease_id: str,
        source: str,
        sequence: int,
        sent_at: datetime,
        ttl_ms: int,
        deadman: bool,
        targets: dict[str, float],
    ) -> dict[str, float]:
        self._expire_lease_if_needed()
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        if not deadman:
            raise ControlRejected("deadman must be held for every motion command")
        bounded = self.preview_targets(targets)
        # safety-B9 (docs/audit/v0.1/safety.md): reject an invalid request before
        # it waits for the bus, so requests can never pile up behind a wedged
        # driver call and starve the server.
        self.validate_command_freshness(sent_at, ttl_ms)
        with self._lock:
            self._check_command_authority_locked(lease_id, source, sequence)

        # Serialize bus traffic separately from state. E-stop/watchdog can still
        # acquire the manager lock while a driver call is slow or hung.
        self._acquire_io_for_request("command")
        failure: Exception | None = None
        fault_message: str | None = None
        try:
            now = datetime.now(timezone.utc)
            # Everything is checked again: the wait may have made it stale.
            self.validate_command_freshness(sent_at, ttl_ms)
            with self._lock:
                lease, follower = self._check_command_authority_locked(lease_id, source, sequence)
                # Reserve the sequence before releasing state lock so another
                # queued request cannot execute the same or an older frame.
                lease.last_sequence = sequence
                authority_epoch = self._authority_epoch
            try:
                commanded = follower.send_goal(bounded)
                observation = follower.observation()
            except Exception as exc:
                # Latch while this request still holds the bus, so no queued
                # command runs before the fault is visible.
                fault_message, _fault_epoch = self._latch_fault_in_memory(exc)
                failure = exc
        finally:
            self._io_lock.release()
        if failure is not None:
            # The same broadcast torque-off with read-back as every stop. The
            # old cleanup was LeRobot's per-motor disable, which stops at the
            # first silent servo: the healthy ones kept their torque, and
            # torque_unconfirmed stayed false (safety-B2,
            # docs/audit/v0.1/safety.md).
            self._cut_torque(f"command failed ({source})")
            assert fault_message is not None
            self._record_fault(fault_message, source=source)
            raise ControlRejected(f"command failed: {failure}") from failure
        with self._lock:
            current_lease = self._lease
            authority_revoked = (
                current_lease is None
                or current_lease.id != lease_id
                or not self._armed
                or self.guard.estopped
                or self._fault
                or self._external_stop_in_progress_locked()
                or self._authority_epoch != authority_epoch
            )
            if not authority_revoked:
                self._last_command_at = now
                accepted_at = self._clock()
                self._last_command_monotonic = accepted_at
                self.mode = Mode.MANUAL
                self._snapshot = {
                    "t": now.timestamp(),
                    "observation": observation,
                    "action": {f"{joint}.pos": value for joint, value in commanded.items()},
                    "fps": 0.0,
                }
                self._snapshot_monotonic = self._clock()
                self._sample_seq += 1
                self._commands_since_audit += 1
                audit_command = accepted_at - self._last_command_audit_monotonic >= 1.0
                accepted_since_audit = self._commands_since_audit
                if audit_command:
                    self._last_command_audit_monotonic = accepted_at
                    self._commands_since_audit = 0
            else:
                audit_command = False
                accepted_since_audit = 0
        if authority_revoked:
            # The stop that revoked authority owns the arm: a Halt holds it, a
            # Collapse lowers it. Only category 0 that is not already owned by
            # an active stop needs this command's late write followed by a
            # confirmed torque-off, which it may not have got while this held
            # the bus.
            with self._lock:
                category0 = self._arm_cleanup_requires_torque_cut_locked(
                    holding=True, authority_epoch=authority_epoch
                )
            if category0:
                self._deenergize_external(f"command after revocation ({source})")
            raise ControlRejected(
                "command completed after control authority was revoked; follower is stopping"
            )
        if audit_command:
            # Persist a bounded 1 Hz sample, not every high-rate control frame.
            self._event(
                "control.command",
                f"accepted command {sequence} from {source}",
                source=source,
                detail={
                    "sequence": sequence,
                    "targets": commanded,
                    "accepted_since_last_event": accepted_since_audit,
                },
            )
        return commanded

    def execute_approved_targets(self, lease_id: str, targets: dict[str, float]) -> dict:
        self._expire_lease_if_needed()
        with self._lock:
            lease = self._require_lease_locked(lease_id)
            sequence = lease.last_sequence + 1
            source = lease.source
        commanded = self.send_external_command(
            lease_id=lease_id,
            source=source,
            sequence=sequence,
            sent_at=datetime.now(timezone.utc),
            ttl_ms=self.watchdog_ms,
            deadman=True,
            targets=targets,
        )
        return {"sequence": sequence, "targets": commanded}

    def _deenergize_external(self, reason: str = "disarm") -> bool:
        """Torque off the follower unless a loop owns it; True when confirmed."""

        with self._lock:
            loop_active = self._thread is not None and self._thread.is_alive()
        if loop_active:
            # The loop owns the bus and its teardown disconnects the follower.
            return False
        return self._cut_torque(reason)

    def _known_buses(self, include_leader: bool) -> list[tuple[str, SO101Follower | SO101Leader]]:
        with self._lock:
            candidates: list[tuple[str, SO101Follower | SO101Leader | None]] = [
                ("follower", self._follower)
            ]
            if include_leader:
                candidates.append(("leader", self._leader))
        return [(name, device) for name, device in candidates if device is not None]

    def _live_buses(self, include_leader: bool) -> list[tuple[str, SO101Follower | SO101Leader]]:
        return [
            (name, device) for name, device in self._known_buses(include_leader) if device.connected
        ]

    def _closed_unconfirmed_buses(
        self, include_leader: bool
    ) -> list[tuple[str, SO101Follower | SO101Leader]]:
        """Closed buses whose torque is not known to be off (safety-B2).

        A bus closes its port even when the torque-off before the close was
        not confirmed, so "not connected" never means "torque off". Its
        ``torque_state`` does, and it survives the close.
        """
        return [
            (name, device)
            for name, device in self._known_buses(include_leader)
            if not device.connected and getattr(device.bus, "torque_state", "off") != "off"
        ]

    def _cut_torque(
        self,
        reason: str,
        *,
        include_leader: bool = False,
        io_timeout: float = 0.5,
        latch_failure: bool = True,
    ) -> bool:
        """Category 0: torque off on every bus the manager owns; never raises.

        A running Collapse descent is cancelled first, so it never writes again,
        and a Halt's hold or a Collapse is over once the cut is done, whatever
        its outcome (a failed cut latches its own fault). See
        :meth:`_cut_torque_pass` for the pass itself.
        """

        with self._lock:
            collapse = self._collapse
        if collapse is not None:
            collapse.cancel.set()
        try:
            result = self._cut_torque_pass(
                reason,
                include_leader=include_leader,
                io_timeout=io_timeout,
                latch_failure=latch_failure,
            )
            if latch_failure:
                with self._lock:
                    if self.guard.estopped:
                        latch = "E-stop"
                    elif self._fault is not None:
                        latch = f"runtime fault: {self._fault}"
                    else:
                        latch = None
                if latch is not None:
                    self._retain_global_safety_latch(f"{latch} retained after torque-off")
            return result
        finally:
            with self._lock:
                if self._collapse is not None:
                    self._collapse.ended = True
                self._collapse = None
                self._halt = None
                self._stop_state = "none"

    def _cut_torque_pass(
        self,
        reason: str,
        *,
        include_leader: bool = False,
        io_timeout: float = 0.5,
        latch_failure: bool = True,
    ) -> bool:
        """Disable torque on every bus the manager owns; never raises.

        Each live bus gets one broadcast Torque_Enable=0 and a per-motor
        read-back, then an acknowledged per-motor write for any motor that
        did not read 0, so a single unanswered servo cannot delay the rest.
        The follower's pass is its ``disconnect()``, which always closes the
        port. A bus whose port closed with its torque-off unconfirmed counts
        as a failure until a later pass confirms it. Mock and simulated
        actuators sit behind the same bus interface.

        Returns True only when torque off is confirmed on every bus. Anything
        less normally latches a fault and the ``torque_unconfirmed`` flag,
        which only a later confirmed torque-off clears (safety-B2,
        docs/audit/v0.1/safety.md). A deliberate fault reset passes
        ``latch_failure=False`` so it can compare-and-clear only the fault it
        started with; the reset itself latches any failed retry after that
        comparison.
        """

        with self._lock:
            recovery_epoch = self._recovery_epoch

        if not self._live_buses(include_leader):
            stale = self._closed_unconfirmed_buses(include_leader)
            with self._lock:
                flagged = self._torque_unconfirmed
            if stale and not flagged and latch_failure:
                names = ", ".join(name for name, _device in stale)
                self._mark_torque_unconfirmed(
                    f"torque-off was not confirmed on {names} before its port closed "
                    f"({reason}); cut physical power before recovery",
                    event="safety.force_deenergize_failed",
                    source=reason,
                    devices={name for name, _device in stale},
                )
            return not (stale or flagged)
        if not self._io_lock.acquire(timeout=io_timeout):
            if latch_failure:
                affected = {
                    name
                    for name, _device in (
                        self._live_buses(include_leader)
                        + self._closed_unconfirmed_buses(include_leader)
                    )
                } or ({"follower", "leader"} if include_leader else {"follower"})
                self._mark_torque_unconfirmed(
                    "torque-off unconfirmed because motor I/O did not yield; "
                    "cut physical power before recovery",
                    event="safety.force_deenergize_io_timeout",
                    source=reason,
                    devices=affected,
                )
            return False
        confirmed_buses: list[str] = []
        failures: list[str] = []
        failed_devices: set[str] = set()
        try:
            handled: set[str] = set()
            for name, device in self._live_buses(include_leader):
                handled.add(name)
                if name == "follower":
                    try:
                        device.disconnect()
                    except TorqueOffUnconfirmed as exc:
                        failures.append(str(exc))
                        failed_devices.add(name)
                        continue
                    except Exception as exc:
                        failures.append(f"follower: {exc}")
                        failed_devices.add(name)
                        continue
                    if device.connected:
                        failures.append("follower: the port is still open after disconnect")
                        failed_devices.add(name)
                        continue
                    confirmed_buses.append(name)
                elif torque_off_all(device.bus):
                    confirmed_buses.append(name)
                else:
                    failures.append(f"{name}: torque-off was not confirmed")
                    failed_devices.add(name)
            for name, _device in self._closed_unconfirmed_buses(include_leader):
                if name not in handled:
                    failures.append(f"{name}: torque-off was not confirmed before its port closed")
                    failed_devices.add(name)
        finally:
            self._io_lock.release()
        if failures:
            if latch_failure:
                self._mark_torque_unconfirmed(
                    f"forced torque-disable failed ({reason}): {'; '.join(failures)}",
                    event="safety.force_deenergize_failed",
                    source=reason,
                    devices=failed_devices,
                )
            return False
        confirmed = set(confirmed_buses)
        with self._lock:
            if self._recovery_epoch != recovery_epoch:
                # A newer fault landed while this pass was in I/O. It owns the
                # recovery truth even when it names a bus read earlier here.
                remaining_recovery = set(self._recovery_devices)
            else:
                remaining_recovery = self._recovery_devices - confirmed
            clear_epoch = self._recovery_epoch
        if remaining_recovery:
            message = (
                "torque-off was not confirmed for every recovery device "
                f"({', '.join(sorted(remaining_recovery))}; {reason})"
            )
            if latch_failure:
                self._mark_torque_unconfirmed(
                    message,
                    event="safety.force_deenergize_incomplete",
                    source=reason,
                    devices=remaining_recovery,
                )
            else:
                with self._lock:
                    self._torque_unconfirmed = True
                self._persist_latches()
            return False
        if not self._clear_global_energized_recovery(
            f"confirmed torque-off: {reason}",
            devices=confirmed,
            expected_recovery_epoch=clear_epoch,
        ):
            message = (
                "torque-off was confirmed, but durable host-global energized state could "
                f"not be cleared ({reason})"
            )
            if latch_failure:
                self._mark_torque_unconfirmed(
                    message,
                    event="safety.global_energized_clear_failed",
                    source=reason,
                    devices=set(confirmed_buses),
                )
            else:
                with self._lock:
                    self._torque_unconfirmed = True
                self._persist_latches()
            return False
        with self._lock:
            if self._recovery_epoch != clear_epoch:
                newer_recovery = True
                self._torque_unconfirmed = True
                was_unconfirmed = had_energized_intent = True
            else:
                newer_recovery = False
                was_unconfirmed = self._torque_unconfirmed
                had_energized_intent = self._energized_recovery
                self._torque_unconfirmed = False
                self._energized_recovery = False
                self._energized_at = None
                self._energized_by = None
                self._recovery_devices.difference_update(confirmed)
                self._recovery_epoch += 1
        if newer_recovery:
            self._persist_latches()
            return False
        if was_unconfirmed or had_energized_intent:
            self._persist_latches()
        self._event(
            "safety.force_deenergized",
            f"torque disabled on {', '.join(confirmed_buses)}",
            level="warning",
            source=reason,
            detail={"buses": confirmed_buses},
        )
        return True

    def _reopen_stale(self, reason: str, *, io_timeout: float = 0.5) -> None:
        """Reopen each bus that closed with its torque-off unconfirmed.

        Only the bus is opened: a device ``connect`` would prime
        Goal_Position, and a recovery path must never write goals. A
        successful open leaves torque off on every motor that answered it; the
        torque-off pass that follows confirms it by read-back.
        """

        stale = self._closed_unconfirmed_buses(include_leader=True)
        if not stale:
            return
        if not self._io_lock.acquire(timeout=io_timeout):
            log.error("torque-off retry (%s): motor I/O did not yield", reason)
            return
        failures: list[str] = []
        try:
            for name, device in stale:
                try:
                    device.bus.connect()
                except Exception as exc:
                    failures.append(f"{name}: {exc}")
        finally:
            self._io_lock.release()
        for failure in failures:
            message = f"torque-off retry could not reopen the {failure}"
            log.error(message)
            self._event("safety.torque_off_retry_failed", message, level="critical", source=reason)

    def _retry_cut(self, reason: str) -> bool:
        """A second torque-off on every bus, through fresh ports where they closed."""
        self._reopen_stale(reason)
        return self._cut_torque(reason, include_leader=True)

    def _release_ports(self) -> list[str]:
        """Release every bus the torque-off retry could not confirm.

        Only after the operator acknowledged a physical power cut. Open ports
        close with no torque write: the retry's handle may be dead (an
        unplugged adapter stays dead after a replug), and keeping it open would
        make every later arm fail on it until a restart. Ports that already
        closed with torque-off unconfirmed stop reporting it, so the next stop
        does not latch the fault again (safety-B2, docs/audit/v0.1/safety.md).
        Returns the buses released. Raises ControlBusy while the bus is held.
        """

        if not (
            self._live_buses(include_leader=True)
            or self._closed_unconfirmed_buses(include_leader=True)
        ):
            return []
        self._acquire_io_for_request("port release")
        try:
            buses = self._live_buses(include_leader=True) + self._closed_unconfirmed_buses(
                include_leader=True
            )
            for _name, device in buses:
                release_port(device.bus)
        finally:
            self._io_lock.release()
        return [name for name, _device in buses]

    def _mark_torque_unconfirmed(
        self,
        message: str,
        *,
        event: str,
        source: str,
        devices: set[str] | None = None,
    ) -> None:
        scope = set(devices or {"follower", "leader"})
        global_error = self._write_global_recovery_intent(message, scope)
        if global_error is not None:
            message += (
                "; host-global energized intent is unsaved "
                f"({type(global_error).__name__}: {global_error})"
            )
        with self._lock:
            self._torque_unconfirmed = True
            if current_backend(self.settings) != "real":
                self._add_recovery_devices_locked(scope)
            self._fault = message
            self._fault_epoch += 1
            self._armed = False
            self._authority_epoch += 1
            self._last_command_monotonic = None
            self._safety_state = self._latched_state_locked()
        self._persist_latches()
        self._event(event, message, level="critical", source=source)

    def _retry_torque_off(self, reason: str, *, io_timeout: float = 0.5) -> bool:
        """A real torque-off with read-back on every bus (C2). True when confirmed.

        A closed follower port (after a restart, a failed connect, or a
        torque-off that could not be confirmed before the close) is reopened
        first, and so is a leader port that closed unconfirmed. Only the bus
        is opened: ``SO101Follower.connect`` would prime Goal_Position, and a
        recovery path must never write goals. A failed pass is returned to
        ``clear_fault`` without replacing the fault it is compare-and-clearing;
        that caller then latches the failure unless a newer fault already won.
        """

        with self._lock:
            recovery_devices = set(self._recovery_devices)
        follower = self.follower()
        stale_names = {
            name for name, _device in self._closed_unconfirmed_buses(include_leader=True)
        }
        targets: list[tuple[str, Any]] = []
        if "follower" in recovery_devices or follower.connected or "follower" in stale_names:
            targets.append(("follower", follower))
        if (
            "leader" in recovery_devices
            or "leader" in stale_names
            or bool(self._leader is not None and self._leader.connected)
        ):
            targets.append(("leader", self.leader()))
        closed_targets = [(name, device) for name, device in targets if not device.connected]
        if closed_targets:
            if not self._io_lock.acquire(timeout=io_timeout):
                log.error("torque-off retry (%s): motor I/O did not yield", reason)
                return False
            try:
                for name, device in closed_targets:
                    try:
                        device.bus.connect()
                    except Exception as exc:
                        message = f"torque-off retry could not open the {name} bus: {exc}"
                        log.error(message)
                        self._event(
                            "safety.torque_off_retry_failed",
                            message,
                            level="critical",
                            source=reason,
                        )
                        return False
            finally:
                self._io_lock.release()
        self._reopen_stale(reason, io_timeout=io_timeout)
        return self._cut_torque(
            reason,
            include_leader=True,
            io_timeout=io_timeout,
            latch_failure=False,
        )

    # --- idle bus diagnostics (bring-up preflight) -----------------------
    def _diagnostic_refusal_locked(self, target: DiagnosticTarget) -> str | None:
        """Why the manager will not lend ``target``'s bus now. Holds ``_lock``."""

        if self.guard.estopped:
            return "the E-stop is latched; inspect the arm and reset it first"
        if self._stop_transitions:
            return "a Halt or Collapse is being established"
        if self._armed or self._safety_state == "arming":
            return "manual control is armed; disarm and release the lease first"
        if self._stop_state == "halted":
            return "a Halt holds the arm with its motors on; Collapse it to the rest pose first"
        if self._stop_state == "collapsing":
            return "a Collapse is lowering the arm; wait for it to finish"
        if self.cal_in_progress:
            return "calibration is running; wait for it to finish"
        if self.mode is not Mode.IDLE or (self._thread is not None and self._thread.is_alive()):
            return f"the robot is busy ({self.mode.value}); stop it first"
        device = self._follower if target == "follower" else self._leader
        if device is not None and bus_torque_state(device.bus) == "on":
            return f"the {target} is energized"
        return None

    def _diagnostic_preempted(self) -> None:
        """Runs before every lent transfer: an E-stop ends the diagnostic."""

        if self.guard.estopped:
            raise BusDiagnosticRefused("an E-stop preempted the bus diagnostic")

    def run_idle_bus_diagnostic(
        self,
        target: DiagnosticTarget,
        fn: Callable[[Any], _T],
        *,
        timeout_s: float | None = None,
        opener: Callable[[str], Any] | None = None,
    ) -> _T:
        """Run a read-only diagnostic against one idle motor bus; return ``fn``'s result.

        The manager stays the only bus owner: ``fn`` runs on the caller's
        thread while the manager holds ``_io_lock``, and receives a read-only
        view of ``target``'s register bus (``ping``, ``read``, ``sync_read``,
        ``motors``; any write or torque method raises ``AttributeError``). An
        E-stop that latches during ``fn`` fails its next read at once, and the
        result is then discarded.

        - ``target``: ``"follower"`` or ``"leader"``.
        - ``timeout_s``: the longest wait for the bus lock; by default
          ``request_io_timeout_s`` (0.25 s). Busy: ``ControlBusy`` (HTTP 503),
          nothing read.
        - ``opener``: opens the port when the manager has it closed (the idle
          case); by default a bare LeRobot ``FeetechMotorsBus`` with
          ``connect(handshake=False)``
          (``so101.bringup.buses.open_feetech_bus``), on the real backend
          only. The port is closed again with ``disconnect(disable_torque=False)``,
          which writes nothing. A bus the manager already has open is lent
          as it is and stays open.

        Refuses with ``BusDiagnosticRefused`` (``ControlConflict``, HTTP 409),
        before and again after the wait for the bus, while E-stopped, armed or
        arming, holding a Halt, collapsing, running a loop, calibrating, with
        ``target`` energized, under the sim backend, on the mock backend
        without an ``opener``, or when the open bus has no register reads to
        lend. A port that does not open raises ``BusDiagnosticOpenError``.
        """

        if target not in _DIAGNOSTIC_TARGETS:
            raise ValueError(f"target must be 'follower' or 'leader', not {target!r}")
        wait = self.request_io_timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(wait) or wait < 0:
            raise ValueError("timeout_s must be finite and non-negative")
        backend = current_backend(self.settings)
        if backend == "sim":
            raise BusDiagnosticRefused("SO101_BACKEND=sim has no serial bus to diagnose")
        if opener is None and backend != "real":
            raise BusDiagnosticRefused(
                f"the {backend} backend has no serial bus; pass a rehearsal opener"
            )
        self._expire_lease_if_needed()
        with self._lock:
            refusal = self._diagnostic_refusal_locked(target)
        if refusal is not None:
            raise BusDiagnosticRefused(f"bus diagnostics need an idle robot: {refusal}")
        if not self._io_lock.acquire(timeout=wait):
            raise ControlBusy(
                f"motor bus is busy; diagnostic not run (waited {wait * 1000:.0f} ms); retry"
            )
        opened: Any | None = None
        try:
            with self._lock:
                # Anything may have changed while this waited for the bus.
                refusal = self._diagnostic_refusal_locked(target)
                device: SO101Follower | SO101Leader | None = (
                    self._follower if target == "follower" else self._leader
                )
            if refusal is not None:
                raise BusDiagnosticRefused(f"bus diagnostics need an idle robot: {refusal}")
            if device is not None and device.connected:
                raw = getattr(device.bus, "register_bus", None)
                if raw is None:
                    raise BusDiagnosticRefused(
                        f"the {target} bus is open in the server and has no register reads to lend"
                    )
            else:
                port = (
                    device.port
                    if device is not None
                    else (
                        self.settings.follower_port
                        if target == "follower"
                        else self.settings.leader_port
                    )
                )
                try:
                    opened = (opener or _open_diagnostic_bus)(port)
                except Exception as exc:
                    raise BusDiagnosticOpenError(f"{type(exc).__name__}: {exc}") from exc
                raw = opened
            result = fn(_LentBus(raw, self._diagnostic_preempted))
            if self.guard.estopped:
                raise BusDiagnosticRefused(
                    "an E-stop preempted the bus diagnostic; its result was discarded"
                )
            return result
        finally:
            if opened is not None:
                try:
                    opened.disconnect(disable_torque=False)
                except Exception as exc:  # the port is ours to close; never a torque write
                    log.warning("bus diagnostic: closing the %s port raised: %s", target, exc)
            self._io_lock.release()

    # --- legacy loop lifecycle -----------------------------------------
    def _start_thread(
        self,
        mode: Mode,
        target: Callable[[], None],
        operator_id: str,
        *,
        active_model: str | None = None,
        prepare: Callable[[], None] | None = None,
    ) -> None:
        self._expire_lease_if_needed()
        with self._lock:
            if self.guard.estopped:
                raise RuntimeError("E-STOP is latched; explicitly reset it first")
            if self._fault:
                raise RuntimeError(f"runtime fault: {self._fault}")
            if self._stop_transitions:
                raise RuntimeError("a Halt or Collapse is being established")
            if self._stop_state == "collapsing":
                raise RuntimeError("a Collapse is lowering the arm to rest; wait for it to finish")
            if self._lease is not None or self._armed:
                raise RuntimeError("external control lease is active; release it first")
            if self.cal_in_progress:
                raise RuntimeError("calibration is in progress")
            if self._camera_preview_thread is not None:
                raise RuntimeError("stop camera preview before starting a robot operation")
            if self._episode_saving:
                raise RuntimeError("a recorded episode is still finalizing")
            if self._thread and self._thread.is_alive():
                raise RuntimeError(f"busy: already in {self.mode.value} mode")
            if self.mode is not Mode.IDLE:
                raise RuntimeError(f"busy: already in {self.mode.value} mode")
            self.require_operator_heartbeat(operator_id)
            if prepare is not None:
                prepare()
            self._stop.clear()
            # A new loop takes over a halt's hold (design doc §5); its connect
            # primes Goal_Position from the measured pose.
            if self._stop_state == "halted":
                self._halt, self._stop_state = None, "none"
            self.mode = mode
            self._operation_owner = operator_id
            self._active_model = active_model

            def run() -> None:
                failed = False
                fault_message: str | None = None
                failure_to_log: Exception | None = None
                stopped_by_estop = False
                try:
                    target()
                except Exception as exc:  # hardware faults must become visible
                    failed = True
                    failure_to_log = exc
                    if isinstance(exc, SafetyViolation) and self.guard.estopped:
                        # The latched guard rejected the loop's next frame: that
                        # is the E-stop working, not a second fault.
                        stopped_by_estop = True
                    else:
                        fault_message, _fault_epoch = self._latch_fault_in_memory(exc)
                finally:
                    # A loop that ends because a stop asked it to leaves the
                    # follower energized: the Halt (or Collapse) that stopped
                    # it takes the arm over, so an episode end never sags
                    # (design doc §5). A loop that failed, or ended under an
                    # E-stop or a fault, gets category 0: confirmed torque-off
                    # with read-back on every bus, since its teardown may have
                    # left a port open or closed unconfirmed (safety-B2,
                    # docs/audit/v0.1/safety.md). With every bus confirmed off
                    # and closed, that does no bus I/O.
                    with self._lock:
                        category0 = failed or self.guard.estopped or self._fault is not None
                    if category0:
                        self._cut_torque(f"{mode.value} loop exit", include_leader=True)
                    if fault_message is not None:
                        self._record_fault(fault_message, source=mode.value)
                    if failure_to_log is not None:
                        if stopped_by_estop:
                            log.info("%s loop stopped by E-stop: %s", mode.value, failure_to_log)
                        else:
                            log.error(
                                "%s control loop failed",
                                mode.value,
                                exc_info=(
                                    type(failure_to_log),
                                    failure_to_log,
                                    failure_to_log.__traceback__,
                                ),
                            )
                    recorder_to_abort: EpisodeRecorder | None = None
                    with self._lock:
                        if self._thread is threading.current_thread():
                            self._thread = None
                        if self._operation_owner == operator_id:
                            self._operation_owner = None
                        if not self._episode_saving:
                            recorder_to_abort = self._recorder
                            self._recorder = None
                        self.mode = Mode.IDLE
                        self._active_model = None
                        self._episode_active = False
                    if recorder_to_abort is not None:
                        try:
                            incomplete = recorder_to_abort.abort(
                                f"{mode.value} loop ended before episode save"
                            )
                        except Exception as exc:
                            self._event(
                                "recording.incomplete_preserve_failed",
                                str(exc),
                                level="error",
                                source="recording",
                            )
                        else:
                            if incomplete is not None:
                                self._event(
                                    "recording.incomplete_preserved",
                                    "incomplete episode preserved after loop exit",
                                    level="warning",
                                    source="recording",
                                    detail={"path": str(incomplete)},
                                )

            self._thread = threading.Thread(target=run, name=f"so101-{mode.value}", daemon=True)
            self._thread.start()
        self._event("runtime.started", f"{mode.value} loop started", source=mode.value)

    def stop(self, join_timeout: float = 3.0) -> bool:
        """End the legacy loop with a loops-only HALT; True once it exited.

        The eval runner calls this between episodes. As in v0.1 it leaves an
        armed external session alone.
        """

        return self._dispatch_stop(
            StopKind.HALT, reason="loop stop", loops_only=True, join_timeout=join_timeout
        ).loop_stopped

    def _stop_loop(self, join_timeout: float = 3.0) -> bool:
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
        alive = bool(thread and thread.is_alive())
        recorder_to_abort: EpisodeRecorder | None = None
        with self._lock:
            if alive:
                self._fault = (
                    f"{self.mode.value} loop did not stop within {join_timeout:.1f} seconds"
                )
                self._fault_epoch += 1
                self._authority_epoch += 1
                self._safety_state = self._latched_state_locked()
            else:
                self._thread = None
                self.mode = Mode.IDLE
                self._operation_owner = None
                self._active_model = None
                if not self._episode_saving:
                    recorder_to_abort = self._recorder
                    self._recorder = None
            self._episode_active = False
        if alive:
            # A wedged loop is a category-0 fallback. Cut before preserving an
            # incomplete episode or writing either latch/event store: all of
            # those may block on disk.
            self._cut_torque("stop-timeout", include_leader=True)
        if recorder_to_abort is not None:
            try:
                incomplete = recorder_to_abort.abort("control loop stopped before episode save")
            except Exception as exc:
                self._event(
                    "recording.incomplete_preserve_failed",
                    f"could not preserve incomplete episode: {exc}",
                    level="error",
                    source="recording",
                )
            else:
                if incomplete is not None:
                    self._event(
                        "recording.incomplete_preserved",
                        "incomplete episode preserved for diagnosis",
                        level="warning",
                        source="recording",
                        detail={"path": str(incomplete)},
                    )
        if alive:
            self._persist_latches()
            self._event(
                "runtime.stop_timeout",
                self._fault or "control loop stop timed out",
                level="error",
            )
        return not alive

    def soft_stop(self, reason: str = "global soft stop") -> bool:
        """Global HALT: end any loop or calibration and disarm manual control.

        A thin wrapper over :meth:`request_stop` (C11 item 3); data-learn's
        routers call it. True when the loop exited and torque-off is confirmed.
        """

        result = self.request_stop(StopKind.HALT, reason=reason)
        self._event(
            "runtime.soft_stop",
            "global soft stop requested",
            source="operator",
            detail={"reason": reason, "stop_kind": StopKind.HALT.value},
        )
        return result.ok

    # --- teleop / recording loop ---------------------------------------
    def start_teleop(self, fps: int = 30, *, operator_id: str) -> None:
        if fps <= 0 or fps > 240:
            raise RuntimeError("fps must be between 1 and 240")
        self._start_thread(
            Mode.TELEOP,
            lambda: self._control_loop(fps, record=False),
            operator_id,
            prepare=lambda: setattr(self, "_rec_cfg", None),
        )

    def start_recording(self, cfg: RecordingConfig, *, operator_id: str) -> None:
        self._require_direct_camera_capture()
        if cfg.fps <= 0 or cfg.fps > 240:
            raise RuntimeError("fps must be between 1 and 240")
        if current_backend(self.settings) == "sim":
            cfg = self._sim_recording_config(cfg)

        def prepare() -> None:
            self._rec_cfg = cfg
            self._episode_active = False
            self._episode_frames = 0

        self._start_thread(
            Mode.RECORDING,
            lambda: self._control_loop(cfg.fps, record=True),
            operator_id,
            prepare=prepare,
        )

    def _sim_recording_config(self, cfg: RecordingConfig) -> RecordingConfig:
        """Record ``observation.environment_state`` under sim, as ``so101 sim demos`` does.

        The layout is the sim task's ``env_state_names``. A dataset that
        already holds episodes with another layout is refused: one dataset,
        one feature layout.
        """
        devices = _sim_devices()
        names = list(devices.get_sim_runtime().task.env_state_names)
        if cfg.environment_state_names and list(cfg.environment_state_names) != names:
            raise RuntimeError(
                f"environment_state_names {list(cfg.environment_state_names)} do not match "
                f"the sim task's {names}"
            )
        recorded = _recorded_environment_state_names(cfg.repo_id)
        if recorded is not None and recorded != names:
            raise RuntimeError(
                f"{cfg.repo_id} already holds episodes with environment state {recorded}; "
                f"this sim recording records {names} (SO101_SIM_TASK="
                f"{self.settings.sim_task}). Use a new repo id."
            )
        return replace(cfg, environment_state_names=names)

    def begin_episode(self) -> None:
        with self._lock:
            self._check_episode_can_begin_locked()
            if current_backend(self.settings) != "sim":
                self._open_episode_locked()
                return
            self._episode_starting = True
        try:
            self._begin_sim_episode()
        finally:
            with self._lock:
                self._episode_starting = False

    def _check_episode_can_begin_locked(self) -> None:
        if self.mode is not Mode.RECORDING or self._rec_cfg is None:
            raise RuntimeError("not in recording mode")
        if self._episode_active:
            raise RuntimeError("an episode is already active")
        if self._episode_saving:
            raise RuntimeError("previous episode is still finalizing")
        if self._episode_starting:
            raise RuntimeError("an episode is already starting")

    def _open_episode_locked(self) -> None:
        assert self._rec_cfg is not None
        self._recorder = EpisodeRecorder(self._rec_cfg)
        self._recorder_epoch = self._scene_epoch
        self._episode_frames = 0
        self._episode_write_error = None
        self._episode_grade = None
        self._episode_grade_error = None
        self._episode_active = True

    def _begin_sim_episode(self) -> None:
        """Reset the simulated scene with the next seed, then open the episode.

        Under ``_io_lock``, so no teleop step interleaves with the reset
        (simulation/devices.py "Integration points"). The reset moves the arm
        to the task's start pose, so the follower's SafetyGuard is re-seeded
        from the new pose; a guard still anchored to the old pose would
        rate-limit the next command from there.
        """
        devices = _sim_devices()
        follower = self.follower()
        self._acquire_io_for_request("episode start")
        try:
            with self._lock:
                if self.mode is not Mode.RECORDING or self._stop.is_set():
                    raise RuntimeError("not in recording mode")
            info = devices.on_episode_start()
            if follower.connected:  # else the loop's connect seeds the guard
                follower.guard.seed(follower.read_positions())
            with self._lock:
                self._scene_epoch += 1
                self._open_episode_locked()
        finally:
            self._io_lock.release()
        self._event(
            "recording.sim_episode_started",
            f"sim scene reset for episode {info.get('episode')} (seed {info.get('seed')})",
            source="recording",
            detail={key: info.get(key) for key in ("task", "seed", "episode", "instruction")},
        )

    def save_episode(self) -> dict:
        sim = current_backend(self.settings) == "sim"
        if sim:
            # The control loop selects/counts a frame and captures its runtime
            # grade before releasing _io_lock. Closing here prevents another
            # selection; after the pending write drains we use that frame's
            # stored grade, never the subsequently advancing world.
            self._io_lock.acquire()
            try:
                with self._lock:
                    recorder = self._begin_episode_save_locked()
            finally:
                self._io_lock.release()
        else:
            with self._lock:
                recorder = self._begin_episode_save_locked()
        try:
            try:
                with self._episode_write_condition:
                    while self._episode_writes:
                        self._episode_write_condition.wait()
                    if self._episode_write_error is not None:
                        raise RuntimeError(
                            f"the final recording frame failed: {self._episode_write_error}"
                        ) from self._episode_write_error
                    grade = self._episode_grade
                    grade_error = self._episode_grade_error
                    if sim and grade is None and grade_error is None:
                        grade_error = RuntimeError(
                            "no simulation predicate snapshot was captured with the final frame"
                        )
                index = recorder.save_episode(retain_operation_lock=sim)
                frames = recorder.last_saved_frames
            except Exception as exc:
                frames = recorder.num_frames
                incomplete: str | None = None
                try:
                    preserved = recorder.abort(f"episode save failed: {exc}")
                    incomplete = str(preserved) if preserved is not None else None
                except Exception as preserve_exc:
                    self._event(
                        "recording.save_and_preserve_failed",
                        str(preserve_exc),
                        level="error",
                        source="recording",
                    )
                self._event(
                    "recording.save_failed",
                    str(exc),
                    level="error",
                    source="recording",
                    detail={"frames": frames, "incomplete_path": incomplete},
                )
                raise RuntimeError(f"episode save failed: {exc}") from exc
            result: dict[str, Any] = {"episode": index, "frames": frames}
            if sim:
                # The retained dataset lock excludes writers and training
                # selection through label publication. Read-only inspection
                # endpoints do not take it and may briefly show ``unlabeled``.
                if grade_error is not None:
                    error = LabelNotStoredError.grading_failed(
                        recorder.cfg.repo_id, index, grade_error
                    )
                    self._record_sim_label_failure(error)
                    raise error
                assert grade is not None
                result.update(
                    self._label_sim_episode(
                        recorder.cfg.repo_id,
                        index,
                        grade,
                        operation_lock_held=True,
                    )
                )
            return result
        finally:
            try:
                recorder.release_operation_lock()
            finally:
                with self._lock:
                    if self._recorder is recorder:
                        self._recorder = None
                    self._episode_saving = False

    def _begin_episode_save_locked(self) -> EpisodeRecorder:
        if not self._episode_active or self._recorder is None:
            raise RuntimeError("no active episode")
        self._episode_active = False
        self._episode_saving = True
        return self._recorder

    def _record_sim_label_failure(self, error: LabelNotStoredError) -> None:
        self._event(
            "recording.sim_label_failed",
            str(error),
            level="error",
            source="recording",
            detail={
                "repo_id": error.repo_id,
                "episode": error.episode,
                "outcome": error.outcome,
                "recoverable": True,
            },
        )

    def _label_sim_episode(
        self,
        repo_id: str,
        index: int,
        grade: dict[str, Any],
        *,
        operation_lock_held: bool,
    ) -> dict[str, Any]:
        """Label a saved sim episode with the task's success predicate.

        The outcome source is ``auto:sim_predicate``, as for ``so101 sim
        demos``. A human label is never overwritten. A storage failure raises
        a recoverable error that names the already-saved episode, so it cannot
        silently enter the default training selection as ``unlabeled``.
        """
        try:
            outcome, label = persist_sim_outcome(
                repo_id,
                index,
                bool(grade["success"]),
                grade["metrics"],
                operation_lock_held=operation_lock_held,
            )
        except LabelNotStoredError as exc:
            self._record_sim_label_failure(exc)
            raise
        self._event(
            "recording.sim_episode_labeled",
            f"episode {index} of {repo_id}: {outcome} by the sim task predicate ({label})",
            level="info" if label in {"auto-labeled", "kept the human label"} else "warning",
            source="recording",
            detail={"repo_id": repo_id, "episode": index, "outcome": outcome, "label": label},
        )
        return {"outcome": outcome, "label": label}

    def _require_direct_camera_capture(self) -> None:
        if (
            current_backend(self.settings) == "real"
            and self.settings.camera_capture_mode == "shared"
        ):
            raise RuntimeError(
                "Cameras are owned by the shared relays. Use leader teleop with MCAP "
                "episodes; direct preview/dataset capture requires an explicit camera handoff."
            )

    def _control_loop(self, fps: int, record: bool) -> None:
        with self.follower().leader_teleop_profile():
            self._leader_control_loop(fps, record)

    def _leader_control_loop(self, fps: int, record: bool) -> None:
        controller = TeleopController(self.leader(), self.follower(), fps=fps)
        shared = (
            current_backend(self.settings) == "real"
            and self.settings.camera_capture_mode == "shared"
            and not record
        )
        cameras = None if shared else self.cameras()
        # Sim recordings carry observation.environment_state (start_recording).
        cfg = self._rec_cfg if record else None
        with_env_state = bool(cfg is not None and cfg.environment_state_names)
        sim_recording = record and current_backend(self.settings) == "sim"
        last = self._clock()
        try:
            if self._stop.is_set():
                return
            self._prepare_global_energized_intent(
                "RobotManager recording" if record else "RobotManager teleop",
                include_leader=True,
            )
            with self._io_lock:
                # A stop may have cut torque while this thread waited for the
                # bus; never connect or step after it.
                if self._stop.is_set():
                    return
                with self._lock:
                    if self._stop.is_set() or self.guard.estopped or self._fault is not None:
                        raise RuntimeError("a stop or fault landed before teleop torque-enable")
                controller.connect()
            if cameras is not None:
                cameras.open()
            pacer = _Pacer(1.0 / fps, clock=self._clock, sleep=self._sleep)
            previous_started: float | None = None
            previous_tail: float | None = None
            previous_pacing: dict[str, float | bool] | None = None
            while not self._stop.is_set():
                env_state = None
                recorder: EpisodeRecorder | None = None
                frame_grade: dict[str, Any] | None = None
                frame_grade_error: Exception | None = None
                tick_started = self._clock()
                with self._io_lock:
                    io_acquired = self._clock()
                    if self._stop.is_set():
                        break
                    step = controller.step()
                    motor_completed = self._clock()
                    scene_epoch = self._scene_epoch
                    with self._lock:
                        recorder = (
                            self._recorder
                            if record
                            and self._episode_active
                            and scene_epoch == self._recorder_epoch
                            else None
                        )
                    if recorder is not None:
                        if sim_recording:
                            try:
                                env_state, frame_grade, frame_grade_error = (
                                    _sim_devices().recording_snapshot()
                                )
                            except Exception as exc:
                                frame_grade_error = exc
                            if frame_grade is not None and (
                                not isinstance(frame_grade, dict)
                                or not isinstance(frame_grade.get("success"), bool)
                                or not isinstance(frame_grade.get("metrics"), dict)
                            ):
                                frame_grade = None
                                frame_grade_error = ValueError(
                                    "simulation grade must contain boolean success and "
                                    "object metrics"
                                )
                        elif with_env_state:
                            env_state = controller.follower.environment_state()
                        with self._lock:
                            self._episode_writes += 1
                tail_started = self._clock()
                try:
                    self.control_trace.append(
                        step,
                        backend=current_backend(self.settings),
                        started=tick_started,
                        completed=self._clock(),
                        fps=fps,
                        profile=controller.follower.command_profile(),
                        timing={
                            "cycle_interval_seconds": (
                                None
                                if previous_started is None
                                else tick_started - previous_started
                            ),
                            "io_lock_wait_seconds": io_acquired - tick_started,
                            "motor_step_seconds": motor_completed - io_acquired,
                            "previous_tail_seconds": previous_tail,
                            "previous_pacing": previous_pacing,
                        },
                    )
                    frames = cameras.read() if cameras is not None else {}
                    now = self._clock()
                    inst_fps = 1.0 / max(1e-6, now - last)
                    last = now
                    with self._lock:
                        self._frames = frames
                        captured_at = self._clock()
                        self._frame_times = {camera: captured_at for camera in frames}
                        self._snapshot = {
                            "t": step.t,
                            "observation": step.observation,
                            "action": step.action,
                            "desired_action": step.desired_action,
                            "fps": round(inst_fps, 1),
                        }
                        self._snapshot_monotonic = captured_at
                        self._sample_seq += 1
                    if recorder is not None:
                        recorder.add(step, frames, environment_state=env_state)
                        with self._episode_write_condition:
                            if recorder is self._recorder:
                                self._episode_frames += 1
                                if sim_recording:
                                    self._episode_grade = frame_grade
                                    self._episode_grade_error = frame_grade_error
                except Exception as exc:
                    if recorder is not None:
                        with self._episode_write_condition:
                            self._episode_write_error = exc
                    raise
                finally:
                    if recorder is not None:
                        with self._episode_write_condition:
                            self._episode_writes -= 1
                            self._episode_write_condition.notify_all()
                previous_started = tick_started
                previous_tail = self._clock() - tail_started
                previous_pacing = pacer.wait()
        finally:
            try:
                # Only the leader: it is passive, back-driven with torque off.
                # The follower stays energized for the stop that ended the
                # loop to hold or lower (design doc §5); a failed loop's
                # wrapper cuts it instead.
                leader_confirmed_off = False
                with self._io_lock:
                    controller.leader.disconnect()
                    leader_confirmed_off = bool(
                        current_backend(self.settings) == "real"
                        and not controller.leader.connected
                        and bus_torque_state(controller.leader.bus) == "off"
                    )
                if leader_confirmed_off:
                    # Journal/SQLite I/O happens only after the motor lock is
                    # free, so a full disk cannot delay category-0 teardown.
                    self._retire_confirmed_recovery_devices(
                        {"leader"},
                        "teleop/recording leader torque-off confirmed",
                    )
            finally:
                if cameras is not None:
                    cameras.close()
                with self._lock:
                    self._frames = {}
                    self._frame_times = {}

    # --- rollout loop ---------------------------------------------------
    def start_rollout(
        self,
        model_path: str,
        fps: int = 30,
        *,
        operator_id: str,
        policy: Any | None = None,
        task: str | None = None,
    ) -> None:
        """Run a policy on the follower until stopped.

        ``policy`` is a policy the caller already loaded (data-B13,
        docs/audit/v0.1/data.md): eval loads once per session and hands the
        same instance to every trial, so a checkpoint load never runs inside a
        trial's time budget. Without it the rollout thread loads
        ``model_path`` as before. Either way the policy is reset at the start
        of every rollout. Refuses while E-stopped or faulted (safety-B1; the
        check lives in ``_start_thread``).

        ``task`` is the language instruction for a language-conditioned policy
        (data-B6). The policy binds it at load time, so it goes to
        ``load_policy(model_path, task=...)``, or onto a preloaded policy's
        ``task`` attribute, which such a policy reads on every inference.
        Policies without that attribute take no instruction and are left
        alone. ``None`` or blank keeps today's calls exactly.
        """
        if fps <= 0 or fps > 240:
            raise RuntimeError("fps must be between 1 and 240")
        if policy is not None and not callable(getattr(policy, "select_action", None)):
            raise TypeError("policy must provide select_action(observation)")
        task = (task or "").strip() or None
        if task is not None and policy is None and not _accepts_keyword(load_policy, "task"):
            raise TypeError(
                "this build's load_policy cannot bind a task; omit the task "
                "or pass a policy loaded with it"
            )
        self._start_thread(
            Mode.ROLLOUT,
            lambda: self._rollout_loop(model_path, fps, policy, task),
            operator_id,
            active_model=str(model_path),
        )

    def _rollout_loop(
        self, model_path: str, fps: int, policy: Any | None = None, task: str | None = None
    ) -> None:
        if policy is None:
            if task is None:
                policy = load_policy(model_path)
            else:
                # start_rollout checked that this build's loader takes task=
                # (data-learn's does); typed loosely to build either way.
                loader: Callable[..., Any] = load_policy
                policy = loader(model_path, task=task)
        elif task is not None and hasattr(policy, "task"):
            policy.task = task
        policy.reset()
        follower = self.follower()
        cameras = self.cameras()
        # data-B13: the rollout clock starts at the first action. No sample is
        # published before it, and neither the load nor the connect counts as
        # a frame interval (the first frame reports fps 0.0).
        last: float | None = None
        try:
            if self._stop.is_set():
                return
            self._prepare_global_energized_intent("RobotManager policy rollout")
            try:
                with self._io_lock:
                    if self._stop.is_set():
                        return
                    with self._lock:
                        if self._stop.is_set() or self.guard.estopped or self._fault is not None:
                            raise RuntimeError(
                                "a stop or fault landed before rollout torque-enable"
                            )
                    follower.connect()
            except BaseException:
                # The with-statement released _io_lock before durable retire.
                self._retire_failed_enable_intent_if_safe(
                    follower,
                    "rollout setup confirmed torque off",
                )
                raise
            cameras.open()
            pacer = _Pacer(1.0 / fps, clock=self._clock, sleep=self._sleep)
            while not self._stop.is_set():
                frames = cameras.read()
                with self._io_lock:
                    if self._stop.is_set():
                        break
                    obs = _build_observation(
                        follower,
                        frames,
                        policy=policy,
                        feature_names=getattr(cameras, "feature_names", None),
                    )
                action = policy.select_action(obs)
                with self._io_lock:
                    if self._stop.is_set():
                        break
                    commanded = follower.send_goal(action)
                    measured = follower.observation()
                now = self._clock()
                with self._lock:
                    self._frames = frames
                    captured_at = self._clock()
                    self._frame_times = {camera: captured_at for camera in frames}
                    self._snapshot = {
                        "t": time.time(),
                        "observation": measured,
                        "action": {f"{key}.pos": value for key, value in commanded.items()},
                        "fps": 0.0 if last is None else round(1.0 / max(1e-6, now - last), 1),
                    }
                    self._snapshot_monotonic = captured_at
                    self._sample_seq += 1
                last = now
                pacer.wait()
        finally:
            # The follower stays energized for the stop that ended the loop to
            # hold or lower (design doc §5); a failed loop's wrapper cuts it.
            cameras.close()
            with self._lock:
                self._frames = {}
                self._frame_times = {}

    def shutdown(self) -> None:
        """Service shutdown (SIGTERM, app lifespan end): a bounded Collapse, then torque off.

        Waits for the descent at most its bound (``collapse_timeout_seconds``,
        at most 12 s, plus a margin); past it, torque is cut with a latched
        fault. Then the remaining ports close under the bus lock.
        """

        self.stop_camera_preview()
        self._dispatch_stop(StopKind.COLLAPSE, reason="service shutdown")
        self._await_collapse()
        self._watchdog_shutdown.set()
        self._health_shutdown.set()
        for thread in (self._watchdog_thread, self._health_thread):
            if thread and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        # Close what the stop left open under the bus lock, with a timed acquire
        # like every teardown. Without the lock, these torque writes shared the
        # serial port with a running driver call, or hung behind a wedged one
        # (safety-B2, docs/audit/v0.1/safety.md).
        buses = [
            (name, device)
            for name, device in (("follower", self._follower), ("leader", self._leader))
            if device is not None
        ]
        shutdown_unconfirmed: set[str] = set()
        if buses and self._io_lock.acquire(timeout=0.5):
            try:
                for name, device in buses:
                    try:
                        device.disconnect()
                    except Exception as exc:
                        log.critical("shutdown: %s", exc)
                        shutdown_unconfirmed.add(name)
                    if bus_torque_state(device.bus) != "off":
                        shutdown_unconfirmed.add(name)
            finally:
                self._io_lock.release()
        elif buses:
            log.critical("shutdown: motor I/O did not yield; the motor ports stay open")
            shutdown_unconfirmed.update(name for name, _device in buses)
        if shutdown_unconfirmed:
            names = ", ".join(sorted(shutdown_unconfirmed))
            self._mark_torque_unconfirmed(
                f"shutdown could not confirm final torque-off on {names}",
                event="safety.shutdown_torque_unconfirmed",
                source="shutdown",
                devices=shutdown_unconfirmed,
            )
        cameras = self._cameras
        try:
            if cameras is not None and self._camera_preview_thread is None:
                cameras.disconnect() if hasattr(cameras, "disconnect") else cameras.close()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        if current_backend(self.settings) == "sim":
            # The simulated arm's physics thread (simulation/devices.py
            # "Integration points"). Only when the sim module is loaded.
            devices = sys.modules.get("so101.simulation.devices")
            if devices is not None:
                devices.stop_sim_runtime()


def _sim_devices() -> Any:
    """``so101.simulation.devices`` (MuJoCo), or RuntimeError naming why it cannot load."""

    try:
        from ..simulation import devices
    except Exception as exc:
        raise RuntimeError(
            f"SO101_BACKEND=sim: the simulated arm cannot load ({type(exc).__name__}: {exc})"
        ) from exc
    return devices


def _recorded_environment_state_names(repo_id: str) -> list[str] | None:
    """The environment-state layout of a dataset's saved episodes; None when it has none."""

    from ..data.dataset import dataset_root

    try:  # an invalid repo id is the recorder's to refuse
        meta = json.loads((dataset_root(repo_id) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not meta.get("episodes"):
        return None
    return [str(name) for name in meta.get("environment_state_names") or []]


_manager: RobotManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> RobotManager:
    global _manager
    settings = get_settings()
    with _manager_lock:
        if _manager is None or _manager.settings.data_dir != settings.data_dir:
            if _manager is not None:
                _manager.shutdown()
            _manager = RobotManager()
        return _manager


def reset_manager() -> None:
    """Dispose of the process singleton (used by isolated app tests)."""

    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
            if current_backend(_manager.settings) == "sim":
                devices = sys.modules.get("so101.simulation.devices")
                if devices is not None:
                    devices.reset_sim_runtime()
        _manager = None

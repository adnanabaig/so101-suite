"""Preflight: ordered, read-only readiness checks before an SO-101 arm moves.

Every check is a pure function of a :class:`PreflightContext`, which holds the
facts gathered up front (host, LeRobot surface, ports, one diagnostic pass per
bus, calibration files, rest pose, cameras, disks, E-stop). Gathering facts is
the only step with side effects, and it never writes a motor register.

Two entry points:

* :func:`run_cli_preflight` (``so101 preflight``). With no server running it
  opens a bus only after :mod:`.ownership` says so and the canonical device's
  host-global lock is free.
  With a Suite server running it asks that server for the bus facts
  (``POST /api/preflight/bus``) and grades them here.
* :func:`run_api_preflight` (``POST /api/preflight``). The bus checks go
  through ``RobotManager.run_idle_bus_diagnostic``
  (:func:`collect_manager_bus_facts`): the manager lends each idle bus
  read-only under its bus lock, or refuses while the robot is busy.

Mock runs use :class:`.buses.FakeFeetechBus` and are labeled
``rehearsal: true`` on the report and on every check.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from .. import JOINT_NAMES, __version__
from ..atomic_io import atomic_write_json
from ..hardware import lerobot_paths
from ..hardware.backend import (
    SIM_EXTRA_REMEDIATION,
    Backend,
    current_backend,
    is_rehearsal,
    sim_devices_problem,
)
from ..hardware.health import (
    JOINT_IDS,
    HealthThresholds,
    convert_temperature,
    convert_voltage,
    decode_status,
    grade_temperature,
    grade_voltage,
)
from .buses import (
    BusDiagnosticOpenError,
    BusFacts,
    BusOpener,
    FakeFeetechBus,
    bus_facts_from_json,
    bus_facts_to_json,
    collect_bus_facts,
    diagnose_port,
    open_feetech_bus,
    synthetic_calibration_for,
)
from .lerobot_surface import (
    DEFAULT_BAUDRATE,
    DEGREES_PER_TICK,
    LEROBOT_PINNED_VERSION,
    STS3215_MODEL_NUMBER,
    STS3215_RESOLUTION,
    LeRobotSurface,
    probe_lerobot_surface,
)
from .ownership import (
    PROBE_CONNECT_TIMEOUT_S,
    STATUS_PATH,
    BusOwnershipError,
    ServerProbe,
    ServerProber,
    acquire_bus_access,
    probe_local_server,
    server_running_error,
)
from .schemas import (
    ARMS,
    ArmName,
    CheckResult,
    CheckStatus,
    MeasuredSource,
    OverallStatus,
    PreflightBusFacts,
    PreflightCheckInfo,
    PreflightOrigin,
    PreflightReport,
)
from .serial_ports import (
    PYSERIAL_REMEDIATION,
    PortListing,
    list_serial_ports,
    match_port,
    not_found_remediation,
    permission_remediation,
    port_key,
    shared_serial_number,
)

#: Skip message for bus checks on the real backend when the manager offers no
#: ``run_idle_bus_diagnostic`` (an older server, or a stand-in manager).
WAVE2_BUS_MESSAGE = (
    "needs idle bus access through RobotManager, which this server does not offer "
    "(run_idle_bus_diagnostic)"
)
#: How long the API waits for the motor bus before it reports it busy. The
#: server's servo-health poll holds it for a few milliseconds at a time.
API_BUS_WAIT_S = 1.0
#: How long ``so101 preflight`` waits for a running server's bus facts. One
#: arm with every motor silent takes about 2 s of read timeouts.
SERVER_BUS_FACTS_TIMEOUT_S = 30.0
#: Remediation when the manager refuses to lend a bus.
REFUSED_REMEDIATION = (
    "Stop teleop, recording or rollout, disarm, and let any Halt or Collapse finish "
    "(or reset the E-stop), then retry."
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_INCOMPLETE = 3
EXIT_REHEARSAL_REJECTED = 4

#: Position spread (raw ticks) tolerated while the arm sits still, torque off.
STABLE_SPREAD_TICKS = 3
#: Calibrated span below this is rejected by the Suite's own calibration
#: (motors.py FeetechBus.calibrate_timed: "less than 100 encoder ticks").
MIN_CALIBRATED_SPAN_TICKS = 100
#: Suite heuristic: a span this small usually means the joint was not swept.
WARN_CALIBRATED_SPAN_TICKS = 400
#: Suite heuristic: tolerance before a resting joint counts as out of range.
RANGE_TOLERANCE_TICKS = 20
#: Suite heuristic: mean pixel value below this is treated as a black frame.
DARK_FRAME_MEAN = 5.0
DISK_FAIL_BYTES = 2 * 1024**3
DISK_WARN_BYTES = 10 * 1024**3
CAMERA_FRAMES = 3
#: Suite heuristic: a depth stream with fewer valid (nonzero) pixels than this
#: fraction is pointed wrong, too close, blocked or in the dark.
DEPTH_MIN_VALID_FRACTION = 0.10
#: Why the camera check skips under SO101_BACKEND=sim.
SIM_CAMERAS_SKIP = "the sim backend renders its cameras; no real camera was probed"

#: Linux kernel names that can be renumbered between boots (a configured value
#: is matched against this; nothing is globbed).
_UNSTABLE_LINUX_PORT = re.compile(r"/dev/tty(ACM|USB)\d+")
_ENV_KEYS = ("LEROBOT_HOME", "HF_LEROBOT_HOME", "HF_LEROBOT_CALIBRATION", "HF_HOME", "SO101_MOCK")


def port_for(settings: Any, arm: str) -> str:
    return str(settings.leader_port if arm == "leader" else settings.follower_port)


def device_id_for(settings: Any, arm: str) -> str:
    return str(settings.leader_id if arm == "leader" else settings.follower_id)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- facts -------------------------------------------------------------------


@dataclass
class HostFacts:
    python: tuple[int, int, int]
    platform: str
    so101_version: str
    environ: dict[str, str]
    calibration_root: str
    calibration_root_how: str


def collect_host_facts(environ: Mapping[str, str] | None = None) -> HostFacts:
    env = os.environ if environ is None else environ
    root, how = lerobot_paths.lerobot_calibration_root()
    info = sys.version_info
    return HostFacts(
        python=(info.major, info.minor, info.micro),
        platform=platform.platform(),
        so101_version=__version__,
        environ={key: env[key] for key in _ENV_KEYS if key in env},
        calibration_root=str(root),
        calibration_root_how=how,
    )


@dataclass
class PortFacts:
    """One configured port. ``None`` means "not checkable on this platform"."""

    arm: str
    path: str
    #: POSIX: the device file exists. Windows: comports() lists the COM port.
    exists: bool = False
    #: comports() lists it (see :func:`.serial_ports.match_port`).
    listed: bool = False
    #: The listed adapter: device, serial_number, vid_pid, description...
    adapter: dict[str, Any] | None = None
    char_device: bool | None = None
    readable: bool | None = None
    writable: bool | None = None
    #: Comparable name (real path on POSIX, upper-case COMx on Windows).
    realpath: str | None = None
    error: str | None = None


def collect_port_facts(
    arm: str, path: str, listing: PortListing | None = None, *, platform: str | None = None
) -> PortFacts:
    """Facts about one configured port, without opening it."""
    where = platform or (listing.platform if listing is not None else sys.platform)
    adapter = match_port(path, listing.ports, platform=where) if listing is not None else None
    facts = PortFacts(
        arm=arm,
        path=path,
        listed=adapter is not None,
        adapter=adapter.to_dict() if adapter is not None else None,
        realpath=port_key(path, platform=where),
    )
    if where == "win32":
        # A COM port is not a file: being listed is the existence test, and only
        # opening it can tell "in use" (the motors check opens it).
        facts.exists = adapter is not None
        return facts
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return facts
    except OSError as exc:
        facts.error = f"{type(exc).__name__}: {exc}"
        return facts
    facts.exists = True
    facts.char_device = stat.S_ISCHR(info.st_mode)
    facts.readable = os.access(path, os.R_OK)
    facts.writable = os.access(path, os.W_OK)
    return facts


PortInspector = Callable[[str, str, PortListing], PortFacts]


def _permission_denied(facts: PortFacts) -> bool:
    return facts.readable is False or facts.writable is False


@dataclass
class CalibrationFacts:
    arm: str
    device_id: str
    lerobot_path: str
    lerobot_exists: bool = False
    lerobot_payload: dict[str, Any] | None = None
    lerobot_error: str | None = None
    lerobot_sha256: str | None = None
    legacy_path: str = ""
    legacy_exists: bool = False
    suite_path: str = ""
    suite_exists: bool = False
    suite_device_id: str | None = None
    suite_backend: str | None = None
    #: The Suite file came from a mock run (``hardware.calibration.is_mock_calibration``).
    suite_mock: bool = False
    suite_error: str | None = None
    synthetic: bool = False


def collect_calibration_facts(
    settings: Any,
    arm: str,
    *,
    synthetic_payload: Mapping[str, Any] | None = None,
) -> CalibrationFacts:
    """Read LeRobot's and the Suite's calibration files for one arm."""
    from ..hardware.calibration import CalibrationResult, is_mock_calibration

    device_id = device_id_for(settings, arm)
    facts = CalibrationFacts(arm=arm, device_id=device_id, lerobot_path="")
    if synthetic_payload is not None:
        facts.lerobot_path = "(synthetic, rehearsal)"
        facts.lerobot_exists = True
        facts.lerobot_payload = {name: dict(entry) for name, entry in synthetic_payload.items()}
        facts.synthetic = True
    else:
        root, _ = lerobot_paths.lerobot_calibration_root()
        path = lerobot_paths.lerobot_calibration_file(arm, device_id, root)
        legacy = lerobot_paths.legacy_calibration_file(arm, device_id, root)
        facts.lerobot_path = str(path)
        facts.legacy_path = str(legacy)
        facts.legacy_exists = legacy.is_file()
        if path.is_file():
            facts.lerobot_exists = True
            try:
                import json

                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not all(
                    isinstance(entry, dict) for entry in payload.values()
                ):
                    raise ValueError("expected an object of per-motor objects")
                facts.lerobot_payload = payload
                facts.lerobot_sha256 = lerobot_paths.file_sha256(path)
            except (OSError, ValueError) as exc:
                facts.lerobot_error = f"{type(exc).__name__}: {exc}"

    suite_path = Path(settings.data_dir) / f"calibration_{arm}.json"
    facts.suite_path = str(suite_path)
    if suite_path.is_file():
        facts.suite_exists = True
        try:
            loaded = CalibrationResult.load(arm, suite_path)
            if loaded is not None:
                facts.suite_device_id = loaded.device_id
                facts.suite_backend = loaded.backend
                facts.suite_mock = is_mock_calibration(loaded)
        except Exception as exc:
            facts.suite_error = f"{type(exc).__name__}: {exc}"
    return facts


@dataclass
class RestPoseFacts:
    resolution: Any | None = None  # RestPoseResolution
    error: str | None = None
    limits_source: str = ""


def collect_rest_pose_facts(
    settings: Any,
    backend: Backend,
    *,
    limits: Mapping[str, tuple[float, float]] | None,
    limits_source: str,
) -> RestPoseFacts:
    """The pose Collapse would use: the same resolver, order and clamp as Collapse.

    ``resolve_collapse_rest_pose``: recorded, then ``SO101_REST_POSE``, then the
    model-derived default, clamped to ``limits`` (the SafetyGuard) ∩ MJCF.
    """
    from ..hardware.rest_pose import resolve_collapse_rest_pose

    source = f"{limits_source} ∩ MJCF limits"
    try:
        resolution = resolve_collapse_rest_pose(settings, limits=limits, backend=backend)
    except Exception as exc:
        return RestPoseFacts(error=f"{type(exc).__name__}: {exc}", limits_source=source)
    return RestPoseFacts(resolution=resolution, limits_source=source)


def saved_guard_limits() -> dict[str, tuple[float, float]]:
    """SafetyGuard limits as the server would build them (no hardware access)."""
    from ..hardware.follower import SO101Follower

    return dict(SO101Follower.guard_from_saved_calibration().limits)


@dataclass
class CameraProbe:
    camera: str
    ok: bool
    error: str | None = None
    mean: float | None = None
    shape: list[int] | None = None
    kind: str | None = None  # "opencv" or "realsense"
    device: dict[str, Any] | None = None  # CameraManager.info()["device"]
    streams: dict[str, Any] | None = None  # per stream: shape, and depth validity
    remediation: str | None = None  # from CameraUnavailableError, OS-specific


@dataclass
class CameraFacts:
    configured: list[str]
    source: MeasuredSource
    probes: list[CameraProbe] = field(default_factory=list)
    live: list[str] | None = None
    skipped_reason: str | None = None


CameraProber = Callable[[list[str]], list[CameraProbe]]


def _depth_stats(depth: Any) -> dict[str, Any]:
    import numpy as np

    valid = depth[depth > 0]
    stats: dict[str, Any] = {
        "shape": [int(n) for n in depth.shape],
        "valid_fraction": round(float(valid.size) / max(1, depth.size), 4),
    }
    if valid.size:
        stats.update(
            min_mm=int(valid.min()), median_mm=int(np.median(valid)), max_mm=int(valid.max())
        )
    return stats


def probe_cameras_locally(
    camera_ids: list[str], *, frames: int = CAMERA_FRAMES
) -> list[CameraProbe]:
    """Open each camera on its own, read a few frames, close it.

    ``camera_ids`` are SO101_CAMERAS entries; each probe is labelled with the
    entry's stream key, which has its credentials stripped. A RealSense
    ``+depth`` entry also reports its depth stream's valid-pixel fraction, and
    every RealSense its device facts (USB link, firmware).
    """
    from ..hardware.camera_spec import (
        DEPTH_SUFFIX,
        parse_camera_entry,
        redact_credentials,
        strip_credentials,
    )
    from ..hardware.cameras import CameraManager

    results: list[CameraProbe] = []
    for entry in camera_ids:
        try:
            spec = parse_camera_entry(entry)
            kind: str | None = spec.kind
            camera = spec.color_key
        except ValueError:
            kind, camera = None, strip_credentials(entry.strip())
        manager: CameraManager | None = None
        try:
            manager = CameraManager([entry])
            manager.open()
            read: dict[str, Any] = {}
            for _ in range(max(1, frames)):
                read = manager.read()
            frame = read[camera]
            streams: dict[str, Any] = {"color": {"shape": [int(n) for n in frame.shape]}}
            depth = read.get(camera + DEPTH_SUFFIX)
            if depth is not None:
                streams["depth"] = _depth_stats(depth)
            results.append(
                CameraProbe(
                    camera=camera,
                    ok=True,
                    mean=round(float(frame.mean()), 2),
                    shape=[int(n) for n in frame.shape],
                    kind=kind,
                    device=manager.info(camera)["device"] or None,
                    streams=streams,
                )
            )
        except Exception as exc:
            detail = redact_credentials(str(getattr(exc, "detail", None) or exc))
            results.append(
                CameraProbe(
                    camera=camera,
                    ok=False,
                    error=f"{type(exc).__name__}: {detail}",
                    kind=kind,
                    remediation=getattr(exc, "remediation", None) or None,
                )
            )
        finally:
            if manager is not None:
                manager.close()
    return results


def _camera_keys(settings: Any) -> list[str]:
    """One color stream key per configured camera, credentials stripped."""
    return [spec.color_key for spec in settings.camera_specs]


def cameras_from_status(
    configured: list[str], status: Mapping[str, Any] | None, *, source: MeasuredSource
) -> CameraFacts:
    if status is None:
        return CameraFacts(
            configured=configured,
            source=source,
            skipped_reason="the server status could not be read",
        )
    # Wave-2 servers also list "<entry>#depth" streams; a camera counts as live
    # by its color stream, so depth keys are kept but never required.
    live = [str(camera) for camera in status.get("live_cameras") or []]
    return CameraFacts(configured=configured, source=source, live=live)


@dataclass
class DirectoryFacts:
    role: str
    path: str
    exists: bool
    writable: bool
    free_bytes: int | None
    total_bytes: int | None
    error: str | None = None


def collect_disk_facts(settings: Any) -> list[DirectoryFacts]:
    facts: list[DirectoryFacts] = []
    for role in ("data_dir", "datasets_dir", "outputs_dir"):
        path = Path(getattr(settings, role))
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            facts.append(
                DirectoryFacts(
                    role=role,
                    path=str(path),
                    exists=path.exists(),
                    writable=os.access(probe, os.W_OK),
                    free_bytes=int(usage.free),
                    total_bytes=int(usage.total),
                )
            )
        except OSError as exc:
            facts.append(
                DirectoryFacts(
                    role=role,
                    path=str(path),
                    exists=path.exists(),
                    writable=False,
                    free_bytes=None,
                    total_bytes=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return facts


@dataclass
class EstopFacts:
    known: bool
    source: MeasuredSource
    estopped: bool | None = None
    fault: str | None = None
    mode: str | None = None
    detail: str = ""
    #: ``/api/robot/status`` carries no fault field; only the in-process
    #: (API) path can see ``RobotManager.fault``.
    fault_known: bool = False


NO_SERVER_ESTOP_DETAIL = (
    "no Suite server is running, so there is no live software E-stop status to read; persisted "
    "E-stop, fault and unconfirmed-torque latches are restored by a server restart and enforced "
    "before standalone hardware commands. While no server runs, the motor power switch is the "
    "only immediate stop."
)


def estop_from_server(probe: ServerProbe) -> EstopFacts:
    if not probe.running:
        return EstopFacts(known=False, source="none", detail=NO_SERVER_ESTOP_DETAIL)
    if probe.status is None:
        return EstopFacts(
            known=False,
            source="server",
            detail=f"something answered at {probe.url} without a Suite status ({probe.detail})",
        )
    return EstopFacts(
        known=True,
        source="server",
        estopped=bool(probe.status.get("estopped")),
        mode=str(probe.status.get("mode")) if probe.status.get("mode") is not None else None,
    )


@dataclass
class BusAccess:
    mode: str  # opened | refused | deferred | synthetic | unavailable
    detail: str = ""
    remediation: str = ""
    #: Whether bus checks skipped under this mode leave readiness unproven.
    blocking: bool = True


@dataclass
class PreflightContext:
    settings: Any
    origin: PreflightOrigin
    backend: Backend
    arms: tuple[ArmName, ...]
    host: HostFacts
    lerobot: LeRobotSurface
    ports: dict[str, PortFacts]
    serial_ports: PortListing
    bus_access: BusAccess
    buses: dict[str, BusFacts]
    calibration: dict[str, CalibrationFacts]
    rest_pose: RestPoseFacts | None
    cameras: CameraFacts
    disks: list[DirectoryFacts]
    estop: EstopFacts
    server: ServerProbe | None = None
    #: Sim backend only: why ``so101.simulation.devices`` (MuJoCo) cannot load,
    #: or None when it loads (:func:`~so101.hardware.backend.sim_devices_problem`).
    sim_devices: str | None = None

    @property
    def rehearsal(self) -> bool:
        return is_rehearsal(self.backend)

    @property
    def bus_source(self) -> MeasuredSource:
        return "real" if self.backend == "real" else "mock"

    def thresholds(self, arm: str) -> HealthThresholds:
        return HealthThresholds.from_settings(self.settings, arm)


# --- check registry -------------------------------------------------------------

CheckFn = Callable[[PreflightContext, ArmName | None], CheckResult]


@dataclass(frozen=True)
class CheckSpec:
    id: str
    title: str
    description: str
    fn: CheckFn
    per_arm: bool = False
    needs_bus: bool = False
    arms: tuple[ArmName, ...] = ARMS


def _spec(check_id: str) -> CheckSpec:
    return CHECKS_BY_ID[check_id]


def _result(
    ctx: PreflightContext,
    check_id: str,
    arm: ArmName | None,
    status: CheckStatus,
    message: str,
    *,
    remediation: str = "",
    measured: Mapping[str, Any] | None = None,
    blocking: bool = True,
    source: MeasuredSource = "host",
    references: Sequence[str] = (),
) -> CheckResult:
    spec = _spec(check_id)
    return CheckResult(
        id=check_id,
        key=f"{check_id}.{arm}" if arm else check_id,
        arm=arm,
        title=f"{spec.title} ({arm})" if arm else spec.title,
        status=status,
        message=message,
        remediation=remediation,
        measured=dict(measured or {}),
        blocking=blocking,
        rehearsal=ctx.rehearsal,
        source=source,
        references=list(references),
    )


def _bus_skip(ctx: PreflightContext, check_id: str, arm: ArmName) -> CheckResult | None:
    """A skip result when the bus facts this check needs are unavailable."""
    access = ctx.bus_access
    facts = ctx.buses.get(arm)
    if facts is None or access.mode in {"refused", "deferred", "unavailable"}:
        return _result(
            ctx,
            check_id,
            arm,
            "skip",
            access.detail or "the bus was not read",
            remediation=access.remediation,
            blocking=access.blocking,
            source="none",
        )
    if not facts.opened:
        return _result(
            ctx,
            check_id,
            arm,
            "skip",
            f"the {arm} bus could not be opened (see motors.{arm})",
            blocking=True,
            source="none",
        )
    if check_id != "motors" and not facts.responding:
        return _result(
            ctx,
            check_id,
            arm,
            "skip",
            f"no {arm} motor answered (see motors.{arm})",
            blocking=True,
            source=ctx.bus_source,
        )
    return None


def _joined(items: Sequence[str]) -> str:
    return "; ".join(items)


def _motor_label(motor: str) -> str:
    return f"{motor} (ID {JOINT_IDS[motor]})"


# --- checks ---------------------------------------------------------------------


def check_environment(ctx: PreflightContext, arm: ArmName | None = None) -> CheckResult:
    host = ctx.host
    fails: list[str] = []
    warns: list[str] = []
    fixes: list[str] = []
    real = ctx.backend == "real"
    if host.python < (3, 12) or host.python[:2] > (3, 13):
        older = host.python < (3, 12)
        fails.append(
            f"Python {'.'.join(map(str, host.python))} is "
            + ("older than 3.12" if older else "newer than 3.13")
        )
        fixes.append("Use Python 3.12 or 3.13 (pyproject requires-python >=3.12,<3.14).")
    baudrate = int(getattr(ctx.settings, "baudrate", DEFAULT_BAUDRATE))
    if baudrate != DEFAULT_BAUDRATE:
        text = (
            f"SO101_BAUDRATE is {baudrate}; SO-101 servos run at 1,000,000 baud "
            "(LeRobot feetech.py:44) and the Suite refuses any other value "
            "(motors.py FeetechBus._build)"
        )
        (fails if real else warns).append(text)
        fixes.append("Unset SO101_BAUDRATE or set it to 1000000.")
    if "LEROBOT_HOME" in host.environ:
        text = (
            "LEROBOT_HOME is set; LeRobot 0.6.1 raises ValueError on import "
            "(utils/constants.py:61-65)"
        )
        (fails if real else warns).append(text)
        fixes.append("Unset LEROBOT_HOME and use HF_LEROBOT_HOME instead.")
    if not real and (warns or fails):
        warns = [f"{text} (would fail on the real backend)" for text in warns]
    if ctx.backend == "sim" and ctx.sim_devices:
        # Without MuJoCo the sim bus refuses to connect, so arming would fail.
        fails.append(f"SO101_BACKEND=sim but the simulated arm cannot load: {ctx.sim_devices}")
        fixes.append(SIM_EXTRA_REMEDIATION)
    measured = {
        "python": ".".join(map(str, host.python)),
        "platform": host.platform,
        "so101_version": host.so101_version,
        "backend": ctx.backend,
        "mock": bool(getattr(ctx.settings, "mock", False)),
        "baudrate": baudrate,
        "environ": host.environ,
        "lerobot_calibration_root": host.calibration_root,
        "lerobot_calibration_root_how": host.calibration_root_how,
    }
    if ctx.backend == "sim":
        measured["sim_devices"] = ctx.sim_devices or "loaded"
    if fails:
        status: CheckStatus = "fail"
        message = _joined(fails + warns)
    elif warns:
        status, message = "warn", _joined(warns)
    else:
        status = "pass"
        message = (
            f"Python {measured['python']} on {host.platform}; backend {ctx.backend}; "
            f"{baudrate:,} baud"
        )
    return _result(
        ctx,
        "environment",
        None,
        status,
        message,
        remediation=" ".join(fixes),
        measured=measured,
        references=["lerobot-0.6.1 src/lerobot/utils/constants.py:61-65", "feetech.py:44"],
    )


def check_lerobot(ctx: PreflightContext, arm: ArmName | None = None) -> CheckResult:
    surface = ctx.lerobot
    problems = surface.problems()
    measured = surface.to_dict()
    references = [
        "lerobot-0.6.1 src/lerobot/motors/feetech/tables.py:41-101",
        "src/so101/bringup/lerobot_surface.py",
    ]
    if not problems:
        return _result(
            ctx,
            "lerobot",
            None,
            "pass",
            f"LeRobot {surface.version} with the Feetech SDK; symbols, signatures and the "
            "STS3215 control table match the 0.6.1 pin",
            measured=measured,
            references=references,
        )
    remediation = (
        f"Install the pinned robot extra (LeRobot {LEROBOT_PINNED_VERSION} with feetech) in a "
        'dedicated environment: `pip install -e ".[robot]"` (docs/bring-up.md, Install).'
    )
    if ctx.backend == "real":
        return _result(
            ctx,
            "lerobot",
            None,
            "fail",
            _joined(problems),
            remediation=remediation,
            measured=measured,
            references=references,
        )
    return _result(
        ctx,
        "lerobot",
        None,
        "warn",
        _joined(problems) + " (the real backend needs it; this rehearsal uses a synthetic bus)",
        remediation=remediation,
        measured=measured,
        references=references,
    )


def check_serial_ports(ctx: PreflightContext, arm: ArmName | None = None) -> CheckResult:
    listing = ctx.serial_ports
    where = listing.platform
    configured = {name: ctx.ports[name].path for name in ctx.arms}
    measured: dict[str, Any] = {
        "platform": where,
        "configured": configured,
        "ports": {
            name: {
                "exists": facts.exists,
                "listed": facts.listed,
                "adapter": facts.adapter,
                "char_device": facts.char_device,
                "readable": facts.readable,
                "writable": facts.writable,
                "realpath": facts.realpath,
                "error": facts.error,
            }
            for name, facts in ctx.ports.items()
        },
        "listed": [info.to_dict() for info in listing.ports],
        "listing_error": listing.error,
    }
    if ctx.backend != "real":
        return _result(
            ctx,
            "serial_ports",
            None,
            "skip",
            f"the {ctx.backend} backend does not open serial ports",
            measured=measured,
            blocking=False,
        )
    fails: list[str] = []
    warns: list[str] = []
    remedies: list[str] = []

    def remedy(text: str) -> None:
        if text not in remedies:
            remedies.append(text)

    if listing.error:
        fails.append(f"serial ports cannot be listed: {listing.error}")
        remedy(PYSERIAL_REMEDIATION)
    for name in ctx.arms:
        facts = ctx.ports[name]
        env = f"SO101_{name.upper()}_PORT"
        if listing.error and where == "win32":
            continue  # a COM port exists only as a listing entry; that failure is above
        if facts.error:
            fails.append(f"{name} port {facts.path} cannot be inspected: {facts.error}")
            remedy(not_found_remediation(listing))
        elif not facts.exists:
            fails.append(f"{name} arm not found: no serial port {facts.path} ({env})")
            remedy(not_found_remediation(listing))
        elif facts.char_device is False:
            fails.append(f"{name} port {facts.path} is not a character device")
            remedy(not_found_remediation(listing))
        elif _permission_denied(facts):
            fails.append(
                f"{name} port {facts.path} is there, but this user may not open it "
                "(no read/write permission)"
            )
            remedy(permission_remediation(facts.path, platform=where))
        else:
            if not facts.listed and listing.error is None:
                warns.append(f"{name} port {facts.path} exists, but pyserial does not list it")
            if where.startswith("linux") and _UNSTABLE_LINUX_PORT.fullmatch(facts.path):
                warns.append(
                    f"{name} port {facts.path} can be renumbered between boots; prefer "
                    f"/dev/so101-{name} (install/install.sh --follower-serial/--leader-serial) "
                    "or its /dev/serial/by-id/ link"
                )
    if len(ctx.arms) == 2:
        follower, leader = ctx.ports["follower"], ctx.ports["leader"]
        if follower.exists and leader.exists:
            shared = shared_serial_number(follower.adapter, leader.adapter)
            if follower.realpath == leader.realpath:
                fails.append(f"follower and leader ports both resolve to {follower.realpath}")
            elif shared:
                warns.append(
                    f"the follower and leader adapters both report USB serial number {shared}, "
                    "so a serial-number lookup cannot tell them apart"
                )
    if fails and not remedies:
        remedy(not_found_remediation(listing))
    remediation = " ".join(remedies)
    if fails:
        return _result(
            ctx,
            "serial_ports",
            None,
            "fail",
            _joined(fails + warns),
            remediation=remediation,
            measured=measured,
        )
    if warns:
        return _result(
            ctx,
            "serial_ports",
            None,
            "warn",
            _joined(warns),
            remediation=remediation or not_found_remediation(listing),
            measured=measured,
        )
    identities = []
    for name, port in configured.items():
        adapter = ctx.ports[name].adapter
        identity = _adapter_identity(adapter)
        identities.append(f"{name} {port}" + (f" ({identity})" if identity else ""))
    return _result(
        ctx,
        "serial_ports",
        None,
        "pass",
        "; ".join(identities) + " present and usable",
        measured=measured,
    )


def _adapter_identity(adapter: Mapping[str, Any] | None) -> str:
    if not adapter:
        return ""
    parts = []
    if adapter.get("serial_number"):
        parts.append(f"serial {adapter['serial_number']}")
    if adapter.get("vid_pid"):
        parts.append(f"VID:PID {adapter['vid_pid']}")
    return ", ".join(parts)


def check_motors(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    assert arm is not None
    skipped = _bus_skip(ctx, "motors", arm)
    facts = ctx.buses.get(arm)
    if facts is not None and facts.attempted and not facts.opened:
        return _result(
            ctx,
            "motors",
            arm,
            "fail",
            f"could not open the {arm} bus on {facts.port}: {facts.open_error}",
            remediation=(
                f"Check the USB cable and SO101_{arm.upper()}_PORT, and close any other "
                "program that has the port open."
            ),
            measured=facts.summary(),
            source=ctx.bus_source,
            references=["motors_bus.py:530-540"],
        )
    if skipped is not None:
        return skipped
    assert facts is not None
    status_raw = facts.register("Status")
    measured = {
        **facts.summary(),
        "model_numbers": facts.models,
        "expected_ids": JOINT_IDS,
        "expected_model_number": STS3215_MODEL_NUMBER,
        "status_raw": status_raw,
        "status_flags": {motor: list(decode_status(value)) for motor, value in status_raw.items()},
        "status_flags_verified": False,
    }
    fails: list[str] = []
    fixes: list[str] = []
    missing = facts.missing
    if len(missing) == len(JOINT_NAMES):
        fails.append(f"no motor answered on {facts.port} at 1,000,000 baud")
        fixes.append(
            "Check that motor power is on and the bus cable reaches the first motor. If the "
            "motors are new, assign IDs and baud with `lerobot-setup-motors`."
        )
    elif missing:
        fails.append("no reply from " + ", ".join(_motor_label(m) for m in missing))
        fixes.append(
            "Check the daisy-chain cable at those motors. IDs are assigned with "
            "`lerobot-setup-motors`."
        )
    wrong = {m: v for m, v in facts.models.items() if v is not None and v != STS3215_MODEL_NUMBER}
    if wrong:
        fails.append(
            "unexpected model numbers: "
            + ", ".join(f"{_motor_label(m)} reports {v}" for m, v in wrong.items())
            + f" (STS3215 is {STS3215_MODEL_NUMBER}, tables.py MODEL_NUMBER_TABLE)"
        )
        fixes.append("Replace the motor with an STS3215 or check for an ID clash.")
    if fails:
        return _result(
            ctx,
            "motors",
            arm,
            "fail",
            _joined(fails),
            remediation=" ".join(fixes),
            measured=measured,
            source=ctx.bus_source,
            references=["lerobot-0.6.1 tables.py:242-247", "motors_bus.py:948 (ping)"],
        )
    flagged = {m: v for m, v in status_raw.items() if v}
    if flagged:
        return _result(
            ctx,
            "motors",
            arm,
            "warn",
            "nonzero Status register on "
            + ", ".join(
                f"{_motor_label(m)} 0x{v:02X} ({', '.join(decode_status(v)) or 'unknown'})"
                for m, v in flagged.items()
            )
            + " (bit meanings unverified)",
            remediation="Read the motor's status in Feetech's software; power-cycle and re-run.",
            measured=measured,
            source=ctx.bus_source,
        )
    return _result(
        ctx,
        "motors",
        arm,
        "pass",
        f"IDs 1-6 answer as STS3215 (model {STS3215_MODEL_NUMBER}) on {facts.port}",
        measured=measured,
        source=ctx.bus_source,
        references=["lerobot-0.6.1 tables.py:242-247"],
    )


def check_firmware(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    assert arm is not None
    skipped = _bus_skip(ctx, "firmware", arm)
    if skipped is not None:
        return skipped
    facts = ctx.buses[arm]
    versions = facts.firmware
    measured = {
        "firmware": versions,
        "errors": facts.firmware_errors,
        "not_checked": facts.missing,
        "minimum_version": None,
    }
    references = ["lerobot-0.6.1 feetech.py:145-157", "feetech.py:431-448"]
    if facts.firmware_errors:
        return _result(
            ctx,
            "firmware",
            arm,
            "fail",
            "could not read firmware from "
            + ", ".join(_motor_label(m) for m in facts.firmware_errors)
            + "; LeRobot's connect handshake reads it from every motor and fails",
            remediation="Check the cable to those motors and re-run.",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    distinct = sorted(set(versions.values()))
    if len(distinct) > 1:
        return _result(
            ctx,
            "firmware",
            arm,
            "fail",
            "motors report different firmware versions ("
            + ", ".join(f"{m} {v}" for m, v in versions.items())
            + "); LeRobot refuses to connect (feetech.py:145-153)",
            remediation="Update every motor to the same firmware with Feetech's software.",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    version = distinct[0] if distinct else "?"
    if facts.missing:
        return _result(
            ctx,
            "firmware",
            arm,
            "warn",
            f"{len(versions)} of {len(JOINT_NAMES)} motors report firmware {version}; the "
            "others did not answer (see motors)",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    return _result(
        ctx,
        "firmware",
        arm,
        "pass",
        f"all {len(versions)} motors report firmware {version}. LeRobot 0.6.1 only requires "
        "them to match; no minimum version is verified",
        measured=measured,
        source=ctx.bus_source,
        references=references,
    )


def check_supply_voltage(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    assert arm is not None
    skipped = _bus_skip(ctx, "supply_voltage", arm)
    if skipped is not None:
        return skipped
    facts = ctx.buses[arm]
    thresholds = ctx.thresholds(arm)
    raw = facts.register("Present_Voltage")
    errors = facts.register_errors.get("Present_Voltage", {})
    volts = {motor: convert_voltage(value) for motor, value in raw.items()}
    limits_raw = (facts.register("Min_Voltage_Limit"), facts.register("Max_Voltage_Limit"))
    servo_limits = {
        motor: [convert_voltage(limits_raw[0][motor]), convert_voltage(limits_raw[1][motor])]
        for motor in volts
        if motor in limits_raw[0] and motor in limits_raw[1]
    }
    critical_low, warn_low, warn_high, critical_high = thresholds.voltage_band
    measured = {
        "profile": thresholds.supply_profile,
        "band_v": {
            "critical_low": critical_low,
            "warn_low": warn_low,
            "warn_high": warn_high,
            "critical_high": critical_high,
        },
        "volts": volts,
        "raw": raw,
        "servo_limits_v": servo_limits,
        "errors": errors,
        "scale": "0.1 V/LSB (unverified)",
    }
    fails: list[str] = []
    warns: list[str] = []
    if errors:
        fails.append("could not read Present_Voltage from " + ", ".join(errors))
    for motor, value in volts.items():
        level, threshold, relation = grade_voltage(value, thresholds)
        if level == "critical":
            fails.append(
                f"{motor} {value:g} V is {relation} the critical threshold {threshold:g} V"
            )
        elif level == "warn":
            warns.append(f"{motor} {value:g} V is {relation} the warn threshold {threshold:g} V")
        limits = servo_limits.get(motor)
        if limits and not limits[0] <= value <= limits[1]:
            warns.append(
                f"{motor} {value:g} V is outside the servo's own limits "
                f"{limits[0]:g}-{limits[1]:g} V (Min/Max_Voltage_Limit)"
            )
    env = f"SO101_HEALTH_{arm.upper()}_SUPPLY"
    remediation = (
        f"Measure the {arm} supply. The {thresholds.supply_profile} profile expects "
        f"{warn_low:g}-{warn_high:g} V at the servos; if this arm uses the other servo "
        f"variant, set {env}. Values use the unverified 0.1 V/LSB scale."
    )
    if fails:
        return _result(
            ctx,
            "supply_voltage",
            arm,
            "fail",
            _joined(fails + warns),
            remediation=remediation,
            measured=measured,
            source=ctx.bus_source,
        )
    if warns:
        return _result(
            ctx,
            "supply_voltage",
            arm,
            "warn",
            _joined(warns),
            remediation=remediation,
            measured=measured,
            source=ctx.bus_source,
        )
    low, high = min(volts.values()), max(volts.values())
    observed = f"{low:g} V" if low == high else f"{low:g}-{high:g} V"
    return _result(
        ctx,
        "supply_voltage",
        arm,
        "pass",
        f"supply {observed} on {len(volts)} motors, inside the "
        f"{thresholds.supply_profile} band ({warn_low:g}-{warn_high:g} V; scale unverified)",
        measured=measured,
        source=ctx.bus_source,
    )


def check_temperature(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    assert arm is not None
    skipped = _bus_skip(ctx, "temperature", arm)
    if skipped is not None:
        return skipped
    facts = ctx.buses[arm]
    thresholds = ctx.thresholds(arm)
    raw = facts.register("Present_Temperature")
    limits = facts.register("Max_Temperature_Limit")
    errors = facts.register_errors.get("Present_Temperature", {})
    temps = {motor: convert_temperature(value) for motor, value in raw.items()}
    graded: dict[str, dict[str, Any]] = {}
    fails: list[str] = []
    warns: list[str] = []
    if errors:
        fails.append("could not read Present_Temperature from " + ", ".join(errors))
    for motor, value in temps.items():
        limit_c = convert_temperature(limits[motor]) if motor in limits else None
        level, threshold, basis = grade_temperature(value, limit_c, thresholds)
        graded[motor] = {
            "temperature_c": value,
            "limit_c": limit_c,
            "warn_c": basis.warn_c,
            "critical_c": basis.critical_c,
            "basis": basis.basis,
            "level": level,
        }
        if level == "critical":
            fails.append(f"{motor} {value:g} degC reached the critical threshold {threshold:g}")
        elif level == "warn":
            warns.append(f"{motor} {value:g} degC reached the warn threshold {threshold:g}")
    measured = {"motors": graded, "errors": errors, "scale": "1 degC/LSB (unverified)"}
    remediation = (
        "Let the arm cool with power off and find the cause (a stalled joint, a load "
        "against a hard stop) before moving it."
    )
    if fails:
        return _result(
            ctx,
            "temperature",
            arm,
            "fail",
            _joined(fails + warns),
            remediation=remediation,
            measured=measured,
            source=ctx.bus_source,
        )
    if warns:
        return _result(
            ctx,
            "temperature",
            arm,
            "warn",
            _joined(warns),
            remediation=remediation,
            measured=measured,
            source=ctx.bus_source,
        )
    hottest = max(temps.values()) if temps else float("nan")
    return _result(
        ctx,
        "temperature",
        arm,
        "pass",
        f"hottest motor {hottest:g} degC, below every warn threshold (scale unverified)",
        measured=measured,
        source=ctx.bus_source,
    )


def validate_calibration_payload(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, tuple[int, int, int]]]:
    """Structural checks of a LeRobot calibration file: (fails, warns, entries)."""
    fails: list[str] = []
    warns: list[str] = []
    entries: dict[str, tuple[int, int, int]] = {}
    names = set(payload)
    if names != set(JOINT_NAMES):
        fails.append(f"motor names {sorted(names)} differ from {list(JOINT_NAMES)}")
    for name in JOINT_NAMES:
        entry = payload.get(name)
        if entry is None:
            continue
        try:
            motor_id = int(entry["id"])
            low, high = int(entry["range_min"]), int(entry["range_max"])
            offset = int(entry["homing_offset"])
        except (KeyError, TypeError, ValueError):
            fails.append(f"{name}: malformed entry")
            continue
        entries[name] = (low, high, offset)
        if motor_id != JOINT_IDS[name]:
            fails.append(f"{name}: id {motor_id}, expected {JOINT_IDS[name]}")
        top = STS3215_RESOLUTION - 1
        if not 0 <= low < high <= top:
            fails.append(f"{name}: range {low}..{high} is empty or outside 0..{top}")
        elif name == "wrist_roll":
            if (low, high) != (0, top):
                warns.append(
                    f"wrist_roll range {low}..{high}; LeRobot calibrates it as the full turn "
                    f"0..{top} (so_follower.py:142-143)"
                )
        elif high - low < MIN_CALIBRATED_SPAN_TICKS:
            fails.append(
                f"{name}: calibrated span {high - low} ticks is below the Suite's own "
                f"{MIN_CALIBRATED_SPAN_TICKS}-tick minimum"
            )
        elif high - low < WARN_CALIBRATED_SPAN_TICKS:
            span = high - low
            warns.append(
                f"{name}: calibrated span {span} ticks ({span * DEGREES_PER_TICK:.0f} deg) is "
                "small; sweep the joint fully when calibrating"
            )
    return fails, warns, entries


_IDENTITY_REGISTERS = (
    ("Min_Position_Limit", 0, "range_min"),
    ("Max_Position_Limit", 1, "range_max"),
    ("Homing_Offset", 2, "homing_offset"),
)


def check_calibration(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    assert arm is not None
    if ctx.backend == "sim":
        return _result(
            ctx,
            "calibration",
            arm,
            "skip",
            "the sim backend has no motor calibration",
            blocking=False,
        )
    facts = ctx.calibration[arm]
    real = ctx.backend == "real"
    measured: dict[str, Any] = {
        "device_id": facts.device_id,
        "lerobot_path": facts.lerobot_path,
        "lerobot_exists": facts.lerobot_exists,
        "lerobot_sha256": facts.lerobot_sha256,
        "legacy_path": facts.legacy_path,
        "legacy_exists": facts.legacy_exists,
        "suite_path": facts.suite_path,
        "suite_exists": facts.suite_exists,
        "suite_device_id": facts.suite_device_id,
        "suite_mock": facts.suite_mock,
        "synthetic": facts.synthetic,
    }
    references = [
        "lerobot-0.6.1 feetech.py:227-266 (is_calibrated, read_calibration)",
        "robots/robot.py:46-56",
        "teleoperators/teleoperator.py:45-56",
    ]
    remediation = (
        f"Recalibrate with `so101 calibrate {arm}` (server stopped). It writes the motors, "
        "LeRobot's file and the Suite's file together."
    )
    fails: list[str] = []
    warns: list[str] = []
    entries: dict[str, tuple[int, int, int]] = {}
    if facts.lerobot_error:
        fails.append(
            f"LeRobot calibration {facts.lerobot_path} is unreadable: {facts.lerobot_error}"
        )
    elif facts.lerobot_payload is None:
        text = (
            f"no LeRobot calibration at {facts.lerobot_path}; the Suite will not read positions "
            "until the arm is calibrated"
        )
        if facts.legacy_exists:
            text += (
                f". A file exists at {facts.legacy_path}, a pre-0.6 location (unverified) that "
                "0.6.1 does not read"
            )
        fails.append(text)
    else:
        structural_fails, structural_warns, entries = validate_calibration_payload(
            facts.lerobot_payload
        )
        fails.extend(structural_fails)
        warns.extend(structural_warns)
        measured["ranges"] = {
            name: {"range_min": low, "range_max": high, "homing_offset": offset}
            for name, (low, high, offset) in entries.items()
        }

    if facts.suite_error:
        fails.append(f"Suite calibration {facts.suite_path} is invalid: {facts.suite_error}")
    elif facts.suite_exists and real and facts.suite_mock:
        # Shared data folder: a mock run's file is no calibration of this arm.
        warns.append(
            f"the Suite calibration at {facts.suite_path} was made on the mock backend, not on "
            "this arm, so it counts as missing"
            + (
                "; SafetyGuard falls back to its conservative default limits. Recalibrate on "
                "the real arm"
                if arm == "follower"
                else ". Recalibrate on the real arm"
            )
        )
    elif facts.suite_exists:
        if facts.suite_device_id and facts.suite_device_id not in {"mock", facts.device_id}:
            fails.append(
                f"Suite calibration belongs to {facts.suite_device_id}; this arm is "
                f"{facts.device_id}"
            )
    elif real:
        if arm == "follower":
            warns.append(
                f"no Suite calibration at {facts.suite_path}; SafetyGuard falls back to its "
                "conservative default limits (follower.py guard_from_saved_calibration)"
            )
        else:
            warns.append(f"no Suite calibration at {facts.suite_path}")

    identity: dict[str, Any] = {"compared": [], "mismatches": [], "unread": []}
    bus_skip: CheckResult | None = None
    if not fails and entries:
        bus_skip = _bus_skip(ctx, "calibration", arm)
        bus = ctx.buses.get(arm)
        if bus_skip is None and bus is not None:
            for motor in bus.responding:
                if motor not in entries:
                    continue
                identity["compared"].append(motor)
                for register, index, key in _IDENTITY_REGISTERS:
                    actual = bus.register(register).get(motor)
                    expected = entries[motor][index]
                    if actual is None:
                        identity["unread"].append(f"{motor}.{register}")
                    elif actual != expected:
                        identity["mismatches"].append(
                            f"{motor}.{key}: motor {actual}, file {expected}"
                        )
            if identity["unread"]:
                fails.append(
                    "could not read calibration registers: " + ", ".join(identity["unread"])
                )
            if identity["mismatches"]:
                fails.append(
                    "motor registers differ from the calibration file ("
                    + ", ".join(identity["mismatches"])
                    + "). LeRobot's is_calibrated is then False and the Suite refuses to read "
                    "positions"
                )
                remediation = (
                    f"Check SO101_{arm.upper()}_ID names this arm. If the motors or the arm "
                    f"changed, recalibrate with `so101 calibrate {arm}` (server stopped)."
                )
            if bus.missing:
                warns.append(
                    f"calibration registers compared on {len(identity['compared'])} of "
                    f"{len(JOINT_NAMES)} motors"
                )
            if bus.position_samples:
                latest = bus.position_samples[-1]
                outside = {
                    motor: value
                    for motor, value in latest.items()
                    if motor in entries
                    and not (
                        entries[motor][0] - RANGE_TOLERANCE_TICKS
                        <= value
                        <= entries[motor][1] + RANGE_TOLERANCE_TICKS
                    )
                }
                if outside:
                    warns.append(
                        "joints rest outside their calibrated range: "
                        + ", ".join(
                            f"{m} at {v} (range {entries[m][0]}..{entries[m][1]})"
                            for m, v in outside.items()
                        )
                    )
    measured["identity"] = identity
    source: MeasuredSource = ctx.bus_source if facts.synthetic or identity["compared"] else "host"

    if fails:
        return _result(
            ctx,
            "calibration",
            arm,
            "fail",
            _joined(fails + warns),
            remediation=remediation,
            measured=measured,
            source=source,
            references=references,
        )
    if bus_skip is not None:
        measured["warnings"] = warns
        return _result(
            ctx,
            "calibration",
            arm,
            "skip",
            "calibration files are well-formed but the motor registers were not compared: "
            + bus_skip.message,
            remediation=bus_skip.remediation,
            measured=measured,
            blocking=bus_skip.blocking,
            source="host",
            references=references,
        )
    if warns:
        return _result(
            ctx,
            "calibration",
            arm,
            "warn",
            _joined(warns),
            remediation=remediation,
            measured=measured,
            source=source,
            references=references,
        )
    what = "synthetic calibration" if facts.synthetic else f"LeRobot calibration {facts.device_id}"
    return _result(
        ctx,
        "calibration",
        arm,
        "pass",
        f"{what} matches the motor registers (range and homing offset on all six motors)",
        measured=measured,
        source=source,
        references=references,
    )


def check_rest_pose(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    arm = arm or "follower"
    remediation = (
        "Fold the arm into its rest, then run `so101 rest-pose record` with the server stopped."
    )
    if ctx.backend == "sim":
        return _result(
            ctx,
            "rest_pose",
            arm,
            "skip",
            "the simulated arm has no recorded rest pose to check; Collapse uses "
            "SO101_REST_POSE or the model-derived default",
            blocking=False,
        )
    facts = ctx.rest_pose
    if facts is None or facts.resolution is None:
        error = facts.error if facts is not None else "not evaluated"
        return _result(
            ctx,
            "rest_pose",
            arm,
            "skip" if ctx.backend == "real" else "warn",
            f"could not resolve the rest pose: {error}",
            remediation=remediation,
        )
    resolution = facts.resolution
    measured = {
        "source": resolution.source,
        "joints": resolution.joints,
        "warnings": list(resolution.warnings),
        "placeholder": False,
        "clamped": {joint: list(pair) for joint, pair in resolution.clamped.items()},
        "calibration_digest": resolution.calibration_digest,
        "recorded": resolution.recorded,
        "limits_source": facts.limits_source,
    }
    references = ["docs/plans/2026-10-01-suite-v02-design.md section 5 (Rest pose)"]
    if resolution.source == "recorded":
        if resolution.clamped:
            return _result(
                ctx,
                "rest_pose",
                arm,
                "skip" if ctx.backend == "real" else "warn",
                "the recorded rest pose lies outside the SafetyGuard ∩ MJCF limits and was "
                "clamped: " + _joined(resolution.warnings),
                remediation=remediation,
                measured=measured,
                references=references,
            )
        recorded_at = (resolution.recorded or {}).get("recorded_at", "?")
        return _result(
            ctx,
            "rest_pose",
            arm,
            "pass",
            f"recorded rest pose (recorded {recorded_at}) matches follower "
            f"{resolution.follower_id} and its calibration",
            measured=measured,
            references=references,
        )
    reasons = [w for w in resolution.warnings if w.startswith("ignored the recorded")]
    if reasons:
        why = _joined(reasons)
    elif resolution.recorded is not None:
        why = "the recorded rest pose does not match"
    else:
        why = "no rest pose has been recorded"
    fallback = (
        "Collapse would use SO101_REST_POSE"
        if resolution.source == "setting"
        else "Collapse would use the model-derived default pose, which is folded and "
        "gravity-stable in MuJoCo but not verified on this arm"
    )
    if resolution.clamped:
        fallback += ", clamped to the SafetyGuard ∩ MJCF limits"
    return _result(
        ctx,
        "rest_pose",
        arm,
        "skip" if ctx.backend == "real" else "warn",
        f"{why}; {fallback}",
        remediation=remediation,
        measured=measured,
        references=references,
    )


def check_torque_off_stability(ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    assert arm is not None
    skipped = _bus_skip(ctx, "torque_off_stability", arm)
    if skipped is not None:
        return skipped
    facts = ctx.buses[arm]
    torque = facts.register("Torque_Enable")
    torque_errors = facts.register_errors.get("Torque_Enable", {})
    samples = facts.position_samples
    spreads: dict[str, int] = {}
    for motor in facts.responding:
        values = [sample[motor] for sample in samples if motor in sample]
        if values:
            spreads[motor] = max(values) - min(values)
    measured = {
        "torque_enable": torque,
        "torque_errors": torque_errors,
        "samples": len(samples),
        "read_errors": facts.position_errors,
        "spread_ticks": spreads,
        "spread_deg": {m: round(v * DEGREES_PER_TICK, 3) for m, v in spreads.items()},
        "degrees_per_tick": round(DEGREES_PER_TICK, 6),
        "stable_spread_ticks": STABLE_SPREAD_TICKS,
    }
    references = [
        "lerobot-0.6.1 motors_bus.py:874-877 (360/4095 deg per tick)",
        "lerobot-0.6.1 tables.py:75-76 (Torque_Enable is in the SRAM section)",
        "so101 hardware/motors.py:195-201 (disable_torque on connect)",
    ]
    if torque_errors:
        return _result(
            ctx,
            "torque_off_stability",
            arm,
            "fail",
            "could not read Torque_Enable from " + ", ".join(torque_errors),
            remediation="Check the cable to those motors and re-run.",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    energized = [motor for motor, value in torque.items() if value]
    if energized:
        return _result(
            ctx,
            "torque_off_stability",
            arm,
            "fail",
            "torque is ON for "
            + ", ".join(_motor_label(m) for m in energized)
            + " while nothing should be driving the arm. The Suite turns torque off when it "
            "connects, and the arm will then drop",
            remediation=(
                "Support the arm by hand, then switch the motor power off and on again "
                "(Torque_Enable is in the SRAM section, so it is not stored). Re-run preflight "
                "before starting the server."
            ),
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    if len(facts.position_errors) > 1 or not samples:
        return _result(
            ctx,
            "torque_off_stability",
            arm,
            "fail",
            f"{len(facts.position_errors)} of {len(facts.position_errors) + len(samples)} "
            "position reads failed",
            remediation="Check the bus cable and connectors, then re-run.",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    top = STS3215_RESOLUTION - 1
    implausible = sorted(
        {motor for sample in samples for motor, value in sample.items() if not 0 <= value <= top}
    )
    if implausible:
        return _result(
            ctx,
            "torque_off_stability",
            arm,
            "fail",
            f"positions outside 0..{top} from " + ", ".join(implausible),
            remediation="Power-cycle the motors; if it persists, the encoder may be faulty.",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    moving = {m: v for m, v in spreads.items() if v > STABLE_SPREAD_TICKS}
    if moving:
        worst = max(moving.values())
        return _result(
            ctx,
            "torque_off_stability",
            arm,
            "warn",
            "positions changed while the arm should be still: "
            + ", ".join(f"{m} {v} ticks" for m, v in moving.items())
            + f" (up to {worst * DEGREES_PER_TICK:.2f} deg)",
            remediation="Keep the arm still and resting (no hand on it), then re-run.",
            measured=measured,
            source=ctx.bus_source,
            references=references,
        )
    note = " (1 read retried)" if facts.position_errors else ""
    message = (
        f"torque is off on all {len(torque)} motors and {len(samples)} position reads are "
        f"steady (max spread {max(spreads.values(), default=0)} ticks){note}"
    )
    status: CheckStatus = "warn" if facts.position_errors else "pass"
    return _result(
        ctx,
        "torque_off_stability",
        arm,
        status,
        message,
        remediation="One position read failed; check the bus cable." if note else "",
        measured=measured,
        source=ctx.bus_source,
        references=references,
    )


def _realsense_warnings(probe: CameraProbe) -> list[tuple[str, str]]:
    """(message, remediation) pairs for an opened RealSense: USB 2, old firmware, sparse depth."""
    from ..hardware.realsense import usb2_hint

    device = probe.device or {}
    found: list[tuple[str, str]] = []
    if device.get("usb2"):
        link = device.get("usb_link_mbps")
        label = f"{link:g} Mbit/s" if link else f"USB {device.get('usb_type')}"
        found.append((f"{probe.camera}: USB 2 link ({label})", usb2_hint(label)))
    if device.get("firmware_outdated"):
        found.append(
            (
                f"{probe.camera}: firmware {device.get('firmware')} is older than the recommended "
                f"{device.get('recommended_firmware')}",
                "Update the camera firmware (docs/realsense.md, Firmware).",
            )
        )
    depth = (probe.streams or {}).get("depth") or {}
    fraction = depth.get("valid_fraction")
    if fraction is not None and fraction < DEPTH_MIN_VALID_FRACTION:
        found.append(
            (
                f"{probe.camera}: only {fraction:.0%} of depth pixels are valid",
                "Aim the camera at the workspace from within its depth range, clear the lenses, "
                "and light the scene.",
            )
        )
    return found


def check_cameras(ctx: PreflightContext, arm: ArmName | None = None) -> CheckResult:
    cameras = ctx.cameras
    measured: dict[str, Any] = {
        "configured": cameras.configured,
        "source": cameras.source,
        "live": cameras.live,
        "probes": [probe.__dict__ for probe in cameras.probes],
        "dark_frame_mean": DARK_FRAME_MEAN,
        "depth_min_valid_fraction": DEPTH_MIN_VALID_FRACTION,
    }
    if ctx.backend == "sim":
        return _result(
            ctx,
            "cameras",
            None,
            "skip",
            SIM_CAMERAS_SKIP,
            measured=measured,
            blocking=False,
            source="sim",
        )
    if not cameras.configured:
        return _result(
            ctx,
            "cameras",
            None,
            "skip",
            "no cameras configured (SO101_CAMERAS is empty)",
            measured=measured,
            blocking=False,
        )
    if cameras.skipped_reason:
        return _result(
            ctx,
            "cameras",
            None,
            "skip",
            cameras.skipped_reason,
            measured=measured,
            blocking=False,
            source=cameras.source,
        )
    if cameras.live is not None:
        missing = [camera for camera in cameras.configured if camera not in cameras.live]
        if not missing:
            return _result(
                ctx,
                "cameras",
                None,
                "pass",
                "live frames from " + ", ".join(cameras.configured) + " (reported by the server)",
                measured=measured,
                source=cameras.source,
            )
        return _result(
            ctx,
            "cameras",
            None,
            "skip",
            "the server owns the cameras and reports no live frame from "
            + ", ".join(missing)
            + "; start a camera-using mode to check them",
            measured=measured,
            blocking=False,
            source=cameras.source,
        )
    failed = [probe for probe in cameras.probes if not probe.ok]
    dark = [
        probe
        for probe in cameras.probes
        if probe.ok and probe.mean is not None and probe.mean < DARK_FRAME_MEAN
    ]
    if failed:
        # A RealSense failure carries its own OS-specific fix (udev rules, USB 3,
        # sudo, another process holding the camera); an OpenCV one does not.
        fixes = dict.fromkeys(probe.remediation for probe in failed if probe.remediation)
        return _result(
            ctx,
            "cameras",
            None,
            "fail",
            "; ".join(f"camera {probe.camera}: {probe.error}" for probe in failed),
            remediation=" ".join(fixes) or "Reconnect the camera or remove it from SO101_CAMERAS.",
            measured=measured,
            source=cameras.source,
        )
    warnings: list[tuple[str, str]] = []
    if dark:
        warnings.append(
            (
                "black frames from " + ", ".join(probe.camera for probe in dark),
                "Remove the lens cap, check lighting, or check the camera index.",
            )
        )
    for probe in cameras.probes:
        if probe.ok and probe.kind == "realsense":
            warnings += _realsense_warnings(probe)
    if warnings:
        return _result(
            ctx,
            "cameras",
            None,
            "warn",
            "; ".join(message for message, _ in warnings),
            remediation=" ".join(dict.fromkeys(fix for _, fix in warnings)),
            measured=measured,
            source=cameras.source,
        )
    return _result(
        ctx,
        "cameras",
        None,
        "pass",
        f"opened and read {len(cameras.probes)} camera(s): " + ", ".join(cameras.configured),
        measured=measured,
        source=cameras.source,
    )


def check_disk_space(ctx: PreflightContext, arm: ArmName | None = None) -> CheckResult:
    fails: list[str] = []
    warns: list[str] = []
    measured: dict[str, Any] = {}
    for facts in ctx.disks:
        free_gib = None if facts.free_bytes is None else round(facts.free_bytes / 1024**3, 2)
        measured[facts.role] = {
            "path": facts.path,
            "exists": facts.exists,
            "writable": facts.writable,
            "free_gib": free_gib,
            "total_gib": None
            if facts.total_bytes is None
            else round(facts.total_bytes / 1024**3, 2),
            "error": facts.error,
        }
        if facts.error or facts.free_bytes is None:
            fails.append(f"{facts.role} {facts.path}: {facts.error or 'unknown free space'}")
            continue
        if not facts.writable:
            fails.append(f"{facts.role} {facts.path} is not writable")
        if facts.free_bytes < DISK_FAIL_BYTES:
            fails.append(f"{facts.role} has {free_gib} GiB free (< 2 GiB)")
        elif facts.free_bytes < DISK_WARN_BYTES:
            warns.append(f"{facts.role} has {free_gib} GiB free (< 10 GiB)")
    remediation = "Free disk space or point SO101_DATA_DIR / DATASETS_DIR / OUTPUTS_DIR elsewhere."
    if fails:
        return _result(
            ctx,
            "disk_space",
            None,
            "fail",
            _joined(fails + warns),
            remediation=remediation,
            measured=measured,
        )
    if warns:
        return _result(
            ctx,
            "disk_space",
            None,
            "warn",
            _joined(warns),
            remediation=remediation,
            measured=measured,
        )
    smallest = min(
        (facts.free_bytes for facts in ctx.disks if facts.free_bytes is not None), default=0
    )
    return _result(
        ctx,
        "disk_space",
        None,
        "pass",
        f"data, datasets and outputs are writable with at least {smallest / 1024**3:.1f} GiB free",
        measured=measured,
    )


def check_estop(ctx: PreflightContext, arm: ArmName | None = None) -> CheckResult:
    estop = ctx.estop
    measured = {
        "known": estop.known,
        "estopped": estop.estopped,
        "fault": estop.fault,
        "fault_known": estop.fault_known,
        "mode": estop.mode,
        "source": estop.source,
    }
    if not estop.known:
        return _result(
            ctx,
            "estop",
            None,
            "skip",
            estop.detail,
            measured=measured,
            blocking=False,
            source=estop.source,
        )
    if estop.estopped:
        return _result(
            ctx,
            "estop",
            None,
            "warn",
            "the software E-stop is latched",
            remediation=(
                "Inspect the workspace, then reset it in the UI or with "
                'POST /api/robot/estop/reset {"operator_id": "<your name>", "confirm": true}.'
            ),
            measured=measured,
            source=estop.source,
        )
    if estop.fault:
        return _result(
            ctx,
            "estop",
            None,
            "warn",
            f"a runtime fault is latched: {estop.fault}",
            remediation=(
                "Fix the cause, then clear it in the UI or with POST /api/robot/fault/reset "
                '{"operator_id": "<your name>", "confirm": true}; add '
                '"power_cut_acknowledged": true only after cutting the arm\'s power when '
                "torque-off stays unconfirmed."
            ),
            measured=measured,
            source=estop.source,
        )
    return _result(
        ctx,
        "estop",
        None,
        "pass",
        "the software E-stop is not latched"
        + (
            " and no runtime fault is latched"
            if estop.fault_known
            else " (runtime faults are not part of /api/robot/status)"
        ),
        measured=measured,
        source=estop.source,
    )


CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        "environment",
        "Environment",
        "Python, platform, backend, baud rate and LeRobot environment variables; on the sim "
        "backend, that the simulated arm (MuJoCo) loads.",
        check_environment,
    ),
    CheckSpec(
        "lerobot",
        "LeRobot 0.6.1 surface",
        "LeRobot version, Feetech SDK, symbols, signatures and the STS3215 control table.",
        check_lerobot,
    ),
    CheckSpec(
        "serial_ports",
        "Serial ports",
        "Configured follower and leader ports exist, are usable and distinct.",
        check_serial_ports,
    ),
    CheckSpec(
        "motors",
        "Motors",
        "IDs 1-6 answer a ping with the STS3215 model number; Status register is clear.",
        check_motors,
        per_arm=True,
        needs_bus=True,
    ),
    CheckSpec(
        "firmware",
        "Firmware",
        "All motors on an arm report the same firmware (LeRobot's connect requirement).",
        check_firmware,
        per_arm=True,
        needs_bus=True,
    ),
    CheckSpec(
        "supply_voltage",
        "Supply voltage",
        "Present_Voltage inside the arm's supply band.",
        check_supply_voltage,
        per_arm=True,
        needs_bus=True,
    ),
    CheckSpec(
        "temperature",
        "Temperature",
        "Present_Temperature below thresholds derived from Max_Temperature_Limit.",
        check_temperature,
        per_arm=True,
        needs_bus=True,
    ),
    CheckSpec(
        "calibration",
        "Calibration",
        "LeRobot and Suite calibration files exist, are sane and match the motor registers.",
        check_calibration,
        per_arm=True,
        needs_bus=True,
    ),
    CheckSpec(
        "rest_pose",
        "Rest pose",
        "A rest pose recorded for this follower and calibration exists (Collapse target).",
        check_rest_pose,
        per_arm=True,
        arms=("follower",),
    ),
    CheckSpec(
        "torque_off_stability",
        "Torque off and steady",
        "Every motor has torque off and positions read steadily while the arm rests.",
        check_torque_off_stability,
        per_arm=True,
        needs_bus=True,
    ),
    CheckSpec(
        "cameras",
        "Cameras",
        "Configured cameras open and return non-black frames.",
        check_cameras,
    ),
    CheckSpec(
        "disk_space",
        "Disk space",
        "Data, datasets and outputs directories are writable with enough free space.",
        check_disk_space,
    ),
    CheckSpec(
        "estop",
        "E-stop latch",
        "The software E-stop and runtime fault are not latched.",
        check_estop,
    ),
)
CHECKS_BY_ID: dict[str, CheckSpec] = {spec.id: spec for spec in CHECKS}


def check_catalog() -> list[PreflightCheckInfo]:
    return [
        PreflightCheckInfo(
            id=spec.id,
            title=spec.title,
            per_arm=spec.per_arm,
            needs_bus=spec.needs_bus,
            description=spec.description,
        )
        for spec in CHECKS
    ]


def _run_one(spec: CheckSpec, ctx: PreflightContext, arm: ArmName | None) -> CheckResult:
    started = time.perf_counter()
    try:
        result = spec.fn(ctx, arm)
    except Exception as exc:  # a crashing check must not hide the others
        result = _result(
            ctx,
            spec.id,
            arm,
            "fail",
            f"the check crashed: {type(exc).__name__}: {exc}",
            remediation="This is a bug in the bring-up kit; report it with the --json output.",
        )
    result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return result


def run_checks(ctx: PreflightContext) -> list[CheckResult]:
    results: list[CheckResult] = []
    for spec in CHECKS:
        if spec.per_arm:
            results.extend(_run_one(spec, ctx, arm) for arm in ctx.arms if arm in spec.arms)
        else:
            results.append(_run_one(spec, ctx, None))
    return results


def overall_status(results: Sequence[CheckResult]) -> OverallStatus:
    if any(result.status == "fail" for result in results):
        return "fail"
    if any(result.status == "skip" and result.blocking for result in results):
        return "incomplete"
    if any(result.status == "warn" for result in results):
        return "warn"
    return "pass"


def exit_code_for(overall: OverallStatus, *, rehearsal: bool, require_hardware: bool) -> int:
    """4 rehearsal rejected > 1 fail > 3 incomplete > 0."""
    if require_hardware and rehearsal:
        return EXIT_REHEARSAL_REJECTED
    if overall == "fail":
        return EXIT_FAIL
    if overall == "incomplete":
        return EXIT_INCOMPLETE
    return EXIT_OK


def build_report(
    ctx: PreflightContext,
    results: list[CheckResult],
    *,
    require_hardware: bool = False,
    notes: Sequence[str] = (),
) -> PreflightReport:
    counts = {status: 0 for status in ("pass", "warn", "fail", "skip")}
    for result in results:
        counts[result.status] += 1
    overall = overall_status(results)
    all_notes = list(notes)
    validated_rest_pose = any(
        result.id == "rest_pose" and result.arm == "follower" and result.status == "pass"
        for result in results
    )
    if not ctx.rehearsal and not validated_rest_pose and overall != "fail":
        # A partial/custom check set must not bypass the required real-arm
        # evidence. Normal runs reach the same result through the blocking
        # rest-pose skip, but keep this report-level invariant for callers that
        # select only the leader or construct a subset of checks.
        overall = "incomplete"
        all_notes.append(
            "Hardware readiness is incomplete until a rest pose recorded on the real follower "
            "matches its current calibration and needs no clamping."
        )
    if ctx.rehearsal:
        all_notes.insert(
            0,
            f"REHEARSAL: the {ctx.backend} backend was checked; this report is not evidence "
            "about a real arm.",
        )
    return PreflightReport(
        created_at=_now().isoformat(),
        so101_version=__version__,
        origin=ctx.origin,
        backend=ctx.backend,
        rehearsal=ctx.rehearsal,
        arms=list(ctx.arms),
        bus_access=ctx.bus_access.mode,
        bus_access_detail=ctx.bus_access.detail,
        overall=overall,
        ready_for_hardware=(not ctx.rehearsal)
        and validated_rest_pose
        and overall in ("pass", "warn"),
        exit_code=exit_code_for(
            overall, rehearsal=ctx.rehearsal, require_hardware=require_hardware
        ),
        counts=counts,
        checks=results,
        notes=all_notes,
    )


# --- runners --------------------------------------------------------------------

FakeBusFactory = Callable[[str], FakeFeetechBus]


def default_fake_bus(settings: Any) -> FakeBusFactory:
    def factory(arm: str) -> FakeFeetechBus:
        profile = (
            settings.health_leader_supply if arm == "leader" else settings.health_follower_supply
        )
        return FakeFeetechBus(arm=arm, supply_profile=profile)

    return factory


def _rehearsal_bus_facts(
    settings: Any,
    arms: Sequence[ArmName],
    factory: FakeBusFactory,
    sleep: Callable[[float], None],
) -> dict[str, BusFacts]:
    buses: dict[str, BusFacts] = {}
    for arm in arms:
        fake = factory(arm)
        try:
            buses[arm] = collect_bus_facts(
                fake, arm=arm, port=port_for(settings, arm), source="mock", sleep=sleep
            )
        finally:
            fake.disconnect(disable_torque=False)
    return buses


def _unopenable_port(
    settings: Any, arm: ArmName, ports: Mapping[str, PortFacts]
) -> BusFacts | None:
    """Facts for a port that cannot be opened at all (missing, or no permission)."""
    facts = ports[arm]
    if facts.exists and not _permission_denied(facts):
        return None
    port = port_for(settings, arm)
    why = "does not exist" if not facts.exists else "may not be opened by this user"
    return BusFacts(
        arm=arm,
        port=port,
        source="real",
        attempted=True,
        opened=False,
        open_error=f"{port} {why} (see serial_ports)",
    )


def _real_bus_facts(
    settings: Any,
    arms: Sequence[ArmName],
    ports: Mapping[str, PortFacts],
    opener: BusOpener,
    sleep: Callable[[float], None],
) -> dict[str, BusFacts]:
    buses: dict[str, BusFacts] = {}
    for arm in arms:
        unopenable = _unopenable_port(settings, arm, ports)
        if unopenable is not None:
            buses[arm] = unopenable
        else:
            port = port_for(settings, arm)
            buses[arm] = diagnose_port(port, arm=arm, source="real", opener=opener, sleep=sleep)
    return buses


def collect_manager_bus_facts(
    manager: Any,
    *,
    arms: Sequence[ArmName],
    ports: Mapping[str, PortFacts],
    surface: LeRobotSurface,
    bus_opener: BusOpener | None = None,
    fake_bus_factory: FakeBusFactory | None = None,
    sleep: Callable[[float], None] = time.sleep,
    bus_wait_s: float = API_BUS_WAIT_S,
) -> tuple[BusAccess, dict[str, BusFacts]]:
    """One diagnostic pass per arm, each on a bus RobotManager lends.

    ``manager.run_idle_bus_diagnostic(arm, fn, timeout_s=..., opener=...)``
    runs ``collect_bus_facts`` under the manager's bus lock on a read-only
    view of the bus, or refuses (armed, halted, collapsing, a loop,
    calibration, E-stop) or reports the bus busy. Any refusal skips every bus
    check as ``refused``. On the mock backend the manager lends the synthetic
    bus, so a rehearsal meets the same refusals. A manager without the method
    defers the real-backend checks (:data:`WAVE2_BUS_MESSAGE`).
    """
    settings = manager.settings
    backend = current_backend(settings)
    if backend == "sim":
        return BusAccess("unavailable", "the sim backend has no serial bus", blocking=False), {}
    runner = getattr(manager, "run_idle_bus_diagnostic", None)
    openers: dict[str, BusOpener]
    if backend == "mock":
        factory = fake_bus_factory or default_fake_bus(settings)
        if not callable(runner):
            access = BusAccess(
                "synthetic",
                "rehearsal: a synthetic STS3215 bus stands in for each arm; no serial port "
                "was opened",
            )
            return access, _rehearsal_bus_facts(settings, arms, factory, sleep)
        openers = {arm: partial(_fake_opener, factory, arm) for arm in arms}
        access = BusAccess(
            "synthetic",
            "rehearsal: RobotManager lent a synthetic STS3215 bus for each arm; no serial port "
            "was opened",
        )
        source = "mock"
    else:
        if not callable(runner):
            return (
                BusAccess(
                    "deferred",
                    WAVE2_BUS_MESSAGE,
                    "Stop the server and run `so101 preflight` for the bus checks.",
                ),
                {},
            )
        if _lerobot_unusable(surface):
            return (
                BusAccess(
                    "unavailable",
                    "LeRobot 0.6.1 or the Feetech SDK is not importable (see lerobot)",
                    "Fix the lerobot check first.",
                ),
                {},
            )
        opener = bus_opener or partial(open_feetech_bus, baudrate=DEFAULT_BAUDRATE)
        openers = dict.fromkeys(arms, opener)
        access = BusAccess("opened", "read-only access through RobotManager, under its bus lock")
        source = "real"
    buses: dict[str, BusFacts] = {}
    for arm in arms:
        port = port_for(settings, arm)
        unopenable = _unopenable_port(settings, arm, ports) if backend == "real" else None
        if unopenable is not None:
            buses[arm] = unopenable
            continue
        try:
            buses[arm] = runner(
                arm,
                partial(collect_bus_facts, arm=arm, port=port, source=source, sleep=sleep),
                timeout_s=bus_wait_s,
                opener=openers[arm],
            )
        except BusDiagnosticOpenError as exc:
            buses[arm] = BusFacts(
                arm=arm, port=port, source=source, attempted=True, opened=False, open_error=str(exc)
            )
        except Exception as exc:  # a refusal (409) or a busy bus (503)
            detail = f"RobotManager refused idle bus access: {exc}"
            return BusAccess("refused", detail, REFUSED_REMEDIATION), {}
    return access, buses


def _fake_opener(factory: FakeBusFactory, arm: str, port: str) -> FakeFeetechBus:
    return factory(arm)


def bus_facts_report(
    manager: Any,
    *,
    arms: Sequence[str] = ARMS,
    lerobot_prober: Callable[[], LeRobotSurface] = probe_lerobot_surface,
    port_inspector: PortInspector = collect_port_facts,
    port_lister: Callable[[], PortListing] | None = None,
    bus_opener: BusOpener | None = None,
    fake_bus_factory: FakeBusFactory | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PreflightBusFacts:
    """``POST /api/preflight/bus``: :func:`collect_manager_bus_facts` as JSON."""
    settings = manager.settings
    selected = _arms_tuple(arms)
    listing = (port_lister or list_serial_ports)()
    ports: dict[str, PortFacts] = {
        arm: port_inspector(arm, port_for(settings, arm), listing) for arm in ARMS
    }
    access, buses = collect_manager_bus_facts(
        manager,
        arms=selected,
        ports=ports,
        surface=lerobot_prober(),
        bus_opener=bus_opener,
        fake_bus_factory=fake_bus_factory,
        sleep=sleep,
    )
    return PreflightBusFacts(
        backend=current_backend(settings),
        arms=list(selected),
        bus_access=access.mode,
        bus_access_detail=access.detail,
        remediation=access.remediation,
        blocking=access.blocking,
        buses={arm: bus_facts_to_json(facts) for arm, facts in buses.items()},
    )


ServerBusFacts = Callable[
    [ServerProbe, Sequence[ArmName], Backend], tuple[BusAccess, dict[str, BusFacts]]
]


def fetch_server_bus_facts(
    probe: ServerProbe,
    arms: Sequence[ArmName],
    backend: Backend,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = SERVER_BUS_FACTS_TIMEOUT_S,
) -> tuple[BusAccess, dict[str, BusFacts]]:
    """Ask the running Suite server for its bus facts (``POST /api/preflight/bus``).

    The server's RobotManager lends each idle bus or refuses; this CLI then
    grades the facts with the same checks as a stopped-server run. Any
    failure, or a server on another backend, skips the bus checks as
    ``refused`` (fail closed). It is a POST because it borrows the bus; the
    request carries no ``Origin`` header, which the server's Origin guard
    allows (only a browser page sends one).
    """
    base = probe.url.removesuffix(STATUS_PATH)
    url = base + "/api/preflight/bus"
    params = {"arm": "both" if len(arms) == len(ARMS) else arms[0]}
    stop_and_retry = "Stop the server (Ctrl-C in its terminal) and re-run `so101 preflight`."
    try:
        limits = httpx.Timeout(timeout_s, connect=PROBE_CONNECT_TIMEOUT_S)
        with httpx.Client(timeout=limits, trust_env=False, transport=transport) as client:
            response = client.post(url, params=params)
    except httpx.HTTPError as exc:
        detail = f"could not get the bus facts from the server at {base} ({type(exc).__name__})"
        return BusAccess("refused", detail, stop_and_retry), {}
    if response.status_code in (404, 405):
        detail = f"the server at {base} does not lend its buses this way (another build)"
        return BusAccess("refused", detail, stop_and_retry), {}
    try:
        report = PreflightBusFacts.model_validate(response.json())
    except ValueError:
        detail = f"HTTP {response.status_code} from {url} is not a bus-facts reply"
        return BusAccess("refused", detail, stop_and_retry), {}
    if report.backend != backend:
        detail = (
            f"the server runs the {report.backend} backend and this command the {backend} "
            "backend, so its bus facts do not apply"
        )
        return BusAccess("refused", detail, "Use the same SO101_MOCK / SO101_BACKEND for both."), {}
    access = BusAccess(
        report.bus_access,
        f"{report.bus_access_detail} (the server at {base})",
        report.remediation,
        report.blocking,
    )
    if report.bus_access not in {"opened", "synthetic"}:
        return access, {}
    wanted = set(arms)
    buses: dict[str, BusFacts] = {
        arm: bus_facts_from_json(payload) for arm, payload in report.buses.items() if arm in wanted
    }
    return access, buses


def _rehearsal_calibration(
    backend: Backend, buses: Mapping[str, BusFacts], arm: str
) -> dict[str, dict[str, int]] | None:
    """Mock runs check the synthetic calibration, even when the bus was not read."""
    if backend != "mock":
        return None
    facts = buses.get(arm)
    if facts is not None and facts.synthetic_calibration is not None:
        return facts.synthetic_calibration
    return synthetic_calibration_for(arm)


def _lerobot_unusable(surface: LeRobotSurface) -> bool:
    return not (surface.importable and surface.sdk_importable) or surface.lerobot_home_set


def _arms_tuple(arms: Sequence[str]) -> tuple[ArmName, ...]:
    selected = tuple(arm for arm in ARMS if arm in set(arms))
    if not selected:
        raise ValueError("select at least one arm: follower, leader")
    return selected


def reports_dir(settings: Any) -> Path:
    return Path(settings.data_dir) / "preflight"


def latest_report_path(
    settings: Any, origin: PreflightOrigin | None = None, backend: Backend | None = None
) -> Path:
    """``latest-<backend>.json`` (either origin) or ``latest-<backend>-<origin>.json``.

    One set per backend (the one ``settings`` selects unless ``backend`` is
    given): real, mock and sim share the data folder, and a rehearsal must
    never stand in for the real backend's report.
    """
    backend = backend or current_backend(settings)
    name = f"latest-{backend}.json" if origin is None else f"latest-{backend}-{origin}.json"
    return reports_dir(settings) / name


def _legacy_latest_report_path(settings: Any, origin: PreflightOrigin | None) -> Path:
    """Where reports before the per-backend copies kept them, for every backend at once."""
    name = "latest.json" if origin is None else f"latest-{origin}.json"
    return reports_dir(settings) / name


def save_report(settings: Any, report: PreflightReport) -> Path:
    """Write the timestamped report plus its backend's any-origin and per-origin ``latest``.

    The per-origin copy matters on the API path: an API run defers the bus checks,
    so it must not hide the last CLI run that did read the bus.
    """
    directory = reports_dir(settings)
    stamp = datetime.fromisoformat(report.created_at).strftime("%Y%m%dT%H%M%S-%f")
    payload = report.model_dump(mode="json")
    path = directory / f"{stamp}-{report.origin}-{report.backend}.json"
    atomic_write_json(path, payload)
    backend = report.backend
    atomic_write_json(latest_report_path(settings, backend=backend), payload)
    atomic_write_json(latest_report_path(settings, report.origin, backend=backend), payload)
    return path


def _read_report(path: Path) -> PreflightReport | None:
    try:
        return PreflightReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_latest_report(
    settings: Any, origin: PreflightOrigin | None = None
) -> PreflightReport | None:
    """The latest report of the backend ``settings`` select; None when it has none.

    A report of another backend never counts: a real-backend reader after a
    sim or mock run gets None, not the rehearsal. A ``latest.json`` written
    before the copies were per backend is read only when its backend matches.
    """
    backend = current_backend(settings)
    path = latest_report_path(settings, origin, backend=backend)
    report = _read_report(path) if path.exists() else None
    if report is None and not path.exists():
        report = _read_report(_legacy_latest_report_path(settings, origin))
    if report is None or report.backend != backend:
        return None
    return report


def run_cli_preflight(
    settings: Any,
    *,
    arms: Sequence[str] = ARMS,
    require_hardware: bool = False,
    server_probe: ServerProber = probe_local_server,
    bus_opener: BusOpener | None = None,
    fake_bus_factory: FakeBusFactory | None = None,
    camera_prober: CameraProber | None = None,
    lerobot_prober: Callable[[], LeRobotSurface] = probe_lerobot_surface,
    guard_limits: Callable[[], Mapping[str, tuple[float, float]]] = saved_guard_limits,
    port_inspector: PortInspector = collect_port_facts,
    port_lister: Callable[[], PortListing] | None = None,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    write_report: bool = True,
    server_bus_facts: ServerBusFacts = fetch_server_bus_facts,
    sim_prober: Callable[[], str | None] = sim_devices_problem,
) -> PreflightReport:
    """``so101 preflight``: gather facts, run every check.

    The bus is read here only when no server runs and its device lock is free.
    With a Suite server running, its RobotManager lends the buses instead
    (``server_bus_facts``, ``POST /api/preflight/bus``).
    """
    selected = _arms_tuple(arms)
    backend = current_backend(settings)
    host = collect_host_facts(environ)
    surface = lerobot_prober()
    listing = (port_lister or list_serial_ports)()
    ports: dict[str, PortFacts] = {
        arm: port_inspector(arm, port_for(settings, arm), listing) for arm in ARMS
    }
    server = server_probe(settings)
    buses: dict[str, BusFacts] = {}

    if backend == "sim":
        access = BusAccess("unavailable", "the sim backend has no serial bus", blocking=False)
    elif server.running and server.status is not None:
        # A Suite server owns the buses: ask its RobotManager to lend them.
        access, buses = server_bus_facts(server, selected, backend)
    elif server.running:
        error = server_running_error(server)
        access = BusAccess("refused", error.detail, error.remediation)
    elif backend == "real" and _lerobot_unusable(surface):
        access = BusAccess(
            "unavailable",
            "LeRobot 0.6.1 or the Feetech SDK is not importable (see lerobot)",
            "Fix the lerobot check first.",
        )
    elif backend == "mock":
        access = BusAccess(
            "synthetic",
            "rehearsal: a synthetic STS3215 bus stands in for each arm; no serial port was opened",
        )
        buses = _rehearsal_bus_facts(
            settings, selected, fake_bus_factory or default_fake_bus(settings), sleep
        )
    else:
        try:
            lock, _ = acquire_bus_access(
                settings,
                purpose="so101 preflight",
                server_probe=lambda _settings: server,
                ports=tuple(port_for(settings, arm) for arm in selected),
            )
        except BusOwnershipError as exc:
            access = BusAccess("refused", exc.detail, exc.remediation)
        else:
            try:
                access = BusAccess(
                    "opened",
                    f"read-only access under device-global lock(s) for {lock.description}",
                )
                opener = bus_opener or (
                    lambda port: open_feetech_bus(port, baudrate=DEFAULT_BAUDRATE)
                )
                buses = _real_bus_facts(settings, selected, ports, opener, sleep)
            finally:
                lock.release()

    calibration: dict[str, CalibrationFacts] = {
        arm: collect_calibration_facts(
            settings,
            arm,
            synthetic_payload=_rehearsal_calibration(backend, buses, arm),
        )
        for arm in selected
    }
    rest_pose = None
    if "follower" in selected:
        try:
            limits: Mapping[str, tuple[float, float]] | None = guard_limits()
            limits_source = "SafetyGuard from the saved Suite calibration"
        except Exception as exc:
            limits, limits_source = None, f"default limits ({type(exc).__name__}: {exc})"
        rest_pose = collect_rest_pose_facts(
            settings, backend, limits=limits, limits_source=limits_source
        )

    entries = list(settings.camera_ids)  # to open; may hold credentials
    configured = _camera_keys(settings)  # to report; credentials stripped
    if backend == "sim":
        # The sim backend renders its cameras; never open a real one here.
        cameras = CameraFacts(configured=configured, source="sim", skipped_reason=SIM_CAMERAS_SKIP)
    elif server.running:
        cameras = cameras_from_status(configured, server.status, source="server")
    else:
        prober = camera_prober or probe_cameras_locally
        cameras = CameraFacts(
            configured=configured,
            source="mock" if backend == "mock" else "host",
            probes=prober(entries) if entries else [],
        )

    ctx = PreflightContext(
        settings=settings,
        origin="cli",
        backend=backend,
        arms=selected,
        host=host,
        lerobot=surface,
        ports=ports,
        serial_ports=listing,
        bus_access=access,
        buses=buses,
        calibration=calibration,
        rest_pose=rest_pose,
        cameras=cameras,
        disks=collect_disk_facts(settings),
        estop=estop_from_server(server),
        server=server,
        sim_devices=sim_prober() if backend == "sim" else None,
    )
    report = build_report(ctx, run_checks(ctx), require_hardware=require_hardware)
    if write_report:
        save_report(settings, report)
    return report


def run_api_preflight(
    manager: Any,
    *,
    arms: Sequence[str] = ARMS,
    bus_opener: BusOpener | None = None,
    fake_bus_factory: FakeBusFactory | None = None,
    lerobot_prober: Callable[[], LeRobotSurface] = probe_lerobot_surface,
    port_inspector: PortInspector = collect_port_facts,
    port_lister: Callable[[], PortListing] | None = None,
    environ: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    bus_wait_s: float = API_BUS_WAIT_S,
    sim_prober: Callable[[], str | None] = sim_devices_problem,
) -> PreflightReport:
    """``POST /api/preflight``: same checks, bus access only through RobotManager."""
    settings = manager.settings
    selected = _arms_tuple(arms)
    backend = current_backend(settings)
    surface = lerobot_prober()
    listing = (port_lister or list_serial_ports)()
    ports: dict[str, PortFacts] = {
        arm: port_inspector(arm, port_for(settings, arm), listing) for arm in ARMS
    }
    status = manager.status()
    access, buses = collect_manager_bus_facts(
        manager,
        arms=selected,
        ports=ports,
        surface=surface,
        bus_opener=bus_opener,
        fake_bus_factory=fake_bus_factory,
        sleep=sleep,
        bus_wait_s=bus_wait_s,
    )

    calibration: dict[str, CalibrationFacts] = {
        arm: collect_calibration_facts(
            settings,
            arm,
            synthetic_payload=_rehearsal_calibration(backend, buses, arm),
        )
        for arm in selected
    }
    rest_pose = None
    if "follower" in selected:
        guard = getattr(manager, "guard", None)
        limits = getattr(guard, "limits", None)
        rest_pose = collect_rest_pose_facts(
            settings,
            backend,
            limits=limits,
            limits_source="the running SafetyGuard" if limits else "default limits",
        )
    estop = EstopFacts(
        known=True,
        source="server",
        estopped=bool(status.get("estopped")),
        fault=getattr(manager, "fault", None),
        mode=status.get("mode"),
        fault_known=True,
    )
    ctx = PreflightContext(
        settings=settings,
        origin="api",
        backend=backend,
        arms=selected,
        host=collect_host_facts(environ),
        lerobot=surface,
        ports=ports,
        serial_ports=listing,
        bus_access=access,
        buses=buses,
        calibration=calibration,
        rest_pose=rest_pose,
        cameras=(
            CameraFacts(
                configured=_camera_keys(settings), source="sim", skipped_reason=SIM_CAMERAS_SKIP
            )
            if backend == "sim"
            else cameras_from_status(_camera_keys(settings), status, source="server")
        ),
        disks=collect_disk_facts(settings),
        estop=estop,
        sim_devices=sim_prober() if backend == "sim" else None,
    )
    return build_report(ctx, run_checks(ctx))

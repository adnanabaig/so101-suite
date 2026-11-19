"""HIL runner: steps, observe-only probes, operator attestation, sealed evidence.

One run lives in ``<data_dir>/hil/<run_id>/``::

    run.json                                        the run (source of truth)
    steps/<step_id>.json                            step definition + record
    artifacts/<step_id>/probe-<n>-<kind>.json       probe bundle: status, control
                                                    state, telemetry, audit excerpt,
                                                    health, verdict
    artifacts/<step_id>/probe-<n>-samples.jsonl     10 Hz samples (sampled probes)
    artifacts/<step_id>/probe-<n>-preflight-*.json  full preflight reports
    artifacts/<step_id>/probe-<n>-frame-<cam>.npy   optional camera frame (uint8)
    report.json, report.md                          written by finalize
    MANIFEST.sha256                                 written last by finalize

Probes only read: robot status, telemetry, the audit log, preflight and
diagnostics. They never command motion; the operator triggers every action.
A finalized run is immutable. ``hil_evidence`` is true only for a finalized
real-backend run whose every probe ran on the real backend.
"""

from __future__ import annotations

import io
import json
import math
import platform
import re
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from .. import JOINT_NAMES, __version__
from ..atomic_io import atomic_write_bytes, atomic_write_json, atomic_write_text
from ..hardware.backend import current_backend, is_rehearsal
from ..hardware.rest_pose import (
    RestPoseResolution,
    follower_calibration_digest,
    resolve_for_settings,
)
from . import preflight as pf
from .evidence import verify_manifest, write_manifest
from .hil_steps import (
    COLLAPSE_SETTLE_DEG,
    HALT_MAX_DRIFT_DEG,
    HALT_MIN_HOLD_S,
    HIL_CATALOG_DIGEST,
    HIL_CATALOG_VERSION,
    STEPS_BY_ID,
    STOP_STATUS_FIELDS,
    select_steps,
    step_catalog,
)
from .schemas import (
    Attestation,
    AttestRequest,
    CheckResult,
    CheckStatus,
    EvidenceField,
    HilReport,
    HilRun,
    HilRunSummary,
    HilStepDefinition,
    HilStepRecord,
    PreflightReport,
    ProbeKind,
    ProbeResult,
    RunStatus,
    StartHilRunRequest,
)

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
ARM_JOINTS: tuple[str, ...] = tuple(joint for joint in JOINT_NAMES if joint != "gripper")

SAMPLE_INTERVAL_S = 0.1
MAX_PROBE_S = 180.0
AUDIT_EXCERPT = 20
#: A CLI preflight report older than this is not used for deferred bus checks.
CLI_REPORT_MAX_AGE = timedelta(hours=2)

# Design §5 defaults, used until the wave-2 settings exist.
DEFAULT_HALT_HOLD_S = 60.0
DEFAULT_SUPERVISION_GRACE_S = 5.0
DEFAULT_COLLAPSE_SPEED_DEG_S = 15.0
COLLAPSE_TIMEOUT_MARGIN_S = 2.0
COLLAPSE_TIMEOUT_MAX_S = 12.0

#: Probe tolerances (the bring-up kit's choices, documented in docs/bring-up.md).
SPEED_WINDOW_S = 0.25
SPEED_TOLERANCE = 1.10
HOLD_TIMER_EARLY_S = 1.0
HOLD_TIMER_LATE_S = 3.0
SUPERVISION_TOLERANCE_S = 3.0
COLLAPSE_TRIGGER_ALLOWANCE_S = 15.0
MOVED_DEG = 1.0

SAMPLED_STOP_PROBES: frozenset[str] = frozenset(
    {"halt_hold", "collapse", "hold_timer", "supervision_loss"}
)
_OPTIONAL_STATUS_FIELDS = (*STOP_STATUS_FIELDS, "supervision")
HEALTH_UNAVAILABLE = "this server reports no servo health (RobotManager.health_snapshot is missing)"


# --- errors ----------------------------------------------------------------------


class HilError(Exception):
    status_code = 400

    def __init__(self, message: str, problems: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.problems = list(problems)


class HilNotFound(HilError):
    status_code = 404


class HilConflict(HilError):
    status_code = 409


class HilInvalid(HilError):
    status_code = 422


# --- the data source -------------------------------------------------------------


class HilClient(Protocol):
    """What probes read. Server-side it wraps RobotManager; tests fake it."""

    @property
    def settings(self) -> Any: ...

    def status(self) -> dict[str, Any]: ...

    def fault(self) -> str | None: ...

    def control_state(self) -> dict[str, Any] | None: ...

    def telemetry(self) -> dict[str, Any]: ...

    def latest_event_id(self) -> int | None: ...

    def events_after(self, after_id: int | None) -> list[dict[str, Any]]: ...

    def diagnostics(self) -> dict[str, Any]: ...

    def preflight(self) -> PreflightReport: ...

    def rest_pose(self) -> RestPoseResolution: ...

    def health(self) -> dict[str, Any] | None: ...

    def frame(self) -> tuple[str, Any] | None: ...


class LocalHilClient:
    """Reads the in-process RobotManager, store and routers. Never touches the bus."""

    def __init__(self, manager: Any, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.manager = manager
        self._sleep = sleep

    @property
    def settings(self) -> Any:
        return self.manager.settings

    def status(self) -> dict[str, Any]:
        return dict(self.manager.status())

    def fault(self) -> str | None:
        fault = getattr(self.manager, "fault", None)
        return str(fault) if fault else None

    def control_state(self) -> dict[str, Any] | None:
        reader = getattr(self.manager, "control_state", None)
        return dict(reader()) if callable(reader) else None

    def telemetry(self) -> dict[str, Any]:
        return dict(self.manager.telemetry())

    def latest_event_id(self) -> int | None:
        from ..services.store import get_store

        rows = get_store().list_events(limit=1)
        return int(rows[0]["id"]) if rows else 0

    def events_after(self, after_id: int | None) -> list[dict[str, Any]]:
        from ..services.store import get_store

        return list(reversed(get_store().list_events(limit=500, after_id=after_id)))

    def diagnostics(self) -> dict[str, Any]:
        from ..services.routers.operations import diagnostics

        return diagnostics().model_dump(mode="json")

    def preflight(self) -> PreflightReport:
        return pf.run_api_preflight(self.manager, sleep=self._sleep)

    def rest_pose(self) -> RestPoseResolution:
        limits = getattr(getattr(self.manager, "guard", None), "limits", None)
        return resolve_for_settings(self.settings, limits=limits)

    def health(self) -> dict[str, Any] | None:
        reader = getattr(self.manager, "health_snapshot", None)
        if not callable(reader):
            return None
        snapshot = reader()
        if snapshot is None:
            return None
        dump = getattr(snapshot, "model_dump", None)
        return dump(mode="json") if callable(dump) else dict(snapshot)

    def frame(self) -> tuple[str, Any] | None:
        for camera in self.status().get("live_cameras") or []:
            image = self.manager.get_frame(camera)
            if image is not None:
                return str(camera), image
        return None


class OfflineHilClient:
    """For reading saved runs (list, status, report) with no robot: every read refuses."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    @property
    def settings(self) -> Any:
        return self._settings

    def _refuse(self) -> Any:
        raise HilConflict("this needs the robot: start the Suite server (`so101 serve`)")

    def status(self) -> dict[str, Any]:
        return self._refuse()

    def fault(self) -> str | None:
        return self._refuse()

    def control_state(self) -> dict[str, Any] | None:
        return self._refuse()

    def telemetry(self) -> dict[str, Any]:
        return self._refuse()

    def latest_event_id(self) -> int | None:
        return self._refuse()

    def events_after(self, after_id: int | None) -> list[dict[str, Any]]:
        return self._refuse()

    def diagnostics(self) -> dict[str, Any]:
        return self._refuse()

    def preflight(self) -> PreflightReport:
        return self._refuse()

    def rest_pose(self) -> RestPoseResolution:
        return self._refuse()

    def health(self) -> dict[str, Any] | None:
        return self._refuse()

    def frame(self) -> tuple[str, Any] | None:
        return self._refuse()


# --- samples and analyzers ---------------------------------------------------------


@dataclass
class Verdict:
    status: CheckStatus
    message: str
    measured: dict[str, Any] = field(default_factory=dict)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def sample_record(
    t: float, wall: datetime, status: Mapping[str, Any], telemetry: Mapping[str, Any]
) -> dict[str, Any]:
    """One 10 Hz sample: time, status summary, wave-2 stop fields, joint positions."""
    observation = telemetry.get("observation")
    positions: dict[str, float] = {}
    if isinstance(observation, Mapping):
        for joint in JOINT_NAMES:
            value = _finite(observation.get(f"{joint}.pos"))
            if value is not None:
                positions[joint] = value
    record: dict[str, Any] = {
        "t": round(t, 3),
        "wall": wall.isoformat(timespec="milliseconds"),
        "mode": status.get("mode"),
        "estopped": status.get("estopped"),
        "seq": telemetry.get("sample_seq"),
        "sample_age_ms": telemetry.get("sample_age_ms"),
        "positions": positions,
    }
    for name in _OPTIONAL_STATUS_FIELDS:
        if name in status:
            record[name] = status[name]
    return record


def _distinct_seq(samples: Sequence[Mapping[str, Any]]) -> int:
    return len({sample.get("seq") for sample in samples if sample.get("seq") is not None})


def _fresh(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The first sample of each telemetry sequence number (repeats carry no news)."""
    kept: list[Mapping[str, Any]] = []
    last: Any = object()
    for sample in samples:
        seq = sample.get("seq")
        if seq is None or seq != last:
            kept.append(sample)
        last = seq
    return kept


def _arm(sample: Mapping[str, Any]) -> dict[str, float]:
    positions = sample.get("positions") or {}
    return {joint: value for joint, value in positions.items() if joint in ARM_JOINTS}


def _ranges(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    seen: dict[str, list[float]] = {}
    for sample in samples:
        for joint, value in (sample.get("positions") or {}).items():
            seen.setdefault(joint, []).append(value)
    return {joint: round(max(values) - min(values), 3) for joint, values in seen.items()}


def _unique(values: Iterable[Any]) -> list[Any]:
    found: list[Any] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _first_index(samples: Sequence[Mapping[str, Any]], state: str, start: int = 0) -> int | None:
    for index in range(start, len(samples)):
        if samples[index].get("stop_state") == state:
            return index
    return None


def analyze_status(
    status: Mapping[str, Any], fault: str | None, control: Mapping[str, Any] | None
) -> Verdict:
    configured = [str(camera) for camera in status.get("cameras") or []]
    live = [str(camera) for camera in status.get("live_cameras") or []]
    missing = [camera for camera in configured if camera not in live]
    measured: dict[str, Any] = {
        "mode": status.get("mode"),
        "estopped": bool(status.get("estopped")),
        "fault": fault,
        "follower_connected": status.get("follower_connected"),
        "leader_connected": status.get("leader_connected"),
        "cameras": configured,
        "live_cameras": live,
        "missing_cameras": missing,
        "camera_frame_age_ms": status.get("camera_frame_age_ms"),
    }
    if control is not None:
        measured["safety_state"] = control.get("safety_state")
        measured["armed"] = control.get("armed")
    for name in STOP_STATUS_FIELDS:
        if name in status:
            measured[name] = status[name]
    message = (
        f"mode={status.get('mode')}, estopped={measured['estopped']}, fault={fault or 'none'}; "
        f"cameras live {len(live)}/{len(configured)}"
        + (f" (missing: {', '.join(missing)})" if missing else "")
    )
    if "torque_state" in status:
        message += f"; torque_state={status['torque_state']}"
    return Verdict("warn" if measured["estopped"] or fault else "pass", message, measured)


def analyze_estop(status: Mapping[str, Any]) -> Verdict:
    measured = {
        "estopped": bool(status.get("estopped")),
        "torque_state": status.get("torque_state"),
        "torque_state_reported": "torque_state" in status,
        "mode": status.get("mode"),
    }
    if not measured["estopped"]:
        return Verdict(
            "fail",
            "the software E-stop is not latched: press E-stop, then probe while it is latched",
            measured,
        )
    if "torque_state" not in status:
        return Verdict(
            "warn",
            "the E-stop is latched, but this incompatible server reports no torque_state; "
            "torque-off is not machine-verified",
            measured,
        )
    torque = status.get("torque_state")
    if torque == "off":
        return Verdict("pass", 'the E-stop is latched and torque_state is "off"', measured)
    if torque == "unconfirmed":
        return Verdict(
            "fail", "torque-off is unconfirmed: cut follower power at the switch", measured
        )
    return Verdict("fail", f"the E-stop is latched but torque_state is {torque!r}", measured)


def analyze_telemetry(samples: Sequence[Mapping[str, Any]]) -> Verdict:
    fresh = _distinct_seq(samples)
    ranges = _ranges(_fresh(samples))
    duration = samples[-1]["t"] - samples[0]["t"] if samples else 0.0
    moved = {joint: span for joint, span in ranges.items() if span >= MOVED_DEG}
    measured: dict[str, Any] = {
        "samples": len(samples),
        "fresh_samples": fresh,
        "window_s": round(duration, 2),
        "range_deg": ranges,
        "moved_joints": sorted(moved),
    }
    if fresh < 2 or not ranges:
        return Verdict(
            "skip",
            "telemetry did not update during the window (sample_seq did not advance): start "
            "teleop or a jog session, then probe again",
            measured,
        )
    if not moved:
        return Verdict(
            "warn",
            f"{fresh} fresh samples over {duration:.1f} s, but no joint moved {MOVED_DEG:g}° "
            "or more",
            measured,
        )
    summary = ", ".join(f"{joint} {span:.1f}°" for joint, span in moved.items())
    return Verdict(
        "pass", f"{fresh} fresh samples over {duration:.1f} s; moved: {summary}", measured
    )


def analyze_events(events: Sequence[Mapping[str, Any]], prefixes: Sequence[str]) -> Verdict:
    by_type = Counter(str(event.get("type", "")) for event in events)
    matched = [
        event
        for event in events
        if any(str(event.get("type", "")).startswith(prefix) for prefix in prefixes)
    ]
    measured: dict[str, Any] = {
        "events_since_run_start": len(events),
        "by_type": dict(sorted(by_type.items())),
        "looked_for": list(prefixes),
        "matched": [
            {key: event.get(key) for key in ("id", "created_at", "type", "level", "message")}
            for event in matched[-50:]
        ],
    }
    if matched:
        kinds = ", ".join(sorted({str(event.get("type")) for event in matched}))
        return Verdict(
            "pass", f"{len(matched)} matching audit events since the run started: {kinds}", measured
        )
    if events:
        return Verdict(
            "warn",
            f"none of the {len(events)} audit events since the run started match "
            f"{', '.join(prefixes)}: do the test, then probe again",
            measured,
        )
    return Verdict(
        "warn", "no audit events since the run started: do the test, then probe again", measured
    )


def analyze_diagnostics(payload: Mapping[str, Any]) -> Verdict:
    checks = [check for check in payload.get("checks") or [] if isinstance(check, Mapping)]
    failed = [check for check in checks if check.get("status") == "fail"]
    warned = [check for check in checks if check.get("status") == "warn"]
    measured = {
        "overall": payload.get("overall"),
        "checks": len(checks),
        "failed": [check.get("id") for check in failed],
        "warned": [check.get("id") for check in warned],
        "failed_checks": len(failed),
    }
    if failed:
        detail = "; ".join(f"{check.get('label')}: {check.get('detail')}" for check in failed)
        return Verdict("fail", f"{len(failed)} failed: {detail}", measured)
    if warned:
        names = ", ".join(str(check.get("label")) for check in warned)
        return Verdict("warn", f"no failed check; warnings: {names}", measured)
    return Verdict("pass", f"all {len(checks)} diagnostics checks pass", measured)


def _cli_report_usable(
    cli: PreflightReport | None, backend: str, now: datetime
) -> tuple[PreflightReport | None, str | None]:
    if cli is None:
        return None, "no saved `so101 preflight` report"
    if cli.backend != backend:
        return None, f"the saved CLI report is for the {cli.backend} backend"
    created = _parse_time(cli.created_at)
    if created is None or now - created > CLI_REPORT_MAX_AGE:
        return None, "the saved CLI report is older than 2 h"
    return cli, None


def grade_preflight(
    api: PreflightReport,
    cli: PreflightReport | None,
    check_ids: Sequence[str],
    *,
    cli_note: str | None = None,
) -> Verdict:
    """Grade the selected checks; bus checks the API deferred come from the CLI report."""
    chosen: list[tuple[CheckResult, str]] = []
    used_cli = False
    for check_id in check_ids:
        from_api = [check for check in api.checks if check.id == check_id]
        deferred = bool(from_api) and all(
            check.status == "skip" and check.message == pf.WAVE2_BUS_MESSAGE for check in from_api
        )
        from_cli = [check for check in cli.checks if check.id == check_id] if cli else []
        if deferred and from_cli:
            chosen.extend((check, "cli") for check in from_cli)
            used_cli = True
        else:
            chosen.extend((check, "api") for check in from_api)
    results = [check for check, _ in chosen]
    if any(check.status == "fail" for check in results):
        status: CheckStatus = "fail"
    elif any(check.status == "skip" and check.blocking for check in results):
        status = "skip"
    elif any(check.status == "warn" for check in results):
        status = "warn"
    else:
        status = "pass"
    measured: dict[str, Any] = {
        "checks": [
            {
                "key": check.key,
                "status": check.status,
                "message": check.message,
                "remediation": check.remediation,
                "rehearsal": check.rehearsal,
                "from": origin,
            }
            for check, origin in chosen
        ],
        "api_report_created_at": api.created_at,
        "api_bus_access": api.bus_access,
        "cli_report_created_at": cli.created_at if used_cli and cli else None,
        "cli_report_note": None if used_cli else cli_note,
    }
    summary = ", ".join(f"{check.key} {check.status}" for check in results) or "no matching check"
    if used_cli and cli:
        summary += f" (bus checks from the `so101 preflight` report of {cli.created_at})"
    elif any(origin == "api" and check.status == "skip" for check, origin in chosen):
        if cli_note:
            summary += f"; {cli_note}"
    return Verdict(status, summary, measured)


def analyze_halt_hold(
    samples: Sequence[Mapping[str, Any]],
    *,
    min_hold_s: float = HALT_MIN_HOLD_S,
    max_drift_deg: float = HALT_MAX_DRIFT_DEG,
) -> Verdict:
    if not samples:
        return Verdict("skip", "no samples were taken")
    measured: dict[str, Any] = {
        "samples": len(samples),
        "stop_states_seen": _unique(sample.get("stop_state") for sample in samples),
        "max_drift_limit_deg": max_drift_deg,
        "min_hold_s": min_hold_s,
    }
    first = samples[0]
    if first.get("stop_state") != "halted":
        return Verdict(
            "fail",
            f"Halt is not active (stop_state={first.get('stop_state')!r}): press Halt, then "
            "probe at once",
            measured,
        )
    span: list[Mapping[str, Any]] = []
    for sample in samples:
        if sample.get("stop_state") != "halted":
            break
        span.append(sample)
    hold = span[-1]["t"] - span[0]["t"]
    measured["hold_duration_s"] = round(hold, 2)
    if len(span) < len(samples):
        measured["ended_as"] = samples[len(span)].get("stop_state")
    measured["fresh_samples"] = _distinct_seq(span)
    reference = _arm(span[0])
    drift: float | None = None
    if measured["fresh_samples"] >= 2 and reference:
        by_joint = {
            joint: round(max(abs(_arm(sample).get(joint, ref) - ref) for sample in span), 3)
            for joint, ref in reference.items()
        }
        worst = max(by_joint, key=lambda joint: by_joint[joint])
        drift = by_joint[worst]
        measured.update(max_drift_deg=drift, worst_joint=worst, drift_by_joint=by_joint)
        gripper = [sample.get("positions", {}).get("gripper") for sample in span]
        grips = [value for value in gripper if value is not None]
        if grips:
            measured["gripper_range"] = round(max(grips) - min(grips), 3)
    problems = []
    if hold < min_hold_s:
        problems.append(
            f"the halt lasted {hold:.1f} s (needs {min_hold_s:g} s)"
            + (f" and ended as {measured['ended_as']!r}" if "ended_as" in measured else "")
        )
    if drift is not None and drift > max_drift_deg:
        problems.append(
            f"{measured['worst_joint']} drifted {drift:.2f}° (limit {max_drift_deg:g}°)"
        )
    if problems:
        return Verdict("fail", "; ".join(problems), measured)
    if drift is None:
        return Verdict(
            "warn",
            f"the halt held for {hold:.1f} s, but telemetry did not update, so drift was not "
            "measured; restore live telemetry and probe again (observation is not stop evidence)",
            measured,
        )
    return Verdict(
        "pass",
        f"held {hold:.1f} s; largest drift {drift:.2f}° ({measured['worst_joint']})",
        measured,
    )


def collapse_timeout(
    start: Mapping[str, float], rest_pose: Mapping[str, float], speed_deg_s: float
) -> float:
    """Design default: distance / speed + 2 s, at most 12 s."""
    distances = [
        abs(value - rest_pose[joint]) for joint, value in start.items() if joint in rest_pose
    ]
    distance = max(distances, default=0.0)
    return min(distance / speed_deg_s + COLLAPSE_TIMEOUT_MARGIN_S, COLLAPSE_TIMEOUT_MAX_S)


def _windowed_speeds(samples: Sequence[Mapping[str, Any]], window_s: float) -> dict[str, float]:
    fresh = _fresh(samples)
    speeds: dict[str, float] = {}
    for i, earlier in enumerate(fresh):
        later = next(
            (sample for sample in fresh[i + 1 :] if sample["t"] - earlier["t"] >= window_s), None
        )
        if later is None:
            break
        dt = later["t"] - earlier["t"]
        before, after = _arm(earlier), _arm(later)
        for joint in before.keys() & after.keys():
            speed = abs(after[joint] - before[joint]) / dt
            speeds[joint] = round(max(speeds.get(joint, 0.0), speed), 3)
    return speeds


def analyze_collapse(
    samples: Sequence[Mapping[str, Any]],
    *,
    rest_pose: Mapping[str, float],
    speed_cap_deg_s: float = DEFAULT_COLLAPSE_SPEED_DEG_S,
    timeout_s: float | None = None,
    settle_deg: float = COLLAPSE_SETTLE_DEG,
    timing_tolerance_s: float = 2 * SAMPLE_INTERVAL_S,
    events: Sequence[Mapping[str, Any]] = (),
) -> Verdict:
    measured: dict[str, Any] = {
        "samples": len(samples),
        "stop_states_seen": _unique(sample.get("stop_state") for sample in samples),
        "speed_cap_deg_s": speed_cap_deg_s,
        "speed_limit_used_deg_s": round(speed_cap_deg_s * SPEED_TOLERANCE, 3),
        "speed_window_s": SPEED_WINDOW_S,
        "settle_deg": settle_deg,
    }
    start = _first_index(samples, "collapsing")
    if start is None:
        return Verdict(
            "fail",
            "no Collapse was seen in the window: start the probe, then press Collapse",
            measured,
        )
    end = next(
        (i for i in range(start + 1, len(samples)) if samples[i].get("stop_state") != "collapsing"),
        None,
    )
    measured["reason"] = _mapping(samples[start].get("collapse")).get("reason")
    begin_pose = _arm(samples[start])
    if timeout_s is None:
        timeout_s = collapse_timeout(begin_pose, rest_pose, speed_cap_deg_s)
        measured["timeout_basis"] = "distance / speed + 2 s, at most 12 s"
    else:
        measured["timeout_basis"] = "SO101_COLLAPSE_TIMEOUT_SECONDS"
    measured["timeout_s"] = round(timeout_s, 2)
    if end is None:
        running = samples[-1]["t"] - samples[start]["t"]
        measured["duration_s"] = round(running, 2)
        return Verdict(
            "fail",
            f"still collapsing {running:.1f} s after it started (timeout {timeout_s:.1f} s)",
            measured,
        )
    duration = samples[end]["t"] - samples[start]["t"]
    within_timeout = duration <= timeout_s + timing_tolerance_s
    speeds = _windowed_speeds(samples[start : end + 1], SPEED_WINDOW_S)
    max_speed = max(speeds.values()) if speeds else None
    speed_limit = speed_cap_deg_s * SPEED_TOLERANCE
    fresh = _distinct_seq(samples[start:])
    final = _arm(samples[-1]) if fresh >= 2 else {}  # stale positions prove nothing
    errors = {
        joint: round(abs(value - rest_pose[joint]), 3)
        for joint, value in final.items()
        if joint in rest_pose
    }
    final_error = max(errors.values()) if errors else None
    torque_end = samples[-1].get("torque_state")
    fallback = any(
        str(event.get("type", "")).startswith("safety.collapse.fallback") for event in events
    )
    measured.update(
        duration_s=round(duration, 2),
        fresh_samples=fresh,
        within_timeout=within_timeout,
        timing_tolerance_s=timing_tolerance_s,
        max_speed_deg_s=max_speed,
        speed_by_joint=speeds,
        speed_within_cap=None if max_speed is None else max_speed <= speed_limit,
        final_error_deg=final_error,
        final_error_by_joint=errors,
        settled=None if final_error is None else final_error <= settle_deg,
        torque_state_end=torque_end,
        torque_off_at_end=torque_end == "off",
        fallback=fallback,
    )
    problems = []
    if not within_timeout:
        problems.append(f"took {duration:.1f} s (timeout {timeout_s:.1f} s)")
    if measured["speed_within_cap"] is False:
        problems.append(
            f"a joint moved at {max_speed:.1f}°/s (cap {speed_cap_deg_s:g}°/s, "
            f"+{(SPEED_TOLERANCE - 1) * 100:.0f}% tolerance)"
        )
    if measured["settled"] is False:
        problems.append(f"ended {final_error:.1f}° from the rest pose (limit {settle_deg:g}°)")
    if torque_end != "off":
        problems.append(f"torque_state at the end is {torque_end!r}, not 'off'")
    if fallback:
        problems.append("the fallback fired (safety.collapse.fallback)")
    if problems:
        return Verdict("fail", "; ".join(problems), measured)
    if max_speed is None or final_error is None:
        return Verdict(
            "warn",
            f"finished in {duration:.1f} s with torque off, but telemetry did not update, so "
            "speed and final error were not measured",
            measured,
        )
    return Verdict(
        "pass",
        f"finished in {duration:.1f} s (timeout {timeout_s:.1f} s); fastest joint "
        f"{max_speed:.1f}°/s; final error {final_error:.1f}°; torque off",
        measured,
    )


def analyze_hold_timer(
    samples: Sequence[Mapping[str, Any]],
    *,
    hold_s: float = DEFAULT_HALT_HOLD_S,
    early_s: float = HOLD_TIMER_EARLY_S,
    late_s: float = HOLD_TIMER_LATE_S,
) -> Verdict:
    measured: dict[str, Any] = {
        "samples": len(samples),
        "stop_states_seen": _unique(sample.get("stop_state") for sample in samples),
        "hold_seconds_setting": hold_s,
        "tolerance_s": [-early_s, late_s],
    }
    if not samples or samples[0].get("stop_state") != "halted":
        state = samples[0].get("stop_state") if samples else None
        return Verdict(
            "fail",
            f"Halt is not active (stop_state={state!r}): press Halt, then probe at once",
            measured,
        )
    first = samples[0]
    halt = _mapping(first.get("halt"))
    expires = _parse_time(halt.get("hold_expires_at"))
    server_now = _parse_time(halt.get("server_time"))
    since = _parse_time(halt.get("since"))
    expected_after = (expires - server_now).total_seconds() if expires and server_now else None
    if expires and since:
        measured["configured_hold_s"] = round((expires - since).total_seconds(), 2)
    measured["expected_after_s"] = None if expected_after is None else round(expected_after, 2)
    collapse_at = _first_index(samples, "collapsing")
    cleared_at = _first_index(samples, "none")
    if cleared_at is not None and (collapse_at is None or cleared_at < collapse_at):
        cleared = samples[cleared_at]["t"] - first["t"]
        return Verdict(
            "fail",
            f"the halt was cleared {cleared:.1f} s into the probe, before the hold expired",
            measured,
        )
    if collapse_at is None:
        waited = samples[-1]["t"] - first["t"]
        due = f" (it was due after {expected_after:.1f} s)" if expected_after is not None else ""
        return Verdict(
            "fail", f"no Collapse within {waited:.1f} s of starting the probe{due}", measured
        )
    seen_after = samples[collapse_at]["t"] - first["t"]
    measured.update(collapse_started=True, collapse_seen_after_probe_start_s=round(seen_after, 2))
    if since and server_now:
        measured["collapse_started_after_s"] = round(
            (server_now - since).total_seconds() + seen_after, 2
        )
    if expected_after is None:
        return Verdict(
            "warn",
            f"Collapse started {seen_after:.1f} s into the probe; halt.hold_expires_at or "
            "halt.server_time is missing, so the timing is not verified",
            measured,
        )
    lateness = seen_after - expected_after
    measured["lateness_s"] = round(lateness, 2)
    if -early_s <= lateness <= late_s:
        return Verdict(
            "pass",
            f"Collapse started {lateness:+.1f} s from the hold expiry (tolerance "
            f"-{early_s:g}/+{late_s:g} s)",
            measured,
        )
    return Verdict(
        "fail",
        f"Collapse started {lateness:+.1f} s from the hold expiry (tolerance "
        f"-{early_s:g}/+{late_s:g} s)",
        measured,
    )


def analyze_supervision_loss(
    samples: Sequence[Mapping[str, Any]],
    *,
    grace_s: float = DEFAULT_SUPERVISION_GRACE_S,
    tolerance_s: float = SUPERVISION_TOLERANCE_S,
) -> Verdict:
    measured: dict[str, Any] = {
        "samples": len(samples),
        "stop_states_seen": _unique(sample.get("stop_state") for sample in samples),
        "grace_s": grace_s,
        "tolerance_s": tolerance_s,
    }
    index = _first_index(samples, "collapsing")
    if index is None:
        waited = samples[-1]["t"] - samples[0]["t"] if samples else 0.0
        return Verdict("fail", f"no Collapse within {waited:.1f} s of starting the probe", measured)
    sample = samples[index]
    collapse = _mapping(sample.get("collapse"))
    reason = collapse.get("reason")
    seen_after = sample["t"] - samples[0]["t"]
    measured.update(
        collapse_started=True,
        collapse_seen_after_probe_start_s=round(seen_after, 2),
        reason=reason,
    )
    reason_note = ""
    if reason is not None and "supervision" not in str(reason).lower():
        reason_note = f"; check the Collapse reason ({reason!r} does not mention supervision)"
    supervision = _mapping(sample.get("supervision"))
    lost_at = _parse_time(supervision.get("lost_at"))
    since = _parse_time(collapse.get("since"))
    if lost_at and since:
        delay = (since - lost_at).total_seconds()
        within = delay <= grace_s + tolerance_s
        measured.update(seconds_after_loss=round(delay, 2), within_grace=within)
        if not within:
            return Verdict(
                "fail",
                f"Collapse started {delay:.1f} s after supervision was lost (grace {grace_s:g} s "
                f"+ {tolerance_s:g} s tolerance)",
                measured,
            )
        return Verdict(
            "warn" if reason_note else "pass",
            f"Collapse started {delay:.1f} s after supervision was lost{reason_note}",
            measured,
        )
    return Verdict(
        "warn",
        f"Collapse started {seen_after:.1f} s after the probe started (reason: {reason!r}); the "
        "server does not report when supervision was lost. A stopwatch cannot replace machine "
        f"timing; use the current server and probe again (limit {grace_s:g} s + "
        f"{tolerance_s:g} s){reason_note}",
        measured,
    )


# --- attestation rules ---------------------------------------------------------------


def _type_problem(spec: EvidenceField, value: Any) -> str | None:
    if spec.kind == "bool":
        return None if isinstance(value, bool) else "must be true or false"
    if spec.kind == "number":
        return None if _finite(value) is not None else "must be a finite number"
    if spec.kind == "text":
        if not isinstance(value, str) or not value.strip():
            return "must be non-empty text"
        return "must be at most 2000 characters" if len(value) > 2000 else None
    if not isinstance(value, str) or value not in spec.choices:
        return "must be one of " + ", ".join(spec.choices)
    return None


def validate_attestation(
    step: HilStepDefinition, request: AttestRequest, latest_probe: ProbeResult | None
) -> list[str]:
    """Problems that block this attestation; empty means it is accepted."""
    problems: list[str] = []
    specs = {spec.name: spec for spec in step.evidence_fields}
    for name in sorted(set(request.evidence) - set(specs)):
        problems.append(f"{name}: not an evidence field of {step.id}")
    passing = request.outcome == "pass"
    for name, spec in specs.items():
        if name not in request.evidence:
            if passing and spec.required:
                problems.append(f"{name}: required for a pass ({spec.label})")
            continue
        value = request.evidence[name]
        problem = _type_problem(spec, value)
        if problem:
            problems.append(f"{name}: {problem}")
            continue
        if not passing:
            continue
        if spec.pass_equals is not None and value != spec.pass_equals:
            problems.append(f"{name}: a pass needs {spec.pass_equals!r}, got {value!r}")
        number = _finite(value) if spec.kind == "number" else None
        if number is not None and spec.pass_min is not None and number < spec.pass_min:
            problems.append(f"{name}: a pass needs at least {spec.pass_min:g}, got {number:g}")
        if number is not None and spec.pass_max is not None and number > spec.pass_max:
            problems.append(f"{name}: a pass needs at most {spec.pass_max:g}, got {number:g}")
    if request.outcome in ("fail", "skip") and not request.note.strip():
        problems.append(f"note: a {request.outcome} needs a note saying why")
    if passing:
        if latest_probe is not None and latest_probe.status == "fail":
            problems.append(
                f"probe: the latest probe failed ({latest_probe.message}); fix the cause and "
                "probe again, or attest fail"
            )
        elif step.group == "stop" and step.probe != "none":
            if latest_probe is None:
                problems.append(
                    "probe: a stop pass must run its probe; observation alone is not stop evidence"
                )
            elif (
                latest_probe.catalog_version != HIL_CATALOG_VERSION
                or latest_probe.catalog_digest != HIL_CATALOG_DIGEST
            ):
                problems.append(
                    "probe: the latest stop probe predates the current evidence contract; "
                    "probe again before attesting pass"
                )
            elif (
                latest_probe.required_status_fields
                != [item.split(".", 1)[1] for item in step.requires if item.startswith("status.")]
                or not latest_probe.required_status_fields_present
            ):
                problems.append(
                    "probe: the latest stop probe lacks the current required status fields; "
                    "probe again before attesting pass"
                )
            elif latest_probe.status == "skip":
                problems.append(
                    f"probe: the latest stop probe skipped ({latest_probe.message}); restore "
                    "the required status fields and probe again, or attest fail/skip"
                )
            elif (
                step.id == "stop-estop"
                and latest_probe.measured.get("torque_state_reported") is False
            ):
                problems.append(
                    "probe: stop-estop did not report torque_state; use the current server and "
                    "probe again before attesting pass"
                )
            elif latest_probe.status == "warn":
                problems.append(
                    f"probe: the latest stop probe warned ({latest_probe.message}); a stop "
                    "pass needs complete machine measurements, so probe again or attest fail"
                )
    return problems


def stop_evidence_gaps(run: HilRun) -> list[str]:
    """Stop steps whose recorded pass/evidence is not machine-verifiable.

    Live attestation rejects these states. This second check makes finalization
    fail closed for an open run written by an older build or edited before its
    manifest was sealed.
    """

    gaps: list[str] = []
    for step_id in run.step_ids:
        step = STEPS_BY_ID[step_id]
        if step.group != "stop" or step.probe == "none":
            continue
        record = run.steps[step_id]
        if not record.probes:
            gaps.append(f"{step_id} has no current probe")
            continue
        for probe in record.probes:
            if probe.status == "skip":
                gaps.append(f"{step_id} probe {probe.sequence} skipped")
            if probe.status == "warn":
                gaps.append(f"{step_id} probe {probe.sequence} warned")
            if step_id == "stop-estop" and probe.measured.get("torque_state_reported") is False:
                gaps.append(f"{step_id} probe {probe.sequence} did not report torque_state")
        latest = record.probes[-1]
        required = [item.split(".", 1)[1] for item in step.requires if item.startswith("status.")]
        if (
            latest.catalog_version != HIL_CATALOG_VERSION
            or latest.catalog_digest != HIL_CATALOG_DIGEST
        ):
            gaps.append(f"{step_id} latest probe predates the current evidence contract")
        elif latest.required_status_fields != required or not latest.required_status_fields_present:
            gaps.append(f"{step_id} latest probe lacks current required status fields")
        if record.status == "pass" and latest.status == "fail":
            gaps.append(f"{step_id} passed over a failed probe")
    return gaps


def catalog_upgrade_problems(run: HilRun) -> list[str]:
    """Why an open run cannot be finalized under the current catalog.

    Old runs retain their id, attestations and probe history. Each existing
    attestation must be deliberately replaced, and every attested measured
    stop must be reprobed, before the run can adopt this catalog digest.
    """

    problems: list[str] = []
    upgrading = (
        run.catalog_version != HIL_CATALOG_VERSION or run.catalog_digest != HIL_CATALOG_DIGEST
    )
    for step_id in run.step_ids:
        step = STEPS_BY_ID[step_id]
        record = run.steps[step_id]
        attestation = record.attestation
        if attestation is not None and (
            attestation.catalog_version != HIL_CATALOG_VERSION
            or attestation.catalog_digest != HIL_CATALOG_DIGEST
        ):
            problems.append(f"re-attest {step_id} under the current evidence contract")
        if step.group != "stop" or step.probe == "none" or attestation is None:
            continue
        latest = record.probes[-1] if record.probes else None
        if latest is None or (
            latest.catalog_version != HIL_CATALOG_VERSION
            or latest.catalog_digest != HIL_CATALOG_DIGEST
            or (
                upgrading
                and (
                    latest.required_status_fields
                    != [
                        item.split(".", 1)[1]
                        for item in step.requires
                        if item.startswith("status.")
                    ]
                    or not latest.required_status_fields_present
                )
            )
        ):
            problems.append(
                f"re-probe {step_id} with the current required status fields before finalize"
            )
    return problems


def final_status(run: HilRun) -> RunStatus:
    statuses = [run.steps[step_id].status for step_id in run.step_ids]
    if any(status == "fail" for status in statuses):
        return "failed"
    if statuses and all(status == "pass" for status in statuses):
        return "incomplete" if stop_evidence_gaps(run) else "passed"
    return "incomplete"


def summarize(run: HilRun) -> dict[str, int]:
    counts = {status: 0 for status in ("pass", "fail", "skip", "pending")}
    probes = 0
    for step_id in run.step_ids:
        record = run.steps[step_id]
        counts[record.status] += 1
        probes += len(record.probes)
    return {**counts, "steps": len(run.step_ids), "probes": probes}


def evidence_verdict(run: HilRun) -> tuple[bool, list[str]]:
    """``hil_evidence`` and the notes explaining it (plus caveats when true)."""
    notes: list[str] = []
    real = run.backend == "real" and not run.rehearsal
    if not real:
        notes.append(
            f"REHEARSAL: the {run.backend} backend was used; this record is not hardware evidence"
        )
    off_backend = [
        f"{step_id} probe {probe.sequence} ({probe.backend})"
        for step_id in run.step_ids
        for probe in run.steps[step_id].probes
        if probe.backend != "real" or probe.rehearsal
    ]
    if real and off_backend:
        notes.append("probes that did not run on the real backend: " + ", ".join(off_backend))
    attested = [
        step_id for step_id in run.step_ids if run.steps[step_id].status in ("pass", "fail")
    ]
    if not attested:
        notes.append("no step was attested pass or fail")
    stop_gaps = stop_evidence_gaps(run)
    if stop_gaps:
        notes.append("stop evidence is incomplete: " + ", ".join(stop_gaps))
    evidence = real and not off_backend and bool(attested) and not stop_gaps

    unprobed = [
        step_id
        for step_id in run.step_ids
        if STEPS_BY_ID[step_id].group != "stop"
        and STEPS_BY_ID[step_id].probe != "none"
        and run.steps[step_id].status == "pass"
        and not run.steps[step_id].probes
    ]
    if unprobed:
        notes.append("attested pass without running the probe: " + ", ".join(unprobed))
    unmeasured = [
        step_id
        for step_id in run.step_ids
        if STEPS_BY_ID[step_id].group != "stop"
        and run.steps[step_id].status == "pass"
        and run.steps[step_id].probes
        and run.steps[step_id].probes[-1].status == "skip"
    ]
    if unmeasured:
        notes.append(
            "attested pass on observation; the probe could not measure: " + ", ".join(unmeasured)
        )
    pending = [step_id for step_id in run.step_ids if run.steps[step_id].status == "pending"]
    if pending:
        notes.append("not attested: " + ", ".join(pending))
    return evidence, notes


# --- Markdown report -------------------------------------------------------------------


def _cell(value: Any) -> str:
    text = json.dumps(value) if isinstance(value, bool) or value is None else str(value)
    return " ".join(text.replace("|", "\\|").split())


def render_markdown(run: HilRun, run_dir: Path) -> str:
    lines = [f"# SO-101 HIL run {run.run_id}", ""]
    if run.hil_evidence:
        lines.append("**Hardware evidence:** real backend, every probe on the real backend.")
    elif run.rehearsal:
        lines.append(
            f"**REHEARSAL ({run.backend} backend). This record is not hardware evidence.**"
        )
    else:
        lines.append("**Not marked as hardware evidence.** See the notes below.")
    lines += [
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Status | {_cell(run.status)} |",
        f"| hil_evidence | {_cell(run.hil_evidence)} |",
        f"| Backend | {_cell(run.backend)} (rehearsal: {_cell(run.rehearsal)}) |",
        f"| Operator | {_cell(run.operator)} |",
        f"| Created | {_cell(run.created_at)} |",
        f"| Finalized | {_cell(run.finalized_at)} |",
        f"| so101 version | {_cell(run.so101_version)} |",
        f"| HIL catalog | v{run.catalog_version} · {_cell(run.catalog_digest)} |",
        f"| Host | {_cell(run.host)} |",
    ]
    lines += [f"| {_cell(key)} | {_cell(value)} |" for key, value in sorted(run.devices.items())]
    if run.note:
        lines += [f"| Note | {_cell(run.note)} |"]
    summary = run.summary or summarize(run)
    lines += [
        "",
        "## Summary",
        "",
        f"pass {summary.get('pass', 0)} · fail {summary.get('fail', 0)} · skip "
        f"{summary.get('skip', 0)} · pending {summary.get('pending', 0)} · probes "
        f"{summary.get('probes', 0)}",
        "",
    ]
    if run.evidence_notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in run.evidence_notes] + [""]
    lines += ["## Steps", ""]
    for step_id in run.step_ids:
        step = STEPS_BY_ID[step_id]
        record = run.steps[step_id]
        lines += [f"### {step.order}. {step.id}: {step.title} ({record.status.upper()})", ""]
        lines += [f"Source: {step.source}", "", f"> {step.quote}", ""]
        lines += ["Expected:", ""] + [f"- {item}" for item in step.expected_results] + [""]
        if step.known_limit:
            lines += ["This step records a known limit; no software stop covers it.", ""]
        for probe in record.probes:
            label = "REHEARSAL " if probe.rehearsal else ""
            lines.append(
                f"- {label}probe {probe.sequence} ({probe.probe}, {probe.backend}) "
                f"{probe.status.upper()}: {_cell(probe.message)}"
            )
            lines += [f"  - artifact: `{artifact}`" for artifact in probe.artifacts]
        if record.probes:
            lines.append("")
        if record.attestation_history:
            lines += ["Prior attestations retained:", ""]
            for prior in record.attestation_history:
                contract = prior.catalog_digest[:12] or "pre-versioning"
                lines.append(
                    f"- **{prior.outcome}** by {_cell(prior.operator)} at "
                    f"{prior.attested_at} (catalog {contract})"
                )
            lines.append("")
        attestation = record.attestation
        if attestation is None:
            lines += ["Not attested.", ""]
            continue
        lines.append(
            f"Attested **{attestation.outcome}** by {_cell(attestation.operator)} at "
            f"{attestation.attested_at}."
        )
        if attestation.note:
            lines.append(f"Note: {_cell(attestation.note)}")
        if attestation.evidence:
            units = {spec.name: spec.unit for spec in step.evidence_fields}
            lines += ["", "| Evidence | Value |", "| --- | --- |"]
            for name, value in attestation.evidence.items():
                unit = f" {units[name]}" if units.get(name) else ""
                lines.append(f"| {_cell(name)} | {_cell(value)}{unit} |")
        lines.append("")
    lines += [
        "## Verify",
        "",
        "`MANIFEST.sha256` lists every file in the run directory, in the `sha256sum` line "
        "format. Check it with the Suite on any OS:",
        "",
        f"    so101 hil report {run.run_id}",
        "",
        "or without the Suite, from the run directory, with coreutils (Linux, Git Bash on "
        "Windows) or Perl's `shasum` (macOS):",
        "",
        "    sha256sum -c MANIFEST.sha256",
        "    shasum -a 256 -c MANIFEST.sha256",
        "",
        f"Run directory: `{run_dir}`",
        "",
    ]
    return "\n".join(lines)


# --- the service ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_PROBING: set[str] = set()


def hil_root(settings: Any) -> Path:
    return Path(settings.data_dir) / "hil"


def new_run_id(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


@dataclass
class _ProbeOutcome:
    verdict: Verdict
    samples: list[dict[str, Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HilService:
    """Start, probe, attest, finalize and report HIL runs under ``<data_dir>/hil``."""

    def __init__(
        self,
        client: HilClient,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        sample_interval_s: float = SAMPLE_INTERVAL_S,
    ) -> None:
        self.client = client
        self.settings = client.settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic
        self._sleep = sleep
        self._interval = sample_interval_s

    # -- paths and persistence --
    @property
    def root(self) -> Path:
        return hil_root(self.settings)

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.match(run_id):
            raise HilNotFound(f"no HIL run {run_id!r}")
        return self.root / run_id

    def _load(self, run_dir: Path) -> HilRun:
        try:
            text = (run_dir / "run.json").read_text(encoding="utf-8")
        except OSError as exc:
            raise HilNotFound(f"no HIL run {run_dir.name!r}") from exc
        return HilRun.model_validate_json(text)

    def _save_run(self, run_dir: Path, run: HilRun) -> None:
        atomic_write_json(run_dir / "run.json", run.model_dump(mode="json"))

    def _save_step(self, run_dir: Path, step: HilStepDefinition, record: HilStepRecord) -> None:
        atomic_write_json(
            run_dir / "steps" / f"{step.id}.json",
            {
                "kind": "so101.hil_step",
                "definition": step.model_dump(mode="json"),
                "record": record.model_dump(mode="json"),
            },
        )

    def _open_run(self, run_id: str) -> tuple[Path, HilRun]:
        run_dir = self._run_dir(run_id)
        run = self._load(run_dir)
        if run.status != "in_progress":
            raise HilConflict(f"run {run_id} is finalized ({run.status}) and cannot change")
        return run_dir, run

    @staticmethod
    def _step_of(run: HilRun, step_id: str) -> HilStepDefinition:
        if step_id not in run.steps or step_id not in STEPS_BY_ID:
            raise HilNotFound(f"step {step_id!r} is not part of run {run.run_id}")
        return STEPS_BY_ID[step_id]

    # -- queries --
    def steps(self) -> list[HilStepDefinition]:
        return step_catalog()

    def get_run(self, run_id: str) -> HilRun:
        return self._load(self._run_dir(run_id))

    def list_runs(self) -> list[HilRunSummary]:
        if not self.root.is_dir():
            return []
        summaries: list[HilRunSummary] = []
        for run_dir in sorted(self.root.iterdir(), reverse=True):
            if not RUN_ID_RE.match(run_dir.name):
                continue
            try:
                run = self._load(run_dir)
            except (HilNotFound, ValueError):
                continue
            summaries.append(
                HilRunSummary(
                    run_id=run.run_id,
                    created_at=run.created_at,
                    backend=run.backend,
                    rehearsal=run.rehearsal,
                    status=run.status,
                    hil_evidence=run.hil_evidence,
                    operator=run.operator,
                    finalized_at=run.finalized_at,
                    steps_total=len(run.step_ids),
                    steps_done=sum(
                        1 for step_id in run.step_ids if run.steps[step_id].status != "pending"
                    ),
                )
            )
        # Run ids carry whole seconds and a random suffix, so two runs started in
        # the same second need the creation time to come out newest first.
        summaries.sort(key=lambda summary: (summary.created_at, summary.run_id), reverse=True)
        return summaries

    # -- lifecycle --
    def _devices(self) -> dict[str, str]:
        settings = self.settings
        devices = {
            "follower_port": str(getattr(settings, "follower_port", "")),
            "leader_port": str(getattr(settings, "leader_port", "")),
            "follower_id": str(getattr(settings, "follower_id", "")),
            "leader_id": str(getattr(settings, "leader_id", "")),
        }
        try:
            digest = follower_calibration_digest(settings)
        except Exception:  # an unreadable file must not block a run
            digest = None
        devices["follower_calibration_digest"] = digest or "none"
        return devices

    def start_run(self, request: StartHilRunRequest) -> HilRun:
        try:
            step_ids = select_steps(request.step_ids, request.groups)
        except ValueError as exc:
            raise HilInvalid(str(exc), [str(exc)]) from exc
        backend = current_backend(self.settings)
        try:
            cursor = self.client.latest_event_id()
        except Exception:
            cursor = None
        now = self._clock()
        with _LOCK:
            run_id = new_run_id(now)
            while (self.root / run_id).exists():
                run_id = new_run_id(now)
            run = HilRun(
                catalog_version=HIL_CATALOG_VERSION,
                catalog_digest=HIL_CATALOG_DIGEST,
                run_id=run_id,
                created_at=_iso(now),
                so101_version=__version__,
                backend=backend,
                rehearsal=is_rehearsal(backend),
                operator=request.operator,
                note=request.note.strip(),
                host=platform.node(),
                devices=self._devices(),
                step_ids=step_ids,
                steps={step_id: HilStepRecord(step_id=step_id) for step_id in step_ids},
                event_cursor=cursor,
            )
            run_dir = self.root / run_id
            run_dir.mkdir(parents=True)
            for step_id in step_ids:
                self._save_step(run_dir, STEPS_BY_ID[step_id], run.steps[step_id])
            self._save_run(run_dir, run)
        return run

    def attest(self, run_id: str, step_id: str, request: AttestRequest) -> HilStepRecord:
        with _LOCK:
            run_dir, run = self._open_run(run_id)
            step = self._step_of(run, step_id)
            record = run.steps[step_id]
            latest = record.probes[-1] if record.probes else None
            problems = validate_attestation(step, request, latest)
            if problems:
                raise HilInvalid(f"attestation for {step_id} rejected", problems)
            now = _iso(self._clock())
            if record.attestation is not None:
                record.attestation_history.append(record.attestation.model_copy(deep=True))
            record.attestation = Attestation(
                operator=request.operator,
                outcome=request.outcome,
                note=request.note.strip(),
                evidence=dict(request.evidence),
                catalog_version=HIL_CATALOG_VERSION,
                catalog_digest=HIL_CATALOG_DIGEST,
                attested_at=now,
            )
            record.status = request.outcome
            record.updated_at = now
            self._save_step(run_dir, step, record)
            self._save_run(run_dir, run)
            return record

    def finalize(self, run_id: str) -> HilRun:
        with _LOCK:
            run_dir, run = self._open_run(run_id)
            if str(run_dir) in _PROBING:
                raise HilConflict(f"a probe is still running in run {run_id}")
            contract_problems = catalog_upgrade_problems(run)
            if contract_problems:
                raise HilInvalid("run uses an older HIL evidence contract", contract_problems)
            if (
                run.catalog_version != HIL_CATALOG_VERSION
                or run.catalog_digest != HIL_CATALOG_DIGEST
            ):
                run.catalog_version = HIL_CATALOG_VERSION
                run.catalog_digest = HIL_CATALOG_DIGEST
                for step_id in run.step_ids:
                    self._save_step(run_dir, STEPS_BY_ID[step_id], run.steps[step_id])
            run.summary = summarize(run)
            run.status = final_status(run)
            run.hil_evidence, run.evidence_notes = evidence_verdict(run)
            run.finalized_at = _iso(self._clock())
            self._save_run(run_dir, run)
            atomic_write_json(
                run_dir / "report.json",
                {
                    "kind": "so101.hil_report",
                    "schema_version": 1,
                    "run": run.model_dump(mode="json"),
                    "steps": {
                        step_id: STEPS_BY_ID[step_id].model_dump(mode="json")
                        for step_id in run.step_ids
                    },
                },
            )
            atomic_write_text(run_dir / "report.md", render_markdown(run, run_dir))
            write_manifest(run_dir)
            return run

    def report(self, run_id: str) -> HilReport:
        run_dir = self._run_dir(run_id)
        run = self._load(run_dir)
        if run.status == "in_progress":
            raise HilConflict(f"run {run_id} is not finalized; finalize it first")
        try:
            markdown = (run_dir / "report.md").read_text(encoding="utf-8")
        except OSError:
            markdown = render_markdown(run, run_dir)
        return HilReport(
            run=run, markdown=markdown, manifest=verify_manifest(run_dir), run_dir=str(run_dir)
        )

    # -- probes --
    def probe(self, run_id: str, step_id: str, duration_s: float | None = None) -> ProbeResult:
        with _LOCK:
            run_dir, run = self._open_run(run_id)
            step = self._step_of(run, step_id)
            if step.probe == "none":
                raise HilInvalid(
                    f"step {step_id} has no automatic probe; attest it directly",
                    [f"{step_id}: no probe"],
                )
            key = str(run_dir)
            if key in _PROBING:
                raise HilConflict(f"another probe is already running in run {run_id}")
            _PROBING.add(key)
        try:
            backend = current_backend(self.settings)
            started = self._clock()
            required_status_fields = [
                item.split(".", 1)[1] for item in step.requires if item.startswith("status.")
            ]
            required_status_fields_present = False
            try:
                status = self.client.status()
                required_status_fields_present = all(
                    name in status for name in required_status_fields
                )
                outcome = self._execute(run, step, duration_s, status=status)
            except HilError:
                raise
            except Exception as exc:  # a broken probe is a result, not a crash
                outcome = _ProbeOutcome(
                    Verdict("fail", f"the probe crashed: {type(exc).__name__}: {exc}")
                )
            bundle = self._context(run)
            finished = self._clock()
            with _LOCK:
                run_dir, run = self._open_run(run_id)
                record = run.steps[step_id]
                result = ProbeResult(
                    step_id=step_id,
                    probe=step.probe,
                    sequence=len(record.probes) + 1,
                    status=outcome.verdict.status,
                    message=outcome.verdict.message,
                    catalog_version=HIL_CATALOG_VERSION,
                    catalog_digest=HIL_CATALOG_DIGEST,
                    required_status_fields=required_status_fields,
                    required_status_fields_present=required_status_fields_present,
                    measured=outcome.verdict.measured,
                    backend=backend,
                    rehearsal=is_rehearsal(backend),
                    started_at=_iso(started),
                    finished_at=_iso(finished),
                )
                result.artifacts = self._write_artifacts(run_dir, step, result, outcome, bundle)
                record.probes.append(result)
                record.updated_at = result.finished_at
                self._save_step(run_dir, step, record)
                self._save_run(run_dir, run)
                return result
        finally:
            with _LOCK:
                _PROBING.discard(key)

    def _context(self, run: HilRun) -> dict[str, Any]:
        """What every probe bundle carries besides its own measurement."""
        context: dict[str, Any] = {}
        readers: dict[str, Callable[[], Any]] = {
            "robot_status": self.client.status,
            "control_state": self.client.control_state,
            "telemetry": self.client.telemetry,
            "health": self.client.health,
        }
        for name, reader in readers.items():
            try:
                context[name] = reader()
            except Exception as exc:
                context[name] = {"error": f"{type(exc).__name__}: {exc}"}
        if context.get("health") is None:
            context["health_note"] = HEALTH_UNAVAILABLE
        try:
            context["audit_excerpt"] = self.client.events_after(run.event_cursor)[-AUDIT_EXCERPT:]
        except Exception as exc:
            context["audit_excerpt"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            context["frame"] = self.client.frame()
        except Exception:
            context["frame"] = None
        return context

    def _write_artifacts(
        self,
        run_dir: Path,
        step: HilStepDefinition,
        result: ProbeResult,
        outcome: _ProbeOutcome,
        context: dict[str, Any],
    ) -> list[str]:
        folder = run_dir / "artifacts" / step.id
        prefix = f"probe-{result.sequence}"
        written: list[str] = []

        def relative(path: Path) -> str:
            written.append(path.relative_to(run_dir).as_posix())
            return written[-1]

        frame = context.pop("frame", None)
        frame_name = None
        if frame is not None:
            camera, image = frame
            import numpy as np

            buffer = io.BytesIO()
            np.save(buffer, np.asarray(image), allow_pickle=False)
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(camera)) or "camera"
            path = folder / f"{prefix}-frame-{safe}.npy"
            atomic_write_bytes(path, buffer.getvalue())
            frame_name = relative(path)
        samples_name = None
        if outcome.samples is not None:
            path = folder / f"{prefix}-samples.jsonl"
            text = "".join(
                json.dumps(sample, sort_keys=True, default=str) + "\n" for sample in outcome.samples
            )
            atomic_write_text(path, text)
            samples_name = relative(path)
        extras: dict[str, str] = {}
        for name, payload in outcome.extra.items():
            path = folder / f"{prefix}-{name}.json"
            atomic_write_json(path, payload)
            extras[name] = relative(path)
        bundle_path = folder / f"{prefix}-{step.probe}.json"
        bundle = {
            "kind": "so101.hil_probe",
            **result.model_dump(mode="json", exclude={"artifacts"}),
            "samples_file": samples_name,
            "frame_file": frame_name,
            "extra_files": extras,
            **context,
        }
        atomic_write_json(bundle_path, bundle)
        return [relative(bundle_path), *written[:-1]]

    def _sample(
        self, duration_s: float, *, until: Callable[[list[dict[str, Any]]], bool] | None = None
    ) -> list[dict[str, Any]]:
        start = self._monotonic()
        samples: list[dict[str, Any]] = []
        while True:
            elapsed = self._monotonic() - start
            status = self.client.status()
            telemetry = self.client.telemetry()
            samples.append(sample_record(elapsed, self._clock(), status, telemetry))
            if elapsed >= duration_s or (until is not None and until(samples)):
                return samples
            self._sleep(self._interval)

    def _stop_settings(self) -> dict[str, Any]:
        settings = self.settings
        timeout = getattr(settings, "collapse_timeout_seconds", None)
        return {
            "hold_s": float(getattr(settings, "halt_hold_seconds", DEFAULT_HALT_HOLD_S)),
            "grace_s": float(
                getattr(settings, "supervision_grace_seconds", DEFAULT_SUPERVISION_GRACE_S)
            ),
            "speed_deg_s": float(
                getattr(settings, "collapse_speed_deg_s", DEFAULT_COLLAPSE_SPEED_DEG_S)
            ),
            "timeout_s": None if timeout is None else float(timeout),
        }

    @staticmethod
    def _window(requested: float | None, default: float, minimum: float = 1.0) -> float:
        value = requested if requested is not None else default
        return min(max(value, minimum), MAX_PROBE_S)

    def _execute(
        self,
        run: HilRun,
        step: HilStepDefinition,
        duration_s: float | None,
        *,
        status: Mapping[str, Any],
    ) -> _ProbeOutcome:
        kind: ProbeKind = step.probe
        if step.requires:
            needed = [req.split(".", 1)[1] for req in step.requires if req.startswith("status.")]
            missing = [name for name in needed if name not in status]
            if missing:
                return _ProbeOutcome(
                    Verdict(
                        "skip",
                        "incompatible server: /api/robot/status has no required stop field(s) "
                        + ", ".join(missing)
                        + "; use this exact build and probe again before attesting pass",
                        {"missing_status_fields": missing},
                    )
                )
        if kind == "status":
            return _ProbeOutcome(
                analyze_status(status, self.client.fault(), self.client.control_state())
            )
        if kind == "estop":
            return _ProbeOutcome(analyze_estop(status))
        if kind == "events":
            events = self.client.events_after(run.event_cursor)
            return _ProbeOutcome(analyze_events(events, step.probe_events))
        if kind == "diagnostics":
            payload = self.client.diagnostics()
            return _ProbeOutcome(analyze_diagnostics(payload), extra={"diagnostics": payload})
        if kind == "preflight":
            return self._probe_preflight(step)
        if kind == "telemetry":
            window = self._window(duration_s, step.probe_window_s or 10.0)
            samples = self._sample(window)
            return _ProbeOutcome(analyze_telemetry(samples), samples=samples)
        stops = self._stop_settings()
        if kind == "halt_hold":
            window = self._window(
                duration_s, step.probe_window_s or HALT_MIN_HOLD_S + 2, HALT_MIN_HOLD_S + 1
            )
            samples = self._sample(window, until=lambda got: got[-1].get("stop_state") != "halted")
            return _ProbeOutcome(analyze_halt_hold(samples), samples=samples)
        if kind == "collapse":
            return self._probe_collapse(run, duration_s, stops)
        if kind == "hold_timer":
            halt = _mapping(status.get("halt"))
            expires = _parse_time(halt.get("hold_expires_at"))
            server_now = _parse_time(halt.get("server_time"))
            remaining = (
                (expires - server_now).total_seconds()
                if expires and server_now
                else stops["hold_s"]
            )
            window = self._window(duration_s, max(remaining, 0.0) + 20.0)
            samples = self._sample(window, until=_after_state("collapsing", 1.0))
            return _ProbeOutcome(
                analyze_hold_timer(samples, hold_s=stops["hold_s"]), samples=samples
            )
        if kind == "supervision_loss":
            window = self._window(duration_s, step.probe_window_s or 30.0)
            samples = self._sample(window, until=_after_state("collapsing", 1.0))
            return _ProbeOutcome(
                analyze_supervision_loss(samples, grace_s=stops["grace_s"]), samples=samples
            )
        raise HilInvalid(f"step {step.id} has no automatic probe", [f"{step.id}: no probe"])

    def _probe_preflight(self, step: HilStepDefinition) -> _ProbeOutcome:
        api = self.client.preflight()
        cli, note = _cli_report_usable(
            pf.load_latest_report(self.settings, "cli"), api.backend, self._clock()
        )
        verdict = grade_preflight(api, cli, step.probe_checks, cli_note=note)
        extra: dict[str, Any] = {"preflight-api": api.model_dump(mode="json")}
        if verdict.measured.get("cli_report_created_at") and cli is not None:
            extra["preflight-cli"] = cli.model_dump(mode="json")
        return _ProbeOutcome(verdict, extra=extra)

    def _probe_collapse(
        self, run: HilRun, duration_s: float | None, stops: Mapping[str, Any]
    ) -> _ProbeOutcome:
        rest = self.client.rest_pose()
        timeout = stops["timeout_s"]
        budget = timeout if timeout is not None else COLLAPSE_TIMEOUT_MAX_S
        window = self._window(duration_s, COLLAPSE_TRIGGER_ALLOWANCE_S + budget + 3.0)
        try:
            cursor = self.client.latest_event_id()
        except Exception:
            cursor = run.event_cursor

        def done(got: list[dict[str, Any]]) -> bool:
            start = _first_index(got, "collapsing")
            if start is None:
                return False
            end = next(
                (i for i in range(start + 1, len(got)) if got[i].get("stop_state") != "collapsing"),
                None,
            )
            return end is not None and got[-1]["t"] - got[end]["t"] >= 1.0

        samples = self._sample(window, until=done)
        try:
            events = self.client.events_after(cursor)
        except Exception:
            events = []
        verdict = analyze_collapse(
            samples,
            rest_pose=rest.joints,
            speed_cap_deg_s=stops["speed_deg_s"],
            timeout_s=timeout,
            events=events,
        )
        verdict.measured["rest_pose"] = {
            "source": rest.source,
            "joints": rest.joints,
            "placeholder": rest.placeholder,
            "warnings": rest.warnings,
        }
        return _ProbeOutcome(verdict, samples=samples)


def _after_state(state: str, linger_s: float) -> Callable[[list[dict[str, Any]]], bool]:
    """Stop sampling ``linger_s`` after ``stop_state`` first equals ``state``."""

    def done(samples: list[dict[str, Any]]) -> bool:
        index = _first_index(samples, state)
        return index is not None and samples[-1]["t"] - samples[index]["t"] >= linger_s

    return done

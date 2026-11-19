"""Bring-up commands for the ``so101`` CLI: preflight, rest pose and HIL runs.

``cli.py`` mounts :data:`app` with one line, so these appear as ``so101 preflight``,
``so101 rest-pose record|show|clear`` and ``so101 hil list|run|status|report``.
Imports are lazy, as in ``cli.py``, so ``so101 --help`` stays fast.

Bus ownership: a command that opens a serial bus first probes
``GET /api/robot/status`` on the configured Suite port and then takes the
canonical device's host-global advisory lock. A running server owns the buses,
so these commands refuse (exit 3) instead of opening a second connection.
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import get_settings

if TYPE_CHECKING:
    from .bringup.schemas import AttestOutcome
    from .bringup.standalone import StandaloneAccess
    from .hardware.backend import Backend

app = typer.Typer(add_completion=False, help="SO-101 bring-up kit")

#: The exact wording the coordinator specified for ``so101 rest-pose record``.
REST_POSE_PROMPT = (
    "with the padded mechanical catch clear of the rest pose, fold the arm by its links "
    "away from pinch points; move hands clear, let go, wait for it to settle, then press Enter"
)
#: Exit code when the bus belongs to someone else (same value as preflight "incomplete").
EXIT_REFUSED = 3
EXIT_FAILED = 1
EXIT_UNSUPPORTED = 2

ARM_CHOICES: dict[str, tuple[str, ...]] = {
    "follower": ("follower",),
    "leader": ("leader",),
    "both": ("follower", "leader"),
}
TONES = {
    "pass": "green",
    "warn": "yellow",
    "fail": "red",
    "skip": "cyan",
    "incomplete": "magenta",
    "passed": "green",
    "failed": "red",
    "in_progress": "yellow",
    "pending": "white",
}

#: Test seam, looked up at call time.
_sleep: Callable[[float], None] = time.sleep
#: Test seam for finding a hash tool on PATH, looked up at call time.
_which: Callable[[str], str | None] = shutil.which


def _console(*, stderr: bool = False) -> Console:
    # Built per call so CliRunner's environment (COLUMNS, captured streams) applies.
    return Console(stderr=stderr, highlight=False, soft_wrap=False)


def _print_line(out: Console, text: str) -> None:
    """One unwrapped line, so a path or command copies and pastes as is."""
    out.print(escape(text), soft_wrap=True)


def _manual_verify_command(run_dir: object, *, platform: str | None = None) -> str | None:
    """A command that checks ``MANIFEST.sha256`` without the Suite, or None.

    Uses coreutils' ``sha256sum`` (Linux; this Mac has it too) or Perl's
    ``shasum`` (macOS), whichever is on PATH. Windows has neither by default and
    its shells chain commands differently, so there the check is
    ``so101 hil report``.
    """
    if (platform or sys.platform) == "win32":
        return None
    if _which("sha256sum"):
        tool = "sha256sum -c"
    elif _which("shasum"):
        tool = "shasum -a 256 -c"
    else:
        return None
    return f"cd {shlex.quote(str(run_dir))} && {tool} MANIFEST.sha256"


def _logs_to_stderr() -> None:
    """Keep stdout clean for ``--json``: the shared RichHandler writes to stdout."""
    from rich.logging import RichHandler

    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            handler.console = Console(stderr=True)


def _quiet_http_logs() -> None:
    # httpx logs every request at INFO through the shared handler, which would
    # interleave with prompts and tables.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _server_probe(settings: Any) -> Any:
    from .bringup import ownership

    _quiet_http_logs()
    return ownership.probe_local_server(settings)


def _guard_limits() -> tuple[Mapping[str, tuple[float, float]] | None, str]:
    """The SafetyGuard the server would build, and how the rest pose is clamped.

    The rest-pose views clamp exactly as Collapse does: that guard ∩ MJCF.
    """
    from .bringup.preflight import saved_guard_limits

    try:
        return saved_guard_limits(), "SafetyGuard from the saved Suite calibration ∩ MJCF limits"
    except Exception as exc:  # the conservative DEFAULT_LIMITS apply instead
        return None, f"default limits ({type(exc).__name__}: {exc}) ∩ MJCF limits"


#: What ``source`` means in the rest-pose views (the Collapse resolver's order).
_REST_SOURCE_LABELS = {
    "recorded": "recorded with `so101 rest-pose record`",
    "setting": "SO101_REST_POSE",
    "default": "model-derived default",
}


def _tone(status: str) -> str:
    return f"[{TONES.get(status, 'white')}]{status.upper()}[/]"


# --- preflight ----------------------------------------------------------------------


@app.command()
def preflight(
    json_out: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
    arm: str = typer.Option("both", "--arm", help="follower | leader | both"),
    require_hardware: bool = typer.Option(
        False, "--require-hardware", help="Exit 4 unless this ran on the real backend."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Keep the report under <data_dir>/preflight/."
    ),
) -> None:
    """Check host, LeRobot, ports, servos, calibration, rest pose, cameras, disk, E-stop.

    Exit codes: 0 ready (warnings allowed), 1 a check failed, 3 incomplete (a blocking
    check was skipped, for example because a server owns the bus), 4 --require-hardware
    on a rehearsal (mock or sim) run.
    """
    from .bringup import preflight as pf

    if arm not in ARM_CHOICES:
        raise typer.BadParameter("must be follower, leader or both", param_hint="--arm")
    if json_out:
        _logs_to_stderr()
    settings = get_settings()
    report = pf.run_cli_preflight(
        settings,
        arms=ARM_CHOICES[arm],
        require_hardware=require_hardware,
        server_probe=_server_probe,
        write_report=save,
        sleep=_sleep,
    )
    if json_out:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_preflight(report, saved=pf.latest_report_path(settings) if save else None)
    raise typer.Exit(report.exit_code)


def _print_preflight(report: Any, *, saved: Path | None) -> None:
    out = _console()
    label = "REHEARSAL · " if report.rehearsal else ""
    table = Table(
        title=f"SO-101 preflight · {label}{report.backend} backend · {_tone(report.overall)}"
    )
    table.add_column("status", no_wrap=True)
    table.add_column("check", no_wrap=True)
    table.add_column("result")
    table.add_column("remediation")
    for check in report.checks:
        table.add_row(
            _tone(check.status),
            escape(check.key),
            escape(check.message),
            escape(check.remediation) or "—",
        )
    out.print(table)
    out.print(f"bus access: {report.bus_access} · {escape(report.bus_access_detail)}")
    for note in report.notes:
        out.print(f"[bold]note:[/] {escape(note)}")
    counts = ", ".join(f"{status} {count}" for status, count in report.counts.items())
    out.print(
        f"{counts} · ready_for_hardware={str(report.ready_for_hardware).lower()} "
        f"· exit {report.exit_code}"
    )
    if saved is not None:
        out.print(f"report: {saved}")


# --- rest pose ----------------------------------------------------------------------

rest_pose_app = typer.Typer(
    help="Record, show or clear the rest pose that the Collapse stop lowers the arm to.",
    no_args_is_help=True,
)
app.add_typer(rest_pose_app, name="rest-pose")


class _RecordFailed(Exception):
    def __init__(self, message: str, *, spread: Mapping[str, float] | None = None) -> None:
        super().__init__(message)
        self.spread = dict(spread or {})


def _measure_with_bus(
    settings: Any,
    backend: Backend,
    out: Console,
    *,
    yes: bool,
    access: StandaloneAccess,
) -> Any:
    """Torque off, wait for the operator, sample, save. Caller holds the device lock."""
    from .hardware import motors
    from .hardware import rest_pose as rp

    if not yes:
        out.print(
            "[bold]This turns torque OFF on every follower motor.[/] An arm that is not "
            "already at rest can drop. Place it low over a padded mechanical catch that does "
            "not obstruct the intended rest pose, keep the physical cutoff reachable, and "
            "clear the arm's envelope. Do not hold or catch the arm by hand."
        )
        typer.confirm(
            "Mechanical catch, clear envelope and physical cutoff confirmed; continue?",
            abort=True,
            err=True,
        )
    bus = motors.make_bus(settings.follower_port, name="follower")
    try:
        try:
            # The Suite connect path disables torque before anything else
            # (hardware/motors.py:195-201); set_torque(False) makes it explicit.
            bus.connect()
            bus.set_torque(False)
        except Exception as exc:
            raise _RecordFailed(
                f"could not open the follower bus: {type(exc).__name__}: {exc}"
            ) from exc
        typer.prompt(REST_POSE_PROMPT, default="", show_default=False, prompt_suffix=" ", err=True)
        try:
            mean, spread = rp.measure_rest_pose(bus.read_positions, sleep=_sleep)
        except rp.RestPoseUnstableError as exc:
            raise _RecordFailed(str(exc), spread=exc.spread) from None
        except Exception as exc:
            raise _RecordFailed(f"position read failed: {type(exc).__name__}: {exc}") from None
    finally:
        access.disconnect_device(bus, device_name="follower")

    digest = rp.follower_calibration_digest(settings)
    if backend == "real" and digest is None:
        raise _RecordFailed(
            "the follower has no LeRobot calibration file to bind the pose to; run "
            "`so101 calibrate follower` first"
        )
    return rp.save_rest_pose(
        settings,
        mean,
        follower_id=settings.follower_id,
        calibration_digest=digest,
        backend=backend,
        spread=spread,
        samples=rp.RECORD_SAMPLES,
    )


@rest_pose_app.command("record")
def rest_pose_record(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirm non-interactively that the external catch/cutoff setup is ready.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the saved record as JSON."),
) -> None:
    """Measure the folded rest pose with torque off and save it for Collapse.

    Samples Present_Position 10 times, 0.1 s apart, and saves the mean only if every
    joint spread is under 1 degree. The pose is bound to the follower id and its
    LeRobot calibration file, so recalibrating invalidates it.
    """
    from .bringup.ownership import BusOwnershipError, acquire_bus_access
    from .bringup.standalone import adopt_acquired_bus_access, standalone_access
    from .hardware import rest_pose as rp
    from .hardware.backend import current_backend, is_rehearsal

    if json_out:
        _logs_to_stderr()
    out = _console(stderr=json_out)
    settings = get_settings()
    backend = current_backend(settings)
    if backend == "sim":
        out.print(
            "[red]refused:[/] the sim backend has no arm to measure; its rest pose comes "
            "from the model."
        )
        raise typer.Exit(EXIT_UNSUPPORTED)
    if is_rehearsal(backend):
        out.print(
            "[bold yellow]REHEARSAL[/] (mock backend): no arm is read. The saved pose is "
            "synthetic and is only used on the mock backend."
        )
    if backend == "mock":
        # Rehearsal owns no serial device and must not contend with a real
        # server or command that happens to use the same configured strings.
        access = standalone_access("so101 rest-pose record", settings=settings)
    else:
        try:
            lock, _ = acquire_bus_access(
                settings,
                purpose="so101 rest-pose record",
                server_probe=_server_probe,
                ports=(str(settings.follower_port),),
            )
        except BusOwnershipError as exc:
            out.print(f"[red]refused:[/] {escape(exc.detail)}")
            if exc.reason == "server":
                out.print(
                    "Stop the server (Ctrl-C in its terminal) and re-run "
                    "`so101 rest-pose record`; the pose is measured with exclusive bus access."
                )
            else:
                out.print(escape(exc.remediation))
            raise typer.Exit(EXIT_REFUSED) from None

        try:
            access = adopt_acquired_bus_access(
                lock,
                backend=backend,
                purpose="so101 rest-pose record",
                settings=settings,
            )
        except BaseException:
            lock.release()
            raise

    try:
        record = _measure_with_bus(settings, backend, out, yes=yes, access=access)
    except typer.Abort:
        raise
    except _RecordFailed as exc:
        out.print(f"[red]not saved:[/] {escape(str(exc))}")
        if exc.spread:
            _print_spread(out, exc.spread)
        raise typer.Exit(EXIT_FAILED) from None
    except Exception as exc:
        out.print(f"[red]not saved:[/] {escape(str(exc))}")
        raise typer.Exit(EXIT_FAILED) from None
    finally:
        access.release()

    limits, limits_source = _guard_limits()
    resolution = rp.resolve_for_settings(settings, limits=limits)
    path = rp.rest_pose_path(settings)
    if json_out:
        payload = {
            "path": str(path),
            "rehearsal": is_rehearsal(backend),
            "record": record.to_json(),
            "resolution": resolution.model_dump(mode="json"),
            "limits_source": limits_source,
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    table = Table(title=f"rest pose saved · {backend} backend")
    table.add_column("joint")
    table.add_column("rest", justify="right")
    table.add_column("spread", justify="right")
    for joint, value in record.joints.items():
        table.add_row(joint, f"{value:.2f}", f"{record.spread.get(joint, 0.0):.2f}")
    out.print(table)
    out.print(f"saved: {path}")
    out.print(f"Collapse will use the [bold]{resolution.source}[/] pose ({escape(limits_source)}).")
    for warning in resolution.warnings:
        out.print(f"[yellow]warning:[/] {escape(warning)}")


def _print_spread(out: Console, spread: Mapping[str, float]) -> None:
    from .hardware.rest_pose import RECORD_MAX_SPREAD

    table = Table(title=f"spread over the samples (must stay under {RECORD_MAX_SPREAD:g})")
    table.add_column("joint")
    table.add_column("spread", justify="right")
    for joint, value in spread.items():
        mark = "[red]" if value >= RECORD_MAX_SPREAD else ""
        table.add_row(joint, f"{mark}{value:.2f}")
    out.print(table)


@rest_pose_app.command("show")
def rest_pose_show(
    json_out: bool = typer.Option(False, "--json", help="Print the resolution as JSON."),
) -> None:
    """Show the rest pose Collapse would use, where it came from, and any warnings.

    Resolved exactly as Collapse resolves it: recorded, then SO101_REST_POSE,
    then the model-derived default, clamped to SafetyGuard ∩ MJCF.
    """
    from .hardware import rest_pose as rp

    if json_out:
        _logs_to_stderr()
    settings = get_settings()
    limits, limits_source = _guard_limits()
    resolution = rp.resolve_for_settings(settings, limits=limits)
    path = rp.rest_pose_path(settings)
    if json_out:
        payload = {
            "path": str(path),
            "limits_source": limits_source,
            **resolution.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    out = _console()
    label = _REST_SOURCE_LABELS.get(resolution.source, resolution.source)
    table = Table(title=f"rest pose Collapse would use · source: {resolution.source} ({label})")
    table.add_column("joint")
    table.add_column("value", justify="right")
    table.add_column("unit")
    for joint, value in resolution.joints.items():
        table.add_row(joint, f"{value:.2f}", "0..100" if joint == "gripper" else "deg")
    out.print(table)
    out.print(
        f"follower {resolution.follower_id} · {resolution.backend} backend · "
        f"limits: {escape(limits_source)}"
    )
    recorded_at = (resolution.recorded or {}).get("recorded_at")
    out.print(f"file: {path}" + (f" (recorded {recorded_at})" if recorded_at else " (none)"))
    for warning in resolution.warnings:
        out.print(f"[yellow]warning:[/] {escape(warning)}")


@rest_pose_app.command("clear")
def rest_pose_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without asking."),
) -> None:
    """Delete the recorded rest pose; Collapse falls back to SO101_REST_POSE or the default."""
    from .hardware import rest_pose as rp

    out = _console()
    settings = get_settings()
    path = rp.rest_pose_path(settings)
    if not path.exists():
        out.print(f"no recorded rest pose at {path}")
        return
    if not yes:
        typer.confirm(f"Delete the recorded rest pose {path}?", abort=True, err=True)
    rp.clear_rest_pose(settings)
    limits, _ = _guard_limits()
    resolution = rp.resolve_for_settings(settings, limits=limits)
    out.print(f"deleted {path}; Collapse now uses the [bold]{resolution.source}[/] pose")
    for warning in resolution.warnings:
        out.print(f"[yellow]warning:[/] {escape(warning)}")


# --- HIL runs -----------------------------------------------------------------------

hil_app = typer.Typer(
    help="Walk the hardware-in-the-loop checklist, attest each step, and seal the evidence.",
    no_args_is_help=True,
)
app.add_typer(hil_app, name="hil")

#: ``so101 hil report``: the run is not finalized (or does not exist).
EXIT_NOT_FINALIZED = 2
#: ``so101 hil run``: stopped early or sealed with steps not passed (as preflight "incomplete").
EXIT_INCOMPLETE = 3
RUN_EXIT = {"passed": 0, "failed": EXIT_FAILED, "incomplete": EXIT_INCOMPLETE}
OUTCOMES = ("pass", "fail", "skip", "quit")

#: Test seam for sampled probes run in this process.
_monotonic: Callable[[], float] = time.monotonic


class _Refused(Exception):
    def __init__(self, detail: str, remediation: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.remediation = remediation


@dataclass
class _HilSession:
    #: A HilService, or an HttpHilApi for a running server (same methods).
    api: Any
    where: str
    close: Callable[[], None]


def _hil_http(base_url: str) -> Any:
    """Test seam: the HIL API of the server at ``base_url``."""
    from .bringup.hil_remote import HttpHilApi

    _quiet_http_logs()
    return HttpHilApi(base_url)


def _open_hil(settings: Any, *, need_robot: bool) -> _HilSession:
    """The running Suite server when one answers; otherwise this process.

    ``so101 hil run`` needs the robot. On the real backend only the server may
    touch it (it owns the bus), so with no server the run is refused. On mock or
    sim the HIL service runs in this process, as a labeled rehearsal. Reading saved
    runs (list, status, report) needs no robot.
    """
    from .bringup import ownership
    from .bringup.hil import HilService, LocalHilClient, OfflineHilClient, hil_root
    from .hardware.backend import current_backend, is_rehearsal

    probe = _server_probe(settings)
    if probe.running and probe.status is not None:
        api = _hil_http(ownership.server_base_url(settings))
        return _HilSession(api, f"the Suite server at {api.base_url}", api.close)
    if not need_robot:
        return _HilSession(
            HilService(OfflineHilClient(settings)), f"saved runs in {hil_root(settings)}", _noop
        )
    if probe.running:
        raise _Refused(
            probe.detail,
            "Stop the program on that port, or set SO101_PORT to the Suite server's port.",
        )
    backend = current_backend(settings)
    if not is_rehearsal(backend):
        raise _Refused(
            f"no Suite server answers at {probe.url}. On the real backend the probes observe "
            "the arm through the server, which owns the bus.",
            "Start the server with `so101 serve` in another terminal, then re-run this command.",
        )
    from .services.robot_manager import get_manager, reset_manager

    service = HilService(
        LocalHilClient(get_manager(), sleep=_sleep_now),
        sleep=_sleep_now,
        monotonic=_monotonic_now,
    )
    return _HilSession(service, f"this process ({backend} backend, no server)", reset_manager)


def _noop() -> None:
    return None


def _sleep_now(seconds: float) -> None:
    _sleep(seconds)


def _monotonic_now() -> float:
    return _monotonic()


def _read_session(out: Console) -> _HilSession:
    try:
        return _open_hil(get_settings(), need_robot=False)
    except _Refused as exc:  # not raised for reads; kept for symmetry
        out.print(f"[red]refused:[/] {escape(exc.detail)}")
        raise typer.Exit(EXIT_REFUSED) from None


def _print_problems(out: Console, exc: Any) -> None:
    out.print(f"[red]{escape(exc.message)}[/]")
    for problem in exc.problems:
        out.print(f"  - {escape(problem)}")


@hil_app.command("run")
def hil_run(
    operator: str = typer.Option(
        ..., "--operator", "-o", help="Your name; recorded with every attestation."
    ),
    group: list[str] = typer.Option(
        [], "--group", "-g", help="Only these groups: gate, stop, verification (repeatable)."
    ),
    step: list[str] = typer.Option(
        [], "--step", "-s", help="Only these step ids (repeatable; `so101 hil list --steps`)."
    ),
    note: str = typer.Option("", "--note", help="A note for the whole run."),
    resume: str = typer.Option("", "--resume", metavar="RUN_ID", help="Continue an open run."),
) -> None:
    """Walk the HIL steps: instructions, an optional probe, then your outcome and evidence.

    Probes only read what the server observes; you trigger every action. Attest
    pass, fail or skip with the step's evidence values (a fail or skip needs a note).
    Answer `quit` at the outcome prompt to stop; `--resume RUN_ID` continues later.
    After the last step the run is finalized: report.json, report.md and
    MANIFEST.sha256 are written and the run can no longer change.

    With no server, mock and sim runs happen in this process as a labeled rehearsal;
    the real backend needs the server (`so101 serve`), which owns the bus.

    Exit codes: 0 passed, 1 a step failed, 2 bad arguments or the run is already
    finalized, 3 incomplete (stopped early or a step skipped) or refused.
    """
    from pydantic import ValidationError

    from .bringup.hil_steps import GROUPS
    from .bringup.schemas import StartHilRunRequest

    unknown = [name for name in group if name not in GROUPS]
    if unknown:
        raise typer.BadParameter(
            f"unknown group {', '.join(unknown)}; use {', '.join(GROUPS)}", param_hint="--group"
        )
    try:
        request = StartHilRunRequest(
            operator=operator,
            note=note,
            step_ids=list(step) or None,
            groups=list(group) or None,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        raise typer.BadParameter(exc.errors()[0]["msg"], param_hint="--operator") from None

    out = _console()
    try:
        session = _open_hil(get_settings(), need_robot=True)
    except _Refused as exc:
        out.print(f"[red]refused:[/] {escape(exc.detail)}")
        out.print(escape(exc.remediation))
        raise typer.Exit(EXIT_REFUSED) from None
    try:
        code = _drive_run(session, out, request, resume=resume)
    finally:
        session.close()
    raise typer.Exit(code)


def _drive_run(session: _HilSession, out: Console, request: Any, *, resume: str) -> int:
    from .bringup.hil import HilError, HilInvalid, HilNotFound
    from .bringup.hil_steps import HIL_CATALOG_DIGEST, HIL_CATALOG_VERSION

    api = session.api
    try:
        if resume:
            run = api.get_run(resume)
        else:
            run = api.start_run(request)
    except HilInvalid as exc:
        _print_problems(out, exc)
        return EXIT_UNSUPPORTED
    except HilNotFound as exc:
        out.print(f"[red]{escape(exc.message)}[/] (see `so101 hil list`)")
        return EXIT_UNSUPPORTED
    except HilError as exc:
        out.print(f"[red]could not start the run:[/] {escape(exc.message)}")
        return EXIT_REFUSED
    if run.status != "in_progress":
        out.print(
            f"[red]run {run.run_id} is already finalized ({run.status}).[/] "
            f"See `so101 hil report {run.run_id}`."
        )
        return EXIT_UNSUPPORTED

    label = "[bold yellow]REHEARSAL[/]" if run.rehearsal else "[bold]REAL HARDWARE[/]"
    out.print(
        f"HIL run [bold]{run.run_id}[/] · {label} · {run.backend} backend · via {session.where}"
    )
    if run.rehearsal:
        out.print(
            "A rehearsal is never hardware evidence: probes record what the "
            f"{run.backend} backend reports, not an arm."
        )
    catalog = {definition.id: definition for definition in api.steps()}

    def needs_refresh(step_id: str) -> bool:
        step = catalog[step_id]
        record = run.steps[step_id]
        attestation = record.attestation
        if attestation is None:
            return record.status == "pending"
        if (
            attestation.catalog_version != HIL_CATALOG_VERSION
            or attestation.catalog_digest != HIL_CATALOG_DIGEST
        ):
            return True
        if step.group != "stop" or step.probe == "none":
            return False
        latest = record.probes[-1] if record.probes else None
        required = [item.split(".", 1)[1] for item in step.requires if item.startswith("status.")]
        return latest is None or (
            latest.catalog_version != HIL_CATALOG_VERSION
            or latest.catalog_digest != HIL_CATALOG_DIGEST
            or latest.required_status_fields != required
            or not latest.required_status_fields_present
        )

    total = len(run.step_ids)
    pending = [step_id for step_id in run.step_ids if needs_refresh(step_id)]
    if len(pending) < total:
        out.print(f"resuming: {total - len(pending)} of {total} steps have current attestations")
    stale = [
        step_id
        for step_id in pending
        if run.steps[step_id].attestation is not None and run.steps[step_id].status != "pending"
    ]
    if stale:
        out.print(
            "[yellow]evidence contract changed:[/] re-probe measured stops and re-attest: "
            + ", ".join(stale)
        )
    for step_id in pending:
        position = run.step_ids.index(step_id) + 1
        try:
            outcome = _walk_step(api, out, run, catalog[step_id], request.operator, position, total)
        except HilError as exc:
            out.print(f"[red]stopped:[/] {escape(exc.message)}")
            return EXIT_REFUSED
        if outcome == "quit":
            if not typer.confirm(
                "Seal the run now? It is marked incomplete and can no longer change.",
                default=False,
                err=True,
            ):
                _print_line(
                    out,
                    f"Run {run.run_id} is still open. Continue with: so101 hil run "
                    f"--operator {shlex.quote(request.operator)} --resume {run.run_id}",
                )
                return EXIT_INCOMPLETE
            break
    try:
        final = api.finalize(run.run_id)
    except HilError as exc:
        out.print(f"[red]could not finalize:[/] {escape(exc.message)}")
        return EXIT_REFUSED
    _print_final(out, api, final)
    return RUN_EXIT.get(final.status, EXIT_INCOMPLETE)


def _walk_step(
    api: Any, out: Console, run: Any, step: Any, operator: str, position: int, total: int
) -> str:
    from .bringup.hil import HilError, HilInvalid
    from .bringup.schemas import AttestRequest

    out.rule(f"[{position}/{total}] {escape(step.id)} · {escape(step.title)}")
    out.print(f"[dim]source: {escape(step.source)}[/]")
    out.print(f"[dim]“{escape(step.quote)}”[/]")
    if step.known_limit:
        out.print(
            "[bold yellow]Known limit:[/] software cannot stop this case; the step records "
            "that the operator is prepared."
        )
    out.print("[bold]Do:[/]")
    for number, line in enumerate(step.instructions, start=1):
        out.print(f"  {number}. {escape(line)}")
    out.print("[bold]Expect:[/]")
    for line in step.expected_results:
        out.print(f"  • {escape(line)}")

    latest = None
    if step.probe != "none":
        window = f", up to {step.probe_window_s:g} s" if step.probe_window_s else ""
        out.print(f"[bold]Probe[/] ({step.probe}{window}): {escape(step.probe_help)}")
        again = typer.confirm("Run the probe now?", default=True, err=True)
        while again:
            try:
                latest = api.probe(run.run_id, step.id)
            except HilError as exc:
                out.print(f"[red]probe refused:[/] {escape(exc.message)}")
                break
            _print_probe(out, latest)
            again = typer.confirm("Run the probe again?", default=False, err=True)

    while True:
        answer = _ask_choice("Outcome", OUTCOMES)
        if answer == "quit":
            return "quit"
        outcome = cast("AttestOutcome", answer)
        evidence = _ask_evidence(step, outcome, latest)
        need_note = outcome != "pass"
        note = _ask_text("Note (required: what happened?)" if need_note else "Note (optional)")
        while need_note and not note:
            note = _ask_text("A fail or skip needs a note; what happened?")
        try:
            record = api.attest(
                run.run_id,
                step.id,
                AttestRequest(operator=operator, outcome=outcome, note=note, evidence=evidence),
            )
        except HilInvalid as exc:
            _print_problems(out, exc)
            out.print("Nothing was recorded; enter the outcome again.")
            continue
        out.print(f"recorded {_tone(record.status)} for {escape(step.id)}")
        return str(record.status)


def _ask_choice(prompt: str, choices: tuple[str, ...]) -> str:
    # Validated here rather than with a click.Choice: typer vendors its own click,
    # whose prompt does not catch the real click's BadParameter.
    while True:
        raw = str(typer.prompt(f"{prompt} ({', '.join(choices)})", err=True)).strip().lower()
        if raw in choices:
            return raw
        typer.echo(f"  enter one of {', '.join(choices)}", err=True)


def _ask_text(prompt: str) -> str:
    return str(typer.prompt(prompt, default="", show_default=False, err=True)).strip()


def _field_hint(spec: Any) -> str:
    parts: list[str] = []
    if spec.kind == "bool":
        parts.append("y/n")
    elif spec.kind == "choice":
        parts.append("/".join(spec.choices))
    if spec.unit:
        parts.append(spec.unit)
    if spec.pass_min is not None:
        parts.append(f"pass ≥ {spec.pass_min:g}")
    if spec.pass_max is not None:
        parts.append(f"pass ≤ {spec.pass_max:g}")
    if not spec.required:
        parts.append("optional")
    return f" ({'; '.join(parts)})" if parts else ""


def _suggestion(spec: Any, measured: Any) -> str:
    """A probe measurement offered as the default, when it fits the field."""
    if spec.kind == "bool" and isinstance(measured, bool):
        return "y" if measured else "n"
    if (
        spec.kind == "number"
        and isinstance(measured, int | float)
        and not isinstance(measured, bool)
    ):
        return f"{measured:g}"
    if spec.kind == "choice" and measured in spec.choices:
        return str(measured)
    return ""


def _parse_field(spec: Any, raw: str) -> tuple[Any, str | None]:
    import math

    if spec.kind == "bool":
        lowered = raw.lower()
        if lowered in {"y", "yes", "true"}:
            return True, None
        if lowered in {"n", "no", "false"}:
            return False, None
        return None, "answer y or n"
    if spec.kind == "number":
        try:
            value = float(raw)
        except ValueError:
            return None, "enter a number"
        return (value, None) if math.isfinite(value) else (None, "enter a finite number")
    if spec.kind == "choice":
        if raw in spec.choices:
            return raw, None
        return None, f"enter one of {', '.join(spec.choices)}"
    return raw, None


def _ask_evidence(step: Any, outcome: str, latest: Any) -> dict[str, Any]:
    if outcome == "skip":
        return {}
    if outcome == "fail":
        typer.echo(
            "Record what you saw. Enter accepts the value in brackets; "
            "with none shown, it leaves the field out.",
            err=True,
        )
    measured = latest.measured if latest is not None else {}
    values: dict[str, Any] = {}
    for spec in step.evidence_fields:
        required = outcome == "pass" and spec.required
        suggestion = _suggestion(spec, measured.get(spec.name))
        while True:
            raw = str(
                typer.prompt(
                    f"{spec.label}{_field_hint(spec)}",
                    default=suggestion,
                    show_default=bool(suggestion),
                    err=True,
                )
            ).strip()
            if not raw:
                if required:
                    typer.echo("  required for a pass", err=True)
                    continue
                break
            value, problem = _parse_field(spec, raw)
            if problem is None:
                values[spec.name] = value
                break
            typer.echo(f"  {problem}", err=True)
    return values


def _print_probe(out: Console, result: Any) -> None:
    tag = " · REHEARSAL" if result.rehearsal else ""
    out.print(f"probe {result.sequence}: {_tone(result.status)}{tag} · {escape(result.message)}")
    shown = 0
    for key, value in result.measured.items():
        if isinstance(value, bool | int | float | str) and shown < 12:
            out.print(f"    {escape(key)}: {escape(str(value))}")
            shown += 1
    out.print(f"    [dim]{len(result.artifacts)} artifact file(s) saved[/]")


def _print_final(out: Console, api: Any, run: Any) -> None:
    from .bringup.hil import HilError

    counts = ", ".join(f"{name} {count}" for name, count in run.summary.items())
    out.rule(f"run {run.run_id}")
    out.print(f"{_tone(run.status)} · hil_evidence={str(run.hil_evidence).lower()} · {counts}")
    for note in run.evidence_notes:
        out.print(f"[yellow]note:[/] {escape(note)}")
    try:
        report = api.report(run.run_id)
    except HilError as exc:
        out.print(f"[yellow]report unavailable:[/] {escape(exc.message)}")
        return
    verdict = "verifies" if report.manifest.ok else "[red]does NOT verify[/]"
    out.print(f"manifest {verdict}")
    _print_line(out, f"report: {Path(report.run_dir) / 'report.md'}")
    command = _manual_verify_command(report.run_dir)
    if command:
        _print_line(out, f"check it yourself: {command}")
    else:
        _print_line(out, f"check it again: so101 hil report {run.run_id}")


@hil_app.command("list")
def hil_list(
    steps: bool = typer.Option(False, "--steps", help="List the step catalog instead of runs."),
    json_out: bool = typer.Option(False, "--json", help="Print JSON."),
) -> None:
    """List HIL runs, newest first, or the step catalog with --steps."""
    if json_out:
        _logs_to_stderr()
    out = _console()
    session = _read_session(out)
    try:
        items = session.api.steps() if steps else session.api.list_runs()
    finally:
        session.close()
    if json_out:
        typer.echo(json.dumps([item.model_dump(mode="json") for item in items], indent=2))
        return
    if steps:
        table = Table(title="HIL steps")
        for column in ("#", "step", "group", "probe", "title"):
            table.add_column(column, no_wrap=column != "title")
        for item in items:
            table.add_row(str(item.order), item.id, item.group, item.probe, escape(item.title))
        out.print(table)
        return
    if not items:
        out.print(f"no HIL runs yet ({session.where})")
        return
    table = Table(title=f"HIL runs · {session.where}")
    for column in ("run", "status", "backend", "evidence", "steps", "operator"):
        table.add_column(column, no_wrap=True)
    for item in items:
        backend = f"{item.backend} (rehearsal)" if item.rehearsal else item.backend
        table.add_row(
            item.run_id,
            _tone(item.status),
            backend,
            "yes" if item.hil_evidence else "no",
            f"{item.steps_done}/{item.steps_total}",
            escape(item.operator),
        )
    out.print(table)


@hil_app.command("status")
def hil_status(
    run_id: str = typer.Argument("", help="Run id (default: the newest run)."),
    json_out: bool = typer.Option(False, "--json", help="Print the run as JSON."),
) -> None:
    """Show one run step by step: status, probes, and who attested what.

    Exit codes: 0 shown, 2 no such run (or no runs yet).
    """
    from .bringup.hil import HilError

    if json_out:
        _logs_to_stderr()
    out = _console(stderr=json_out)
    session = _read_session(out)
    try:
        if not run_id:
            runs = session.api.list_runs()
            if not runs:
                out.print(f"no HIL runs yet ({session.where})")
                raise typer.Exit(EXIT_NOT_FINALIZED)
            run_id = runs[0].run_id
        run = session.api.get_run(run_id)
    except HilError as exc:
        out.print(f"[red]{escape(exc.message)}[/]")
        raise typer.Exit(EXIT_NOT_FINALIZED) from None
    finally:
        session.close()
    if json_out:
        typer.echo(run.model_dump_json(indent=2))
        return
    label = " · REHEARSAL" if run.rehearsal else ""
    table = Table(title=f"HIL run {run.run_id} · {run.backend}{label} · {_tone(run.status)}")
    for column in ("#", "step", "status", "probe", "attested"):
        table.add_column(column, no_wrap=column != "attested")
    for position, step_id in enumerate(run.step_ids, start=1):
        record = run.steps[step_id]
        probe = "—"
        if record.probes:
            latest = record.probes[-1]
            probe = f"{_tone(latest.status)} ×{len(record.probes)}"
        attested = "—"
        if record.attestation is not None:
            attested = f"{escape(record.attestation.operator)} {record.attestation.attested_at}"
            if record.attestation.note:
                attested += f": {escape(record.attestation.note[:60])}"
        table.add_row(str(position), step_id, _tone(record.status), probe, attested)
    out.print(table)
    out.print(
        f"operator {escape(run.operator)} · started {run.created_at}"
        + (f" · finalized {run.finalized_at}" if run.finalized_at else "")
        + f" · hil_evidence={str(run.hil_evidence).lower()}"
    )
    for note in run.evidence_notes:
        out.print(f"[yellow]note:[/] {escape(note)}")


@hil_app.command("report")
def hil_report(
    run_id: str = typer.Argument(..., help="Run id (see `so101 hil list`)."),
    json_out: bool = typer.Option(
        False, "--json", help="Print the run, report.md and the manifest check as JSON."
    ),
    markdown: bool = typer.Option(False, "--markdown", help="Print report.md as written."),
) -> None:
    """Show a finalized run and re-verify its MANIFEST.sha256.

    Exit codes: 0 the manifest verifies, 1 it does not (a file changed, went
    missing or was added), 2 the run is not finalized or does not exist.
    """
    from .bringup.hil import HilError

    if json_out:
        _logs_to_stderr()
    out = _console(stderr=json_out)
    session = _read_session(out)
    try:
        report = session.api.report(run_id)
    except HilError as exc:
        out.print(f"[red]{escape(exc.message)}[/]")
        raise typer.Exit(EXIT_NOT_FINALIZED) from None
    finally:
        session.close()
    code = 0 if report.manifest.ok else EXIT_FAILED
    if json_out:
        typer.echo(report.model_dump_json(indent=2))
        raise typer.Exit(code)
    if markdown:
        typer.echo(report.markdown)
        raise typer.Exit(code)
    run = report.run
    label = "REHEARSAL · " if run.rehearsal else ""
    out.print(
        f"HIL run [bold]{run.run_id}[/] · {label}{run.backend} backend · {_tone(run.status)} "
        f"· hil_evidence={str(run.hil_evidence).lower()}"
    )
    for note in run.evidence_notes:
        out.print(f"[yellow]note:[/] {escape(note)}")
    manifest = report.manifest
    if manifest.ok:
        out.print(f"manifest: [green]OK[/] · {manifest.checked} files match MANIFEST.sha256")
    else:
        out.print("manifest: [red]FAILED[/]")
        problems = {
            "not present": [] if manifest.manifest_present else ["MANIFEST.sha256"],
            "missing": manifest.missing,
            "changed": manifest.mismatched,
            "not listed": manifest.extra,
            "malformed": manifest.malformed,
        }
        for kind, names in problems.items():
            for name in names:
                out.print(f"  {kind}: {escape(name)}")
    _print_line(out, f"report: {Path(report.run_dir) / 'report.md'}")
    command = _manual_verify_command(report.run_dir)
    if command:
        _print_line(out, f"check it yourself: {command}")
    raise typer.Exit(code)

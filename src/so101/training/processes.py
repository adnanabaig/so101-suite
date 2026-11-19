"""Find and stop a trainer that outlived the API server that started it.

``lerobot-train`` runs in its own session (``start_new_session=True``), so a
terminal's SIGINT or SIGHUP never reaches it. When the server is killed
(SIGKILL, a crash, the OOM killer) the trainer keeps running and keeps the GPU.
The job record carries the trainer's pid, process group and OS start time;
on the next start the server checks that the pid is still *that* trainer and
stops its process group.

A pid is only ever signalled after it is verified: it must still lead the
recorded process group, have the recorded start time (Linux
``/proc/<pid>/stat``, macOS ``ps -o lstart``) and a command line that runs the
training module for this job's output directory (Linux ``/proc/<pid>/cmdline``,
macOS ``ps -o command``). A pid that the OS has since given to an unrelated
process fails these checks and is left alone.

If the parent died between ``Popen`` and persisting the pid, Linux recovery
searches same-user process-group leaders for the durable output-directory
argument. An incomplete process-table read is reported as unknown, never as
gone; the job runner then keeps the GPU slot closed for the next safe retry.

POSIX only. On Windows nothing is inspected or signalled: the trainer's
process-tree stop is still a TODO there (see ``trainer._terminate_process_tree``).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .jobs import TrainingJob

POSIX = os.name == "posix"
_LINUX = sys.platform.startswith("linux")
_PS_TIMEOUT_S = 5.0
# By path where it exists: the server's PATH is whatever started it.
_PS = "/bin/ps" if os.path.exists("/bin/ps") else "ps"


@dataclass(frozen=True)
class ProcessInfo:
    """What the OS reports about a live process."""

    pgid: int
    state: str  # ps/proc state letters; "Z" is a zombie
    start: str  # opaque start-time token, compared only for equality
    cmdline: str  # the arguments joined by spaces


def _linux_info(pid: int) -> ProcessInfo | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # "pid (comm) state ppid pgrp ...": comm may hold spaces and parentheses.
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 20:
        raise ValueError(f"/proc/{pid}/stat is malformed")
    args = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return ProcessInfo(
        pgid=int(fields[2]), state=fields[0], start=fields[19], cmdline=" ".join(args)
    )


def _ps_info(pid: int) -> ProcessInfo | None:
    result = subprocess.run(
        [_PS, "-ww", "-o", "pgid=,stat=,lstart=,command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_PS_TIMEOUT_S,
        env={**os.environ, "LC_ALL": "C"},  # a fixed lstart format
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "no diagnostic"
        raise OSError(f"ps could not inspect pid {pid} (exit {result.returncode}: {detail})")
    # lstart is five words in the C locale: "Wed Oct  7 12:10:44 2026".
    parts = result.stdout.strip().split(None, 7)
    if len(parts) != 8:
        raise ValueError(f"ps returned malformed process metadata for pid {pid}")
    return ProcessInfo(
        pgid=int(parts[0]), state=parts[1], start=" ".join(parts[2:7]), cmdline=parts[7]
    )


def process_info(pid: int) -> ProcessInfo | None:
    """The live process ``pid``, or None when there is none.

    Raises OSError (or ``subprocess.SubprocessError``) when the process exists
    but cannot be inspected; callers then must not touch it.
    """

    if not POSIX:
        raise OSError("process inspection is implemented for POSIX only")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # exists, owned by someone else: the checks below decide
    return _linux_info(pid) if _LINUX else _ps_info(pid)


def process_start_marker(pid: int) -> str | None:
    """The start-time token of ``pid`` (see :class:`ProcessInfo`), best effort."""

    try:
        info = process_info(pid)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return info.start if info is not None else None


def own_process_group(pid: int) -> int | None:
    """The process group a ``start_new_session`` child leads: its pid on POSIX."""

    if not POSIX:
        return None
    try:
        return os.getpgid(pid)
    except OSError:  # already gone: it led its own group all the same
        return pid


@dataclass(frozen=True)
class TrainerCheck:
    """Result of checking an unfinished job's trainer.

    ``state`` is ``gone``, ``ours`` (verified; ``pgid`` set), ``other``
    (positively identified as a different process), or ``unknown`` (inspection
    could not prove either).  ``unknown`` must fail closed: a caller cannot
    assume the GPU is free.
    """

    state: str
    pgid: int | None = None
    detail: str = ""
    pid: int | None = None
    start: str | None = None


def _has_arg(cmdline: str, arg: str) -> bool:
    return re.search(rf"(?:^|\s){re.escape(arg)}(?:\s|$)", cmdline) is not None


def _matches_trainer_command(info: ProcessInfo, job: TrainingJob, module: str) -> bool:
    return bool(
        re.search(rf"(?:^|\s)-m\s+{re.escape(module)}(?:\s|$)", info.cmdline)
        and _has_arg(info.cmdline, f"--output_dir={job.output_dir}")
    )


def _discover_linux_trainer(job: TrainingJob, *, module: str) -> TrainerCheck:
    """Find a trainer whose parent died before it could persist the child's pid.

    The output directory contains the job id and is fixed in the durable job
    record before launch.  A matching process must also lead its own process
    group, as every trainer launched by :mod:`so101.training.trainer` does.
    Any same-user process that cannot be inspected makes the result unknown:
    it could be the missing trainer, so absence is not proven.
    """

    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        return TrainerCheck("unknown", detail=f"the process table could not be read ({exc})")

    uid = os.geteuid()
    matches: list[tuple[int, ProcessInfo]] = []
    matching_followers: set[int] = set()
    errors: list[str] = []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        proc_dir = Path("/proc") / entry
        try:
            if proc_dir.stat().st_uid != uid:
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            errors.append(f"pid {pid}: {exc}")
            continue
        try:
            info = _linux_info(pid)
        except (OSError, ValueError) as exc:
            # A same-UID process can disappear between the directory scan and
            # these reads. Treat that ordinary /proc race as gone, while a
            # still-live process with EIO/invalid metadata remains unknown.
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                pass
            errors.append(f"pid {pid}: {exc}")
            continue
        if info is None or info.state.startswith("Z"):
            continue
        if not _matches_trainer_command(info, job, module):
            continue
        if info.pgid != pid:
            # Forked DataLoader workers can retain the trainer's original
            # argv. They corroborate a matching group leader but cannot alone
            # establish which process is safe to signal.
            matching_followers.add(info.pgid)
            continue
        matches.append((pid, info))

    if errors:
        example = errors[0]
        more = f" and {len(errors) - 1} more" if len(errors) > 1 else ""
        return TrainerCheck(
            "unknown",
            detail=f"the process table could not be inspected completely ({example}{more})",
        )
    if not matches and matching_followers:
        groups = ", ".join(str(pgid) for pgid in sorted(matching_followers))
        return TrainerCheck(
            "unknown",
            detail=(
                f"trainer-like workers remain in process group(s) {groups}, "
                "but no leader was verified"
            ),
        )
    if not matches:
        return TrainerCheck(
            "gone", detail="no trainer for this job was found after its pid was not recorded"
        )
    if len(matches) != 1:
        return TrainerCheck(
            "unknown",
            detail=f"{len(matches)} live trainers match this job; none was signalled",
        )
    pid, info = matches[0]
    if matching_followers - {pid}:
        groups = ", ".join(str(pgid) for pgid in sorted(matching_followers - {pid}))
        return TrainerCheck(
            "unknown",
            detail=f"trainer-like processes also belong to unverified process group(s) {groups}",
        )
    return TrainerCheck(
        "ours",
        pgid=info.pgid,
        pid=pid,
        start=info.start,
        detail=f"found its unrecorded trainer at pid {pid}",
    )


def inspect_trainer(job: TrainingJob, *, module: str) -> TrainerCheck:
    """Is the trainer recorded for ``job`` still running, and verifiably that trainer?"""

    if job.pid is None:
        if not POSIX:
            return TrainerCheck(
                "unknown", detail="this platform cannot search for a trainer whose pid is missing"
            )
        if not _LINUX:
            return TrainerCheck(
                "unknown",
                detail="this platform cannot search for a trainer whose pid was not recorded",
            )
        return _discover_linux_trainer(job, module=module)
    if not POSIX:
        return TrainerCheck(
            "unknown", detail=f"this platform cannot check whether pid {job.pid} still runs"
        )
    try:
        info = process_info(job.pid)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return TrainerCheck("unknown", detail=f"pid {job.pid} could not be inspected ({exc})")
    expected = job.pgid if job.pgid is not None else job.pid
    if info is None or info.state.startswith("Z"):
        if group_alive(expected):
            state = "missing" if info is None else "a zombie"
            return TrainerCheck(
                "unknown",
                detail=(
                    f"the recorded trainer leader pid {job.pid} is {state}, but process group "
                    f"{expected} still has live or uninspectable members"
                ),
            )
        return TrainerCheck("gone", detail=f"its trainer (pid {job.pid}) is no longer running")
    reasons = []
    if info.pgid != expected:
        reasons.append(f"it is in process group {info.pgid}, not {expected}")
    if job.pid_start is not None and info.start != job.pid_start:
        reasons.append("it started at a different time")
    if not _matches_trainer_command(info, job, module):
        reasons.append("its command line is not this job's trainer")
    if reasons:
        return TrainerCheck(
            "other",
            detail=f"pid {job.pid} now belongs to another process ({'; '.join(reasons)}); "
            "it was left alone",
        )
    return TrainerCheck("ours", pgid=expected, pid=job.pid, start=info.start)


def _linux_group_has_live_member(pgid: int) -> bool:
    """True if ``pgid`` is live, including an ambiguous process-table read.

    ``killpg(..., 0)`` has already proved that the group exists. A stat entry
    that cannot be read or parsed could be its remaining worker, so treating
    that ambiguity as gone would open the GPU slot before stop was verified.
    """

    try:
        entries = os.listdir("/proc")
    except OSError:
        return True
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            fields = Path(f"/proc/{entry}/stat").read_text(errors="replace").rpartition(")")[2]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return True
        parts = fields.split()
        if len(parts) <= 2:
            return True
        try:
            process_group = int(parts[2])
        except ValueError:
            return True
        if process_group == pgid and parts[0] not in ("Z", "X"):
            return True
    return False


def group_alive(pgid: int) -> bool:
    """True while a process of group ``pgid`` that can still run exists.

    EPERM counts as gone: macOS answers it for a group of zombies, and this is
    only asked about a group this user could signal before.
    """

    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return _linux_group_has_live_member(pgid) if _LINUX else True


def _wait_gone(pgid: int, timeout: float, hurry: threading.Event | None, poll: float) -> bool:
    deadline = time.monotonic() + timeout
    while group_alive(pgid):
        if time.monotonic() >= deadline or (hurry is not None and hurry.is_set()):
            return not group_alive(pgid)
        time.sleep(poll)
    return True


def stop_process_group(
    pgid: int,
    *,
    grace_s: float,
    kill_wait_s: float,
    hurry: threading.Event | None = None,
    poll_s: float = 0.05,
) -> bool:
    """SIGTERM the group, SIGKILL it after ``grace_s``; True once nothing is left.

    ``hurry`` (a cancel or a server shutdown) cuts the grace period short.
    """

    if not POSIX:
        return False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return not group_alive(pgid)
    if _wait_gone(pgid, grace_s, hurry, poll_s):
        return True
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return _wait_gone(pgid, kill_wait_s, None, poll_s)

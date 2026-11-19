"""Build the handoff folder for the Zephyrus (Ubuntu 24.04) from a clean commit.

    python scripts/make_handoff.py --out DIR [--wheel FILE] [--allow-no-ui]
    make handoff OUT=DIR

    python scripts/make_handoff.py --archive FILE.tar.gz --expected-commit FULL_SHA
    make handoff-archive OUT=FILE.tar.gz HANDOFF_APPROVED_COMMIT=FULL_SHA

DIR, copied to the target by USB stick or scp, holds:

- ``so101/``: a git clone of the current branch with its full history (only
  tracked files: no .venv, node_modules, datasets, outputs or data), with no
  remote;
- ``wheels/so101-<version>-py3-none-any.whl``: the suite with the web UI built
  in, so the target needs no Node.js. Built from a temporary clone of
  ``so101/`` checked out (detached) at its exact commit (npm ci, the UI build,
  then ``python -m build``), or taken from ``--wheel``. Either way the wheel is
  checked against that commit: its METADATA version is the commit's, its
  ``so101/`` files are byte for byte the commit's ``src/so101`` (the UI aside),
  and its UI carries a build stamp (hatch_build.py) whose commit is this one,
  whose sources digest is that of the commit's frontend sources, and whose
  manifest lists exactly the UI files with their hashes;
- ``install.sh``: runs ``so101/install/install.sh`` with that wheel, through
  ``bash``, so it works from a FAT32 or exFAT USB stick that drops the exec
  bit (start it with ``bash install.sh``). A deliberately UI-less handoff adds
  ``--no-launcher --allow-no-ui`` to that fixed wrapper;
- ``START_HERE.md``: the runbook;
- ``SHA256SUMS``: every file outside ``so101/.git`` (git checks its own
  objects: ``git -C so101 fsck``). Check it with ``sha256sum -c SHA256SUMS``.

It refuses a working tree with uncommitted changes or untracked files, an output
folder inside the repository, and an output folder that isn't empty. It never
writes into the repository. The folder is assembled in a sibling temporary
directory and published only when complete, so a failed build leaves OUT
untouched and can be retried with the same path.
"""

from __future__ import annotations

import argparse
import errno
import functools
import hashlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
UI_WEB = "so101/web"
UI_INDEX = f"{UI_WEB}/index.html"
PACKAGE_INIT = "so101/__init__.py"
PACKAGE_SRC = "src/so101"
LISTED = 10  # mismatched files named in an error
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ARCHIVE_ROOT = "so101-handoff"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class HandoffError(Exception):
    """Something the user has to fix; the message says what."""


def _run(cmd: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    kwargs.setdefault("text", True)
    return subprocess.run(list(cmd), check=True, **kwargs)  # type: ignore[call-overload,no-any-return]


@functools.cache
def _hatch_build() -> ModuleType:
    """The build hook, for its UI-stamp helpers (it imports hatchling, a dev dependency)."""
    spec = importlib.util.spec_from_file_location("so101_hatch_build", REPO / "hatch_build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str, run: Runner = _run) -> str:
    result = run(["git", "-C", str(repo), *args], capture_output=True)
    return str(result.stdout).strip()


def check_clean(repo: Path, run: Runner = _run) -> None:
    status = git(repo, "status", "--porcelain", "--untracked-files=normal", run=run)
    if status:
        lines = status.splitlines()
        shown = "\n  ".join(lines[:10]) + ("\n  ..." if len(lines) > 10 else "")
        raise HandoffError(
            "the working tree isn't clean; commit or stash these first, so the handoff is "
            f"exactly a commit:\n  {shown}"
        )


def check_out_dir(out: Path, repo: Path) -> None:
    out_abs, repo_abs = out.resolve(), repo.resolve()
    if out_abs == repo_abs or repo_abs in out_abs.parents:
        raise HandoffError(f"{out} is inside the repository; pick a folder outside {repo}")
    if out_abs.exists() and (not out_abs.is_dir() or any(out_abs.iterdir())):
        raise HandoffError(f"{out} exists and isn't an empty folder")


def check_archive_out(out: Path, repo: Path) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz", out.name) is None:
        raise HandoffError("archive output must have a plain NAME.tar.gz filename")
    out_abs, repo_abs = out.resolve(), repo.resolve()
    if out_abs == repo_abs or repo_abs in out_abs.parents:
        raise HandoffError(f"{out} is inside the repository; pick a file outside {repo}")
    checksum = Path(str(out) + ".sha256")
    occupied = [path for path in (out, checksum) if path.exists()]
    if occupied:
        raise HandoffError("archive output already exists: " + ", ".join(map(str, occupied)))
    return checksum


def check_expected_head(repo: Path, expected_commit: str, run: Runner = _run) -> str:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise HandoffError("the approved commit must be a full 40-character lowercase Git commit")
    actual = git(repo, "rev-parse", "HEAD", run=run)
    if actual != expected_commit:
        raise HandoffError(f"HEAD is {actual}, not the approved commit {expected_commit}")
    return actual


def _file_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    return metadata.st_dev, metadata.st_ino


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise


def _publish_exclusive(source: Path, destination: Path) -> tuple[int, int]:
    """Publish ``source`` without replacing a path that appeared meanwhile."""

    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise HandoffError(f"archive output appeared while building: {destination}") from exc
    except OSError as link_error:
        unsupported = {errno.EACCES, errno.EPERM, errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP}
        if link_error.errno not in unsupported:
            raise
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(destination, flags, 0o600)
        except FileExistsError as exc:
            raise HandoffError(f"archive output appeared while building: {destination}") from exc
        identity = os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino
        try:
            with os.fdopen(descriptor, "wb") as outgoing, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, outgoing, 1 << 20)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except BaseException:
            try:
                if destination.exists() and _file_identity(destination) == identity:
                    destination.unlink()
            finally:
                raise
    identity = _file_identity(destination)
    try:
        _fsync_directory(destination.parent)
    except BaseException:
        _rollback_publication(destination, identity)
        raise
    return identity


def _rollback_publication(path: Path, identity: tuple[int, int]) -> None:
    try:
        if path.exists() and _file_identity(path) == identity:
            path.unlink()
            _fsync_directory(path.parent)
    except OSError:
        # The primary publication error remains authoritative. A replacement
        # from another process is never unlinked or overwritten.
        pass


def commit_tree(repo: Path, commit: str, run: Runner = _run) -> dict[str, bytes]:
    """Every file of ``commit``, path -> contents, as a checkout of it has them."""
    archived = run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True, text=False
    )
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as archive:  # type: ignore[arg-type]
        for member in archive.getmembers():
            extracted = archive.extractfile(member) if member.isfile() else None
            if extracted is not None:
                files[member.name] = extracted.read()
    return files


def project_version(tree: Mapping[str, bytes]) -> str:
    return str(tomllib.loads(tree["pyproject.toml"].decode("utf-8"))["project"]["version"])


def wheel_name(version: str) -> str:
    return f"so101-{version}-py3-none-any.whl"


def _mismatch(what: str, paths: list[str]) -> str:
    shown = ", ".join(paths[:LISTED]) + (
        f" and {len(paths) - LISTED} more" if len(paths) > LISTED else ""
    )
    return f"{what}: {shown}"


def check_wheel(
    wheel: Path, version: str, commit: str, tree: Mapping[str, bytes], *, allow_no_ui: bool
) -> bool:
    """Refuse ``wheel`` unless it is the suite at ``commit`` (``tree``: the commit's files).

    The name, the METADATA version, the package files (byte for byte the
    commit's src/so101) and, when it has a UI, the UI's build stamp: built at
    ``commit`` from the commit's frontend sources, its manifest exactly the UI
    files. Without a UI it needs ``allow_no_ui``.
    """

    if wheel.name != wheel_name(version):
        raise HandoffError(f"{wheel.name} isn't {wheel_name(version)} (the commit's version)")
    try:
        with zipfile.ZipFile(wheel) as archive:
            files = {
                info.filename: archive.read(info)
                for info in archive.infolist()
                if not info.is_dir()
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise HandoffError(f"{wheel} isn't a wheel: {exc}") from exc
    metadata = [
        name for name in files if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
    ]
    if len(metadata) != 1:
        raise HandoffError(f"{wheel.name} has {len(metadata)} .dist-info/METADATA files, not one")
    fields = Parser().parsestr(files[metadata[0]].decode("utf-8", "replace"))
    if fields.get("Name") != "so101" or fields.get("Version") != version:
        raise HandoffError(
            f"{wheel.name}: METADATA says {fields.get('Name') or '(no Name)'} "
            f"{fields.get('Version') or '(no Version)'}, "
            f"not so101 {version} (the commit's version)"
        )
    allowed = {"so101", metadata[0].split("/")[0]}
    others = sorted({name.split("/")[0] for name in files} - allowed)
    if others:
        # pip would install them next to so101 (a sitecustomize.py runs in every python).
        raise HandoffError(f"{wheel.name} holds more than so101: {', '.join(others)}")
    if PACKAGE_INIT not in files:
        raise HandoffError(f"{wheel.name} has no {PACKAGE_INIT}, so it isn't the suite")
    packaged = {
        name: data
        for name, data in files.items()
        if name.startswith("so101/") and not name.startswith(f"{UI_WEB}/")
    }
    expected = {
        "so101/" + name.removeprefix(f"{PACKAGE_SRC}/"): data
        for name, data in tree.items()
        if name.startswith(f"{PACKAGE_SRC}/")
    }
    problems = []
    for what, paths in (
        (f"files the commit's {PACKAGE_SRC} doesn't have", sorted(set(packaged) - set(expected))),
        (f"files of the commit's {PACKAGE_SRC} it lacks", sorted(set(expected) - set(packaged))),
        (
            "files that differ from the commit",
            sorted(n for n in set(packaged) & set(expected) if packaged[n] != expected[n]),
        ),
    ):
        if paths:
            problems.append(_mismatch(what, paths))
    if problems:
        raise HandoffError(
            f"{wheel.name} isn't the suite at {commit[:12]}:\n  " + "\n  ".join(problems)
        )
    ui = {
        name.removeprefix(f"{UI_WEB}/"): data
        for name, data in files.items()
        if name.startswith(f"{UI_WEB}/")
    }
    if not ui:
        if not allow_no_ui:
            raise HandoffError(
                f"{wheel.name} has no {UI_INDEX}, so the target would have no web UI; build it "
                "with the UI (make ci-wheel) or pass --allow-no-ui"
            )
        return False
    _check_wheel_ui(wheel.name, ui, commit, tree)
    return True


def _check_wheel_ui(
    name: str, ui: Mapping[str, bytes], commit: str, tree: Mapping[str, bytes]
) -> None:
    hb = _hatch_build()
    frontend = {
        path.removeprefix("frontend/"): data
        for path, data in tree.items()
        if path.startswith("frontend/")
    }
    hashes = {path: hb.sha256_hex(data) for path, data in ui.items() if path != hb.UI_STAMP}
    stamp, problems = hb.ui_build_problems(
        hashes, ui.get(hb.UI_STAMP), hb.ui_sources_digest(frontend)
    )
    if stamp is not None and stamp["git_head"] != commit:
        problems.insert(
            0, f"its UI was built at {stamp['git_head'] or 'an unknown commit'}, not {commit}"
        )
    if "index.html" not in ui:
        problems.insert(0, f"it has no {UI_INDEX}")
    if problems:
        raise HandoffError(
            f"{name}: the web UI isn't the build of {commit[:12]}:\n  " + "\n  ".join(problems)
        )


def clone(repo: Path, branch: str, dest: Path, run: Runner = _run) -> None:
    """A self-contained clone of ``branch`` with its full history and no remote.

    ``--no-local`` transfers only the objects the branch reaches, so leftovers
    such as amended or reset commits stay behind.
    """

    run(
        ["git", "clone", "--quiet", "--no-local", "--single-branch", "--branch", branch]
        + [str(repo), str(dest)],
        capture_output=True,
    )
    git(dest, "remote", "remove", "origin", run=run)
    # FAT/exFAT drops executable bits. This is clone-local: M0 still verifies
    # every tracked byte and untracked path, but mode-only loss cannot make the
    # freshly extracted candidate look dirty before its Bash entry points run.
    git(dest, "config", "--local", "core.filemode", "false", run=run)


def build_wheel(source: Path, commit: str, version: str, dest: Path, run: Runner = _run) -> Path:
    """Build the wheel with the UI at exactly ``commit``; return its path in ``dest``.

    ``source`` is the handoff's own clone. The build runs in a temporary clone
    of it checked out (detached) at ``commit``, so a commit landing on the
    branch meanwhile cannot pair this source with another commit's wheel.
    """

    with tempfile.TemporaryDirectory(prefix="so101-handoff-") as tmp:
        build = Path(tmp) / "src"
        run(
            ["git", "clone", "--quiet", "--no-local", "--no-checkout", str(source), str(build)],
            capture_output=True,
        )
        git(build, "checkout", "--quiet", "--detach", commit, run=run)
        checked_out = git(build, "rev-parse", "HEAD", run=run)
        if checked_out != commit:
            raise HandoffError(f"the build clone is at {checked_out}, not {commit}")
        frontend = build / "frontend"
        run(["npm", "ci"], cwd=str(frontend))
        run(["npm", "run", "build"], cwd=str(frontend))
        dist = Path(tmp) / "dist"
        run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(build)],
            env={**os.environ, "SO101_REQUIRE_UI": "1"},
        )
        built = dist / wheel_name(version)
        if not built.is_file():
            found = ", ".join(sorted(p.name for p in dist.glob("*.whl"))) or "nothing"
            raise HandoffError(f"the build made {found}, not {built.name}")
        dest.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy2(built, dest / built.name))


def wrapper_script(version: str, *, allow_no_ui: bool = False) -> str:
    backend_only = " --no-launcher --allow-no-ui" if allow_no_ui else ""
    install = (
        'exec bash "$HERE/so101/install/install.sh" '
        f'--wheel "$HERE/wheels/{wheel_name(version)}"{backend_only} "$@"'
    )
    return f"""#!/usr/bin/env bash
# SO-101 Suite handoff: install the prebuilt wheel with so101/install/install.sh.
# Options go through, for example: bash install.sh --realsense
# Everything runs through bash, not the exec bit: a FAT32 or exFAT USB stick
# (udisks mounts it with showexec) drops the exec bit from every file.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
{install}
"""


def runbook(version: str, branch: str, commit: str, built: str) -> str:
    return f"""# SO-101 Suite {version}: start here

Target: ASUS ROG Zephyrus G15 (RTX 3070 Laptop GPU) running Ubuntu 24.04 LTS,
with both arms and the RealSense camera available. Keep both arm motor supplies
switched off until the physical checks in step 4 are complete.

This folder is branch `{branch}` at commit `{commit}`, built {built}.
On the laptop, copy or extract this whole folder, enter that copied folder, and
verify it before following any setup or install step below:

    sha256sum -c SHA256SUMS

Only continue after every bundled file passes.
The bundled clone has clone-local `core.filemode=false`, so a FAT/exFAT copy
that removes executable bits remains clean; Git still detects changed bytes and
untracked files, and `SHA256SUMS` still covers every bundled file outside
`.git`. Always invoke the scripts through Bash as shown below.

## M0. One-shot software gate from the fresh extraction

Leave every arm and camera disconnected and unpowered. Before installing the
suite or creating any reusable build state, run the exact candidate's software
gate once from inside the freshly extracted `so101/` repository:

    (
        cd so101
        /usr/bin/make g15-ci-evidence \\
          G15_CI_EXPECTED_COMMIT={commit} \\
          G15_CI_EVIDENCE="$PWD/../../evidence/g15-ci-{commit[:12]}"
    )

The G15 gate requires Ubuntu 24.04, system Python 3.12 and Node 24. Its evidence
path resolves outside both the Git repository and this bundle. If it refuses or
fails, preserve that evidence and start the next attempt from another fresh
extraction; never delete state or retry a failed command to make it green. Only
continue below after M0 passes. M0 never runs doctor, preflight, calibration,
camera discovery or HIL and does not authorize connecting hardware.

## 1. Ubuntu 24.04 and the NVIDIA driver

Install Ubuntu 24.04 LTS, then the recommended NVIDIA driver:

    sudo ubuntu-drivers install

With Secure Boot on, the installer asks for a password to sign the driver; at
the next boot choose "Enroll MOK" and enter it, or the driver doesn't load.
Hybrid graphics (the integrated GPU drives the screen) is fine for CUDA. Reboot,
then `nvidia-smi` should list the RTX 3070 and the driver version.

## 2. Install the suite

The arm USB adapters may be connected so their serial numbers can be listed,
but leave both motor supplies off. Review a dry run and then install as your
normal user (do not run the script itself with sudo):

    bash install.sh --realsense --profile cuda --dry-run
    bash install.sh --realsense --profile cuda

(Start it with `bash`: a FAT32 or exFAT USB stick drops the exec bit, so
`./install.sh` would fail with "Permission denied". Running it straight from
the stick is fine.)

It installs into `~/.local/opt/so101-suite`. `--profile cuda` refuses to hide
a broken NVIDIA setup behind a CPU-only PyTorch build. Every step that needs
root (apt packages, the dialout group, udev rules) is shown first and runs only
after you answer y. Read each command before confirming it.
To give the arm adapters stable names, add both serial numbers (the platform
report at the end of the install lists them). Physically trace each cable or
unplug one USB adapter at a time; never guess which arm a serial belongs to:

    bash install.sh --realsense --follower-serial SERIAL --leader-serial SERIAL --profile cuda

The installer keeps settings and data in `~/.local/share/so101`. Its `.env`
there gets the data folders and, with the serial numbers, the arm ports
(`/dev/so101-follower` and `/dev/so101-leader`); each added line is shown,
and a value already there is replaced only if you answer y.

## 3. Log out and back in

So the dialout group and `~/.local/bin` (where `so101` and `so101-suite` are
linked) take effect.

## 4. Physical safety and device identity

Keep both arm motor supplies switched off while two people, where possible,
confirm all of these:

- Clamp or weigh down both bases, clear the whole arm envelope, remove the
  payload and put a padded mechanical catch under the follower. Position it to
  arrest a drop without obstructing the recorded rest pose or Collapse path.
  Never plan to catch a falling or moving arm by hand.
- Put the follower power switch where a dedicated power-switch operator can
  cut power without reaching across the arm. A second person should operate
  the Suite during first power and motion.
- Confirm the follower and leader supply labels and servo variants match the
  configured health supply profiles. An ambiguous or wrong supply is a no-go.
- Follow each USB cable to its arm. Unplug and replug one at a time to prove
  `/dev/so101-follower` and `/dev/so101-leader` name the correct, different
  adapters and remain stable. Confirm both udev properties include
  `ID_MM_DEVICE_IGNORE=1` so ModemManager leaves them alone.

The operator, and the power-switch operator when that is a separate person,
confirm the workspace, mechanical catch, supplies, aliases and cutoff before
anyone may switch on an arm supply.

## 5. Doctor (arm power still off)

Every `so101` command reads `~/.local/share/so101/.env` and keeps its data in
that folder, whatever folder you run it from:

    so101 doctor

The "Settings and data folders" row names the settings files and folders in
use (`so101 info` lists them too).

Fix every FAIL. The laptop-sleep warning matters: a suspended laptop can't stop
the arm. Confirm the report says backend `real`, CUDA sees the RTX 3070,
MuJoCo's EGL render passes, the serial aliases and RealSense are ready, and
automatic suspend on AC is disabled. A mock or sim report is only a rehearsal.

## 6. First power and read-only preflight

Switch on the arm supplies only after the operator completed step 4. Keep the
follower low over the mechanical catch, hands outside its envelope and the
power-switch operator at the cutoff. Stop any running Suite, then run the
read-only preflight; it pings and reads registers but never writes or changes
torque:

    so101 preflight

Before first calibration, only the missing calibration checks and a missing
rest-pose warning are expected. Any other failure is a no-go: cut power,
diagnose it and repeat preflight rather than continuing.

## 7. Calibrate and record the rest pose (server stopped)

Calibration and `rest-pose record` turn torque off. The follower can drop.
Keep it low over the padded mechanical catch, keep hands out of its envelope,
and have the second person, when present, remain at the physical cutoff. The
operator must confirm those conditions at each prompt; do not use `--yes` on
this first run. Calibration is a hands-on sweep: move only the intended links
when prompted, keep fingers away from pinch points and move clear before the
command finishes. Each real calibration command asks for this external safety
confirmation before it opens a bus.

Place each arm at its midpoint before starting its calibration command. Do
that while torque is off, over the mechanical catch and before the bus opens;
the timed sweep uses that starting pose as its midpoint convention.

    so101 calibrate follower
    so101 calibrate leader
    so101 rest-pose record

Record the rest pose after calibration because recalibrating invalidates it.
Then `so101 preflight --require-hardware` must exit 0 with real hardware
evidence before the server may start.

## 8. Start the suite

Start the Suite in one terminal and leave it running:

    so101-suite

It opens http://localhost:8000 in the browser. Keep that terminal visible;
closing it shuts the server down and can move the arm through its shutdown
Collapse.

## 9. The HIL checklist

In a second terminal, run all three groups in this safety-first order, one run
each:

    so101 hil run --operator "Your Name" --group gate
    so101 hil run --operator "Your Name" --group stop
    so101 hil run --operator "Your Name" --group verification

The runner does not check that a group passed before you start the next one,
so the human operator does: start `stop` only after `gate` passed (exit 0),
and start data, training and rollout `verification` only after `stop` passed.

The `stop` group is the real-arm E-stop, Halt and Collapse test. The arm drops
when torque goes off and lowers itself on a Collapse. Keep the follower low
over its padded mechanical catch for the whole `stop` group, keep hands and
people clear, and keep the power-switch operator at the cutoff. Cut power at
once if the arm does anything the step does not describe. Every latch reset
and every HIL attestation is a deliberate human confirmation; never automate a
reset to make a step pass.

Do not stop the server between HIL steps and never run standalone calibration
or `rest-pose record` while it owns the buses. The collection/resume step
deliberately restarts it to test recovery; follow that step's controlled
shutdown and resume the same RUN_ID. For a non-default hold-timer setting,
pause without sealing, confirm torque off, restart the same build/backend/data
folder, and continue with `so101 hil run --operator "Your Name" --resume RUN_ID`.
Release evidence uses the default hold-timer setting without a restart.

Each run ends with its run id and the path of its `report.md`. Check each run:

    so101 hil report RUN_ID

docs/bring-up.md (in `so101/`) has the details. Preserve each whole sealed HIL
run directory as evidence; editing one invalidates its manifest.

## 10. Send this back

    so101 doctor --json > ~/doctor.json

Send `~/doctor.json` and the complete sealed HIL run directories for gate,
stop and verification back. They include each `report.md` and its checksum
manifest. Review them for operator names, device serials and camera frames
before sharing; redact a copy, never the sealed original.
"""


def checksums(out: Path) -> str:
    """``<sha256>  <path>`` for every file outside so101/.git, sorted by path."""

    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() or p.is_symlink()):
        relative = path.relative_to(out).as_posix()
        if relative == "SHA256SUMS" or relative.startswith("so101/.git/"):
            continue
        if path.is_symlink():
            continue  # sha256sum follows links; the target file is listed itself
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    return "\n".join(lines) + "\n"


def folder_size(out: Path) -> int:
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())


def make_handoff(
    out: Path,
    *,
    repo: Path = REPO,
    wheel: Path | None = None,
    allow_no_ui: bool = False,
    expected_commit: str | None = None,
    run: Runner | None = None,
) -> Path:
    run = run or _run
    repo = Path(git(repo, "rev-parse", "--show-toplevel", run=run))
    check_clean(repo, run=run)
    check_out_dir(out, repo)
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD", run=run)
    if branch == "HEAD":
        raise HandoffError("HEAD is detached; check out the branch to hand off")
    commit = (
        check_expected_head(repo, expected_commit, run=run)
        if expected_commit is not None
        else git(repo, "rev-parse", "HEAD", run=run)
    )
    # The commit's own files, read by id, so a commit landing meanwhile can't change them.
    tree = commit_tree(repo, commit, run=run)
    version = project_version(tree)
    if wheel is not None:
        check_wheel(wheel, version, commit, tree, allow_no_ui=allow_no_ui)

    # Build beside the destination and publish only once it is complete. A
    # failed npm/build/git command therefore cannot leave OUT non-empty and
    # block a retry. The sibling is also on the same filesystem, including a
    # FAT/exFAT handoff drive, so the final rename does not become a copy.
    staging_root: Path | None = None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
        staged = staging_root / "handoff"
        staged.mkdir()
    except OSError as exc:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        raise HandoffError(f"cannot create the handoff beside {out}: {exc}") from exc
    assert staging_root is not None
    try:
        clone(repo, branch, staged / "so101", run=run)
        cloned = git(staged / "so101", "rev-parse", "HEAD", run=run)
        if cloned != commit:
            raise HandoffError(f"the clone is at {cloned}, not {commit}")
        wheels = staged / "wheels"
        wheel_has_ui: bool
        if wheel is None:
            built = build_wheel(staged / "so101", commit, version, wheels, run=run)
            wheel_has_ui = check_wheel(built, version, commit, tree, allow_no_ui=allow_no_ui)
        else:
            wheels.mkdir()
            copied = Path(shutil.copy2(wheel, wheels / wheel.name))
            # The source wheel can be replaced during the clone above. Trust
            # only the staged bytes that will actually be published.
            wheel_has_ui = check_wheel(copied, version, commit, tree, allow_no_ui=allow_no_ui)

        script = staged / "install.sh"
        script.write_text(
            wrapper_script(version, allow_no_ui=not wheel_has_ui),
            encoding="utf-8",
            newline="\n",
        )
        try:
            script.chmod(0o755)
        except OSError:
            # udisks FAT/exFAT mounts can reject chmod; START_HERE deliberately
            # invokes this wrapper with bash, so its exec bit is not required.
            pass
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        (staged / "START_HERE.md").write_text(
            runbook(version, branch, commit, stamp), encoding="utf-8", newline="\n"
        )
        (staged / "SHA256SUMS").write_text(checksums(staged), encoding="utf-8", newline="\n")

        # Re-check after the potentially long build. Never replace a folder
        # somebody populated while it was running. The final same-filesystem
        # replace is atomic and, on failure/interruption, leaves a preexisting
        # empty OUT (including its inode, mode and ACL) untouched.
        if expected_commit is not None:
            check_expected_head(repo, expected_commit, run=run)
            check_clean(repo, run=run)
        check_out_dir(out, repo)
        try:
            os.replace(staged, out)
        except OSError as exc:
            raise HandoffError(f"could not publish the completed handoff at {out}: {exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return out


def make_handoff_archive(
    out: Path,
    *,
    expected_commit: str,
    repo: Path = REPO,
    wheel: Path | None = None,
    allow_no_ui: bool = False,
    run: Runner | None = None,
) -> tuple[Path, Path]:
    """Publish one extractable bundle plus its external SHA-256 sidecar."""

    run = run or _run
    repo = Path(git(repo, "rev-parse", "--show-toplevel", run=run))
    check_expected_head(repo, expected_commit, run=run)
    check_clean(repo, run=run)
    checksum_out = check_archive_out(out, repo)
    staging_root: Path | None = None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
        bundle = make_handoff(
            staging_root / ARCHIVE_ROOT,
            repo=repo,
            wheel=wheel,
            allow_no_ui=allow_no_ui,
            expected_commit=expected_commit,
            run=run,
        )
        staged_archive = staging_root / out.name
        with tarfile.open(staged_archive, "w:gz") as archive:
            archive.add(bundle, arcname=ARCHIVE_ROOT, recursive=True)
        digest = hashlib.sha256(staged_archive.read_bytes()).hexdigest()
        staged_checksum = staging_root / checksum_out.name
        staged_checksum.write_text(f"{digest}  {out.name}\n", encoding="utf-8", newline="\n")

        # Construction and compression may be long.  Approval applies to the
        # still-current HEAD, not merely to the revision observed at startup.
        check_expected_head(repo, expected_commit, run=run)
        check_clean(repo, run=run)
        check_archive_out(out, repo)
        published: list[tuple[Path, tuple[int, int]]] = []
        try:
            published.append((out, _publish_exclusive(staged_archive, out)))
            published.append((checksum_out, _publish_exclusive(staged_checksum, checksum_out)))
        except BaseException:
            for path, identity in reversed(published):
                _rollback_publication(path, identity)
            raise
    except OSError as exc:
        raise HandoffError(f"could not publish the handoff archive at {out}: {exc}") from exc
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
    return out, checksum_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", type=Path, help="an empty or new folder")
    output.add_argument("--archive", type=Path, help="a new one-download .tar.gz bundle")
    parser.add_argument("--wheel", type=Path, help="use this wheel instead of building one")
    parser.add_argument("--expected-commit", help="required exact commit for archive publication")
    parser.add_argument(
        "--allow-no-ui",
        action="store_true",
        help="accept a wheel without the built web UI and generate a backend-only wrapper",
    )
    parser.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.archive is not None:
            if not args.expected_commit:
                raise HandoffError("--archive requires --expected-commit")
            out, checksum = make_handoff_archive(
                args.archive,
                expected_commit=args.expected_commit,
                repo=args.repo,
                wheel=args.wheel,
                allow_no_ui=args.allow_no_ui,
            )
        else:
            assert args.out is not None
            out = make_handoff(
                args.out,
                repo=args.repo,
                wheel=args.wheel,
                allow_no_ui=args.allow_no_ui,
                expected_commit=args.expected_commit,
            )
            checksum = None
    except HandoffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        command = " ".join(str(part) for part in exc.cmd)
        detail = (exc.stderr or "").strip().splitlines()[-1:] if exc.stderr else []
        print(
            f"error: {command} exited with code {exc.returncode}"
            + (f": {detail[0]}" if detail else ""),
            file=sys.stderr,
        )
        return 1
    if args.archive is not None:
        assert checksum is not None
        print(f"handoff archive: {out.resolve()}")
        print(f"archive checksum: {checksum.resolve()}")
        print(f"size: {out.stat().st_size / 2**20:.1f} MB")
    else:
        size = folder_size(out)
        print(f"handoff folder: {out.resolve()}")
        print(f"size: {size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

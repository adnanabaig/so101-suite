"""Provision local launch prerequisites without opening hardware or changing torque."""

from __future__ import annotations

import argparse
import grp
import os
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE_ROOT = Path("/var/lib/so101/device-ownership-v2")
DEVICE_DIRS = (
    DEVICE_ROOT,
    DEVICE_ROOT / "locks",
    DEVICE_ROOT / "intent-events",
    DEVICE_ROOT / "intent-events" / "aliases",
)


def camera_entries(directory: Path = Path("/dev/v4l/by-id")) -> str:
    """Stable capture endpoints only; metadata discovery never opens a camera."""
    found: list[Path] = []
    targets: set[Path] = set()
    for path in sorted(directory.glob("*-video-index0")):
        target = path.resolve()
        if path.exists() and target not in targets:
            targets.add(target)
            found.append(path)
    return ",".join(f"camera_{index}={path}" for index, path in enumerate(found, start=1))


def create_profile(root: Path, cameras: str) -> bool:
    """Publish a private initial profile exclusively; preserve existing settings."""
    destination = root / ".env"
    if destination.is_symlink():
        raise RuntimeError("Refusing a symlinked .env; review it before launching real hardware.")
    if destination.exists():
        return False
    content = (root / "config/rigs/so101.env.example").read_text(encoding="utf-8")
    content = content.replace("/home/adnan/so101-suite", str(root))
    content = content.replace("SO101_CAMERAS=\n", f"SO101_CAMERAS={cameras}\n")
    content = content.replace(
        "# Start without cameras; discover and identify them before enabling capture.",
        "# Cameras discovered by stable path; camera_1/2 do not imply wrist/overview roles.",
    )
    with tempfile.NamedTemporaryFile(dir=root, prefix=".env-start-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                return False
        finally:
            temporary.unlink(missing_ok=True)
    return True


def namespace_ready() -> bool:
    group = grp.getgrnam("dialout").gr_gid
    ready = True
    for path in DEVICE_DIRS:
        try:
            info = path.lstat()
        except FileNotFoundError:
            ready = False
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"Refusing non-directory or symlink at {path}")
        ready &= info.st_uid == 0 and info.st_gid == group and stat.S_IMODE(info.st_mode) == 0o3770
    return ready


def provision_namespace() -> None:
    if namespace_ready():
        return
    print("Setting up shared motor-device locks; sudo may ask for your password.", flush=True)
    subprocess.run(
        ["sudo", "install", "-d", "-o", "root", "-g", "dialout", "-m", "3770"]
        + [str(path) for path in DEVICE_DIRS],
        check=True,
    )
    if not namespace_ready():
        raise RuntimeError("Shared motor-device directories did not pass ownership/mode checks.")


def port_busy() -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", 8000)) == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Inspect only; no sudo, writes or server"
    )
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        raise RuntimeError(
            "Run ./start as your normal user; only directory provisioning uses sudo."
        )
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise RuntimeError("Built UI missing; run make build-ui first.")
    if port_busy():
        raise RuntimeError("Port 8000 is occupied. Stop that server first; ./start never kills it.")
    cameras = camera_entries()
    print(f"Stable camera sources: {cameras or '(none found)'}", flush=True)
    if args.check:
        profile = (ROOT / ".env").is_file()
        namespace = namespace_ready()
        print(f"Real profile: {'present' if profile else 'will be created'}")
        print(f"Shared lock directories: {'ready' if namespace else 'need sudo setup'}")
        print("Inspection only: no hardware opened and no server started.")
        return 0 if profile and namespace else 1
    if create_profile(ROOT, cameras):
        print("Created private .env from the proposed rig profile; cameras have generic names.")
    else:
        print("Using existing .env unchanged (including its camera settings).")
    provision_namespace()
    if grp.getgrnam("dialout").gr_gid not in {os.getgid(), *os.getgroups()}:
        raise RuntimeError(
            "This session lacks dialout membership; log out/in after adding the group."
        )
    launcher = ROOT / "config/launch-real.sh"
    subprocess.run([str(launcher), "info"], check=True)
    print(
        "Starting REAL dashboard at http://localhost:8000 — no auto-reload.\n"
        "Verify USB roles and supply variants before hardware operations.\n"
        "Calibration, rest pose, preflight bus probes and motion are manual.\n"
        "The previous full software gate failed; this is a diagnostic launch, not qualification.\n"
        "Cameras are configured; live capture starts only in a camera-using operation.\n"
        "Stop this server before standalone calibration; Ctrl+C stops it.",
        flush=True,
    )
    os.execv(str(launcher), [str(launcher)])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"start: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

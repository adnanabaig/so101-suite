"""Operator-run leader calibration, installation and ten-sweep verification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.hardware.backend import current_backend

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verification", ROOT / "scripts/verify_arm_manual.py")
assert spec and spec.loader
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)
manual = verification.installer.arm_manual


def latest_session(settings):
    candidates = []
    root = settings.data_dir / "manual_arm_calibrations/leader"
    for path in root.glob("*/arm_session.json"):
        manifest = json.loads(path.read_text())
        if (
            manifest.get("arm") != "leader"
            or manifest.get("device_id") != settings.leader_id
            or manifest.get("port") != settings.leader_port
            or manifest.get("validations") != 5
            or manifest.get("joints") != list(manual.ORDER)
        ):
            continue
        completed = [
            name for name, item in manifest["results"].items() if item["state"] == "validated"
        ]
        for name in completed:
            manual.verify_completed(path.parent, manifest, name)
        candidates.append((len(completed), path.stat().st_mtime_ns, path.parent))
    return max(candidates)[2] if candidates else None


def fetch(path):
    with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=3) as response:
        return json.load(response)


def stop_idle_motor_server():
    try:
        status = fetch("/api/robot/status")
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return
        raise
    control = fetch("/api/control/state")
    if (
        status.get("backend") != "real"
        or status.get("mode") != "idle"
        or status.get("torque_state") != "off"
        or status.get("torque_unconfirmed")
        or status.get("stop_state") != "none"
        or control.get("armed")
        or control.get("safety_state") != "disarmed"
    ):
        raise RuntimeError("Motor server must be idle, fault-free and confirmed torque OFF first")
    result = subprocess.run(
        ["systemctl", "--user", "show", "so101-motor.service", "-p", "MainPID", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    process = Path("/proc") / result.stdout.strip() / "cmdline"
    if not process.exists() or b"/so101\x00serve" not in process.read_bytes():
        raise RuntimeError("Stop the externally managed motor server in its own terminal first")
    subprocess.run(["systemctl", "--user", "stop", "so101-motor.service"], check=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            fetch("/api/robot/status")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                print("Motor server stopped; camera and MCAP services remain running.")
                return
            raise
        time.sleep(0.2)
    raise RuntimeError("Motor server did not stop; calibration refused")


def run_child(command, *, check):
    child = subprocess.Popen(command)
    try:
        status = child.wait()
    except KeyboardInterrupt:
        child.wait(timeout=30)
        raise
    except BaseException:
        child.terminate()
        child.wait(timeout=30)
        raise
    if check and status:
        raise subprocess.CalledProcessError(status, command)


def run(args, settings, *, execute=run_child, stop_server=stop_idle_motor_server):
    if current_backend(settings) != "real" or settings.health_leader_supply != "12v":
        raise RuntimeError("This rig requires its confirmed real, 12 V leader profile")
    if settings.leader_port == settings.follower_port or settings.leader_id == settings.follower_id:
        raise RuntimeError("Leader and follower must have distinct ports and identities")
    folder = args.resume or (None if args.new else latest_session(settings))
    resume = folder is not None
    if folder is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = (
            settings.data_dir / "manual_arm_calibrations/leader" / f"{stamp}-{uuid.uuid4().hex[:8]}"
        )
    print(
        f"LEADER: {settings.leader_id}\nPort: {settings.leader_port}\nSession: {folder}", flush=True
    )
    print("Flow: six accepted sweeps per joint → install → four additional verification sweeps.")
    print(
        "Existing follower calibration is preserved. No powered motion or automatic fault resets."
    )
    command = [str(ROOT / "calibrate-arm-manual"), "--arm", "leader"]
    command += ["--resume" if resume else "--output", str(folder)]
    if args.dry_run:
        execute([*command, "--dry-run"], check=True)
        print(
            "DRY RUN: no server stop, bus access or writes. "
            "Installation and verification follow capture."
        )
        return folder
    stop_server()
    follower_paths = [
        settings.data_dir / "calibration_follower.json",
        verification.lerobot_calibration_file("follower", settings.follower_id),
    ]
    follower_hashes = {path: verification.digest(path) for path in follower_paths}
    execute(command, check=True)
    try:
        verification.load_reference(folder, settings, "leader")
        print("REUSE: leader installation already matches the validated recipes.")
    except (OSError, ValueError, KeyError, RuntimeError):
        execute(
            [str(ROOT / "install-arm-calibration"), "--arm", "leader", "--session", str(folder)],
            check=True,
        )
    execute(
        [str(ROOT / "verify-arm-manual"), "--arm", "leader", "--source", str(folder)], check=True
    )
    reference = verification.load_reference(folder, settings, "leader")
    evidence = verification.find_resume(settings, "leader", folder, list(manual.ORDER))
    report = json.loads((evidence / "verification.json").read_text())
    if report["state"] != "ten_sweeps_verified" or report["reference"] != reference:
        raise RuntimeError("Ten-sweep verification is incomplete or belongs to a changed reference")
    if not all(report["comparison"][name]["successful_sweeps"] == 10 for name in JOINT_NAMES):
        raise RuntimeError("Ten successful sweeps required for every leader joint")
    for path, expected in follower_hashes.items():
        if verification.digest(path) != expected:
            raise RuntimeError("Follower calibration changed; stop and investigate")
    summary = {
        "arm": "leader",
        "device_id": settings.leader_id,
        "state": "installed_and_verified",
        "session": str(folder),
        "verification": str(evidence),
        "reference_hashes": reference["source_hashes"],
        "follower_calibration_unchanged": True,
        "powered_teleop_verified": False,
    }
    verification.installer.atomic_write_json(folder / "leader_bringup.json", summary)
    print(f"LEADER INSTALLED AND VERIFIED: ten accepted sweeps per joint. Reference: {folder}")
    print(
        "Motor server stays stopped. Run services up after reviewing results; no motion is enabled."
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--resume", type=Path)
    selection.add_argument("--new", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("Use an interactive terminal for manual calibration")

    def terminate(signum, frame):
        raise RuntimeError(f"Signal {signum}: stop and let the active worker restore registers")

    handlers = {
        signum: signal.signal(signum, terminate) for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        run(args, get_settings())
    except (Exception, KeyboardInterrupt) as exc:
        print(
            f"Stopped: {exc}\nRun calibrate-leader again to reuse completed progress.",
            file=sys.stderr,
        )
        return 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

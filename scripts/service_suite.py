"""Start and inspect the local workstation services without commanding motion."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = (
    "so101-motor.service",
    "so101-arm-camera.service",
    "so101-camera-1.service",
    "so101-camera-2.service",
    "so101-realsense.service",
    "so101-foxglove.service",
    "so101-recording.service",
)


def require_idle_restart():
    """Refuse to interrupt an operation or restart an unobservable live server."""
    try:
        motor = fetch("http://127.0.0.1:8000/api/robot/status")
    except urllib.error.URLError as exc:
        if not isinstance(exc.reason, ConnectionRefusedError):
            raise RuntimeError("Motor API is unresponsive; refusing to interrupt it") from exc
        state = subprocess.run(
            ["systemctl", "--user", "is-active", "so101-motor.service"],
            capture_output=True,
            text=True,
            start_new_session=True,
        ).stdout.strip()
        if state not in {"inactive", "failed", "unknown"}:
            raise RuntimeError(
                "Motor service is not observable; refusing to interrupt it"
            ) from None
        recorder_state = subprocess.run(
            ["systemctl", "--user", "is-active", "so101-recording.service"],
            capture_output=True,
            text=True,
            start_new_session=True,
        ).stdout.strip()
        if recorder_state not in {"inactive", "failed", "unknown"}:
            raise RuntimeError(
                "Recorder is running without the motor API; inspect it before refresh"
            ) from None
        return
    control = fetch("http://127.0.0.1:8000/api/control/state")
    capture = fetch("http://127.0.0.1:8000/api/mcap/status")
    if (
        motor.get("torque_state") != "off"
        or motor.get("torque_unconfirmed", True)
        or motor.get("mode") != "idle"
        or motor.get("stop_state") != "none"
        or motor.get("operation_owner") is not None
        or control.get("armed") is not False
        or control.get("lease") is not None
        or capture.get("active") is not None
        or capture.get("recorder", {}).get("active") is not False
        or capture.get("recorder", {}).get("recording") is not False
    ):
        raise RuntimeError(
            "Refresh refused: finish recording/calibration and the operator stop first; "
            "the arm must be idle, disarmed and confirmed torque OFF"
        )


def fetch(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def status():
    result = {"services": {}}
    for unit in UNITS:
        state = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            start_new_session=True,
        ).stdout.strip()
        result["services"][unit] = {"state": state}
    for name, url in (
        ("motor", "http://127.0.0.1:8000/api/robot/status"),
        ("control", "http://127.0.0.1:8000/api/control/state"),
        ("health", "http://127.0.0.1:8000/api/robot/health"),
        ("obsbot", "http://127.0.0.1:8765/status.json"),
        ("camera_1", "http://127.0.0.1:8768/status.json"),
        ("camera_2", "http://127.0.0.1:8769/status.json"),
        ("realsense", "http://127.0.0.1:8770/status.json"),
    ):
        try:
            result[name] = fetch(url)
        except Exception as exc:
            result[name] = {"online": False, "error": str(exc)}
    try:
        recording = json.loads((ROOT / "data/real/mcap/status.json").read_text())
        recording["fresh"] = time.time() - recording["updated_at"] < 5
        recording["healthy"] = recording["fresh"] and (
            (recording.get("ready") and time.time() - recording.get("api_received_at", 0) < 5)
            or (
                recording["active"]
                and recording.get("bytes", 0) > 0
                and time.time() - recording.get("telemetry_received_at", 0) < 5
            )
        )
        result["recording"] = recording
    except (OSError, ValueError, KeyError) as exc:
        result["recording"] = {"healthy": False, "error": str(exc)}
    result["upload"] = {"enabled": False, "state": "awaiting destination and recording selection"}
    result["all_local_online"] = (
        all(item["state"] == "active" for item in result["services"].values())
        and result["motor"].get("backend") == "real"
        and all(
            result[name].get("live") for name in ("obsbot", "camera_1", "camera_2", "realsense")
        )
        and result["recording"]["healthy"]
    )
    result["motion_state"] = {
        "armed": result["control"].get("armed", False),
        "safety_state": result["control"].get("safety_state", "unavailable"),
        "torque_state": result["motor"].get("torque_state", "unconfirmed"),
        "note": "Service availability is not calibration or motion qualification",
    }
    return result


def start_services():
    destination = Path.home() / ".config/systemd/user"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*UNITS, "so101-suite.target"):
        shutil.copyfile(ROOT / "config/services" / name, destination / name)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, start_new_session=True)
    subprocess.run(
        ["systemctl", "--user", "start", *UNITS, "so101-suite.target"],
        check=True,
        start_new_session=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("up", "restart", "status", "down"), nargs="?", default="up"
    )
    parser.add_argument("--json", action="store_true", help="Print complete status as JSON")
    parser.add_argument(
        "--open", action="store_true", help="Open a fresh data collection dashboard"
    )
    args = parser.parse_args()
    if args.action == "restart":
        require_idle_restart()
        subprocess.run(["systemctl", "--user", "stop", "so101-suite.target", *UNITS], check=True)
    if args.action in {"up", "restart"}:
        start_services()
        deadline = time.monotonic() + 30
        while not status()["all_local_online"] and time.monotonic() < deadline:
            time.sleep(0.25)
    elif args.action == "down":
        motor = fetch("http://127.0.0.1:8000/api/robot/status")
        control = fetch("http://127.0.0.1:8000/api/control/state")
        if motor["torque_state"] != "off" or control["armed"] or motor["stop_state"] != "none":
            raise RuntimeError(
                "Finish the operator stop and confirm torque OFF before stopping services"
            )
        subprocess.run(["systemctl", "--user", "stop", "so101-suite.target", *UNITS], check=True)
    report = status()
    if args.open and report["all_local_online"]:
        webbrowser.open(f"http://127.0.0.1:8000/teleop?refresh={time.time_ns()}")
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for name, item in report["services"].items():
        print(f"{name.removeprefix('so101-').removesuffix('.service'):14} {item['state']}")
    for name in ("obsbot", "camera_1", "camera_2", "realsense"):
        camera = report[name]
        measured = camera.get("actual_preview_fps")
        rate = f"{measured:.1f} measured fps" if measured is not None else "fps not reported"
        print(f"{name:14} {'live' if camera.get('live') else 'unavailable'} · {rate}")
    print(
        f"Motor safety: {report['motion_state']['safety_state']} · "
        f"torque {report['motion_state']['torque_state']}"
    )
    recording = report["recording"]
    capture_state = (
        ("recording" if recording.get("active") else "ready · waiting for activity")
        if recording["healthy"]
        else "NOT HEALTHY"
    )
    capture_location = recording.get("path") or recording.get("storage_path")
    print(f"MCAP: {capture_state} · {capture_location or recording.get('error', 'unavailable')}")
    print("Upload: disabled · awaiting destination and selected recordings")
    print(
        f"All local services online: {report['all_local_online']} (does not mean ready for motion)"
    )
    if args.action in {"up", "restart"} and not report["all_local_online"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

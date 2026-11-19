"""Read-only shoulder reference capture; no motor or calibration writes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path

from so101 import JOINT_NAMES
from so101.atomic_io import atomic_write_json
from so101.bringup.buses import open_feetech_bus
from so101.bringup.standalone import standalone_access
from so101.config import get_settings
from so101.hardware.backend import current_backend
from so101.hardware.lerobot_paths import calibration_digest, lerobot_calibration_file
from so101.simulation.mjcf import so101_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture", ROOT / "scripts/record_wrist_roll.py")
assert spec and spec.loader
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
JOINT = "shoulder_lift"


def model_angle(joint_degrees: float) -> float:
    """Shoulder-pivot to elbow-pivot tilt from vertical, in model sagittal plane."""
    model = so101_model()
    joints = dict.fromkeys(JOINT_NAMES, 0.0)
    joints[JOINT] = math.radians(joint_degrees)
    poses = model.body_poses(joints)
    vector = poses["lower_arm"].pos - poses["upper_arm"].pos
    return math.degrees(math.atan2(float(vector[0]), float(vector[2])))


def measure(bus, calibration, *, sleep=time.sleep):
    before = capture.registers(bus, JOINT)
    if before["Operating_Mode"] != 0:
        raise RuntimeError("Shoulder must already be in position mode")
    for register, field in (
        ("Homing_Offset", "homing_offset"),
        ("Min_Position_Limit", "range_min"),
        ("Max_Position_Limit", "range_max"),
    ):
        if before[register] != calibration[field]:
            raise RuntimeError(f"{register}: motor differs from installed calibration")
    samples = []
    for _ in range(30):
        capture.torque_off(bus)
        value = bus.read("Present_Position", JOINT, normalize=False, num_retry=1)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError("Invalid shoulder encoder reading")
        if not calibration["range_min"] <= value <= calibration["range_max"]:
            raise RuntimeError("Shoulder reading outside measured calibration range")
        samples.append({"reported_ticks": value, "raw_ticks": value + before["Homing_Offset"]})
        sleep(0.05)
    if capture.registers(bus, JOINT) != before:
        raise RuntimeError("Shoulder calibration changed during capture")
    capture.torque_off(bus)
    values = [sample["reported_ticks"] for sample in samples]
    spread = (max(values) - min(values)) * 360 / 4095
    if spread >= 1.0:
        raise RuntimeError(f"Reference pose moved by {spread:.2f} degrees; support it stably")
    measured = (
        (statistics.mean(values) - (calibration["range_min"] + calibration["range_max"]) / 2)
        * 360
        / 4095
    )
    return {"suite_degrees": measured, "spread_degrees": spread, "samples": samples}


def collect(args, settings, *, ask=input, opener=open_feetech_bus, measure_fn=measure):
    if current_backend(settings) != "real":
        raise RuntimeError("Real profile required")
    zero_tilt = model_angle(0)
    print(
        f"Model shoulder-to-elbow pivot line at joint zero: {zero_tilt:.3f}° forward of vertical."
    )
    print(f"A vertical pivot line corresponds to model shoulder_lift {-zero_tilt:.3f}°.")
    if args.dry_run:
        print("DRY RUN: model inspected; no locks, serial access or file writes.")
        return None
    path = lerobot_calibration_file("follower", settings.follower_id)
    installed = json.loads(path.read_text())
    digest = calibration_digest("follower", settings.follower_id)
    print(
        "Stop the dashboard. All six motors must ALREADY be torque-off.\n"
        "Clamp the base level, support every link over a padded catch, clear pinch points\n"
        "and keep the physical cutoff reachable. Software E-stop is unavailable here.\n"
        "This tool ONLY READS registers; it never changes torque, offsets or limits.\n"
        "Measure the SIDE-VIEW line between SHOULDER and ELBOW pivot centers,\n"
        "not a plastic edge. First: elbow pivot directly ABOVE the shoulder pivot.\n"
        "Second: tilt that line gently 15–30° away from vertical and measure the angle.\n"
        "Keep shoulder pan fixed between captures. Save a side-view photo of both poses\n"
        "for frame/direction review; the script does not assume forward/back direction.\n"
        "Use a plumb line/protractor or an angle gauge; do not guess from encoder counts.\n"
        "We capture two supported poses, about 15–30° apart, within gentle free travel."
    )
    if ask("Type READY to confirm setup and measurement method: ").strip().casefold() != "ready":
        raise RuntimeError("Setup not confirmed; no bus opened")
    evidence = settings.data_dir / "shoulder_reference_checks" / uuid.uuid4().hex[:12]
    report = {
        "backend": "real",
        "follower_id": settings.follower_id,
        "calibration_digest": digest,
        "model_zero_tilt_degrees": zero_tilt,
        "state": "recording",
        "poses": [],
        "calibration_applied": False,
    }
    with standalone_access("read-only shoulder frame reference", settings=settings):
        evidence.mkdir(parents=True, mode=0o700)
        bus = None
        try:
            bus = opener(settings.follower_port, baudrate=settings.baudrate)
            capture.torque_off(bus)
            for index in range(2):
                print(
                    "Pose 1/2: shoulder-to-elbow pivot line VERTICAL, elbow above shoulder."
                    if index == 0
                    else "Pose 2/2: tilt 15–30° gently; measure the pivot-line angle."
                )
                angle = 0.0
                if index:
                    while True:
                        text = ask(
                            "Measured tilt magnitude from vertical (15..45°, q to abort): "
                        ).strip()
                        if text.casefold() == "q":
                            raise RuntimeError("Operator cancelled")
                        try:
                            angle = float(text)
                        except ValueError:
                            print(
                                "Enter the measured angle as a number, such as 20, or q to stop. "
                                "Pose 1 is retained; blank Enter does not capture Pose 2."
                            )
                            continue
                        if math.isfinite(angle) and 15 <= angle <= 45:
                            break
                        print("Use a measured tilt magnitude within 15..45 degrees, or q to stop.")
                if ask(
                    "Hands clear and supported: press Enter to CAPTURE, q then Enter to abort: "
                ).strip():
                    raise RuntimeError("Operator cancelled")
                result = measure_fn(bus, installed[JOINT])
                result.update(
                    measured_tilt_magnitude_degrees=angle,
                    expected_model_alternatives_degrees=[angle - zero_tilt, -angle - zero_tilt],
                )
                report["poses"].append(result)
                atomic_write_json(evidence / "reference.json", report)
                print(
                    f"Captured: encoder {result['suite_degrees']:.2f}°, measured tilt {angle:.2f}°."
                )
            if calibration_digest("follower", settings.follower_id) != digest:
                raise RuntimeError("Native calibration file changed during capture")
            first, second = report["poses"]
            physical_delta = second["measured_tilt_magnitude_degrees"]
            encoder_delta = second["suite_degrees"] - first["suite_degrees"]
            report.update(
                state="captured",
                physical_delta_magnitude_degrees=physical_delta,
                encoder_delta_degrees=encoder_delta,
                encoder_to_measured_angle_ratio=abs(encoder_delta) / physical_delta,
                vertical_reference_model_minus_suite_degrees=-zero_tilt - first["suite_degrees"],
                sufficient_separation=abs(physical_delta) >= 15,
                note=(
                    "Operator angle accuracy and physical direction require review; "
                    "no automatic correction"
                ),
            )
        except BaseException as exc:
            report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                if bus is not None:
                    bus.disconnect(disable_torque=False)
            finally:
                atomic_write_json(evidence / "reference.json", report)
                print(f"Evidence: {evidence}")
    print(
        "Reference captured. Calibration and limits unchanged. Send the evidence path for review."
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("Use an interactive terminal")
    try:
        collect(args, get_settings())
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

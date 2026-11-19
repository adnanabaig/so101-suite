"""Endpoint-first wrist-roll calibration with manual validation and register restoration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from so101 import JOINT_NAMES, __version__
from so101.atomic_io import atomic_write_json
from so101.bringup.standalone import standalone_access
from so101.config import get_settings
from so101.hardware.backend import current_backend
from so101.hardware.calibration import CalibrationResult, JointCalibration
from so101.hardware.motors import FeetechBus
from so101.services.store import persist_standalone_fault

ROOT = Path(__file__).resolve().parents[1]
JOINT = "wrist_roll"
DEFAULT_BASELINE = "20261018T222815Z-8408fed4"
spec = importlib.util.spec_from_file_location(
    "wrist_capture", ROOT / "scripts/record_wrist_roll.py"
)
assert spec is not None and spec.loader is not None
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def load_baseline(path: Path, settings: Any, arm: str, *, joint: str = JOINT) -> dict[str, Any]:
    """Use original traces, not an assumed offset or the prior report's verdict."""
    session = json.loads((path / "session.json").read_text())
    if (
        session["backend"] != "real"
        or session["synthetic"]
        or session["state"] != "complete"
        or session["completed_trials"] < 10
        or session["joint"] != joint
        or session["expected_motor_id"] != JOINT_NAMES.index(joint) + 1
        or session["arm"] != arm
        or session["device_id"] != getattr(settings, f"{arm}_id")
        or session["port"] != getattr(settings, f"{arm}_port")
    ):
        raise ValueError(
            "Baseline must contain ten completed real trials for this exact arm/profile."
        )
    lows, highs, hashes = [], [], {}
    for index in range(1, session["requested_trials"] + 1):
        name = f"trial_{index:02d}"
        summary_path, trace_path = path / f"{name}.json", path / f"{name}.jsonl"
        summary = json.loads(summary_path.read_text())
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        samples = [event for event in events if event["type"] == "sample"]
        if not summary["usable_for_comparison"] or len(samples) != summary["sample_count"]:
            raise ValueError(f"{name}: baseline trace is incomplete or flagged")
        if summary["registers_before"] != summary["registers_after"]:
            raise ValueError(f"{name}: baseline registers changed")
        if (
            not samples
            or events[-1]["type"] != "trial_end"
            or any(
                sample["raw_ticks"] != sample["reported_ticks"] + sample["homing_offset"]
                or sample["homing_offset"] != summary["registers_before"]["Homing_Offset"]
                for sample in samples
            )
            or any(
                second["elapsed_seconds"] <= first["elapsed_seconds"]
                for first, second in zip(samples, samples[1:], strict=False)
            )
            or not all(
                len(event["torque"]) == 6 and all(value == 0 for value in event["torque"].values())
                for event in events
                if "torque" in event
            )
        ):
            raise ValueError(f"{name}: baseline trace integrity failed")
        raw = [sample["raw_ticks"] for sample in samples]
        if (min(raw), max(raw)) != (summary["raw_min"], summary["raw_max"]):
            raise ValueError(f"{name}: baseline extrema mismatch")
        lows.append(min(raw))
        highs.append(max(raw))
        hashes[trace_path.name] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        hashes[summary_path.name] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    hashes["session.json"] = hashlib.sha256((path / "session.json").read_bytes()).hexdigest()
    result = {
        "path": str(path.resolve()),
        "session_id": session["session_id"],
        "raw_min": min(lows),
        "raw_max": max(highs),
        "source_sha256": hashes,
    }
    proposed = capture.candidate(result["raw_min"], result["raw_max"])
    if not proposed["offset_representable"] or not proposed["range_representable"]:
        raise ValueError("Baseline envelope cannot fit one encoder interval with a valid offset")
    return result


def snapshot(native, *, joint: str = JOINT) -> dict[str, int]:
    return {
        **capture.registers(native, joint),
        "Lock": int(native.read("Lock", joint, normalize=False)),
    }


def position_at_midpoint(
    native,
    target: float,
    *,
    stop_key=capture.keyboard_stop,
    clock=time.monotonic,
    sleep=time.sleep,
    timeout: float = 180,
    joint: str = JOINT,
) -> dict[str, Any]:
    """Operator positioning only; never issues a motor command."""
    print(
        f"Move {joint} manually toward raw midpoint {target:.1f}. "
        "Watch the live counts, then hold still and press Enter.\n"
        "Within 64 ticks is accepted for positioning; q then Enter aborts.",
        flush=True,
    )
    start = clock()
    last_print = -1.0
    readings: list[int] = []
    offset = int(native.read("Homing_Offset", joint, normalize=False))
    while clock() - start < timeout:
        capture.torque_off(native)
        reported = int(native.sync_read("Present_Position", [joint], normalize=False)[joint])
        raw = reported + offset
        readings = (readings + [raw])[-3:]
        if clock() - last_print >= 0.25:
            print(
                f"  midpoint guide: raw {raw} · target {target:.1f} · delta {raw - target:+.1f}",
                flush=True,
            )
            last_print = clock()
        key = stop_key()
        if key == "abort":
            raise RuntimeError("Midpoint positioning cancelled; no offset written")
        if key == "complete":
            if (
                abs(raw - target) <= 64
                and len(readings) == 3
                and max(readings) - min(readings) <= 10
            ):
                print("Midpoint accepted. Hold still during the temporary offset test.", flush=True)
                return {
                    "target_raw": target,
                    "raw": raw,
                    "reported": reported,
                    "existing_offset": offset,
                    "recent_raw": readings,
                }
            print(
                "Not yet centered and settled. Move gently, hold still, then press Enter again.",
                flush=True,
            )
        sleep(0.02)
    raise RuntimeError("Midpoint positioning timed out; no offset written")


def write_offset(
    native,
    offset: int,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    record=None,
    joint: str = JOINT,
) -> dict[str, Any]:
    """Require stable direct AND sync feedback within a bounded observation window."""
    probe: dict[str, Any] = {
        "requested": offset,
        "observations": [],
        "status": "preparing",
        "feedback_timeout_seconds": 1.0,
        "required_consecutive_matches": 3,
    }

    def save() -> None:
        if record:
            record(probe)

    save()
    try:
        capture.torque_off(native)
        old_offset = int(native.read("Homing_Offset", joint, normalize=False))
        before = int(native.sync_read("Present_Position", [joint], normalize=False)[joint])
        before_direct = int(native.read("Present_Position", joint, normalize=False))
        probe.update(
            before=before,
            before_direct=before_direct,
            previous_offset=old_offset,
            expected_after=before + old_offset - offset,
            status="write_pending",
        )
        save()
        if abs(before_direct - before) > 10:
            raise RuntimeError(
                "Direct/sync positions differ before write; hold still and investigate"
            )
        native.write("Lock", joint, 0, normalize=False)
        native.write("Homing_Offset", joint, offset, normalize=False)
        probe["read_back"] = int(native.read("Homing_Offset", joint, normalize=False))
        probe["status"] = "observing_feedback"
        save()
        if probe["read_back"] != offset:
            raise RuntimeError(
                f"{joint} homing read-back mismatch: requested {offset}, got {probe['read_back']}"
            )
        started = clock()
        consecutive = 0
        while clock() - started <= 1.0:
            capture.torque_off(native)
            actual_offset = int(native.read("Homing_Offset", joint, normalize=False))
            direct = int(native.read("Present_Position", joint, normalize=False))
            synced = int(native.sync_read("Present_Position", [joint], normalize=False)[joint])
            probe["observations"].append(
                {
                    "elapsed_seconds": clock() - started,
                    "homing_offset": actual_offset,
                    "direct_position": direct,
                    "sync_position": synced,
                }
            )
            probe["after"] = synced
            save()
            if actual_offset != offset:
                raise RuntimeError("Temporary homing offset changed during feedback observation")
            matches = (
                abs(direct - probe["expected_after"]) <= 10
                and abs(synced - probe["expected_after"]) <= 10
            )
            consecutive = consecutive + 1 if matches else 0
            if consecutive == 3:
                probe["status"] = "verified"
                save()
                return probe
            sleep(0.02)
        raise RuntimeError(
            f"{joint} feedback did not converge: offset read-back {probe['read_back']}, "
            f"expected position {probe['expected_after']}, final direct {direct}, sync {synced}. "
            "The register write alone is insufficient; cause remains unconfirmed. Probe saved."
        )
    except BaseException as exc:
        probe.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def restore(arm_bus, native, previous: dict[str, int], *, joint: str = JOINT) -> None:
    arm_bus.set_torque(False)
    native.write("Lock", joint, 0, normalize=False)
    # Restore only changed fields, keeping Homing_Offset after any mode change.
    for register in ("Operating_Mode", "Min_Position_Limit", "Max_Position_Limit", "Homing_Offset"):
        if int(native.read(register, joint, normalize=False)) != previous[register]:
            native.write(register, joint, previous[register], normalize=False)
    native.write("Lock", joint, previous["Lock"], normalize=False)
    actual = snapshot(native, joint=joint)
    if actual != previous:
        raise RuntimeError(f"{joint} restore read-back mismatch: expected {previous}, got {actual}")
    capture.torque_off(native)


def check_trial(
    folder: Path,
    trial: dict[str, Any],
    reference: dict[str, Any],
    *,
    offset: int | None,
    dwell_seconds: float,
) -> dict[str, Any]:
    samples = [
        event
        for event in map(json.loads, (folder / trial["samples_file"]).read_text().splitlines())
        if event["type"] == "sample"
    ]
    if not trial["usable_for_comparison"]:
        raise RuntimeError(
            f"Trial {trial['trial']} is {trial['status']}: {trial['quality_warnings']}"
        )
    checks = capture.settled_endpoints(samples, trial["raw_min"], trial["raw_max"])
    checks.update(
        covers_reference_low=(
            checks["low_settled_ticks"] is not None
            and trial["raw_min"] <= reference["raw_min"] + 64
            and checks["low_settled_ticks"] <= reference["raw_min"] + 64
        ),
        covers_reference_high=(
            checks["high_settled_ticks"] is not None
            and trial["raw_max"] >= reference["raw_max"] - 64
            and checks["high_settled_ticks"] >= reference["raw_max"] - 64
        ),
    )
    if (
        checks["low_dwell_seconds"] < dwell_seconds
        or checks["high_dwell_seconds"] < dwell_seconds
        or not checks["covers_reference_low"]
        or not checks["covers_reference_high"]
    ):
        raise RuntimeError(
            f"Trial {trial['trial']} needs settled endpoint coverage: {checks}. "
            "Use gentle pauses of about one second, without extra force."
        )
    if offset is not None:
        if trial["registers_before"]["Homing_Offset"] != offset:
            raise RuntimeError("Temporary offset changed before validation")
        if not 0 <= trial["reported_min"] < trial["reported_max"] <= 4095:
            raise RuntimeError(
                f"Validation exceeds 0..4095: {trial['reported_min']}..{trial['reported_max']}; "
                "candidate rejected, samples retained"
            )
    return checks


def workflow(
    args,
    settings,
    *,
    ask=input,
    bus_factory=FeetechBus,
    capture_fn=capture.capture_trial,
    position_fn=position_at_midpoint,
) -> Path:
    if current_backend(settings) != "real":
        raise RuntimeError("This workflow requires an explicitly configured real backend.")
    joint = getattr(args, "joint", JOINT)
    if joint not in JOINT_NAMES:
        raise ValueError("unknown calibration joint")
    motor_id = JOINT_NAMES.index(joint) + 1
    joint_kwargs = {"joint": joint} if joint != JOINT else {}
    baseline_path = args.baseline
    if baseline_path is None and not getattr(args, "fresh_reference", False):
        baseline_path = settings.data_dir / "manual_joint_baselines" / joint / DEFAULT_BASELINE
    baseline = (
        load_baseline(baseline_path, settings, args.arm, joint=joint) if baseline_path else None
    )
    print(f"Manual joint calibration: {joint} · motor ID {motor_id}")
    if baseline:
        preview = capture.candidate(baseline["raw_min"], baseline["raw_max"])
        print(
            f"Baseline: {baseline['session_id']} · "
            f"raw {baseline['raw_min']}..{baseline['raw_max']}\n"
            f"Initial calculated offset: {preview['homing_offset']} "
            "(recomputed after fresh capture)."
        )
    else:
        print("A fresh endpoint capture will establish this joint's reference.")
    print(
        f"Flow: fresh endpoints → temporary offset → {args.validations} manual validation sweeps\n"
        "→ original registers restored/read back → separate validated calibration recipe saved.\n"
        "No goal commands or torque-on. Full-arm and native LeRobot files stay unchanged."
    )
    if args.dry_run:
        print("DRY RUN: files read only; no bus lock, serial access or motor writes.")
        return baseline_path or settings.data_dir
    print(
        "\nStop the dashboard. Clamp/support the whole arm over its padded catch, verify\n"
        "cable/motor mapping, clear pinch points and keep the physical cutoff reachable.\n"
        f"The arm will be torque-off. Move only {joint}; keep other links supported.\n"
        "Software E-stop is unavailable here. Pause gently at each safe end; never force."
    )
    if ask("Type READY to confirm setup: ").strip().casefold() != "ready":
        raise RuntimeError("Setup not confirmed; no motor bus opened.")
    with standalone_access(f"endpoint-first {joint} calibration", settings=settings) as access:
        folder = args.output or (
            settings.data_dir
            / "manual_joint_calibrations"
            / joint
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        )
        folder.mkdir(parents=True, mode=0o700, exist_ok=False)
        session: dict[str, Any] = {
            "schema_version": 1,
            "method": "endpoint-first manual",
            "backend": "real",
            "suite_version": __version__,
            "started_at": capture.utc_now(),
            "arm": args.arm,
            "joint": joint,
            "device_id": getattr(settings, f"{args.arm}_id"),
            "port": getattr(settings, f"{args.arm}_port"),
            "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline": baseline,
            "state": "preparing",
            "trials": [],
            "validation_sweeps": args.validations,
            "minimum_endpoint_dwell_seconds": 0.5,
            "endpoint_band_ticks": 10,
            "endpoint_check_method": "stable-window-v2",
            "endpoint_proximity_ticks": 64,
            "maximum_endpoint_sample_gap_seconds": 0.1,
            "reference_coverage_tolerance_ticks": 64,
            "calibration_applied": False,
            "original_registers_restored": False,
        }
        atomic_write_json(folder / "session.json", session)
        arm_bus = bus_factory(getattr(settings, f"{args.arm}_port"), settings.baudrate, args.arm)
        native, previous, failure = None, None, None
        try:
            arm_bus.connect()
            native = arm_bus._bus
            if native.motors[joint].id != motor_id:
                raise RuntimeError(f"Configured {joint} motor ID is not {motor_id}")
            previous = snapshot(native, joint=joint)
            session["registers_before"] = previous
            if previous["Operating_Mode"] != 0:
                raise RuntimeError("Position mode required; no mode conversion will be attempted")
            if joint == JOINT and (
                previous["Min_Position_Limit"],
                previous["Max_Position_Limit"],
            ) != (0, 4095):
                raise RuntimeError("Expected wrist limits 0..4095; review existing setup first")

            if not 0 <= previous["Min_Position_Limit"] < previous["Max_Position_Limit"] <= 4095:
                raise RuntimeError("Invalid existing joint limits; inspect before calibrating")

            def trial(number: int, label: str, reference, offset=None):
                if ask(f"{label}: Enter to START, or q then Enter to abort: ").strip():
                    raise RuntimeError("Capture cancelled by operator")

                def ready():
                    print(
                        "RECORDING: visit both safe endpoints; pause gently about 1 s at each.\n"
                        "Watch LOW/HIGH READY for settled pauses near observed endpoints.\n"
                        "Enter to FINISH; q then Enter or Ctrl+C aborts.",
                        flush=True,
                    )

                result = capture_fn(
                    native,
                    folder,
                    number,
                    capture.keyboard_stop,
                    max_seconds=args.max_seconds,
                    ready=ready,
                    **joint_kwargs,
                )
                session["trials"].append(result)
                atomic_write_json(folder / "session.json", session)
                checks = check_trial(
                    folder,
                    result,
                    reference or {"raw_min": result["raw_min"], "raw_max": result["raw_max"]},
                    offset=offset,
                    dwell_seconds=0.5,
                )
                session.setdefault("checks", []).append(checks)
                atomic_write_json(folder / "session.json", session)
                print(f"PASS: raw {result['raw_min']}..{result['raw_max']}; {checks}")
                return result

            first = trial(1, "Fresh endpoint capture (existing offset)", baseline)
            low = min(baseline["raw_min"], first["raw_min"]) if baseline else first["raw_min"]
            high = max(baseline["raw_max"], first["raw_max"]) if baseline else first["raw_max"]
            proposed = capture.candidate(low, high)
            if not proposed["offset_representable"] or not proposed["range_representable"]:
                raise RuntimeError(
                    "Observed envelope cannot fit a valid offset; candidate rejected"
                )
            session["candidate"] = proposed
            session["state"] = "candidate_review"
            atomic_write_json(folder / "session.json", session)
            print(
                f"\nCandidate offset {proposed['homing_offset']}; "
                f"observed counts become {proposed['range_min']}..{proposed['range_max']}.\n"
                "Next: live midpoint positioning, then a temporary Homing_Offset test."
            )
            if (
                ask("Type TEST to position at the measured midpoint, then test the offset: ")
                .strip()
                .casefold()
                != "test"
            ):
                raise RuntimeError("Temporary offset not confirmed")
            session["state"] = "positioning"
            atomic_write_json(folder / "session.json", session)
            session["midpoint_positioning"] = position_fn(native, (low + high) / 2, **joint_kwargs)
            session["state"] = "temporary_write_pending"
            atomic_write_json(folder / "session.json", session)  # recovery evidence before write

            def record_probe(probe):
                session["homing_probe"] = probe
                atomic_write_json(folder / "homing_probe.json", probe)

            session["homing_readback"] = write_offset(
                native,
                proposed["homing_offset"],
                record=record_probe,
                joint=joint,
            )
            session["state"] = "validating"
            atomic_write_json(folder / "session.json", session)
            reference = {"raw_min": low, "raw_max": high}
            for index in range(args.validations):
                trial(
                    index + 2,
                    f"Validation {index + 1}/{args.validations}",
                    reference,
                    offset=proposed["homing_offset"],
                )
            session["state"] = "validated_pending_restore"
        except BaseException as exc:
            failure = exc
            session.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if native is not None and previous is not None:
                try:
                    restore(arm_bus, native, previous, joint=joint)
                    session["original_registers_restored"] = True
                except BaseException as exc:
                    session.update(state="failed", restore_error=str(exc))
                    failure = failure or exc
                    # A register rollback failure must block subsequent local motion.
                    try:
                        persist_standalone_fault(
                            settings,
                            RuntimeError(f"{joint} calibration restore failed: {exc}"),
                            device_name=args.arm,
                            torque_unconfirmed=False,
                            energized=False,
                        )
                    except Exception as persist_error:
                        session["fault_persistence_error"] = str(persist_error)
            try:
                access.disconnect_device(arm_bus, device_name=args.arm)
            except BaseException as exc:
                session.update(state="failed", cleanup_error=str(exc))
                failure = failure or exc
            session["finished_at"] = capture.utc_now()
            atomic_write_json(folder / "session.json", session)
        if failure is not None:
            raise RuntimeError(
                f"Calibration not saved: {type(failure).__name__}: {failure}. "
                f"Details: {folder}/session.json"
            ) from failure
        assert previous is not None
        diagnostics = {
            "method": "endpoint-first manual",
            "validated": True,
            "restored": True,
            "calibration_applied": False,
            "neutral_orientation_verified": False,
            "powered_motion_verified": False,
            "validation_sweeps": args.validations,
            "capture_folder": str(folder.resolve()),
            "motor_ids": {joint: motor_id},
            "homing_offsets_requested": {joint: session["candidate"]["homing_offset"]},
            "homing_readback": {
                joint: {
                    "homing_offset": session["homing_readback"]["read_back"],
                    "range_min": previous["Min_Position_Limit"],
                    "range_max": previous["Max_Position_Limit"],
                }
            },
            "range_mins": {
                joint: min(low, *(entry["raw_min"] for entry in session["trials"]))
                - session["candidate"]["homing_offset"]
            },
            "range_maxes": {
                joint: max(high, *(entry["raw_max"] for entry in session["trials"]))
                - session["candidate"]["homing_offset"]
            },
        }
        half_degrees = (
            (diagnostics["range_maxes"][joint] - diagnostics["range_mins"][joint]) * 180 / 4095
        )
        result = CalibrationResult(
            arm=args.arm,
            joint=joint,
            device_id=getattr(settings, f"{args.arm}_id"),
            port=getattr(settings, f"{args.arm}_port"),
            backend="lerobot",
            joints={
                joint: (
                    JointCalibration(0.0, 100.0, 50.0)
                    if joint == "gripper"
                    else JointCalibration(-half_degrees, half_degrees, 0)
                )
            },
            diagnostics=diagnostics,
        )
        # Archive the complete evidence before publishing the latest joint recipe.
        result.save(folder / "validated_calibration.json")
        latest = settings.data_dir / f"calibration_joint_{args.arm}_{joint}.json"
        if latest.exists():
            atomic_write_json(
                folder / "previous_joint_baseline.json", json.loads(latest.read_text())
            )
        result.save()
        session.update(state="validated", saved_joint_recipe=str(latest), validated=True)
        atomic_write_json(folder / "session.json", session)
        print(
            f"\nVALIDATED: {args.validations}/{args.validations} fresh sweeps passed.\n"
            f"Original registers restored and read back. Torque confirmed OFF.\n"
            f"Recipe: {latest}\nEvidence: {folder}\n"
            "The recipe is saved for joint/arm bring-up; it has not been installed for motion."
        )
        return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("follower", "leader"), default="follower")
    parser.add_argument("--baseline", type=Path, help="Ten-trial baseline session directory")
    parser.add_argument("--validations", type=int, choices=(3, 4, 5), default=5)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument(
        "--output", type=Path, help="New evidence directory; existing paths refused"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Inspect baseline and plan; no device access"
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 600:
        parser.error("max-seconds must be finite within 1–600")
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("Use an interactive terminal for capture and register-write confirmation")

    def terminate(signum, frame):
        raise RuntimeError(f"Signal {signum}: abort and restore prior calibration")

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, terminate)
    try:
        workflow(args, get_settings())
    except Exception as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

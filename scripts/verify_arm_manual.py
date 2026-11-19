"""Four read-only endpoint sweeps per joint, extending the accepted six to ten."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import signal
import sys
import uuid
from pathlib import Path

from so101 import JOINT_NAMES
from so101.atomic_io import atomic_write_json
from so101.bringup.buses import open_feetech_bus
from so101.bringup.standalone import standalone_access
from so101.config import get_settings
from so101.hardware.lerobot_paths import lerobot_calibration_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "installer", ROOT / "scripts/install_arm_calibration.py"
)
assert spec and spec.loader
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
capture = installer.manual.capture


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(folder, settings, arm):
    plan = installer.compile_plan(folder, settings, arm)
    native_path = lerobot_calibration_file(arm, plan["device_id"])
    suite_path = settings.data_dir / f"calibration_{arm}.json"
    if json.loads(native_path.read_text()) != plan["native"]:
        raise RuntimeError("Native calibration differs from accepted recipes")
    suite = json.loads(suite_path.read_text())
    if suite["joints"] != plan["suite"] or suite["device_id"] != plan["device_id"]:
        raise RuntimeError("Suite calibration differs from accepted recipes")
    hashes = {str(path): digest(path) for path in (native_path, suite_path)}
    manifest = json.loads((folder / "arm_session.json").read_text())
    seeds = {}
    for name in JOINT_NAMES:
        recipe_path = folder / manifest["results"][name]["recipe"]
        hashes[str(recipe_path)] = digest(recipe_path)
        session_path = recipe_path.parent / "session.json"
        source = json.loads(session_path.read_text())
        if (
            source["state"] != "validated"
            or not source["original_registers_restored"]
            or len(source["trials"]) != 6
        ):
            raise RuntimeError(f"{name}: six accepted source sweeps required")
        hashes[str(session_path)] = digest(session_path)
        seeds[name] = []
        for trial in source["trials"]:
            path = recipe_path.parent / trial["samples_file"]
            samples = [
                event
                for event in map(json.loads, path.read_text().splitlines())
                if event.get("type") == "sample"
            ]
            if (
                not trial["usable_for_comparison"]
                or len(samples) != trial["sample_count"]
                or min(event["raw_ticks"] for event in samples) != trial["raw_min"]
                or max(event["raw_ticks"] for event in samples) != trial["raw_max"]
            ):
                raise RuntimeError(f"{name}: source trace differs from accepted summary")
            hashes[str(path)] = digest(path)
            seeds[name].append(
                {
                    "raw_min": trial["raw_min"],
                    "raw_max": trial["raw_max"],
                    "sample_count": len(samples),
                    "source": str(path),
                }
            )
    return {"plan": plan, "source_hashes": hashes, "seeds": seeds}


def verify_registers(bus, plan):
    capture.torque_off(bus)
    for name in JOINT_NAMES:
        actual = capture.registers(bus, name)
        expected = plan["native"][name]
        if actual != {
            "Homing_Offset": expected["homing_offset"],
            "Operating_Mode": 0,
            "Min_Position_Limit": expected["range_min"],
            "Max_Position_Limit": expected["range_max"],
        }:
            raise RuntimeError(
                f"{name}: live registers differ from installed reference; stop and investigate"
            )


def find_resume(settings, arm, source, selected):
    candidates = []
    root = settings.data_dir / "manual_arm_verifications" / arm
    for path in root.glob("*/verification.json"):
        try:
            report = json.loads(path.read_text())
            if (
                report["arm"] != arm
                or Path(report["source"]).resolve() != source.resolve()
                or report["joints"] != selected
            ):
                continue
            passed = sum(
                item["state"] == "pass" for name in selected for item in report["attempts"][name]
            )
            candidates.append((passed, path.stat().st_mtime_ns, path.parent))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Cannot inspect saved progress {path}: {exc}") from exc
    return max(candidates)[2] if candidates else None


def repair_bundle_metadata(folder, settings):
    path = folder / "verification.json"
    original = path.read_bytes()
    report = json.loads(original)
    reference = load_reference(Path(report["source"]), settings, report["arm"])
    if report["reference"] == reference:
        return
    previous = report["reference"]
    candidate = copy.deepcopy(previous)
    candidate["plan"]["bundle_sha256"] = reference["plan"]["bundle_sha256"]
    if candidate != reference:
        raise RuntimeError("Repair refused: changes extend beyond the completed bundle hash")
    source = Path(report["source"])
    manifest = json.loads((source / "arm_session.json").read_text())
    bundle = json.loads((source / "validated_joint_bundle.json").read_text())
    recipes = {
        name: installer.arm_manual.verify_completed(source, manifest, name) for name in JOINT_NAMES
    }
    if bundle != installer.arm_manual.make_bundle(manifest, recipes, bundle["created_at"]):
        raise RuntimeError("Repair refused: bundle metadata or recipes differ from the session")
    for name, attempts in report["attempts"].items():
        for attempt in attempts:
            if attempt["state"] != "pass":
                continue
            trial = attempt["trial"]
            trace = folder / name / trial["samples_file"]
            samples = [
                item
                for item in map(json.loads, trace.read_text().splitlines())
                if item.get("type") == "sample"
            ]
            if (
                digest(trace) != attempt["sha256"]
                or len(samples) != trial["sample_count"]
                or min(item["raw_ticks"] for item in samples) != trial["raw_min"]
                or max(item["raw_ticks"] for item in samples) != trial["raw_max"]
            ):
                raise RuntimeError("Repair refused: passed verification trace differs")
            plan = reference["plan"]
            offset = plan["native"][name]["homing_offset"]
            bounds = {
                "raw_min": plan["observed"][name]["min"] + offset,
                "raw_max": plan["observed"][name]["max"] + offset,
            }
            installer.manual.check_trial(
                folder / name, trial, bounds, offset=offset, dwell_seconds=0.5
            )
            if (
                trial["raw_min"] < bounds["raw_min"] - 64
                or trial["raw_max"] > bounds["raw_max"] + 64
            ):
                raise RuntimeError("Repair refused: passed endpoints drift outside reference")
    backup = folder / f"verification.before-bundle-repair-{uuid.uuid4().hex[:8]}.json"
    installer.atomic_write_bytes(backup, original)
    report.setdefault("reference_repairs", []).append(
        {
            "reason": "completed_bundle_republished_with_unchanged_calibration_recipes_and_traces",
            "at": capture.utc_now(),
            "previous_journal": backup.name,
            "previous_journal_sha256": digest(backup),
            "old_bundle_sha256": previous["plan"]["bundle_sha256"],
            "new_bundle_sha256": reference["plan"]["bundle_sha256"],
        }
    )
    report["reference"] = reference
    atomic_write_json(path, report)
    print(f"Repaired bundle metadata reference; passed sweeps retained. Backup: {backup}")


def collect(
    args, settings, *, ask=input, opener=open_feetech_bus, capture_fn=capture.capture_trial
):
    resumed = json.loads((args.resume / "verification.json").read_text()) if args.resume else None
    source = (
        args.source
        or (Path(resumed["source"]) if resumed else None)
        or settings.data_dir / "manual_arm_calibrations" / args.arm / installer.DEFAULT_SESSION
    )
    reference = load_reference(source, settings, args.arm)
    plan = reference["plan"]
    selected = list(args.joint or installer.arm_manual.ORDER)
    if len(selected) != len(set(selected)):
        raise ValueError("Select each joint once")
    if not args.resume and not getattr(args, "new", False):
        args.resume = find_resume(settings, args.arm, source, selected)
        if args.resume:
            resumed = json.loads((args.resume / "verification.json").read_text())
    folder = (
        args.resume
        or settings.data_dir / "manual_arm_verifications" / args.arm / uuid.uuid4().hex[:12]
    )
    report = {
        "schema_version": 1,
        "arm": args.arm,
        "source": str(source),
        "reference": reference,
        "joints": selected,
        "attempts": {name: [] for name in selected},
        "state": "ready",
        "calibration_applied": False,
        "powered_motion_verified": False,
    }
    if args.resume:
        report = resumed
        if report["reference"] != reference or report["arm"] != args.arm:
            raise RuntimeError("Resume reference changed; original calibration/evidence must match")
        if args.joint and report["joints"] != selected:
            raise RuntimeError("Resume must preserve joint selection")
        selected = report["joints"]
        for name in selected:
            for attempt in report["attempts"][name]:
                if attempt["state"] == "pass":
                    path = folder / name / attempt["trial"]["samples_file"]
                    if digest(path) != attempt["sha256"]:
                        raise RuntimeError("Previously passed verification trace changed")
    print(
        "READ-ONLY verification: 6 accepted sweeps + 4 fresh sweeps = 10 per selected joint.\n"
        "No homing, limits, torque, goal-position or calibration-file writes."
    )
    if args.resume:
        print(f"RESUMING: {folder}")
        for name in selected:
            passed = sum(item["state"] == "pass" for item in report["attempts"][name])
            print(f"  {name}: {6 + passed}/10 accepted · {4 - passed} remaining")
        if (
            all(
                sum(item["state"] == "pass" for item in report["attempts"][name]) == 4
                for name in selected
            )
            and report["state"] == "ten_sweeps_verified"
        ):
            print(
                "Already complete; saved traces verified. No bus opened. Use --new for a fresh run."
            )
            return folder
    if args.dry_run:
        print("Order: " + " → ".join(selected) + "\nDRY RUN: no bus access or evidence writes.")
        return folder
    print(
        "Stop the motor dashboard. All six motors must ALREADY be torque-off.\n"
        "Support the entire arm over a padded catch, verify IDs/cables, clear pinch points,\n"
        "and keep the physical cutoff reachable. Software E-stop is unavailable here.\n"
        "Move only the named joint, gently; respect wrist-roll cable travel. Never force stops."
    )
    if ask("Type READY to confirm setup: ").strip().casefold() != "ready":
        raise RuntimeError("Setup not confirmed; no bus opened")
    with standalone_access("read-only arm endpoint verification", settings=settings):
        folder.mkdir(parents=True, mode=0o700, exist_ok=bool(args.resume))
        bus = None
        try:
            atomic_write_json(folder / "verification.json", report)
            bus = opener(plan["port"], baudrate=settings.baudrate)
            verify_registers(bus, plan)
            for name in selected:
                joint_folder = folder / name
                joint_folder.mkdir(exist_ok=True)
                attempts = report["attempts"][name]
                while sum(item["state"] == "pass" for item in attempts) < 4:
                    count = sum(item["state"] == "pass" for item in attempts)
                    print(
                        f"\n{name} · motor ID {plan['native'][name]['id']} · sweep {7 + count}/10"
                    )
                    while True:
                        answer = ask(
                            "Press Enter to START (empty input), q then Enter to stop: "
                        ).strip()
                        if answer.casefold() == "q":
                            raise RuntimeError("Operator stopped; evidence retained")
                        if not answer:
                            break
                        print("Press the Enter key without typing a word.")
                    verify_registers(bus, plan)
                    attempt = {"state": "recording"}
                    attempts.append(attempt)
                    atomic_write_json(folder / "verification.json", report)
                    try:
                        trial = capture_fn(
                            bus,
                            joint_folder,
                            len(attempts),
                            capture.keyboard_stop,
                            joint=name,
                            max_seconds=180,
                            ready=lambda: print(
                                "RECORDING: pause gently 1–2 s at BOTH ends.\n"
                                "Watch LOW/HIGH READY. Press Enter to FINISH; q then Enter aborts.",
                                flush=True,
                            ),
                        )
                        attempt["trial"] = trial
                        verify_registers(bus, plan)
                        offset = plan["native"][name]["homing_offset"]
                        bounds = {
                            "raw_min": plan["observed"][name]["min"] + offset,
                            "raw_max": plan["observed"][name]["max"] + offset,
                        }
                        attempt["checks"] = installer.manual.check_trial(
                            joint_folder, trial, bounds, offset=offset, dwell_seconds=0.5
                        )
                        if (
                            trial["raw_min"] < bounds["raw_min"] - 64
                            or trial["raw_max"] > bounds["raw_max"] + 64
                        ):
                            raise RuntimeError(
                                "Observed endpoints drifted more than 64 ticks outside reference"
                            )
                        attempt.update(
                            state="pass", sha256=digest(joint_folder / trial["samples_file"])
                        )
                        print(f"PASS: {name}, raw {trial['raw_min']}..{trial['raw_max']}")
                    except BaseException as exc:
                        attempt.update(state="failed", error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        atomic_write_json(folder / "verification.json", report)
                all_trials = reference["seeds"][name] + [
                    item["trial"] for item in attempts if item["state"] == "pass"
                ]
                lows, highs = (
                    [item["raw_min"] for item in all_trials],
                    [item["raw_max"] for item in all_trials],
                )
                report.setdefault("comparison", {})[name] = {
                    "successful_sweeps": len(all_trials),
                    "low_endpoint_spread_ticks": max(lows) - min(lows),
                    "high_endpoint_spread_ticks": max(highs) - min(highs),
                    "raw_envelope": [min(lows), max(highs)],
                }
            verify_registers(bus, plan)
            for path, expected in reference["source_hashes"].items():
                if digest(Path(path)) != expected:
                    raise RuntimeError("Source evidence or calibration changed during capture")
            report["state"] = "ten_sweeps_verified"
        except BaseException as exc:
            report.update(state="partial", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                if bus is not None:
                    bus.disconnect(disable_torque=False)
            except BaseException as exc:
                report.update(state="partial", cleanup_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                atomic_write_json(folder / "verification.json", report)
                print(f"Evidence/resume folder: {folder}")
    print(
        "VERIFIED: ten accepted sweeps per selected joint. Calibration unchanged.\n"
        "Powered operation and physical/model reference still require separate verification."
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("follower", "leader"), default="follower")
    parser.add_argument("--source", type=Path)
    session = parser.add_mutually_exclusive_group()
    session.add_argument("--resume", type=Path, help="Resume this evidence folder")
    session.add_argument("--new", action="store_true", help="Start fresh instead of auto-resuming")
    parser.add_argument("--joint", action="append", choices=JOINT_NAMES)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument(
        "--repair-bundle-metadata",
        action="store_true",
        help="Audit and repair a republished bundle hash; requires --resume",
    )
    args = parser.parse_args()
    if args.repair_bundle_metadata and not args.resume:
        parser.error("--repair-bundle-metadata requires --resume")
    if not args.dry_run and not args.repair_bundle_metadata and not sys.stdin.isatty():
        parser.error("Use an interactive terminal")

    def terminate(signum, frame):
        raise RuntimeError(f"Signal {signum}: stop read-only capture")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, terminate)
    try:
        if args.repair_bundle_metadata:
            repair_bundle_metadata(args.resume, get_settings())
            return 0
        collect(args, get_settings())
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

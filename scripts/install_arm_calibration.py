"""Install six validated manual joint recipes with torque off and verified rollback."""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from so101 import JOINT_NAMES
from so101.atomic_io import atomic_write_bytes, atomic_write_json
from so101.bringup.standalone import standalone_access
from so101.config import get_settings
from so101.hardware.backend import current_backend
from so101.hardware.motors import FeetechBus
from so101.services.store import persist_standalone_fault

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = "20261018T230018Z-83f0c786"
spec = importlib.util.spec_from_file_location(
    "arm_manual", ROOT / "scripts/calibrate_arm_manual.py"
)
assert spec and spec.loader
arm_manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm_manual)
manual = arm_manual.manual


def integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer calibration register, got {value!r}")
    return value


def compile_plan(folder: Path, settings: Any, arm: str) -> dict[str, Any]:
    if current_backend(settings) != "real":
        raise ValueError("Real profile required; no mock fallback")
    manifest = json.loads((folder / "arm_session.json").read_text())
    bundle = json.loads((folder / "validated_joint_bundle.json").read_text())
    if (
        manifest.get("state") != "validated"
        or manifest.get("backend") != "real"
        or manifest.get("arm") != arm
        or manifest.get("device_id") != getattr(settings, f"{arm}_id")
        or manifest.get("port") != getattr(settings, f"{arm}_port")
        or set(manifest.get("joints", [])) != set(JOINT_NAMES)
        or set(manifest.get("results", {})) != set(JOINT_NAMES)
        or manifest.get("validations") != 5
        or bundle.get("all_six_joints_validated") is not True
        or set(bundle.get("joint_recipes", {})) != set(JOINT_NAMES)
    ):
        raise ValueError("Need a completed six-joint, five-validation session for this profile")
    native, suite, observed = {}, {}, {}
    for name in JOINT_NAMES:
        if manifest["results"][name].get("state") != "validated":
            raise ValueError(f"{name}: recipe is not validated")
        recipe = arm_manual.verify_completed(folder, manifest, name)
        if recipe != bundle["joint_recipes"][name]:
            raise ValueError(f"{name}: bundle differs from hashed recipe")
        diag = recipe["diagnostics"]
        offset = integer(diag["homing_offsets_requested"][name])
        low = integer(diag["range_mins"][name])
        high = integer(diag["range_maxes"][name])
        if (
            diag["motor_ids"][name] != JOINT_NAMES.index(name) + 1
            or abs(offset) > 2047
            or not 0 <= low < high <= 4095
            or high - low < 100
            or diag["homing_readback"][name]["homing_offset"] != offset
        ):
            raise ValueError(f"{name}: invalid ID, offset or measured range")
        # Preserve LeRobot's full-turn wrist-roll coordinate scale. The Suite
        # receives measured bounds expressed in that exact same coordinate frame.
        native_low, native_high = (0, 4095) if name == "wrist_roll" else (low, high)
        native[name] = {
            "id": JOINT_NAMES.index(name) + 1,
            "drive_mode": 0,
            "homing_offset": offset,
            "range_min": native_low,
            "range_max": native_high,
        }
        center = (native_low + native_high) / 2
        lo, hi = (
            (0.0, 100.0)
            if name == "gripper"
            else (
                (low - center) * 360 / 4095,
                (high - center) * 360 / 4095,
            )
        )
        suite[name] = {"min": lo, "max": hi, "offset": (lo + hi) / 2}
        observed[name] = {"min": low, "max": high, "raw_midpoint": (low + high) / 2 + offset}
    return {
        "arm": arm,
        "device_id": manifest["device_id"],
        "port": manifest["port"],
        "session": str(folder.resolve()),
        "bundle_sha256": arm_manual.sha256(folder / "validated_joint_bundle.json"),
        "native": native,
        "suite": suite,
        "observed": observed,
        "normalization": "LeRobot 0.6.1 degrees; gripper 0..100; wrist_roll native 0..4095",
        "neutral_orientation_verified": False,
        "powered_motion_verified": False,
    }


def install(args, settings, *, ask=input, bus_factory=FeetechBus, position_fn=None, offset_fn=None):
    folder = args.session or (
        settings.data_dir / "manual_arm_calibrations" / args.arm / DEFAULT_SESSION
    )
    plan = compile_plan(folder, settings, args.arm)
    print("Validated installation plan (no goal commands or torque-on):")
    for name, value in plan["native"].items():
        print(
            f"  {name}: offset {value['homing_offset']:+d}, "
            f"native limits {value['range_min']}..{value['range_max']}, "
            f"Suite bounds {plan['suite'][name]['min']:.2f}..{plan['suite'][name]['max']:.2f}"
        )
    print("Folded rest pose, physical neutral frame and powered motion remain unverified.")
    if args.dry_run:
        print("DRY RUN: files inspected only; no locks, device access or writes.")
        return plan
    print(
        "Stop the dashboard. Support the entire arm over a padded catch; verify cable/ID "
        "mapping, clear pinch points and keep the physical cutoff reachable.\n"
        "All motors will stay torque-off. You will manually center each named joint.\n"
        "Software E-stop is unavailable here. Wrist roll must stay within safe cable travel."
    )
    if ask("Type READY to confirm physical setup: ").strip().casefold() != "ready":
        raise RuntimeError("Setup not confirmed; no bus opened")
    position_fn = position_fn or manual.position_at_midpoint
    offset_fn = offset_fn or manual.write_offset
    evidence = settings.data_dir / "arm_calibration_installs" / args.arm / uuid.uuid4().hex[:12]
    with standalone_access("install validated manual calibration", settings=settings) as access:
        evidence.mkdir(parents=True, mode=0o700, exist_ok=False)
        atomic_write_json(evidence / "plan.json", plan)
        state: dict[str, Any] = {"state": "preparing", "plan": plan, "probes": {}, "backups": []}
        arm_bus = bus_factory(getattr(settings, f"{args.arm}_port"), settings.baudrate, args.arm)
        native = None
        previous: dict[str, dict] = {}
        backups: dict[Path, bytes | None] = {}
        changed = False
        committed = False

        def save():
            atomic_write_json(evidence / "install.json", state)

        def fault(exc):
            persist_standalone_fault(
                settings, exc, device_name=args.arm, torque_unconfirmed=False, energized=False
            )

        try:
            arm_bus.connect()
            arm_bus.set_torque(False)
            native = arm_bus._bus
            manual.capture.torque_off(native)
            if set(native.motors) != set(JOINT_NAMES):
                raise RuntimeError("Motor mapping must contain exactly six joints")
            for name in JOINT_NAMES:
                if native.motors[name].id != plan["native"][name]["id"]:
                    raise RuntimeError(f"{name}: motor ID mismatch")
                previous[name] = manual.snapshot(native, joint=name)
                if previous[name]["Operating_Mode"] != 0:
                    raise RuntimeError(f"{name}: position mode required")
            suite_path = settings.data_dir / f"calibration_{args.arm}.json"
            native_path = Path(arm_bus._device.calibration_fpath)
            for index, path in enumerate((suite_path, native_path)):
                old = path.read_bytes() if path.exists() else None
                backups[path] = old
                backup = evidence / f"previous_file_{index}.bin"
                if old is not None:
                    atomic_write_bytes(backup, old)
                state["backups"].append(
                    {
                        "path": str(path),
                        "existed": old is not None,
                        "backup": backup.name if old is not None else None,
                    }
                )
            state.update(registers_before=previous, state="backed_up")
            save()
            if (
                ask("Type INSTALL to apply this plan, retaining torque OFF: ").strip().casefold()
                != "install"
            ):
                raise RuntimeError("Installation cancelled before register changes")
            for name in arm_manual.ORDER:
                print(
                    f"\n{name}: manually center this joint; support every other link.", flush=True
                )
                state["positioning"] = name
                save()
                position_fn(native, plan["observed"][name]["raw_midpoint"], joint=name)
                changed = True
                state.update(state="writing", active_joint=name)
                save()

                def record(probe, joint=name):
                    state["probes"][joint] = probe
                    save()

                offset_fn(native, plan["native"][name]["homing_offset"], joint=name, record=record)
                for register, field in (
                    ("Min_Position_Limit", "range_min"),
                    ("Max_Position_Limit", "range_max"),
                ):
                    native.write(register, name, plan["native"][name][field], normalize=False)
                native.write("Lock", name, previous[name]["Lock"], normalize=False)
            for name in JOINT_NAMES:
                actual = manual.snapshot(native, joint=name)
                expected = {
                    **previous[name],
                    "Homing_Offset": plan["native"][name]["homing_offset"],
                    "Min_Position_Limit": plan["native"][name]["range_min"],
                    "Max_Position_Limit": plan["native"][name]["range_max"],
                }
                if actual != expected:
                    raise RuntimeError(f"{name}: final register read-back mismatch")
            manual.capture.torque_off(native)
            positions = native.sync_read("Present_Position", list(JOINT_NAMES), normalize=False)
            if set(positions) != set(JOINT_NAMES) or any(
                not plan["observed"][name]["min"]
                <= positions[name]
                <= plan["observed"][name]["max"]
                for name in JOINT_NAMES
            ):
                raise RuntimeError("Final arm readings outside measured ranges; support and review")
            state.update(
                state="registers_verified",
                positions_after=positions,
                registers_after={name: manual.snapshot(native, joint=name) for name in JOINT_NAMES},
            )
            save()
            atomic_write_json(native_path, plan["native"])
            atomic_write_json(
                suite_path,
                {
                    "schema_version": 2,
                    "arm": args.arm,
                    "created_at": time.time(),
                    "device_id": plan["device_id"],
                    "port": plan["port"],
                    "backend": "lerobot",
                    "joints": plan["suite"],
                    "diagnostics": {
                        "installation_evidence": str(evidence),
                        "neutral_orientation_verified": False,
                        "powered_motion_verified": False,
                    },
                },
            )
            if json.loads(native_path.read_text()) != plan["native"]:
                raise RuntimeError("Native file read-back mismatch")
            if json.loads(suite_path.read_text())["joints"] != plan["suite"]:
                raise RuntimeError("Suite file read-back mismatch")
            manual.capture.torque_off(native)
            state.update(state="installed", calibration_applied=True, torque_off_verified=True)
            save()
            committed = True
        except BaseException as exc:
            state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            errors = []
            if changed and native is not None:
                for name in reversed(JOINT_NAMES):
                    try:
                        manual.restore(arm_bus, native, previous[name], joint=name)
                    except BaseException as rollback_error:
                        errors.append(f"{name}: {rollback_error}")
                for path, old in backups.items():
                    try:
                        if old is None:
                            path.unlink(missing_ok=True)
                        else:
                            atomic_write_bytes(path, old)
                        if (path.read_bytes() if path.exists() else None) != old:
                            raise RuntimeError("File restore mismatch")
                    except BaseException as rollback_error:
                        errors.append(f"{path}: {rollback_error}")
                state["rollback_verified"] = not errors
            if errors:
                state["rollback_errors"] = errors
                fault(RuntimeError(f"Calibration install rollback failed: {errors}"))
            raise
        finally:
            try:
                access.disconnect_device(arm_bus, device_name=args.arm)
            except BaseException as cleanup_error:
                state.update(
                    state="installed_cleanup_failed" if committed else "failed",
                    cleanup_error=str(cleanup_error),
                )
                fault(RuntimeError(f"Calibration install cleanup failed: {cleanup_error}"))
                raise
            finally:
                save()
                print(f"Installation evidence: {evidence}", flush=True)
    print(
        "INSTALLED: all six offsets/limits and native/Suite files verified. Torque OFF.\n"
        "No motion commanded. Next: read-only folded rest-pose capture and frame review."
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("follower", "leader"), default="follower")
    parser.add_argument("--session", type=Path, help="Completed manual arm session")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("Use an interactive terminal for physical confirmation")

    def terminate(signum, frame):
        raise RuntimeError(f"Signal {signum}: abort installation and restore")

    handlers = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        install(args, get_settings())
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())

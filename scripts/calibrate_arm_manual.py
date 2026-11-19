"""Joint-by-joint manual SO101 calibration recipes; no commanded motion or UI changes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from so101 import JOINT_NAMES, __version__
from so101.atomic_io import atomic_write_json
from so101.config import get_settings
from so101.hardware.backend import current_backend

ROOT = Path(__file__).resolve().parents[1]
ORDER = tuple(reversed(JOINT_NAMES))
spec = importlib.util.spec_from_file_location(
    "joint_manual_worker", ROOT / "scripts/calibrate_wrist_roll.py"
)
assert spec is not None and spec.loader is not None
manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(manifest, recipes, created_at):
    return {
        "schema_version": 1,
        "arm": manifest["arm"],
        "device_id": manifest["device_id"],
        "port": manifest["port"],
        "backend": "real",
        "created_at": created_at,
        "method": manifest["method"],
        "validation_sweeps_per_joint": manifest["validations"],
        "all_six_joints_validated": set(recipes) == set(JOINT_NAMES),
        "calibration_applied": False,
        "powered_motion_ready": False,
        "neutral_orientation_verified": False,
        "normalization_policy_verified": False,
        "joint_recipes": recipes,
    }


def verify_completed(folder: Path, manifest: dict[str, Any], joint: str) -> dict[str, Any]:
    entry = manifest["results"][joint]
    path = (folder / entry["recipe"]).resolve()
    if not path.is_relative_to(folder.resolve()) or sha256(path) != entry["sha256"]:
        raise RuntimeError(f"{joint}: saved recipe is missing, changed or outside this session")
    recipe = json.loads(path.read_text())
    diagnostics = recipe.get("diagnostics", {})
    if (
        recipe.get("arm") != manifest["arm"]
        or recipe.get("joint") != joint
        or recipe.get("device_id") != manifest["device_id"]
        or recipe.get("port") != manifest["port"]
        or recipe.get("backend") != "lerobot"
        or set(recipe.get("joints", {})) != {joint}
        or diagnostics.get("validated") is not True
        or diagnostics.get("restored") is not True
        or diagnostics.get("calibration_applied") is not False
        or diagnostics.get("validation_sweeps", 0) < manifest["validations"]
    ):
        raise RuntimeError(
            f"{joint}: saved recipe does not match this session's validation contract"
        )
    return recipe


def run_session(args, settings, *, calibrate_fn=manual.workflow) -> Path | None:
    if current_backend(settings) != "real":
        raise RuntimeError("Real backend required; this command never substitutes a mock arm")
    selected = list(args.joint or ORDER)
    if len(set(selected)) != len(selected) or any(name not in JOINT_NAMES for name in selected):
        raise ValueError("Select each valid joint at most once")
    if args.resume:
        folder = args.resume.resolve()
        manifest = json.loads((folder / "arm_session.json").read_text())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("backend") != "real"
            or manifest.get("arm") != args.arm
            or manifest.get("device_id") != getattr(settings, f"{args.arm}_id")
            or manifest.get("port") != getattr(settings, f"{args.arm}_port")
        ):
            raise ValueError("Resume session belongs to another arm/profile")
        selected = manifest["joints"]
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(name not in JOINT_NAMES for name in selected)
            or manifest.get("validations") not in (3, 4, 5)
            or (args.joint and selected != args.joint)
        ):
            raise ValueError("Resume selection or validation count is invalid")
        if args.validations is not None and args.validations != manifest["validations"]:
            raise ValueError("Keep the original validation count when resuming")
        for name, saved_entry in manifest["results"].items():
            if name not in selected:
                raise ValueError("Resume results contain an unselected joint")
            if saved_entry["state"] == "validated":
                verify_completed(folder, manifest, name)
        validations = manifest["validations"]
    else:
        folder = args.output
        validations = args.validations or 5
        manifest = None
    print(
        f"MANUAL JOINT CALIBRATION · {args.arm} · {validations} validation sweeps per joint\n"
        "Order: " + " → ".join(f"{name} (ID {JOINT_NAMES.index(name) + 1})" for name in selected)
    )
    print(
        "Each joint: fresh endpoints → midpoint guide → temporary offset/read-back\n"
        "→ validation sweeps → original registers restored → joint recipe saved.\n"
        "All motors stay torque-off. Support the entire arm. Move only the named joint.\n"
        "Wrist roll: keep safe cable travel; never force a stop or wind cables.\n"
        "No full-arm/native calibration is installed and no motion is commanded."
    )
    if manifest and manifest["state"] == "validated":
        recipes = {name: verify_completed(folder, manifest, name) for name in selected}
        bundle = json.loads((folder / "validated_joint_bundle.json").read_text())
        if bundle != make_bundle(manifest, recipes, bundle["created_at"]):
            raise RuntimeError("Completed bundle differs from the verified session recipes")
        print("REUSE: all selected joints validated; completed bundle preserved unchanged.")
        if args.dry_run:
            print("DRY RUN: no bus locks, serial access or writes.")
        return folder
    if args.dry_run:
        if manifest:
            complete = [
                name
                for name in selected
                if manifest["results"].get(name, {}).get("state") == "validated"
            ]
            print("Resume will reuse these verified session recipes: " + ", ".join(complete))
        print("DRY RUN: plan only; no bus locks, serial access or motor writes.")
        return folder
    if folder is None:
        folder = (
            settings.data_dir
            / "manual_arm_calibrations"
            / args.arm
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        )
    if manifest is None:
        folder.mkdir(parents=True, mode=0o700, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "method": "joint-by-joint endpoint-first manual",
            "backend": "real",
            "suite_version": __version__,
            "recorder_sha256": sha256(Path(__file__)),
            "worker_sha256": sha256(ROOT / "scripts/calibrate_wrist_roll.py"),
            "capture_sha256": sha256(ROOT / "scripts/record_wrist_roll.py"),
            "arm": args.arm,
            "device_id": getattr(settings, f"{args.arm}_id"),
            "port": getattr(settings, f"{args.arm}_port"),
            "joints": selected,
            "validations": validations,
            "started_at": manual.capture.utc_now(),
            "results": {},
            "attempts": [],
            "state": "running",
            "calibration_applied": False,
            "powered_motion_ready": False,
        }
    manifest["state"] = "running"
    manifest.pop("error", None)
    atomic_write_json(folder / "arm_session.json", manifest)
    print(f"Session: {folder}", flush=True)
    try:
        for name in selected:
            if manifest["results"].get(name, {}).get("state") == "validated":
                verify_completed(folder, manifest, name)
                print(f"REUSE: {name}, already validated in this session.")
                continue
            attempt = (
                folder
                / "attempts"
                / f"{name}-{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
            )
            attempt.parent.mkdir(mode=0o700, exist_ok=True)
            entry: dict[str, Any] = {
                "joint": name,
                "state": "running",
                "folder": str(attempt.relative_to(folder)),
                "started_at": manual.capture.utc_now(),
            }
            manifest["attempts"].append(entry)
            manifest["results"][name] = entry
            atomic_write_json(folder / "arm_session.json", manifest)
            print(f"\n=== {name} · motor ID {JOINT_NAMES.index(name) + 1} ===", flush=True)
            # A new reference is captured for each joint. Earlier recipes are
            # preserved in their evidence folders; no guessed pose or offset is reused.
            worker_args = SimpleNamespace(
                arm=args.arm,
                joint=name,
                baseline=None,
                fresh_reference=True,
                validations=validations,
                max_seconds=args.max_seconds,
                output=attempt,
                dry_run=False,
            )
            try:
                evidence = calibrate_fn(worker_args, settings)
                recipe_path = evidence / "validated_calibration.json"
                entry.update(
                    state="validated",
                    recipe=str(recipe_path.relative_to(folder)),
                    sha256=sha256(recipe_path),
                    finished_at=manual.capture.utc_now(),
                )
                verify_completed(folder, manifest, name)
                atomic_write_json(folder / "arm_session.json", manifest)
            except BaseException as exc:
                entry.update(state="failed", error=f"{type(exc).__name__}: {exc}")
                raise
        recipes = {name: verify_completed(folder, manifest, name) for name in selected}
        bundle = make_bundle(manifest, recipes, manual.capture.utc_now())
        atomic_write_json(folder / "validated_joint_bundle.json", bundle)
        manifest.update(state="validated", bundle="validated_joint_bundle.json")
    except BaseException as exc:
        manifest.update(state="partial", error=f"{type(exc).__name__}: {exc}")
        print(f"Stopped; completed joint recipes retained. Resume session: {folder}", flush=True)
        raise
    finally:
        manifest["updated_at"] = manual.capture.utc_now()
        atomic_write_json(folder / "arm_session.json", manifest)
    print(
        f"\nVALIDATED: {len(selected)}/{len(selected)} selected joints.\n"
        f"Bundle: {folder}/validated_joint_bundle.json\n"
        "Original registers restored. Joint recipes are saved separately; no motion enabled."
    )
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("follower", "leader"), default="follower")
    parser.add_argument("--joint", action="append", choices=ORDER, help="Select one or more joints")
    parser.add_argument("--validations", type=int, choices=(3, 4, 5), default=None)
    parser.add_argument("--max-seconds", type=float, default=180)
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--output", type=Path, help="New session directory")
    paths.add_argument(
        "--resume", type=Path, help="Resume this session, preserving validated joints"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan; never open hardware")
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 600:
        parser.error("max-seconds must be finite within 1–600")
    if not args.dry_run and not sys.stdin.isatty():
        parser.error("Use an interactive terminal for physical confirmations and Enter controls")

    def terminate(signum, frame):
        raise RuntimeError(f"Signal {signum}: abort and restore the active joint")

    handlers = {
        signum: signal.signal(signum, terminate) for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        run_session(args, get_settings())
    except Exception as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

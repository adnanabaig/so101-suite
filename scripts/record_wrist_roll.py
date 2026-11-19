"""Ten manually delimited, read-only wrist-roll sweeps; no calibration writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import statistics
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from so101 import JOINT_NAMES, __version__
from so101.atomic_io import atomic_write_json
from so101.bringup.buses import FakeFeetechBus, RegisterBus, open_feetech_bus
from so101.bringup.standalone import standalone_access
from so101.config import get_settings
from so101.hardware.backend import current_backend

JOINT = "wrist_roll"
REGISTERS = ("Homing_Offset", "Operating_Mode", "Min_Position_Limit", "Max_Position_Limit")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def torque_off(bus: RegisterBus) -> dict[str, int]:
    values = bus.sync_read("Torque_Enable", list(JOINT_NAMES), normalize=False, num_retry=1)
    if set(values) != set(JOINT_NAMES) or any(value != 0 for value in values.values()):
        raise RuntimeError(
            f"All six motors must read torque OFF before manual movement: {values}. "
            "This recorder never changes torque. Stop and resolve this through the Suite."
        )
    return values


def registers(bus: RegisterBus, joint: str = JOINT) -> dict[str, int]:
    return {name: int(bus.read(name, joint, normalize=False, num_retry=1)) for name in REGISTERS}


def candidate(low: int, high: int) -> dict[str, Any]:
    offset = round((low + high) / 2) - 2047
    return {
        "homing_offset": offset,
        "range_min": low - offset,
        "range_max": high - offset,
        "offset_representable": abs(offset) <= 2047,
        "range_representable": 0 <= low - offset < high - offset <= 4095,
        "applied": False,
    }


def settled_endpoints(samples: list[dict], low: int, high: int) -> dict[str, Any]:
    """Stable windows near observed extremes; raw extrema remain untouched.

    Use the existing 64-tick coverage tolerance for proximity, but require
    a <=10-tick spread for >=0.5 s without a sampling gap above 100 ms.
    """
    checks: dict[str, Any] = {
        "low_dwell_seconds": 0.0,
        "high_dwell_seconds": 0.0,
        "low_settled_ticks": None,
        "high_settled_ticks": None,
    }
    if high - low < 100:
        return checks
    minima: deque[int] = deque()
    maxima: deque[int] = deque()
    left = 0
    for index, sample in enumerate(samples):
        if (
            index
            and not 0
            < sample["elapsed_seconds"] - samples[index - 1]["elapsed_seconds"]
            <= 0.100001
        ):
            minima.clear()
            maxima.clear()
            left = index
        value = sample["raw_ticks"]
        while minima and samples[minima[-1]]["raw_ticks"] >= value:
            minima.pop()
        while maxima and samples[maxima[-1]]["raw_ticks"] <= value:
            maxima.pop()
        minima.append(index)
        maxima.append(index)
        while samples[maxima[0]]["raw_ticks"] - samples[minima[0]]["raw_ticks"] > 10:
            left += 1
            if minima[0] < left:
                minima.popleft()
            if maxima[0] < left:
                maxima.popleft()
        duration = sample["elapsed_seconds"] - samples[left]["elapsed_seconds"]
        window_low = samples[minima[0]]["raw_ticks"]
        window_high = samples[maxima[0]]["raw_ticks"]
        for side, near, edge in (
            ("low", window_high <= low + 64, window_high),
            ("high", window_low >= high - 64, window_low),
        ):
            if not near:
                continue
            checks[f"{side}_dwell_seconds"] = max(checks[f"{side}_dwell_seconds"], duration)
            if duration >= 0.5:
                previous = checks[f"{side}_settled_ticks"]
                if previous is None or (edge < previous if side == "low" else edge > previous):
                    checks[f"{side}_settled_ticks"] = edge
    return checks


def capture_trial(
    bus: RegisterBus,
    folder: Path,
    number: int,
    stop_key,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    max_seconds: float = 180.0,
    ready=None,
    joint: str = JOINT,
) -> dict[str, Any]:
    """Main owning thread performs every read. stop_key: None, complete or abort."""
    if joint not in JOINT_NAMES:
        raise ValueError("unknown calibration joint")
    started = clock()
    result: dict[str, Any] = {
        "trial": number,
        "joint": joint,
        "started_at": utc_now(),
        "status": "recording",
        "sample_count": 0,
        "error": None,
        "operator_confirmed_safe_endpoints": False,
        "samples_file": f"trial_{number:02d}.jsonl",
    }
    raw_low: int | None = None
    raw_high: int | None = None
    reported_low: int | None = None
    reported_high: int | None = None
    previous_raw: int | None = None
    largest_step = 0
    feedback_samples: list[dict] = []
    with (folder / result["samples_file"]).open("x", encoding="utf-8") as journal:

        def emit(event: dict[str, Any]) -> None:
            journal.write(json.dumps(event, allow_nan=False) + "\n")
            journal.flush()

        emit({"type": "trial_start", **result})
        last_check = last_flush = started
        try:
            initial = registers(bus, joint)
            result["registers_before"] = initial
            emit({"type": "initial_state", "registers": initial, "torque": torque_off(bus)})
            os.fsync(journal.fileno())
            if ready:
                ready()
            while True:
                key = stop_key()
                if key is not None:
                    result["status"] = "complete" if key == "complete" else "interrupted"
                    result["operator_confirmed_safe_endpoints"] = key == "complete"
                    break
                if clock() - started >= max_seconds:
                    result["status"] = "timed_out"
                    result["error"] = "Trial time limit reached; both endpoints not confirmed."
                    break
                if clock() - last_check >= 0.5:
                    emit({"type": "torque_check", "torque": torque_off(bus)})
                    last_check = clock()
                read_started = clock()
                values = bus.sync_read("Present_Position", [joint], normalize=False, num_retry=1)
                value = values[joint]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"Invalid encoder reading: {value!r}")
                if int(value) != value:
                    raise ValueError(f"Non-integer encoder reading: {value!r}")
                reported = int(value)
                raw = reported + initial["Homing_Offset"]
                captured = clock()
                feedback_samples.append({"raw_ticks": raw, "elapsed_seconds": captured - started})
                emit(
                    {
                        "type": "sample",
                        "utc": utc_now(),
                        "elapsed_seconds": captured - started,
                        "read_seconds": captured - read_started,
                        "reported_ticks": reported,
                        "homing_offset": initial["Homing_Offset"],
                        "raw_ticks": raw,
                    }
                )
                raw_low = raw if raw_low is None else min(raw_low, raw)
                raw_high = raw if raw_high is None else max(raw_high, raw)
                reported_low = reported if reported_low is None else min(reported_low, reported)
                reported_high = reported if reported_high is None else max(reported_high, reported)
                if previous_raw is not None:
                    largest_step = max(largest_step, abs(raw - previous_raw))
                previous_raw = raw
                result["sample_count"] += 1
                if captured - last_flush >= 1.0:
                    os.fsync(journal.fileno())
                    last_flush = captured
                    print(
                        f"  {result['sample_count']} samples · raw {raw} · "
                        f"min {raw_low} · max {raw_high}",
                        flush=True,
                    )
                    settled = settled_endpoints(feedback_samples, raw_low, raw_high)
                    labels = {
                        side: "READY" if settled[side + "_settled_ticks"] is not None else "WAIT"
                        for side in ("low", "high")
                    }
                    print(
                        "  Settled endpoints (observed so far): "
                        + " · ".join(
                            f"{side.upper()} {labels[side]} "
                            f"({settled[f'{side}_dwell_seconds']:.2f}/0.50 s, "
                            f"ticks {settled[f'{side}_settled_ticks']})"
                            for side in ("low", "high")
                        ),
                        flush=True,
                    )
                sleep(0.02)
            final = registers(bus, joint)
            result["registers_after"] = final
            emit({"type": "final_state", "registers": final, "torque": torque_off(bus)})
            if final != initial:
                raise RuntimeError(
                    "Calibration/mode registers changed during capture; trial is invalid."
                )
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Ctrl+C; partial readings retained")
        except Exception as exc:
            result.update(status="error", error=str(exc))
        finally:
            result.update(
                finished_at=utc_now(),
                duration_seconds=max(0.0, clock() - started),
                raw_min=raw_low,
                raw_max=raw_high,
                reported_min=reported_low,
                reported_max=reported_high,
                largest_step_ticks=largest_step,
            )
            warnings = []
            if result["sample_count"] < 2:
                warnings.append("fewer than two samples")
            if raw_low is not None and raw_high is not None:
                result["span_ticks"] = raw_high - raw_low
                result["candidate_not_applied"] = candidate(raw_low, raw_high)
                if raw_high - raw_low < 100:
                    warnings.append("observed span below 100 ticks")
                if raw_high - raw_low > 4095:
                    warnings.append("observed span exceeds one encoder revolution")
            if largest_step > 2048:
                warnings.append("large adjacent jump: inspect possible wrap or discontinuity")
            result["quality_warnings"] = warnings
            result["usable_for_comparison"] = (
                result["status"] == "complete"
                and not warnings
                and result["operator_confirmed_safe_endpoints"]
            )
            emit({"type": "trial_end", **result})
            os.fsync(journal.fileno())
    atomic_write_json(folder / f"trial_{number:02d}.json", result)
    return result


def analyze(folder: Path) -> dict[str, Any]:
    session = json.loads((folder / "session.json").read_text())
    trials = [json.loads(path.read_text()) for path in sorted(folder.glob("trial_*.json"))]
    usable = [trial for trial in trials if trial["usable_for_comparison"]]
    report: dict[str, Any] = {
        "session_id": session["session_id"],
        "backend": session["backend"],
        "joint": JOINT,
        "requested_trials": session["requested_trials"],
        "recorded_trials": len(trials),
        "usable_trials": len(usable),
        "all_requested_trials_usable": len(usable) == session["requested_trials"],
        "hardware_safety_qualified": False,
        "calibration_applied": False,
        "trials": trials,
    }
    if usable:
        lows = [trial["raw_min"] for trial in usable]
        highs = [trial["raw_max"] for trial in usable]
        report["comparison"] = {
            "median_low": statistics.median(lows),
            "median_high": statistics.median(highs),
            "low_spread_ticks": max(lows) - min(lows),
            "high_spread_ticks": max(highs) - min(highs),
            "envelope_min": min(lows),
            "envelope_max": max(highs),
            "candidate_for_observed_envelope_not_applied": candidate(min(lows), max(highs)),
        }
    atomic_write_json(folder / "report.json", report)
    return report


def keyboard_stop() -> str | None:
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    line = sys.stdin.readline()
    if not line or line.strip().lower() == "q":
        return "abort"
    return "complete" if not line.strip() else None


def collect(args, settings, *, ask=input, opener=open_feetech_bus) -> Path:
    backend = current_backend(settings)
    if backend == "sim":
        raise RuntimeError("Simulation has no real encoder baseline; use real or explicit --mock.")
    if not args.mock and backend != "real":
        raise RuntimeError("Real backend required; --mock explicitly labels a rehearsal.")
    print("MOCK REHEARSAL — synthetic readings" if args.mock else "REAL encoder capture")
    print(
        "READ-ONLY WRIST-ROLL BASELINE · motor ID 5\n"
        "No torque, homing, limits or calibration-file writes.\n"
        "Stop the dashboard first. Clamp/support the arm over a padded catch, verify\n"
        "the follower/leader cable and motor mapping, clear pinch points and keep\n"
        "the physical power cutoff reachable. All six motors must already be torque-off.\n"
        "Software E-stop is unavailable while the dashboard is stopped.\n"
        "Only move wrist roll within safe cable travel; never force or wind cables."
    )
    if (
        ask("Type READY to confirm this physical setup, or anything else to quit: ")
        .strip()
        .casefold()
        != "ready"
    ):
        raise RuntimeError("Setup not confirmed; no motor bus opened.")
    # Canonical locks + persistent local/global latches; no competing driver.
    with standalone_access("manual wrist-roll baseline capture", settings=settings):
        port = getattr(settings, f"{args.arm}_port")
        bus = (
            FakeFeetechBus(
                arm=args.arm,
                registers={"Operating_Mode": dict.fromkeys(JOINT_NAMES, 0)},
            )
            if args.mock
            else opener(port, baudrate=settings.baudrate)
        )
        session: dict[str, Any] | None = None
        folder: Path | None = None
        try:
            torque_off(bus)
            session_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
            )
            folder = args.output or (
                settings.data_dir / "manual_joint_baselines" / JOINT / session_id
            )
            folder.mkdir(parents=True, exist_ok=False, mode=0o700)
            session = {
                "schema_version": 1,
                "session_id": session_id,
                "started_at": utc_now(),
                "backend": backend,
                "synthetic": args.mock,
                "suite_version": __version__,
                "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python_version": sys.version,
                "arm": args.arm,
                "joint": JOINT,
                "expected_motor_id": 5,
                "device_id": getattr(settings, f"{args.arm}_id"),
                "port": str(port),
                "baudrate": settings.baudrate,
                "requested_trials": args.trials,
                "state": "recording",
                "completed_trials": 0,
                "trial_results": [],
                "sampling_target_hz": 50,
                "maximum_trial_seconds": args.max_seconds,
                "coordinates": "raw_ticks = reported_ticks + unchanged Homing_Offset",
                "calibration_applied": False,
            }
            atomic_write_json(folder / "session.json", session)
            print(f"\nSaving to: {folder}\n")
            for number in range(1, args.trials + 1):
                key = (
                    ask(f"Trial {number}/{args.trials}: Enter to START, or q then Enter to quit: ")
                    .strip()
                    .lower()
                )
                if key == "q":
                    session["state"] = "interrupted"
                    break
                if key:
                    raise RuntimeError(
                        "Expected Enter or q; capture stopped without starting this trial."
                    )
                print("Checking torque and registers — hold still…", flush=True)

                def ready() -> None:
                    print(
                        "RECORDING: gently visit BOTH safe end positions; pause briefly at each.\n"
                        "Enter to FINISH; q then Enter or Ctrl+C aborts and keeps partial data.",
                        flush=True,
                    )

                result = capture_trial(
                    bus, folder, number, keyboard_stop, max_seconds=args.max_seconds, ready=ready
                )
                session["trial_results"].append(f"trial_{number:02d}.json")
                session["completed_trials"] += result["status"] == "complete"
                atomic_write_json(folder / "session.json", session)
                print(
                    f"Trial {number}: {result['status']} · {result['sample_count']} samples · "
                    f"raw {result['raw_min']} → {result['raw_max']} · "
                    f"warnings: {result['quality_warnings']}\n",
                    flush=True,
                )
                if result["status"] != "complete":
                    session["state"] = result["status"]
                    break
            else:
                session["state"] = "complete"
        except (KeyboardInterrupt, EOFError):
            if session is not None:
                session["state"] = "interrupted"
        except Exception as exc:
            if session is not None:
                session.update(state="error", error=str(exc))
            raise
        finally:
            try:
                # Bare read-only bus: close host port only; no torque packets.
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                if session is not None:
                    session.update(state="error", close_error=str(exc))
                raise
            finally:
                if session is not None and folder is not None:
                    session["finished_at"] = utc_now()
                    atomic_write_json(folder / "session.json", session)
                    report = analyze(folder)
                    print(
                        f"\nSaved: {folder}\n"
                        f"Usable comparisons: {report['usable_trials']}/{args.trials}. "
                        "No calibration applied.\n"
                        "Send this folder path for analysis.",
                        flush=True,
                    )
        assert folder is not None
        return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("follower", "leader"), default="follower")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, help="New directory; existing paths are refused")
    parser.add_argument(
        "--analyze", type=Path, help="Regenerate an existing session report offline"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Explicit synthetic rehearsal; no real bus"
    )
    args = parser.parse_args()
    if args.analyze:
        report = analyze(args.analyze)
        print(
            json.dumps({key: value for key, value in report.items() if key != "trials"}, indent=2)
        )
        return 0
    if (
        not 1 <= args.trials <= 100
        or not math.isfinite(args.max_seconds)
        or not 1 <= args.max_seconds <= 600
    ):
        parser.error("trials must be 1–100 and max-seconds must be finite within 1–600")
    if not sys.stdin.isatty():
        parser.error("Run interactively in a terminal; Enter controls each trial.")
    settings = get_settings()
    if args.mock:
        settings = settings.model_copy(
            update={
                "backend": "mock",
                "mock": True,
                "data_dir": settings.data_dir.parent / "mock",
            }
        )
    try:
        folder = collect(args, settings)
        return 0 if json.loads((folder / "session.json").read_text())["state"] == "complete" else 1
    except Exception as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

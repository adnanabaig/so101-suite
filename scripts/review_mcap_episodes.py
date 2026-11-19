"""Export and assess saved episodes without opening a motor bus or replaying commands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
storage_spec = importlib.util.spec_from_file_location(
    "recording_storage", ROOT / "scripts/recording_storage.py"
)
assert storage_spec and storage_spec.loader
storage_module = importlib.util.module_from_spec(storage_spec)
storage_spec.loader.exec_module(storage_module)
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def timing(timestamps):
    intervals = [
        (later - earlier) / 1e9 for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    ]
    span = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0
    return {
        "samples": len(timestamps),
        "measured_fps": (len(timestamps) - 1) / span if span > 0 else 0,
        "max_gap_seconds": max(intervals, default=0),
        "timestamps_strictly_increase": all(value > 0 for value in intervals),
    }


def collect_health_advisories(advisories, health, timestamp_ns):
    for arm_name, arm in health.get("arms", {}).items():
        if not isinstance(arm, dict):
            continue
        for motor_name, motor in arm.get("motors", {}).items():
            if not isinstance(motor, dict) or motor.get("level") not in ("warn", "critical"):
                continue
            reasons = tuple(str(reason) for reason in motor.get("reasons", []))
            key = (arm_name, motor_name, motor["level"], reasons)
            if key not in advisories:
                advisories[key] = {
                    "arm": arm_name,
                    "motor": motor_name,
                    "level": motor["level"],
                    "reasons": list(reasons),
                    "first_log_time_ns": timestamp_ns,
                    "last_log_time_ns": timestamp_ns,
                    "health_message_count": 0,
                }
            advisory = advisories[key]
            advisory["last_log_time_ns"] = timestamp_ns
            advisory["health_message_count"] += 1


def assess(episode, ticks, cameras, topics, camera_clocks, *, allow_mock=False):
    failures = []
    duration = (episode["ended_at_ns"] - episode["started_at_ns"]) / 1e9
    if episode["status"] != "saved":
        failures.append("Episode was interrupted or not saved")
    if duration < 5:
        failures.append("Episode is shorter than the five-second pilot requirement")
    control = timing([tick["timestamp_ns"] for tick in ticks])
    if control["samples"] < 2 or control["measured_fps"] < 27:
        failures.append("Control tick capture is below 27 Hz (30 Hz target)")
    if control["max_gap_seconds"] > 0.1 or not control["timestamps_strictly_increase"]:
        failures.append("Control timestamps contain a gap over 100 ms or fail ordering")
    if ticks and (
        abs(ticks[0]["timestamp_ns"] - episode["started_at_ns"]) > 150_000_000
        or abs(episode["ended_at_ns"] - ticks[-1]["timestamp_ns"]) > 150_000_000
    ):
        failures.append("Control capture does not cover both episode boundaries within 150 ms")
    stream = None
    previous = None
    tracking = defaultdict(list)
    leader_error = defaultdict(list)
    stationary_demand = Counter()
    longest_stationary_demand = Counter()
    stationary_runs = Counter()
    previous_positions = {}
    previous_timestamp = None
    clamped = Counter()
    for tick in ticks:
        if not allow_mock and tick.get("backend") != "real":
            failures.append("Control samples are not from the real backend")
        if stream is not None and stream != tick.get("stream_id"):
            failures.append("Control stream restarted inside the episode")
            previous_positions.clear()
            stationary_runs.clear()
            previous_timestamp = None
        if previous is not None and tick["seq"] != previous + 1:
            failures.append("Control tick sequence contains a gap or duplicate")
            previous_positions.clear()
            stationary_runs.clear()
            previous_timestamp = None
        stream, previous = tick.get("stream_id"), tick["seq"]
        timestamp = tick["timestamp_ns"]
        delta_seconds = (
            (timestamp - previous_timestamp) / 1e9 if previous_timestamp is not None else 0.0
        )
        for joint in JOINTS:
            key = f"{joint}.pos"
            values = [
                tick.get(field, {}).get(key)
                for field in ("observation", "action", "desired_action")
            ]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in values
            ):
                failures.append(f"Missing or non-finite {joint} observation/command/intent")
                previous_positions.pop(joint, None)
                stationary_runs[joint] = 0.0
                continue
            observed, accepted, desired = values
            tracking[joint].append(abs(observed - accepted))
            leader_error[joint].append(abs(observed - desired))
            clamped[joint] += abs(accepted - desired) > 1e-6
            stationary = (
                joint in previous_positions
                and 0 < delta_seconds <= 0.1
                and abs(observed - previous_positions[joint]) <= 0.1
                and abs(observed - desired) > 10.0
            )
            stationary_runs[joint] = stationary_runs[joint] + delta_seconds if stationary else 0
            if stationary:
                stationary_demand[joint] += delta_seconds
                longest_stationary_demand[joint] = max(
                    longest_stationary_demand[joint], stationary_runs[joint]
                )
            previous_positions[joint] = observed
        previous_timestamp = timestamp
    camera_report = {name: timing(stamps) for name, stamps in cameras.items()}
    camera_targets = {}
    for name in ("obsbot", "camera_1", "camera_2", "realsense"):
        stats = camera_report.get(name, timing([]))
        stamps = cameras.get(name, [])
        covered = (
            bool(stamps)
            and abs(stamps[0] - episode["started_at_ns"]) <= 250_000_000
            and abs(episode["ended_at_ns"] - stamps[-1]) <= 250_000_000
        )
        camera_targets[name] = {
            "target_fps": 30,
            "minimum_fps": 27,
            "present": bool(stamps),
            "passes": bool(
                stats["samples"] >= 2
                and stats["measured_fps"] >= 27
                and stats["max_gap_seconds"] <= 0.15
                and stats["timestamps_strictly_increase"]
                and covered
                and camera_clocks.get(name) == {"host_camera_read_completed"}
            ),
        }
    primary = camera_report.get("obsbot", timing([]))
    if primary["samples"] < 2 or primary["measured_fps"] < 27 or primary["max_gap_seconds"] > 0.15:
        failures.append("Required OBSBOT view misses the 27 fps / 150 ms gap requirement")
    primary_stamps = cameras.get("obsbot", [])
    if primary_stamps and (
        abs(primary_stamps[0] - episode["started_at_ns"]) > 250_000_000
        or abs(episode["ended_at_ns"] - primary_stamps[-1]) > 250_000_000
    ):
        failures.append("Required camera does not cover both episode boundaries within 250 ms")
    if camera_clocks.get("obsbot") != {"host_camera_read_completed"}:
        failures.append("Required camera capture-clock provenance is missing or inconsistent")
    if topics.get("/robot/encoder/scene", 0) == 0:
        failures.append("Recorded encoder-driven 3D model is missing")
    if topics.get("/robot/health", 0) == 0:
        failures.append("Servo-health observations are missing")
    return {
        "data_checks_passed": not failures,
        "issues": sorted(set(failures)),
        "human_task_review": "pending",
        "duration_seconds": duration,
        "control": control,
        "cameras": camera_report,
        "camera_target_review": {
            "all_four_views_ready": all(item["passes"] for item in camera_targets.values()),
            "views": camera_targets,
            "note": (
                "Separate from the historical OBSBOT/control integrity checks. "
                "Saved tuning runs are not accepted task demonstrations."
            ),
        },
        "tracking_error": {
            joint: {
                "mean_absolute": statistics.mean(values),
                "maximum_absolute": max(values),
                "unit": "percent" if joint == "gripper" else "degrees",
            }
            for joint, values in tracking.items()
        },
        "clamped_tick_counts": dict(clamped),
        "leader_following": {
            joint: {
                "mean_absolute_error": statistics.mean(values),
                "maximum_absolute_error": max(values),
                "stationary_demand_seconds": stationary_demand[joint],
                "longest_stationary_demand_seconds": longest_stationary_demand[joint],
                "unit": "percent" if joint == "gripper" else "degrees",
            }
            for joint, values in leader_error.items()
        },
        "following_metric_definition": {
            "stationary_position_change_max": 0.1,
            "requested_position_error_min": 10.0,
            "consecutive_sample_gap_seconds_max": 0.1,
            "note": "Diagnostic symptoms, not proof of a mechanical stall or its cause.",
        },
        "note": (
            "Integrity checks do not establish task success, geometric alignment "
            "or optimal motor tuning."
        ),
    }


def export_episode(episode, raw_dir, output, *, allow_mock=False):
    from mcap.reader import make_reader
    from mcap.writer import Writer

    start, end = episode["started_at_ns"], episode["ended_at_ns"]
    starting = Path(episode["starting_segment"]).name.removesuffix(".active")
    if not starting:
        raise ValueError("Episode has no acknowledged starting segment")
    ending_stamp = datetime.fromtimestamp(end / 1e9, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sources = sorted(
        path
        for path in raw_dir.glob("*.mcap")
        if path.name >= starting and path.name[:16] <= ending_stamp
    )
    if not sources or not (raw_dir / starting).exists():
        raise ValueError(
            "Episode segments are not finalized yet or the starting segment is unavailable"
        )
    partial = output.with_suffix(".mcap.partial")
    ticks, cameras, clocks = [], defaultdict(list), defaultdict(set)
    topics = Counter()
    schemas, channels = {}, {}
    used = []
    runtime_issues = set()
    health_advisories = {}
    with partial.open("xb") as handle:
        writer = Writer(handle, enable_data_crcs=True)
        writer.start(profile="", library="so101-episode-review")

        def copy_message(schema, channel, message):
            schema_key = (schema.name, schema.encoding, schema.data) if schema else None
            if schema_key not in schemas:
                schemas[schema_key] = writer.register_schema(*schema_key) if schema_key else 0
            channel_key = (
                channel.topic,
                channel.message_encoding,
                schemas[schema_key],
                tuple(sorted(channel.metadata.items())),
            )
            if channel_key not in channels:
                channels[channel_key] = writer.register_channel(
                    channel.topic, channel.message_encoding, schemas[schema_key], channel.metadata
                )
            writer.add_message(
                channels[channel_key],
                message.log_time,
                message.data,
                message.publish_time,
                message.sequence,
            )
            topics[channel.topic] += 1

        with sources[0].open("rb") as reader_handle:
            prelude = {}
            for schema, channel, message in make_reader(
                reader_handle, validate_crcs=True
            ).iter_messages(end_time=start):
                if channel.topic.startswith("/calibration/"):
                    prelude[channel.topic] = (schema, channel, message)
                elif channel.topic == "/recording/metadata":
                    metadata = json.loads(message.data)
                    if metadata.get("episode_id") == episode["id"]:
                        prelude[channel.topic] = (schema, channel, message)
            for record in sorted(prelude.values(), key=lambda record: record[2].log_time):
                copy_message(*record)
            if prelude:
                used.append(str(sources[0]))
        for source in sources:
            added = 0
            with source.open("rb") as reader_handle:
                reader = make_reader(reader_handle, validate_crcs=True)
                for schema, channel, message in reader.iter_messages(
                    start_time=start, end_time=end + 1
                ):
                    copy_message(schema, channel, message)
                    added += 1
                    if channel.topic == "/robot/control_tick":
                        ticks.append(json.loads(message.data))
                    if channel.topic.startswith("/camera/") and channel.topic.endswith("/image"):
                        cameras[channel.topic.split("/")[2]].append(message.log_time)
                    if channel.topic.startswith("/camera/") and channel.topic.endswith("/status"):
                        status = json.loads(message.data)
                        if "timestamp_source" in status:
                            clocks[channel.topic.split("/")[2]].add(status["timestamp_source"])
                    if channel.topic == "/robot/health":
                        health = json.loads(message.data)
                        collect_health_advisories(health_advisories, health, message.log_time)
                        if health.get("level") == "critical" or any(
                            isinstance(arm, dict) and arm.get("level") == "critical"
                            for arm in health.get("arms", {}).values()
                        ):
                            runtime_issues.add("Critical servo health inside episode")
                    if channel.topic == "/robot/telemetry":
                        telemetry = json.loads(message.data)
                        if telemetry.get("estopped") or telemetry.get("fault"):
                            runtime_issues.add("Safety stop or fault inside episode")
            if added and str(source) not in used:
                used.append(str(source))
        channel_id = writer.register_channel("/recording/review", "json", 0)
        public_episode = {key: value for key, value in episode.items() if key != "operator_id"}
        writer.add_message(
            channel_id,
            end,
            json.dumps(
                {
                    "episode": public_episode,
                    "source_segments": used,
                    "kind": "derived_episode_export",
                    "source_timestamps_preserved": True,
                    "calibration_prelude_topics": sorted(
                        topic for topic in prelude if topic.startswith("/calibration/")
                    ),
                    "prelude_topics": sorted(prelude),
                }
            ).encode(),
            end,
        )
        topics["/recording/review"] += 1
        writer.finish()
        handle.flush()
        import os

        os.fsync(handle.fileno())
    partial.rename(output)
    with output.open("rb") as handle:
        verified = Counter(
            channel.topic
            for _, channel, _ in make_reader(handle, validate_crcs=True).iter_messages()
        )
    if verified != topics:
        raise ValueError("Exported message counts differ from the source interval")
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    report = assess(episode, ticks, cameras, topics, clocks, allow_mock=allow_mock)
    if runtime_issues:
        report["issues"].extend(sorted(runtime_issues))
        report["data_checks_passed"] = False
    return {
        "episode_id": episode["id"],
        "task": episode.get("task", ""),
        "path": str(output),
        "sha256": digest.hexdigest(),
        "source_segments": used,
        "topics": dict(topics),
        "health_review": {
            "attention_required": bool(health_advisories),
            "advisories": list(health_advisories.values()),
            "note": (
                "Repeated health messages can carry the same hardware sample. "
                "Warnings require review; they do not establish overload causality "
                "or calibrated current measurements."
            ),
        },
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/real")
    parser.add_argument("--mcap-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--since", help="ISO timestamp, for example 2026-10-27T21:03:26Z")
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Explicit software validation; never counts as real data",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    profile_path = ROOT / "config/rigs/mcap-storage.local.json"
    raw_dir = args.mcap_dir
    if raw_dir is None:
        storage = storage_module.load_storage(ROOT, profile=profile_path)
        storage.check()
        raw_dir = storage.output
    index = args.data_dir / "mcap/episodes.json"
    if not index.exists():
        print(
            "No saved episodes yet. Complete the physical checks and record a pilot episode first."
        )
        return 1
    entries = json.loads(index.read_text())["episodes"]
    since_ns = (
        round(datetime.fromisoformat(args.since.replace("Z", "+00:00")).timestamp() * 1e9)
        if args.since
        else 0
    )
    selected = [
        entry
        for entry in entries
        if entry["status"] == "saved" and entry["started_at_ns"] >= since_ns
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or raw_dir.parent / "episode-reviews" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    if args.mcap_dir is None:
        storage.output = output
        storage.check()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for episode in reversed(selected):
        destination = output / f"episode-{len(reports) + 1:02d}-{episode['id'][:8]}.mcap"
        try:
            result = export_episode(episode, raw_dir, destination, allow_mock=args.allow_mock)
        except Exception as exc:
            result = {
                "episode_id": episode["id"],
                "data_checks_passed": False,
                "issues": [str(exc)],
            }
        reports.append(result)
        verdict = "DATA CHECKS PASS" if result["data_checks_passed"] else "REVIEW REQUIRED"
        print(f"{episode['id'][:8]}: {verdict}", flush=True)
        if sum(item["data_checks_passed"] for item in reports) >= args.count:
            break
    report = {
        "schema_version": 1,
        "requested_count": args.count,
        "data_checks_passed": sum(item["data_checks_passed"] for item in reports),
        "mock_allowed": args.allow_mock,
        "human_review": "pending",
        "thresholds": {
            "motor_hz_min": 27,
            "motor_gap_seconds_max": 0.1,
            "obsbot_fps_min": 27,
            "obsbot_gap_seconds_max": 0.15,
        },
        "episodes": reports,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output / 'report.json'}")
    print(
        "Open each exported MCAP locally in Foxglove with the episode review layout. "
        "No commands are replayed."
    )
    return 0 if report["data_checks_passed"] >= args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())

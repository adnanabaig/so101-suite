"""Record existing local feeds. No motor bus access or motion authority."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("viewer", ROOT / "scripts/foxglove_viewer.py")
assert spec and spec.loader
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)
camera_publisher, fetch = viewer.camera_publisher, viewer.fetch
storage_spec = importlib.util.spec_from_file_location(
    "recording_storage", ROOT / "scripts/recording_storage.py"
)
assert storage_spec and storage_spec.loader
storage_module = importlib.util.module_from_spec(storage_spec)
storage_spec.loader.exec_module(storage_module)
activity_spec = importlib.util.spec_from_file_location(
    "capture_activity", ROOT / "src/so101/capture_activity.py"
)
assert activity_spec and activity_spec.loader
activity_module = importlib.util.module_from_spec(activity_spec)
activity_spec.loader.exec_module(activity_module)


def log_episodes(state_dir, *, episode_id=None):
    import foxglove

    path = state_dir / "episodes.json"
    if path.exists():
        annotations = json.loads(path.read_text())
        for episode in [annotations.get("active"), *annotations.get("episodes", [])]:
            if episode is not None:
                episode.pop("operator_id", None)
        foxglove.log("/recording/episodes", annotations)
        for episode in [annotations.get("active"), *annotations.get("episodes", [])]:
            if episode is not None and episode["id"] == episode_id:
                foxglove.log("/recording/episode", {"schema_version": 1, **episode})


def segment_metadata(state, requested, active_path):
    sources = (
        "src/so101/teleop/controller.py",
        "src/so101/hardware/follower.py",
        "src/so101/hardware/motors.py",
        "src/so101/hardware/safety.py",
        "src/so101/config.py",
        "src/so101/services/robot_manager.py",
        "src/so101/services/control_trace.py",
        "scripts/record_mcap.py",
        "scripts/arm_camera.py",
        "scripts/foxglove_viewer.py",
        "src/so101/hardware/camera_controls.py",
    )
    return {
        "schema_version": 1,
        "episode_id": requested["id"] if requested["kind"] == "episode" else None,
        "activity_id": requested["id"],
        "activity_kind": requested["kind"],
        "task": requested.get("label", ""),
        "segment_id": active_path.name.removesuffix(".active"),
        "control_stream_id": state.get("control_stream_id"),
        "created_at_ns": time.time_ns(),
        "clock": {
            "log_time": "unix_nanoseconds",
            "control_sample": "host_control_step_completed",
            "duration": "monotonic_seconds",
            "camera": "host_camera_read_completed",
        },
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in sources
        },
        "human_review": "pending",
        "camera_requirements": {
            "expected_views": ["obsbot", "camera_1", "camera_2", "realsense"],
            "target_fps": 30,
            "acceptance_minimum_fps": 27,
        },
    }


def finalize_segment(writer, active_path, state, state_dir):
    import foxglove

    log_episodes(state_dir, episode_id=state.get("activity_id"))
    foxglove.log(
        "/recording/activity",
        {"id": state.get("activity_id"), "kind": state.get("activity_kind"), "phase": "closed"},
    )
    writer.close()
    final = active_path.with_suffix("")
    active_path.rename(final)
    state.update(last_completed_path=str(final), closed_activity_id=state.get("activity_id"))


def log_control_ticks(state, api_base, *, seed=False):
    import foxglove

    after = state.get("control_last_seq", 0)
    suffix = "&latest_only=true" if seed else ""
    batch = json.loads(fetch(f"{api_base}/api/mcap/control-ticks?after_seq={after}{suffix}"))
    if not seed and state.get("control_stream_id") not in (None, batch["stream_id"]):
        batch = json.loads(fetch(f"{api_base}/api/mcap/control-ticks?after_seq=0"))
        state["control_gap_count"] = state.get("control_gap_count", 0) + 1
    state["control_stream_id"] = batch["stream_id"]
    if seed:
        state["control_last_seq"] = batch["last_seq"]
        return
    state["control_gap_count"] = state.get("control_gap_count", 0) + batch["missing_before_buffer"]
    for tick in batch["ticks"]:
        tick = {
            **tick,
            "recording_activity_id": state.get("activity_id"),
            "episode_id": state.get("activity_id")
            if state.get("activity_kind") == "episode"
            else None,
        }
        foxglove.log("/robot/control_tick", tick, log_time=tick["timestamp_ns"])
        state["control_last_seq"] = tick["seq"]
        state["control_ticks_recorded"] = state.get("control_ticks_recorded", 0) + 1
        state["control_received_at"] = time.time()
    state.pop("control_error", None)


def publish_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def telemetry_publisher(stop, state, api_base="http://127.0.0.1:8000"):
    import foxglove
    from websockets.sync.client import connect

    while not stop.is_set():
        try:
            token = os.environ.get("SO101_E2E_TOKEN")
            headers = {"X-SO101-E2E-Token": token} if token else {}
            with connect(
                api_base.replace("http://", "ws://") + "/ws/telemetry",
                open_timeout=3,
                additional_headers=headers,
            ) as socket:
                while not stop.is_set():
                    try:
                        message = json.loads(socket.recv(timeout=1))
                    except TimeoutError:
                        continue
                    foxglove.log("/robot/telemetry", message)
                    state["telemetry_received_at"] = time.time()
                    state.pop("telemetry_error", None)
        except Exception as exc:
            state["telemetry_error"] = str(exc)
            stop.wait(1)


def main():
    import foxglove

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--storage-profile", type=Path)
    parser.add_argument("--min-free-gib", type=float)
    parser.add_argument("--segment-seconds", type=float, default=300)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    from urllib.parse import urlsplit

    parsed = urlsplit(args.api_base)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        parser.error("api-base must be a local HTTP origin without credentials or a path")
    if args.segment_seconds < 5 or args.duration < 0:
        parser.error("segment must be at least five seconds; duration cannot be negative")
    try:
        storage = storage_module.load_storage(
            ROOT,
            output=args.output,
            state_dir=args.state_dir,
            profile=args.storage_profile,
            min_free_gib=args.min_free_gib,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output = storage.output
    os.umask(0o077)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *unused: stop.set())
    state = {
        "pid": os.getpid(),
        "active": False,
        "ready": False,
        "uploads_enabled": False,
        "storage_path": str(storage.output),
        "state_dir": str(storage.state_dir),
        "min_free_gib": storage.min_free_gib,
    }
    writer = None
    active_path = None
    threads = []
    started = time.monotonic()
    segment_started = 0.0
    after_id = 0
    history_saved = False
    activity_id = None
    exit_code = 0
    try:
        storage.prepare()
        while not stop.is_set():
            now = time.monotonic()
            if args.duration and now - started >= args.duration:
                break
            storage.check()
            requested = (
                {"id": "explicit-diagnostic", "kind": "diagnostic", "label": "bounded capture"}
                if args.duration
                else activity_module.requested_activity(storage.state_dir)
            )
            requested_id = requested["id"] if requested else None
            if requested is not None and requested["kind"] == "episode":
                try:
                    robot = json.loads(fetch(args.api_base + "/api/robot/status"))
                    if robot.get("mode") != "teleop":
                        requested = None
                        requested_id = None
                except Exception:
                    requested = None
                    requested_id = None
            if writer is not None and activity_id != requested_id:
                try:
                    log_control_ticks(state, args.api_base)
                except Exception as exc:
                    state["control_error"] = str(exc)
                finalize_segment(writer, active_path, state, storage.state_dir)
                writer = None
                active_path = None
                state.update(active=False, bytes=0)
                state.pop("path", None)
                state.pop("activity_id", None)
                state.pop("activity_kind", None)
            if requested is None:
                state.update(ready=True, active=False, updated_at=time.time())
                try:
                    message = json.loads(fetch(args.api_base + "/api/robot/status"))
                    state["api_received_at"] = time.time()
                    state["robot_mode"] = message.get("mode")
                    state.pop("api_error", None)
                except Exception as exc:
                    state["api_error"] = str(exc)
                publish_status(storage.state_dir / "status.json", state)
                stop.wait(0.2)
                continue
            if writer is None or now - segment_started >= args.segment_seconds:
                new_activity = activity_id != requested_id
                if writer is not None:
                    finalize_segment(writer, active_path, state, storage.state_dir)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                active_path = args.output / f"{stamp}-{uuid.uuid4().hex[:8]}.mcap.active"
                writer = foxglove.open_mcap(active_path)
                segment_started = now
                activity_id = requested_id
                if new_activity:
                    state.update(control_gap_count=0, control_ticks_recorded=0)
                    state.pop("control_received_at", None)
                state.update(
                    ready=True,
                    active=True,
                    path=str(active_path),
                    started_at=time.time(),
                    activity_id=activity_id,
                    activity_kind=requested["kind"],
                )
                foxglove.log("/recording/activity", {**requested, "phase": "opened"})
                for topic, record in viewer.saved_records(storage.state_dir.parent).items():
                    foxglove.log(topic, record)
                try:
                    log_control_ticks(state, args.api_base, seed=new_activity)
                except Exception as exc:
                    state["control_error"] = str(exc)
                foxglove.log("/recording/metadata", segment_metadata(state, requested, active_path))
                log_episodes(storage.state_dir, episode_id=activity_id)
                print(f"RECORDING {active_path}", flush=True)
            if not threads:
                threads = [
                    threading.Thread(
                        target=telemetry_publisher, args=(stop, state, args.api_base), daemon=True
                    ),
                    threading.Thread(
                        target=viewer.simulation_publisher,
                        args=(stop, "encoders", args.api_base),
                        daemon=True,
                    ),
                ]
                for name, port in (
                    ("obsbot", 8765),
                    ("camera_1", 8768),
                    ("camera_2", 8769),
                    ("realsense", 8770),
                ):
                    threads.append(
                        threading.Thread(
                            target=camera_publisher,
                            args=(name, f"http://127.0.0.1:{port}", stop),
                            daemon=True,
                        )
                    )
                threads.append(
                    threading.Thread(
                        target=camera_publisher,
                        args=("obsbot_cv", args.api_base + "/api/mcap/vision", stop),
                        daemon=True,
                    )
                )
                for thread in threads:
                    thread.start()
            for name, route in (
                ("status", "/api/robot/status"),
                ("health", "/api/robot/health"),
                ("vision", "/api/mcap/vision/status"),
            ):
                try:
                    message = json.loads(fetch(args.api_base + route))
                    foxglove.log(f"/robot/{name}", message)
                    state[f"{name}_received_at"] = time.time()
                    state.pop(f"{name}_error", None)
                except Exception as exc:
                    state[f"{name}_error"] = str(exc)
            try:
                events = json.loads(
                    fetch(f"{args.api_base}/api/events?limit=500&after_id={after_id}")
                )["events"]
                if not history_saved:
                    foxglove.log("/robot/event_history", {"historical": True, "events": events})
                    after_id = max((event["id"] for event in events), default=0)
                    history_saved = True
                else:
                    for event in sorted(events, key=lambda item: item["id"]):
                        foxglove.log("/robot/events", event)
                        after_id = event["id"]
                state["last_event_id"] = after_id
                state.pop("events_error", None)
            except Exception as exc:
                state["events_error"] = str(exc)
            log_episodes(storage.state_dir, episode_id=activity_id)
            try:
                log_control_ticks(state, args.api_base)
            except Exception as exc:
                state["control_error"] = str(exc)
            writer.flush()
            state.update(updated_at=time.time(), bytes=active_path.stat().st_size)
            publish_status(storage.state_dir / "status.json", state)
            stop.wait(0.2)
    except (storage_module.StorageUnavailable, OSError) as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["storage_blocked"] = True
        exit_code = 78
        print(state["error"], flush=True)
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=4)
        if writer is not None:
            try:
                finalize_segment(writer, active_path, state, storage.state_dir)
            except Exception as exc:
                state["close_error"] = f"{type(exc).__name__}: {exc}"
                exit_code = 78
        state.update(ready=False, active=False, updated_at=time.time())
        try:
            publish_status(storage.state_dir / "status.json", state)
        except OSError as exc:
            print(f"Recorder status could not be saved: {exc}", flush=True)
            exit_code = 78
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

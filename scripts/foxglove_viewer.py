"""Local camera and saved-evidence publisher. Never opens a motor bus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMERA = "http://127.0.0.1:8765"
OPERATIONAL_TOPICS = {
    "/robot/status": "/api/robot/status",
    "/robot/health": "/api/robot/health",
    "/control/state": "/api/control/state",
    "/recording/status": "/api/mcap/status",
    "/diagnostics/preflight": "/api/preflight/latest",
}


def operational_snapshot(api_base="http://127.0.0.1:8000"):
    records = {}
    for topic, path in OPERATIONAL_TOPICS.items():
        try:
            record = json.loads(fetch(api_base + path))
            record["bridge"] = {"available": True, "received_at_ns": time.time_ns()}
        except Exception as exc:
            record = {
                "bridge": {"available": False, "error": str(exc), "received_at_ns": time.time_ns()}
            }
        records[topic] = record
    return records


def operational_publisher(stop, api_base="http://127.0.0.1:8000"):
    import foxglove

    while not stop.is_set():
        for topic, record in operational_snapshot(api_base).items():
            foxglove.log(topic, record)
        stop.wait(1)


def control_publisher(stop, api_base="http://127.0.0.1:8000"):
    import foxglove

    cursor = None
    sequence = 0
    while not stop.is_set():
        try:
            suffix = "&latest_only=true" if cursor is None else ""
            batch = json.loads(
                fetch(f"{api_base}/api/mcap/control-ticks?after_seq={sequence}{suffix}")
            )
            if cursor is not None and cursor != batch["stream_id"]:
                sequence = 0
                cursor = None
                continue
            for tick in batch["ticks"]:
                foxglove.log("/robot/control_tick", tick, log_time=tick["timestamp_ns"])
            sequence = batch["last_seq"]
            cursor = batch["stream_id"]
            foxglove.log(
                "/robot/control_capture",
                {
                    "available": True,
                    "stream_id": cursor,
                    "last_seq": sequence,
                    "missing_before_buffer": batch["missing_before_buffer"],
                },
            )
        except Exception as exc:
            foxglove.log("/robot/control_capture", {"available": False, "error": str(exc)})
            stop.wait(1)
        stop.wait(0.1)


def saved_records(data_dir: Path) -> dict:
    sources = {
        "/calibration/installed": data_dir / "calibration_follower.json",
        "/calibration/installed_leader": data_dir / "calibration_leader.json",
        "/calibration/rest_pose": data_dir / "rest_pose.json",
    }
    references = list((data_dir / "shoulder_reference_checks").glob("*/reference.json"))
    if references:
        sources["/calibration/shoulder_reference"] = max(
            references, key=lambda path: path.stat().st_mtime
        )
    records = {}
    for topic, path in sources.items():
        record = {
            "source_kind": "saved_evidence",
            "live_joint_telemetry": False,
            "path": str(path),
            "available": False,
        }
        try:
            contents = path.read_bytes()
            record.update(
                data=json.loads(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
                available=True,
                file_modified_at=path.stat().st_mtime,
            )
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
        records[topic] = record
    return records


def fresh_image(fetch, base=CAMERA):
    status = json.loads(fetch(base + "/status.json"))
    if not status.get("live") or status.get("age_seconds", 999) >= 2:
        raise RuntimeError("Camera has no fresh frame")
    return fetch(base + "/snapshot.jpg"), status


def fetch(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.read()


def timestamped_image(base):
    status = json.loads(fetch(base + "/status.json"))
    if not status.get("live") or status.get("age_seconds", 999) >= 2:
        raise RuntimeError("Camera has no fresh frame")
    with urllib.request.urlopen(base + "/snapshot.jpg", timeout=2) as response:
        jpeg = response.read()
        captured = int(response.headers.get("X-SO101-Captured-At-Ns", 0))
        sequence = response.headers.get("X-SO101-Sequence")
    if sequence is not None:
        status["sequence"] = int(sequence)
    status["capture_timestamp_ns"] = captured or time.time_ns()
    status["timestamp_source"] = "host_camera_read_completed" if captured else "recorder_arrival"
    return jpeg, status


def camera_publisher(name, base, stop):
    import foxglove
    from foxglove.channels import CompressedImageChannel
    from foxglove.messages import CompressedImage, Timestamp

    channel = CompressedImageChannel(f"/camera/{name}/image")
    last_sequence = None
    while not stop.is_set():
        started = time.monotonic()
        try:
            jpeg, status = timestamped_image(base)
            stamp = status["capture_timestamp_ns"]
            if status["sequence"] != last_sequence:
                channel.log(
                    CompressedImage(
                        timestamp=Timestamp(stamp // 10**9, stamp % 10**9),
                        frame_id=f"{name}_optical",
                        format="jpeg",
                        data=jpeg,
                    ),
                    log_time=stamp,
                )
                last_sequence = status["sequence"]
            foxglove.log(f"/camera/{name}/status", status)
            if name == "obsbot":
                foxglove.log("/camera/status", status)
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            foxglove.log(f"/camera/{name}/status", {"live": False, "error": str(exc)})
        stop.wait(max(0, 1 / 30 - (time.monotonic() - started)))


class Assets(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):
        pass


def simulation_publisher(stop, source="simulation", api_base="http://127.0.0.1:8000"):
    import foxglove
    from foxglove.channels import SceneUpdateChannel
    from foxglove.messages import (
        Color,
        Duration,
        ModelPrimitive,
        Pose,
        Quaternion,
        SceneEntity,
        SceneUpdate,
        Timestamp,
        Vector3,
    )

    prefix = "/robot/encoder" if source == "encoders" else "/simulation"
    frame_id = "robot_base" if source == "encoders" else "sim_base"
    channel = SceneUpdateChannel(prefix + "/scene")
    environment = os.environ | {"SO101_BACKEND": "sim", "SO101_MOCK": "0"}
    worker = subprocess.Popen(
        [
            str(ROOT / ".venv/bin/python"),
            "-I",
            str(ROOT / "scripts/foxglove_simulation.py"),
            "--source",
            source,
            "--api-base",
            api_base,
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert worker.stdout is not None
        for line in worker.stdout:
            if stop.is_set():
                break
            snapshot = json.loads(line)
            stamp = time.time_ns()
            timestamp = Timestamp(stamp // 10**9, stamp % 10**9)
            entities = []
            for item in snapshot.pop("models"):
                position = item["position"]
                quat = item["quaternion_wxyz"]
                rgba = item["rgba"]
                entities.append(
                    SceneEntity(
                        timestamp=timestamp,
                        frame_id=frame_id,
                        id=item["id"],
                        lifetime=Duration(0, 500_000_000),
                        models=[
                            ModelPrimitive(
                                pose=Pose(
                                    position=Vector3(x=position[0], y=position[1], z=position[2]),
                                    orientation=Quaternion(
                                        w=quat[0], x=quat[1], y=quat[2], z=quat[3]
                                    ),
                                ),
                                scale=Vector3(x=1, y=1, z=1),
                                color=Color(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3]),
                                override_color=True,
                                url=f"http://127.0.0.1:8767/{Path(item['mesh']).stem}.glb",
                                media_type="model/gltf-binary",
                            )
                        ],
                    )
                )
            channel.log(SceneUpdate(entities=entities), log_time=stamp)
            foxglove.log(prefix + "/joints", snapshot)
            foxglove.log(
                prefix + "/status",
                {
                    "running": True,
                    "engine": "cached_encoders_mjcf" if source == "encoders" else "mujoco",
                    "source": source,
                    "live": snapshot.get("live", source == "simulation"),
                    "physical_alignment_verified": False,
                    "physical_arm_connected": False,
                },
            )
        if not stop.is_set():
            raise RuntimeError(f"MuJoCo worker stopped: {worker.poll()}")
    except Exception as exc:
        foxglove.log(
            prefix + "/status",
            {"running": False, "error": str(exc), "physical_arm_connected": False},
        )
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/real")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    import foxglove

    server = foxglove.start_server(
        name="SO101 read-only viewer",
        host="127.0.0.1",
        port=args.port,
        capabilities=[],
        services=[],
        message_backlog_size=1,
    )
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stop.set())
    print(f"Foxglove: ws://127.0.0.1:{args.port}; cameras + MuJoCo + SAVED evidence", flush=True)
    assets = ThreadingHTTPServer(
        ("127.0.0.1", 8767), partial(Assets, directory=str(ROOT / ".artifacts/foxglove-models"))
    )
    threads = [threading.Thread(target=assets.serve_forever, daemon=True)]
    threads.append(threading.Thread(target=operational_publisher, args=(stop,), daemon=True))
    threads.append(threading.Thread(target=control_publisher, args=(stop,), daemon=True))
    for name, port in (
        ("obsbot", 8765),
        ("camera_1", 8768),
        ("camera_2", 8769),
        ("realsense", 8770),
    ):
        threads.append(
            threading.Thread(
                target=camera_publisher, args=(name, f"http://127.0.0.1:{port}", stop), daemon=True
            )
        )
    threads.append(threading.Thread(target=simulation_publisher, args=(stop,), daemon=True))
    threads.append(
        threading.Thread(target=simulation_publisher, args=(stop, "encoders"), daemon=True)
    )
    threads.append(
        threading.Thread(
            target=camera_publisher,
            args=("obsbot_cv", "http://127.0.0.1:8000/api/mcap/vision", stop),
            daemon=True,
        )
    )
    for thread in threads:
        thread.start()
    next_records = 0.0
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now >= next_records:
                for topic, record in saved_records(args.data_dir).items():
                    foxglove.log(topic, record)
                next_records = now + 1
            stop.wait(0.2)
    finally:
        stop.set()
        assets.shutdown()
        assets.server_close()
        for thread in threads:
            thread.join(timeout=6)
        server.stop()


if __name__ == "__main__":
    main()

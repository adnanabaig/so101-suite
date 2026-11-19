"""Local OBSBOT preview relay. Single camera reader; no robot, audio or cloud access."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

DEVICE = "/dev/v4l/by-id/usb-Remo_Tech_Co.__Ltd._OBSBOT_Tiny_2_Lite-video-index0"
PAGE = b"""<!doctype html><html><head><meta name="viewport" content="width=device-width">
<title>SO101 arm camera</title><style>body{margin:0;background:#10141c;color:#eee;
font:18px sans-serif}img{width:100%;height:90vh;object-fit:contain}p{margin:12px}</style>
</head><body><img id="feed" alt="Waiting for OBSBOT camera"><p id="status">Connecting...</p>
<script>
const feed=document.getElementById('feed'),label=document.getElementById('status');
let connected=false;
async function update(){try{const r=await fetch('/status.json',{cache:'no-store'});
const s=await r.json();label.textContent=s.live?'Camera | live | '+s.width+'x'+s.height+
' | '+s.actual_preview_fps.toFixed(1)+' fps measured / '+s.requested_preview_fps+' requested':
'Camera unavailable / reconnecting';
if(s.live&&!connected){feed.src='/stream.mjpg?t='+Date.now();connected=true;}
if(!s.live){connected=false;feed.removeAttribute('src');}}
catch(e){connected=false;feed.removeAttribute('src');
label.textContent='Camera service unavailable';}}
feed.onerror=()=>{connected=false;};setInterval(update,1000);update();
</script></body></html>"""


class Frames:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.updated = 0.0
        self.sequence = 0
        self.captured_at_ns = 0
        self.width = self.height = 0
        self.error: str | None = "Camera starting"
        self.frame_times: deque[float] = deque(maxlen=120)
        self.capture_times: deque[float] = deque(maxlen=240)
        self.requested_capture_fps = 0
        self.requested_preview_fps = 0
        self.device_reported_fps = 0.0
        self.controls = {"controls": {}, "profile_error": "Camera controls starting"}
        self.control_requests = queue.Queue()
        self.image_quality = {}

    @staticmethod
    def measured_rate(timestamps, live):
        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        return (len(timestamps) - 1) / elapsed if elapsed and live else 0.0

    def snapshot(self):
        with self.condition:
            return self.jpeg if time.monotonic() - self.updated < 2.0 else None

    def packet(self):
        with self.condition:
            if self.jpeg is None or time.monotonic() - self.updated >= 2:
                return None, {}
            return self.jpeg, {
                "X-SO101-Sequence": str(self.sequence),
                "X-SO101-Captured-At-Ns": str(self.captured_at_ns),
            }

    def status(self):
        with self.condition:
            age = time.monotonic() - self.updated if self.updated else None
            live = self.jpeg is not None and age is not None and age < 2.0
            measured = self.measured_rate(self.frame_times, live)
            return {
                "live": live,
                "age_seconds": age,
                "sequence": self.sequence,
                "width": self.width,
                "height": self.height,
                "error": self.error,
                "requested_capture_fps": self.requested_capture_fps,
                "requested_preview_fps": self.requested_preview_fps,
                "device_reported_fps": self.device_reported_fps,
                "actual_capture_fps": self.measured_rate(self.capture_times, live),
                "actual_preview_fps": measured,
                "fps_target_met": live and measured >= self.requested_preview_fps * 0.95,
                "camera_profile": self.controls,
                "image_quality": self.image_quality,
            }


class RealSenseColorCapture:
    def __init__(self, width, height, fps, serial=None):
        import pyrealsense2 as rs

        context = rs.context()
        devices = list(context.query_devices())
        if serial is not None:
            if not serial:
                raise ValueError("RealSense serial selector must not be empty")
            devices = [
                device
                for device in devices
                if device.get_info(rs.camera_info.serial_number) == serial
            ]
        if len(devices) != 1:
            raise RuntimeError(
                "RealSense selection must match exactly one connected camera; "
                "use --device realsense:SERIAL when multiple cameras are connected"
            )
        self.pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(devices[0].get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        self.fps = fps
        self.first = True

    def isOpened(self):
        return True

    def set(self, *args):
        return False

    def get(self, property_id):
        return self.fps if property_id == cv2.CAP_PROP_FPS else 0

    def read(self):
        import numpy as np

        frames = self.pipeline.wait_for_frames(10000 if self.first else 2000)
        self.first = False
        color = frames.get_color_frame()
        return (True, np.asanyarray(color.get_data())) if color else (False, None)

    def release(self):
        self.pipeline.stop()


def producer(
    frames: Frames,
    stop: threading.Event,
    device: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    preview_fps: int = 15,
    profile_directory: Path | None = None,
):
    while not stop.is_set():
        camera = None
        driver = None
        try:
            is_realsense = device == "realsense" or device.startswith("realsense:")
            camera = (
                RealSenseColorCapture(
                    width, height, fps, device.partition(":")[2] if ":" in device else None
                )
                if is_realsense
                else cv2.VideoCapture(device, cv2.CAP_V4L2)
            )
            if not camera.isOpened():
                raise RuntimeError("Camera not connected or camera already in use")
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            camera.set(cv2.CAP_PROP_FPS, fps)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            from so101.hardware.camera_controls import (
                CameraProfile,
                RealSenseControls,
                V4L2Controls,
            )

            try:
                driver = RealSenseControls(camera) if is_realsense else V4L2Controls(device)
                profile = CameraProfile(
                    driver,
                    profile_directory
                    or Path(__file__).resolve().parents[1] / "data/real/camera-profiles",
                )
                frames.controls = profile.snapshot()
            except Exception as exc:
                profile = None
                frames.controls = {"controls": {}, "profile_error": str(exc)}
            with frames.condition:
                frames.requested_capture_fps = fps
                frames.requested_preview_fps = preview_fps
                frames.device_reported_fps = camera.get(cv2.CAP_PROP_FPS)
                frames.frame_times.clear()
                frames.capture_times.clear()
            next_encoded = 0.0
            while not stop.is_set():
                while not frames.control_requests.empty():
                    request = frames.control_requests.get_nowait()
                    if request["cancelled"].is_set():
                        request["done"].set()
                        continue
                    try:
                        if profile is None:
                            raise ValueError("Camera controls are unavailable")
                        frames.controls = profile.apply(request["changes"])
                        request["result"] = frames.controls
                    except Exception as exc:
                        request["error"] = str(exc)
                    finally:
                        request["done"].set()
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed; reconnecting")
                now = time.monotonic()
                captured_at_ns = time.time_ns()
                with frames.condition:
                    frames.capture_times.append(now)
                    while frames.capture_times and now - frames.capture_times[0] > 4:
                        frames.capture_times.popleft()
                if now + 0.001 < next_encoded:
                    continue
                ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError("Camera JPEG encode failed")
                with frames.condition:
                    frames.jpeg = data.tobytes()
                    frames.height, frames.width = frame.shape[:2]
                    frames.updated = now
                    frames.captured_at_ns = captured_at_ns
                    frames.sequence += 1
                    frames.frame_times.append(now)
                    while frames.frame_times and now - frames.frame_times[0] > 4:
                        frames.frame_times.popleft()
                    frames.error = None
                    if frames.sequence % 30 == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames.image_quality = {
                            "mean_luma": float(gray.mean()),
                            "dark_fraction": float((gray < 8).mean()),
                            "clipped_fraction": float((gray > 247).mean()),
                        }
                        if profile is not None:
                            frames.controls = profile.snapshot()
                    frames.condition.notify_all()
                next_encoded = max(next_encoded + 1 / preview_fps, now)
        except Exception as exc:
            with frames.condition:
                frames.jpeg = None
                frames.error = str(exc)
                frames.condition.notify_all()
            print(str(exc), flush=True)
        finally:
            if driver is not None:
                driver.close()
            if camera is not None:
                try:
                    camera.release()
                except Exception as exc:
                    with frames.condition:
                        frames.jpeg = None
                        frames.error = f"Camera cleanup failed; reconnecting: {exc}"
                        frames.condition.notify_all()
                    print(str(exc), flush=True)
        stop.wait(1)


def handler_for(frames: Frames, stop: threading.Event):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, content_type, body, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Resource-Policy", "same-site")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    self.send(200, "text/html; charset=utf-8", PAGE)
                elif path == "/status.json":
                    self.send(200, "application/json", json.dumps(frames.status()).encode())
                elif path == "/snapshot.jpg":
                    frame, headers = frames.packet()
                    self.send(
                        200 if frame else 503,
                        "image/jpeg" if frame else "text/plain",
                        frame or b"No fresh camera frame",
                        headers,
                    )
                elif path == "/stream.mjpg":
                    if frames.snapshot() is None:
                        self.send(503, "text/plain", b"No fresh camera frame")
                        return
                    self.connection.settimeout(5)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Cross-Origin-Resource-Policy", "same-site")
                    self.end_headers()
                    seen = -1
                    while not stop.is_set():
                        with frames.condition:
                            frames.condition.wait_for(
                                lambda previous=seen: frames.sequence != previous or stop.is_set(),
                                timeout=1,
                            )
                            frame = frames.snapshot()
                            seq = frames.sequence
                        if frame is None:
                            break
                        if seq == seen:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode()
                            + b"\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                        self.wfile.flush()
                        seen = seq
                    self.close_connection = True
                else:
                    self.send(404, "text/plain", b"Not found")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True

        def do_POST(self):
            if (
                self.path != "/controls"
                or self.headers.get("X-SO101-Control") != "camera-settings"
                or self.headers.get("Origin")
            ):
                self.send(
                    403, "application/json", b'{"error":"Use the workstation camera settings"}'
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid camera request length")
                changes = json.loads(self.rfile.read(length))
                request = {
                    "changes": changes,
                    "done": threading.Event(),
                    "cancelled": threading.Event(),
                }
                frames.control_requests.put(request)
                if not request["done"].wait(3):
                    request["cancelled"].set()
                    raise ValueError(
                        "Camera did not acknowledge controls; verify readback before retrying"
                    )
                if request.get("error"):
                    raise ValueError(request["error"])
                self.send(200, "application/json", json.dumps(request["result"]).encode())
            except (ValueError, TypeError) as exc:
                self.send(422, "application/json", json.dumps({"error": str(exc)}).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-fps", type=int, choices=(5, 15, 30), default=15)
    args = parser.parse_args()
    frames, stop = Frames(), threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(frames, stop))
    thread = threading.Thread(
        target=producer,
        args=(frames, stop, args.device, args.width, args.height, args.fps, args.preview_fps),
        daemon=True,
    )
    thread.start()

    def terminate(*args):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, terminate)
    print(f"Arm camera: http://127.0.0.1:{args.port} (local only; audio disabled)", flush=True)
    try:
        server.serve_forever()
    finally:
        stop.set()
        with frames.condition:
            frames.condition.notify_all()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()

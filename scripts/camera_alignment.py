"""Camera-only phone target guide. Reads existing relays; never opens motor buses."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import secrets
import socket
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MARKER_ID = 42
PHONE_PPI = 460
PHONE_CSS_MARKER = 340
TABLE_HEIGHT_MM = 29.125 * 25.4
CAMERA_BOTTOM_HEIGHT_MM = 53.5 * 25.4


def phone_width(data: dict) -> float:
    values = [
        float(data[key])
        for key in (
            "screen_width",
            "screen_height",
            "dpr",
            "scale",
            "viewport_width",
            "marker_css_width",
        )
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid phone display report")
    width, height, dpr, scale, viewport, marker = values
    if (
        abs(width * dpr - 1290) > 1
        or abs(height * dpr - 2796) > 1
        or abs(dpr - 3) > 0.001
        or abs(scale - 1) > 0.001
        or abs(viewport - width) > 1
        or abs(marker - PHONE_CSS_MARKER) > 0.1
    ):
        raise ValueError("Use iPhone 14 Pro Max in portrait at standard display / 100% page zoom")
    return marker * dpr / PHONE_PPI * 25.4


def make_target(path: Path) -> None:
    sheet = np.full((2796, 1290), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sheet[850:1900, 120:1170] = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, 1050)
    for label, position, scale in [
        ("SO101 CAMERA ALIGNMENT", (75, 210), 1.9),
        ("TOP - marker 42", (190, 650), 2.1),
        ("Measure BLACK square width", (85, 2130), 1.8),
        ("Do not measure the white margin", (65, 2240), 1.6),
        ("Keep this image still and uncropped", (45, 2470), 1.5),
    ]:
        cv2.putText(sheet, label, position, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 3, cv2.LINE_AA)
    cv2.arrowedLine(sheet, (120, 2040), (1170, 2040), 0, 6, tipLength=0.025)
    cv2.arrowedLine(sheet, (1170, 2040), (120, 2040), 0, 6, tipLength=0.025)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError("Could not write target")


def table_from_marker(
    width_mm: float, x_mm: float, y_mm: float, height_mm: float, yaw_deg: float
) -> np.ndarray:
    values = [width_mm, x_mm, y_mm, height_mm, yaw_deg]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Measurements must be finite")
    if not 20 <= width_mm <= 100 or not 0 < height_mm <= 100:
        raise ValueError("Enter square width 20–100 mm and screen height above table 0–100 mm")
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = np.array([[cosine, sine, 0], [sine, -cosine, 0], [0, 0, -1]])
    transform[:3, 3] = np.array([x_mm, y_mm, height_mm]) / 1000
    return transform


def pose_candidates(corners: np.ndarray, width_mm: float, profile: dict) -> list[dict]:
    if not math.isfinite(width_mm) or not 20 <= width_mm <= 100:
        raise ValueError("Measure the displayed black square in mm first")
    if any(abs(value) > 1e-12 for value in profile["coefficients"]):
        raise ValueError("This guide requires zero-distortion intrinsics; camera model unsupported")
    width = width_mm / 1000
    points = np.array([[0, 0, 0], [width, 0, 0], [width, width, 0], [0, width, 0]], float)
    matrix = np.array(
        [[profile["fx"], 0, profile["ppx"]], [0, profile["fy"], profile["ppy"]], [0, 0, 1]], float
    )
    result = cv2.solvePnPGeneric(
        points, corners.astype(float), matrix, None, flags=cv2.SOLVEPNP_IPPE
    )
    candidates = []
    for rotation, translation in zip(result[1], result[2], strict=True):
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rotation)[0]
        transform[:3, 3] = translation.ravel()
        camera_points = (transform[:3, :3] @ points.T).T + translation.ravel()
        if np.any(camera_points[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(points, rotation, translation, matrix, None)[0].reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
        candidates.append({"camera_from_marker": transform.tolist(), "rms_px": error})
    return sorted(candidates, key=lambda candidate: candidate["rms_px"])


def summarize_samples(samples: list[dict], measurements: dict) -> dict:
    quick = measurements.get("mode") == "phone_reference"
    if not quick and not measurements.get("level_confirmed"):
        raise ValueError("Level the screen with a stable shim and confirm its measured height")
    placement = (
        None
        if quick
        else table_from_marker(
            *[
                float(measurements[key])
                for key in ("width_mm", "x_mm", "y_mm", "height_mm", "yaw_deg")
            ]
        )
    )
    if len(samples) < 15 or samples[-1]["at"] - samples[0]["at"] < 2:
        raise ValueError("Keep the marker still for at least two seconds / 15 fresh frames")
    if time.monotonic() - samples[-1]["at"] > 1:
        raise ValueError("Camera/marker is stale")
    corners = np.array([sample["corners"] for sample in samples])
    spread = float(np.max(np.linalg.norm(corners - np.median(corners, axis=0), axis=2)))
    if spread > 2:
        raise ValueError("Marker moved; keep phone still until corner jitter is below 2 pixels")
    candidates = [sample["candidates"] for sample in samples]
    if any(not item or item[0]["rms_px"] > 1.5 for item in candidates):
        raise ValueError("Pose fit is poor; improve lighting or move target closer")
    best = samples[-1]
    camera_from_marker = np.array(best["candidates"][0]["camera_from_marker"])
    alternatives = best["candidates"]
    ambiguous = len(alternatives) > 1 and alternatives[1]["rms_px"] < max(
        alternatives[0]["rms_px"] * 2, alternatives[0]["rms_px"] + 0.3
    )
    return {
        "table_from_camera": None
        if quick
        else (placement @ np.linalg.inv(camera_from_marker)).tolist(),
        "table_from_marker": None if placement is None else placement.tolist(),
        "reference_from_camera": (
            (np.diag([1.0, -1.0, -1.0, 1.0]) if placement is None else placement)
            @ np.linalg.inv(camera_from_marker)
        ).tolist(),
        "reference_frame": "phone_screen" if quick else "measured_table",
        "screen_height_above_table_mm": None if quick else measurements["height_mm"],
        "screen_height_status": "not_measured" if quick else "operator_measured",
        "measurements": measurements,
        "corner_jitter_px": spread,
        "reprojection_rms_px": alternatives[0]["rms_px"],
        "planar_ambiguity": ambiguous,
        "pose_candidates": alternatives,
        "sample_count": len(samples),
        "reference_pixels": best["corners"],
        "frame_sequence": best.get("sequence"),
        "captured_at_ns": best.get("captured_at_ns"),
        "status": "candidate_only",
        "robot_base_alignment_verified": False,
        "independent_validation_required": True,
    }


def compare_placements(first: dict, second: dict) -> dict:
    if first.get("table_from_camera") is None or second.get("table_from_camera") is None:
        raise ValueError(
            "Quick captures use phone reference frames. Comparing table alignment "
            "requires measured placements in Advanced settings."
        )
    if first["intrinsics_sha256"] != second["intrinsics_sha256"]:
        raise ValueError("Camera intrinsics changed between placements")
    if first["serial"] != second["serial"]:
        raise ValueError("Placements must use the same camera")
    a, b = np.array(first["table_from_camera"]), np.array(second["table_from_camera"])
    separation = float(
        np.linalg.norm(
            np.array(first["table_from_marker"])[:2, 3]
            - np.array(second["table_from_marker"])[:2, 3]
        )
        * 1000
    )
    position_error = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    angle_error = math.degrees(math.acos(float(cosine)))
    consistent = separation >= 100 and position_error <= 10 and angle_error <= 2
    return {
        "placement_separation_mm": separation,
        "camera_position_difference_mm": position_error,
        "camera_rotation_difference_deg": angle_error,
        "within_provisional_tolerance": consistent,
        "planar_ambiguity": first["planar_ambiguity"] or second["planar_ambiguity"],
        "robot_base_alignment_verified": False,
        "message": "Compare candidates only: use placements at least 100 mm apart; "
        "provisional consistency thresholds are 10 mm / 2 degrees. "
        "Independent known-distance and robot-base checks are still required.",
    }


def frame_identity(headers, now_ns: int) -> tuple[int, int]:
    sequence = int(headers["X-SO101-Sequence"])
    captured_ns = int(headers["X-SO101-Captured-At-Ns"])
    if not 0 <= (now_ns - captured_ns) / 1e9 <= 0.5:
        raise ValueError("Snapshot capture timestamp is stale or in the future")
    return sequence, captured_ns


def local_request(host: str | None, origin: str | None, port: int) -> bool:
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    return host in allowed and (origin is None or origin == f"http://{host}")


class Guide:
    def __init__(self, intrinsics: Path, output: Path):
        self.source = intrinsics
        self.output = output
        self.provenance = hashlib.sha256(intrinsics.read_bytes()).hexdigest()
        data = json.loads(intrinsics.read_text())
        device = next(device for device in data["devices"] if device["serial"] == "347622075852")
        self.profile = device["streams"]["color"]
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.jpeg = b""
        self.status = {"message": "Waiting for camera", "ready": False}
        self.measurements: dict = {}
        self.configuration_id = ""
        self.samples: list[dict] = []
        self.saved: list[dict] = []
        self.last_sequence: int | None = None
        self.last_captured_ns = 0
        self.last_fresh = 0.0
        self.phone: dict = {}
        self.phone_updated = 0.0
        self.phone_url = ""
        self.phone_error: dict = {}
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters(),
        )

    def configure(self, data: dict) -> str:
        for key in ("width_mm", "x_mm", "y_mm", "height_mm", "yaw_deg"):
            data[key] = float(data[key])
        table_from_marker(
            *[data[key] for key in ("width_mm", "x_mm", "y_mm", "height_mm", "yaw_deg")]
        )
        if data.get("level_confirmed") is not True:
            raise ValueError("Confirm screen level and height before measuring")
        with self.lock:
            self.measurements = data
            self.configuration_id = secrets.token_urlsafe(12)
            self.samples.clear()
            return self.configuration_id

    def report_phone(self, data: dict) -> dict:
        try:
            width = phone_width(data)
        except (ValueError, KeyError, TypeError) as error:
            with self.lock:
                self.phone_error = {
                    "message": str(error),
                    "display": data,
                    "received_at": time.time(),
                }
            raise
        with self.lock:
            self.phone_error = {}
            self.phone = {
                "model": "iPhone 14 Pro Max",
                "width_mm": width,
                "size_source": "controlled Safari page / Apple nominal 460 ppi",
                "display": data,
            }
            self.phone_updated = time.monotonic()
            if not self.measurements or self.measurements.get("mode") == "phone_reference":
                if self.measurements.get("width_mm") != width:
                    self.configuration_id = secrets.token_urlsafe(12)
                    self.samples.clear()
                self.measurements = {
                    "mode": "phone_reference",
                    "width_mm": width,
                    "phone_model": "iPhone 14 Pro Max",
                    "width_source": self.phone["size_source"],
                    "nominal_body_thickness_mm": 7.85,
                    "table_height_from_floor_mm": TABLE_HEIGHT_MM,
                    "camera_bottom_from_floor_mm": CAMERA_BOTTOM_HEIGHT_MM,
                    "camera_bottom_above_table_mm": CAMERA_BOTTOM_HEIGHT_MM - TABLE_HEIGHT_MM,
                    "camera_height_reference": "bottom of housing; approximate",
                }
        return {"ok": True, "width_mm": width}

    def frame(self) -> None:
        with urllib.request.urlopen("http://127.0.0.1:8770/status.json", timeout=2) as response:
            status = json.load(response)
        if status.get("age_seconds", 999) > 0.5 or not status.get("live"):
            raise ValueError("D435i relay is stale or offline")
        if status.get("camera_profile", {}).get("identity") != "realsense:347622075852":
            raise ValueError("Relay camera identity does not match the D435i intrinsics")
        with urllib.request.urlopen("http://127.0.0.1:8770/snapshot.jpg", timeout=2) as response:
            payload = response.read()
            sequence, captured_ns = frame_identity(response.headers, time.time_ns())
        if sequence == self.last_sequence or captured_ns <= self.last_captured_ns:
            if time.monotonic() - self.last_fresh > 0.5:
                raise ValueError("D435i snapshot stopped updating")
            return
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode D435i image")
        height, width = frame.shape[:2]
        if (width, height) != (self.profile["width"], self.profile["height"]):
            raise ValueError("Frame dimensions differ from saved intrinsics")
        self.last_sequence = sequence
        self.last_captured_ns = captured_ns
        self.last_fresh = time.monotonic()
        corners, ids, _ = self.detector.detectMarkers(frame)
        message = "Place phone face-up in the dashed guide, TOP toward table +Y"
        found = False
        with self.lock:
            measurements = self.measurements.copy()
            configuration_id = self.configuration_id
            phone_recent = time.monotonic() - self.phone_updated < 5
        if measurements.get("mode") == "phone_reference" and not phone_recent:
            raise ValueError("Open the phone target page and leave it visible")
        sample = None
        if ids is not None and MARKER_ID in ids.ravel():
            index = ids.ravel().tolist().index(MARKER_ID)
            quad = corners[index].reshape(4, 2)
            cv2.aruco.drawDetectedMarkers(frame, [corners[index]], np.array([[MARKER_ID]]))
            side = float(np.min(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)))
            inside = bool(
                np.all(quad > 12)
                and np.all(quad[:, 0] < width - 12)
                and np.all(quad[:, 1] < height - 12)
            )
            found = side >= 40 and inside
            message = f"Marker visible, shortest edge {side:.0f}px"
            if not found:
                message = "Need 40 pixels per side and image-edge clearance; use a larger target"
            elif measurements:
                candidates = pose_candidates(quad, measurements["width_mm"], self.profile)
                sample = {
                    "at": time.monotonic(),
                    "corners": quad.tolist(),
                    "candidates": candidates,
                    "sequence": sequence,
                    "captured_at_ns": captured_ns,
                }
            else:
                message = "Marker found. Open the phone target page to set its size automatically."
        for start, end in [
            ((int(width * 0.15), int(height * 0.15)), (int(width * 0.85), int(height * 0.85)))
        ]:
            cv2.rectangle(frame, start, end, (255, 200, 0), 1)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Could not encode guide frame")
        with self.lock:
            if configuration_id == self.configuration_id:
                if sample is None:
                    self.samples.clear()
                else:
                    self.samples.append(sample)
                    self.samples = self.samples[-30:]
            ready = False
            if self.samples:
                try:
                    result = summarize_samples(self.samples, self.measurements)
                    ready = True
                    message = "Phone is steady. Ready to capture."
                    if result["planar_ambiguity"]:
                        message += " Alignment will need an independent check."
                except ValueError as error:
                    message = str(error)
            self.jpeg = encoded.tobytes()
            self.status = {
                "message": message,
                "ready": ready,
                "marker_found": found,
                "sample_count": len(self.samples),
                "frame_sequence": sequence,
                "updated_at": time.time(),
            }

    def save(self, configuration_id: str) -> dict:
        with self.lock:
            if configuration_id != self.configuration_id:
                raise ValueError("Measurements changed; apply the current form again before saving")
            if self.measurements.get("mode") == "phone_reference" and (
                time.monotonic() - self.phone_updated >= 5
            ):
                raise ValueError("Phone target is no longer active; reopen its page")
            result = summarize_samples(self.samples, self.measurements)
            jpeg = self.jpeg
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = self.output / timestamp
        folder.mkdir(parents=True, mode=0o700)
        result.update(
            schema_version=1,
            camera="realsense",
            serial="347622075852",
            captured_at_utc=timestamp,
            intrinsics=self.profile,
            intrinsics_sha256=self.provenance,
            target={"dictionary": "DICT_4X4_50", "id": MARKER_ID},
            convention="Transforms map camera into the named reference; metres. "
            "Quick mode leaves the table transform unset until screen/table geometry is measured.",
        )
        (folder / "candidate.json").write_text(json.dumps(result, indent=2) + "\n")
        (folder / "frame.jpg").write_bytes(jpeg)
        with self.lock:
            self.saved.append(result)
        return {
            "path": str(folder / "candidate.json"),
            "status": result["status"],
            "planar_ambiguity": result["planar_ambiguity"],
            "reference_frame": result["reference_frame"],
        }

    def run(self) -> None:
        while True:
            try:
                self.frame()
            except Exception as error:
                with self.lock:
                    self.samples.clear()
                    self.status = {"message": str(error), "ready": False}
            time.sleep(0.1)


def serve_phone(guide: Guide, host: str, port: int, key: str | None = None) -> ThreadingHTTPServer:
    key = key or secrets.token_urlsafe(18)
    guide.phone_url = f"http://{host}:{port}/?key={key}"
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID, 1020
    )
    encoded = cv2.imencode(".png", marker)[1].tobytes()
    page = Path(__file__).with_name("camera_alignment_phone.html").read_text()
    page = page.replace("__KEY__", key).replace("__MARKER__", base64.b64encode(encoded).decode())

    class PhoneHandler(BaseHTTPRequestHandler):
        def reply(self, code: int, payload: bytes, mime: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path != "/" or parse_qs(parsed.query).get("key") != [key]:
                self.reply(404, b"Not found", "text/plain")
                return
            self.reply(200, page.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if (
                self.path != "/display"
                or self.headers.get("X-Target-Key") != key
                or self.headers.get("Origin") not in (None, f"http://{host}:{port}")
                or self.headers.get("Host") != f"{host}:{port}"
            ):
                self.reply(403, b"Forbidden", "text/plain")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048:
                    raise ValueError("Invalid display report")
                report = guide.report_phone(json.loads(self.rfile.read(size)))
                self.reply(200, json.dumps(report).encode(), "application/json")
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, json.dumps({"error": str(error)}).encode(), "application/json")

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer((host, port), PhoneHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def serve(
    guide: Guide, port: int, phone_host: str | None = None, phone_key: str | None = None
) -> None:
    phone_server = serve_phone(guide, phone_host, port + 1, phone_key) if phone_host else None

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, payload: bytes, mime: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if not local_request(self.headers.get("Host"), self.headers.get("Origin"), port):
                self.reply(403, b"Loopback access only", "text/plain")
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                content = Path(__file__).with_suffix(".html").read_text()
                qr = (
                    cast(Any, cv2).QRCodeEncoder_create().encode(guide.phone_url)
                    if guide.phone_url
                    else None
                )
                qr_uri = ""
                if qr is not None:
                    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
                    qr = cv2.resize(qr, (240, 240), interpolation=cv2.INTER_NEAREST)
                    qr_uri = (
                        "data:image/png;base64,"
                        + base64.b64encode(cv2.imencode(".png", qr)[1].tobytes()).decode()
                    )
                content = content.replace("__TOKEN__", guide.token)
                content = content.replace("__PHONE_URL__", guide.phone_url)
                content = content.replace("__PHONE_QR__", qr_uri)
                self.reply(200, content.encode(), "text/html")
            elif path == "/frame.jpg":
                with guide.lock:
                    image = guide.jpeg
                self.reply(200 if image else 503, image, "image/jpeg")
            elif path == "/status":
                with guide.lock:
                    status = guide.status.copy()
                    status.update(
                        configuration_id=guide.configuration_id,
                        mode=guide.measurements.get("mode", "measured_table"),
                        phone=guide.phone,
                        phone_error=guide.phone_error,
                        phone_connected=time.monotonic() - guide.phone_updated < 5,
                    )
                self.reply(200, json.dumps(status).encode(), "application/json")
            else:
                self.reply(404, b"Not found", "text/plain")

        def do_POST(self) -> None:
            if not local_request(self.headers.get("Host"), self.headers.get("Origin"), port) or (
                self.headers.get("X-Guide-Token") != guide.token
            ):
                self.reply(403, b"Invalid local session", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid body length")
                data = json.loads(self.rfile.read(length))
                if self.path == "/configure":
                    result = {"ok": True, "configuration_id": guide.configure(data)}
                elif self.path == "/save":
                    result = guide.save(data["configuration_id"])
                elif self.path == "/compare":
                    with guide.lock:
                        saved = guide.saved[-2:]
                    if len(saved) < 2:
                        raise ValueError("Save two separately measured placements first")
                    result = compare_placements(*saved)
                else:
                    self.reply(404, b"Not found", "text/plain")
                    return
                self.reply(200, json.dumps(result).encode(), "application/json")
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, json.dumps({"error": str(error)}).encode(), "application/json")

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=guide.run, daemon=True).start()
    print(f"Camera-only alignment guide: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if phone_server:
            phone_server.shutdown()
            phone_server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8782)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=ROOT / ".artifacts" / "vision-20261031" / "realsense-factory-calibration.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/real/camera_alignment")
    parser.add_argument("--target-only", type=Path)
    parser.add_argument("--phone-host", help="LAN IPv4 for the target-only phone page")
    parser.add_argument(
        "--phone-key", help="Preserve an existing target-only link during a restart"
    )
    args = parser.parse_args()
    if args.target_only:
        make_target(args.target_only)
    else:
        phone_host = args.phone_host
        if not phone_host:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("192.0.2.1", 9))
                phone_host = probe.getsockname()[0]
        serve(Guide(args.intrinsics, args.output), args.port, phone_host, args.phone_key)


if __name__ == "__main__":
    main()

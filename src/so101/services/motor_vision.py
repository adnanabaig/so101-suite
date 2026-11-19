"""Independent image observations using numbered tags, with no motion authority."""

from __future__ import annotations

import threading
import time
import urllib.request
from typing import Any

import numpy as np

from .. import JOINT_NAMES


def detect_motors(frame: np.ndarray) -> list[dict[str, Any]]:
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, identifiers, _ = detector.detectMarkers(frame)
    if identifiers is None:
        return []
    points = []
    duplicate_ids = {
        int(value) for value in identifiers.ravel() if list(identifiers.ravel()).count(value) > 1
    }
    for corner, identifier in zip(corners, identifiers.ravel(), strict=True):
        motor_id = int(identifier)
        if motor_id not in range(1, 7) or motor_id in duplicate_ids:
            continue
        center = corner.reshape(4, 2).mean(axis=0)
        points.append(
            {
                "id": motor_id,
                "name": JOINT_NAMES[motor_id - 1],
                "pixel": [float(center[0]), float(center[1])],
            }
        )
    return sorted(points, key=lambda point: point["id"])


def annotate(frame: np.ndarray, points: list[dict[str, Any]]) -> bytes:
    import cv2

    image = frame.copy()
    coordinates = {point["id"]: tuple(round(value) for value in point["pixel"]) for point in points}
    for motor_id, center in coordinates.items():
        if motor_id + 1 in coordinates:
            cv2.line(image, center, coordinates[motor_id + 1], (80, 220, 90), 2)
        cv2.circle(image, center, 7, (80, 220, 90), 2)
        cv2.putText(
            image,
            str(motor_id),
            (center[0] + 9, center[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 220, 90),
            2,
        )
    ok, jpeg = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode vision frame")
    return jpeg.tobytes()


class MotorVision:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._updated = 0.0
        self._state: dict[str, Any] = {"running": False, "detected": [], "error": None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = time.monotonic() - self._updated
            live = self._state["running"] and age < 1
            return {
                **self._state,
                "live": live,
                "age_seconds": age if self._updated else None,
                "detected": self._state["detected"] if live else [],
                "method": "aruco_4x4_50",
                "requires_numbered_motor_tags": True,
                "camera_projection_calibrated": False,
                "tag_centers_are_joint_pivots": False,
            }

    def snapshot(self) -> bytes | None:
        with self._lock:
            if not self._state["running"] or time.monotonic() - self._updated >= 1:
                return None
            return self._jpeg

    def start(self) -> None:
        import cv2

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("The camera environment needs OpenCV ArUco support")
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._state = {"running": True, "detected": [], "error": None}
            self._thread = threading.Thread(
                target=self._run, name="so101-motor-vision", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            self._state["running"] = False

    def _run(self) -> None:
        import cv2

        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8765/snapshot.jpg", timeout=2
                ) as response:
                    encoded = response.read(8 * 1024 * 1024)
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("OBSBOT did not return a decodable image")
                points = detect_motors(frame)
                jpeg = annotate(frame, points)
                with self._lock:
                    self._jpeg = jpeg
                    self._updated = time.monotonic()
                    self._state.update(
                        detected=points, error=None, width=frame.shape[1], height=frame.shape[0]
                    )
                    self._state["sequence"] = self._state.get("sequence", 0) + 1
            except (OSError, RuntimeError, cv2.error) as exc:
                with self._lock:
                    self._state.update(error=str(exc), detected=[])
                    self._updated = 0.0
            self._stop.wait(max(0, 0.1 - (time.monotonic() - started)))


vision = MotorVision()

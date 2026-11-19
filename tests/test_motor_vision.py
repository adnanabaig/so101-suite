"""Image detections identify tags independently of encoder positions."""

import cv2
import numpy as np
import pytest

from so101.services.motor_vision import MotorVision, annotate, detect_motors


def test_numbered_tags_are_detected_from_pixels_and_unrelated_ids_ignored():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((400, 1100, 3), 255, np.uint8)
    for index, tag in enumerate((6, 3, 1, 19)):
        marker = cv2.aruco.generateImageMarker(dictionary, tag, 160)
        image[100:260, 50 + index * 260 : 210 + index * 260] = cv2.cvtColor(
            marker, cv2.COLOR_GRAY2BGR
        )
    points = detect_motors(image)
    assert [point["id"] for point in points] == [1, 3, 6]
    assert [point["name"] for point in points] == ["shoulder_pan", "elbow_flex", "gripper"]
    assert points[0]["pixel"] == pytest.approx([649.5, 179.5], abs=1)
    assert annotate(image, points).startswith(b"\xff\xd8")


def test_blank_image_never_invents_motor_positions():
    assert detect_motors(np.zeros((300, 400, 3), np.uint8)) == []


def test_duplicate_tag_ids_are_ambiguous_and_not_reported():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 2, 160)
    image = np.full((300, 600), 255, np.uint8)
    image[60:220, 60:220] = marker
    image[60:220, 360:520] = marker
    assert detect_motors(image) == []


def test_stale_vision_cannot_display_cached_image_or_detections(monkeypatch):
    vision = MotorVision()
    vision._state = {"running": True, "detected": [{"id": 1}], "error": None}
    vision._jpeg = b"old"
    vision._updated = 10
    monkeypatch.setattr("so101.services.motor_vision.time.monotonic", lambda: 12)
    assert not vision.status()["live"]
    assert vision.status()["detected"] == []
    assert vision.snapshot() is None
    assert not vision.status()["camera_projection_calibrated"]

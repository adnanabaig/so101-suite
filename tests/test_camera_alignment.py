"""Geometric and freshness contracts for the camera-only phone alignment guide."""

import importlib.util
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "camera_alignment", Path(__file__).resolve().parents[1] / "scripts/camera_alignment.py"
)
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)


@pytest.fixture
def profile():
    return {
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "coefficients": [0.0] * 5,
        "width": 640,
        "height": 480,
    }


def test_generated_phone_target_is_detectable(tmp_path):
    path = tmp_path / "target.png"
    alignment.make_target(path)
    image = cv2.imread(str(path))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    _, ids, _ = detector.detectMarkers(image)
    assert image.shape[:2] == (2796, 1290)
    assert ids.ravel().tolist() == [42]


def test_pose_recovery_from_known_projection(profile):
    points = np.array([[0, 0, 0], [0.06, 0, 0], [0.06, 0.06, 0], [0, 0.06, 0]])
    matrix = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    rotation = np.array([0.25, -0.15, 0.08])
    translation = np.array([-0.02, -0.03, 0.5])
    pixels = cv2.projectPoints(points, rotation, translation, matrix, None)[0].reshape(4, 2)
    candidates = alignment.pose_candidates(pixels, 60, profile)
    recovered = np.array(candidates[0]["camera_from_marker"])
    np.testing.assert_allclose(recovered[:3, 3], translation, atol=1e-6)
    np.testing.assert_allclose(recovered[:3, :3], cv2.Rodrigues(rotation)[0], atol=1e-6)
    assert candidates[0]["rms_px"] < 1e-6


def test_phone_screen_height_and_axis_direction_are_preserved():
    transform = alignment.table_from_marker(60, 200, 300, 12, 0)
    top_left = transform @ [0, 0, 0, 1]
    bottom_right = transform @ [0.06, 0.06, 0, 1]
    np.testing.assert_allclose(top_left[:3], [0.2, 0.3, 0.012])
    np.testing.assert_allclose(bottom_right[:3], [0.26, 0.24, 0.012])
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)


def phone_report():
    return {
        "screen_width": 430,
        "screen_height": 932,
        "dpr": 3,
        "scale": 1,
        "viewport_width": 430,
        "marker_css_width": 340,
    }


def test_controlled_phone_size_and_user_height_conversion():
    assert alignment.phone_width(phone_report()) == pytest.approx(56.32173913)
    assert alignment.TABLE_HEIGHT_MM == pytest.approx(739.775)
    assert alignment.CAMERA_BOTTOM_HEIGHT_MM - alignment.TABLE_HEIGHT_MM == pytest.approx(619.125)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", 1.2),
        ("dpr", 2),
        ("screen_width", 390),
        ("viewport_width", 400),
        ("marker_css_width", 300),
        ("screen_height", 430),
    ],
)
def test_zoomed_wrong_phone_or_resized_target_rejected(field, value):
    report = phone_report()
    report[field] = value
    with pytest.raises(ValueError):
        alignment.phone_width(report)


def test_quick_capture_keeps_unknown_phone_height_out_of_table_transform():
    samples, _ = sample_data()
    result = alignment.summarize_samples(
        samples, {"mode": "phone_reference", "width_mm": alignment.phone_width(phone_report())}
    )
    assert result["reference_frame"] == "phone_screen"
    assert result["table_from_camera"] is None
    assert result["screen_height_above_table_mm"] is None
    assert result["screen_height_status"] == "not_measured"
    assert not result["robot_base_alignment_verified"]
    with pytest.raises(ValueError, match="Advanced"):
        alignment.compare_placements(result, result)


def test_phone_report_autoconfigures_quick_capture_without_overriding_manual(tmp_path, profile):
    source = tmp_path / "intrinsics.json"
    source.write_text(
        json.dumps({"devices": [{"serial": "347622075852", "streams": {"color": profile}}]})
    )
    guide = alignment.Guide(source, tmp_path / "output")
    report = guide.report_phone(phone_report())
    assert report["width_mm"] == pytest.approx(56.32173913)
    assert guide.measurements["mode"] == "phone_reference"
    assert guide.measurements["camera_bottom_above_table_mm"] == pytest.approx(619.125)
    identifier = guide.configuration_id
    guide.report_phone(phone_report())
    assert guide.configuration_id == identifier
    _, measurements = sample_data()
    manual_id = guide.configure(measurements)
    guide.report_phone(phone_report())
    assert guide.configuration_id == manual_id
    assert guide.measurements["height_mm"] == 12


def test_rejected_phone_report_is_diagnosable_without_enabling_capture(tmp_path, profile):
    source = tmp_path / "intrinsics.json"
    source.write_text(
        json.dumps({"devices": [{"serial": "347622075852", "streams": {"color": profile}}]})
    )
    guide = alignment.Guide(source, tmp_path / "output")
    report = phone_report()
    report["screen_width"] = 390
    with pytest.raises(ValueError):
        guide.report_phone(report)
    assert guide.phone_error["display"]["screen_width"] == 390
    assert not guide.measurements
    assert not guide.phone


@pytest.mark.parametrize(
    "values",
    [(60, 0, 0, 0, 0), (0, 0, 0, 8, 0), (60, float("nan"), 0, 8, 0), (60, 0, 0, 8, float("inf"))],
)
def test_invalid_measurements_rejected(values):
    with pytest.raises(ValueError):
        alignment.table_from_marker(*values)


def sample_data():
    now = time.monotonic()
    corners = [[200.0, 200.0], [260.0, 200.0], [260.0, 260.0], [200.0, 260.0]]
    samples = [
        {
            "at": now - (29 - index) * 0.1,
            "corners": corners,
            "candidates": [
                {"camera_from_marker": np.eye(4).tolist(), "rms_px": 0.2},
                {"camera_from_marker": np.eye(4).tolist(), "rms_px": 0.21},
            ],
        }
        for index in range(30)
    ]
    measured = {
        "width_mm": 60.0,
        "x_mm": 100.0,
        "y_mm": 200.0,
        "height_mm": 12.0,
        "yaw_deg": 0.0,
        "level_confirmed": True,
    }
    return samples, measured


def test_candidate_never_claims_robot_alignment_and_preserves_ambiguity():
    samples, measurements = sample_data()
    result = alignment.summarize_samples(samples, measurements)
    assert result["status"] == "candidate_only"
    assert result["planar_ambiguity"]
    assert result["independent_validation_required"]
    assert not result["robot_base_alignment_verified"]


def test_advanced_reference_transform_matches_measured_table():
    samples, measurements = sample_data()
    measurements["yaw_deg"] = 35
    result = alignment.summarize_samples(samples, measurements)
    assert result["reference_frame"] == "measured_table"
    np.testing.assert_allclose(result["reference_from_camera"], result["table_from_camera"])
    np.testing.assert_allclose(np.array(result["reference_from_camera"])[:3, 3], [0.1, 0.2, 0.012])


@pytest.mark.parametrize("problem", ["stale", "moving", "brief", "not_level", "poor_fit"])
def test_unreliable_capture_rejected(problem):
    samples, measurements = sample_data()
    if problem == "stale":
        for sample in samples:
            sample["at"] -= 5
    elif problem == "moving":
        samples[0]["corners"] = [[220.0, 200.0]] * 4
    elif problem == "brief":
        samples = samples[-10:]
    elif problem == "not_level":
        measurements["level_confirmed"] = False
    else:
        samples[0]["candidates"][0]["rms_px"] = 3
    with pytest.raises(ValueError):
        alignment.summarize_samples(samples, measurements)


def test_nonzero_inverse_distortion_not_silently_treated_as_opencv(profile):
    profile["coefficients"][0] = 0.1
    with pytest.raises(ValueError, match="unsupported"):
        alignment.pose_candidates(np.zeros((4, 2)), 60, profile)


def test_comparison_requires_separate_positions_and_reports_inconsistency():
    samples, measurements = sample_data()
    first = alignment.summarize_samples(samples, measurements)
    first.update(serial="D435i", intrinsics_sha256="same")
    second = json.loads(json.dumps(first))
    assert not alignment.compare_placements(first, second)["within_provisional_tolerance"]
    second["table_from_marker"][0][3] += 0.15
    assert alignment.compare_placements(first, second)["within_provisional_tolerance"]
    second["table_from_camera"][2][3] += 0.02
    assert not alignment.compare_placements(first, second)["within_provisional_tolerance"]
    second["intrinsics_sha256"] = "different"
    with pytest.raises(ValueError):
        alignment.compare_placements(first, second)


@pytest.mark.parametrize("age_ns", [-1, 600_000_000])
def test_stale_or_future_snapshot_rejected(age_ns):
    now = time.time_ns()
    with pytest.raises(ValueError):
        alignment.frame_identity(
            {"X-SO101-Sequence": "12", "X-SO101-Captured-At-Ns": str(now - age_ns)}, now
        )


def test_snapshot_metadata_used_for_freshness():
    now = time.time_ns()
    assert alignment.frame_identity(
        {"X-SO101-Sequence": "12", "X-SO101-Captured-At-Ns": str(now - 10_000)}, now
    ) == (12, now - 10_000)


@pytest.mark.parametrize(
    "host,origin,expected",
    [
        ("127.0.0.1:8782", None, True),
        ("localhost:8782", "http://localhost:8782", True),
        ("attacker.example:8782", None, False),
        ("127.0.0.1:8782", "http://attacker.example", False),
    ],
)
def test_local_host_and_origin_only(host, origin, expected):
    assert alignment.local_request(host, origin, 8782) is expected


def test_real_target_detection_to_saved_candidate(tmp_path, profile, monkeypatch):
    source = tmp_path / "intrinsics.json"
    source.write_text(
        json.dumps({"devices": [{"serial": "347622075852", "streams": {"color": profile}}]})
    )
    guide = alignment.Guide(source, tmp_path / "output")
    _, measurements = sample_data()
    configuration_id = guide.configure(measurements)
    image = np.full((480, 640), 255, np.uint8)
    image[170:310, 250:390] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 42, 140
    )
    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
    clock = [100.0]
    sequence = [0]
    identity = ["realsense:347622075852"]

    class Response(io.BytesIO):
        headers = {}

    def fetch(url, timeout):
        if url.endswith("status.json"):
            return Response(
                json.dumps(
                    {"live": True, "age_seconds": 0.01, "camera_profile": {"identity": identity[0]}}
                ).encode()
            )
        response = Response(jpeg)
        response.headers = {
            "X-SO101-Sequence": str(sequence[0]),
            "X-SO101-Captured-At-Ns": str(time.time_ns()),
        }
        return response

    monkeypatch.setattr(alignment.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(alignment.time, "monotonic", lambda: clock[0])
    for index in range(30):
        sequence[0] = index + 1
        clock[0] += 0.1
        guide.frame()
    assert guide.status["ready"]
    saved = guide.save(configuration_id)
    record = json.loads(Path(saved["path"]).read_text())
    assert record["sample_count"] == 30
    assert record["serial"] == "347622075852"
    assert record["frame_sequence"] == 30
    assert record["measurements"]["height_mm"] == 12
    assert not record["robot_base_alignment_verified"]
    assert Path(saved["path"]).with_name("frame.jpg").exists()
    with pytest.raises(ValueError, match="Measurements changed"):
        guide.save("old-configuration")
    before = len(guide.samples)
    guide.frame()
    assert len(guide.samples) == before
    identity[0] = "realsense:419122270120"
    with pytest.raises(ValueError, match="identity"):
        guide.frame()

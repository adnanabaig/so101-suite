"""Saved calibration evidence stays distinct from live camera observations."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "viewer", Path(__file__).resolve().parents[1] / "scripts/foxglove_viewer.py"
)
assert spec and spec.loader
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_saved_evidence_is_labeled_and_missing_files_visible(tmp_path):
    saved = {"joints": {"shoulder_lift": {"min": -107, "max": 107}}}
    (tmp_path / "calibration_follower.json").write_text(json.dumps(saved))
    records = viewer.saved_records(tmp_path)
    assert records["/calibration/installed"]["data"] == saved
    for record in records.values():
        assert record["source_kind"] == "saved_evidence"
        assert record["live_joint_telemetry"] is False
    assert not records["/calibration/rest_pose"]["available"]


def test_corrupt_evidence_is_not_published_as_valid(tmp_path):
    (tmp_path / "rest_pose.json").write_text("{")
    record = viewer.saved_records(tmp_path)["/calibration/rest_pose"]
    assert not record["available"] and "error" in record


def test_image_sequence_and_timestamp_are_bound_to_the_jpeg_response(monkeypatch):
    monkeypatch.setattr(
        viewer, "fetch", lambda url: b'{"live":true,"age_seconds":0.01,"sequence":4}'
    )

    class Response:
        headers = {"X-SO101-Sequence": "5", "X-SO101-Captured-At-Ns": "123456789"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"frame-five"

    monkeypatch.setattr(viewer.urllib.request, "urlopen", lambda *a, **kw: Response())
    jpeg, status = viewer.timestamped_image(viewer.CAMERA)
    assert jpeg == b"frame-five" and status["sequence"] == 5
    assert status["capture_timestamp_ns"] == 123456789
    assert status["timestamp_source"] == "host_camera_read_completed"


@pytest.mark.parametrize("live", [True, False])
def test_stale_camera_does_not_emit_a_frame(live):
    def fetch(url):
        assert url.endswith("status.json")
        return json.dumps({"live": live, "age_seconds": 3}).encode()

    with pytest.raises(RuntimeError, match="no fresh frame"):
        viewer.fresh_image(fetch)


def test_fresh_camera_uses_relay_not_device():
    requested = []

    def fetch(url):
        requested.append(url)
        return b'{"live":true,"age_seconds":0.01}' if url.endswith("json") else b"jpeg"

    frame, status = viewer.fresh_image(fetch)
    assert frame == b"jpeg" and status["live"]
    assert requested == [viewer.CAMERA + "/status.json", viewer.CAMERA + "/snapshot.jpg"]


def test_operational_dashboard_only_reads_cached_get_endpoints(monkeypatch):
    requested = []

    def fetch(url):
        requested.append(url)
        if url.endswith("/api/preflight/latest"):
            raise RuntimeError("No saved preflight")
        return b'{"mode":"idle"}'

    monkeypatch.setattr(viewer, "fetch", fetch)
    result = viewer.operational_snapshot()
    assert result["/robot/status"]["bridge"]["available"]
    assert not result["/diagnostics/preflight"]["bridge"]["available"]
    assert set(requested) == {
        "http://127.0.0.1:8000" + path for path in viewer.OPERATIONAL_TOPICS.values()
    }
    assert not any("/start" in url or "/pose/" in url for url in requested)

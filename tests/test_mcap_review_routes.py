import io
import json
from types import SimpleNamespace

import pytest

from so101.services.routers import mcap


@pytest.fixture
def idle(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        status=lambda: {"mode": "idle"}, settings=SimpleNamespace(data_dir=tmp_path)
    )
    monkeypatch.setattr(mcap, "get_manager", lambda: manager)
    return manager


def test_recorded_routes_refuse_motion_and_never_request_arm_commands(monkeypatch, idle):
    idle.status = lambda: {"mode": "teleop"}

    def forbidden():
        pytest.fail("read recording while motion active")

    monkeypatch.setattr(mcap, "review_store", forbidden)
    with pytest.raises(mcap.HTTPException) as error:
        mcap.review_timeline("a" * 32)
    assert error.value.status_code == 409


def test_open_foxglove_launches_only_selected_contained_recording(monkeypatch, idle, tmp_path):
    root = tmp_path / "episode-reviews"
    root.mkdir()
    path = root / "chosen.mcap"
    path.write_bytes(b"mcap")
    store = SimpleNamespace(
        raw_dir=tmp_path / "mcap",
        episode=lambda identity: {},
        reports=lambda: {"a" * 32: {"path": str(path)}},
    )
    monkeypatch.setattr(mcap, "review_store", lambda: store)
    monkeypatch.setattr(mcap.shutil, "which", lambda name: "/usr/bin/foxglove-studio")
    calls = []
    monkeypatch.setattr(
        mcap.subprocess, "Popen", lambda arguments, **kwargs: calls.append(arguments)
    )
    assert mcap.open_recorded_foxglove("a" * 32)["launch_requested"]
    assert calls == [["/usr/bin/foxglove-studio", str(path)]]


def test_missing_foxglove_is_visible_and_never_exports(monkeypatch, idle):
    monkeypatch.setattr(
        mcap,
        "review_store",
        lambda: SimpleNamespace(episode=lambda identity: {}, reports=lambda: {}),
    )
    monkeypatch.setattr(mcap.shutil, "which", lambda name: None)
    with pytest.raises(mcap.HTTPException) as error:
        mcap.open_recorded_foxglove("a" * 32)
    assert error.value.status_code == 503


def test_camera_settings_refused_during_recording(monkeypatch, idle):
    monkeypatch.setattr(
        mcap, "sessions", lambda: SimpleNamespace(snapshot=lambda: {"active": None})
    )
    monkeypatch.setattr(mcap, "recorder_status", lambda folder: {"recording": True})
    with pytest.raises(mcap.HTTPException) as error:
        mcap.set_camera_controls("camera_1", mcap.CameraSettingsRequest(values={"brightness": 1}))
    assert error.value.status_code == 409


def test_camera_settings_use_capture_owner_and_report_readback(monkeypatch, idle):
    monkeypatch.setattr(
        mcap, "sessions", lambda: SimpleNamespace(snapshot=lambda: {"active": None})
    )
    monkeypatch.setattr(mcap, "recorder_status", lambda folder: {"recording": False})
    outgoing = []

    def response(request, timeout):
        outgoing.append(request)
        return io.BytesIO(json.dumps({"controls": {"brightness": {"value": 1}}}).encode())

    monkeypatch.setattr(mcap.urllib.request, "urlopen", response)
    result = mcap.set_camera_controls(
        "camera_1", mcap.CameraSettingsRequest(values={"brightness": 1})
    )
    assert result["controls"]["brightness"]["value"] == 1
    assert outgoing[0].full_url == "http://127.0.0.1:8768/controls"
    assert json.loads(outgoing[0].data) == {"brightness": 1}

"""MCAP episode boundaries are durable observations, never motor authority."""

import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from so101.config import get_settings
from so101.services.app import create_app
from so101.services.mcap_sessions import McapEpisodes, recorder_status
from so101.services.routers import mcap


def test_camera_snapshot_uses_existing_relay_and_never_caches(monkeypatch):
    import io

    requested = []

    def read(url, timeout):
        requested.append((url, timeout))
        return io.BytesIO(b"jpeg-frame")

    monkeypatch.setattr(mcap.urllib.request, "urlopen", read)
    result = mcap.camera_snapshot("realsense")
    assert result.body == b"jpeg-frame"
    assert result.headers["cache-control"] == "no-store"
    assert requested == [("http://127.0.0.1:8770/snapshot.jpg", 2)]
    with pytest.raises(mcap.HTTPException) as error:
        mcap.camera_snapshot("unconfigured")
    assert error.value.status_code == 404


def test_saved_episodes_are_owned_idempotent_and_survive_restart(tmp_path):
    episodes = McapEpisodes(tmp_path)
    started = episodes.start("operator", "move cube", "leader_teleop", "segment.mcap")
    assert "operator_id" not in started
    with pytest.raises(ValueError, match="already active"):
        episodes.start("operator", "duplicate", "leader_teleop", "segment.mcap")
    with pytest.raises(ValueError, match="Only"):
        episodes.finish("other", started["id"])
    saved = episodes.finish("operator", started["id"])
    assert saved["status"] == "saved"
    assert saved["ended_at_ns"] >= saved["started_at_ns"]
    assert episodes.finish("operator", started["id"]) == saved
    assert McapEpisodes(tmp_path).snapshot()["episodes"] == [saved]


def test_interrupted_episode_is_retained_instead_of_claiming_success(tmp_path):
    episodes = McapEpisodes(tmp_path)
    started = episodes.start("operator", "move cube", "leader_teleop", "segment.mcap")
    recovered = McapEpisodes(tmp_path).snapshot()
    assert recovered["active"] is None
    assert recovered["episodes"][0]["id"] == started["id"]
    assert recovered["episodes"][0]["status"] == "interrupted"
    assert recovered["episodes"][0]["end_reason"] == "server_restarted"


def test_failed_metadata_write_preserves_active_episode(tmp_path, monkeypatch):
    from so101.services import mcap_sessions

    episodes = McapEpisodes(tmp_path)
    started = episodes.start("operator", "task", "leader_teleop", "segment.mcap")
    before = episodes.path.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(mcap_sessions, "atomic_write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        episodes.finish("operator", started["id"])
    assert episodes.snapshot()["active"]["id"] == started["id"]
    assert episodes.path.read_bytes() == before


@pytest.fixture
def api(monkeypatch):
    settings = get_settings()
    folder = settings.data_dir / "mcap"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(
        json.dumps(
            {
                "active": True,
                "updated_at": time.time(),
                "bytes": 100,
                "telemetry_received_at": time.time(),
                "path": str(folder / "segment.mcap.active"),
            }
        )
    )
    state = {"mode": "teleop", "backend": "mock", "torque_state": "on"}

    def heartbeat(operator):
        if operator != "operator":
            raise RuntimeError("Fresh operator heartbeat required")

    manager = SimpleNamespace(
        settings=settings,
        status=lambda: state,
        require_operator_heartbeat=heartbeat,
        owns_operation=lambda operator: operator == "operator",
    )
    monkeypatch.setattr(mcap, "get_manager", lambda: manager)
    monkeypatch.setattr(mcap, "relay_status", lambda: {"obsbot": {"live": True}})
    return TestClient(create_app()), state, folder


def test_api_episode_flow_requires_live_recorder_and_leader_mode(api):
    client, state, folder = api
    state["mode"] = "idle"
    assert (
        client.post("/api/mcap/episodes/start", json={"operator_id": "operator"}).status_code == 409
    )
    state["mode"] = "teleop"
    assert client.post("/api/mcap/episodes/start", json={"operator_id": "other"}).status_code == 409
    response = client.post(
        "/api/mcap/episodes/start", json={"operator_id": "operator", "task": "pick"}
    )
    assert response.status_code == 200, response.text
    episode = response.json()
    assert client.get("/api/mcap/status").json()["active"]["id"] == episode["id"]
    assert (
        client.post(
            "/api/mcap/episodes/save", json={"operator_id": "other", "episode_id": episode["id"]}
        ).status_code
        == 409
    )
    saved = client.post(
        "/api/mcap/episodes/save", json={"operator_id": "operator", "episode_id": episode["id"]}
    )
    assert saved.json()["status"] == "saved"
    assert client.get("/api/mcap/status").json()["active"] is None
    (folder / "status.json").write_text('{"active":false,"updated_at":0}')
    assert (
        client.post("/api/mcap/episodes/start", json={"operator_id": "operator"}).status_code == 409
    )


def test_stop_marks_active_episode_interrupted_and_stale_recording_is_visible(api):
    client, state, folder = api
    client.post("/api/mcap/episodes/start", json={"operator_id": "operator"})
    state["mode"] = "idle"
    before = (folder / "episodes.json").read_bytes()
    status = client.get("/api/mcap/status").json()
    assert status["active"] is not None
    assert (folder / "episodes.json").read_bytes() == before
    manager = mcap.get_manager()
    stop, thread = mcap.start_supervisor(manager)
    try:
        deadline = time.monotonic() + 2
        while mcap.sessions().snapshot()["active"] and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=2)
    status = client.get("/api/mcap/status").json()
    assert status["active"] is None and status["episodes"][0]["status"] == "interrupted"
    data = json.loads((folder / "status.json").read_text())
    data["updated_at"] = 0
    (folder / "status.json").write_text(json.dumps(data))
    assert not recorder_status(folder.parent)["recording"]


def test_camera_proxy_only_reads_fixed_local_relays(api, monkeypatch):
    client, _, _ = api
    calls = []

    class Response:
        def read1(self, size):
            return b""

        def close(self):
            calls.append("closed")

    def open_url(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(mcap.urllib.request, "urlopen", open_url)
    assert client.get("/api/mcap/cameras/unknown/stream.mjpg").status_code == 404
    assert not calls
    response = client.get("/api/mcap/cameras/obsbot/stream.mjpg")
    assert response.status_code == 200
    assert calls == ["http://127.0.0.1:8765/stream.mjpg", "closed"]


@pytest.mark.parametrize("failure", ["gap", "stale", "error", "wrong_activity"])
def test_real_episode_is_interrupted_when_control_capture_is_lost(api, failure):
    client, state, folder = api
    episode = client.post("/api/mcap/episodes/start", json={"operator_id": "operator"}).json()
    state["backend"] = "real"
    recording = json.loads((folder / "status.json").read_text())
    recording.update(activity_id=episode["id"], control_received_at=time.time())
    if failure == "gap":
        recording["control_gap_count"] = 1
    elif failure == "stale":
        recording["control_received_at"] = 0
    elif failure == "error":
        recording["control_error"] = "Trace request failed"
    else:
        recording["activity_id"] = "different"
    (folder / "status.json").write_text(json.dumps(recording))
    stop, thread = mcap.start_supervisor(mcap.get_manager())
    try:
        deadline = time.monotonic() + 2
        while mcap.sessions().snapshot()["active"] and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=2)
    saved = mcap.sessions().snapshot()["episodes"][-1]
    assert saved["status"] == "interrupted"
    assert saved["end_reason"] == "control_capture_unhealthy"
    assert state["mode"] == "teleop"


def test_ready_idle_recorder_acknowledges_capture_before_episode_can_be_demonstrated(
    api, monkeypatch
):
    client, _, folder = api
    original_start = []

    def recording_status(data_dir):
        active = mcap.sessions().snapshot()["active"]
        if active is None:
            return {
                "recording": False,
                "telemetry_fresh": False,
                "service_ready": True,
                "api_fresh": True,
            }
        original_start.append(active["started_at_ns"])
        return {
            "recording": True,
            "telemetry_fresh": True,
            "service_ready": True,
            "activity_id": active["id"],
            "path": str(folder / "episode.mcap.active"),
        }

    monkeypatch.setattr(mcap, "recorder_status", recording_status)
    response = client.post(
        "/api/mcap/episodes/start", json={"operator_id": "operator", "task": "pick"}
    )
    assert response.status_code == 200, response.text
    episode = response.json()
    assert episode["status"] == "recording"
    assert episode["started_at_ns"] >= original_start[0]
    assert episode["starting_segment"].endswith("episode.mcap.active")

"""Only live explicit activities request disk capture; stale PIDs cannot keep it running."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101 import capture_activity as module


@pytest.fixture
def recorder_module():
    spec = importlib.util.spec_from_file_location(
        "metadata_recorder_test", Path(__file__).resolve().parents[1] / "scripts/record_mcap.py"
    )
    assert spec and spec.loader
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    return recorder


def test_segment_metadata_links_rotation_and_diagnostic_identity(recorder_module, tmp_path):
    requested = {"id": "episode-a", "kind": "episode", "label": "slow shoulder sweep"}
    first = recorder_module.segment_metadata(
        {"control_stream_id": "stream-a"}, requested, tmp_path / "first.mcap.active"
    )
    second = recorder_module.segment_metadata(
        {"control_stream_id": "stream-a"}, requested, tmp_path / "second.mcap.active"
    )
    assert first["episode_id"] == second["episode_id"] == "episode-a"
    assert first["segment_id"] != second["segment_id"]
    assert first["control_stream_id"] == "stream-a"
    assert first["task"] == "slow shoulder sweep"
    assert all(len(value) == 64 for value in first["source_sha256"].values())
    diagnostic = recorder_module.segment_metadata(
        {}, {"id": "diagnostic-a", "kind": "diagnostic"}, tmp_path / "test.mcap.active"
    )
    assert diagnostic["episode_id"] is None
    assert diagnostic["activity_id"] == "diagnostic-a"


def test_recorded_tick_and_final_annotation_have_episode_identity(
    recorder_module, tmp_path, monkeypatch
):
    messages = []
    monkeypatch.setitem(
        sys.modules,
        "foxglove",
        SimpleNamespace(log=lambda *args, **kwargs: messages.append((args, kwargs))),
    )
    tick = {"timestamp_ns": 123, "seq": 4, "control_profile": {"max_velocity": 60}}
    monkeypatch.setattr(
        recorder_module,
        "fetch",
        lambda url: json.dumps(
            {"stream_id": "stream", "missing_before_buffer": 0, "ticks": [tick]}
        ).encode(),
    )
    recorder_module.log_control_ticks(
        {"activity_id": "episode-a", "activity_kind": "episode"}, "http://localhost"
    )
    payload = messages[0][0][1]
    assert payload["episode_id"] == payload["recording_activity_id"] == "episode-a"
    assert payload["control_profile"] == tick["control_profile"]
    assert messages[0][1]["log_time"] == 123
    assert "episode_id" not in tick
    (tmp_path / "episodes.json").write_text(
        json.dumps(
            {
                "active": None,
                "episodes": [{"id": "episode-a", "status": "saved", "operator_id": "private"}],
            }
        )
    )
    recorder_module.log_episodes(tmp_path, episode_id="episode-a")
    annotation = [args[1] for args, _ in messages if args[0] == "/recording/episode"][0]
    assert annotation["id"] == "episode-a" and annotation["status"] == "saved"
    assert "operator_id" not in annotation


def test_idle_and_nested_activity_lifetimes(tmp_path):
    state = tmp_path / "mcap"
    assert module.requested_activity(state) is None
    with module.capture_activity(tmp_path, "calibration", "wrist sweep") as outer:
        assert module.requested_activity(state)["id"] == outer["id"]
        with module.capture_activity(tmp_path, "diagnostic", "nested read") as inner:
            assert inner["id"] == outer["id"]
        assert module.requested_activity(state)["kind"] == "calibration"
    assert module.requested_activity(state) is None
    assert not list((state / "requests").glob("*.json"))


def test_reused_pid_and_failed_operation_do_not_leave_active_requests(tmp_path, monkeypatch):
    with module.capture_activity(tmp_path, "diagnostic", "read"):
        monkeypatch.setattr(module, "process_identity", lambda pid: "different process")
        assert module.requested_activity(tmp_path / "mcap") is None
    (tmp_path / "mcap/episodes.json").write_text(
        json.dumps(
            {
                "active": {
                    "id": "episode",
                    "task": "pick",
                    "status": "starting",
                }
            }
        )
    )
    assert module.requested_activity(tmp_path / "mcap")["id"] == "episode"


def test_idle_recorder_is_ready_without_creating_any_mcap(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "idle_recorder_test", Path(__file__).resolve().parents[1] / "scripts/record_mcap.py"
    )
    assert spec and spec.loader
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, seconds):
            self.stopped = True

        def set(self):
            self.stopped = True

    monkeypatch.setattr(recorder.threading, "Event", Stop)
    monkeypatch.setattr(recorder.signal, "signal", lambda *args: None)
    monkeypatch.setattr(recorder, "fetch", lambda url: b'{"mode":"idle"}')
    monkeypatch.setitem(
        sys.modules,
        "foxglove",
        SimpleNamespace(
            open_mcap=lambda *args: (_ for _ in ()).throw(AssertionError("Idle disk recording")),
        ),
    )
    output = tmp_path / "raw"
    state = tmp_path / "state"
    monkeypatch.setattr(
        sys, "argv", ["record_mcap.py", "--output", str(output), "--state-dir", str(state)]
    )
    previous = recorder.os.umask(0o077)
    try:
        assert recorder.main() == 0
    finally:
        recorder.os.umask(previous)
    assert list(output.iterdir()) == []
    result = json.loads((state / "status.json").read_text())
    assert result["active"] is False and result["ready"] is False


def test_idle_recorder_clears_recovered_api_error(recorder_module, tmp_path, monkeypatch):
    class Stop:
        waits = 0

        def is_set(self):
            return self.waits >= 2

        def wait(self, seconds):
            self.waits += 1

        def set(self):
            self.waits = 2

    responses = iter([ConnectionRefusedError("API restarting"), b'{"mode":"idle"}'])

    def fetch(url):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    published = []
    monkeypatch.setattr(recorder_module.threading, "Event", Stop)
    monkeypatch.setattr(recorder_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(recorder_module, "fetch", fetch)
    monkeypatch.setattr(
        recorder_module, "publish_status", lambda path, state: published.append(dict(state))
    )
    monkeypatch.setitem(sys.modules, "foxglove", SimpleNamespace())
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_mcap.py", "--output", str(tmp_path / "raw"), "--state-dir", str(tmp_path)],
    )
    previous = recorder_module.os.umask(0o077)
    try:
        assert recorder_module.main() == 0
    finally:
        recorder_module.os.umask(previous)
    assert published[0]["api_error"] == "API restarting"
    assert "telemetry_error" not in published[0]
    assert "api_error" not in published[1]
    assert published[1]["robot_mode"] == "idle"
    assert not list((tmp_path / "raw").glob("*.mcap*"))


@pytest.mark.parametrize("acknowledge", [True, False])
def test_standalone_capture_waits_for_ack_and_releases_request_and_lock(
    tmp_path, monkeypatch, acknowledge
):
    from so101.bringup import standalone

    state = tmp_path / "mcap"
    state.mkdir()
    status = state / "status.json"
    status.write_text(json.dumps({"ready": True, "updated_at": time.time()}))
    released = []
    monkeypatch.setattr(standalone, "_drop_lock", lambda *args, **kwargs: released.append(True))
    clock = [0.0]
    monkeypatch.setattr(standalone.time, "monotonic", lambda: clock[0])

    def advance(seconds):
        clock[0] += 1
        request = module.requested_activity(state)
        assert request is not None and not released
        if acknowledge:
            status.write_text(json.dumps({"active": True, "activity_id": request["id"]}))

    monkeypatch.setattr(standalone.time, "sleep", advance)
    access = standalone.StandaloneAccess(
        "real",
        "calibration sweep",
        lock_path=tmp_path / "lock",
        lock_key=("fake",),
        settings=SimpleNamespace(data_dir=tmp_path, camera_capture_mode="shared"),
    )
    if acknowledge:
        with access:
            assert module.requested_activity(state)["kind"] == "calibration"
            assert not released
    else:
        with pytest.raises(RuntimeError, match="did not acknowledge"):
            with access:
                pytest.fail("Operation entered without capture acknowledgement")
    assert released == [True]
    assert module.requested_activity(state) is None
    access.release()
    assert released == [True]

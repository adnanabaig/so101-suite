"""A missing, wrong or full recording drive never falls back to the laptop disk."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recording_storage", ROOT / "scripts/recording_storage.py"
)
assert spec and spec.loader
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def write_profile(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values))
    return path


def test_default_local_profile_selects_drive_and_keeps_metadata_local(tmp_path):
    output = tmp_path / "external/mcap"
    metadata = tmp_path / "local/state"
    write_profile(
        tmp_path / "config/rigs/mcap-storage.local.json",
        output=str(output),
        state_dir=str(metadata),
        min_free_gib=50,
    )
    config = storage.load_storage(tmp_path)
    assert config.output == output and config.state_dir == metadata and config.min_free_gib == 50
    isolated = storage.load_storage(tmp_path, output=tmp_path / "isolated")
    assert isolated.output == isolated.state_dir == tmp_path / "isolated"
    assert isolated.required_mount is None


def test_missing_explicit_profile_is_an_error_not_internal_fallback(tmp_path):
    with pytest.raises(ValueError, match="Cannot load"):
        storage.load_storage(tmp_path, profile=tmp_path / "missing.json")


def test_missing_drive_refuses_before_creating_any_output(tmp_path):
    mount = tmp_path / "unmounted"
    config = storage.RecordingStorage(mount / "mcap", tmp_path / "state", required_mount=mount)
    with pytest.raises(storage.StorageUnavailable, match="not mounted"):
        config.prepare()
    assert not mount.exists()
    assert not config.state_dir.exists()


def test_uuid_guard_and_unplug_are_checked_on_each_pass(tmp_path, monkeypatch):
    mount = tmp_path / "drive"
    mount.mkdir()
    monkeypatch.setattr(storage.Path, "is_mount", lambda path: path == mount)
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * 1024**3)
    )
    monkeypatch.setattr(
        storage.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="abcd-1234\n")
    )
    config = storage.RecordingStorage(mount / "mcap", tmp_path / "state", mount, "ABCD-1234", 50)
    config.prepare()
    existing = config.output / "evidence.mcap"
    existing.write_bytes(b"preserved")
    monkeypatch.setattr(
        storage.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="wrong")
    )
    with pytest.raises(storage.StorageUnavailable, match="UUID"):
        config.check()
    monkeypatch.setattr(storage.Path, "is_mount", lambda path: False)
    with pytest.raises(storage.StorageUnavailable, match="not mounted"):
        config.check()
    assert existing.read_bytes() == b"preserved"


def test_symlink_cannot_redirect_recording_off_the_required_drive(tmp_path, monkeypatch):
    mount = tmp_path / "drive"
    mount.mkdir()
    outside = tmp_path / "internal"
    outside.mkdir()
    (mount / "redirect").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(storage.Path, "is_mount", lambda path: path == mount)
    config = storage.RecordingStorage(mount / "redirect/mcap", tmp_path / "state", mount)
    with pytest.raises(storage.StorageUnavailable, match="outside"):
        config.prepare()
    assert not (outside / "mcap").exists()


def test_low_space_pauses_without_deleting_files(tmp_path, monkeypatch):
    output = tmp_path / "mcap"
    output.mkdir()
    evidence = output / "keep.mcap"
    evidence.write_bytes(b"original")
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=49 * 1024**3)
    )
    with pytest.raises(storage.StorageUnavailable, match="50 GiB reserve"):
        storage.RecordingStorage(output, tmp_path / "state", min_free_gib=50).check()
    assert evidence.read_bytes() == b"original"


@pytest.mark.parametrize("reserve", [0, -1, float("nan"), float("inf"), True])
def test_invalid_reserves_are_rejected(tmp_path, reserve):
    with pytest.raises(ValueError):
        storage.RecordingStorage(tmp_path, tmp_path, min_free_gib=reserve)


def test_recorder_missing_drive_returns_non_retryable_status_without_opening_writer(
    tmp_path, monkeypatch
):
    module_spec = importlib.util.spec_from_file_location(
        "recorder_storage_test", ROOT / "scripts/record_mcap.py"
    )
    assert module_spec and module_spec.loader
    recorder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(recorder)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    profile = write_profile(
        tmp_path / "profile.json",
        output=str(tmp_path / "missing/mcap"),
        state_dir=str(state_dir),
        required_mount=str(tmp_path / "missing"),
    )
    monkeypatch.setattr(sys, "argv", ["record_mcap.py", "--storage-profile", str(profile)])
    monkeypatch.setattr(recorder.signal, "signal", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "foxglove",
        SimpleNamespace(open_mcap=lambda *args: pytest.fail("Must not open writer")),
    )
    previous = recorder.os.umask(0o077)
    try:
        assert recorder.main() == 78
    finally:
        recorder.os.umask(previous)
    state = json.loads((state_dir / "status.json").read_text())
    assert not state["active"] and state["storage_blocked"]
    assert not (tmp_path / "missing").exists()
    unit = (ROOT / "config/services/so101-recording.service").read_text()
    assert "RestartPreventExitStatus=2 78" in unit


def test_recorder_finalizes_external_file_and_reads_local_episode_metadata(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "recorder_success_test", ROOT / "scripts/record_mcap.py"
    )
    assert module_spec and module_spec.loader
    recorder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(recorder)
    output = tmp_path / "external/mcap"
    state_dir = tmp_path / "local/state"
    state_dir.mkdir(parents=True)
    episode = {"id": "episode-1", "operator_id": "private", "task": "pick block"}
    (state_dir / "episodes.json").write_text(json.dumps({"active": episode, "episodes": []}))
    profile = write_profile(
        tmp_path / "profile.json", output=str(output), state_dir=str(state_dir), min_free_gib=0.001
    )
    monkeypatch.setattr(
        sys, "argv", ["record_mcap.py", "--storage-profile", str(profile), "--duration", "1.5"]
    )
    monkeypatch.setattr(recorder.signal, "signal", lambda *args: None)
    moments = iter([0, 1, 2])
    monkeypatch.setattr(recorder.time, "monotonic", lambda: next(moments))
    stop = SimpleNamespace(is_set=lambda: False, set=lambda: None, wait=lambda *args: None)
    monkeypatch.setattr(recorder.threading, "Event", lambda: stop)
    monkeypatch.setattr(
        recorder.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        recorder, "fetch", lambda route: b'{"events": []}' if "/api/events?" in route else b"{}"
    )
    logged = []

    def open_writer(path):
        path.write_bytes(b"MCAP fixture")
        return SimpleNamespace(flush=lambda: None, close=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "foxglove",
        SimpleNamespace(
            open_mcap=open_writer, log=lambda topic, data: logged.append((topic, data))
        ),
    )
    previous = recorder.os.umask(0o077)
    try:
        assert recorder.main() == 0
    finally:
        recorder.os.umask(previous)
    state = json.loads((state_dir / "status.json").read_text())
    final = Path(state["last_completed_path"])
    assert final.parent == output and final.read_bytes() == b"MCAP fixture"
    assert not state["active"] and not list(output.glob("*.active"))
    assert not (output / "status.json").exists()
    annotations = next(data for topic, data in logged if topic == "/recording/episodes")
    assert annotations["active"]["id"] == "episode-1"
    assert "operator_id" not in annotations["active"]
    assert json.loads((state_dir / "episodes.json").read_text())["active"] == episode


def test_recorder_separates_raw_output_from_local_episode_metadata(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "recorder_split_storage_test", ROOT / "scripts/record_mcap.py"
    )
    assert module_spec and module_spec.loader
    recorder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(recorder)
    output = tmp_path / "external/mcap"
    state_dir = tmp_path / "local/state"
    state_dir.mkdir(parents=True)
    annotations = {"active": {"id": "episode-1", "operator_id": "private"}, "episodes": []}
    (state_dir / "episodes.json").write_text(json.dumps(annotations))
    profile = write_profile(
        tmp_path / "profile.json", output=str(output), state_dir=str(state_dir), min_free_gib=0.001
    )
    monkeypatch.setattr(
        sys, "argv", ["record_mcap.py", "--storage-profile", str(profile), "--duration", "1"]
    )
    monkeypatch.setattr(recorder.signal, "signal", lambda *args: None)
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(recorder.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        recorder.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kw: None),
    )
    messages = []

    class Writer:
        def __init__(self, path):
            self.path = path
            path.write_bytes(b"test recording")

        def close(self):
            pass

        def flush(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "foxglove",
        SimpleNamespace(open_mcap=Writer, log=lambda topic, value: messages.append((topic, value))),
    )
    monkeypatch.setattr(
        recorder, "fetch", lambda url: b'{"events": []}' if "/api/events?" in url else b"{}"
    )
    previous = recorder.os.umask(0o077)
    try:
        assert recorder.main() == 0
    finally:
        recorder.os.umask(previous)
    completed = list(output.glob("*.mcap"))
    assert len(completed) == 1 and not list(output.glob("*.active"))
    status = json.loads((state_dir / "status.json").read_text())
    assert status["last_completed_path"] == str(completed[0]) and not status["active"]
    assert not (output / "status.json").exists()
    saved_annotations = json.loads((state_dir / "episodes.json").read_text())
    assert saved_annotations == annotations
    published = [value for topic, value in messages if topic == "/recording/episodes"]
    assert published and all(
        item == {"active": {"id": "episode-1"}, "episodes": []} for item in published
    )

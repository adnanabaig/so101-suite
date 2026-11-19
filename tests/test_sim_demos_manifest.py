"""``sim_demos.json``: written durably, and never silently wiped (data review finding 9)."""

from __future__ import annotations

import json
import os
import threading

import pytest

from so101 import atomic_io, cli_sim
from so101.config import get_settings
from so101.data.dataset import (
    FALLBACK_PUBLISH_TRANSACTION,
    DatasetBusyError,
    DatasetIdConflictError,
    dataset_root,
)
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.training import toy_reach as toy_reach_module
from so101.training.toy_reach import generate_toy_reach_episodes, write_toy_reach_dataset


def _path(repo_id: str):
    root = dataset_root(repo_id)
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta.json"
    if not meta.exists():
        meta.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "format": "so101-fallback-v2",
                    "fps": 30,
                    "task": "Perform the task.",
                    "camera_keys": [],
                    "episodes": 0,
                    "total_frames": 0,
                }
            ),
            encoding="utf-8",
        )
    return root / cli_sim.DEMOS_MANIFEST


def _run(created_at: float) -> dict:
    return {"created_at": created_at, "task": "reach", "episodes": []}


def _noncanonical_dataset(repo_id: str):
    root = get_settings().datasets_dir / "zz-imported-dataset"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "format": "so101-fallback-v2",
                "fps": 30,
                "task": "Perform the task.",
                "camera_keys": [],
                "episodes": 0,
                "total_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_first_party_metadata_writers_resolve_noncanonical_dataset_root():
    repo_id = "sim/noncanonical"
    imported = _noncanonical_dataset(repo_id)
    canonical = dataset_root(repo_id)

    manifest = cli_sim._save_manifest(repo_id, _run(1.0))
    assert manifest.parent == imported
    assert not canonical.exists()

    with pytest.raises(FileExistsError) as refused:
        write_toy_reach_dataset(repo_id, generate_toy_reach_episodes(1, seed=0))
    assert str(imported) in str(refused.value)
    assert not canonical.exists()


def test_toy_dataset_publication_excludes_episode_recorder(monkeypatch):
    repo_id = "toy/concurrent-writer"
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    write_json = toy_reach_module._write_json
    first = True

    def blocked_first_metadata(path, payload):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(5), "test did not release toy metadata write"
        return write_json(path, payload)

    def write() -> None:
        try:
            write_toy_reach_dataset(repo_id, generate_toy_reach_episodes(1, seed=0))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(toy_reach_module, "_write_json", blocked_first_metadata)
    thread = threading.Thread(target=write)
    thread.start()
    try:
        assert entered.wait(5), "toy writer did not reach metadata publication"
        with pytest.raises(DatasetBusyError):
            EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert (dataset_root(repo_id) / "meta.json").is_file()


@pytest.mark.parametrize("marker_hidden", [False, True])
def test_toy_dataset_refuses_case_alias_under_publication_lock(marker_hidden):
    existing_id = "toy/case-alias"
    requested_id = "TOY/CASE-ALIAS"
    episodes = generate_toy_reach_episodes(1, seed=0)
    existing_root = write_toy_reach_dataset(existing_id, episodes)
    if marker_hidden:
        (existing_root / FALLBACK_PUBLISH_TRANSACTION).write_text("{}", encoding="utf-8")
    requested_root = dataset_root(requested_id)

    with pytest.raises(DatasetIdConflictError, match="differs.*only in letter case"):
        write_toy_reach_dataset(requested_id, episodes)

    matching_roots = [
        child
        for child in get_settings().datasets_dir.iterdir()
        if child.name.casefold() == existing_root.name.casefold()
    ]
    assert matching_roots == [existing_root]
    if requested_root.resolve() != existing_root.resolve():
        assert not requested_root.exists()


@pytest.mark.parametrize(
    "content",
    [b'{"schema_version": 1, "runs": [', b"\xff\xfe", b"[]", b'{"runs": "x"}', b'{"runs": [1]}'],
)
def test_an_unreadable_manifest_is_kept_aside_and_reported(content, capsys):
    path = _path("sim/corrupt")
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    assert cli_sim._save_manifest("sim/corrupt", _run(1.0)) == path
    [backup] = path.parent.glob(f"{cli_sim.DEMOS_MANIFEST}.corrupt-*")
    assert backup.read_bytes() == content  # the old runs are not lost
    assert json.loads(path.read_text())["runs"] == [_run(1.0)]
    message = " ".join(capsys.readouterr().err.split())
    assert "could not be read" in message and f"kept it as {backup.name}" in message
    # A second corrupt file the same second gets its own name.
    path.write_bytes(content)
    cli_sim._save_manifest("sim/corrupt", _run(2.0))
    assert len(list(path.parent.glob(f"{cli_sim.DEMOS_MANIFEST}.corrupt-*"))) == 2


def test_a_readable_manifest_is_extended_in_place(capsys):
    _path("sim/ok")
    cli_sim._save_manifest("sim/ok", _run(1.0))
    cli_sim._save_manifest("sim/ok", _run(2.0))
    cli_sim._save_manifest("sim/ok", {**_run(2.0), "task": "pick_lift"})  # same run, updated
    manifest = json.loads(_path("sim/ok").read_text())
    assert manifest["schema_version"] == 1
    assert [(r["created_at"], r["task"]) for r in manifest["runs"]] == [
        (1.0, "reach"),
        (2.0, "pick_lift"),
    ]
    assert not list(_path("sim/ok").parent.glob("*.corrupt-*"))
    assert capsys.readouterr().err == ""


def test_concurrent_manifest_updates_hold_one_dataset_lock(monkeypatch):
    path = _path("sim/concurrent")
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []
    write_json = cli_sim._write_json
    first = True

    def blocked_first_write(destination, payload):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(5), "test did not release first manifest write"
        return write_json(destination, payload)

    def save(run, *, second=False):
        if second:
            second_started.set()
        try:
            cli_sim._save_manifest("sim/concurrent", run)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if second:
                second_done.set()

    monkeypatch.setattr(cli_sim, "_write_json", blocked_first_write)
    first_thread = threading.Thread(target=save, args=(_run(1.0),))
    second_thread = threading.Thread(target=save, args=(_run(2.0),), kwargs={"second": True})
    first_thread.start()
    try:
        assert entered.wait(5), "first manifest write did not block"
        second_thread.start()
        assert second_started.wait(5)
        assert not second_done.wait(0.1), "second update bypassed the dataset lock"
    finally:
        release.set()
        first_thread.join(timeout=5)
        if second_thread.ident is not None:
            second_thread.join(timeout=5)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert errors == []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["created_at"] for entry in manifest["runs"]] == [1.0, 2.0]


def test_an_unreadable_file_that_is_not_corrupt_is_left_alone(monkeypatch):
    path = _path("sim/locked")
    path.parent.mkdir(parents=True)
    path.write_text('{"runs": []}')
    monkeypatch.setattr(
        type(path), "read_text", lambda self, **kwargs: (_ for _ in ()).throw(PermissionError("no"))
    )
    with pytest.raises(PermissionError):
        cli_sim._save_manifest("sim/locked", _run(1.0))
    assert not list(path.parent.glob("*.corrupt-*"))


def test_the_manifest_is_fsynced_and_its_rename_persisted(monkeypatch):
    synced: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def fsync(fd):
        info = os.fstat(fd)
        synced.append((info.st_dev, info.st_ino))
        return real_fsync(fd)

    monkeypatch.setattr(atomic_io.os, "fsync", fsync)
    path = _path("sim/durable")
    assert cli_sim._save_manifest("sim/durable", _run(1.0)) == path
    file_id = (os.stat(path).st_dev, os.stat(path).st_ino)
    assert file_id in synced
    if os.name != "nt":
        directory = os.stat(path.parent)
        assert (directory.st_dev, directory.st_ino) in synced


def test_strict_json_is_kept():
    path = _path("sim/nan")
    with pytest.raises(ValueError):
        cli_sim._save_manifest("sim/nan", {**_run(1.0), "sim_time_s": float("nan")})
    assert not path.exists()

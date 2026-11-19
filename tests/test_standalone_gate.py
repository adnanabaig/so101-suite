"""The standalone gate: commands that open an arm without the server (finding 1).

``so101 teleop``, ``record``, ``replay``, ``run``, ``eval`` and ``calibrate``,
and the library calls that open the follower themselves (``replay_episode``
without a follower, ``PolicyRunner.connect``, ``EvalRun``), must read the
safety latches the server persists before anything is opened, refuse while
one is set or the store is unreadable, and on the real backend take the
canonical device-global locks (refusing while a server answers). The server
takes the same locks for its lifetime. Mock stays unaffected.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from so101 import cli as cli_module
from so101.bringup import ownership, standalone
from so101.bringup.ownership import BusLock, BusOwnershipError, ServerProbe, bus_lock_path
from so101.bringup.standalone import StandaloneAccess, StandaloneRefused, standalone_access
from so101.cli import EXIT_REFUSED, app
from so101.config import get_settings
from so101.hardware.follower import SO101Follower
from so101.hardware.leader import SO101Leader
from so101.hardware.motors import FeetechBus, TorqueOffUnconfirmed
from so101.services import store as control_store_module
from so101.services.store import (
    ControlStore,
    control_store_path,
    legacy_latch_marker_path,
    persist_standalone_torque_off_failure,
    safety_latch_identity,
)

runner = CliRunner()

IDLE = ServerProbe(running=False, url="http://127.0.0.1:8000/api/robot/status", detail="idle")
RUNNING = ServerProbe(
    running=True,
    url="http://127.0.0.1:8000/api/robot/status",
    detail="HTTP 200 from http://127.0.0.1:8000/api/robot/status",
    http_status=200,
    status={"estopped": False, "mode": "idle"},
)

#: Every CLI path that can energize or command an arm, with arguments that
#: would reach the arm on a clear store.
CLI_PATHS = {
    "teleop": ["teleop", "--duration", "0.05"],
    "record": [
        "record",
        "gate/demo",
        "--episodes",
        "1",
        "--episode-seconds",
        "0.05",
        "--no-prompt",
    ],
    "replay": ["replay", "gate/demo", "--episode", "0"],
    "run": ["run", "no-such-model", "--duration", "0.05"],
    "eval": ["eval", "no-such-model", "--episodes", "1", "--max-seconds", "0.05"],
    "calibrate": ["calibrate", "follower", "--yes"],
}


class Opened(Exception):
    """An arm or bus connect was reached."""


def _flat(text: str) -> str:
    return " ".join(text.split())


def _set_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", backend)
    monkeypatch.setenv("SO101_CAMERAS", "")  # no real camera to open
    get_settings.cache_clear()


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every arm/bus connect; each one raises ``Opened`` (nothing real opens)."""

    calls: list[str] = []

    def connect(name: str):
        def _connect(self: Any, *args: Any, **kwargs: Any) -> None:
            calls.append(name)
            raise Opened(name)

        return _connect

    monkeypatch.setattr(SO101Follower, "connect", connect("follower"))
    monkeypatch.setattr(SO101Leader, "connect", connect("leader"))
    monkeypatch.setattr(FeetechBus, "connect", connect("bus"))
    return calls


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> list[ServerProbe]:
    """The server probe answers from this list (idle by default); never the network."""

    answers = [IDLE]
    monkeypatch.setattr(standalone, "server_probe", lambda settings: answers[0])
    return answers


@pytest.fixture
def real(
    monkeypatch: pytest.MonkeyPatch,
    probe: list[ServerProbe],
) -> None:
    _set_backend(monkeypatch, "real")


@pytest.fixture
def real_motion_internal_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise teardown internals that public real-motion entry points now refuse."""

    import so101.data.replay as replay_module
    import so101.policy.runner as policy_runner_module

    def low_level_access(purpose: str, **_kwargs: Any) -> StandaloneAccess:
        access = standalone_access(purpose)
        access._real_motion_allowed = True
        return access

    monkeypatch.setattr(StandaloneAccess, "_require_motion_allowed", lambda _self: None)
    monkeypatch.setattr(cli_module, "_standalone_gate", low_level_access)
    monkeypatch.setattr(replay_module, "standalone_access", low_level_access)
    monkeypatch.setattr(policy_runner_module, "standalone_access", low_level_access)


def _latch(**fields: Any) -> Path:
    path = control_store_path(get_settings())
    latches = {"estopped": False, "fault": None, "torque_unconfirmed": False, **fields}
    ControlStore(path).save_latches(latches, version=10)
    return path


def _marker_path() -> Path:
    from so101.services.store import latch_marker_path

    settings = get_settings()
    return latch_marker_path(control_store_path(settings), safety_latch_identity(settings))


def _legacy_marker_path() -> Path:
    return legacy_latch_marker_path(control_store_path(get_settings()))


def _write_marker(*, version: int, **latches: Any) -> Path:
    path = _marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "kind": "so101.latch-unsaved",
                "schema_version": 2,
                "identity": safety_latch_identity(get_settings()).as_dict(),
                "version": version,
                "latches": {
                    "estopped": False,
                    "fault": None,
                    "torque_unconfirmed": False,
                    **latches,
                },
                "error": "OSError: disk refused the latch",
            }
        ),
        encoding="utf-8",
    )
    return path


def _assert_failed_torque_off_was_latched() -> None:
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None
    assert saved["torque_unconfirmed"] is True
    assert "torque-off unconfirmed" in str(saved["fault"])
    assert _lock_is_free()

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("the next standalone command")
    assert refused.value.reason == "latched"
    assert "torque-off unconfirmed" in refused.value.detail
    assert _lock_is_free()


def _lock_is_free() -> bool:
    try:
        BusLock(bus_lock_path(get_settings()), purpose="test probe").acquire().release()
    except BusOwnershipError:
        return False
    return True


def _record_demo_dataset(repo_id: str = "gate/demo") -> None:
    """A one-episode fallback dataset, recorded on the mock arm."""

    from so101.data.recorder import EpisodeRecorder, RecordingConfig
    from so101.teleop.controller import TeleopController

    rec = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=30, camera_keys=[]))
    with TeleopController(SO101Leader(), SO101Follower(), fps=120) as tc:
        for step in itertools.islice(tc.run(), 3):
            rec.add(step)
    rec.save_episode()


# --- the CLI ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["teleop", "--duration", "nan"],
        ["teleop", "--fps", "0"],
        ["record", "gate/demo", "--episode-seconds", "nan"],
        ["record", "gate/demo", "--reset-seconds", "nan"],
        ["record", "gate/demo", "--fps", "241"],
        ["run", "no-such-model", "--duration", "nan"],
        ["eval", "no-such-model", "--max-seconds", "nan"],
    ],
)
def test_invalid_motion_timing_is_rejected_before_the_standalone_gate(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    def unexpected_gate(purpose: str, **_kwargs: Any) -> None:
        raise AssertionError(f"standalone gate reached for {purpose}")

    monkeypatch.setattr(cli_module, "_standalone_gate", unexpected_gate)

    result = runner.invoke(app, argv)

    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


@pytest.mark.parametrize("command", list(CLI_PATHS))
def test_a_latched_store_blocks_every_cli_path_before_any_connect(
    command: str, real: None, opened: list[str]
) -> None:
    store = _latch(estopped=True, fault="Halt could not hold position", torque_unconfirmed=True)

    result = runner.invoke(app, CLI_PATHS[command])

    assert result.exit_code == EXIT_REFUSED, result.output
    assert opened == []
    err = _flat(result.stderr)
    assert "refused:" in err
    assert "E-stop" in err and "fault: Halt could not hold position" in err
    assert "torque-off unconfirmed" in err
    assert str(store) in err
    # How to clear it, in the order the server accepts the resets.
    assert "POST /api/robot/estop/reset, then POST /api/robot/fault/reset" in err
    assert '{"operator_id": "<your name>", "confirm": true}' in err
    assert '"power_cut_acknowledged": true' in err
    assert "so101 serve" in err
    assert _lock_is_free()  # a refusal gives bus.lock back


@pytest.mark.parametrize(
    ("fields", "named", "resets"),
    [
        ({"estopped": True}, "E-stop", "POST /api/robot/estop/reset"),
        ({"fault": "bus I/O failed"}, "fault: bus I/O failed", "POST /api/robot/fault/reset"),
        ({"torque_unconfirmed": True}, "torque-off unconfirmed", "POST /api/robot/fault/reset"),
    ],
)
def test_each_latch_alone_blocks_and_names_its_own_reset(
    fields: dict[str, Any], named: str, resets: str, real: None, opened: list[str]
) -> None:
    _latch(**fields)

    result = runner.invoke(app, CLI_PATHS["teleop"])

    assert result.exit_code == EXIT_REFUSED
    assert opened == []
    err = _flat(result.stderr)
    assert named in err and resets in err
    if "estopped" not in fields:
        assert "estop/reset" not in err


@pytest.mark.parametrize("command", ["calibrate"])
def test_a_clear_store_lets_the_command_reach_the_arm_holding_bus_lock(
    command: str, real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _latch()  # saved, nothing latched
    seen: list[bool] = []

    def connect(self: Any, *args: Any, **kwargs: Any) -> None:
        seen.append(_lock_is_free())
        raise Opened("arm")

    monkeypatch.setattr(SO101Leader, "connect", connect)
    monkeypatch.setattr(SO101Follower, "connect", connect)
    monkeypatch.setattr(FeetechBus, "connect", connect)

    result = runner.invoke(app, CLI_PATHS[command])

    assert seen == [False]  # it got to the arm, with bus.lock held
    assert isinstance(result.exception, Opened) or result.exit_code == 1, result.output
    assert _lock_is_free()  # and gave it back


@pytest.mark.parametrize("command", ["teleop", "record", "replay", "run", "eval"])
def test_a_clear_real_store_still_refuses_standalone_motion(
    command: str, real: None, opened: list[str]
) -> None:
    _latch()

    result = runner.invoke(app, CLI_PATHS[command])

    assert result.exit_code == EXIT_REFUSED
    assert opened == []
    assert "real standalone motion is disabled" in _flat(result.stderr)
    assert _lock_is_free()


def test_no_store_file_means_nothing_was_ever_latched(real: None, opened: list[str]) -> None:
    assert not control_store_path(get_settings()).exists()

    result = runner.invoke(app, CLI_PATHS["teleop"])

    assert result.exit_code == EXIT_REFUSED
    assert opened == []
    assert "real standalone motion is disabled" in _flat(result.stderr)
    assert not control_store_path(get_settings()).exists()  # the gate creates nothing


def test_a_real_bringup_access_cannot_be_repurposed_to_energize(real: None) -> None:
    access = standalone_access("misdeclared bring-up operation")
    try:
        with pytest.raises(StandaloneRefused, match="real standalone motion is disabled"):
            access.prepare_to_energize()
    finally:
        access.release()
    assert _lock_is_free()


def test_an_unreadable_store_blocks(real: None, opened: list[str]) -> None:
    path = control_store_path(get_settings())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a sqlite database" * 100)

    result = runner.invoke(app, CLI_PATHS["teleop"])

    assert result.exit_code == EXIT_REFUSED
    assert opened == []
    err = _flat(result.stderr)
    assert "could not read the persisted safety latches" in err and str(path) in err
    assert "fails closed" in err


def test_a_newer_unsaved_latch_marker_blocks_standalone_hardware(real: None) -> None:
    _latch()  # the last SQLite snapshot was clear
    marker = _write_marker(version=11, estopped=True, fault="unsaved bus failure")

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("so101 teleop")

    assert refused.value.reason == "latched"
    assert "E-stop" in refused.value.detail
    assert "unsaved bus failure" in refused.value.detail
    assert str(marker) in refused.value.detail
    assert _lock_is_free()


def test_an_unreadable_unsaved_latch_marker_fails_closed(real: None) -> None:
    _latch()
    marker = _marker_path()
    marker.write_bytes(b"not valid JSON")

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("so101 teleop")

    assert "unreadable" in refused.value.detail
    assert str(marker) in refused.value.detail
    assert _lock_is_free()


def test_a_marker_older_than_the_stored_snapshot_is_ignored(real: None) -> None:
    _latch()  # version 10
    _write_marker(version=9, estopped=True, fault="obsolete")

    access = standalone_access("so101 teleop")

    assert access.latches is not None and not standalone.latched_items(access.latches)
    access.release()
    assert _lock_is_free()


def test_a_running_server_blocks_under_real_without_taking_the_lock(
    real: None,
    probe: list[ServerProbe],
    opened: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe[0] = RUNNING
    attempts: list[str] = []
    original = ownership.DeviceBusLock.acquire
    monkeypatch.setattr(
        ownership.DeviceBusLock,
        "acquire",
        lambda self: (attempts.append(self.purpose), original(self))[1],
    )

    result = runner.invoke(app, CLI_PATHS["teleop"])

    assert result.exit_code == EXIT_REFUSED
    assert opened == []
    err = _flat(result.stderr)
    assert "a Suite server appears to be running" in err
    assert "outside the server's E-stop" in err
    assert attempts == []


def test_a_held_bus_lock_blocks_and_names_the_holder(real: None, opened: list[str]) -> None:
    with BusLock(bus_lock_path(get_settings()), purpose="so101 preflight"):
        result = runner.invoke(app, CLI_PATHS["replay"])

    assert result.exit_code == EXIT_REFUSED
    assert opened == []
    assert "so101 preflight" in _flat(result.stderr)


def test_mock_is_unaffected_by_a_latched_store(opened: list[str]) -> None:
    _latch(estopped=True, fault="left over from a real run")

    access = standalone_access("test")

    assert access.backend == "mock" and not access.holds_bus_lock
    access.release()


def test_mock_cli_replay_still_runs_with_a_latched_store() -> None:
    _record_demo_dataset()
    _latch(estopped=True)

    result = runner.invoke(app, ["replay", "gate/demo", "--episode", "0", "--fps", "240"])

    assert result.exit_code == 0, result.output
    assert "replayed 3 frames" in result.output


@pytest.mark.parametrize("stop_early", [False, True])
def test_record_normal_halt_always_emits_the_supervision_notice(
    stop_early: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[str] = []
    monkeypatch.setenv("SO101_CAMERAS", "")
    get_settings.cache_clear()
    monkeypatch.setattr(cli_module, "_standalone_halt_notice", lambda: notices.append("halt"))
    if stop_early:
        monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    episodes = "2" if stop_early else "1"

    result = runner.invoke(
        app,
        [
            "record",
            "notice/demo",
            "--episodes",
            episodes,
            "--episode-seconds",
            "0.02",
            "--reset-seconds",
            "0",
            "--fps",
            "240",
        ],
    )

    assert result.exit_code == 0, result.output
    assert notices == ["halt"]


def test_eval_report_failure_after_halt_still_emits_supervision_notice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import so101.eval.evaluator as evaluator
    import so101.policy.checkpoint as checkpoint

    notices: list[str] = []

    class Info:
        path = tmp_path
        task = ""

    class FailedPublicationRun:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.ended_holding = False
            self.report = None
            self.report_path = None

        def run(self) -> None:
            self.ended_holding = True
            raise RuntimeError("terminal report publication failed")

    monkeypatch.setattr(cli_module, "_standalone_halt_notice", lambda: notices.append("halt"))
    monkeypatch.setattr(checkpoint, "preflight_policy", lambda path, task=None: Info())
    monkeypatch.setattr(checkpoint, "resolve_control_fps", lambda requested, info: 30)
    monkeypatch.setattr(evaluator, "EvalRun", FailedPublicationRun)

    result = runner.invoke(
        app,
        ["eval", str(tmp_path), "--episodes", "1", "--max-seconds", "0.01"],
    )

    assert result.exit_code == 1
    assert "terminal report publication failed" in result.output
    assert notices == ["halt"]


def test_mock_torque_failure_does_not_create_a_hardware_latch() -> None:
    class Follower:
        def disconnect(self) -> None:
            raise TorqueOffUnconfirmed("mock follower", ["shoulder_pan"])

    access = standalone_access("mock command")
    with pytest.raises(TorqueOffUnconfirmed):
        access.disconnect_follower(Follower())
    access.release()

    assert not control_store_path(get_settings()).exists()


def test_sim_is_gated_on_the_latches_but_takes_no_bus_lock(
    monkeypatch: pytest.MonkeyPatch, probe: list[ServerProbe]
) -> None:
    _set_backend(monkeypatch, "sim")
    probe[0] = RUNNING  # the sim never opens a real bus, so a server does not matter
    _latch(estopped=True)

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("so101 teleop")
    assert refused.value.reason == "latched"
    assert "Then re-run this command" in refused.value.remediation
    assert "holds the bus" not in refused.value.remediation

    # A newer snapshot with nothing latched (what a deliberate reset writes).
    ControlStore(control_store_path(get_settings())).save_latches({}, version=20)
    access = standalone_access("so101 teleop")
    assert access.backend == "sim" and not access.holds_bus_lock


@pytest.mark.parametrize("rehearsal_backend", ["mock", "sim"])
def test_a_rehearsal_snapshot_cannot_clear_a_real_arm_latch(
    rehearsal_backend: str,
    monkeypatch: pytest.MonkeyPatch,
    probe: list[ServerProbe],
) -> None:
    """Mock/sim may share a data folder, but not the real arm's latch row."""

    _set_backend(monkeypatch, "real")
    path = control_store_path(get_settings())
    real_store = ControlStore(path)
    assert real_store.save_latches(
        {
            "estopped": True,
            "fault": "real follower bus failed",
            "safety_state": "estopped",
            "torque_unconfirmed": True,
        },
        version=10,
    )

    _set_backend(monkeypatch, rehearsal_backend)
    rehearsal_store = ControlStore(path)
    assert rehearsal_store.load_latches() is None
    assert rehearsal_store.save_latches({}, version=20)

    _set_backend(monkeypatch, "real")
    restored = ControlStore(path).load_latches()
    assert restored is not None
    assert restored["estopped"] is True
    assert restored["fault"] == "real follower bus failed"
    assert restored["torque_unconfirmed"] is True


def test_legacy_unscoped_latches_migrate_to_real_even_when_mock_opens_first(
    monkeypatch: pytest.MonkeyPatch,
    probe: list[ServerProbe],
) -> None:
    """An upgrade rehearsal must not adopt and then clear an old real latch."""

    _set_backend(monkeypatch, "mock")
    path = control_store_path(get_settings())
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE safety_latches (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                estopped INTEGER NOT NULL,
                fault TEXT,
                safety_state TEXT NOT NULL,
                torque_unconfirmed INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO safety_latches VALUES (1, 7, 'legacy', 1, ?, 'estopped', 1)",
            ("legacy real-arm failure",),
        )

    assert ControlStore(path).load_latches() is None

    _set_backend(monkeypatch, "real")
    restored = ControlStore(path).load_latches()
    assert restored is not None
    assert restored["estopped"] is True
    assert restored["fault"] == "legacy real-arm failure"
    assert restored["torque_unconfirmed"] is True


def test_real_identity_b_cannot_retire_identity_a_fallback_marker(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101.services.robot_manager import RobotManager

    monkeypatch.setenv("SO101_FOLLOWER_ID", "arm-a")
    monkeypatch.setenv("SO101_LEADER_ID", "leader-a")
    get_settings.cache_clear()
    save_latches = ControlStore.save_latches

    def refuse_save(self: ControlStore, latches: dict[str, Any], *, version: int) -> bool:
        raise OSError("A's store write failed")

    monkeypatch.setattr(ControlStore, "save_latches", refuse_save)
    persist_standalone_torque_off_failure(
        get_settings(),
        TorqueOffUnconfirmed("follower", ["shoulder_pan"]),
        device_name="follower",
    )
    marker_a = _marker_path()
    assert marker_a.is_file()

    monkeypatch.setattr(ControlStore, "save_latches", save_latches)
    monkeypatch.setenv("SO101_FOLLOWER_ID", "arm-b")
    monkeypatch.setenv("SO101_LEADER_ID", "leader-b")
    get_settings.cache_clear()
    store_b = ControlStore(control_store_path(get_settings()))
    assert store_b.save_latches({}, version=time.time_ns() // 1_000)
    manager_b = RobotManager(store=store_b)
    manager_b._persist_latches()
    manager_b._watchdog_shutdown.set()
    manager_b._health_shutdown.set()
    access_b = standalone_access("identity B")
    access_b.release()
    assert marker_a.is_file()

    monkeypatch.setenv("SO101_FOLLOWER_ID", "arm-a")
    monkeypatch.setenv("SO101_LEADER_ID", "leader-a")
    get_settings.cache_clear()
    with pytest.raises(StandaloneRefused, match="torque-off unconfirmed"):
        standalone_access("identity A")
    assert marker_a.is_file()


def test_sim_ignores_a_legacy_real_arm_marker(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old unscoped marker is conservatively real, never a rehearsal latch."""

    _set_backend(monkeypatch, "real")
    marker = _legacy_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "kind": "so101.latch-unsaved",
                "version": 11,
                "latches": {
                    "estopped": True,
                    "fault": "legacy real-arm marker",
                    "torque_unconfirmed": False,
                },
                "error": "legacy store failure",
            }
        ),
        encoding="utf-8",
    )
    _set_backend(monkeypatch, "sim")

    access = standalone_access("sim rehearsal")
    access.release()

    _set_backend(monkeypatch, "real")
    with pytest.raises(StandaloneRefused, match="legacy real-arm marker"):
        standalone_access("real arm")
    assert marker.is_file()


def test_corrupt_legacy_marker_cannot_hide_or_clear_scoped_estop(
    real: None,
) -> None:
    from so101.services.robot_manager import ControlConflict, RobotManager

    scoped = _write_marker(
        version=time.time_ns() // 1_000,
        estopped=True,
        fault="identity-scoped E-stop",
    )
    legacy = _legacy_marker_path()
    legacy.write_bytes(b"corrupt legacy marker")

    manager = RobotManager(store=ControlStore(control_store_path(get_settings())))
    try:
        assert manager.guard.estopped is True
        assert manager.fault is not None and "identity-scoped E-stop" in manager.fault
        with pytest.raises(ControlConflict, match="reset the E-stop"):
            manager.clear_fault(operator_id="operator")
        assert scoped.is_file()
        assert legacy.is_file()
    finally:
        manager._watchdog_shutdown.set()
        manager._health_shutdown.set()


# --- the library paths ------------------------------------------------------------------


class _StillPolicy:
    def reset(self) -> None:
        pass

    def select_action(self, observation: dict[str, Any]) -> dict[str, float]:
        return {}


class _NoCameras:
    streams: list[str] = []

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self) -> dict[str, Any]:
        return {}


def test_replay_episode_without_a_follower_is_gated(
    monkeypatch: pytest.MonkeyPatch, probe: list[ServerProbe]
) -> None:
    from so101.data.replay import replay_episode

    _record_demo_dataset()  # on the mock arm
    _set_backend(monkeypatch, "real")
    _latch(estopped=True)
    connects: list[str] = []
    monkeypatch.setattr(SO101Follower, "connect", lambda self, **kw: connects.append("follower"))

    with pytest.raises(StandaloneRefused, match="E-stop"):
        replay_episode("gate/demo", 0)
    assert connects == []
    assert _lock_is_free()


def test_replay_episode_refuses_owned_real_motion_before_connect(
    monkeypatch: pytest.MonkeyPatch, probe: list[ServerProbe]
) -> None:
    from so101.data.replay import replay_episode

    _record_demo_dataset()
    _set_backend(monkeypatch, "real")
    seen: list[bool] = []

    def connect(self: Any, *args: Any, **kwargs: Any) -> None:
        seen.append(_lock_is_free())
        raise Opened("follower")

    monkeypatch.setattr(SO101Follower, "connect", connect)

    with pytest.raises(StandaloneRefused, match="real standalone motion is disabled"):
        replay_episode("gate/demo", 0)
    assert seen == []
    assert _lock_is_free()


def test_replay_episode_with_a_callers_follower_is_refused_on_real(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follower passed in (a caller that owns the connection) is used as it is."""

    import so101.data.replay as replay_mod

    _latch(estopped=True)
    goals: list[dict[str, float]] = []

    class CallerFollower:
        connected = True

        def send_goal(self, goal: dict[str, float]) -> dict[str, float]:
            goals.append(goal)
            return goal

    class Dataset:
        format = "so101-fallback-v1"
        path = "."
        fps = 240
        repo_id = "gate/demo"

    monkeypatch.setattr(replay_mod, "require_dataset", lambda repo_id: Dataset())
    monkeypatch.setattr(
        replay_mod, "_load_fallback_actions", lambda root, ep: [[0.0] * 6, [1.0] * 6]
    )

    follower: Any = CallerFollower()
    with pytest.raises(StandaloneRefused, match="caller-owned follower"):
        replay_mod.replay_episode("gate/demo", 0, follower=follower, fps=240)
    assert goals == []
    assert _lock_is_free()


def test_policy_runner_connect_is_gated(real: None, opened: list[str]) -> None:
    from so101.policy.runner import PolicyRunner

    _latch(fault="servo overheated")
    policy_runner = PolicyRunner(_StillPolicy(), cameras=_NoCameras())  # type: ignore[arg-type]

    with pytest.raises(StandaloneRefused, match="servo overheated"):
        policy_runner.rollout(0.05)
    with pytest.raises(StandaloneRefused):
        policy_runner.connect()
    assert opened == []
    assert _lock_is_free()


def test_eval_run_is_gated_before_its_first_connect(
    real: None, opened: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import so101.eval.evaluator as evaluator
    from so101.eval.evaluator import EvalConfig, EvalRun

    monkeypatch.setattr(evaluator, "load_policy", lambda path, task=None: _StillPolicy())
    _latch(estopped=True)

    run = EvalRun(EvalConfig(model_path=str(tmp_path), n_episodes=1, max_seconds=0.05), save=True)
    with pytest.raises(StandaloneRefused):
        run.run()
    assert opened == []
    assert run.report_path is None  # nothing ran, nothing saved
    assert _lock_is_free()


def test_policy_runner_releases_the_lock_after_a_rollout(
    monkeypatch: pytest.MonkeyPatch,
    probe: list[ServerProbe],
    real_motion_internal_harness: None,
) -> None:
    from so101.policy.runner import PolicyRunner

    _set_backend(monkeypatch, "real")
    held_during: list[bool] = []

    class Follower:
        connected = False
        torque_state = "off"
        torque_off_disconnects = 0
        halts = 0

        def connect(self) -> None:
            held_during.append(not _lock_is_free())
            self.connected = True
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.torque_off_disconnects += 1
            self.connected = False
            self.torque_state = "off"

        def halt_and_disconnect(self) -> None:
            self.halts += 1
            self.connected = False
            self.torque_state = "on"

        def read_positions(self) -> dict[str, float]:
            return {}

        def send_goal(self, goal: dict[str, float]) -> dict[str, float]:
            return goal

    follower = Follower()
    policy_runner = PolicyRunner(
        _StillPolicy(),  # type: ignore[arg-type]
        follower=follower,  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
        fps=200,
    )
    policy_runner.rollout(0.02)

    assert held_during == [True]
    assert follower.halts == 1
    assert follower.torque_off_disconnects == 0
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None and saved["energized"] is True
    assert _lock_is_free()

    with pytest.raises(StandaloneRefused, match="arm may be energized"):
        standalone_access("a restarted process")


def test_policy_runner_persists_failed_torque_off_before_releasing_the_lock(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    from so101.policy.runner import PolicyRunner

    saved_while_lock_free: list[bool] = []
    save_latches = ControlStore.save_latches

    def observed_save(self: ControlStore, latches: dict[str, Any], *, version: int) -> bool:
        saved_while_lock_free.append(_lock_is_free())
        return save_latches(self, latches, version=version)

    monkeypatch.setattr(ControlStore, "save_latches", observed_save)

    class Follower:
        connected = False

        def connect(self) -> None:
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    policy_runner = PolicyRunner(
        _StillPolicy(),  # type: ignore[arg-type]
        follower=Follower(),  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
    )
    policy_runner.connect()

    with pytest.raises(TorqueOffUnconfirmed):
        policy_runner.disconnect()

    assert saved_while_lock_free == [False]
    _assert_failed_torque_off_was_latched()


def test_write_ahead_energized_intent_is_durable_before_follower_connect(
    real: None,
    real_motion_internal_harness: None,
) -> None:
    from so101.policy.runner import PolicyRunner

    seen: list[dict[str, Any]] = []

    class Follower:
        connected = False
        torque_state = "off"

        def connect(self) -> None:
            saved = ControlStore(control_store_path(get_settings())).load_latches()
            assert saved is not None
            seen.append(saved)
            self.connected = True
            self.torque_state = "on"

        def halt_and_disconnect(self) -> None:
            self.connected = False
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.connected = False
            self.torque_state = "off"

        def read_positions(self) -> dict[str, float]:
            return {}

        def send_goal(self, goal: dict[str, float]) -> dict[str, float]:
            return goal

    PolicyRunner(
        _StillPolicy(),  # type: ignore[arg-type]
        follower=Follower(),  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
    ).rollout(0)

    assert len(seen) == 1
    assert seen[0]["energized"] is True
    assert seen[0]["energized_by"] == "PolicyRunner (policy rollout)"


def test_confirmed_failed_rollout_teardown_clears_energized_intent(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    real_motion_internal_harness: None,
) -> None:
    from so101.policy.runner import PolicyRunner

    class Follower:
        connected = False
        torque_state = "off"

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.connected = False
            self.torque_state = "off"

    def fail(self: PolicyRunner, duration: float = 30.0) -> Iterator[dict[str, float]]:
        raise RuntimeError("policy failed")
        yield  # pragma: no cover

    monkeypatch.setattr(PolicyRunner, "run", fail)
    runner_ = PolicyRunner(
        _StillPolicy(),  # type: ignore[arg-type]
        follower=Follower(),  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        runner_.rollout(0)

    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None and saved["energized"] is False
    access = standalone_access("recovery after confirmed torque-off")
    access.release()

    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "other-data-after-off"))
    get_settings.cache_clear()
    cross_data = standalone_access("same devices from another data directory")
    cross_data.release()


def test_server_restart_adopts_standalone_energized_intent_as_unconfirmed_fault(
    real: None,
    real_motion_internal_harness: None,
) -> None:
    from so101.services.robot_manager import RobotManager

    access = standalone_access("completed standalone teleop")
    access.prepare_to_energize()
    access.release()  # normal Halt/port release deliberately retains intent

    manager = RobotManager(store=ControlStore(control_store_path(get_settings())))
    try:
        assert manager.fault is not None
        assert "may have left the arm energized" in manager.fault
        status = manager.status()
        assert status["torque_unconfirmed"] is True
        assert status["torque_state"] == "unconfirmed"
    finally:
        manager._watchdog_shutdown.set()
        manager._health_shutdown.set()


def test_energized_halt_refuses_same_devices_from_another_data_directory(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    real_motion_internal_harness: None,
) -> None:
    first = standalone_access("data-dir A teleop")
    first.prepare_to_energize()
    first.release()  # represents successful Halt + host-port release

    other_data = tmp_path / "other-data-dir"
    monkeypatch.setenv("SO101_DATA_DIR", str(other_data))
    get_settings.cache_clear()

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("data-dir B replay")

    assert refused.value.reason == "latched"
    assert "host-global motor-device energized marker" in refused.value.detail
    assert "arm may be energized" in refused.value.detail
    assert not control_store_path(get_settings()).exists()


@pytest.mark.parametrize("latch", ["estop", "fault"])
def test_server_safety_latch_blocks_the_same_devices_from_another_data_directory(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    latch: str,
) -> None:
    from so101.services.robot_manager import RobotManager

    ownership_access = standalone_access("server safety-latch harness")
    manager = RobotManager(store=ControlStore(control_store_path(get_settings())))
    if latch == "estop":
        manager.estop("cross-data E-stop")
    else:
        manager._set_fault("cross-data runtime fault", source="test")
    assert ownership.read_global_energized_intent(get_settings()) is not None
    ownership_access.release()  # represents server exit after the latch was saved
    manager._watchdog_shutdown.set()
    manager._health_shutdown.set()

    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / f"other-data-{latch}"))
    get_settings.cache_clear()

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access(f"data-dir B after {latch}")
    assert refused.value.reason == "latched"
    assert "host-global motor-device energized marker" in refused.value.detail


def test_deliberate_estop_reset_retires_the_host_global_latch(real: None) -> None:
    from so101.services.robot_manager import RobotManager

    ownership_access = standalone_access("server E-stop reset harness")
    manager = RobotManager(store=ControlStore(control_store_path(get_settings())))
    try:
        manager.estop("resettable E-stop")
        assert ownership.read_global_energized_intent(get_settings()) is not None

        manager.reset_estop(operator_id="test operator")

        assert manager.guard.estopped is False
        assert ownership.read_global_energized_intent(get_settings()) is None
    finally:
        manager._watchdog_shutdown.set()
        manager._health_shutdown.set()
        ownership_access.release()


def test_crashed_server_write_ahead_refuses_another_data_directory(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from so101.services.robot_manager import RobotManager

    ownership_access = standalone_access("server lock harness")
    manager = RobotManager(store=ControlStore(control_store_path(get_settings())))
    manager._prepare_global_energized_intent("RobotManager teleop before injected crash")
    ownership_access.release()  # crash: no RobotManager shutdown/torque-off
    manager._watchdog_shutdown.set()
    manager._health_shutdown.set()

    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "server-crash-other-data"))
    get_settings.cache_clear()

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("new server/data-dir")
    assert "host-global motor-device energized marker" in refused.value.detail
    assert "RobotManager teleop before injected crash" in refused.value.detail


def test_leader_calibration_unconfirmed_survives_server_crash_across_data_dirs(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from so101.services.robot_manager import RobotManager

    ownership_access = standalone_access("server leader-calibration lock harness")
    manager = RobotManager(store=ControlStore(control_store_path(get_settings())))
    manager._mark_torque_unconfirmed(
        "leader calibration torque-off unconfirmed",
        event="safety.force_deenergize_failed",
        source="calibration",
        devices={"leader"},
    )
    ownership_access.release()  # crash before shutdown
    manager._watchdog_shutdown.set()
    manager._health_shutdown.set()

    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "leader-crash-other-data"))
    get_settings.cache_clear()
    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("new data-dir after leader calibration crash")
    assert "host-global motor-device energized marker" in refused.value.detail
    assert "arm may be energized" in refused.value.detail


def test_server_republishes_pre_global_local_recovery_before_serving(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from so101.services.robot_manager import RobotManager

    ownership_access = standalone_access("server restored-state lock harness")
    store = ControlStore(control_store_path(get_settings()))
    store.save_latches(
        {
            "fault": "pre-global unknown motor bus",
            "safety_state": "fault",
            "torque_unconfirmed": True,
        },
        version=time.time_ns() // 1_000,
    )
    manager = RobotManager(store=store)
    manager.publish_restored_recovery_intent()
    ownership_access.release()  # injected crash
    manager._watchdog_shutdown.set()
    manager._health_shutdown.set()

    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "pre-global-other-data"))
    get_settings.cache_clear()
    with pytest.raises(StandaloneRefused, match="host-global"):
        standalone_access("new data-dir after restored local fault")


def test_untrusted_global_energized_state_refuses_before_connect(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ownership,
        "read_global_energized_intent",
        lambda settings: (_ for _ in ()).throw(
            ownership.GlobalEnergizedIntentError("marker identity mismatch")
        ),
    )

    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("real arm")

    assert refused.value.reason == "unreadable"
    assert "treating the arm as energized" in refused.value.detail
    assert _lock_is_free()


def test_global_write_ahead_failure_prevents_follower_connect_and_clears_local_intent(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    connected: list[bool] = []

    class Follower:
        def connect(self) -> None:
            connected.append(True)

    def refuse_global(*args: Any, **kwargs: Any) -> None:
        raise ownership.GlobalEnergizedIntentError("global marker is read-only")

    monkeypatch.setattr(ownership, "write_global_energized_intent", refuse_global)
    access = standalone_access("real arm")

    with pytest.raises(ownership.GlobalEnergizedIntentError):
        access.connect_follower(Follower())

    assert connected == []
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None and saved["energized"] is False
    access.release()


@pytest.mark.parametrize(
    "torque_state",
    ["on", "unconfirmed", pytest.param(["corrupt"], id="unhashable")],
)
def test_disconnected_follower_with_possible_torque_fails_closed(
    torque_state: Any, real: None
) -> None:
    class Follower:
        connected = False

        def __init__(self) -> None:
            self.torque_state = torque_state
            self.disconnects = 0

        def disconnect(self) -> None:
            self.disconnects += 1

    follower = Follower()
    access = standalone_access("closed follower cleanup")

    with pytest.raises(TorqueOffUnconfirmed):
        access.disconnect_follower(follower)

    assert follower.disconnects == 1
    assert access.holds_bus_lock
    access.release()
    _assert_failed_torque_off_was_latched()


def test_unreadable_device_torque_truth_still_persists_the_latch(real: None) -> None:
    class Follower:
        def disconnect(self) -> None:
            pass

        @property
        def bus(self) -> Any:
            raise OSError("bus wrapper is unreadable")

    access = standalone_access("unreadable follower cleanup")

    with pytest.raises(TorqueOffUnconfirmed) as unconfirmed:
        access.disconnect_follower(Follower())

    assert isinstance(unconfirmed.value.__cause__, OSError)
    access.release()
    _assert_failed_torque_off_was_latched()


def test_generic_disconnect_error_with_uncertain_torque_is_latched(real: None) -> None:
    class Follower:
        torque_state = "on"

        def disconnect(self) -> None:
            self.torque_state = "unconfirmed"
            raise OSError("serial close failed")

    access = standalone_access("failed follower cleanup")

    with pytest.raises(TorqueOffUnconfirmed) as unconfirmed:
        access.disconnect_follower(Follower())

    assert isinstance(unconfirmed.value.__cause__, OSError)
    access.release()
    _assert_failed_torque_off_was_latched()


def test_real_device_without_torque_truth_fails_closed(real: None) -> None:
    class Follower:
        def disconnect(self) -> None:
            pass

    access = standalone_access("missing follower torque truth")

    with pytest.raises(TorqueOffUnconfirmed):
        access.disconnect_follower(Follower())

    access.release()
    _assert_failed_torque_off_was_latched()


def test_policy_rollout_does_not_release_a_closed_unconfirmed_follower(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    from so101.policy.runner import PolicyRunner

    class Follower:
        connected = False
        torque_state = "off"
        disconnects = 0

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.disconnects += 1

    def lose_connection(self: PolicyRunner, duration: float = 30.0) -> Iterator[dict[str, float]]:
        self.follower.connected = False  # type: ignore[attr-defined]
        self.follower.torque_state = "unconfirmed"  # type: ignore[attr-defined]
        return
        yield  # pragma: no cover - makes this a generator without sleeping

    follower = Follower()
    monkeypatch.setattr(PolicyRunner, "run", lose_connection)
    policy_runner = PolicyRunner(
        _StillPolicy(),  # type: ignore[arg-type]
        follower=follower,  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
    )

    with pytest.raises(TorqueOffUnconfirmed):
        policy_runner.rollout(0)

    assert follower.disconnects == 1
    _assert_failed_torque_off_was_latched()


def test_policy_rollout_refuses_a_caller_owned_real_follower(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101.policy.runner import PolicyRunner

    class Follower:
        connected = True
        torque_state = "on"
        disconnects = 0

        def disconnect(self) -> None:
            self.disconnects += 1

    def lose_connection(self: PolicyRunner, duration: float = 30.0) -> Iterator[dict[str, float]]:
        self.follower.connected = False  # type: ignore[attr-defined]
        self.follower.torque_state = "unconfirmed"  # type: ignore[attr-defined]
        return
        yield  # pragma: no cover - makes this a generator without sleeping

    follower = Follower()
    monkeypatch.setattr(PolicyRunner, "run", lose_connection)
    policy_runner = PolicyRunner(
        _StillPolicy(),  # type: ignore[arg-type]
        follower=follower,  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
    )

    with pytest.raises(StandaloneRefused, match="caller-owned follower"):
        policy_runner.rollout(0)

    assert follower.disconnects == 0
    assert not control_store_path(get_settings()).exists()
    assert _lock_is_free()


def test_teleop_persists_failed_torque_off_before_releasing_the_lock(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    import so101.hardware.follower as follower_module
    import so101.hardware.leader as leader_module

    class Follower:
        connected = False

        def connect(self) -> None:
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    class Leader:
        torque_state = "off"

        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            self.torque_state = "off"

    monkeypatch.setattr(follower_module, "SO101Follower", Follower)
    monkeypatch.setattr(leader_module, "SO101Leader", Leader)

    result = runner.invoke(app, ["teleop", "--duration", "0"])

    assert result.exit_code == 1
    assert isinstance(result.exception, TorqueOffUnconfirmed)
    _assert_failed_torque_off_was_latched()


def test_teleop_normal_completion_halts_without_torque_off(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    import so101.hardware.follower as follower_module
    import so101.hardware.leader as leader_module

    calls: list[str] = []

    class Follower:
        connected = False
        torque_state = "off"

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def halt_and_disconnect(self) -> None:
            calls.append("halt")
            self.connected = False
            self.torque_state = "on"

        def disconnect(self) -> None:
            calls.append("torque-off")
            self.connected = False
            self.torque_state = "off"

    class Leader:
        torque_state = "off"

        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            calls.append("leader-off")

    monkeypatch.setattr(follower_module, "SO101Follower", Follower)
    monkeypatch.setattr(leader_module, "SO101Leader", Leader)

    result = runner.invoke(app, ["teleop", "--duration", "0"])

    assert result.exit_code == 0, result.output
    assert calls == ["leader-off", "halt"]
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None and saved["energized"] is True


def test_teleop_controller_persists_closed_unconfirmed_leader(
    real: None,
    real_motion_internal_harness: None,
) -> None:
    from so101.teleop.controller import TeleopController

    class Follower:
        connected = False
        torque_state = "off"

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.connected = False
            self.torque_state = "off"

    class Leader:
        connected = False
        torque_state = "off"
        disconnects = 0

        def connect(self) -> None:
            self.connected = True

        def disconnect(self) -> None:
            self.disconnects += 1
            self.connected = False
            self.torque_state = "unconfirmed"

    leader = Leader()
    access = standalone_access("so101 record")
    controller = TeleopController(
        leader,  # type: ignore[arg-type]
        Follower(),  # type: ignore[arg-type]
        standalone_access=access,
    )
    controller.connect()

    with pytest.raises(TorqueOffUnconfirmed):
        controller.disconnect()

    assert leader.disconnects == 1
    access.release()
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None and "standalone leader" in str(saved["fault"])
    _assert_failed_torque_off_was_latched()


def test_teleop_controller_checks_leader_after_failed_connect(
    real: None, real_motion_internal_harness: None
) -> None:
    from so101.teleop.controller import TeleopController

    class Follower:
        def connect(self) -> None:
            raise AssertionError("follower must not open after the leader failed")

        def disconnect(self) -> None:
            raise AssertionError("follower was never owned")

    class Leader:
        connected = False
        torque_state = "off"
        disconnects = 0

        def connect(self) -> None:
            self.torque_state = "unconfirmed"
            raise RuntimeError("leader port closed during connect")

        def disconnect(self) -> None:
            self.disconnects += 1

    leader = Leader()
    access = standalone_access("so101 record")
    controller = TeleopController(
        leader,  # type: ignore[arg-type]
        Follower(),  # type: ignore[arg-type]
        standalone_access=access,
    )

    with pytest.raises(TorqueOffUnconfirmed):
        controller.connect()

    assert leader.disconnects == 1
    access.release()
    _assert_failed_torque_off_was_latched()


def test_replay_persists_failed_torque_off_before_releasing_the_lock(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    import so101.data.replay as replay_module

    class Dataset:
        format = "so101-fallback-v1"
        path = "."
        fps = 30
        repo_id = "gate/demo"

    class Follower:
        connected = False

        def connect(self) -> None:
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    monkeypatch.setattr(replay_module, "require_dataset", lambda repo_id: Dataset())
    monkeypatch.setattr(replay_module, "_load_fallback_actions", lambda root, episode: [])
    monkeypatch.setattr(replay_module, "SO101Follower", Follower)

    with pytest.raises(TorqueOffUnconfirmed):
        replay_module.replay_episode("gate/demo", 0)

    _assert_failed_torque_off_was_latched()


def test_replay_normal_completion_halts_without_torque_off(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    import so101.data.replay as replay_module

    class Dataset:
        format = "so101-fallback-v1"
        path = "."
        fps = 30
        repo_id = "gate/demo"

    class Follower:
        connected = False
        torque_state = "off"
        halts = 0
        disconnects = 0

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def halt_and_disconnect(self) -> None:
            self.halts += 1
            self.connected = False
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.disconnects += 1
            self.connected = False
            self.torque_state = "off"

    follower = Follower()
    monkeypatch.setattr(replay_module, "require_dataset", lambda repo_id: Dataset())
    monkeypatch.setattr(replay_module, "_load_fallback_actions", lambda root, episode: [])
    monkeypatch.setattr(replay_module, "SO101Follower", lambda: follower)

    assert replay_module.replay_episode("gate/demo", 0) == 0

    assert follower.halts == 1
    assert follower.disconnects == 0
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None and saved["energized"] is True


def test_replay_does_not_release_a_closed_unconfirmed_follower(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    real_motion_internal_harness: None,
) -> None:
    import so101.data.replay as replay_module

    class Dataset:
        format = "so101-fallback-v1"
        path = "."
        fps = 30
        repo_id = "gate/demo"

    class Follower:
        connected = False
        torque_state = "off"
        disconnects = 0

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def send_goal(self, goal: dict[str, float]) -> dict[str, float]:
            self.connected = False
            self.torque_state = "unconfirmed"
            raise RuntimeError("the port closed during replay")

        def disconnect(self) -> None:
            self.disconnects += 1

    follower = Follower()
    monkeypatch.setattr(replay_module, "require_dataset", lambda repo_id: Dataset())
    monkeypatch.setattr(replay_module, "_load_fallback_actions", lambda root, episode: [[0.0] * 6])
    monkeypatch.setattr(replay_module, "SO101Follower", lambda: follower)

    with pytest.raises(TorqueOffUnconfirmed):
        replay_module.replay_episode("gate/demo", 0)

    assert follower.disconnects == 1
    _assert_failed_torque_off_was_latched()


def test_eval_cleanup_does_not_release_a_closed_unconfirmed_follower(
    real: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    real_motion_internal_harness: None,
) -> None:
    import so101.eval.evaluator as evaluator
    import so101.policy.runner as runner_module
    from so101.eval.evaluator import EvalConfig, EvalRun
    from so101.policy.runner import PolicyRunner

    class Defaults:
        task = ""
        fps = 30

    class Follower:
        connected = False
        torque_state = "off"
        disconnects = 0

        def connect(self) -> None:
            self.connected = True
            self.torque_state = "on"

        def disconnect(self) -> None:
            self.disconnects += 1

    def lose_connection(self: PolicyRunner, duration: float = 30.0) -> Iterator[dict[str, float]]:
        self.follower.connected = False  # type: ignore[attr-defined]
        self.follower.torque_state = "unconfirmed"  # type: ignore[attr-defined]
        raise RuntimeError("the port closed during evaluation")
        yield  # pragma: no cover - makes this a generator without sleeping

    follower = Follower()
    monkeypatch.setattr(evaluator, "load_policy", lambda path, task=None: _StillPolicy())
    monkeypatch.setattr(evaluator, "checkpoint_defaults", lambda path: Defaults())
    monkeypatch.setattr(evaluator, "resolve_control_fps", lambda requested, defaults: 30)
    monkeypatch.setattr(runner_module, "SO101Follower", lambda: follower)
    monkeypatch.setattr(runner_module, "CameraManager", _NoCameras)
    monkeypatch.setattr(PolicyRunner, "run", lose_connection)
    run = EvalRun(EvalConfig(model_path=str(tmp_path), n_episodes=1), save=False)

    with pytest.raises(TorqueOffUnconfirmed):
        run.run()

    assert follower.disconnects == 1
    _assert_failed_torque_off_was_latched()


def test_failed_standalone_latch_store_uses_marker_before_unlock(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Follower:
        def disconnect(self) -> None:
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    def refuse_save(self: ControlStore, latches: dict[str, Any], *, version: int) -> bool:
        raise OSError("disk refused the latch")

    monkeypatch.setattr(ControlStore, "save_latches", refuse_save)
    access = standalone_access("standalone torque failure")

    with pytest.raises(TorqueOffUnconfirmed):
        access.disconnect_follower(Follower())

    assert access.holds_bus_lock
    assert _marker_path().is_file()
    access.release()
    with pytest.raises(StandaloneRefused) as refused:
        standalone_access("the next standalone command")
    assert refused.value.reason == "latched"
    assert "torque-off unconfirmed" in refused.value.detail
    assert _lock_is_free()


def test_standalone_latch_store_open_failure_uses_marker(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Follower:
        def disconnect(self) -> None:
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    access = standalone_access("standalone torque failure")

    def refuse_open(self: ControlStore, path: Path) -> None:
        raise OSError("database cannot be opened")

    monkeypatch.setattr(ControlStore, "__init__", refuse_open)
    with pytest.raises(TorqueOffUnconfirmed):
        access.disconnect_follower(Follower())

    assert _marker_path().is_file()
    assert access.holds_bus_lock
    access.release()


def test_access_stays_locked_when_neither_latch_target_can_be_written(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Follower:
        def disconnect(self) -> None:
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    def refuse_save(self: ControlStore, latches: dict[str, Any], *, version: int) -> bool:
        raise OSError("database refused the latch")

    def refuse_marker(path: Path, payload: Any) -> None:
        raise OSError("marker refused the latch")

    def refuse_global(*args: Any, **kwargs: Any) -> None:
        raise OSError("global marker refused the latch")

    monkeypatch.setattr(ControlStore, "save_latches", refuse_save)
    monkeypatch.setattr(control_store_module, "atomic_write_json", refuse_marker)
    monkeypatch.setattr(ownership, "write_global_energized_intent", refuse_global)
    access = standalone_access("standalone torque failure")

    try:
        with pytest.raises(RuntimeError, match="neither .* nor its fallback marker"):
            access.disconnect_follower(Follower())

        assert access.holds_bus_lock
        with pytest.raises(RuntimeError, match="refusing to release motor-bus locks"):
            access.release()
        with pytest.raises(StandaloneRefused) as refused:
            standalone_access("the next command in this process")
        assert refused.value.reason == "unreadable"
    finally:
        # Production recovery is to end the process. This test process must
        # continue, so emulate process teardown even when an assertion fails.
        with standalone._held_guard:
            held = standalone._held.get(access.lock_key)
            if held is not None:
                held.release_blocked = None
        access.release()
    assert _lock_is_free()


def test_global_marker_preserves_unconfirmed_truth_when_both_local_targets_fail(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Follower:
        def disconnect(self) -> None:
            raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    def refuse_save(self: ControlStore, latches: dict[str, Any], *, version: int) -> bool:
        raise OSError("database refused the latch")

    def refuse_marker(path: Path, payload: Any) -> None:
        raise OSError("local marker refused the latch")

    monkeypatch.setattr(ControlStore, "save_latches", refuse_save)
    monkeypatch.setattr(control_store_module, "atomic_write_json", refuse_marker)
    access = standalone_access("standalone torque failure")

    with pytest.raises(TorqueOffUnconfirmed):
        access.disconnect_follower(Follower())

    access.release()
    with pytest.raises(StandaloneRefused, match="host-global"):
        standalone_access("next process")
    assert _lock_is_free()


def test_a_newer_concurrent_unconfirmed_latch_satisfies_failed_disconnect(
    real: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_latches = ControlStore.save_latches

    def displaced_save(self: ControlStore, latches: dict[str, Any], *, version: int) -> bool:
        save_latches(
            self,
            {
                "estopped": False,
                "fault": "newer concurrent torque failure",
                "safety_state": "fault",
                "torque_unconfirmed": True,
            },
            version=version + 1,
        )
        return False

    monkeypatch.setattr(ControlStore, "save_latches", displaced_save)
    access = standalone_access("standalone torque failure")
    error = TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    access.record_torque_off_failure(error)
    access.record_torque_off_failure(error)  # idempotent

    assert access.latches is not None
    assert access.latches["fault"] == "newer concurrent torque failure"
    assert access.latches["torque_unconfirmed"] is True
    assert not _marker_path().exists()
    access.release()


def test_disconnect_after_releasing_access_is_rejected(real: None) -> None:
    class Follower:
        def disconnect(self) -> None:
            raise AssertionError("released access must not touch the follower")

    access = standalone_access("finished command")
    access.release()

    with pytest.raises(RuntimeError, match="released before device disconnect"):
        access.disconnect_follower(Follower())


def test_the_gate_is_reentrant_within_one_process(real: None) -> None:
    outer = standalone_access("so101 run")
    inner = standalone_access("PolicyRunner (policy rollout)")
    assert outer.holds_bus_lock and inner.holds_bus_lock

    holder = ownership.read_lock_holder(bus_lock_path(get_settings()))
    assert holder.get("purpose") == "so101 run"  # the outer command wrote the record

    inner.release()
    assert not _lock_is_free()  # the command still holds it
    inner.release()  # idempotent
    assert not _lock_is_free()

    outer.release()
    assert _lock_is_free()


def test_reentrancy_does_not_admit_an_unrelated_thread(real: None) -> None:
    outer = standalone_access("thread A")
    outcomes: list[StandaloneAccess | StandaloneRefused] = []

    def take_from_thread_b() -> None:
        try:
            access = standalone_access("thread B")
        except StandaloneRefused as exc:
            outcomes.append(exc)
        else:
            outcomes.append(access)
            access.release()

    thread = threading.Thread(target=take_from_thread_b)
    thread.start()
    thread.join()

    assert len(outcomes) == 1
    assert isinstance(outcomes[0], StandaloneRefused)
    assert outcomes[0].reason == "lock"
    assert outer.holds_bus_lock
    outer.release()
    assert _lock_is_free()


def test_bus_access_can_only_be_released_by_its_owning_thread(real: None) -> None:
    access = standalone_access("thread A")
    errors: list[Exception] = []

    def release_from_thread_b() -> None:
        try:
            access.release()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=release_from_thread_b)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owning thread" in str(errors[0])
    assert access.holds_bus_lock
    access.release()
    assert _lock_is_free()


def test_an_existing_bus_lock_must_be_held_and_untracked_before_adoption(real: None) -> None:
    path = bus_lock_path(get_settings())
    loose = BusLock(path, purpose="loose bring-up lock")
    with pytest.raises(RuntimeError, match="not held"):
        standalone.adopt_acquired_bus_access(
            loose, backend="real", purpose="bring-up", settings=get_settings()
        )

    access = standalone_access("tracked command")
    with standalone._held_guard:
        tracked = standalone._held[access.lock_key].lock
    with pytest.raises(RuntimeError, match="already tracks"):
        standalone.adopt_acquired_bus_access(
            tracked, backend="real", purpose="bring-up", settings=get_settings()
        )
    access.release()
    assert _lock_is_free()


def test_a_nested_call_rereads_the_latches(real: None) -> None:
    with standalone_access("so101 run"):
        _latch(estopped=True)  # set while the command runs (another process)
        with pytest.raises(StandaloneRefused):
            standalone_access("PolicyRunner (policy rollout)")
        assert not _lock_is_free()  # the outer access still holds the lock
    assert _lock_is_free()


# --- the server's device-global locks ---------------------------------------------------


@pytest.fixture
def real_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    pytest.importorskip("fastapi")
    _set_backend(monkeypatch, "real")
    yield


def test_the_real_server_holds_bus_lock_for_its_lifetime(real_app: None) -> None:
    from fastapi.testclient import TestClient

    from so101.services.app import create_app

    path = bus_lock_path(get_settings())
    with TestClient(create_app()) as client:
        assert client.get("/api/robot/status").json()["backend"] == "real"
        with pytest.raises(BusOwnershipError) as held:
            BusLock(path, purpose="so101 teleop").acquire()
        assert held.value.holder.get("purpose") == ownership.SERVER_LOCK_PURPOSE
    assert _lock_is_free()  # released on shutdown


def test_a_cli_command_is_refused_while_the_server_holds_the_lock(
    real_app: None, probe: list[ServerProbe], opened: list[str]
) -> None:
    """Even when the probe misses the server (another port), the lock excludes it."""

    from fastapi.testclient import TestClient

    from so101.services.app import create_app

    with TestClient(create_app()):
        result = runner.invoke(app, CLI_PATHS["teleop"])
    assert result.exit_code == EXIT_REFUSED
    assert ownership.SERVER_LOCK_PURPOSE in _flat(result.stderr)
    assert opened == []


def test_the_real_server_will_not_start_while_a_cli_command_holds_the_lock(
    real_app: None,
) -> None:
    from fastapi.testclient import TestClient

    import so101.services.robot_manager as robot_manager
    from so101.services.app import create_app

    with BusLock(bus_lock_path(get_settings()), purpose="so101 teleop"):
        with pytest.raises(ownership.ServerBusLockError) as refused:
            with TestClient(create_app()):
                pass
    message = str(refused.value)
    assert "so101 teleop" in message and "cannot start the Suite server" in message
    assert robot_manager._manager is None  # nothing owned the arm


def test_the_mock_server_takes_no_bus_lock() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from so101.services.app import create_app

    with TestClient(create_app()):
        assert _lock_is_free()

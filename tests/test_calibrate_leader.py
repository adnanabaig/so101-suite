"""Leader orchestration is resumable and never bypasses server ownership or validation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from so101.config import get_settings

spec = importlib.util.spec_from_file_location(
    "leader_workflow", Path(__file__).resolve().parents[1] / "scripts/calibrate_leader.py"
)
assert spec and spec.loader
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def options(**updates):
    return SimpleNamespace(**({"resume": None, "new": False, "dry_run": False} | updates))


def config(tmp_path):
    return get_settings().model_copy(
        update={
            "backend": "real",
            "mock": False,
            "health_leader_supply": "12v",
            "data_dir": tmp_path,
        }
    )


def test_dry_run_never_stops_server_or_runs_installation(tmp_path):
    commands = []
    folder = cli.run(
        options(dry_run=True),
        config(tmp_path),
        execute=lambda command, **kwargs: commands.append(command),
        stop_server=lambda: pytest.fail("Dry run cannot stop server"),
    )
    assert len(commands) == 1
    assert "--dry-run" in commands[0] and "leader" in commands[0]
    assert not folder.exists()


def test_wrong_profile_and_shared_arm_identity_are_rejected(tmp_path):
    settings = config(tmp_path)
    for invalid in (
        settings.model_copy(update={"health_leader_supply": "7.4v"}),
        settings.model_copy(update={"leader_id": settings.follower_id}),
        settings.model_copy(update={"leader_port": settings.follower_port}),
    ):
        with pytest.raises(RuntimeError):
            cli.run(options(dry_run=True), invalid)


def test_auto_resume_keeps_most_completed_joints(tmp_path, monkeypatch):
    settings = config(tmp_path)
    root = tmp_path / "manual_arm_calibrations/leader"
    for name, completed in (("older-progress", ["gripper", "wrist_roll"]), ("new-empty", [])):
        folder = root / name
        folder.mkdir(parents=True)
        manifest = {
            "arm": "leader",
            "device_id": settings.leader_id,
            "port": settings.leader_port,
            "validations": 5,
            "joints": list(cli.manual.ORDER),
            "results": {joint: {"state": "validated"} for joint in completed},
        }
        (folder / "arm_session.json").write_text(json.dumps(manifest))
    verified = []
    monkeypatch.setattr(
        cli.manual, "verify_completed", lambda folder, manifest, name: verified.append(name)
    )
    assert cli.latest_session(settings) == root / "older-progress"
    assert verified == ["gripper", "wrist_roll"]


def test_energized_server_cannot_be_stopped_for_calibration(monkeypatch):
    monkeypatch.setattr(
        cli,
        "fetch",
        lambda route: {
            "backend": "real",
            "mode": "idle",
            "torque_state": "on",
            "armed": False,
        },
    )
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("No stop allowed")
    )
    with pytest.raises(RuntimeError, match="torque OFF"):
        cli.stop_idle_motor_server()


def test_calibration_failure_never_installs_or_restarts_server(tmp_path, monkeypatch):
    settings = config(tmp_path)
    follower = tmp_path / "calibration_follower.json"
    follower.write_text("preserved")
    monkeypatch.setattr(cli.verification, "lerobot_calibration_file", lambda *args: follower)
    commands = []
    stopped = []

    def execute(command, **kwargs):
        commands.append(command)
        raise RuntimeError("joint capture failed")

    with pytest.raises(RuntimeError, match="joint capture failed"):
        cli.run(options(), settings, execute=execute, stop_server=lambda: stopped.append(True))
    assert stopped == [True]
    assert len(commands) == 1
    assert "calibrate-arm-manual" in commands[0][0]
    assert follower.read_text() == "preserved"


@pytest.mark.parametrize("already_installed", [False, True])
def test_full_workflow_installs_once_and_requires_ten_sweeps(
    tmp_path, monkeypatch, already_installed
):
    settings = config(tmp_path)
    follower = tmp_path / "calibration_follower.json"
    follower.write_text("preserved")
    monkeypatch.setattr(cli.verification, "lerobot_calibration_file", lambda *args: follower)
    folder = tmp_path / "session"
    folder.mkdir()
    evidence = tmp_path / "verification"
    evidence.mkdir()
    reference = {"source_hashes": {"leader": "hash"}}
    installed = [already_installed]

    def load_reference(*args):
        if not installed[0]:
            raise FileNotFoundError("No installed leader calibration")
        return reference

    monkeypatch.setattr(cli.verification, "load_reference", load_reference)
    monkeypatch.setattr(cli.verification, "find_resume", lambda *args: evidence)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        if command[0].endswith("install-arm-calibration"):
            installed[0] = True
        if command[0].endswith("verify-arm-manual"):
            (evidence / "verification.json").write_text(
                json.dumps(
                    {
                        "state": "ten_sweeps_verified",
                        "reference": reference,
                        "comparison": {name: {"successful_sweeps": 10} for name in cli.JOINT_NAMES},
                    }
                )
            )

    cli.run(options(resume=folder), settings, execute=execute, stop_server=lambda: None)
    assert len(commands) == (2 if already_installed else 3)
    summary = json.loads((folder / "leader_bringup.json").read_text())
    assert summary["state"] == "installed_and_verified"
    assert summary["follower_calibration_unchanged"]
    assert not summary["powered_teleop_verified"]


def test_parent_abort_signals_worker_and_waits_for_restoration(monkeypatch):
    calls = []

    class Child:
        def wait(self, **kwargs):
            calls.append(("wait", kwargs))
            if len(calls) == 1:
                raise RuntimeError("parent interrupted")
            return 0

        def terminate(self):
            calls.append(("terminate", {}))

    monkeypatch.setattr(cli.subprocess, "Popen", lambda command: Child())
    with pytest.raises(RuntimeError, match="parent interrupted"):
        cli.run_child(["worker"], check=True)
    assert calls == [("wait", {}), ("terminate", {}), ("wait", {"timeout": 30})]

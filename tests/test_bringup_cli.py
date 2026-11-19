"""CLI: ``so101 preflight`` and ``so101 rest-pose record|show|clear`` (mock and real paths)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from so101 import JOINT_NAMES, cli_bringup
from so101.bringup import ownership
from so101.bringup.ownership import BusLock, ServerProbe, bus_lock_path
from so101.cli import app as root_app
from so101.config import get_settings
from so101.hardware import backend as backend_module
from so101.hardware import lerobot_paths, motors
from so101.hardware import rest_pose as rp
from so101.services.store import ControlStore, control_store_path
from so101.simulation.rest_pose import MODEL_REST_POSE

FOLDED = {
    "shoulder_pan": 1.0,
    "shoulder_lift": -95.0,
    "elbow_flex": 92.0,
    "wrist_flex": 70.0,
    "wrist_roll": 3.0,
    "gripper": 2.0,
}


def no_server(settings: Any) -> ServerProbe:
    return ServerProbe(running=False, url="http://127.0.0.1:8816/api/robot/status", detail="none")


def running_server(settings: Any) -> ServerProbe:
    return ServerProbe(
        running=True,
        url="http://127.0.0.1:8816/api/robot/status",
        detail="HTTP 200 from http://127.0.0.1:8816/api/robot/status",
        http_status=200,
        status={"mode": "idle", "estopped": False, "live_cameras": ["0"], "connected": True},
    )


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # No real server probe, no real LeRobot calibration directory, no real sleeps.
    monkeypatch.setattr(ownership, "probe_local_server", no_server)
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    monkeypatch.setattr(cli_bringup, "_sleep", lambda _: None)
    monkeypatch.setenv("COLUMNS", "200")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def invoke(args: list[str], input: str | None = None) -> Any:
    return CliRunner().invoke(root_app, args, input=input)


def use_real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()


def write_lerobot_calibration(content: str = '{"shoulder_pan": {"id": 1}}') -> str:
    settings = get_settings()
    path = lerobot_paths.lerobot_calibration_file("follower", settings.follower_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


class FakeArm:
    """Stands in for the follower bus; records every call."""

    def __init__(self, readings: list[dict[str, float]] | None = None, fail_connect: bool = False):
        self.readings = readings or [dict(FOLDED)]
        self.fail_connect = fail_connect
        self.calls: list[Any] = []
        self._connected = False
        self._index = 0
        self.torque_state = "off"

    def connect(self) -> None:
        self.calls.append("connect")
        if self.fail_connect:
            raise ConnectionError("Could not connect on port '/dev/fake'")
        self._connected = True
        self.torque_state = "off"

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self._connected = False
        self.torque_state = "off"

    @property
    def connected(self) -> bool:
        return self._connected

    def set_torque(self, enabled: bool) -> None:
        self.calls.append(("set_torque", enabled))

    def read_positions(self) -> dict[str, float]:
        self.calls.append("read")
        reading = self.readings[self._index % len(self.readings)]
        self._index += 1
        return dict(reading)

    def write_positions(self, goal: dict[str, float]) -> dict[str, float]:  # pragma: no cover
        raise AssertionError("rest-pose record must never command a position")


def install_arm(monkeypatch: pytest.MonkeyPatch, arm: FakeArm) -> list[tuple[str, str]]:
    opened: list[tuple[str, str]] = []

    def make_bus(port: str, *, name: str, force_mock: bool | None = None) -> FakeArm:
        opened.append((port, name))
        return arm

    monkeypatch.setattr(motors, "make_bus", make_bus)
    return opened


def lock_is_free() -> bool:
    try:
        BusLock(bus_lock_path(get_settings()), purpose="test").acquire().release()
    except ownership.BusOwnershipError:
        return False
    return True


# --- mounting -----------------------------------------------------------------------


def test_commands_are_mounted_on_the_root_cli() -> None:
    result = invoke(["--help"])
    assert result.exit_code == 0
    assert "preflight" in result.output and "rest-pose" in result.output
    assert "record" in invoke(["rest-pose", "--help"]).output


# --- preflight ----------------------------------------------------------------------


def test_preflight_json_is_a_labeled_rehearsal_and_is_saved() -> None:
    result = invoke(["preflight", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["kind"] == "so101.preflight" and report["origin"] == "cli"
    assert report["rehearsal"] is True and report["backend"] == "mock"
    assert report["ready_for_hardware"] is False and report["exit_code"] == 0
    assert report["notes"][0].startswith("REHEARSAL")
    assert all(check["rehearsal"] for check in report["checks"])
    latest = Path(get_settings().data_dir) / "preflight" / "latest-mock.json"
    assert json.loads(latest.read_text(encoding="utf-8"))["created_at"] == report["created_at"]
    assert lock_is_free()


def test_preflight_human_output_and_no_save() -> None:
    result = invoke(["preflight", "--no-save", "--arm", "follower"])
    assert result.exit_code == 0, result.output
    assert "REHEARSAL" in result.output and "rest_pose.follower" in result.output
    assert "motors.leader" not in result.output
    assert "exit 0" in result.output
    assert not (Path(get_settings().data_dir) / "preflight").exists()


def test_preflight_require_hardware_rejects_a_rehearsal() -> None:
    result = invoke(["preflight", "--json", "--no-save", "--require-hardware"])
    assert result.exit_code == 4
    assert json.loads(result.stdout)["exit_code"] == 4


def test_preflight_rejects_an_unknown_arm() -> None:
    result = invoke(["preflight", "--arm", "both-hands"])
    assert result.exit_code == 2
    assert "follower, leader or both" in result.output


def test_preflight_refuses_the_bus_while_a_server_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ownership, "probe_local_server", running_server)
    result = invoke(["preflight", "--json", "--no-save"])
    report = json.loads(result.stdout)
    assert result.exit_code == 3 and report["overall"] == "incomplete"
    assert report["bus_access"] == "refused"
    estop = next(check for check in report["checks"] if check["id"] == "estop")
    assert estop["source"] == "server" and estop["status"] == "pass"


# --- rest-pose record ---------------------------------------------------------------


def test_record_rehearsal_saves_a_mock_pose_and_shows_the_exact_prompt() -> None:
    result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == 0, result.output
    assert cli_bringup.REST_POSE_PROMPT in result.output
    assert cli_bringup.REST_POSE_PROMPT == (
        "with the padded mechanical catch clear of the rest pose, fold the arm by its links "
        "away from pinch points; move hands clear, let go, wait for it to settle, then press "
        "Enter"
    )
    assert "REHEARSAL" in result.output
    record = rp.load_recorded_rest_pose(get_settings())
    assert record is not None and record.backend == "mock" and record.samples == 10
    assert record.calibration_digest is None
    assert "Collapse will use the recorded pose" in result.output
    assert lock_is_free()


def test_record_json_keeps_prompts_off_stdout() -> None:
    result = invoke(["rest-pose", "record", "--yes", "--json"], input="\n")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["rehearsal"] is True
    assert payload["record"]["backend"] == "mock"
    assert payload["resolution"]["source"] == "recorded"
    assert cli_bringup.REST_POSE_PROMPT in result.stderr


def test_record_asks_before_turning_torque_off(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = FakeArm()
    opened = install_arm(monkeypatch, arm)
    result = invoke(["rest-pose", "record"], input="n\n")
    assert result.exit_code == 1 and "Aborted" in result.output
    assert "padded mechanical catch" in result.output
    assert "physical cutoff" in result.output
    assert "Do not hold or catch the arm by hand" in result.output
    assert "Support the arm" not in result.output
    assert opened == [] and arm.calls == []
    assert rp.load_recorded_rest_pose(get_settings()) is None
    assert lock_is_free()


def test_record_refuses_while_a_server_owns_the_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    use_real_backend(monkeypatch)
    monkeypatch.setattr(ownership, "probe_local_server", running_server)
    opened = install_arm(monkeypatch, FakeArm())
    result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == cli_bringup.EXIT_REFUSED == 3
    assert "Stop the server" in result.output
    assert opened == []
    assert lock_is_free()


def test_record_refuses_while_another_command_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_real_backend(monkeypatch)
    opened = install_arm(monkeypatch, FakeArm())
    with BusLock(bus_lock_path(get_settings()), purpose="so101 preflight"):
        result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == 3
    assert "so101 preflight" in result.output and opened == []


def test_mock_rest_pose_record_does_not_contend_with_real_device_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = FakeArm()
    install_arm(monkeypatch, arm)
    with BusLock(bus_lock_path(get_settings()), purpose="real arm owner"):
        result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == 0, result.output
    assert "REHEARSAL" in result.output
    assert "saved" in result.output


def test_record_rejects_an_arm_that_has_not_settled(monkeypatch: pytest.MonkeyPatch) -> None:
    moving = [dict(FOLDED), {**FOLDED, "elbow_flex": FOLDED["elbow_flex"] + 2.5}]
    arm = FakeArm(readings=moving)
    install_arm(monkeypatch, arm)
    result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == 1
    assert "not saved" in result.output and "has not settled" in result.output
    assert "elbow_flex" in result.output and "2.50" in result.output
    assert arm.calls[-1] == "disconnect"
    assert rp.load_recorded_rest_pose(get_settings()) is None
    assert lock_is_free()


def test_record_on_the_real_backend_binds_the_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_real_backend(monkeypatch)
    digest = write_lerobot_calibration()
    arm = FakeArm(readings=[dict(FOLDED), {**FOLDED, "wrist_roll": 3.4}])
    opened = install_arm(monkeypatch, arm)
    result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == 0, result.output
    assert "REHEARSAL" not in result.output
    settings = get_settings()
    assert opened == [(settings.follower_port, "follower")]
    assert arm.calls[:2] == ["connect", ("set_torque", False)]
    assert arm.calls.count("read") == rp.RECORD_SAMPLES and arm.calls[-1] == "disconnect"
    assert ("set_torque", True) not in arm.calls
    record = rp.load_recorded_rest_pose(settings)
    assert record is not None and record.backend == "real"
    assert record.calibration_digest == digest and record.follower_id == settings.follower_id
    assert record.joints["wrist_roll"] == pytest.approx(3.2)
    assert record.spread["wrist_roll"] == pytest.approx(0.4)
    assert set(record.joints) == set(JOINT_NAMES)
    resolution = rp.resolve_for_settings(settings)
    assert resolution.source == "recorded"
    assert lock_is_free()


def test_record_on_the_real_backend_needs_a_calibration_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_real_backend(monkeypatch)
    install_arm(monkeypatch, FakeArm())
    result = invoke(["rest-pose", "record", "--yes"], input="\n")
    assert result.exit_code == 1
    assert "so101 calibrate follower" in result.output
    assert rp.load_recorded_rest_pose(get_settings()) is None


def test_record_fails_closed_when_bus_disconnect_leaves_torque_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_real_backend(monkeypatch)
    write_lerobot_calibration()

    class UnconfirmedArm(FakeArm):
        torque_state = "off"

        def connect(self) -> None:
            super().connect()
            self.torque_state = "off"

        def disconnect(self) -> None:
            super().disconnect()
            self.torque_state = "unconfirmed"

    arm = UnconfirmedArm()
    install_arm(monkeypatch, arm)

    result = invoke(["rest-pose", "record", "--yes"], input="\n")

    assert result.exit_code == 1
    assert "not saved" in result.output and "torque-off unconfirmed" in result.output
    assert rp.load_recorded_rest_pose(get_settings()) is None
    saved = ControlStore(control_store_path(get_settings())).load_latches()
    assert saved is not None
    assert saved["fault"] and "standalone follower" in saved["fault"]
    assert saved["torque_unconfirmed"] is True
    assert lock_is_free()


def test_record_reports_a_bus_that_will_not_open(monkeypatch: pytest.MonkeyPatch) -> None:
    use_real_backend(monkeypatch)
    install_arm(monkeypatch, FakeArm(fail_connect=True))
    result = invoke(["rest-pose", "record", "--yes"])
    assert result.exit_code == 1
    assert "could not open the follower bus" in result.output
    assert cli_bringup.REST_POSE_PROMPT not in result.output  # failed before the prompt
    assert lock_is_free()


def test_record_refuses_the_sim_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module, "current_backend", lambda settings: "sim")
    opened = install_arm(monkeypatch, FakeArm())
    result = invoke(["rest-pose", "record", "--yes"])
    assert result.exit_code == cli_bringup.EXIT_UNSUPPORTED and opened == []
    assert "sim backend" in result.output


# --- rest-pose show / clear ---------------------------------------------------------


def test_show_reports_the_model_default_until_a_pose_is_recorded() -> None:
    """``show`` resolves exactly as Collapse does: the model pose, never a placeholder."""
    result = invoke(["rest-pose", "show", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source"] == "default" and payload["placeholder"] is False
    assert payload["joints"] == MODEL_REST_POSE
    assert "MJCF" in payload["limits_source"]
    assert any("so101 rest-pose record" in warning for warning in payload["warnings"])
    human = invoke(["rest-pose", "show"])
    assert "model-derived default" in human.output and "placeholder" not in human.output
    assert payload["path"].endswith("rest_pose.json")

    assert invoke(["rest-pose", "record", "--yes"], input="\n").exit_code == 0
    human = invoke(["rest-pose", "show"])
    assert human.exit_code == 0
    assert "source: recorded" in human.output and "recorded 20" in human.output


def test_show_names_the_setting_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SO101_REST_POSE", "0,-95,90,70,0,2")
    get_settings.cache_clear()
    payload = json.loads(invoke(["rest-pose", "show", "--json"]).stdout)
    assert payload["source"] == "setting" and payload["joints"]["shoulder_lift"] == -95.0


def test_clear_confirms_before_deleting() -> None:
    assert "no recorded rest pose" in invoke(["rest-pose", "clear"]).output
    assert invoke(["rest-pose", "record", "--yes"], input="\n").exit_code == 0
    path = rp.rest_pose_path(get_settings())

    kept = invoke(["rest-pose", "clear"], input="n\n")
    assert kept.exit_code == 1 and path.exists()

    deleted = invoke(["rest-pose", "clear"], input="y\n")
    assert deleted.exit_code == 0 and not path.exists()
    assert "now uses the default pose" in deleted.output


def test_clear_yes_skips_the_question() -> None:
    assert invoke(["rest-pose", "record", "--yes"], input="\n").exit_code == 0
    result = invoke(["rest-pose", "clear", "--yes"])
    assert result.exit_code == 0 and not rp.rest_pose_path(get_settings()).exists()

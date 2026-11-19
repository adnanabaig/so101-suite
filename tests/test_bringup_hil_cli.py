"""CLI: ``so101 hil list|run|status|report`` in process, over HTTP, and refusing."""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from so101 import cli_bringup
from so101.bringup import ownership
from so101.bringup.evidence import verify_manifest
from so101.bringup.hil import HilConflict, HilError, HilInvalid, HilNotFound, hil_root
from so101.bringup.hil_remote import HilUnavailable, HttpHilApi
from so101.bringup.hil_steps import STEPS_BY_ID
from so101.bringup.ownership import ServerProbe
from so101.cli import app as root_app
from so101.config import get_settings
from so101.hardware import lerobot_paths

URL = "http://127.0.0.1:8816/api/robot/status"


def no_server(settings: Any) -> ServerProbe:
    return ServerProbe(running=False, url=URL, detail="nothing is listening")


def suite_server(settings: Any) -> ServerProbe:
    return ServerProbe(
        running=True,
        url=URL,
        detail=f"HTTP 200 from {URL}",
        http_status=200,
        status={"mode": "idle", "estopped": False},
    )


def foreign_server(settings: Any) -> ServerProbe:
    return ServerProbe(
        running=True,
        url=URL,
        detail=f"HTTP 404 from {URL} (not a Suite status reply; if another program uses "
        "this port, stop it)",
        http_status=404,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clock = FakeClock()
    monkeypatch.setattr(ownership, "probe_local_server", no_server)
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    monkeypatch.setattr(cli_bringup, "_sleep", clock.sleep)
    monkeypatch.setattr(cli_bringup, "_monotonic", clock.monotonic)
    monkeypatch.setenv("COLUMNS", "200")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def invoke(args: list[str], answers: list[str] | None = None) -> Any:
    text = None if answers is None else "\n".join(answers) + "\n"
    return CliRunner().invoke(root_app, args, input=text)


def runs() -> list[Path]:
    root = hil_root(get_settings())
    return sorted(root.iterdir()) if root.is_dir() else []


def run_json(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


TORQUE_PASS = ["y", "n", "pass", "y", "yes", ""]
HOST_KILL_PASS = ["pass", "left of the bench", "y", "y", ""]


def test_hil_commands_are_mounted() -> None:
    result = invoke(["hil", "--help"])
    assert result.exit_code == 0
    for command in ("list", "run", "status", "report"):
        assert command in result.output


def test_a_rehearsal_runs_in_process_and_re_asks_bad_answers() -> None:
    answers = [
        # v01-torque-transitions: probe, then a bad outcome before the real one
        *["y", "n", "maybe", "pass", "y", "yes", ""],
        # stop-halt: the mock server reports the wave-2 stop fields, so the
        # probe runs and fails (no Halt is active); a skip still needs a note
        *["y", "n", "skip", "", "rehearsal: halt not pressed"],
        # stop-host-kill: the server rejects acknowledged=n for a pass; enter it again
        *["pass", "left of the bench", "y", "n", ""],
        *HOST_KILL_PASS,
    ]
    result = invoke(
        [
            "hil",
            "run",
            "--operator",
            "Ada",
            "--step",
            "stop-host-kill",
            "--step",
            "v01-torque-transitions",
            "--step",
            "stop-halt",
        ],
        answers,
    )
    assert result.exit_code == 3, result.output
    output = result.output
    assert "REHEARSAL" in output and "this process (mock backend, no server)" in output
    assert "enter one of pass, fail, skip, quit" in output
    assert "Halt is not active" in output
    assert "acknowledged: a pass needs True" in output and "Nothing was recorded" in output
    assert "INCOMPLETE" in output and "manifest verifies" in output

    (run_dir,) = runs()
    run = run_json(run_dir)
    assert run["status"] == "incomplete" and run["rehearsal"] and not run["hil_evidence"]
    assert run["evidence_notes"][0].startswith("REHEARSAL")
    statuses = {step_id: record["status"] for step_id, record in run["steps"].items()}
    assert statuses == {
        "v01-torque-transitions": "pass",
        "stop-halt": "skip",
        "stop-host-kill": "pass",
    }
    host = run["steps"]["stop-host-kill"]["attestation"]
    assert host["evidence"] == {
        "power_switch_location": "left of the bench",
        "operator_can_reach_switch": True,
        "acknowledged": True,
    }
    assert run["steps"]["stop-halt"]["attestation"]["note"] == "rehearsal: halt not pressed"
    assert verify_manifest(run_dir).ok


def test_an_all_pass_run_exits_zero_and_a_failed_step_exits_one() -> None:
    passed = invoke(
        ["hil", "run", "-o", "Ada", "-s", "stop-host-kill", "-s", "v01-torque-transitions"],
        [*TORQUE_PASS, *HOST_KILL_PASS],
    )
    assert passed.exit_code == 0, passed.output
    assert "PASSED" in passed.output and "hil_evidence=false" in passed.output

    before = set(runs())
    failed = invoke(
        ["hil", "run", "-o", "Ada", "-s", "stop-host-kill"],
        ["fail", "", "n", "", "the switch is behind the arm"],
    )
    assert failed.exit_code == 1, failed.output
    (failed_dir,) = set(runs()) - before
    run = run_json(failed_dir)
    assert run["status"] == "failed"
    attestation = run["steps"]["stop-host-kill"]["attestation"]
    assert attestation["evidence"] == {"operator_can_reach_switch": False}


def test_a_probe_measurement_is_offered_as_the_default() -> None:
    result = invoke(
        ["hil", "run", "-o", "Ada", "-s", "gate4-diagnostics"], ["y", "n", "fail", "", "x"]
    )
    assert result.exit_code == 1, result.output
    (run_dir,) = runs()
    record = run_json(run_dir)["steps"]["gate4-diagnostics"]
    measured = record["probes"][0]["measured"]["failed_checks"]
    assert record["attestation"]["evidence"] == {"failed_checks": float(measured)}


def test_quit_leaves_the_run_open_and_resume_finishes_it() -> None:
    quit_run = invoke(
        ["hil", "run", "-o", "Ada", "-s", "stop-host-kill", "-s", "v01-training"], ["quit", "n"]
    )
    assert quit_run.exit_code == 3, quit_run.output
    (run_dir,) = runs()
    assert run_json(run_dir)["status"] == "in_progress"
    assert f"--resume {run_dir.name}" in quit_run.output

    resumed = invoke(
        ["hil", "run", "-o", "Bo", "--resume", run_dir.name],
        ["skip", "no GPU on this desk", *HOST_KILL_PASS],
    )
    assert resumed.exit_code == 3, resumed.output
    run = run_json(run_dir)
    assert run["status"] == "incomplete"
    assert run["steps"]["v01-training"]["attestation"]["operator"] == "Bo"

    again = invoke(["hil", "run", "-o", "Bo", "--resume", run_dir.name])
    assert again.exit_code == 2 and "already finalized" in again.output


def test_resume_skips_steps_already_attested() -> None:
    invoke(
        ["hil", "run", "-o", "Ada", "-s", "stop-host-kill", "-s", "v01-training"],
        ["skip", "later", "quit", "n"],
    )
    (run_dir,) = runs()
    resumed = invoke(["hil", "run", "-o", "Ada", "--resume", run_dir.name], HOST_KILL_PASS)
    assert resumed.exit_code == 3, resumed.output
    assert "1 of 2 steps have current attestations" in resumed.output
    assert "v01-training ·" not in resumed.output


def test_resume_rewalks_pre_contract_attestations_without_changing_run_id() -> None:
    invoke(
        ["hil", "run", "-o", "Ada", "-s", "gate1-workspace", "-s", "v01-training"],
        ["skip", "later", "quit", "n"],
    )
    (run_dir,) = runs()
    payload = run_json(run_dir)
    payload["catalog_version"] = 0
    payload["catalog_digest"] = ""
    attestation = payload["steps"]["gate1-workspace"]["attestation"]
    attestation["catalog_version"] = 0
    attestation["catalog_digest"] = ""
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    resumed = invoke(
        ["hil", "run", "-o", "Ada", "--resume", run_dir.name],
        ["skip", "re-attested", "skip", "not run"],
    )

    assert resumed.exit_code == 3, resumed.output
    assert "evidence contract changed" in resumed.output
    assert "gate1-workspace ·" in resumed.output
    current = run_json(run_dir)
    assert current["run_id"] == run_dir.name
    history = current["steps"]["gate1-workspace"]["attestation_history"]
    assert len(history) == 1 and history[0]["catalog_digest"] == ""


def test_quit_can_seal_the_run_as_incomplete() -> None:
    result = invoke(["hil", "run", "-o", "Ada", "-s", "v01-training"], ["quit", "y"])
    assert result.exit_code == 3, result.output
    (run_dir,) = runs()
    run = run_json(run_dir)
    assert run["status"] == "incomplete" and run["summary"]["pending"] == 1
    assert verify_manifest(run_dir).ok


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--group", "robot"], "unknown group robot"),
        (["--step", "nope"], "unknown HIL step or group: nope"),
        (["--resume", "20990101T000000Z-abcdef"], "no HIL run"),
    ],
)
def test_bad_arguments_exit_two(args: list[str], expected: str) -> None:
    result = invoke(["hil", "run", "-o", "Ada", *args])
    assert result.exit_code == 2 and expected in result.output, result.output
    assert runs() == []


def test_a_blank_operator_is_rejected() -> None:
    result = invoke(["hil", "run", "--operator", "   "])
    assert result.exit_code == 2 and "operator" in result.output


def test_the_real_backend_needs_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    result = invoke(["hil", "run", "-o", "Ada", "-s", "stop-host-kill"])
    assert result.exit_code == 3, result.output
    assert "refused" in result.output and "so101 serve" in result.output
    assert runs() == []


def test_a_foreign_program_on_the_port_refuses_runs_but_not_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ownership, "probe_local_server", foreign_server)
    result = invoke(["hil", "run", "-o", "Ada", "-s", "stop-host-kill"])
    assert result.exit_code == 3 and "not a Suite status reply" in result.output
    listed = invoke(["hil", "list"])
    assert listed.exit_code == 0 and "no HIL runs yet" in listed.output


# HttpHilApi passes per-request timeouts, which only TestClient warns about.
@pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")
def test_a_running_server_is_driven_over_http(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from so101.services.app import create_app

    seen: list[str] = []

    class SpyClient(TestClient):
        def request(self, method: str, url: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            seen.append(f"{method} {httpx.URL(str(url)).path}")
            return super().request(method, url, **kwargs)

    spy = SpyClient(create_app())
    monkeypatch.setattr(ownership, "probe_local_server", suite_server)
    monkeypatch.setattr(
        cli_bringup, "_hil_http", lambda base: HttpHilApi("http://testserver", client=spy)
    )
    result = invoke(["hil", "run", "-o", "Ada", "-s", "v01-torque-transitions"], TORQUE_PASS)
    assert result.exit_code == 0, result.output
    assert "via the Suite server at http://testserver" in result.output
    (run_dir,) = runs()
    run_id = run_dir.name
    assert seen == [
        "POST /api/hil/runs",
        "GET /api/hil/steps",
        f"POST /api/hil/runs/{run_id}/steps/v01-torque-transitions/probe",
        f"POST /api/hil/runs/{run_id}/steps/v01-torque-transitions/attest",
        f"POST /api/hil/runs/{run_id}/finalize",
        f"GET /api/hil/runs/{run_id}/report",
    ]
    listed = invoke(["hil", "list", "--json"])
    assert [item["run_id"] for item in json.loads(listed.stdout)] == [run_id]
    assert seen[-1] == "GET /api/hil/runs"


def test_list_and_status() -> None:
    assert invoke(["hil", "list"]).output.count("no HIL runs yet") == 1
    empty = invoke(["hil", "status"])
    assert empty.exit_code == 2 and "no HIL runs yet" in empty.output

    invoke(["hil", "run", "-o", "Ada", "-s", "stop-host-kill"], HOST_KILL_PASS)
    (run_dir,) = runs()
    table = invoke(["hil", "list"])
    assert run_dir.name in table.output and "mock (rehearsal)" in table.output
    listed = json.loads(invoke(["hil", "list", "--json"]).stdout)
    assert listed[0]["status"] == "passed" and listed[0]["steps_done"] == 1
    steps = json.loads(invoke(["hil", "list", "--steps", "--json"]).stdout)
    assert [step["id"] for step in steps] == list(STEPS_BY_ID)

    status = invoke(["hil", "status"])
    assert status.exit_code == 0 and "stop-host-kill" in status.output
    assert "REHEARSAL" in status.output and "hil_evidence=false" in status.output
    as_json = json.loads(invoke(["hil", "status", run_dir.name, "--json"]).stdout)
    assert as_json["run_id"] == run_dir.name and as_json["status"] == "passed"
    missing = invoke(["hil", "status", "20990101T000000Z-abcdef"])
    assert missing.exit_code == 2 and "no HIL run" in missing.output


def test_report_verifies_the_manifest() -> None:
    invoke(["hil", "run", "-o", "Ada", "-s", "stop-host-kill", "-s", "v01-training"], ["quit", "n"])
    (open_dir,) = runs()
    not_final = invoke(["hil", "report", open_dir.name])
    assert not_final.exit_code == 2 and "not finalized" in not_final.output

    invoke(
        ["hil", "run", "-o", "Ada", "--resume", open_dir.name], ["skip", "later", *HOST_KILL_PASS]
    )
    good = invoke(["hil", "report", open_dir.name])
    assert good.exit_code == 0, good.output
    assert "manifest: OK" in good.output
    hint = cli_bringup._manual_verify_command(open_dir)
    assert hint is None or f"check it yourself: {hint}" in good.output
    markdown = invoke(["hil", "report", open_dir.name, "--markdown"])
    assert markdown.exit_code == 0 and "REHEARSAL (mock backend)" in markdown.stdout

    with (open_dir / "report.md").open("a", encoding="utf-8") as handle:
        handle.write("edited later\n")
    bad = invoke(["hil", "report", open_dir.name])
    assert bad.exit_code == 1 and "changed: report.md" in bad.output
    as_json = invoke(["hil", "report", open_dir.name, "--json"])
    assert as_json.exit_code == 1
    assert json.loads(as_json.stdout)["manifest"]["mismatched"] == ["report.md"]


def test_paths_and_commands_print_on_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    invoke(["hil", "run", "-o", "Ada", "-s", "stop-host-kill"], HOST_KILL_PASS)
    (run_dir,) = runs()
    monkeypatch.setenv("COLUMNS", "40")
    monkeypatch.setattr(cli_bringup, "_which", lambda name: f"/usr/bin/{name}")
    result = invoke(["hil", "report", run_dir.name])
    assert result.exit_code == 0, result.output
    assert f"report: {run_dir / 'report.md'}\n" in result.output
    command = cli_bringup._manual_verify_command(run_dir)
    if command is not None:  # None on Windows
        assert f"check it yourself: {command}\n" in result.output


@pytest.mark.parametrize(
    ("platform", "tools", "expected"),
    [
        ("linux", {"sha256sum", "shasum"}, "sha256sum -c MANIFEST.sha256"),
        ("darwin", {"shasum"}, "shasum -a 256 -c MANIFEST.sha256"),
        ("linux", set(), None),
        ("win32", {"sha256sum", "shasum"}, None),  # Windows: so101 hil report
    ],
)
def test_manual_verify_command_per_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    tools: set[str],
    expected: str | None,
) -> None:
    monkeypatch.setattr(cli_bringup, "_which", lambda name: name if name in tools else None)
    run_dir = tmp_path / "a b"
    command = cli_bringup._manual_verify_command(run_dir, platform=platform)
    if expected is None:
        assert command is None
    else:
        assert command == f"cd {shlex.quote(str(run_dir))} && {expected}"


def test_run_summary_points_windows_at_so101_hil_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_bringup, "_which", lambda name: None)  # as on Windows
    result = invoke(["hil", "run", "-o", "Ada", "-s", "stop-host-kill"], HOST_KILL_PASS)
    assert result.exit_code == 0, result.output
    (run_dir,) = runs()
    assert f"check it again: so101 hil report {run_dir.name}" in result.output
    assert "check it yourself" not in result.output


# --- prompt helpers ------------------------------------------------------------------


def field(step_id: str, name: str) -> Any:
    return next(spec for spec in STEPS_BY_ID[step_id].evidence_fields if spec.name == name)


def test_field_hints_and_suggestions() -> None:
    drift = field("stop-halt", "max_drift_deg")
    assert cli_bringup._field_hint(drift) == " (deg; pass ≤ 1)"
    assert cli_bringup._field_hint(field("stop-estop", "torque_state")) == " (on/off/unconfirmed)"
    assert cli_bringup._field_hint(field("stop-halt", "object_released")) == " (y/n)"
    assert cli_bringup._suggestion(drift, 0.4) == "0.4"
    assert cli_bringup._suggestion(drift, True) == ""
    assert cli_bringup._suggestion(field("stop-halt", "object_released"), False) == "n"
    assert cli_bringup._suggestion(field("stop-estop", "torque_state"), "off") == "off"
    assert cli_bringup._suggestion(field("stop-estop", "torque_state"), "maybe") == ""


def test_field_parsing() -> None:
    flag = field("stop-halt", "object_released")
    assert cli_bringup._parse_field(flag, "Yes") == (True, None)
    assert cli_bringup._parse_field(flag, "n") == (False, None)
    assert cli_bringup._parse_field(flag, "sure") == (None, "answer y or n")
    number = field("stop-halt", "max_drift_deg")
    assert cli_bringup._parse_field(number, "0.25") == (0.25, None)
    assert cli_bringup._parse_field(number, "abc") == (None, "enter a number")
    assert cli_bringup._parse_field(number, "1e999") == (None, "enter a finite number")
    choice = field("stop-estop", "torque_state")
    assert cli_bringup._parse_field(choice, "off") == ("off", None)
    assert cli_bringup._parse_field(choice, "gone")[1] == "enter one of on, off, unconfirmed"


# --- the HTTP facade ----------------------------------------------------------------------


def api_replying(status: int, payload: Any, seen: list[httpx.Request] | None = None) -> HttpHilApi:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if isinstance(payload, str):
            return httpx.Response(status, text=payload)
        return httpx.Response(status, json=payload)

    return HttpHilApi("http://suite", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_http_errors_map_to_hil_errors() -> None:
    invalid = {"detail": {"message": "attestation rejected", "problems": ["note: needed"]}}
    with pytest.raises(HilInvalid) as rejected:
        api_replying(422, invalid).get_run("x")
    assert rejected.value.message == "attestation rejected"
    assert rejected.value.problems == ["note: needed"]

    validation = {"detail": [{"loc": ["body", "operator"], "msg": "Value error, blank"}]}
    with pytest.raises(HilInvalid) as body:
        api_replying(422, validation).list_runs()
    assert body.value.problems == ["operator: Value error, blank"]

    with pytest.raises(HilNotFound, match="no HIL run"):
        api_replying(404, {"detail": "no HIL run 'x'"}).get_run("x")
    with pytest.raises(HilConflict):
        api_replying(409, {"detail": "finalized"}).finalize("x")
    with pytest.raises(HilError) as crashed:
        api_replying(500, "Internal Server Error").steps()
    assert crashed.value.status_code == 500 and "HTTP 500" in crashed.value.message


def test_http_unreachable_server() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    api = HttpHilApi("http://suite", client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(HilUnavailable, match="did not answer"):
        api.list_runs()


def test_http_paths_are_quoted_and_probe_bodies_optional() -> None:
    seen: list[httpx.Request] = []
    probe = {
        "step_id": "stop-halt",
        "probe": "halt_hold",
        "sequence": 1,
        "status": "skip",
        "message": "m",
        "backend": "mock",
        "rehearsal": True,
        "started_at": "a",
        "finished_at": "b",
    }
    api = api_replying(200, probe, seen)
    api.probe("../x", "stop-halt")
    api.probe("r", "stop-halt", duration_s=5)
    assert seen[0].url.raw_path == b"/api/hil/runs/..%2Fx/steps/stop-halt/probe"
    assert seen[0].content == b""
    assert json.loads(seen[1].content) == {"duration_s": 5}

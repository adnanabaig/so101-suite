"""HIL runner: step catalog, evidence manifest, attestation, analyzers, service and router."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from so101 import JOINT_NAMES
from so101.bringup import hil
from so101.bringup import preflight as pf
from so101.bringup.evidence import (
    MANIFEST_NAME,
    evidence_files,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from so101.bringup.hil_steps import (
    DESIGN,
    DESIGN_COMMIT,
    GROUPS,
    HIL_CATALOG_DIGEST,
    HIL_CATALOG_VERSION,
    HIL_STEPS,
    STEPS_BY_ID,
    STOP_STATUS_FIELDS,
    select_steps,
    step_catalog,
)
from so101.bringup.schemas import (
    AttestRequest,
    CheckResult,
    HilRun,
    HilStepRecord,
    PreflightReport,
    ProbeResult,
    StartHilRunRequest,
)
from so101.config import get_settings
from so101.hardware import lerobot_paths

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
REST = {joint: 0.0 for joint in JOINT_NAMES}

EXPECTED_IDS = [
    "gate1-workspace",
    "gate2-bus",
    "gate3-install",
    "gate4-diagnostics",
    "gate5-calibration",
    "gate6-first-jog",
    "gate7-stop-tests",
    "stop-estop",
    "stop-halt",
    "stop-collapse",
    "stop-hold-timer",
    "stop-supervision-loss",
    "stop-host-kill",
    "v01-calibration-signs",
    "v01-torque-transitions",
    "v01-deadman-release",
    "v01-browser-loss",
    "v01-estop-power",
    "v01-camera-faults",
    "v01-collection-resume",
    "v01-training",
    "v01-rollout-cancel",
]


@pytest.fixture(autouse=True)
def _quiet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    get_settings.cache_clear()
    yield
    hil._PROBING.clear()


def iso(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


# --- the step catalog ------------------------------------------------------------


def test_catalog_has_every_step_in_order() -> None:
    assert HIL_CATALOG_VERSION >= 2
    assert re.fullmatch(r"[0-9a-f]{64}", HIL_CATALOG_DIGEST)
    canonical = json.dumps(
        [step.model_dump(mode="json") for step in HIL_STEPS],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert HIL_CATALOG_DIGEST == hashlib.sha256(canonical).hexdigest()
    assert GROUPS == ("gate", "stop", "verification")
    assert [step.id for step in HIL_STEPS] == EXPECTED_IDS
    assert [step.order for step in HIL_STEPS] == list(range(1, len(EXPECTED_IDS) + 1))
    groups = [step.group for step in HIL_STEPS]
    assert groups == sorted(groups, key=GROUPS.index), "groups must be contiguous, in order"
    assert set(STEPS_BY_ID) == set(EXPECTED_IDS)


@pytest.mark.parametrize("step", HIL_STEPS, ids=lambda step: step.id)
def test_each_step_is_well_formed(step: Any) -> None:
    assert step.title and step.instructions and step.expected_results and step.evidence_fields
    names = [spec.name for spec in step.evidence_fields]
    assert len(names) == len(set(names))
    for spec in step.evidence_fields:
        if spec.kind == "choice":
            assert spec.choices and (spec.pass_equals is None or spec.pass_equals in spec.choices)
        if spec.kind == "bool":
            assert spec.pass_equals in (True, False, None)
        if spec.kind != "number":
            assert spec.pass_min is None and spec.pass_max is None
    if step.probe == "preflight":
        assert step.probe_checks and set(step.probe_checks) <= set(pf.CHECKS_BY_ID)
    else:
        assert not step.probe_checks
    assert bool(step.probe_events) == (step.probe == "events")
    for requirement in step.requires:
        assert requirement.startswith("status.")
        assert requirement.split(".", 1)[1] in STOP_STATUS_FIELDS
    if step.probe in hil.SAMPLED_STOP_PROBES:
        assert step.requires, "sampled stop probes must name the status fields they need"
    if step.group == "stop" and step.probe != "none":
        assert step.requires, "every measured stop must name the status fields needed to pass"
    assert step.known_limit == (step.id == "stop-host-kill")


def test_stop_steps_carry_the_directive_evidence() -> None:
    def spec(step_id: str, name: str) -> Any:
        return next(f for f in STEPS_BY_ID[step_id].evidence_fields if f.name == name)

    assert spec("stop-estop", "arm_dropped").pass_equals is True
    assert spec("stop-estop", "torque_off_all_motors").pass_equals is True
    assert spec("stop-estop", "torque_state").pass_equals == "off"
    assert spec("stop-halt", "object_released").pass_equals is False
    assert spec("stop-halt", "max_drift_deg").pass_max == 1.0
    assert spec("stop-halt", "hold_duration_s").pass_min == 10.0
    assert spec("stop-collapse", "final_error_deg").pass_max == 3.0
    for name in ("speed_within_cap", "torque_off_at_end", "within_timeout"):
        assert spec("stop-collapse", name).pass_equals is True
    assert spec("stop-hold-timer", "collapse_started").pass_equals is True
    assert spec("stop-supervision-loss", "within_grace").pass_equals is True
    assert spec("stop-host-kill", "power_switch_location").kind == "text"
    assert spec("stop-host-kill", "acknowledged").pass_equals is True
    assert STEPS_BY_ID["stop-estop"].requires == ["status.torque_state"]
    for name in (
        "unplugged_session_refused",
        "no_motion_on_failed_start",
        "post_test_recovery_idle",
        "post_test_latches_clear",
    ):
        assert spec("gate3-install", name).pass_equals is True
    for name in (
        "physical_stop_cuts_power",
        "no_motion_on_power_restore",
        "recovery_idle",
        "latches_clear",
        "torque_off_before_stop_group",
    ):
        assert spec("gate7-stop-tests", name).pass_equals is True


def test_catalog_instructions_are_runnable_and_use_mechanical_catches() -> None:
    def instructions(step_id: str) -> str:
        return " ".join(STEPS_BY_ID[step_id].instructions)

    gate2 = instructions("gate2-bus")
    assert "manager lends" in gate2 and "read-only" in gate2
    assert "stop the server" not in gate2.lower() and "wave 2" not in gate2

    gate3 = instructions("gate3-install")
    assert "Suite UI" in gate3 and "server's teleop route" in gate3
    assert "Do not run `so101 teleop` while the server is running" in gate3
    assert "HTTP 200" in gate3 and "terminal fault" in gate3
    assert "deliberately reset" in gate3 and "no motion" in gate3

    gate5 = instructions("gate5-calibration")
    assert "Before starting this HIL run" in gate5
    assert "mechanical catch" in gate5 and "turn torque off" in gate5
    assert "hands-on calibration sweep" in gate5 and "pinch points" in gate5
    assert "During this HIL step leave the server running" in gate5
    assert "then start the server" not in gate5.lower()

    physical_stop_steps = (
        "gate1-workspace",
        "gate7-stop-tests",
        "v01-estop-power",
        "stop-estop",
        "stop-host-kill",
    )
    assert all("mechanical catch" in instructions(step_id) for step_id in physical_stop_steps)
    assert all("support the arm" not in instructions(step.id).lower() for step in HIL_STEPS)

    gate7 = instructions("gate7-stop-tests")
    assert "restore power" in gate7 and "idle" in gate7 and "latches" in gate7
    assert "before starting the stop group" in gate7.lower()

    collection = instructions("v01-collection-resume")
    assert "only HIL step that requires a server restart" in collection
    assert "same build, backend and data folder" in collection
    hold_timer = instructions("stop-hold-timer")
    assert "default 60-second" in hold_timer
    assert "pause this run without sealing it" in hold_timer


def test_every_catalog_step_obeys_the_server_lifecycle_contract() -> None:
    for step in HIL_STEPS:
        text = " ".join(step.instructions)
        if "so101 rest-pose record" in text or "so101 calibrate" in text:
            assert step.id == "gate5-calibration", step.id
            assert "Before starting this HIL run" in text
        if "server restart" in text or "restart the server" in text:
            assert step.id in {
                "gate5-calibration",  # explicitly forbidden during this step
                "v01-collection-resume",
                "stop-hold-timer",
            }, step.id

    collapse = " ".join(STEPS_BY_ID["stop-collapse"].instructions)
    assert "so101 rest-pose record" not in collapse
    assert "gate5-calibration" in collapse and "server is running" in collapse


_RANGE = re.compile(r"(\d+)(?:-(\d+))?")


def _cited(source: str) -> list[tuple[str, list[int]]]:
    """``[(path, line numbers)]`` for each ``;``-separated part of a step source."""
    parts: list[tuple[str, list[int]]] = []
    for part in source.split("; "):
        head = part.split()[0]
        path, _, spec = head.partition(":")
        if not spec:
            spec = part.split("lines ", 1)[1]
        numbers: list[int] = []
        for match in _RANGE.finditer(spec):
            low = int(match.group(1))
            numbers.extend(range(low, int(match.group(2) or low) + 1))
        parts.append((path, numbers))
    return parts


def _doc_lines(path: str) -> list[str] | None:
    if path != DESIGN:
        return (REPO / path).read_text(encoding="utf-8").splitlines()
    # Design §5 is cited at the commit that added it; it may not be in this checkout.
    try:
        shown = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{DESIGN_COMMIT}:{DESIGN}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        shown = None
    if shown is not None and shown.returncode == 0:
        return shown.stdout.splitlines()
    source = REPO / DESIGN
    if not source.is_file():
        return None
    text = source.read_text(encoding="utf-8")
    expected = "002fe19fcb55642233e82780e1c4d65ee5cf4ff11007f66fb68686f9101b59fd"
    if hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError("Archived HIL design differs from the cited a62fda0 source")
    return text.splitlines()


def test_design_quotes_remain_verifiable_in_source_snapshots(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    lines = _doc_lines(DESIGN)
    assert lines == (REPO / DESIGN).read_text(encoding="utf-8").splitlines()


def test_source_snapshot_rejects_changed_historical_design(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    source = tmp_path / DESIGN
    source.parent.mkdir(parents=True)
    source.write_text("changed design\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        _doc_lines(DESIGN)


def _norm(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("step", HIL_STEPS, ids=lambda step: step.id)
def test_step_quotes_are_verbatim_at_the_cited_lines(step: Any) -> None:
    texts = []
    for path, numbers in _cited(step.source):
        lines = _doc_lines(path)
        if lines is None:
            pytest.skip(f"{path} at {DESIGN_COMMIT} is not available in this checkout")
        assert numbers and max(numbers) <= len(lines), step.source
        texts.append(_norm(" ".join(lines[number - 1] for number in numbers)))
    for piece in step.quote.split(" … "):
        assert any(_norm(piece) in text for text in texts), f"{step.id}: {piece!r}"


def test_select_steps() -> None:
    assert select_steps() == EXPECTED_IDS
    assert select_steps([], []) == EXPECTED_IDS
    stops = select_steps(groups=["stop"])
    assert stops == [step_id for step_id in EXPECTED_IDS if step_id.startswith("stop-")]
    mixed = select_steps(["stop-halt", "gate2-bus"], ["verification"])
    assert mixed[:2] == ["gate2-bus", "stop-halt"]
    assert all(step_id.startswith("v01-") for step_id in mixed[2:])
    with pytest.raises(ValueError, match="nope, group robot"):
        select_steps(["gate2-bus", "nope"], ["robot"])


def test_step_catalog_returns_copies() -> None:
    catalog = step_catalog()
    catalog[0].instructions.append("changed")
    assert "changed" not in STEPS_BY_ID[catalog[0].id].instructions


def test_run_ids_sort_by_time_and_reject_paths() -> None:
    run_id = hil.new_run_id(T0)
    assert hil.RUN_ID_RE.match(run_id) and run_id.startswith("20261002T120000Z-")
    assert hil.new_run_id(T0 + timedelta(seconds=1)) > run_id
    for bad in ("../etc", "20261002T120000Z-zzzzzz", "x"):
        assert not hil.RUN_ID_RE.match(bad)


# --- the evidence manifest -----------------------------------------------------------


def _run_dir(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "artifacts" / "stop-halt").mkdir(parents=True)
    (root / "run.json").write_text('{"a": 1}\n', encoding="utf-8")
    (root / "report.md").write_text("# report\n", encoding="utf-8")
    (root / "artifacts" / "stop-halt" / "probe-1-samples.jsonl").write_text(
        '{"t": 0}\n', encoding="utf-8"
    )
    (root / ".tmp-in-flight").write_text("partial", encoding="utf-8")
    (root / "artifacts" / ".DS_Store").write_text("finder", encoding="utf-8")
    return root


def test_manifest_round_trip(tmp_path: Path) -> None:
    root = _run_dir(tmp_path)
    assert evidence_files(root) == [
        "artifacts/stop-halt/probe-1-samples.jsonl",
        "report.md",
        "run.json",
    ]
    path = write_manifest(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    digest, name = lines[2].split("  ")
    assert name == "run.json" and digest == sha256_file(root / "run.json")
    result = verify_manifest(root)
    assert result.ok and result.manifest_present and result.checked == 3
    (root / ".another-hidden").write_text("ignored", encoding="utf-8")
    assert verify_manifest(root).ok


@pytest.mark.parametrize(
    ("tool", "flags"), [("sha256sum", ["-c"]), ("shasum", ["-a", "256", "-c"])]
)
def test_manifest_checks_with_sha256sum_and_shasum(
    tmp_path: Path, tool: str, flags: list[str]
) -> None:
    executable = shutil.which(tool)
    if executable is None:
        pytest.skip(f"no {tool} on PATH")
    root = _run_dir(tmp_path)
    write_manifest(root)
    checked = subprocess.run([executable, *flags, MANIFEST_NAME], cwd=root, capture_output=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    (root / "run.json").write_text('{"a": 3}\n', encoding="utf-8")
    tampered = subprocess.run([executable, *flags, MANIFEST_NAME], cwd=root, capture_output=True)
    assert tampered.returncode != 0


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    root = _run_dir(tmp_path)
    write_manifest(root)
    (root / "run.json").write_text('{"a": 2}\n', encoding="utf-8")
    (root / "report.md").unlink()
    (root / "artifacts" / "added.json").write_text("{}", encoding="utf-8")
    result = verify_manifest(root)
    assert not result.ok
    assert result.mismatched == ["run.json"]
    assert result.missing == ["report.md"]
    assert result.extra == ["artifacts/added.json"]


def test_manifest_treats_a_symlink_as_missing(tmp_path: Path) -> None:
    root = _run_dir(tmp_path)
    write_manifest(root)
    target = root / "run.json"
    (tmp_path / "elsewhere.json").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.unlink()
    try:
        target.symlink_to(tmp_path / "elsewhere.json")
    except OSError:  # Windows without the symlink privilege
        pytest.skip("cannot create symlinks here")
    result = verify_manifest(root)
    assert not result.ok and result.missing == ["run.json"]
    assert "run.json" not in evidence_files(root)


def test_manifest_rejects_malformed_lines(tmp_path: Path) -> None:
    root = _run_dir(tmp_path)
    write_manifest(root)
    good = (root / MANIFEST_NAME).read_text(encoding="utf-8")
    digest = "0" * 64
    for bad in (
        "not a line",
        f"{digest}  /etc/passwd",
        f"{digest}  ../outside",
        f"{digest}  ./run.json",
        f"{digest}  .hidden",
        f"{digest}  {MANIFEST_NAME}",
        good.splitlines()[0],  # a duplicate
    ):
        (root / MANIFEST_NAME).write_text(good + bad + "\n", encoding="utf-8")
        result = verify_manifest(root)
        assert not result.ok and len(result.malformed) == 1, bad


def test_manifest_absent_or_empty_is_not_ok(tmp_path: Path) -> None:
    root = _run_dir(tmp_path)
    result = verify_manifest(root)
    assert not result.ok and not result.manifest_present
    (root / MANIFEST_NAME).write_text("", encoding="utf-8")
    result = verify_manifest(root)
    assert result.manifest_present and not result.ok and result.checked == 0


# --- attestation rules ---------------------------------------------------------------


def attest(outcome: str = "pass", note: str = "", **evidence: Any) -> AttestRequest:
    return AttestRequest(operator="Ada", outcome=outcome, note=note, evidence=evidence)  # type: ignore[arg-type]


def probe_result(status: str, step_id: str = "stop-halt") -> ProbeResult:
    required = [
        item.split(".", 1)[1]
        for item in STEPS_BY_ID[step_id].requires
        if item.startswith("status.")
    ]
    return ProbeResult(
        step_id=step_id,
        probe=STEPS_BY_ID[step_id].probe,
        sequence=1,
        status=status,  # type: ignore[arg-type]
        message="probe said so",
        catalog_version=HIL_CATALOG_VERSION,
        catalog_digest=HIL_CATALOG_DIGEST,
        required_status_fields=required,
        required_status_fields_present=True,
        backend="mock",
        rehearsal=True,
        started_at=iso(0),
        finished_at=iso(1),
    )


HALT_OK = {"object_released": False, "max_drift_deg": 0.4, "hold_duration_s": 12.0}


def test_a_valid_pass_is_accepted() -> None:
    step = STEPS_BY_ID["stop-halt"]
    passed = probe_result("pass")
    assert hil.validate_attestation(step, attest(**HALT_OK), passed) == []
    boundary = {**HALT_OK, "max_drift_deg": 1.0, "hold_duration_s": 10}
    assert hil.validate_attestation(step, attest(**boundary), passed) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"object_released": True}, "object_released: a pass needs False"),
        ({"max_drift_deg": 1.01}, "max_drift_deg: a pass needs at most 1"),
        ({"hold_duration_s": 9.99}, "hold_duration_s: a pass needs at least 10"),
        ({"object_released": "no"}, "object_released: must be true or false"),
        ({"max_drift_deg": True}, "max_drift_deg: must be a finite number"),
        ({"max_drift_deg": math.nan}, "max_drift_deg: must be a finite number"),
        ({"extra": 1}, "extra: not an evidence field of stop-halt"),
    ],
)
def test_a_pass_outside_the_bounds_is_rejected(change: dict[str, Any], expected: str) -> None:
    problems = hil.validate_attestation(
        STEPS_BY_ID["stop-halt"], attest(**{**HALT_OK, **change}), probe_result("pass")
    )
    assert any(problem.startswith(expected) for problem in problems), problems


def test_a_pass_needs_every_required_field() -> None:
    problems = hil.validate_attestation(
        STEPS_BY_ID["stop-halt"], attest(object_released=False), probe_result("pass")
    )
    assert sorted(problem.split(":")[0] for problem in problems) == [
        "hold_duration_s",
        "max_drift_deg",
    ]
    optional = hil.validate_attestation(
        STEPS_BY_ID["v01-calibration-signs"], attest(all_signs_match=True), None
    )
    assert optional == []


def test_choice_and_text_fields_are_checked() -> None:
    estop = STEPS_BY_ID["stop-estop"]
    passed = probe_result("pass", "stop-estop")
    good = {"arm_dropped": True, "torque_off_all_motors": True, "torque_state": "off"}
    assert hil.validate_attestation(estop, attest(**good), passed) == []
    wrong = hil.validate_attestation(estop, attest(**{**good, "torque_state": "maybe"}), passed)
    assert wrong == ["torque_state: must be one of on, off, unconfirmed"]
    on = hil.validate_attestation(estop, attest(**{**good, "torque_state": "on"}), passed)
    assert on == ["torque_state: a pass needs 'off', got 'on'"]
    unreported = probe_result("warn", "stop-estop")
    unreported.measured = {"torque_state_reported": False}
    blocked = hil.validate_attestation(estop, attest(**good), unreported)
    assert blocked == [
        "probe: stop-estop did not report torque_state; use the current server and probe "
        "again before attesting pass"
    ]
    host = STEPS_BY_ID["stop-host-kill"]
    blank = {"power_switch_location": "  ", "operator_can_reach_switch": True, "acknowledged": True}
    assert hil.validate_attestation(host, attest(**blank), None) == [
        "power_switch_location: must be non-empty text"
    ]


def test_fail_and_skip_need_a_note_but_not_every_field() -> None:
    step = STEPS_BY_ID["stop-halt"]
    for outcome in ("fail", "skip"):
        assert hil.validate_attestation(step, attest(outcome), None) == [
            f"note: a {outcome} needs a note saying why"
        ]
        assert hil.validate_attestation(step, attest(outcome, note="arm sagged"), None) == []
    # Values are still type-checked, but pass bounds do not apply.
    failed = attest("fail", note="drifted", max_drift_deg=4.0, object_released="yes")
    assert hil.validate_attestation(step, failed, None) == [
        "object_released: must be true or false"
    ]


def test_a_stop_pass_requires_a_passing_machine_probe() -> None:
    step = STEPS_BY_ID["stop-halt"]
    missing = hil.validate_attestation(step, attest(**HALT_OK), None)
    assert len(missing) == 1 and "must run its probe" in missing[0]
    blocked = hil.validate_attestation(step, attest(**HALT_OK), probe_result("fail"))
    assert len(blocked) == 1 and blocked[0].startswith("probe: the latest probe failed")
    skipped = hil.validate_attestation(step, attest(**HALT_OK), probe_result("skip"))
    assert len(skipped) == 1 and "latest stop probe skipped" in skipped[0]
    fields_missing = probe_result("pass")
    fields_missing.required_status_fields_present = False
    missing_fields = hil.validate_attestation(step, attest(**HALT_OK), fields_missing)
    assert len(missing_fields) == 1
    assert "lacks the current required status fields" in missing_fields[0]
    assert hil.validate_attestation(step, attest(**HALT_OK), probe_result("pass")) == []
    warned = hil.validate_attestation(step, attest(**HALT_OK), probe_result("warn"))
    assert len(warned) == 1 and "latest stop probe warned" in warned[0]
    assert hil.validate_attestation(step, attest("fail", note="x"), probe_result("fail")) == []

    # Non-stop observational steps retain their documented ability to attest
    # from observation when a probe cannot measure.
    verification = STEPS_BY_ID["v01-torque-transitions"]
    assert (
        hil.validate_attestation(
            verification, attest(**TORQUE_OK), probe_result("skip", verification.id)
        )
        == []
    )


# --- analyzers (synthetic wave-2 samples) -----------------------------------------------


def make_samples(
    seconds: float,
    *,
    state: Callable[[float], str] | str = "halted",
    positions: Callable[[float], dict[str, float]] | None = None,
    extra: Callable[[float], dict[str, Any]] | None = None,
    stale: bool = False,
    dt: float = 0.1,
) -> list[dict[str, Any]]:
    samples = []
    for i in range(int(round(seconds / dt)) + 1):
        t = round(i * dt, 3)
        status: dict[str, Any] = {
            "mode": "idle",
            "estopped": False,
            "stop_state": state(t) if callable(state) else state,
        }
        if extra is not None:
            status.update(extra(t))
        pose = positions(t) if positions else {joint: 10.0 for joint in JOINT_NAMES}
        telemetry = {
            "observation": {f"{joint}.pos": value for joint, value in pose.items()},
            "sample_seq": 0 if stale else i,
            "sample_age_ms": 5.0,
        }
        samples.append(hil.sample_record(t, T0 + timedelta(seconds=t), status, telemetry))
    return samples


def pose(**overrides: float) -> dict[str, float]:
    return {**{joint: 10.0 for joint in JOINT_NAMES}, **overrides}


def test_sample_record_keeps_positions_and_wave2_fields() -> None:
    record = hil.sample_record(
        1.23456,
        T0,
        {"mode": "teleop", "estopped": False, "stop_state": "halted", "halt": {"since": "x"}},
        {"observation": {"elbow_flex.pos": 12.5, "bogus.pos": 1, "wrist_roll.pos": "nan"}},
    )
    assert record["t"] == 1.235 and record["positions"] == {"elbow_flex": 12.5}
    assert record["stop_state"] == "halted" and record["halt"] == {"since": "x"}
    assert "torque_state" not in record and record["seq"] is None


def test_halt_hold_passes_when_still() -> None:
    noise = make_samples(12, positions=lambda t: pose(shoulder_lift=10.0 + 0.2 * math.sin(t)))
    verdict = hil.analyze_halt_hold(noise)
    assert verdict.status == "pass", verdict.message
    assert verdict.measured["hold_duration_s"] == 12.0
    assert verdict.measured["max_drift_deg"] <= 0.2


def test_halt_hold_drift_boundary_and_gripper() -> None:
    def at(value: float) -> Callable[[float], dict[str, float]]:
        return lambda t: pose(elbow_flex=value if t >= 6 else 10.0, gripper=10.0 + t)

    exact = hil.analyze_halt_hold(make_samples(12, positions=at(11.0)))
    assert exact.status == "pass" and exact.measured["max_drift_deg"] == 1.0
    assert exact.measured["gripper_range"] == 12.0, "the gripper is reported, not graded"
    over = hil.analyze_halt_hold(make_samples(12, positions=at(11.01)))
    assert over.status == "fail" and "elbow_flex drifted 1.01°" in over.message


def test_halt_hold_fails_short_or_inactive_and_warns_when_stale() -> None:
    short = hil.analyze_halt_hold(
        make_samples(12, state=lambda t: "halted" if t < 9.95 else "none")
    )
    assert short.status == "fail" and "ended as 'none'" in short.message
    assert short.measured["hold_duration_s"] == 9.9
    ten = hil.analyze_halt_hold(make_samples(12, state=lambda t: "halted" if t <= 10 else "none"))
    assert ten.status == "pass", ten.message
    inactive = hil.analyze_halt_hold(make_samples(12, state="none"))
    assert inactive.status == "fail" and "Halt is not active" in inactive.message
    stale = hil.analyze_halt_hold(make_samples(12, stale=True))
    assert stale.status == "warn" and "drift was not measured" in stale.message


def collapse_samples(
    *,
    speed: float = 12.0,
    start: float = 48.0,
    end: float = 0.0,
    torque_end: str = "off",
    still_collapsing: bool = False,
    stale: bool = False,
) -> list[dict[str, Any]]:
    pre, descent, post = 1.0, abs(start - end) / speed, 1.5

    def state(t: float) -> str:
        if t < pre:
            return "none"
        return "collapsing" if t < pre + descent or still_collapsing else "none"

    def where(t: float) -> dict[str, float]:
        value = start if t < pre else max(end, start - speed * (t - pre))
        return {**REST, "shoulder_lift": value, "gripper": 30.0}

    def extra(t: float) -> dict[str, Any]:
        collapsing = state(t) == "collapsing"
        done = t >= pre + descent and not still_collapsing
        return {
            "torque_state": torque_end if done else "on",
            "collapse": {"since": iso(pre), "reason": "operator"} if collapsing else None,
        }

    return make_samples(
        pre + descent + post, state=state, positions=where, extra=extra, stale=stale
    )


def test_collapse_passes_within_cap_timeout_and_settle() -> None:
    verdict = hil.analyze_collapse(collapse_samples(), rest_pose=REST)
    assert verdict.status == "pass", verdict.message
    measured = verdict.measured
    assert measured["timeout_s"] == round(48 / 15 + 2, 2)
    assert measured["duration_s"] == 4.0 and measured["within_timeout"]
    assert measured["max_speed_deg_s"] == pytest.approx(12.0, abs=0.01)
    assert measured["final_error_deg"] == 0.0 and measured["torque_off_at_end"]
    assert measured["reason"] == "operator"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"speed": 20.0}, "a joint moved at 20.0°/s"),
        ({"end": 4.0}, "ended 4.0° from the rest pose"),
        ({"speed": 8.0}, "took 6.0 s (timeout 5.2 s)"),
        ({"torque_end": "on"}, "torque_state at the end is 'on'"),
        ({"torque_end": "unconfirmed"}, "torque_state at the end is 'unconfirmed'"),
        ({"still_collapsing": True}, "still collapsing"),
    ],
)
def test_collapse_failures(kwargs: dict[str, Any], expected: str) -> None:
    verdict = hil.analyze_collapse(collapse_samples(**kwargs), rest_pose=REST)
    assert verdict.status == "fail" and expected in verdict.message, verdict.message


def test_collapse_needs_a_collapse_and_honors_the_timeout_setting() -> None:
    none = hil.analyze_collapse(make_samples(5, state="none"), rest_pose=REST)
    assert none.status == "fail" and "no Collapse was seen" in none.message
    slow = hil.analyze_collapse(collapse_samples(speed=8.0), rest_pose=REST, timeout_s=10.0)
    assert (
        slow.status == "pass" and slow.measured["timeout_basis"] == "SO101_COLLAPSE_TIMEOUT_SECONDS"
    )
    fallback = hil.analyze_collapse(
        collapse_samples(), rest_pose=REST, events=[{"type": "safety.collapse.fallback"}]
    )
    assert fallback.status == "fail" and "fallback" in fallback.message
    stale = hil.analyze_collapse(collapse_samples(stale=True), rest_pose=REST)
    assert stale.status == "warn" and stale.measured["final_error_deg"] is None


def test_collapse_timeout_default_is_capped() -> None:
    assert hil.collapse_timeout({"shoulder_lift": 30.0}, REST, 15.0) == 4.0
    assert hil.collapse_timeout({"shoulder_lift": 300.0}, REST, 15.0) == 12.0


HALT_INFO = {"since": iso(-50), "server_time": iso(0), "hold_expires_at": iso(10)}


def hold_samples(
    *, collapse_at: float | None = None, clear_at: float | None = None, halt: Any = HALT_INFO
) -> list[dict[str, Any]]:
    def state(t: float) -> str:
        if clear_at is not None and t >= clear_at:
            return "none"
        if collapse_at is not None and t >= collapse_at:
            return "collapsing"
        return "halted"

    return make_samples(20, state=state, extra=lambda t: {"halt": halt})


@pytest.mark.parametrize(("collapse_at", "status"), [(10.5, "pass"), (9.0, "pass"), (13.0, "pass")])
def test_hold_timer_on_time(collapse_at: float, status: str) -> None:
    verdict = hil.analyze_hold_timer(hold_samples(collapse_at=collapse_at))
    assert verdict.status == status, verdict.message
    assert verdict.measured["expected_after_s"] == 10.0
    assert verdict.measured["collapse_started_after_s"] == pytest.approx(50 + collapse_at)


def test_hold_timer_failures_and_missing_timing() -> None:
    late = hil.analyze_hold_timer(hold_samples(collapse_at=13.5))
    assert late.status == "fail" and "+3.5 s" in late.message
    early = hil.analyze_hold_timer(hold_samples(collapse_at=8.5))
    assert early.status == "fail" and "-1.5 s" in early.message
    cleared = hil.analyze_hold_timer(hold_samples(clear_at=5.0, collapse_at=12.0))
    assert cleared.status == "fail" and "cleared" in cleared.message
    never = hil.analyze_hold_timer(hold_samples())
    assert never.status == "fail" and "no Collapse" in never.message
    untimed = hil.analyze_hold_timer(hold_samples(collapse_at=10.0, halt={}))
    assert untimed.status == "warn" and "not verified" in untimed.message


def supervision_samples(
    *, collapse_at: float | None = 6.5, lost: float | None = 2.0, reason: str = "supervision_lost"
) -> list[dict[str, Any]]:
    def state(t: float) -> str:
        return "collapsing" if collapse_at is not None and t >= collapse_at else "halted"

    def extra(t: float) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if state(t) == "collapsing":
            fields["collapse"] = {"since": iso(collapse_at or 0), "reason": reason}
        if lost is not None:
            fields["supervision"] = {"lost_at": iso(lost)}
        return fields

    return make_samples(15, state=state, extra=extra)


def test_supervision_loss() -> None:
    ok = hil.analyze_supervision_loss(supervision_samples())
    assert ok.status == "pass" and ok.measured["seconds_after_loss"] == 4.5
    boundary = hil.analyze_supervision_loss(supervision_samples(collapse_at=10.0))
    assert boundary.status == "pass" and boundary.measured["within_grace"]
    late = hil.analyze_supervision_loss(supervision_samples(collapse_at=11.5))
    assert late.status == "fail" and "9.5 s after supervision was lost" in late.message
    stopwatch = hil.analyze_supervision_loss(supervision_samples(lost=None))
    assert stopwatch.status == "warn" and "stopwatch" in stopwatch.message
    other = hil.analyze_supervision_loss(supervision_samples(reason="operator"))
    assert other.status == "warn" and "does not mention supervision" in other.message
    none = hil.analyze_supervision_loss(supervision_samples(collapse_at=None))
    assert none.status == "fail" and "no Collapse" in none.message


def test_estop_analyzer() -> None:
    assert hil.analyze_estop({"estopped": False}).status == "fail"
    unverified = hil.analyze_estop({"estopped": True})
    assert unverified.status == "warn" and "not machine-verified" in unverified.message
    assert hil.analyze_estop({"estopped": True, "torque_state": "off"}).status == "pass"
    unconfirmed = hil.analyze_estop({"estopped": True, "torque_state": "unconfirmed"})
    assert unconfirmed.status == "fail" and "cut follower power" in unconfirmed.message
    assert hil.analyze_estop({"estopped": True, "torque_state": "on"}).status == "fail"


def test_status_analyzer_reports_cameras_and_stop_fields() -> None:
    status = {
        "mode": "idle",
        "estopped": False,
        "cameras": ["wrist", "top"],
        "live_cameras": ["top"],
    }
    verdict = hil.analyze_status(status, None, {"safety_state": "disarmed", "armed": False})
    assert verdict.status == "pass" and verdict.measured["missing_cameras"] == ["wrist"]
    assert "missing: wrist" in verdict.message
    faulted = hil.analyze_status({**status, "torque_state": "off"}, "bus timeout", None)
    assert faulted.status == "warn" and faulted.measured["torque_state"] == "off"


def test_telemetry_analyzer() -> None:
    stale = hil.analyze_telemetry(make_samples(5, stale=True))
    assert stale.status == "skip" and "did not update" in stale.message
    still = hil.analyze_telemetry(make_samples(5))
    assert still.status == "warn"
    moved = hil.analyze_telemetry(make_samples(5, positions=lambda t: pose(wrist_flex=10 + t)))
    assert moved.status == "pass" and moved.measured["moved_joints"] == ["wrist_flex"]
    assert moved.measured["range_deg"]["wrist_flex"] == 5.0


def test_events_analyzer() -> None:
    events = [
        {"id": 1, "type": "control.armed"},
        {"id": 2, "type": "runtime.operator_timeout", "message": "browser lost"},
    ]
    found = hil.analyze_events(events, ["runtime.operator_timeout", "safety."])
    assert found.status == "pass" and [e["id"] for e in found.measured["matched"]] == [2]
    missing = hil.analyze_events(events[:1], ["safety."])
    assert missing.status == "warn" and "none of the 1 audit events" in missing.message
    assert hil.analyze_events([], ["safety."]).status == "warn"


def test_diagnostics_analyzer() -> None:
    def payload(*statuses: str) -> dict[str, Any]:
        return {
            "overall": "x",
            "checks": [
                {"id": f"c{i}", "label": f"Check {i}", "status": s, "detail": "d"}
                for i, s in enumerate(statuses)
            ],
        }

    failed = hil.analyze_diagnostics(payload("pass", "fail", "warn"))
    assert failed.status == "fail" and failed.measured["failed_checks"] == 1
    assert hil.analyze_diagnostics(payload("pass", "warn")).status == "warn"
    assert hil.analyze_diagnostics(payload("pass")).status == "pass"


def report(
    origin: str, backend: str, checks: list[CheckResult], *, age: timedelta = timedelta()
) -> PreflightReport:
    return PreflightReport(
        created_at=(datetime.now(timezone.utc) - age).isoformat(),
        so101_version="test",
        origin=origin,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        rehearsal=backend != "real",
        arms=["follower"],
        bus_access="deferred" if origin == "api" else "opened",
        overall="pass",
        ready_for_hardware=False,
        exit_code=0,
        counts={},
        checks=checks,
    )


def check(
    check_id: str, status: str, message: str = "ok", *, rehearsal: bool = False
) -> CheckResult:
    return CheckResult(
        id=check_id,
        key=f"{check_id}.follower",
        arm="follower",
        title=check_id,
        status=status,  # type: ignore[arg-type]
        message=message,
        rehearsal=rehearsal,
    )


def test_grade_preflight_uses_a_fresh_cli_report_for_deferred_bus_checks() -> None:
    deferred = check("motors", "skip", pf.WAVE2_BUS_MESSAGE)
    api = report("api", "real", [deferred, check("environment", "pass")])
    cli = report("cli", "real", [check("motors", "pass")])
    graded = hil.grade_preflight(api, cli, ["motors", "environment"])
    assert graded.status == "pass", graded.message
    assert [c["from"] for c in graded.measured["checks"]] == ["cli", "api"]
    assert graded.measured["cli_report_created_at"] == cli.created_at

    alone = hil.grade_preflight(api, None, ["motors"], cli_note="no saved report")
    assert alone.status == "skip" and "no saved report" in alone.message
    failing = report("api", "real", [check("motors", "fail", "ID 3 missing")])
    assert hil.grade_preflight(failing, cli, ["motors"]).status == "fail"
    warned = report("api", "real", [check("environment", "warn")])
    assert hil.grade_preflight(warned, None, ["environment"]).status == "warn"


def test_cli_report_must_match_the_backend_and_be_recent() -> None:
    now = datetime.now(timezone.utc)
    cli = report("cli", "real", [check("motors", "pass")])
    assert hil._cli_report_usable(cli, "real", now) == (cli, None)
    assert "mock backend" in str(hil._cli_report_usable(report("cli", "mock", []), "real", now)[1])
    old = report("cli", "real", [], age=timedelta(hours=3))
    assert "older than 2 h" in str(hil._cli_report_usable(old, "real", now)[1])
    assert hil._cli_report_usable(None, "real", now)[0] is None


# --- the service with a fake client ---------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.mono = 0.0

    def monotonic(self) -> float:
        return self.mono

    def now(self) -> datetime:
        return T0 + timedelta(seconds=self.mono)

    def sleep(self, seconds: float) -> None:
        self.mono += seconds


class FakeClient:
    """A HilClient whose status and telemetry follow the fake clock."""

    def __init__(self, settings: Any, clock: FakeClock) -> None:
        self.settings = settings
        self.clock = clock
        self.status_at: Callable[[float], dict[str, Any]] = lambda t: {
            "mode": "idle",
            "estopped": False,
            "cameras": [],
            "live_cameras": [],
        }
        self.positions_at: Callable[[float], dict[str, float]] = lambda t: dict(REST)
        self.events: list[dict[str, Any]] = []
        self.api_report: PreflightReport | None = None
        self.image: Any = None
        self.on_status: Callable[[], None] | None = None

    def status(self) -> dict[str, Any]:
        if self.on_status is not None:
            self.on_status()
        return self.status_at(self.clock.mono)

    def fault(self) -> str | None:
        return None

    def control_state(self) -> dict[str, Any]:
        return {"armed": False, "safety_state": "disarmed"}

    def telemetry(self) -> dict[str, Any]:
        positions = self.positions_at(self.clock.mono)
        return {
            "observation": {f"{joint}.pos": value for joint, value in positions.items()},
            "sample_seq": int(round(self.clock.mono * 10)),
            "sample_age_ms": 3.0,
        }

    def latest_event_id(self) -> int:
        return self.events[-1]["id"] if self.events else 0

    def events_after(self, after_id: int | None) -> list[dict[str, Any]]:
        return [e for e in self.events if after_id is None or e["id"] > after_id]

    def diagnostics(self) -> dict[str, Any]:
        return {"overall": "pass", "checks": [{"id": "a", "label": "A", "status": "pass"}]}

    def preflight(self) -> PreflightReport:
        assert self.api_report is not None
        return self.api_report

    def rest_pose(self) -> Any:
        return SimpleNamespace(joints=dict(REST), source="recorded", placeholder=False, warnings=[])

    def health(self) -> None:
        return None

    def frame(self) -> Any:
        return None if self.image is None else ("wrist", self.image)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def service_for(client: FakeClient, clock: FakeClock) -> hil.HilService:
    return hil.HilService(client, clock=clock.now, monotonic=clock.monotonic, sleep=clock.sleep)


@pytest.fixture
def client(clock: FakeClock) -> FakeClient:
    return FakeClient(get_settings(), clock)


@pytest.fixture
def service(client: FakeClient, clock: FakeClock) -> hil.HilService:
    return service_for(client, clock)


def start(service: hil.HilService, *step_ids: str) -> HilRun:
    return service.start_run(StartHilRunRequest(operator="Ada", step_ids=list(step_ids) or None))


HOST_KILL_OK = {
    "power_switch_location": "left of the bench",
    "operator_can_reach_switch": True,
    "acknowledged": True,
}
TORQUE_OK = {"no_jump_on_arm": True, "stale_goal_held": True}


def test_start_run_writes_the_layout_and_labels_rehearsals(
    service: hil.HilService, client: FakeClient
) -> None:
    client.events = [{"id": 7, "type": "control.armed"}]
    run = start(service, "stop-halt", "gate1-workspace")
    assert run.step_ids == ["gate1-workspace", "stop-halt"]
    assert run.catalog_version == HIL_CATALOG_VERSION
    assert run.catalog_digest == HIL_CATALOG_DIGEST
    assert run.backend == "mock" and run.rehearsal and not run.hil_evidence
    assert run.event_cursor == 7 and run.status == "in_progress"
    assert run.devices["follower_calibration_digest"] == "none"
    run_dir = hil.hil_root(get_settings()) / run.run_id
    assert (run_dir / "run.json").is_file()
    step_file = json.loads((run_dir / "steps" / "stop-halt.json").read_text(encoding="utf-8"))
    assert (
        step_file["definition"]["id"] == "stop-halt" and step_file["record"]["status"] == "pending"
    )
    assert service.get_run(run.run_id) == run

    with pytest.raises(hil.HilInvalid) as invalid:
        start(service, "nope")
    assert invalid.value.problems == ["unknown HIL step or group: nope"]
    for bad in ("../../etc", "20991231T000000Z-abcdef"):
        with pytest.raises(hil.HilNotFound):
            service.get_run(bad)


def test_list_runs_is_newest_first(service: hil.HilService, clock: FakeClock) -> None:
    first = start(service, "gate1-workspace")
    clock.sleep(5)
    second = start(service, "gate1-workspace")
    (hil.hil_root(get_settings()) / "not-a-run").mkdir()
    runs = service.list_runs()
    assert [summary.run_id for summary in runs] == [second.run_id, first.run_id]
    assert runs[0].steps_total == 1 and runs[0].steps_done == 0


def test_list_runs_orders_runs_started_in_the_same_second(
    service: hil.HilService, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The older run gets the suffix that sorts last by name.
    suffixes = iter(["ffffff", "000000"])
    monkeypatch.setattr(hil, "new_run_id", lambda now: f"{now:%Y%m%dT%H%M%SZ}-{next(suffixes)}")
    older = start(service, "gate1-workspace")
    clock.sleep(0.25)
    newer = start(service, "gate1-workspace")
    assert older.run_id[:16] == newer.run_id[:16]
    assert [summary.run_id for summary in service.list_runs()] == [newer.run_id, older.run_id]


def test_attest_reattest_and_finalize_statuses(service: hil.HilService) -> None:
    run = start(service, "stop-host-kill", "v01-training")
    with pytest.raises(hil.HilInvalid) as rejected:
        service.attest(run.run_id, "stop-host-kill", attest())
    assert len(rejected.value.problems) == 3
    with pytest.raises(hil.HilNotFound):
        service.attest(run.run_id, "stop-halt", attest(**HALT_OK))
    service.attest(run.run_id, "stop-host-kill", attest("fail", note="switch is behind the arm"))
    record = service.attest(run.run_id, "stop-host-kill", attest(**HOST_KILL_OK))
    assert record.status == "pass" and record.attestation and record.attestation.note == ""
    assert len(record.attestation_history) == 1
    assert record.attestation_history[0].outcome == "fail"

    service.attest(run.run_id, "v01-training", attest("skip", note="no GPU today"))
    assert service.finalize(run.run_id).status == "incomplete"


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [(("pass", "pass"), "passed"), (("pass", "fail"), "failed"), (("pass", None), "incomplete")],
)
def test_finalize_grades_the_run(
    service: hil.HilService, outcomes: tuple[str | None, ...], expected: str
) -> None:
    run = start(service, "stop-host-kill", "v01-torque-transitions")
    evidence = {"stop-host-kill": HOST_KILL_OK, "v01-torque-transitions": TORQUE_OK}
    for step_id, outcome in zip(run.step_ids, outcomes, strict=True):
        if outcome == "pass":
            service.attest(run.run_id, step_id, attest(**evidence[step_id]))
        elif outcome == "fail":
            service.attest(run.run_id, step_id, attest("fail", note="no"))
    final = service.finalize(run.run_id)
    assert final.status == expected and final.finalized_at
    assert final.summary["steps"] == 2 and not final.hil_evidence
    assert final.evidence_notes[0].startswith("REHEARSAL")


def test_a_finalized_run_is_immutable(service: hil.HilService) -> None:
    run = start(service, "stop-host-kill", "v01-torque-transitions")
    service.finalize(run.run_id)
    for action in (
        lambda: service.attest(run.run_id, "stop-host-kill", attest(**HOST_KILL_OK)),
        lambda: service.probe(run.run_id, "v01-torque-transitions"),
        lambda: service.finalize(run.run_id),
    ):
        with pytest.raises(hil.HilConflict):
            action()


def test_pre_fix_open_run_requires_current_stop_probe_and_reattestation(
    service: hil.HilService,
) -> None:
    run = start(service, "stop-halt")
    first = service.probe(run.run_id, "stop-halt")
    assert first.status == "fail"
    service.attest(run.run_id, "stop-halt", attest("fail", note="not halted"))

    run_dir = hil.hil_root(get_settings()) / run.run_id
    legacy = service.get_run(run.run_id)
    legacy.catalog_version = 0
    legacy.catalog_digest = ""
    record = legacy.steps["stop-halt"]
    assert record.attestation is not None
    record.attestation.catalog_version = 0
    record.attestation.catalog_digest = ""
    record.probes[0].catalog_version = 0
    record.probes[0].catalog_digest = ""
    service._save_step(run_dir, STEPS_BY_ID["stop-halt"], record)
    service._save_run(run_dir, legacy)

    with pytest.raises(hil.HilInvalid) as stale:
        service.finalize(run.run_id)
    assert any("re-attest stop-halt" in problem for problem in stale.value.problems)
    assert any("re-probe stop-halt" in problem for problem in stale.value.problems)

    second = service.probe(run.run_id, "stop-halt")
    assert second.catalog_digest == HIL_CATALOG_DIGEST
    with pytest.raises(hil.HilInvalid, match="older HIL evidence contract"):
        service.finalize(run.run_id)
    service.attest(run.run_id, "stop-halt", attest("fail", note="still not halted"))
    final = service.finalize(run.run_id)

    assert final.run_id == run.run_id and final.status == "failed"
    assert final.catalog_version == HIL_CATALOG_VERSION
    assert final.catalog_digest == HIL_CATALOG_DIGEST
    assert len(final.steps["stop-halt"].probes) == 2
    assert final.steps["stop-halt"].attestation is not None
    assert final.steps["stop-halt"].attestation.catalog_digest == HIL_CATALOG_DIGEST
    assert len(final.steps["stop-halt"].attestation_history) == 1
    assert final.steps["stop-halt"].attestation_history[0].catalog_digest == ""


def test_report_verifies_the_manifest_and_detects_tampering(service: hil.HilService) -> None:
    run = start(service, "stop-host-kill", "v01-torque-transitions")
    with pytest.raises(hil.HilConflict):
        service.report(run.run_id)
    service.probe(run.run_id, "v01-torque-transitions")
    service.attest(run.run_id, "stop-host-kill", attest(**HOST_KILL_OK))
    service.finalize(run.run_id)

    result = service.report(run.run_id)
    assert result.manifest.ok and result.manifest.checked >= 6
    assert "REHEARSAL (mock backend)" in result.markdown
    assert "shasum -a 256 -c MANIFEST.sha256" in result.markdown
    assert "sha256sum -c MANIFEST.sha256" in result.markdown
    assert f"so101 hil report {run.run_id}" in result.markdown
    run_dir = Path(result.run_dir)
    names = set(evidence_files(run_dir))
    assert {"run.json", "report.json", "report.md", "steps/stop-host-kill.json"} <= names
    for name in [run_dir.name, *names]:  # legal on Windows: no ':' from a timestamp
        assert not set('<>:"|?*\\') & set(name), name
    report_json = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert (
        report_json["kind"] == "so101.hil_report" and report_json["run"]["status"] == "incomplete"
    )

    artifact = run_dir / result.run.steps["v01-torque-transitions"].probes[0].artifacts[0]
    artifact.write_text(
        artifact.read_text(encoding="utf-8").replace("pass", "PASS"), encoding="utf-8"
    )
    tampered = service.report(run.run_id).manifest
    assert not tampered.ok and tampered.mismatched == [artifact.relative_to(run_dir).as_posix()]


def test_hil_evidence_needs_the_real_backend(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    service = service_for(FakeClient(get_settings(), clock), clock)
    run = start(service, "stop-host-kill", "v01-torque-transitions")
    assert run.backend == "real" and not run.rehearsal
    probe = service.probe(run.run_id, "v01-torque-transitions")
    assert probe.backend == "real" and not probe.rehearsal and probe.status == "pass"
    service.attest(run.run_id, "v01-torque-transitions", attest(**TORQUE_OK))
    service.attest(run.run_id, "stop-host-kill", attest(**HOST_KILL_OK))
    final = service.finalize(run.run_id)
    assert final.status == "passed" and final.hil_evidence and final.evidence_notes == []
    assert service.report(run.run_id).markdown.splitlines()[2].startswith("**Hardware evidence:**")

    unprobed = start(service, "v01-torque-transitions")
    service.attest(unprobed.run_id, "v01-torque-transitions", attest(**TORQUE_OK))
    caveat = service.finalize(unprobed.run_id)
    assert caveat.hil_evidence
    assert caveat.evidence_notes == [
        "attested pass without running the probe: v01-torque-transitions"
    ]

    skipped = start(service, "v01-training")
    service.attest(skipped.run_id, "v01-training", attest("skip", note="later"))
    nothing = service.finalize(skipped.run_id)
    assert (
        not nothing.hil_evidence and "no step was attested pass or fail" in nothing.evidence_notes
    )


def test_evidence_verdict_rejects_rehearsal_probes_in_a_real_run() -> None:
    record = HilStepRecord(step_id="v01-torque-transitions", status="pass")
    record.probes.append(probe_result("pass", "v01-torque-transitions"))
    run = HilRun(
        run_id="20261002T120000Z-abcdef",
        created_at=iso(0),
        so101_version="test",
        backend="real",
        rehearsal=False,
        operator="Ada",
        step_ids=["v01-torque-transitions"],
        steps={"v01-torque-transitions": record},
    )
    evidence, notes = hil.evidence_verdict(run)
    assert not evidence
    assert notes == [
        "probes that did not run on the real backend: v01-torque-transitions probe 1 (mock)"
    ]


@pytest.mark.parametrize("record_status", ["pending", "pass", "fail", "skip"])
def test_every_measured_stop_without_a_probe_is_an_evidence_gap(record_status: str) -> None:
    record = HilStepRecord(step_id="stop-halt", status=record_status)  # type: ignore[arg-type]
    run = HilRun(
        run_id="20261002T120000Z-abcdef",
        created_at=iso(0),
        so101_version="test",
        backend="real",
        rehearsal=False,
        operator="Ada",
        step_ids=["stop-halt"],
        steps={"stop-halt": record},
        catalog_version=HIL_CATALOG_VERSION,
        catalog_digest=HIL_CATALOG_DIGEST,
    )
    evidence, notes = hil.evidence_verdict(run)
    assert evidence is False
    assert any("stop-halt has no current probe" in note for note in notes)
    if record_status == "pass":
        assert hil.final_status(run) == "incomplete"


def test_stop_probe_history_with_skip_or_legacy_unverified_estop_never_counts() -> None:
    passed = probe_result("pass", "stop-halt")
    skipped = probe_result("skip", "stop-halt")
    halt_record = HilStepRecord(step_id="stop-halt", status="pass", probes=[skipped, passed])

    legacy_warn = probe_result("warn", "stop-estop")
    legacy_warn.catalog_version = 0
    legacy_warn.catalog_digest = ""
    legacy_warn.measured = {"torque_state_reported": False}
    verified = probe_result("pass", "stop-estop")
    verified.measured = {"torque_state_reported": True, "torque_state": "off"}
    estop_record = HilStepRecord(
        step_id="stop-estop", status="pass", probes=[legacy_warn, verified]
    )

    run = HilRun(
        run_id="20261002T120000Z-abcdef",
        created_at=iso(0),
        so101_version="test",
        backend="real",
        rehearsal=False,
        operator="Ada",
        step_ids=["stop-halt", "stop-estop"],
        steps={"stop-halt": halt_record, "stop-estop": estop_record},
        catalog_version=HIL_CATALOG_VERSION,
        catalog_digest=HIL_CATALOG_DIGEST,
    )

    gaps = hil.stop_evidence_gaps(run)
    assert any("stop-halt probe 1 skipped" in gap for gap in gaps)
    assert any("stop-estop probe 1 did not report torque_state" in gap for gap in gaps)
    assert hil.final_status(run) == "incomplete"
    evidence, notes = hil.evidence_verdict(run)
    assert evidence is False and any("stop evidence is incomplete" in note for note in notes)


def test_warned_stop_measurement_cannot_finalize_as_passed_hardware_evidence() -> None:
    warned = probe_result("warn", "stop-halt")
    warned.backend = "real"
    warned.rehearsal = False
    warned.measured = {"fresh_samples": 1, "max_drift_deg": None}
    record = HilStepRecord(step_id="stop-halt", status="pass", probes=[warned])
    run = HilRun(
        run_id="20261002T120000Z-abcdef",
        created_at=iso(0),
        so101_version="test",
        backend="real",
        rehearsal=False,
        operator="Ada",
        step_ids=["stop-halt"],
        steps={"stop-halt": record},
        catalog_version=HIL_CATALOG_VERSION,
        catalog_digest=HIL_CATALOG_DIGEST,
    )

    assert any("warned" in gap for gap in hil.stop_evidence_gaps(run))
    assert hil.final_status(run) == "incomplete"
    evidence, notes = hil.evidence_verdict(run)
    assert evidence is False
    assert any("stop evidence is incomplete" in note for note in notes)


def test_a_sampled_probe_writes_samples_and_a_bundle(
    service: hil.HilService, client: FakeClient
) -> None:
    import numpy as np

    halt = {"since": iso(0), "server_time": iso(0), "hold_expires_at": iso(60)}
    client.status_at = lambda t: {
        "mode": "idle",
        "estopped": False,
        "stop_state": "halted",
        "torque_state": "on",
        "halt": halt,
        "live_cameras": ["wrist"],
    }
    client.image = np.zeros((2, 3, 3), dtype=np.uint8)
    client.events = [{"id": 1, "type": "control.armed"}]
    run = start(service, "stop-halt")
    client.events.append({"id": 2, "type": "safety.halt"})

    result = service.probe(run.run_id, "stop-halt")
    assert result.status == "pass" and result.sequence == 1, result.message
    assert result.measured["hold_duration_s"] >= 12.0
    run_dir = hil.hil_root(get_settings()) / run.run_id
    bundle_name, *others = result.artifacts
    assert bundle_name == "artifacts/stop-halt/probe-1-halt_hold.json"
    assert set(others) == {
        "artifacts/stop-halt/probe-1-frame-wrist.npy",
        "artifacts/stop-halt/probe-1-samples.jsonl",
    }
    bundle = json.loads((run_dir / bundle_name).read_text(encoding="utf-8"))
    assert bundle["kind"] == "so101.hil_probe" and bundle["rehearsal"] is True
    assert bundle["robot_status"]["stop_state"] == "halted"
    assert bundle["control_state"]["safety_state"] == "disarmed"
    assert [event["id"] for event in bundle["audit_excerpt"]] == [2]
    assert bundle["health"] is None and "no servo health" in bundle["health_note"]
    lines = (
        (run_dir / "artifacts/stop-halt/probe-1-samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) >= 121 and json.loads(lines[0])["stop_state"] == "halted"
    assert np.load(run_dir / "artifacts/stop-halt/probe-1-frame-wrist.npy").shape == (2, 3, 3)
    assert service.get_run(run.run_id).steps["stop-halt"].probes[0] == result


def test_a_collapse_probe_grades_the_descent(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The suite's test settings speed up mock collapses; this grades a 15 deg/s
    # descent against the shipped default.
    monkeypatch.setenv("SO101_COLLAPSE_SPEED_DEG_S", "15")
    get_settings.cache_clear()
    client = FakeClient(get_settings(), clock)
    service = service_for(client, clock)

    def status_at(t: float) -> dict[str, Any]:
        collapsing = 2.0 <= t < 6.0
        return {
            "mode": "idle",
            "estopped": False,
            "stop_state": "collapsing" if collapsing else "none",
            "torque_state": "off" if t >= 6.0 else "on",
            "collapse": {"since": iso(2), "reason": "operator"} if collapsing else None,
        }

    client.status_at = status_at
    client.positions_at = lambda t: {
        **REST,
        "shoulder_lift": min(48.0, max(0.0, 48 - 12 * (t - 2))),
    }
    run = start(service, "stop-collapse")
    result = service.probe(run.run_id, "stop-collapse")
    assert result.status == "pass", result.message
    assert result.measured["rest_pose"]["source"] == "recorded"
    assert result.measured["duration_s"] == pytest.approx(4.0, abs=0.11)


def test_stop_probes_skip_without_required_status_fields(service: hil.HilService) -> None:
    run = start(
        service,
        "stop-estop",
        "stop-halt",
        "stop-collapse",
        "stop-hold-timer",
        "stop-supervision-loss",
    )
    for step_id in run.step_ids:
        result = service.probe(run.run_id, step_id)
        assert result.status == "skip" and "incompatible server" in result.message
        assert result.measured["missing_status_fields"]


def test_stop_probe_status_crash_never_claims_required_fields_or_hil_evidence(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    client = FakeClient(get_settings(), clock)

    def status_crash() -> dict[str, Any]:
        raise RuntimeError("status unavailable before observation")

    client.status = status_crash  # type: ignore[method-assign]
    service = service_for(client, clock)
    step_ids = [
        "stop-estop",
        "stop-halt",
        "stop-collapse",
        "stop-hold-timer",
        "stop-supervision-loss",
    ]
    run = start(service, *step_ids)
    for step_id in step_ids:
        result = service.probe(run.run_id, step_id)
        expected = [item.split(".", 1)[1] for item in STEPS_BY_ID[step_id].requires]
        assert result.status == "fail" and "status unavailable" in result.message
        assert result.required_status_fields == expected
        assert result.required_status_fields_present is False
        service.attest(run.run_id, step_id, attest("fail", note="status read crashed"))

    final = service.finalize(run.run_id)
    assert final.status == "failed" and final.hil_evidence is False
    assert any("lacks current required status fields" in note for note in final.evidence_notes)


def test_probe_rules(service: hil.HilService, client: FakeClient) -> None:
    run = start(service, "gate1-workspace", "v01-estop-power", "gate4-diagnostics")
    with pytest.raises(hil.HilInvalid):
        service.probe(run.run_id, "gate1-workspace")
    with pytest.raises(hil.HilNotFound):
        service.probe(run.run_id, "stop-halt")
    assert service.probe(run.run_id, "v01-estop-power").status == "fail"
    client.status_at = lambda t: {"mode": "idle", "estopped": True}
    second = service.probe(run.run_id, "v01-estop-power")
    assert second.status == "warn" and second.sequence == 2
    diag = service.probe(run.run_id, "gate4-diagnostics")
    assert diag.status == "pass"
    assert "artifacts/gate4-diagnostics/probe-1-diagnostics.json" in diag.artifacts

    def boom() -> dict[str, Any]:
        raise RuntimeError("store is gone")

    client.diagnostics = boom  # type: ignore[method-assign]
    crashed = service.probe(run.run_id, "gate4-diagnostics")
    assert crashed.status == "fail" and "RuntimeError: store is gone" in crashed.message


def test_a_preflight_probe_falls_back_to_the_cli_report(
    service: hil.HilService, client: FakeClient
) -> None:
    deferred = check("motors", "skip", pf.WAVE2_BUS_MESSAGE, rehearsal=True)
    client.api_report = report(
        "api",
        "mock",
        [deferred, check("serial_ports", "pass", rehearsal=True)],
    )
    pf.save_report(get_settings(), report("cli", "mock", [check("motors", "pass", rehearsal=True)]))
    run = start(service, "gate2-bus")
    result = service.probe(run.run_id, "gate2-bus")
    assert result.status == "pass", result.message
    assert "from the `so101 preflight` report" in result.message
    assert {
        "artifacts/gate2-bus/probe-1-preflight-api.json",
        "artifacts/gate2-bus/probe-1-preflight-cli.json",
    } <= set(result.artifacts)


def test_concurrent_probes_and_finalize_conflict(
    service: hil.HilService, client: FakeClient
) -> None:
    run = start(service, "v01-torque-transitions", "v01-camera-faults")
    entered, release = threading.Event(), threading.Event()

    def block_once() -> None:
        client.on_status = None
        entered.set()
        assert release.wait(10)

    client.on_status = block_once
    results: list[Any] = []
    worker = threading.Thread(
        target=lambda: results.append(service.probe(run.run_id, "v01-torque-transitions"))
    )
    worker.start()
    try:
        assert entered.wait(10)
        with pytest.raises(hil.HilConflict, match="already running"):
            service.probe(run.run_id, "v01-camera-faults")
        with pytest.raises(hil.HilConflict, match="still running"):
            service.finalize(run.run_id)
        service.attest(run.run_id, "v01-camera-faults", attest("skip", note="no cameras"))
    finally:
        release.set()
        worker.join(10)
    assert results and results[0].status == "pass"
    assert service.finalize(run.run_id).status == "incomplete"


# --- the HTTP router (mock backend) -----------------------------------------------------------

fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from so101.services.app import create_app
    from so101.services.routers import hil as hil_router
    from so101.services.routers import preflight as preflight_router

    fake = FakeClock()
    monkeypatch.setattr(hil_router, "_sleep", fake.sleep)
    monkeypatch.setattr(hil_router, "_monotonic", fake.monotonic)
    monkeypatch.setattr(preflight_router, "_sleep", lambda seconds: None)
    yield TestClient(create_app())


def test_router_runs_a_rehearsal_end_to_end(api: Any) -> None:
    steps = api.get("/api/hil/steps").json()
    assert [step["id"] for step in steps] == EXPECTED_IDS
    created = api.post("/api/hil/runs", json={"operator": "Ada", "groups": ["stop"]})
    assert created.status_code == 201
    run = created.json()
    assert run["rehearsal"] is True and run["backend"] == "mock"
    base = f"/api/hil/runs/{run['run_id']}"

    halt = api.post(f"{base}/steps/stop-halt/probe").json()
    # The server reports the wave-2 stop fields, so the probe runs; with no
    # Halt pressed it fails rather than skipping.
    assert halt["status"] == "fail" and "Halt is not active" in halt["message"]
    no_probe = api.post(f"{base}/steps/stop-host-kill/probe")
    assert no_probe.status_code == 422 and no_probe.json()["detail"]["problems"]
    assert api.post(f"{base}/steps/gate1-workspace/probe").status_code == 404
    assert api.post(f"{base}/steps/stop-halt/probe", json={"duration_s": 0}).status_code == 422
    assert api.post(f"{base}/steps/stop-halt/probe", json={"duration_s": 181}).status_code == 422

    bad = api.post(
        f"{base}/steps/stop-host-kill/attest", json={"operator": "Ada", "outcome": "pass"}
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["message"] == "attestation for stop-host-kill rejected"
    good = api.post(
        f"{base}/steps/stop-host-kill/attest",
        json={"operator": "Ada", "outcome": "pass", "evidence": HOST_KILL_OK},
    )
    assert good.status_code == 200 and good.json()["status"] == "pass"

    assert api.get(f"{base}/report").status_code == 409
    final = api.post(f"{base}/finalize").json()
    assert final["status"] == "incomplete" and final["hil_evidence"] is False
    assert api.post(f"{base}/finalize").status_code == 409
    assert api.post(f"{base}/steps/stop-halt/probe").status_code == 409
    sealed = api.get(f"{base}/report").json()
    assert sealed["manifest"]["ok"] is True and sealed["run"]["run_id"] == run["run_id"]
    listed = api.get("/api/hil/runs").json()
    assert listed[0]["run_id"] == run["run_id"] and listed[0]["steps_done"] == 1


def test_router_estop_probe_on_the_mock_server(api: Any) -> None:
    from so101.services.robot_manager import get_manager

    run = api.post("/api/hil/runs", json={"operator": "Ada", "step_ids": ["stop-estop"]}).json()
    url = f"/api/hil/runs/{run['run_id']}/steps/stop-estop/probe"
    assert api.post(url).json()["status"] == "fail"
    get_manager().estop()
    try:
        latched = api.post(url).json()
    finally:
        get_manager().reset_estop(operator_id="test-operator")
    # The server reports torque_state, so the latch and torque-off are verified.
    assert latched["status"] == "pass" and 'torque_state is "off"' in latched["message"]
    assert latched["rehearsal"] is True


def test_router_validation_errors(api: Any) -> None:
    assert api.post("/api/hil/runs", json={"operator": "  "}).status_code == 422
    assert api.post("/api/hil/runs", json={"operator": "Ada", "groups": ["x"]}).status_code == 422
    unknown = api.post("/api/hil/runs", json={"operator": "Ada", "step_ids": ["nope"]})
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["problems"] == ["unknown HIL step or group: nope"]
    assert api.get("/api/hil/runs/20261002T120000Z-abcdef").status_code == 404
    assert api.get("/api/hil/runs/not-a-run").status_code == 404
    assert api.get("/api/hil/runs").json() == []

"""Doc claims that the code can check (review findings L15, L16, L17).

Each test ties a sentence in README, docs/ or the Makefile to the code that
makes it true, so the sentence cannot go stale silently again.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import typer

from so101.cli import app
from so101.config import default_arm_port
from so101.services.schemas import ExecuteAgentProposalRequest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _prose_files() -> list[Path]:
    return [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "Makefile",
        *sorted((REPO_ROOT / "docs").glob("*.md")),
    ]


def test_no_doc_says_a_restart_clears_the_estop_latch():
    # RobotManager._restore_latches brings the E-stop latch, faults and the
    # unconfirmed-torque flag back after any restart, --reload included.
    claim = re.compile(r"(restart|reload)\w*\s+clears?\s+the\s+(in-memory\s+)?E-stop\s+latch", re.I)
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _prose_files()
        if claim.search(" ".join(path.read_text(encoding="utf-8").split()))
    ]
    assert offenders == []


def test_make_verify_is_not_presented_as_the_release_gate():
    assert "verify: test lint build-ui ## Full software release gate" not in _text("Makefile")
    assert "2. `make verify` passes" not in _text("CONTRIBUTING.md")
    assert "Run `make verify` before treating a build as ready" not in _text(
        "docs/getting-started.md"
    )
    suite_v0 = _text("docs/SUITE_V0.md")
    release = suite_v0[suite_v0.index("## Release verification") :]
    assert "```bash\nmake verify\n```" not in release
    assert "```bash\nmake ci\n```" in release
    assert "`make ci` passes" in _text("CONTRIBUTING.md")


def test_architecture_describes_the_approval_token_not_approved_true():
    fields = ExecuteAgentProposalRequest.model_fields
    assert fields["approval_token"].is_required() and "approved" not in fields
    with pytest.raises(ValueError):
        ExecuteAgentProposalRequest.model_validate({"lease_id": "l", "approved": True})
    architecture = " ".join(_text("docs/architecture.md").split())
    assert "approval_token" in architecture
    assert "`approved=true`;" not in architecture
    assert "one arm/command/disarm step" not in architecture


def test_the_agent_limitation_no_longer_waits_for_the_stop_dispatcher():
    security = " ".join(_text("docs/control-plane-security.md").split())
    assert "Once the v0.2 stop dispatcher lands" not in security
    assert "A run ends with a global disarm" not in security


def test_architecture_shows_the_fallback_dataset_folder_name(tmp_path, monkeypatch):
    from so101.config import get_settings
    from so101.data.dataset import dataset_root

    monkeypatch.setenv("SO101_DATASETS_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        name = dataset_root("demo/pick").name
    finally:
        get_settings.cache_clear()
    assert name == "v2--demo%2Fpick"
    architecture = _text("docs/architecture.md")
    assert f"-> {name}" in architecture
    assert "datasets/v2--<percent-encoded repo id>/" in architecture


def _record_options() -> set[str]:
    record = typer.main.get_command(app).commands["record"]
    names: set[str] = set()
    for param in record.params:
        names.update(param.opts)
        names.update(getattr(param, "secondary_opts", []))
    return names


def test_readme_record_examples_use_real_flags_and_do_not_wait_for_input():
    options = _record_options()
    assert "--no-prompt" in options
    lines = [line for line in _text("README.md").splitlines() if " so101 record " in line]
    assert lines, "README has a so101 record example"
    for line in lines:
        words = shlex.split(line.split("#", 1)[0])
        flags = {word.split("=", 1)[0] for word in words if word.startswith("--")}
        assert flags <= options, (line, flags - options)
        assert "--no-prompt" in flags, f"the mock CLI example waits for Enter: {line}"


def test_bring_up_names_the_linux_default_ports():
    text = " ".join(_text("docs/bring-up.md").split())
    for arm in ("follower", "leader"):
        assert default_arm_port(arm, "linux") in text
    assert "so Linux and Windows must always set both ports" not in text
    assert "whatever directory it runs from" in text


def test_bring_up_orders_stop_evidence_before_data_and_training():
    text = _text("docs/bring-up.md")
    gate = text.index("--group gate")
    stop = text.index("`--group stop`", gate)
    verification = text.index("`--group verification`", stop)
    assert gate < stop < verification
    assert "mechanical catch" in text
    assert "The server must still be running and idle" in text
    assert "version and SHA-256" in text
    assert "never erases probe history" in text


def test_v1_plan_runs_stop_hil_before_verification():
    text = _text("docs/plans/2026-10-07-v1-e2e-execution.md")
    milestone = text[text.index("### M6") : text.index("### M7")]
    gate = milestone.index("1. `gate`")
    stop = milestone.index("2. `stop`")
    verification = milestone.index("3. `verification`")
    assert gate < stop < verification
    assert "Stop HIL passes before collection, training or rollout" in milestone
    assert "G15 is the only v1 robot host and HIL evidence source" in text
    assert "Thor workstation" in text and "does not own motor buses" in text

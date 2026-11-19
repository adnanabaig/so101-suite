"""Eval statistics and durable, self-describing eval reports (audit B12, B14).

Wilson 95% intervals; reports that record where their numbers come from and
how the run ended; atomic, uniquely named saves; the per-trial journal; and
the CLI's workspace-reset gate with partial reports on Ctrl-C or a declined
prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from so101.cli import app as cli_app
from so101.config import get_settings
from so101.eval import evaluator
from so101.eval.evaluator import (
    EvalAborted,
    EvalConfig,
    EvalJournalError,
    EvalReport,
    EvalRun,
)
from so101.eval.stats import success_summary, wilson_interval
from so101.training.run_manifest import RUN_MANIFEST_NAME

# -- helpers ---------------------------------------------------------------------------


def _mock_checkpoint(fps: int = 15) -> Path:
    run_dir = get_settings().outputs_dir / "run"
    path = run_dir / "checkpoints" / "last" / "pretrained_model"
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"type": "act", "mock": True}), encoding="utf-8")
    (run_dir / RUN_MANIFEST_NAME).write_text(json.dumps({"fps": fps}), encoding="utf-8")
    return path


def _eval_files(suffix: str) -> list[Path]:
    return sorted(get_settings().outputs_dir.glob(f"eval_*{suffix}"))


def _journal(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _cfg(path: Path, episodes: int) -> EvalConfig:
    return EvalConfig(model_path=str(path), n_episodes=episodes, max_seconds=0.02)


def _invoke(*args: str, input: str = ""):
    result = CliRunner().invoke(cli_app, list(args), input=input)
    return result, " ".join(result.output.split())


# -- Wilson interval -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.2775),
        (1, 10, 0.0179, 0.4042),
        (5, 10, 0.2366, 0.7634),
        (8, 10, 0.4902, 0.9433),
        (10, 10, 0.7225, 1.0),
    ],
)
def test_wilson_interval_matches_reference_values(successes, n, low, high):
    got = wilson_interval(successes, n)
    assert got == pytest.approx((low, high), abs=5e-5)


def test_wilson_interval_is_symmetric_and_bounded():
    for n in (1, 2, 7, 30):
        for successes in range(n + 1):
            low, high = wilson_interval(successes, n)
            mirror_low, mirror_high = wilson_interval(n - successes, n)
            assert 0.0 <= low <= successes / n <= high <= 1.0
            assert low == pytest.approx(1.0 - mirror_high, abs=1e-12)
            assert high == pytest.approx(1.0 - mirror_low, abs=1e-12)


def test_wilson_interval_without_trials_carries_no_information():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("successes", "n", "z"),
    [(11, 10, 1.96), (-1, 10, 1.96), (0, -1, 1.96), (True, 1, 1.96), (1, 2, 0.0)],
)
def test_wilson_interval_rejects_impossible_inputs(successes, n, z):
    with pytest.raises(ValueError):
        wilson_interval(successes, n, z)


def test_success_summary_shows_the_interval():
    assert success_summary(8, 10) == "8/10 = 80% (95% CI 49-94%)"
    assert success_summary(0, 10) == "0/10 = 0% (95% CI 0-28%)"
    assert success_summary(0, 0) == "0 scored trials"


# -- EvalReport ------------------------------------------------------------------------


def test_report_built_the_old_way_gains_consistent_derived_fields(tmp_path):
    """Reports from older positional producers stay valid."""

    episodes = [
        {"episode": 1, "success": True, "duration": 1.0},
        {"episode": 2, "success": False, "duration": 1.0},
        {"episode": 3, "success": None, "duration": 0.4, "interrupted": "stopped"},
    ]
    report = EvalReport(
        model_path="m",
        n_episodes=3,
        successes=1,
        success_rate=1 / 3,
        per_episode=[*episodes, {"error": "boom", "phase": "failed"}],
    )
    # The producer's own numbers are kept, and the interval matches them.
    assert report.success_rate == pytest.approx(1 / 3)
    assert report.n == 3
    assert report.ci95 == [round(v, 4) for v in wilson_interval(1, 3)]
    assert (report.scored, report.interrupted) == (2, 1)
    assert (report.phase, report.error) == ("failed", "boom")
    assert report.requested_episodes is None  # not recorded by that producer
    assert report.source == "mock"
    assert report.report_version == 2

    reports = tmp_path / "reports"
    path = report.save(reports / "eval_explicit.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_path", "n_episodes", "successes", "success_rate", "per_episode"):
        assert key in saved
    assert saved["source"] == "mock"
    assert [p.name for p in reports.iterdir()] == ["eval_explicit.json"]


def test_from_trials_scores_only_scored_trials():
    report = EvalReport.from_trials(
        "m",
        [
            {"episode": 1, "success": True},
            {"episode": 2, "success": False},
            {"episode": 3, "success": None, "interrupted": "KeyboardInterrupt"},
        ],
        requested_episodes=5,
        phase="aborted",
        seeds=[3, 4, 5],
        source="sim",
    )
    assert report.success_rate == 0.5
    assert (report.n, report.scored, report.interrupted) == (2, 2, 1)
    assert report.n_episodes == 3
    assert report.requested_episodes == 5
    assert report.ci95 == [round(v, 4) for v in wilson_interval(1, 2)]
    assert report.seeds == [3, 4, 5]
    assert report.source == "sim"
    with pytest.raises(ValueError, match="phase"):
        EvalReport.from_trials("m", [], phase="paused")


def test_reports_saved_in_the_same_millisecond_never_collide():
    first = EvalReport("m", 1, 1, 1.0, created_at=1000.0)
    second = EvalReport("m", 1, 0, 0.0, created_at=1000.0)
    paths = {first.save(), second.save()}
    assert len(paths) == 2
    assert all(path.name.startswith("eval_1000000_") for path in paths)


def test_a_failed_save_leaves_no_partial_file(tmp_path):
    report = EvalReport("m", 1, 1, 1.0, per_episode=[{"success": True, "blob": object()}])
    reports = tmp_path / "reports"
    with pytest.raises(TypeError):
        report.save(reports / "eval.json")
    assert list(reports.iterdir()) == []


def test_report_directory_fsync_error_propagates_and_quarantines_the_file(tmp_path, monkeypatch):
    report = EvalReport("m", 1, 1, 1.0, per_episode=[{"success": True}])
    path = tmp_path / "reports" / "eval.json"

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError("injected directory fsync ENOSPC")

    monkeypatch.setattr(evaluator, "fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync ENOSPC"):
        report.save(path)

    assert not path.exists()
    assert len(list(path.parent.glob("eval.json.invalid-*"))) == 1


def test_report_directory_fsync_and_quarantine_rename_failure_remove_false_green_file(
    tmp_path, monkeypatch
):
    report = EvalReport("m", 1, 1, 1.0, per_episode=[{"success": True}])
    path = tmp_path / "reports" / "eval.json"
    real_replace = evaluator.os.replace
    replaces = 0

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError("injected directory fsync EIO")

    def replace(source, target) -> None:
        nonlocal replaces
        replaces += 1
        if replaces == 2:
            raise OSError("injected quarantine rename failure")
        real_replace(source, target)

    monkeypatch.setattr(evaluator, "fsync_directory", fail_directory_fsync)
    monkeypatch.setattr(evaluator.os, "replace", replace)
    with pytest.raises(OSError, match="directory fsync EIO"):
        report.save(path)

    assert not path.exists()
    assert list(path.parent.glob("eval.json.invalid-*")) == []


def test_journal_lines_are_fsynced(tmp_path, monkeypatch):
    synced: list[int] = []
    real_fsync = evaluator.os.fsync

    def counting_fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(evaluator.os, "fsync", counting_fsync)
    journal = evaluator.EvalJournal(tmp_path / "eval.jsonl")
    journal.write("start", requested_episodes=2)
    journal.write("trial", episode=1, success=True)
    journal.close()
    lines = _journal(tmp_path / "eval.jsonl")
    assert [line["event"] for line in lines] == ["start", "trial"]
    assert lines[1]["success"] is True and "t" in lines[1]
    assert len(synced) == 3  # first line + new directory entry, then second line


def test_saved_eval_refuses_before_connect_when_initial_journal_fsync_fails(monkeypatch):
    path = _mock_checkpoint()
    connected: list[bool] = []
    monkeypatch.setattr(evaluator.PolicyRunner, "connect", lambda self: connected.append(True))

    def fail_fsync(fd: int) -> None:
        raise OSError("injected initial journal failure")

    monkeypatch.setattr(evaluator.os, "fsync", fail_fsync)
    run = EvalRun(_cfg(path, 1), save=True)

    with pytest.raises(EvalJournalError, match="initial journal failure"):
        run.run()

    assert connected == []
    assert run.report_path is None and run.journal_path is None
    assert _eval_files(".json") == [] and _eval_files(".jsonl") == []


def test_saved_eval_trial_journal_failure_stops_before_the_next_trial(monkeypatch):
    path = _mock_checkpoint()
    real_fsync = evaluator.os.fsync
    fsync_calls = 0
    rollout_calls = 0
    real_rollout = evaluator.PolicyRunner.rollout

    def fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 3:
            raise OSError("injected trial journal failure")
        real_fsync(fd)

    def rollout(self, duration=30.0):
        nonlocal rollout_calls
        rollout_calls += 1
        return real_rollout(self, duration)

    monkeypatch.setattr(evaluator.os, "fsync", fsync)
    monkeypatch.setattr(evaluator.PolicyRunner, "rollout", rollout)
    run = EvalRun(_cfg(path, 2), score_fn=lambda episode: True, save=True)

    with pytest.raises(EvalJournalError, match="trial journal failure"):
        run.run()

    assert rollout_calls == 1
    assert run.report is not None
    assert (run.report.phase, run.report.scored, run.report.successes) == ("failed", 1, 1)
    assert "trial journal failure" in run.report.error
    assert run.report_path is not None
    saved = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert saved["per_episode"][0]["success"] is True


def test_cli_report_save_failure_is_journaled_as_failed(monkeypatch):
    path = _mock_checkpoint()

    def fail_save(self, path=None):
        raise OSError("injected report publication failure")

    monkeypatch.setattr(EvalReport, "save", fail_save)
    run = EvalRun(_cfg(path, 1), score_fn=lambda episode: True, save=True)
    report = run.run()

    assert report.phase == "failed"
    assert "report not saved: injected report publication failure" in report.error
    assert run.report_path is None
    assert run.journal_path is not None
    final = _journal(run.journal_path)[-1]
    assert final["event"] == "end"
    assert final["phase"] == "failed"
    assert final["report"] is None
    assert "report not saved: injected report publication failure" in final["error"]


def test_teardown_error_stays_primary_when_terminal_journal_fsync_also_fails(monkeypatch):
    path = _mock_checkpoint()
    real_fsync = evaluator.os.fsync
    fsync_calls = 0
    real_disconnect = evaluator.PolicyRunner.disconnect_owned

    def fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 6:
            raise OSError("injected terminal journal failure")
        real_fsync(fd)

    def disconnect(self) -> None:
        real_disconnect(self)
        raise RuntimeError("injected torque-off cleanup failure")

    monkeypatch.setattr(evaluator.os, "fsync", fsync)
    monkeypatch.setattr(evaluator.PolicyRunner, "disconnect_owned", disconnect)
    run = EvalRun(_cfg(path, 1), score_fn=lambda episode: True, save=True)

    with pytest.raises(RuntimeError, match="torque-off cleanup failure") as raised:
        run.run()

    assert not isinstance(raised.value, EvalJournalError)
    assert run.report is not None
    assert "policy teardown failed: injected torque-off cleanup failure" in run.report.error
    assert "terminal journal failure" in run.report.error
    assert run.report_path is not None
    saved = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "failed"
    assert "torque-off cleanup failure" in saved["error"]
    assert "terminal journal failure" in saved["error"]


def test_journal_write_error_is_not_masked_when_close_also_fails(tmp_path):
    class BrokenStream:
        def write(self, text: str) -> None:
            raise OSError("primary write failure")

        def close(self) -> None:
            raise OSError("secondary close failure")

    journal = evaluator.EvalJournal(tmp_path / "eval.jsonl")
    journal._stream = BrokenStream()  # type: ignore[assignment]

    with pytest.raises(EvalJournalError, match="primary write failure") as raised:
        journal.write("trial", episode=1)

    assert "secondary close failure" in str(raised.value)
    assert isinstance(raised.value.__cause__, OSError)
    assert "primary write failure" in str(raised.value.__cause__)


# -- EvalRun ---------------------------------------------------------------------------


def test_a_complete_run_saves_a_self_describing_report_and_journal():
    path = _mock_checkpoint(fps=15)
    run = EvalRun(_cfg(path, 3), score_fn=lambda ep: ep != 1, save=True)
    report = run.run()

    assert run.report_path is not None and run.journal_path is not None
    saved = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "done"
    assert (saved["successes"], saved["n"], saved["requested_episodes"]) == (2, 3, 3)
    assert saved["success_rate"] == pytest.approx(2 / 3)
    assert saved["ci95"] == [round(v, 4) for v in wilson_interval(2, 3)]
    assert saved["source"] == "mock"
    assert saved["fps"] == 15
    assert saved["label"] == (
        "deterministic MockPolicy stand-in (checkpoint weights were not loaded)"
    )
    assert [trial["episode"] for trial in saved["per_episode"]] == [1, 2, 3]
    assert all(trial["steps"] > 0 for trial in saved["per_episode"])
    assert report.successes == 2
    assert run.report_path.stem == run.journal_path.stem

    events = _journal(run.journal_path)
    assert [event["event"] for event in events] == ["start", "trial", "trial", "trial", "end"]
    assert events[0]["requested_episodes"] == 3
    assert events[-1]["report"] == str(run.report_path)


def test_evaluate_keeps_returning_an_unsaved_report():
    path = _mock_checkpoint()
    report = evaluator.evaluate(_cfg(path, 2))
    assert (report.successes, report.n, report.phase) == (2, 2, "done")
    assert _eval_files(".json") == [] and _eval_files(".jsonl") == []


def test_stopping_at_the_reset_gate_saves_a_partial_report():
    path = _mock_checkpoint()

    def gate(ep: int) -> None:
        if ep == 2:
            raise EvalAborted("stopped before rollout 3")

    run = EvalRun(_cfg(path, 4), score_fn=lambda ep: True, on_reset=gate, save=True)
    with pytest.raises(EvalAborted):
        run.run()

    assert run.report_path is not None
    saved = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "aborted"
    assert saved["error"] == "stopped before rollout 3"
    assert (saved["n_episodes"], saved["requested_episodes"], saved["n"]) == (2, 4, 2)
    end = _journal(run.journal_path)[-1]
    assert (end["event"], end["phase"], end["report"]) == ("end", "aborted", str(run.report_path))


def test_ctrl_c_during_a_rollout_records_the_trial_as_interrupted(monkeypatch):
    path = _mock_checkpoint()
    real_rollout = evaluator.PolicyRunner.rollout
    calls = {"n": 0}

    def rollout(self, duration=30.0):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_rollout(self, duration)

    monkeypatch.setattr(evaluator.PolicyRunner, "rollout", rollout)
    run = EvalRun(_cfg(path, 3), score_fn=lambda ep: True, save=True)
    with pytest.raises(KeyboardInterrupt):
        run.run()

    assert run.report is not None
    report = run.report
    assert report.phase == "aborted"
    assert (report.n_episodes, report.scored, report.interrupted, report.n) == (2, 1, 1, 1)
    assert report.success_rate == 1.0  # the interrupted trial is not a failure
    assert report.per_episode[1]["success"] is None
    assert report.per_episode[1]["interrupted"] == "KeyboardInterrupt"
    assert run.report_path is not None and run.report_path.is_file()
    trials = [e for e in _journal(run.journal_path) if e["event"] == "trial"]
    assert [trial["success"] for trial in trials] == [True, None]


def test_a_run_that_never_started_leaves_no_files(monkeypatch):
    path = _mock_checkpoint()

    def refuse(self):
        raise OSError("follower port busy")

    monkeypatch.setattr(evaluator.PolicyRunner, "connect", refuse)
    run = EvalRun(_cfg(path, 2), save=True)
    with pytest.raises(OSError, match="port busy"):
        run.run()
    assert run.report is not None and run.report.phase == "failed"
    assert run.report_path is None and run.journal_path is None
    assert _eval_files(".json") == [] and _eval_files(".jsonl") == []


def test_a_failing_report_save_never_masks_the_run_error(monkeypatch):
    path = _mock_checkpoint()

    def disk_full(self, path=None):
        raise OSError("no space left on device")

    monkeypatch.setattr(EvalReport, "save", disk_full)

    def gate(ep: int) -> None:
        if ep == 1:
            raise EvalAborted("stopped before rollout 2")

    run = EvalRun(_cfg(path, 2), on_reset=gate, save=True)
    with pytest.raises(EvalAborted):
        run.run()
    assert run.report_path is None
    assert _journal(run.journal_path)[-1]["report"] is None


# -- CLI: reset gate, journal, partial reports -----------------------------------------


def test_cli_eval_gates_every_rollout_after_the_first():
    path = _mock_checkpoint()
    # score 1: y; reset gate before 2: y; score 2: n
    result, text = _invoke(
        "eval", str(path), "--episodes", "2", "--max-seconds", "0.02", input="y\ny\nn\n"
    )
    assert result.exit_code == 0, text
    assert "Reset the workspace, then start rollout 2/2?" in text
    assert "1/2 = 50%" in text
    assert "not a real-robot result" in text
    (report_path,) = _eval_files(".json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "done"
    assert [trial["success"] for trial in saved["per_episode"]] == [True, False]
    assert str(report_path) in result.output.replace("\n", "")
    assert len(_eval_files(".jsonl")) == 1


def test_cli_eval_declining_the_reset_saves_a_partial_report():
    path = _mock_checkpoint()
    result, text = _invoke(
        "eval", str(path), "--episodes", "3", "--max-seconds", "0.02", input="y\nn\n"
    )
    assert result.exit_code == 1, text
    assert "stopped before rollout 2" in text
    assert "partial result after 1 of 3 rollouts" in text
    (report_path,) = _eval_files(".json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert (saved["phase"], saved["n_episodes"], saved["requested_episodes"]) == ("aborted", 1, 3)


def test_cli_eval_end_of_input_at_the_score_prompt_keeps_the_unscored_trial():
    path = _mock_checkpoint()
    result, text = _invoke("eval", str(path), "--episodes", "2", "--max-seconds", "0.02")
    assert result.exit_code == 1, text
    assert "0 scored trials" in text
    (report_path,) = _eval_files(".json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert (saved["n_episodes"], saved["interrupted"], saved["n"]) == (1, 1, 0)
    assert saved["phase"] == "aborted"


def test_cli_eval_ctrl_c_mid_rollout_exits_130_with_a_partial_report(monkeypatch):
    path = _mock_checkpoint()

    def interrupted(self, duration=30.0):
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluator.PolicyRunner, "rollout", interrupted)
    result, text = _invoke("eval", str(path), "--episodes", "2", "--max-seconds", "0.02")
    assert result.exit_code == 130, text
    assert "evaluation interrupted" in text
    (report_path,) = _eval_files(".json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert (saved["phase"], saved["interrupted"]) == ("aborted", 1)

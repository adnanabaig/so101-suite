"""Cancellation must stop both mock jobs and blocked real subprocesses."""

from __future__ import annotations

import threading

import pytest

from so101.training.jobs import TrainingJob
from so101.training.trainer import _run_mock, run_training


def test_mock_training_honors_cancel_before_checkpoint(tmp_path):
    cancel = threading.Event()
    job = TrainingJob(
        repo_id="test/cancel",
        policy="act",
        steps=200,
        batch_size=1,
        output_dir=str(tmp_path / "cancelled"),
    )
    stream = _run_mock(job, fast=False, cancel_event=cancel)
    next(stream)
    cancel.set()
    assert list(stream) == []
    assert not (tmp_path / "cancelled" / "checkpoints").exists()


def test_real_mode_never_creates_synthetic_checkpoint(tmp_path, monkeypatch):
    from so101.config import get_settings

    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    # No LeRobot trainer importable, whatever is on PATH.
    monkeypatch.setattr("so101.training.trainer.TRAIN_MODULE", "so101_no_such_trainer")
    job = TrainingJob(
        repo_id="test/real",
        policy="act",
        steps=1,
        batch_size=1,
        output_dir=str(tmp_path / "real"),
    )
    with pytest.raises(RuntimeError, match="SO101_MOCK=0"):
        list(run_training(job))
    assert not (tmp_path / "real" / "checkpoints").exists()
    get_settings.cache_clear()


def test_training_jobs_never_share_output_directory():
    from so101.services.jobs import JobRunner

    runner = JobRunner()
    first = runner.submit("test/data", "act", 1, 1)
    second = runner.submit("test/data", "act", 1, 1)
    runner.shutdown()
    assert first.output_dir != second.output_dir

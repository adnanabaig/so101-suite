"""scripts/bc_baseline_measure.py: the measurement table and the toy dataset writer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from so101.data.loader import load_training_data
from so101.training.toy_reach import ENV_STATE_NAMES, generate_toy_reach_episodes

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bc_baseline_measure.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bc_baseline_measure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_quick_run_measures_every_variant_against_the_bounds(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert script.main(["--quick"]) == 0
    out = capsys.readouterr().out
    rows = {line[:30].strip(): line for line in out.splitlines()}
    expert = rows["scripted expert (upper bound)"].split()
    assert expert[-4:] == ["10", "100.0%", "[72.2%,", "100.0%]"]
    assert rows["hold (lower bound)"].split()[-4:] == ["0", "0.0%", "[0.0%,", "27.8%]"]
    for variant in script.VARIANTS:
        assert variant.name in rows, variant.name
    # The mixed dataset's kept selection trains on its successful demonstrations only.
    mixed = generate_toy_reach_episodes(12, seed=0, failure_fraction=0.3)
    succeeded = sum(episode.success for episode in mixed)
    assert 0 < succeeded < 12
    assert rows["privileged, mixed data, kept"].split()[-7] == str(succeeded)
    assert rows["privileged, mixed data, all"].split()[-7] == "12"
    assert "not simulator or real-arm success" in out


def test_writes_a_dataset_the_trainer_reads(
    script: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--write-dataset", "toy/reach_demo", "--episodes", "8", "--failure-fraction", "0.5"]
    assert script.main(args) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "so101 train toy/reach_demo --policy bc_baseline" in out
    everything = load_training_data("toy/reach_demo", selection="all")
    kept = load_training_data("toy/reach_demo")
    assert len(everything.episodes) == 8
    assert tuple(everything.env_state_names) == ENV_STATE_NAMES
    failures = sum(episode.outcome == "failure" for episode in everything.episodes)
    assert failures > 0 and len(kept.episodes) == 8 - failures
    assert f"({failures} labeled failure)" in out

    assert script.main(args) == 1
    assert "already exists" in capsys.readouterr().err

    assert script.main(["--write-dataset", "toy/blind", "--episodes", "3", "--no-env-state"]) == 0
    blind = load_training_data("toy/blind", selection="all")
    assert blind.env_state_names == [] and blind.episodes[0].env_state is None


@pytest.mark.parametrize(
    "argv",
    [["--variants", "9"], ["--seeds", "a"], ["--hidden", "0"], ["--failure-fraction", "1"]],
)
def test_rejects_bad_arguments(script: ModuleType, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        script.main(argv)
    assert exit_info.value.code == 2

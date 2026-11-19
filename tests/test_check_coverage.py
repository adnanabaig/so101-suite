"""scripts/check_coverage.py and the coverage-floors.toml it enforces."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOORS = REPO_ROOT / "coverage-floors.toml"


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_coverage.py"
    spec = importlib.util.spec_from_file_location("check_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module


cc = _load_script()


def _report(files: dict[str, Any], branch: bool = True) -> dict[str, Any]:
    return {"meta": {"format": 3, "branch_coverage": branch}, "files": files}


def _file(percent: float, functions: dict[str, float] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"summary": {"percent_covered": percent}}
    entry["functions"] = {
        name: {"summary": {"percent_covered": value}} for name, value in (functions or {}).items()
    }
    return entry


def _run(tmp_path: Path, floors: str, report: dict[str, Any]) -> int:
    floors_path = tmp_path / "floors.toml"
    floors_path.write_text(floors, encoding="utf-8")
    report_path = tmp_path / "coverage.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return int(cc.main([str(report_path), "--floors", str(floors_path)]))


FLOORS_TOML = """
[settings]
branch = true

[files]
"src/so101/a.py" = 80.0

[functions]
"src/so101/a.py::Thing.method" = 50.0
"""


def test_passes_when_every_floor_holds(tmp_path, capsys):
    report = _report({"src/so101/a.py": _file(80.0, {"Thing.method": 75.0})})
    assert _run(tmp_path, FLOORS_TOML, report) == 0
    out = capsys.readouterr().out
    assert "All 2 coverage floors hold." in out
    assert "consider ratcheting up" in out  # 75 is well above the 50 floor


def test_fails_when_a_file_drops_below_its_floor(tmp_path, capsys):
    report = _report({"src/so101/a.py": _file(79.99, {"Thing.method": 75.0})})
    assert _run(tmp_path, FLOORS_TOML, report) == 1
    err = capsys.readouterr().err
    assert "src/so101/a.py is 79.99%, below its 80.00% floor" in err
    assert "do not lower a floor" in err


def test_fails_when_a_function_drops_below_its_floor(tmp_path, capsys):
    report = _report({"src/so101/a.py": _file(90.0, {"Thing.method": 49.0})})
    assert _run(tmp_path, FLOORS_TOML, report) == 1
    assert "Thing.method is 49.00%" in capsys.readouterr().err


def test_a_target_missing_from_the_report_fails_instead_of_skipping(tmp_path, capsys):
    report = _report({"src/so101/renamed.py": _file(100.0)})
    assert _run(tmp_path, FLOORS_TOML, report) == 1
    captured = capsys.readouterr()
    assert "MISSING from report" in captured.out
    assert "src/so101/a.py is not in the coverage report" in captured.err


def test_a_function_missing_from_the_report_fails(tmp_path, capsys):
    report = _report({"src/so101/a.py": _file(90.0, {"Thing.renamed": 100.0})})
    assert _run(tmp_path, FLOORS_TOML, report) == 1
    assert "Thing.method is not in the coverage report" in capsys.readouterr().err


@pytest.mark.parametrize(
    "spelling",
    [
        "/home/runner/work/so101/so101/src/so101/a.py",
        "/tmp/venv/lib/python3.12/site-packages/so101/a.py",
        "so101/a.py",
    ],
)
def test_report_paths_are_matched_by_package_path(tmp_path, spelling):
    report = _report({spelling: _file(85.0, {"Thing.method": 60.0})})
    assert _run(tmp_path, FLOORS_TOML, report) == 0


def test_floors_reject_a_report_measured_without_branch_coverage(tmp_path, capsys):
    report = _report({"src/so101/a.py": _file(99.0, {"Thing.method": 99.0})}, branch=False)
    assert _run(tmp_path, FLOORS_TOML, report) == 2
    assert "branch coverage" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("floors", "message"),
    [
        ('[files]\n"src/so101/a.py" = "high"\n', "must be a number"),
        ('[files]\n"src/so101/a.py" = 101\n', "within 0..100"),
        ('[functions]\n"src/so101/a.py" = 50\n', "expected '<file>::<qualified name>'"),
        ("[settings]\nbranch = true\n", "no floors defined"),
        ('[file]\n"src/so101/a.py" = 50\n', "unknown section"),
        ("[files\n", "invalid TOML"),
    ],
)
def test_malformed_floors_are_a_usage_error(tmp_path, capsys, floors, message):
    report = _report({"src/so101/a.py": _file(100.0, {"Thing.method": 100.0})})
    assert _run(tmp_path, floors, report) == 2
    assert message in capsys.readouterr().err


def test_a_missing_report_is_a_usage_error(tmp_path, capsys):
    floors_path = tmp_path / "floors.toml"
    floors_path.write_text(FLOORS_TOML, encoding="utf-8")
    code = cc.main([str(tmp_path / "absent.json"), "--floors", str(floors_path)])
    assert code == 2
    assert "make coverage" in capsys.readouterr().err


def _qualified_names(source: Path) -> set[str]:
    names: set[str] = set()

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                names.add(name)
                visit(child, f"{name}.")

    visit(ast.parse(source.read_text(encoding="utf-8")), "")
    return names


def test_repo_floors_cover_the_safety_critical_targets():
    _settings, files, functions = cc.load_floors(FLOORS)
    assert _settings.get("branch") is True
    for required in (
        "src/so101/hardware/safety.py",
        "src/so101/services/robot_manager.py",
        "src/so101/services/routers/control.py",
    ):
        assert required in files
    lease = [name for name in functions if "lease" in name.split("::", 1)[1].lower()]
    assert len(lease) >= 4, "the lease paths need function-level floors"
    critical_stops = {
        "src/so101/services/robot_manager.py::RobotManager._fail_closed_runtime_io",
        "src/so101/services/robot_manager.py::RobotManager._latch_fault_in_memory",
        "src/so101/services/robot_manager.py::RobotManager.request_stop",
        "src/so101/services/robot_manager.py::RobotManager.latch_estop",
        "src/so101/services/robot_manager.py::RobotManager.reset_estop",
        "src/so101/services/robot_manager.py::RobotManager.clear_fault",
        "src/so101/services/robot_manager.py::RobotManager._estop_category0",
        "src/so101/services/robot_manager.py::RobotManager._dispatch_stop",
        "src/so101/services/robot_manager.py::RobotManager._external_stop_in_progress_locked",
        "src/so101/services/robot_manager.py::RobotManager._any_stop_in_progress_locked",
        "src/so101/services/robot_manager.py::RobotManager._stop_transition_is_current_locked",
        "src/so101/services/robot_manager.py::RobotManager._arm_cleanup_requires_torque_cut_locked",
        "src/so101/services/robot_manager.py::RobotManager._revoke_authority_locked",
        "src/so101/services/robot_manager.py::RobotManager._cut_torque",
        "src/so101/services/robot_manager.py::RobotManager._cut_torque_pass",
        "src/so101/services/robot_manager.py::RobotManager._retry_torque_off",
        "src/so101/services/robot_manager.py::RobotManager._halt_stop",
        "src/so101/services/robot_manager.py::RobotManager._hold_in_place",
        "src/so101/services/robot_manager.py::RobotManager._halt_fallback",
        "src/so101/services/robot_manager.py::RobotManager._collapse_stop",
        "src/so101/services/robot_manager.py::RobotManager._plan_collapse",
        "src/so101/services/robot_manager.py::RobotManager._run_descent",
        "src/so101/services/robot_manager.py::RobotManager._collapse_fallback",
        "src/so101/services/robot_manager.py::RobotManager._collapse_done",
        "src/so101/services/robot_manager.py::RobotManager._supervise_stops",
        "src/so101/services/robot_manager.py::RobotManager._sample_hold",
        "src/so101/services/robot_manager.py::RobotManager._police_collapse",
        "src/so101/services/robot_manager.py::RobotManager._freeze_collapse",
        "src/so101/services/robot_manager.py::RobotManager._claim_collapse_end",
        "src/so101/services/robot_manager.py::RobotManager._await_collapse",
        "src/so101/services/robot_manager.py::RobotManager._on_health_critical",
        "src/so101/services/robot_manager.py::RobotManager.servo_health_critical",
        "src/so101/services/robot_manager.py::RobotManager._mark_torque_unconfirmed",
        "src/so101/services/robot_manager.py::RobotManager._reopen_stale",
        "src/so101/services/robot_manager.py::RobotManager._retry_cut",
        "src/so101/services/robot_manager.py::RobotManager._release_ports",
        "src/so101/services/robot_manager.py::RobotManager._add_recovery_devices_locked",
        "src/so101/services/robot_manager.py::RobotManager._prepare_global_energized_intent",
        "src/so101/services/robot_manager.py::RobotManager._write_global_recovery_intent",
        "src/so101/services/robot_manager.py::RobotManager._clear_global_energized_recovery",
        "src/so101/services/robot_manager.py::RobotManager.publish_restored_recovery_intent",
        "src/so101/services/robot_manager.py::RobotManager._retire_confirmed_recovery_devices",
        "src/so101/services/robot_manager.py::RobotManager.shutdown",
        "src/so101/services/robot_manager.py::RobotManager._stop_loop",
    }
    assert critical_stops <= functions.keys()
    initial_unmeasured = {
        "src/so101/services/robot_manager.py::RobotManager._fail_closed_runtime_io",
        "src/so101/services/robot_manager.py::RobotManager._latch_fault_in_memory",
        "src/so101/services/robot_manager.py::RobotManager.latch_estop",
        "src/so101/services/robot_manager.py::RobotManager.reset_estop",
        "src/so101/services/robot_manager.py::RobotManager.clear_fault",
        "src/so101/services/robot_manager.py::RobotManager._estop_category0",
        "src/so101/services/robot_manager.py::RobotManager._dispatch_stop",
        "src/so101/services/robot_manager.py::RobotManager._cut_torque",
        "src/so101/services/robot_manager.py::RobotManager._cut_torque_pass",
        "src/so101/services/robot_manager.py::RobotManager._retry_torque_off",
        "src/so101/services/robot_manager.py::RobotManager._supervise_stops",
        "src/so101/services/robot_manager.py::RobotManager._external_stop_in_progress_locked",
        "src/so101/services/robot_manager.py::RobotManager._any_stop_in_progress_locked",
        "src/so101/services/robot_manager.py::RobotManager._stop_transition_is_current_locked",
        "src/so101/services/robot_manager.py::RobotManager._arm_cleanup_requires_torque_cut_locked",
        "src/so101/services/robot_manager.py::RobotManager._revoke_authority_locked",
        "src/so101/services/robot_manager.py::RobotManager._sample_hold",
        "src/so101/services/robot_manager.py::RobotManager._police_collapse",
        "src/so101/services/robot_manager.py::RobotManager._freeze_collapse",
        "src/so101/services/robot_manager.py::RobotManager._claim_collapse_end",
        "src/so101/services/robot_manager.py::RobotManager._plan_collapse",
        "src/so101/services/robot_manager.py::RobotManager._await_collapse",
        "src/so101/services/robot_manager.py::RobotManager._mark_torque_unconfirmed",
        "src/so101/services/robot_manager.py::RobotManager._reopen_stale",
        "src/so101/services/robot_manager.py::RobotManager._retry_cut",
        "src/so101/services/robot_manager.py::RobotManager._release_ports",
        "src/so101/services/robot_manager.py::RobotManager._add_recovery_devices_locked",
        "src/so101/services/robot_manager.py::RobotManager._prepare_global_energized_intent",
        "src/so101/services/robot_manager.py::RobotManager._write_global_recovery_intent",
        "src/so101/services/robot_manager.py::RobotManager._clear_global_energized_recovery",
        "src/so101/services/robot_manager.py::RobotManager.publish_restored_recovery_intent",
        "src/so101/services/robot_manager.py::RobotManager._retire_confirmed_recovery_devices",
        "src/so101/services/robot_manager.py::RobotManager.shutdown",
        "src/so101/services/robot_manager.py::RobotManager._stop_loop",
    }
    assert all(functions[target] >= 80.0 for target in initial_unmeasured)


def test_every_repo_floor_names_code_that_exists():
    """Typos and renames fail here in a plain test run, not only under coverage."""
    _settings, files, functions = cc.load_floors(FLOORS)
    for target in files:
        assert (REPO_ROOT / target).is_file(), f"{target} does not exist"
    for target in functions:
        file_name, qualname = target.split("::", 1)
        assert (REPO_ROOT / file_name).is_file(), f"{file_name} does not exist"
        assert qualname in _qualified_names(REPO_ROOT / file_name), (
            f"{qualname} is not defined in {file_name}"
        )

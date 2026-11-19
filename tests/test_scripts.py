"""Developer scripts under scripts/ (run as subprocesses, the way make runs them)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _run_script(name: str, cwd: Path, extra_env: dict[str, str] | None = None) -> str:
    """Run scripts/<name> from ``cwd`` and return so101's import location + output."""
    script = SCRIPTS / name
    env = {**os.environ, **(extra_env or {})}
    # Without PYTHONPATH, a cwd-relative sys.path hack would import some other
    # so101 (or none) when the script runs outside the repo root.
    env.pop("PYTHONPATH", None)
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(script)!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "import so101\n"
        "print('SO101_IMPORTED_FROM=' + so101.__file__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_setup_motors_prints_both_arms_lerobot_commands_from_any_directory(tmp_path):
    out = _run_script(
        "setup_motors.py",
        cwd=tmp_path,
        extra_env={
            "SO101_FOLLOWER_PORT": "/dev/ttyACM-follower",
            "SO101_LEADER_PORT": "/dev/ttyACM-leader",
            "SO101_FOLLOWER_ID": "arm-f",
            "SO101_LEADER_ID": "arm-l",
        },
    )
    # Flags as LeRobot 0.6.1's lerobot-setup-motors defines them: --robot.* for
    # the so101_follower robot type, --teleop.* for the so101_leader teleoperator.
    assert (
        "lerobot-setup-motors --robot.type=so101_follower "
        "--robot.port=/dev/ttyACM-follower --robot.id=arm-f"
    ) in out
    assert (
        "lerobot-setup-motors --teleop.type=so101_leader "
        "--teleop.port=/dev/ttyACM-leader --teleop.id=arm-l"
    ) in out
    # It prints instructions; it must not claim to have assigned anything.
    assert "does not talk to the motors" in out
    imported_from = Path(out.rsplit("SO101_IMPORTED_FROM=", 1)[1].strip()).resolve()
    assert imported_from.is_relative_to(REPO_ROOT / "src")


def test_setup_motors_does_not_create_data_directories(tmp_path):
    _run_script("setup_motors.py", cwd=tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_find_cameras_reports_from_any_directory_without_creating_files(tmp_path):
    out = _run_script(
        "find_cameras.py",
        cwd=tmp_path,
        extra_env={"SO101_MOCK": "0", "SO101_BACKEND": "real"},
    )
    # Whatever this host has installed, both sections print and nothing raises.
    assert "Intel RealSense" in out and "OpenCV indices 0..9:" in out
    assert "Traceback" not in out
    imported_from = Path(out.rsplit("SO101_IMPORTED_FROM=", 1)[1].strip()).resolve()
    assert imported_from.is_relative_to(REPO_ROOT / "src")
    assert sorted(path.name for path in tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_find_cameras_script_refuses_non_real_backends(tmp_path, backend):
    script = SCRIPTS / "find_cameras.py"
    env = {
        **os.environ,
        "SO101_MOCK": "0",
        "SO101_BACKEND": backend,
        "SO101_HOME": str(tmp_path / "home"),
    }
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 3
    assert proc.stdout == ""
    assert f"{backend} backend" in proc.stderr
    assert "probes physical USB/video devices" in proc.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == []

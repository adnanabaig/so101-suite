"""The camera launcher must not open another OBS or another setup dialog."""

import os
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "obs-arms"


@pytest.mark.parametrize("already_running", [True, False])
def test_obs_launcher_avoids_duplicates_and_selects_profile(tmp_path, already_running):
    tools = tmp_path / "bin"
    tools.mkdir()
    marker = tmp_path / "flatpak-called"
    commands = {
        "systemctl": "exit 0",
        "pgrep": f"exit {0 if already_running else 1}",
        "flatpak": 'printf "%s\\n" "$@" > "$OBS_TEST_MARKER"\n'
        'printf "%s\\n" "$XDG_DATA_DIRS" >> "$OBS_TEST_MARKER"',
    }
    for name, body in commands.items():
        path = tools / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = os.environ | {
        "PATH": f"{tools}:/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(tmp_path),
        "OBS_TEST_MARKER": str(marker),
        "XDG_DATA_DIRS": "/usr/share",
    }
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if already_running:
        assert "already running" in result.stdout
        assert not marker.exists()
    else:
        lines = marker.read_text().splitlines()
        assert lines[:6] == [
            "run",
            "com.obsproject.Studio",
            "--collection",
            "SO101 Arms",
            "--profile",
            "SO101 Arms",
        ]
        assert "/var/lib/flatpak/exports/share" in lines[-1]

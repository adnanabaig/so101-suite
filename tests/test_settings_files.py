"""Where settings and data come from (so101.config), from any working folder.

The rule: ``SO101_*`` environment variables, then ``./.env``, then
``<data folder>/.env`` (SO101_HOME, else the per-user folder), then defaults.
The installer writes absolute storage folders into the data folder's .env, so
the installed suite (the G15) finds the same settings and data whatever folder
``so101`` runs in, while a source checkout's own ``.env`` and ``./data`` still
work.

Every test here sets HOME and SO101_HOME to temp folders (conftest sets
SO101_HOME for every test), so nothing reads the developer's real data folder.
Env files are off for the rest of the suite; these tests turn the rule back on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from so101 import config, platform_info
from so101.cli import app as cli_app


@pytest.fixture
def rule(tmp_path, monkeypatch):
    """The real env-file rule, with a fake home and data folder, and no storage env vars."""

    monkeypatch.setitem(config.Settings.model_config, "env_file", config.AUTO_ENV_FILES)
    for key in ("SO101_DATA_DIR", "SO101_DATASETS_DIR", "SO101_OUTPUTS_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    suite = tmp_path / "home" / ".local" / "share" / "so101"
    monkeypatch.setenv("SO101_HOME", str(suite))
    config.get_settings.cache_clear()
    yield suite
    config.get_settings.cache_clear()


def _write(path: Path, *lines: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_installed_suite_reads_its_data_folder_from_any_folder(rule, tmp_path, monkeypatch):
    """G15: install.sh wrote ~/.local/share/so101/.env; doctor runs from ~/Downloads."""

    _write(
        rule / ".env",
        *(f"{key}={value}" for key, value in platform_info.storage_settings(rule).items()),
        "SO101_FOLLOWER_PORT=/dev/so101-follower",
        "SO101_FOLLOWER_ID=g15-follower",
    )
    elsewhere = tmp_path / "home" / "Downloads"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)

    settings = config.Settings()
    assert settings.follower_port == "/dev/so101-follower"
    assert settings.follower_id == "g15-follower"
    assert settings.data_dir == rule / "data"
    assert settings.outputs_dir == rule / "outputs"
    assert config.env_files() == (rule / ".env", elsewhere / ".env")
    sources = config.settings_sources(settings)
    assert sources["warning"] is None and sources["storage_from_cwd"] == []
    assert [entry["role"] for entry in sources["env_files"]] == ["data folder", "current folder"]
    assert [entry["exists"] for entry in sources["env_files"]] == [True, False]


def test_a_checkout_keeps_its_own_env_and_data(rule, tmp_path, monkeypatch):
    """Dev checkout, no installed suite: ./.env and ./data, as before."""

    checkout = tmp_path / "so101"
    _write(checkout / ".env", "SO101_FOLLOWER_ID=bench-arm", "SO101_MOCK=1")
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("SO101_MOCK")

    settings = config.Settings()
    assert settings.follower_id == "bench-arm" and settings.mock is True
    assert settings.data_dir == checkout / "data"
    assert settings.datasets_dir == checkout / "datasets"
    assert config.settings_sources(settings)["warning"] is None  # no installed suite


def test_the_current_folder_overrides_the_data_folder_and_the_environment_both(
    rule, tmp_path, monkeypatch
):
    _write(rule / ".env", "SO101_FOLLOWER_ID=installed", "SO101_LEADER_ID=installed")
    checkout = tmp_path / "checkout"
    _write(checkout / ".env", "SO101_FOLLOWER_ID=checkout", "SO101_BAUDRATE=1000000")
    monkeypatch.chdir(checkout)

    settings = config.Settings()
    assert settings.follower_id == "checkout"  # ./.env beats the data folder's
    assert settings.leader_id == "installed"  # what ./.env doesn't set still applies
    monkeypatch.setenv("SO101_FOLLOWER_ID", "shell")
    assert config.Settings().follower_id == "shell"  # the environment beats both
    assert config.Settings(_env_file=None).leader_id == "so101-leader"  # explicit wins


def test_running_from_the_data_folder_reads_its_env_once(rule, monkeypatch):
    _write(rule / ".env", "SO101_FOLLOWER_ID=installed")
    monkeypatch.chdir(rule)
    assert config.env_files() == (rule / ".env",)
    assert config.Settings().follower_id == "installed"


def test_doctor_warns_when_the_data_folders_env_leaves_storage_to_the_working_folder(
    rule, tmp_path, monkeypatch
):
    _write(rule / ".env", "SO101_FOLLOWER_ID=installed")  # an older .env: no storage folders
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    sources = config.settings_sources(config.Settings())
    assert sources["storage_from_cwd"] == ["data_dir", "datasets_dir", "outputs_dir"]
    assert sources["warning"] is not None
    assert "SO101_DATA_DIR, SO101_DATASETS_DIR, SO101_OUTPUTS_DIR" in sources["warning"]
    assert "re-running install/install.sh adds them" in sources["warning"]
    assert str(work) in sources["warning"]

    monkeypatch.chdir(rule)  # from the data folder the defaults land in it: fine
    assert config.settings_sources(config.Settings())["warning"] is None
    monkeypatch.chdir(work)
    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("SO101_DATASETS_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("SO101_OUTPUTS_DIR", str(tmp_path / "o"))
    assert config.settings_sources(config.Settings())["warning"] is None


def test_env_files_reports_what_settings_reads(monkeypatch, tmp_path):
    # The test suite turns env files off (conftest), so tests never read a real .env.
    assert config.env_files() == ()
    assert config.settings_sources(config.get_settings())["env_files"] == []
    monkeypatch.setitem(config.Settings.model_config, "env_file", str(tmp_path / "a.env"))
    assert config.env_files() == (tmp_path / "a.env",)
    monkeypatch.setitem(config.Settings.model_config, "env_file", ["x.env", tmp_path / "y.env"])
    assert config.env_files() == (Path("x.env"), tmp_path / "y.env")
    assert [e["role"] for e in config.settings_sources(config.Settings())["env_files"]] == [
        "configured",
        "configured",
    ]


def test_tests_never_see_the_real_data_folder(tmp_path):
    """conftest points SO101_HOME at this test's temp folder."""

    assert os.environ["SO101_HOME"] == str(tmp_path / "so101-home")
    assert platform_info.current_data_home() == tmp_path / "so101-home"
    assert platform_info.current_env_files()[0] == tmp_path / "so101-home" / ".env"


@pytest.mark.parametrize(
    ("platform", "follower", "leader"),
    [
        ("linux", "/dev/so101-follower", "/dev/so101-leader"),
        ("darwin", "/dev/tty.usbmodem585A0077531", "/dev/tty.usbmodem585A0077541"),
        ("win32", "/dev/tty.usbmodem585A0077531", "/dev/tty.usbmodem585A0077541"),
    ],
)
def test_arm_port_defaults_follow_the_os(monkeypatch, platform, follower, leader):
    assert config.default_arm_port("follower", platform) == follower
    assert config.default_arm_port("leader", platform) == leader
    monkeypatch.setattr(config.sys, "platform", platform)
    monkeypatch.delenv("SO101_FOLLOWER_PORT", raising=False)
    settings = config.Settings()
    assert (settings.follower_port, settings.leader_port) == (follower, leader)


def test_the_linux_defaults_are_the_names_the_installer_creates():
    from so101.bringup.serial_ports import LINUX_ARM_LINKS

    assert (
        platform_info.ARM_LINKS == LINUX_ARM_LINKS == ("/dev/so101-follower", "/dev/so101-leader")
    )
    install = (Path(__file__).resolve().parents[1] / "install" / "install.sh").read_text("utf-8")
    assert 'FOLLOWER_LINK="/dev/so101-follower"' in install
    assert 'LEADER_LINK="/dev/so101-leader"' in install


def test_so101_info_prints_the_settings_files_and_folders(rule, tmp_path, monkeypatch):
    _write(rule / ".env", "SO101_FOLLOWER_ID=installed")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("COLUMNS", "300")
    result = CliRunner().invoke(cli_app, ["info"])
    assert result.exit_code == 0, result.output
    assert f"settings (data folder)    │ {rule / '.env'} (read)" in result.output
    assert f"{work / '.env'} (not found)" in result.output
    assert f"data dir                  │ {work / 'data'}" in result.output
    assert "warning:" in result.output and "re-running install/install.sh" in result.output


def test_so101_info_says_when_env_files_are_off(monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    result = CliRunner().invoke(cli_app, ["info"])
    assert result.exit_code == 0, result.output
    assert "no .env files (env files are off)" in result.output
    assert "warning:" not in result.output


def test_doctor_reports_the_settings_files_and_data_folders(rule, tmp_path, monkeypatch):
    from so101.services.robot_manager import reset_manager

    _write(
        rule / ".env",
        *(f"{key}={value}" for key, value in platform_info.storage_settings(rule).items()),
    )
    monkeypatch.chdir(tmp_path)
    # A cheap platform report: no torch import, no MuJoCo render child.
    report = platform_info.collect(
        platform_info.Probes(torch=lambda: None, dist_version=lambda name: None)
    )
    monkeypatch.setattr(platform_info, "collect", lambda: report)
    reset_manager()
    try:
        result = CliRunner().invoke(cli_app, ["doctor", "--json"])
    finally:
        reset_manager()
    payload = json.loads(result.stdout)
    settings = payload["settings"]
    assert settings["data_home"] == str(rule)
    assert settings["data_dir"] == str(rule / "data")
    assert settings["env_files"][0] == {
        "path": str(rule / ".env"),
        "exists": True,
        "role": "data folder",
    }
    (check,) = [c for c in payload["checks"] if c["id"] == "settings"]
    assert check["status"] == "pass"
    assert f"{rule / '.env'} (read)" in check["detail"]
    assert f"Data {rule / 'data'}" in check["detail"]

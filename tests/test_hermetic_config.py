"""The suite must not depend on the developer's ``.env`` or shell (audit B5).

``Settings`` reads ``SO101_*`` variables, ``./.env`` and the data folder's
``.env``. The README tells users to ``cp .env.example .env`` and customize it,
so without isolation a local device id or planner provider silently changes
test results.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from so101 import config

# Values a real operator plausibly puts in .env or exports in their shell.
POLLUTION = {
    "SO101_MOCK": "0",
    "SO101_FOLLOWER_ID": "lab-arm-7",
    "SO101_AGENT_PROVIDER": "openai_responses",
    "SO101_AGENT_MODEL": "gpt-test",
    "SO101_OPENAI_API_KEY": "sk-test-not-real",
    "SO101_CORS_ORIGINS": "https://evil.example",
}


def _write_dotenv(directory: Path) -> None:
    lines = [f"{key}={value}" for key, value in POLLUTION.items()]
    (directory / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_defaults(settings: config.Settings) -> None:
    assert settings.mock is True
    assert settings.follower_id == "so101-follower"
    assert settings.agent_provider == "deterministic"
    assert settings.agent_model == ""
    assert settings.openai_api_key == ""
    assert "https://evil.example" not in settings.allowed_origins


def test_settings_inside_a_test_are_defaults():
    _assert_defaults(config.get_settings())


def test_dotenv_in_working_directory_is_ignored(tmp_path, monkeypatch):
    _write_dotenv(tmp_path)
    monkeypatch.chdir(tmp_path)
    config.get_settings.cache_clear()
    _assert_defaults(config.get_settings())


def test_polluted_shell_and_dotenv_do_not_leak_into_a_test_session(tmp_path):
    """Run a child pytest from a directory holding a customized .env, with the
    same values exported, and require the in-test settings to be defaults."""
    _write_dotenv(tmp_path)
    # An installed suite's data folder with the same values, found through
    # SO101_HOME, XDG_DATA_HOME and HOME alike.
    for folder in ("suite", "xdg/so101", "home/.local/share/so101"):
        (tmp_path / folder).mkdir(parents=True)
        _write_dotenv(tmp_path / folder)
    env = {**os.environ, **POLLUTION}
    env.update(
        SO101_HOME=str(tmp_path / "suite"),
        XDG_DATA_HOME=str(tmp_path / "xdg"),
        HOME=str(tmp_path / "home"),
    )
    env.pop("PYTEST_ADDOPTS", None)
    target = f"{Path(__file__).resolve()}::test_settings_inside_a_test_are_defaults"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", target],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_importing_the_app_cli_and_launcher_creates_nothing(tmp_path):
    """Importing a module must not build the app (audit devex-B8, B5).

    Building it reads the settings and creates data/, datasets/ and outputs/ in
    whatever folder pytest or a tool runs from. uvicorn builds it through the
    factory, so a plain import must leave the working directory, HOME and the
    data folder untouched.
    """
    import so101

    work = tmp_path / "work"
    work.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("SO101_")}
    env.pop("XDG_DATA_HOME", None)
    env["HOME"] = str(tmp_path / "home")
    env["SO101_HOME"] = str(tmp_path / "so101-home")
    source = str(Path(so101.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    code = "import so101.services.app, so101.cli, so101.launcher, so101.config"
    subprocess.run([sys.executable, "-c", code], cwd=work, env=env, check=True, timeout=120)
    assert list(work.iterdir()) == []
    assert not (tmp_path / "home").exists() and not (tmp_path / "so101-home").exists()


def test_the_module_level_app_is_built_on_first_access(monkeypatch):
    """``uvicorn so101.services.app:app`` keeps working: the attribute is lazy."""
    import importlib

    from fastapi import FastAPI

    from so101 import launcher

    module = importlib.import_module("so101.services.app")
    # setitem then delitem: undoing them removes whatever this test caches.
    monkeypatch.setitem(module.__dict__, "app", None)
    monkeypatch.delitem(module.__dict__, "app")
    built = module.app
    assert isinstance(built, FastAPI)
    assert module.app is built  # cached after the first access
    try:
        module.not_an_attribute  # noqa: B018
    except AttributeError as exc:
        assert "not_an_attribute" in str(exc)
    else:
        raise AssertionError("an unknown attribute must raise AttributeError")
    assert launcher.APP == module.APP_FACTORY

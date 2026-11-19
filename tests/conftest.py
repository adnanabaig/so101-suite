"""Shared fixtures. Hermetic config: mock hardware, temp dirs, no developer state.

``Settings`` reads ``SO101_*`` variables, ``<data folder>/.env`` and ``./.env``
(so101.config). Tests must not see the developer's shell exports or either
``.env`` (audit B5), so env files are off and ``SO101_HOME`` points into a temp
folder: nothing reads the real ``~/.local/share/so101`` (platform_info's
report, child processes and make recipes included). Importing the app at
collection time must not create ``data/`` etc. in the working directory.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# The skip budget (`--skip-budget`, passed by make test / make coverage).
from skip_budget import pytest_addoption, pytest_configure  # noqa: F401
from so101 import config


def _scrub_so101_environment() -> None:
    for key in [name for name in os.environ if name.startswith("SO101_")]:
        del os.environ[key]


# Runs at conftest import, before any test module is collected, so collection
# is hermetic as well (importing so101.services.app no longer builds the app,
# but a module-scoped fixture may).
config.Settings.model_config["env_file"] = None
_scrub_so101_environment()
_SESSION_ROOT = Path(tempfile.mkdtemp(prefix="so101-tests-"))
os.environ["SO101_MOCK"] = "1"
os.environ["SO101_DATA_DIR"] = str(_SESSION_ROOT / "data")
os.environ["SO101_DATASETS_DIR"] = str(_SESSION_ROOT / "datasets")
os.environ["SO101_OUTPUTS_DIR"] = str(_SESSION_ROOT / "outputs")
os.environ["SO101_HOME"] = str(_SESSION_ROOT / "so101-home")
# Module-scoped fixtures run before _isolated_dirs; give them the same Host baseline.
os.environ["SO101_ALLOWED_HOSTS"] = "testserver"


def pytest_unconfigure() -> None:
    shutil.rmtree(_SESSION_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    # Drop anything a previous test (or the shell) left behind, then set the
    # test baseline. monkeypatch restores the previous environment afterwards.
    for key in [name for name in os.environ if name.startswith("SO101_")]:
        monkeypatch.delenv(key)
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    monkeypatch.setenv("SO101_MOCK", "1")
    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SO101_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("SO101_OUTPUTS_DIR", str(tmp_path / "outputs"))
    # The data folder (where <data folder>/.env lives) is the test's own.
    monkeypatch.setenv("SO101_HOME", str(tmp_path / "so101-home"))
    monkeypatch.setenv("SO101_CAMERAS", "0")
    # C1 Host guard: Starlette's TestClient sends ``Host: testserver``.
    monkeypatch.setenv("SO101_ALLOWED_HOSTS", "testserver")
    monkeypatch.delenv("SO101_UI_DIR", raising=False)
    # Host-global energized intent is independent of SO101_DATA_DIR in
    # production. Give every test its own equivalent persistent namespace so
    # a deliberate retained-Halt marker cannot contaminate another test.
    from so101.bringup import ownership

    device_lock_root = tmp_path / "global-device-locks"
    device_lock_root.mkdir()
    monkeypatch.setattr(ownership, "_device_lock_root", lambda: device_lock_root)
    monkeypatch.setattr(ownership, "_validate_lock_namespace_root", lambda _root: None)
    global_marker_root = tmp_path / "global-energized-state"
    global_marker_root.mkdir()
    monkeypatch.setattr(ownership, "_global_marker_root", lambda: global_marker_root)
    monkeypatch.setattr(ownership, "_validate_global_marker_root", lambda _root: None)
    # LeRobot 0.6.1 refuses to import with LEROBOT_HOME set; FeetechBus checks it.
    monkeypatch.delenv("LEROBOT_HOME", raising=False)
    # A halted mock arm is energized, and every app shutdown collapses it to
    # the rest pose (design doc §5). At the real 15 deg/s and 30 Hz that
    # descent takes seconds of wall time per test. The SafetyGuard still caps
    # every tick (450 deg/s), so these only shorten the mock descent; tests of
    # the cap and the physics tests set their own speed and rate.
    monkeypatch.setenv("SO101_COLLAPSE_SPEED_DEG_S", "450")
    from so101.services.robot_manager import RobotManager

    monkeypatch.setattr(RobotManager, "collapse_rate_hz", 240.0)
    # Reset the cached settings so the temp dirs take effect.
    from so101.services.agent_executor import reset_executor as reset_agent_executor
    from so101.services.robot_manager import reset_manager
    from so101.services.routers.eval import reset_session as reset_eval_session
    from so101.services.store import reset_store

    config.get_settings.cache_clear()
    reset_agent_executor()
    reset_eval_session()
    reset_manager()
    reset_store()
    yield
    reset_agent_executor()
    reset_eval_session()
    reset_manager()
    reset_store()
    config.get_settings.cache_clear()


class ManagerClock:
    """Monotonic time for the robot manager that moves only when told to.

    The manager reads one injected clock for leases, operator heartbeats, the
    command deadman and the watchdog (devex-B4, docs/audit/v0.1/devex.md), so
    a test freezes time and advances it instead of sleeping, and does not
    depend on how busy the machine is.
    """

    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        # A control loop's pacing sleep moves fake time by exactly what it
        # asked for; the short real nap only yields the CPU.
        self.now += max(0.0, seconds)
        import time

        time.sleep(0.001)


@pytest.fixture
def manager_clock(monkeypatch) -> ManagerClock:
    """Freeze the robot manager's clock; ``advance`` moves it (devex-B4)."""

    import time

    from so101.services.robot_manager import get_manager

    manager = get_manager()
    clock = ManagerClock(time.monotonic())
    monkeypatch.setattr(manager, "_clock", clock)
    monkeypatch.setattr(manager, "_sleep", clock.sleep)
    return clock


@pytest.fixture
def supervise(monkeypatch):
    """One watchdog pass, run only when the test calls it (devex-B4).

    The watchdog thread keeps running but its passes do nothing, so every
    supervision decision comes from the injected clock at a point the test
    chose, however busy the machine is. The servo-health thread is held the
    same way (its polls read the bus and can stop the arm); a test that wants
    them drives them with ``health_poll``.
    """

    from so101.services.robot_manager import get_manager

    manager = get_manager()
    tick = manager._watchdog_tick
    monkeypatch.setattr(manager, "_watchdog_tick", lambda: None)
    monkeypatch.setattr(manager, "_health_tick", lambda: None)
    return tick


@pytest.fixture
def health_poll(monkeypatch):
    """One servo-health pass, run only when the test calls it.

    The ``so101-health`` thread keeps running but its passes do nothing; the
    returned callable runs the real pass (``RobotManager._health_tick``), which
    polls each arm whose period is due on the injected manager clock.
    """

    from functools import partial

    from so101.services.robot_manager import RobotManager, get_manager

    manager = get_manager()
    monkeypatch.setattr(manager, "_health_tick", lambda: None)
    return partial(RobotManager._health_tick, manager)

"""Start the SO-101 suite and open it in a browser: ``so101-suite``.

The installers' Start Menu, desktop and app-menu launchers run this
(``python -X utf8 -m so101.launcher --pause-on-error``). It:

1. makes the per-user data folder the working directory and ``SO101_HOME``
   before ``so101.config`` is imported, and creates ``<folder>/.env`` with the
   absolute data, datasets and outputs folders if it has none. Settings reads
   ``<data folder>/.env`` (then a ``./.env``), so settings and data stay in one
   place however the launcher was started (``so101/config.py``). The folder is
   ``--home``, else ``SO101_HOME``, else :func:`so101.platform_info.data_home`;
2. puts the venv's bin (Scripts) folder first on PATH, so console scripts such
   as ``lerobot-find-port`` are found without an activated venv (training runs
   ``python -m lerobot.scripts.lerobot_train`` and needs no PATH); sets
   ``PYTHONUTF8=1`` for child processes, drops a ``MUJOCO_GL`` value that
   would make ``import mujoco`` raise on this OS, and on Linux picks the offscreen
   renderer: egl when the NVIDIA driver is loaded, else one that works without a
   display;
3. opens the browser on a suite that is already running on the port, or else
4. runs the server in this process and opens the browser once ``/api/health``
   answers. Ctrl+C stops it, and so does closing the terminal: SIGHUP (the
   app-menu launcher on Linux runs in a terminal) takes the same graceful
   path as SIGINT and SIGTERM, so training jobs are cancelled and the robot
   manager collapses the arm before the process exits.
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import platform
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

from .platform_info import (
    ENV_FILE_NAME,
    NVIDIA_PROC,
    data_home,
    mujoco_gl_advice,
    os_name,
    storage_settings,
    update_env_file,
)

# uvicorn builds the app with this factory (so101.services.app.APP_FACTORY).
APP = "so101.services.app:create_app"
READY_TIMEOUT_S = 90.0
# uvicorn exits with this code when it cannot bind the port (uvicorn/server.py).
STARTUP_FAILURE = 3

# Talk to the local server directly: an HTTP(S)_PROXY setting must not route
# requests for 127.0.0.1 through a proxy.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

QUIT_WARNING = (
    "Before you quit, lower the arm to its rest pose: quitting can switch the motors "
    "off. The power switch always cuts them."
)


def resolve_home(
    arg: str | None,
    os_kind: str,
    env: MutableMapping[str, str],
    home: Path,
) -> Path:
    """The data folder as an absolute path (not created here)."""

    path = Path(arg).expanduser() if arg else Path(data_home(os_kind, env, home))
    return path.absolute()


def init_settings_file(home: Path) -> str | None:
    """Create ``<home>/.env`` with the absolute storage folders if there is none.

    An existing file is never changed here (install/install.sh adds missing
    folders to it, showing the edit). Returns a note for the log, or None.
    """

    path = home / ENV_FILE_NAME
    if path.exists():
        return None
    update_env_file(path, storage_settings(home))
    return f"Created {path} with the data, datasets and outputs folders."


def prepare_environment(
    env: MutableMapping[str, str],
    os_kind: str,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
    exists: Callable[[str], bool] = os.path.exists,
    scripts_dir: str | None = None,
) -> list[str]:
    """Adjust ``env`` in place for this OS and return what was changed, for the log.

    ``scripts_dir`` is the virtual environment's bin (Scripts on Windows) folder,
    by default this interpreter's: the launchers start the venv's Python without
    activating the venv, so console scripts such as ``lerobot-find-port``, which
    the suite's hints name, would otherwise be missing.
    """

    notes = []
    env.setdefault("PYTHONUTF8", "1")  # child processes: trainer, scripts
    scripts = scripts_dir if scripts_dir is not None else str(Path(sys.executable).parent)
    entries = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    if scripts and scripts not in entries:
        env["PATH"] = os.pathsep.join([scripts, *entries])
        notes.append(f"Put {scripts} first on PATH (the suite's console scripts).")
    nvidia = os_kind == "linux" and exists(NVIDIA_PROC)
    advice = mujoco_gl_advice(os_kind, env, find_library, nvidia=nvidia)
    if advice.problem:
        env.pop("MUJOCO_GL", None)
        notes.append(f"Ignoring MUJOCO_GL={advice.current}: `import mujoco` fails with it here.")
        advice = mujoco_gl_advice(os_kind, env, find_library, nvidia=nvidia)
    if advice.current is None and advice.recommended:
        env["MUJOCO_GL"] = advice.recommended
        notes.append(f"Set MUJOCO_GL={advice.recommended}: {advice.reason}.")
    return notes


def browser_host(host: str) -> str:
    """Where a browser on this machine reaches a server bound to ``host``."""

    if host in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def suite_url(host: str, port: int) -> str:
    return f"http://{browser_host(host)}:{port}/"


def can_open_browser(os_kind: str, env: MutableMapping[str, str]) -> bool:
    """False on Linux without a display, where webbrowser could start a text browser."""

    if os_kind != "linux":
        return True
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def health_ok(
    url: str,
    timeout: float = 1.0,
    urlopen: Callable[..., Any] = _DIRECT.open,
) -> bool:
    """True if an SO-101 suite answers ``GET <url>api/health`` with ``"ok": true``."""

    try:
        with urlopen(url + "api/health", timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                return False
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(body, dict) and body.get("ok") is True


def open_when_ready(
    url: str,
    *,
    timeout: float = READY_TIMEOUT_S,
    poll: Callable[[str], bool] = health_ok,
    opener: Callable[[str], object] = webbrowser.open,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> bool:
    """Open ``url`` once the suite answers; give up after ``timeout`` seconds."""

    deadline = clock() + timeout
    while clock() < deadline:
        if poll(url):
            opener(url)
            return True
        sleep(0.25)
    return False


class _HangupStopsServer:
    """SIGHUP handler: the graceful stop uvicorn gives SIGINT and SIGTERM.

    uvicorn handles only SIGINT and SIGTERM. Closing the terminal the suite
    runs in sends SIGHUP, whose default action ends the process at once: no
    lifespan shutdown, so the trainer (in its own session) is orphaned and the
    arm is not collapsed. This forwards SIGHUP to the SIGTERM handler uvicorn
    installs while it serves. Before that handler exists, nothing has started
    and the default action (exit) is kept.
    """

    def __init__(self) -> None:
        self.forwarded = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        if self.forwarded:
            # A repeated hang-up, or uvicorn re-raising the signal it handled
            # once its shutdown is complete: the shutdown is already under way.
            return
        stop = signal.getsignal(signal.SIGTERM)
        if not callable(stop):
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
            return
        self.forwarded = True
        stop(signum, frame)


@contextmanager
def hangup_stops_server() -> Iterator[None]:
    """While the server runs, SIGHUP stops it gracefully (POSIX, main thread only)."""

    hangup = getattr(signal, "SIGHUP", None)
    if hangup is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(hangup, _HangupStopsServer())
    try:
        yield
    finally:
        signal.signal(hangup, previous)


def run_server(host: str, port: int) -> None:
    """Run the API server in this process until SIGINT, SIGTERM or SIGHUP."""

    import uvicorn

    with hangup_stops_server():
        uvicorn.run(APP, host=host, port=port, factory=True)


def _serve(host: str, port: int) -> None:
    run_server(host, port)


def _pause(enabled: bool) -> None:
    """Keep a launcher-opened console window open so the error can be read."""

    if not enabled:
        return
    try:
        input("Press Enter to close this window.")
    except (EOFError, OSError, RuntimeError):
        pass


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="so101-suite",
        description="Start the SO-101 suite and open it in a browser.",
    )
    parser.add_argument("--home", help="data folder (default: SO101_HOME, else the per-user one)")
    parser.add_argument("--host", help="address to bind (default: SO101_HOST, else 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: SO101_PORT, else 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser")
    parser.add_argument(
        "--pause-on-error",
        action="store_true",
        help="wait for Enter before exiting on an error (for launcher windows)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = os.environ
    kind = os_name(platform.system())
    try:
        home = resolve_home(args.home, kind, env, Path.home())
        home.mkdir(parents=True, exist_ok=True)
        os.chdir(home)
        created = init_settings_file(home)
    except OSError as exc:
        print(f"so101-suite: can't use the data folder: {exc}", file=sys.stderr)
        _pause(args.pause_on_error)
        return 1
    # Settings reads <SO101_HOME>/.env: make that this folder, --home included.
    env["SO101_HOME"] = str(home)
    notes = prepare_environment(env, kind)
    if created:
        notes.insert(0, created)
    # The server reads its own settings; pass flags through the same variables
    # so the request guard sees the address actually bound.
    if args.host:
        env["SO101_HOST"] = args.host
    if args.port:
        env["SO101_PORT"] = str(args.port)

    from .config import get_settings

    settings = get_settings()
    host, port = settings.host, settings.port
    url = suite_url(host, port)
    browser = not args.no_browser and can_open_browser(kind, env)

    print("SO-101 suite")
    print(f"  data folder  {home}  (settings go in .env here)")
    print(f"  address      {url}")
    for note in notes:
        print(f"  note         {note}")

    if health_ok(url):
        print("  The suite is already running.")
        if browser:
            webbrowser.open(url)
        return 0

    print(f"  {QUIT_WARNING}")
    print("  Press Ctrl+C to stop.", flush=True)
    if browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    try:
        _serve(host, port)
    except KeyboardInterrupt:
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            return 0
        if code == STARTUP_FAILURE:
            print(
                f"so101-suite: couldn't start on {host}:{port}. If another program uses "
                f"port {port}, set SO101_PORT=<free port> in {home / '.env'} or pass --port.",
                file=sys.stderr,
            )
        _pause(args.pause_on_error)
        return code
    except Exception as exc:
        print(f"so101-suite: the server failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        _pause(args.pause_on_error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

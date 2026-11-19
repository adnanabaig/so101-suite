#!/usr/bin/env python3
"""Install the built wheel into a throwaway venv and run the so101 CLI from it.

    python scripts/wheel_smoke.py DIST_DIR [--ui required|absent|auto] [--keep]

The venv, and everything the CLI writes, lives in a temp directory that is
deleted afterwards (kept with --keep). The CLI runs from an empty working
directory with PYTHONPATH, PYTHONHOME and every SO101_* variable removed, so
so101 comes from the wheel and no checkout or developer .env leaks in. pip
takes its index and certificate settings (PIP_INDEX_URL, PIP_CERT, ...) from
the environment.

Checks: `so101 --help` and `so101 info` succeed; so101 imports from the venv;
importlib.resources finds so101/web/index.html with --ui required and does not
with --ui absent. Then the installed app serves: the venv's
`python -m uvicorn --factory so101.services.app:create_app` starts on a free
loopback port with SO101_MOCK=1 and the temp data folders, and

- with the UI, GET / returns the installed so101/web/index.html, whose SHA-256
  is the one the build stamp (so101-ui-stamp.json) lists, and every
  /assets/... file that index.html references loads with the installed bytes;
- without it, GET / returns the JSON placeholder.

The server is stopped afterwards (SIGTERM, then SIGKILL). Last, nothing was
written into the working directory by the CLI or the server.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

PROBE = """
import importlib.resources, json, so101
web = importlib.resources.files("so101") / "web"
print(json.dumps({
    "file": so101.__file__,
    "version": so101.__version__,
    "web": (web / "index.html").is_file(),
    "web_dir": str(web),
}))
"""
DROPPED_VARIABLES = frozenset({"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"})
APP = "so101.services.app:create_app"
UI_STAMP = "so101-ui-stamp.json"
# src="/assets/..." and href="/assets/..." in the built index.html.
ASSET_REFERENCE = re.compile(r"""(?:src|href)=["'](/assets/[^"'?#]+)["']""")
SERVER_START_TIMEOUT = 60.0
SERVER_STOP_TIMEOUT = 15.0

# GET of a path on the server -> (status, content type, body).
Fetch = Callable[[str], tuple[int, str, bytes]]


class SmokeError(Exception):
    pass


def clean_env(home: Path) -> dict[str, str]:
    """The environment for the venv: no checkout on sys.path, mock hardware, temp data."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in DROPPED_VARIABLES and not key.startswith("SO101_")
    }
    env.update(
        {
            "SO101_MOCK": "1",
            # The data folder, whose .env the suite reads: a temp one, never the user's.
            "SO101_HOME": str(home / "so101-home"),
            "SO101_DATA_DIR": str(home / "data"),
            "SO101_DATASETS_DIR": str(home / "datasets"),
            "SO101_OUTPUTS_DIR": str(home / "outputs"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return env


def find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("so101-*.whl"))
    if len(wheels) != 1:
        found = ", ".join(path.name for path in wheels) or "none"
        raise SmokeError(f"expected one so101 wheel in {dist_dir}, found {found}")
    return wheels[0]


def probe_problems(probe: dict[str, object], venv: Path, ui: str) -> list[str]:
    """What is wrong with the installed package, judged from the PROBE output."""
    problems = []
    location = Path(str(probe.get("file", "")))
    if not location.resolve().is_relative_to(venv.resolve()):
        problems.append(f"so101 imported from {location}, not from the venv at {venv}")
    if ui == "required" and probe.get("web") is not True:
        problems.append("so101/web/index.html is not in the installed package (--ui required)")
    if ui == "absent" and probe.get("web") is not False:
        problems.append("so101/web/index.html is in the installed package (--ui absent)")
    return problems


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str, timeout: float = 10.0) -> tuple[int, str, bytes]:
    """(status, content type, body); an HTTP error status is returned, not raised."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.headers.get("content-type", ""), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read()


def _wait_for_health(base: str, server: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SmokeError(f"the installed server exited with code {server.returncode}")
        try:
            status, _, body = _get(f"{base}/api/health", timeout=2)
        except (OSError, http.client.HTTPException):
            time.sleep(0.2)
            continue
        if status == 200 and json.loads(body).get("ok") is True:
            return
        time.sleep(0.2)
    raise SmokeError(f"the installed server did not answer {base}/api/health in {timeout:g} s")


def _stop(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=SERVER_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=SERVER_STOP_TIMEOUT)


def ui_problems(web_dir: Path, fetch: Fetch) -> list[str]:
    """What is wrong with how the server serves the UI installed in ``web_dir``."""
    status, content_type, body = fetch("/")
    if status != 200 or not content_type.startswith("text/html"):
        return [f"GET / returned {status} {content_type!r}, not the built index.html"]
    problems = []
    if body != (web_dir / "index.html").read_bytes():
        problems.append("GET / is not the installed so101/web/index.html")
    try:
        manifest = json.loads((web_dir / UI_STAMP).read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"the installed {UI_STAMP} is missing or unreadable ({exc})")
    else:
        if manifest.get("index.html") != hashlib.sha256(body).hexdigest():
            problems.append(f"GET / is not the index.html that {UI_STAMP} lists")
    assets = sorted(set(ASSET_REFERENCE.findall(body.decode("utf-8", "replace"))))
    if not assets:
        problems.append("the served index.html references no /assets/ file")
    for asset in assets:
        status, _, data = fetch(asset)
        installed = web_dir / asset.lstrip("/")
        if status != 200:
            problems.append(f"GET {asset} returned {status}")
        elif not installed.is_file() or data != installed.read_bytes():
            problems.append(f"GET {asset} is not the installed so101/web{asset}")
    return problems


def placeholder_problems(fetch: Fetch) -> list[str]:
    """What is wrong with GET / of a server whose package has no UI."""
    status, _, body = fetch("/")
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
    if status != 200 or not isinstance(payload, dict) or "ui" not in payload:
        return [f"GET / returned {status}, not the JSON placeholder of a package without a UI"]
    return []


def serve_checks(
    python: Path, home: Path, work: Path, env: dict[str, str], web_dir: Path | None
) -> str:
    """Start the installed app, check what GET / serves, stop it. Returns what passed."""
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = home / "server.log"
    command = [str(python), "-m", "uvicorn", "--factory", APP, "--host", "127.0.0.1"]
    with log.open("wb") as log_file:
        server = subprocess.Popen(
            [*command, "--port", str(port)],
            cwd=work,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_health(base, server, SERVER_START_TIMEOUT)

            def fetch(path: str) -> tuple[int, str, bytes]:
                return _get(f"{base}{path}")

            problems = ui_problems(web_dir, fetch) if web_dir else placeholder_problems(fetch)
        except (SmokeError, OSError, http.client.HTTPException, ValueError) as exc:
            problems = [str(exc)]
        finally:
            _stop(server)
    if problems:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise SmokeError("\n".join(problems) + f"\nserver log:\n{tail}")
    if web_dir:
        return "the installed app serves the built UI at / (index.html and its assets)"
    return "the installed app serves the JSON placeholder at / (no UI)"


def _run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> str:
    proc = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise SmokeError(
            f"`{' '.join(command)}` exited {proc.returncode}\n{proc.stdout}{proc.stderr}".rstrip()
        )
    return proc.stdout


def smoke(wheel: Path, home: Path, ui: str) -> list[str]:
    """Install ``wheel`` under ``home`` and run the checks; returns what passed."""
    venv, work = home / "venv", home / "work"
    work.mkdir(parents=True)
    env = clean_env(home)
    passed = []
    _run([sys.executable, "-m", "venv", str(venv)], cwd=work, env=env, timeout=300)
    python, so101 = venv / "bin" / "python", venv / "bin" / "so101"
    _run(
        [str(python), "-m", "pip", "install", "--no-input", str(wheel.resolve())],
        cwd=work,
        env=env,
        timeout=900,
    )
    passed.append(f"installed {wheel.name} into a clean venv")
    usage = _run([str(so101), "--help"], cwd=work, env=env, timeout=120)
    if "Usage" not in usage:
        raise SmokeError(f"`so101 --help` printed no usage:\n{usage}")
    passed.append("so101 --help")
    _run([str(so101), "info"], cwd=work, env=env, timeout=120)
    passed.append("so101 info")
    probe = json.loads(_run([str(python), "-c", PROBE], cwd=work, env=env, timeout=120))
    problems = probe_problems(probe, venv, ui)
    if problems:
        raise SmokeError("\n".join(problems))
    passed.append(f"so101 {probe['version']} imports from the venv: {probe['file']}")
    passed.append(f"bundled UI (so101/web/index.html): {'yes' if probe['web'] else 'no'} --ui {ui}")
    web_dir = Path(str(probe["web_dir"])) if probe["web"] else None
    passed.append(serve_checks(python, home, work, env, web_dir))
    leftovers = sorted(path.name for path in work.iterdir())
    if leftovers:
        raise SmokeError(f"the CLI or the server wrote into its working directory: {leftovers}")
    passed.append("nothing written into the working directory")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dist_dir", type=Path, help="directory holding the built wheel")
    parser.add_argument("--ui", choices=("required", "absent", "auto"), default="auto")
    parser.add_argument("--keep", action="store_true", help="keep the temp venv for debugging")
    args = parser.parse_args(argv)
    home = Path(tempfile.mkdtemp(prefix="so101-wheel-smoke-"))
    try:
        wheel = find_wheel(args.dist_dir)
        for line in smoke(wheel, home, args.ui):
            print(f"  ok    {line}")
    except (SmokeError, subprocess.TimeoutExpired) as exc:
        print(f"wheel_smoke: FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep:
            print(f"wheel_smoke: kept {home}")
        else:
            shutil.rmtree(home, ignore_errors=True)
    print(f"wheel_smoke: {wheel.name} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

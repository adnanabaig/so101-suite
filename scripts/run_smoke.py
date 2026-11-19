#!/usr/bin/env python3
"""Start a throwaway mock API, run scripts/smoke_suite.py against it, then stop it.

    python scripts/run_smoke.py [--host 127.0.0.1] [--port 0] [--timeout 60]

This is `make ci-smoke` and the CI smoke job. The server is
`python -m uvicorn --factory so101.services.app:create_app`, without auto-reload. It runs in a
temp directory with SO101_MOCK=1, its data, datasets and outputs dirs inside
that directory, SO101_HOME (the data folder, whose .env the suite reads) set to
it too, and every other SO101_* variable removed, so neither the checkout's nor
the installed suite's .env applies, and no hardware setting does. --port 0 (the
default) picks a free port.

The smoke suite only runs once /api/health reports mock=true; smoke_suite.py
itself also refuses (exit 3) anything but a fresh mock server, because it
E-stops, resets and moves the arm. Whatever happens, the server is stopped
(SIGTERM, then SIGKILL after 10 s) and the temp directory is deleted.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
# Built by uvicorn --factory (so101.services.app.APP_FACTORY).
APP = "so101.services.app:create_app"
LOG_TAIL_LINES = 40
SMOKE_TIMEOUT_SECONDS = 300


class SmokeServerError(Exception):
    pass


def client_url(host: str, port: int) -> str:
    """The URL a client uses for a server bound to ``host``; wildcards become loopback."""
    if host in {"", "0.0.0.0", "::"}:
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}"


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def free_port(host: str) -> int:
    with socket.socket(_family(host), socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def port_in_use(host: str, port: int) -> bool:
    """True if uvicorn could not bind host:port (it sets SO_REUSEADDR as well).

    Checked before starting: while our server fails to bind, whatever holds the
    port would answer the health check in its place.
    """
    with socket.socket(_family(host), socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def server_env(home: Path) -> dict[str, str]:
    """Mock hardware and temp data dirs; nothing from the developer's SO101_* settings."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("SO101_")}
    env.update(
        {
            "SO101_MOCK": "1",
            # The data folder: <home>/.env, which doesn't exist, is the only
            # settings file besides ./.env, which is the same file.
            "SO101_HOME": str(home),
            "SO101_DATA_DIR": str(home / "data"),
            "SO101_DATASETS_DIR": str(home / "datasets"),
            "SO101_OUTPUTS_DIR": str(home / "outputs"),
        }
    )
    # The server's working directory is the temp dir, where relative PYTHONPATH
    # entries would point somewhere else and an empty entry means the temp dir.
    entries = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(str(Path(entry).resolve()) for entry in entries)
    else:
        env.pop("PYTHONPATH", None)
    return env


def log_tail(log: Path) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    return "\n".join(lines[-LOG_TAIL_LINES:]) or "(no server output)"


def wait_for_health(base: str, server: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    """Poll /api/health until it answers; fail if the server exits or the time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SmokeServerError(f"the server exited with code {server.returncode} on startup")
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as response:
                health = json.loads(response.read())
        except (OSError, ValueError, http.client.HTTPException):
            time.sleep(0.2)
            continue
        if not isinstance(health, dict):
            raise SmokeServerError(f"{base}/api/health returned {health!r}, not an object")
        return health
    raise SmokeServerError(f"no answer from {base}/api/health within {timeout:g}s")


def stop(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=10)


def run(host: str, port: int, timeout: float, app: str = APP) -> int:
    if port and port_in_use(host, port):
        print(
            f"run_smoke: FAIL: {host}:{port} is already in use. Stop whatever is listening "
            "there, or use --port 0 for a free port.",
            file=sys.stderr,
        )
        return 1
    port = port or free_port(host)
    base = client_url(host, port)
    home = Path(tempfile.mkdtemp(prefix="so101-smoke-"))
    log = home / "server.log"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "--factory",
        app,
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"run_smoke: starting the mock API on {base} (temp dir {home})", flush=True)
    try:
        with log.open("wb") as log_file:
            server = subprocess.Popen(
                command, cwd=home, env=server_env(home), stdout=log_file, stderr=subprocess.STDOUT
            )
            try:
                health = wait_for_health(base, server, timeout)
                if health.get("mock") is not True:
                    raise SmokeServerError(
                        f"/api/health reports mock={health.get('mock')!r}, not true; the smoke "
                        "suite runs only against a mock server"
                    )
                smoke = subprocess.run(
                    [sys.executable, str(SCRIPTS / "smoke_suite.py"), "--base", base],
                    timeout=SMOKE_TIMEOUT_SECONDS,
                )
            finally:
                stop(server)
        if smoke.returncode != 0:
            print(
                f"run_smoke: FAIL: smoke_suite.py exited {smoke.returncode}; the server stopped. "
                f"Server log:\n{log_tail(log)}",
                file=sys.stderr,
            )
            return smoke.returncode
        print(f"run_smoke: passed; the server on {base} stopped")
        return 0
    except (SmokeServerError, subprocess.TimeoutExpired) as exc:
        print(
            f"run_smoke: FAIL: {exc}; the server stopped. Server log:\n{log_tail(log)}",
            file=sys.stderr,
        )
        return 1
    finally:
        shutil.rmtree(home, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port (the default)")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for startup")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error(f"--port must be 0-65535, not {args.port}")
    return run(args.host, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())

"""Foreground Suite lifecycle: Ctrl+C Halts; a second Ctrl+C E-stops."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import queue
import signal
import subprocess
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("service_suite", ROOT / "scripts/service_suite.py")
assert SPEC and SPEC.loader
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)
AUXILIARY = tuple(unit for unit in suite.UNITS if unit != "so101-motor.service")


def post_stop(kind: str) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:8000/api/robot/{kind}",
        method="POST",
        data=json.dumps({"reason": "foreground session interrupt"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        reply = json.load(response)
    if reply.get("ok") is not True:
        raise RuntimeError(f"{kind} was not acknowledged")


def force_units(units) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", "--no-block", *units],
        check=False,
        start_new_session=True,
    )
    subprocess.run(
        ["systemctl", "--user", "kill", "--signal=SIGKILL", *units],
        check=False,
        start_new_session=True,
    )


class Shutdown:
    def __init__(self):
        self.stage = 0
        self.lock = threading.RLock()
        self.requested = threading.Event()
        self.halt_done = threading.Event()
        self.estop_done = threading.Event()
        self.motor_off = threading.Event()
        self.last_emergency = 0.0
        self.signals: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self.closed = False

    def enqueue_signal(self, signum, frame=None) -> None:
        self.signals.put(signum)

    def dispatch_signals(self) -> None:
        while True:
            number = self.signals.get()
            if number is None:
                return
            if not self.closed:
                self.interrupt(number)

    def interrupt(self, signum=None, frame=None) -> None:
        with self.lock:
            previous = self.stage
            self.stage = min(2, self.stage + 1)
            current = self.stage
        self.requested.set()
        if current == 1:
            print(
                "\nHALT requested: holding torque, then the existing rest/off timer applies. "
                "Press Ctrl+C again for E-stop / forced worker shutdown.",
                flush=True,
            )
            threading.Thread(target=self.halt, daemon=True).start()
        elif previous < 2:
            print(
                "\nE-STOP requested: torque cut can let the arm drop. "
                "Force-stopping camera/recording workers.",
                flush=True,
            )
            threading.Thread(target=self.emergency, daemon=True).start()
            threading.Thread(target=self.force_workers, daemon=True).start()

    def force_workers(self) -> None:
        force_units(AUXILIARY)

    def halt(self) -> None:
        try:
            post_stop("halt")
            self.halt_done.set()
        except Exception as error:
            print(f"Halt not confirmed: {error}. Requesting E-stop.", flush=True)
            if self.stage < 2:
                self.interrupt()

    def emergency(self) -> None:
        self.last_emergency = time.monotonic()
        try:
            post_stop("estop")
            self.estop_done.set()
        except Exception as error:
            print(
                f"E-stop torque-off NOT CONFIRMED: {error}. "
                "Use the physical power cutoff. Motor service stays alive to finish the stop.",
                flush=True,
            )

    def safe_to_stop_motor(self) -> bool:
        try:
            state = suite.fetch("http://127.0.0.1:8000/api/robot/status")
        except Exception:
            return False
        if state.get("torque_state") != "off" or state.get("torque_unconfirmed", True):
            return False
        if self.stage == 2:
            return self.estop_done.is_set() or state.get("estopped") is True
        return self.halt_done.is_set()

    def finish(self) -> None:
        while True:
            if self.stage == 2 and not self.estop_done.is_set():
                if time.monotonic() - self.last_emergency > 5:
                    self.emergency()
            if self.safe_to_stop_motor():
                if self.stage == 2:
                    if not self.safe_to_stop_motor():
                        continue
                    force_units(suite.UNITS)
                self.motor_off.set()
                break
            time.sleep(0.2)
        subprocess.run(
            ["systemctl", "--user", "stop", "so101-suite.target", *suite.UNITS],
            check=True,
            start_new_session=True,
        )
        print("Suite stopped. Cameras, simulation publisher and recorder are off.", flush=True)


def run(open_browser: bool) -> None:
    folder = ROOT / ".artifacts/run"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "suite-session.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A foreground Suite session is already running") from None
        shutdown = Shutdown()
        threading.Thread(target=shutdown.dispatch_signals, daemon=True).start()
        old = {
            number: signal.signal(number, shutdown.enqueue_signal)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        started = False
        try:
            suite.require_idle_restart()
            if shutdown.requested.is_set():
                return
            started = True
            suite.start_services()
            deadline = time.monotonic() + 30
            while not shutdown.requested.is_set():
                if suite.status()["all_local_online"]:
                    print(
                        "Suite running: http://127.0.0.1:8000/operate\n"
                        "Ctrl+C: Halt and safe shutdown. Ctrl+C again: E-stop / force workers.",
                        flush=True,
                    )
                    if open_browser:
                        webbrowser.open(f"http://127.0.0.1:8000/operate?refresh={time.time_ns()}")
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Suite did not become ready; shutting down")
                shutdown.requested.wait(0.2)
            shutdown.requested.wait()
        finally:
            try:
                if started or shutdown.requested.is_set():
                    if not shutdown.requested.is_set():
                        shutdown.interrupt()
                    shutdown.finish()
            finally:
                shutdown.closed = True
                for number, handler in old.items():
                    signal.signal(number, handler)
                shutdown.signals.put(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="Open the Operate dashboard")
    args = parser.parse_args()
    run(args.open)


if __name__ == "__main__":
    main()

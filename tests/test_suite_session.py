"""Foreground stop escalation preserves the real manager's torque-off boundary."""

import importlib.util
import json
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "suite_session", Path(__file__).resolve().parents[1] / "scripts/suite_session.py"
)
session = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(session)


def test_second_interrupt_does_not_wait_for_blocked_halt(monkeypatch):
    halt_started = threading.Event()
    release_halt = threading.Event()
    estop_called = threading.Event()
    forced = threading.Event()
    calls = []

    def stop(kind):
        calls.append(kind)
        if kind == "halt":
            halt_started.set()
            release_halt.wait(2)
        else:
            estop_called.set()

    def force(units):
        assert "so101-motor.service" not in units
        forced.set()

    monkeypatch.setattr(session, "post_stop", stop)
    monkeypatch.setattr(session, "force_units", force)
    shutdown = session.Shutdown()
    try:
        shutdown.interrupt()
        assert halt_started.wait(1)
        shutdown.interrupt()
        assert estop_called.wait(1)
        assert forced.wait(1)
        assert not shutdown.halt_done.is_set()
        assert calls == ["halt", "estop"]
    finally:
        release_halt.set()
    assert shutdown.halt_done.wait(1)


@pytest.mark.parametrize(
    "state",
    [
        {"torque_state": "on", "torque_unconfirmed": False, "estopped": True},
        {"torque_state": "off", "torque_unconfirmed": True, "estopped": True},
        {"torque_state": "off"},
    ],
)
def test_emergency_ack_never_overrides_unconfirmed_torque(monkeypatch, state):
    shutdown = session.Shutdown()
    shutdown.stage = 2
    shutdown.estop_done.set()
    monkeypatch.setattr(session.suite, "fetch", lambda url: state)
    assert not shutdown.safe_to_stop_motor()


def test_unreachable_api_is_not_torque_off(monkeypatch):
    shutdown = session.Shutdown()

    def unavailable(url):
        raise OSError("API unreachable")

    monkeypatch.setattr(session.suite, "fetch", unavailable)
    assert not shutdown.safe_to_stop_motor()


@pytest.mark.parametrize("stage", [1, 2])
def test_shutdown_only_stops_motor_after_fresh_torque_off(monkeypatch, stage):
    shutdown = session.Shutdown()
    shutdown.stage = stage
    shutdown.halt_done.set()
    shutdown.estop_done.set()
    monkeypatch.setattr(
        session.suite,
        "fetch",
        lambda url: {
            "torque_state": "off",
            "torque_unconfirmed": False,
            "estopped": stage == 2,
        },
    )
    commands = []
    forced = []
    monkeypatch.setattr(session.subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(session, "force_units", lambda units: forced.append(units))
    shutdown.finish()
    assert shutdown.motor_off.is_set()
    assert commands[-1][2] == "stop"
    assert "so101-motor.service" in commands[-1]
    assert bool(forced) == (stage == 2)


def test_cached_torque_off_never_allows_force_worker_to_kill_motor(monkeypatch):
    shutdown = session.Shutdown()
    shutdown.motor_off.set()
    forced = []
    monkeypatch.setattr(session, "force_units", lambda units: forced.extend(units))
    shutdown.force_workers()
    assert "so101-motor.service" not in forced


def test_reentrant_interrupt_cannot_deadlock_on_stage_lock(monkeypatch):
    shutdown = session.Shutdown()
    monkeypatch.setattr(session, "post_stop", lambda kind: None)
    complete = threading.Event()

    def nested():
        with shutdown.lock:
            shutdown.interrupt()
        complete.set()

    worker = threading.Thread(target=nested, daemon=True)
    worker.start()
    assert complete.wait(1)


def test_signal_handler_never_enters_event_or_shutdown_locks():
    shutdown = session.Shutdown()
    with shutdown.requested._cond:
        shutdown.enqueue_signal(signal.SIGINT)
        shutdown.enqueue_signal(signal.SIGINT)
    assert not shutdown.requested.is_set()
    assert shutdown.signals.get_nowait() == signal.SIGINT
    assert shutdown.signals.get_nowait() == signal.SIGINT


def test_halt_failure_escalates_to_estop(monkeypatch):
    shutdown = session.Shutdown()
    shutdown.stage = 1
    called = threading.Event()
    monkeypatch.setattr(session, "force_units", lambda units: None)

    def stop(kind):
        if kind == "halt":
            raise OSError("hold failed")
        called.set()

    monkeypatch.setattr(session, "post_stop", stop)
    shutdown.halt()
    assert called.wait(1)
    assert shutdown.stage == 2


@pytest.mark.parametrize("second_interrupt", [False, True])
def test_foreground_process_handles_real_terminal_signals(tmp_path, second_interrupt):
    script = Path(session.__file__).resolve()
    code = f"""
import importlib.util, pathlib, threading, json
spec=importlib.util.spec_from_file_location("runner", {str(script)!r})
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.ROOT=pathlib.Path({str(tmp_path)!r})
state={{"torque_state":"on","torque_unconfirmed":False,"estopped":False}}
runner.suite.require_idle_restart=lambda:None
runner.suite.start_services=lambda:None
runner.suite.status=lambda:{{"all_local_online":True}}
runner.suite.fetch=lambda url:dict(state)
def stop(kind):
 print("STOP:"+kind,flush=True)
 if kind=="estop":state.update(torque_state="off",estopped=True)
 elif not {second_interrupt!r}:
  threading.Timer(.1,lambda:state.update(torque_state="off")).start()
runner.post_stop=stop
def command(args,**kwargs):
 print("COMMAND:"+json.dumps(args),flush=True)
runner.subprocess.run=command
runner.run(False)
"""
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = queue.Queue()
    reader = threading.Thread(
        target=lambda: [lines.put(line) for line in child.stdout], daemon=True
    )
    reader.start()
    observed = []
    try:
        while True:
            line = lines.get(timeout=5)
            observed.append(line)
            if line.startswith("Ctrl+C:"):
                break
        child.send_signal(signal.SIGINT)
        if second_interrupt:
            while True:
                line = lines.get(timeout=5)
                observed.append(line)
                if line.startswith("STOP:halt"):
                    break
            child.send_signal(signal.SIGINT)
        assert child.wait(timeout=5) == 0
        reader.join(timeout=1)
        while not lines.empty():
            observed.append(lines.get_nowait())
        output = "".join(observed)
        assert "STOP:halt" in output
        assert ("STOP:estop" in output) is second_interrupt
        assert "Suite stopped." in output
        commands = [
            json.loads(line.removeprefix("COMMAND:"))
            for line in observed
            if line.startswith("COMMAND:")
        ]
        assert any("so101-motor.service" in command and "stop" in command for command in commands)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

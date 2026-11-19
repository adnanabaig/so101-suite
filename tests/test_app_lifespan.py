"""Application lifespan teardown keeps the robot's safety stop mandatory."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from so101.services import app as app_module


class _BusLock:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def release(self) -> None:
        self._calls.append("bus.release")


class _Manager:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def shutdown(self) -> None:
        self._calls.append("manager.shutdown")


class _AdoptingManager(_Manager):
    def adopt_global_energized_intent(self, intent, *, error=None) -> None:
        assert error is None
        self._calls.append(f"manager.adopt:{intent['purpose']}")

    def publish_restored_recovery_intent(self) -> None:
        pass


class _Runner:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def shutdown(self) -> None:
        self._calls.append("trainer.shutdown")
        raise RuntimeError("trainer teardown failed")


class _CleanRunner:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def shutdown(self) -> None:
        self._calls.append("trainer.shutdown")


def test_lifespan_always_shuts_down_manager_after_other_teardown_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    manager = _Manager(calls)
    runner = _Runner(calls)

    monkeypatch.setattr(app_module, "take_server_bus_lock", lambda _settings: _BusLock(calls))
    monkeypatch.setattr(app_module, "get_manager", lambda: manager)
    monkeypatch.setattr(app_module, "recover_interrupted_proposals", lambda: None)
    monkeypatch.setattr(app_module, "get_runner", lambda: runner)

    def fail_agent_shutdown() -> None:
        calls.append("agent.shutdown")
        raise RuntimeError("agent teardown failed")

    def fail_eval_shutdown() -> None:
        calls.append("eval.shutdown")
        raise RuntimeError("eval teardown failed")

    monkeypatch.setattr(app_module, "shutdown_executor", fail_agent_shutdown)
    monkeypatch.setattr(app_module.eval, "shutdown_session", fail_eval_shutdown)

    async def run_lifespan() -> None:
        async with app_module.lifespan(FastAPI()):
            calls.append("serving")

    with pytest.raises(BaseExceptionGroup) as raised:
        asyncio.run(run_lifespan())

    assert [str(error) for error in raised.value.exceptions] == [
        "agent teardown failed",
        "eval teardown failed",
        "trainer teardown failed",
    ]
    assert calls == [
        "serving",
        "agent.shutdown",
        "eval.shutdown",
        "trainer.shutdown",
        "manager.shutdown",
        "bus.release",
    ]


def test_lifespan_body_failure_still_runs_manager_safety_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    manager = _Manager(calls)
    runner = _CleanRunner(calls)

    monkeypatch.setattr(app_module, "take_server_bus_lock", lambda _settings: _BusLock(calls))
    monkeypatch.setattr(app_module, "get_manager", lambda: manager)
    monkeypatch.setattr(app_module, "recover_interrupted_proposals", lambda: None)
    monkeypatch.setattr(app_module, "get_runner", lambda: runner)
    monkeypatch.setattr(app_module, "shutdown_executor", lambda: calls.append("agent.shutdown"))
    monkeypatch.setattr(app_module.eval, "shutdown_session", lambda: calls.append("eval.shutdown"))

    async def run_lifespan() -> None:
        async with app_module.lifespan(FastAPI()):
            calls.append("serving")
            raise RuntimeError("application failed")

    with pytest.raises(RuntimeError, match="application failed"):
        asyncio.run(run_lifespan())

    assert calls == [
        "serving",
        "agent.shutdown",
        "eval.shutdown",
        "trainer.shutdown",
        "manager.shutdown",
        "bus.release",
    ]


def test_lifespan_startup_interruption_still_runs_manager_safety_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    manager = _Manager(calls)
    runner = _CleanRunner(calls)

    monkeypatch.setattr(app_module, "take_server_bus_lock", lambda _settings: _BusLock(calls))
    monkeypatch.setattr(app_module, "get_manager", lambda: manager)
    monkeypatch.setattr(app_module, "get_runner", lambda: runner)
    monkeypatch.setattr(app_module, "shutdown_executor", lambda: calls.append("agent.shutdown"))
    monkeypatch.setattr(app_module.eval, "shutdown_session", lambda: calls.append("eval.shutdown"))

    def interrupt_startup() -> None:
        calls.append("startup.interrupted")
        raise SystemExit("startup interrupted")

    monkeypatch.setattr(app_module, "recover_interrupted_proposals", interrupt_startup)

    async def run_lifespan() -> None:
        async with app_module.lifespan(FastAPI()):
            calls.append("serving")

    with pytest.raises(SystemExit, match="startup interrupted"):
        asyncio.run(run_lifespan())

    assert calls == [
        "startup.interrupted",
        "agent.shutdown",
        "eval.shutdown",
        "trainer.shutdown",
        "manager.shutdown",
        "bus.release",
    ]


def test_real_lifespan_adopts_global_energized_truth_before_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    manager = _AdoptingManager(calls)
    runner = _CleanRunner(calls)

    monkeypatch.setattr(app_module, "current_backend", lambda _settings: "real")
    monkeypatch.setattr(app_module, "take_server_bus_lock", lambda _settings: _BusLock(calls))
    monkeypatch.setattr(
        app_module,
        "read_global_energized_intent",
        lambda _settings: {"purpose": "RobotManager teleop before server crash"},
    )
    monkeypatch.setattr(app_module, "get_manager", lambda: manager)
    monkeypatch.setattr(app_module, "recover_interrupted_proposals", lambda: None)
    monkeypatch.setattr(app_module, "get_runner", lambda: runner)
    monkeypatch.setattr(app_module, "shutdown_executor", lambda: calls.append("agent.shutdown"))
    monkeypatch.setattr(app_module.eval, "shutdown_session", lambda: calls.append("eval.shutdown"))

    async def run_lifespan() -> None:
        async with app_module.lifespan(FastAPI()):
            calls.append("serving")

    asyncio.run(run_lifespan())

    assert calls[:2] == [
        "manager.adopt:RobotManager teleop before server crash",
        "serving",
    ]

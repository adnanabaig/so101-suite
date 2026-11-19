"""scripts/smoke_suite.py: it runs only against a fresh mock server (finding 2),
and its Collapse check requires a real finish at rest (finding 3).

The full run against a throwaway mock API is `make ci-smoke`
(tests/test_run_smoke.py covers that harness).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "smoke_suite.py"
    spec = importlib.util.spec_from_file_location("smoke_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()

REST = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 95.0,
    "wrist_flex": 60.0,
    "wrist_roll": 0.0,
}


def legacy_collapse_check(status: dict[str, Any]) -> bool:
    """The Collapse check smoke_suite.py had at v0.2.0 (4c79533), verbatim."""
    return status.get("stop_state") == "none" and status.get("torque_state") == "off"


def _start(event_id: int = 10) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "safety.collapse.start",
        "message": "Collapse: lowering to the rest pose (owner disarm)",
        "detail": {"reason": "owner disarm", "rest_pose": dict(REST)},
    }


def _done(event_id: int = 12, *, torque_off: bool = True, error: float = 0.4) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "safety.collapse.done",
        "message": "Collapse finished at the rest pose; torque off",
        "detail": {"reason": "owner disarm", "torque_off": torque_off, "final_error_deg": error},
    }


def _fallback(event_id: int = 12, cause: str = "stall") -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "safety.collapse.fallback",
        "message": f"Collapse fell back to torque-off ({cause}): no progress for 0.5 s",
        "detail": {"reason": "owner disarm", "cause": cause, "torque_off": True},
    }


AT_REST = {f"{joint}.pos": value + 0.3 for joint, value in REST.items()} | {"gripper.pos": 12.0}
IDLE_STATUS = {
    "backend": "mock",
    "stop_state": "none",
    "torque_state": "off",
    "estopped": False,
    "torque_unconfirmed": False,
}
DISARMED = {"safety_state": "disarmed"}
FAULTED = {"safety_state": "fault"}


# --- finding 3: the Collapse check ------------------------------------------------------


def test_the_old_check_passes_a_fallback_and_the_new_one_fails_it() -> None:
    events = [_start(), _fallback()]
    # What a fallback leaves behind: no stop running, torque cut, a fault.
    status = dict(IDLE_STATUS)

    assert legacy_collapse_check(status) is True  # the v0.2.0 smoke passed this
    problems = smoke.collapse_problems(events, status, FAULTED, AT_REST)
    assert any("safety.collapse.fallback (stall)" in problem for problem in problems)
    assert any("no safety.collapse.done" in problem for problem in problems)
    assert any("safety_state is 'fault'" in problem for problem in problems)


def test_a_finish_at_rest_with_torque_off_passes() -> None:
    assert smoke.collapse_problems([_done(), _start()], IDLE_STATUS, DISARMED, AT_REST) == []


@pytest.mark.parametrize(
    ("events", "status", "control", "observation", "expected"),
    [
        ([_start(), _done(), _fallback(13)], IDLE_STATUS, DISARMED, AT_REST, "fallback"),
        ([_start(), _done(torque_off=False)], IDLE_STATUS, DISARMED, AT_REST, "torque-off"),
        ([_start(), _done(error=7.5)], IDLE_STATUS, DISARMED, AT_REST, "final error"),
        ([_done()], IDLE_STATUS, DISARMED, AT_REST, "no safety.collapse.start"),
        ([], IDLE_STATUS, DISARMED, AT_REST, "no safety.collapse.done"),
        (
            [_start(), _done()],
            IDLE_STATUS | {"stop_state": "collapsing"},
            DISARMED,
            AT_REST,
            "stop_state",
        ),
        (
            [_start(), _done()],
            IDLE_STATUS | {"torque_state": "unconfirmed"},
            DISARMED,
            AT_REST,
            "torque_state",
        ),
        ([_start(), _done()], IDLE_STATUS, FAULTED, AT_REST, "safety_state"),
        (
            [_start(), _done()],
            IDLE_STATUS,
            DISARMED,
            AT_REST | {"elbow_flex.pos": 80.0},
            "elbow_flex ended at 80.00",
        ),
        ([_start(), _done()], IDLE_STATUS, DISARMED, None, "could not be read"),
    ],
)
def test_anything_short_of_a_clean_finish_at_rest_fails(
    events: list[dict[str, Any]],
    status: dict[str, Any],
    control: dict[str, Any],
    observation: dict[str, float] | None,
    expected: str,
) -> None:
    problems = smoke.collapse_problems(events, status, control, observation)
    assert any(expected in problem for problem in problems), problems


def test_an_earlier_collapses_fallback_does_not_count_against_the_newest() -> None:
    events = [_start(1), _fallback(2), _start(10), _done(12)]
    assert smoke.collapse_problems(events, IDLE_STATUS, DISARMED, AT_REST) == []


# --- finding 3 against the real manager (mock backend) -----------------------------------


@pytest.fixture
def client(manager_clock):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from so101.services.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _collapse_through_the_manager(client: Any, *, stall: bool) -> dict[str, Any]:
    from so101.services.robot_manager import StopKind, get_manager

    manager = get_manager()
    lease = manager.acquire_lease("smoke-test", 60)
    manager.arm_control(lease["id"])
    marker = client.get("/api/events", params={"limit": 1}).json()["events"][0]["id"]
    if stall:
        manager.follower().bus.smoothing = 0.0  # the arm stops following its goal
    manager.release_lease(lease["id"])
    assert manager.request_stop(StopKind.COLLAPSE, reason="smoke test").collapsing
    deadline = time.monotonic() + 10
    while manager.status()["stop_state"] != "none" and time.monotonic() < deadline:
        time.sleep(0.01)
    events = client.get("/api/events", params={"after_id": marker, "limit": 500}).json()
    status = client.get("/api/robot/status").json()
    control = client.get("/api/control/state").json()
    observation = None
    if control["safety_state"] not in {"fault", "estopped"}:
        lease = client.post("/api/control/lease", json={"source": "manual", "ttl_seconds": 5})
        lease_id = lease.json()["id"]
        observed = client.post("/api/control/observe", json={"lease_id": lease_id})
        observation = observed.json()["observation"]
        client.delete(f"/api/control/lease/{lease_id}")
    return {
        "events": events["events"],
        "status": status,
        "control": control,
        "observation": observation,
    }


def test_a_real_stall_fallback_passes_the_old_check_and_fails_the_new_one(client: Any) -> None:
    seen = _collapse_through_the_manager(client, stall=True)

    assert legacy_collapse_check(seen["status"]) is True
    problems = smoke.collapse_problems(
        seen["events"], seen["status"], seen["control"], seen["observation"]
    )
    assert any("safety.collapse.fallback (stall)" in problem for problem in problems), problems


def test_a_real_finish_at_rest_passes_the_new_check(client: Any) -> None:
    seen = _collapse_through_the_manager(client, stall=False)

    assert (
        smoke.collapse_problems(
            seen["events"], seen["status"], seen["control"], seen["observation"]
        )
        == []
    )


# --- finding 2: only a fresh mock server -------------------------------------------------


@pytest.mark.parametrize(
    ("status", "control", "expected"),
    [
        (IDLE_STATUS | {"backend": "real"}, DISARMED, "backend is 'real'"),
        (IDLE_STATUS | {"backend": "sim"}, DISARMED, "backend is 'sim'"),
        ({k: v for k, v in IDLE_STATUS.items() if k != "backend"}, DISARMED, "backend is None"),
        (IDLE_STATUS | {"estopped": True}, {"safety_state": "estopped"}, "already E-stopped"),
        (IDLE_STATUS, FAULTED, "latched fault"),
        (IDLE_STATUS | {"torque_unconfirmed": True}, DISARMED, "unconfirmed torque-off"),
    ],
)
def test_refusal_reasons(status: dict[str, Any], control: dict[str, Any], expected: str) -> None:
    reasons = smoke.refusal_reasons(status, control)
    assert any(expected in reason for reason in reasons), reasons


def test_a_fresh_mock_server_is_accepted() -> None:
    assert smoke.refusal_reasons(IDLE_STATUS, DISARMED) == []


class FakeServer:
    """Answers smoke_suite.request() like a server; records every call."""

    def __init__(self, status: dict[str, Any], control: dict[str, Any]) -> None:
        self.status, self.control = dict(status), dict(control)
        self.calls: list[tuple[str, str]] = []

    def request(self, base: str, method: str, path: str, body: dict | None = None) -> Any:
        self.calls.append((method, path))
        if path == "/api/robot/status":
            return dict(self.status)
        if path == "/api/control/state":
            return dict(self.control)
        if path == "/api/health":
            return {"ok": True, "version": "0.2.0", "mock": self.status["backend"] == "mock"}
        if path == "/api/capabilities":
            return {"capabilities": ["x"]}
        if path == "/api/diagnostics":
            return {"overall": "pass"}
        if path == "/api/simulation/scenarios":
            return ["pick"]
        if path == "/api/robot/estop":
            self.status["estopped"] = True
            return {}
        if path == "/api/robot/estop/reset":
            self.status["estopped"] = False
            return {}
        raise smoke.SmokeFailure(f"{method} {path}: stop here")


@pytest.mark.parametrize(
    ("status", "control"),
    [
        (IDLE_STATUS | {"backend": "real"}, DISARMED),
        (IDLE_STATUS | {"backend": "sim"}, DISARMED),
        (IDLE_STATUS | {"estopped": True}, {"safety_state": "estopped"}),
        (IDLE_STATUS, FAULTED),
    ],
    ids=["real", "sim", "estopped", "faulted"],
)
def test_a_real_or_latched_server_is_refused_before_anything_is_sent(
    status: dict[str, Any],
    control: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server = FakeServer(status, control)
    monkeypatch.setattr(smoke, "request", server.request)
    monkeypatch.setattr(sys, "argv", ["smoke_suite.py", "--base", "http://robot.local:8000"])

    try:
        code = smoke.main()
    except smoke.SmokeFailure:
        code = None

    assert code not in (0, None)
    assert all(method == "GET" for method, _ in server.calls), server.calls
    assert ("POST", "/api/robot/estop/reset") not in server.calls
    assert "REFUSED" in capsys.readouterr().err


def test_the_smoke_resets_only_the_e_stop_it_set(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(IDLE_STATUS, DISARMED)
    monkeypatch.setattr(smoke, "request", server.request)
    monkeypatch.setattr(sys, "argv", ["smoke_suite.py"])

    with pytest.raises(smoke.SmokeFailure):  # the fake stops at the lease
        smoke.main()

    posts = [path for method, path in server.calls if method == "POST"]
    assert posts[:2] == ["/api/robot/estop", "/api/robot/estop/reset"]
    assert "/api/robot/fault/reset" not in posts
    # The status was read before the smoke sent anything.
    assert server.calls[0] == ("GET", "/api/robot/status")

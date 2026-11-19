"""The stop routes and the visibility contract (design doc §5).

- ``POST /api/robot/halt`` is exempt from the Origin guard and never refused
  over its body; ``POST /api/robot/collapse`` is Origin-guarded.
- Status and every telemetry frame carry ``torque_state``, ``stop_state``,
  ``halt`` and ``collapse``, declared in the pydantic models.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import RobotManager, StopKind, get_manager  # noqa: E402
from so101.services.schemas import RobotStatus, Telemetry  # noqa: E402
from so101.services.store import get_store  # noqa: E402

OPERATOR = "test-operator"
EVIL = "https://evil.example"


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _armed(manager: RobotManager) -> str:
    lease = manager.acquire_lease("routes", 30)
    manager.arm_control(lease["id"])
    return lease["id"]


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# --- POST /api/robot/halt ----------------------------------------------------------


def test_halt_route_holds_an_energized_arm(client: TestClient):
    _armed(get_manager())

    response = client.post("/api/robot/halt", json={"operator_id": OPERATOR, "reason": "pause"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "detail": "halted: motors on, holding position"}
    status = client.get("/api/robot/status").json()
    assert status["stop_state"] == "halted" and status["halt"]["reason"] == "pause"


def test_halt_route_with_the_motors_off_is_ok(client: TestClient):
    response = client.post("/api/robot/halt")

    assert response.status_code == 200, response.text
    assert "motors are off" in response.json()["detail"]


@pytest.mark.parametrize(
    "headers",
    [{"Origin": EVIL}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}],
    ids=["foreign", "null", "cross-site"],
)
def test_halt_route_is_exempt_from_the_origin_guard(client: TestClient, headers: dict):
    _armed(get_manager())

    response = client.post("/api/robot/halt", headers=headers)

    assert response.status_code == 200, response.text
    assert get_manager().status()["stop_state"] == "halted"


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        ("{not json", "application/json"),
        ("[1, 2]", "application/json"),
        ('{"operator_id": NaN, "reason": Infinity}', "application/json"),
        ('{"operator_id": 7, "reason": ["x"]}', "application/json"),
        ("operator_id=x", "application/x-www-form-urlencoded"),
        ("halt please", "text/plain"),
    ],
    ids=["malformed", "array", "non-finite", "wrong-types", "form", "text"],
)
def test_halt_route_never_refuses_a_stop_over_its_body(
    client: TestClient, content: str, content_type: str
):
    _armed(get_manager())

    response = client.post(
        "/api/robot/halt", content=content, headers={"Content-Type": content_type}
    )

    assert response.status_code == 200, response.text
    status = get_manager().status()
    assert status["stop_state"] == "halted"
    assert status["halt"]["reason"] == "operator halt"


def test_estop_and_stop_ignore_a_non_finite_body(client: TestClient):
    """StrictJSONMiddleware no longer answers 422 before a stop route runs."""

    for path in ("/api/robot/estop", "/api/robot/stop"):
        response = client.post(path, content="NaN", headers={"Content-Type": "application/json"})
        assert response.status_code == 200, (path, response.text)
    assert get_manager().guard.estopped is True


# --- a stop never waits on its body (STOP_PATHS) -------------------------------------

STOP_ROUTES = ["/api/robot/halt", "/api/robot/stop", "/api/robot/estop"]
CHUNK = 64 * 1024
TEN_MB = 10 * 1024 * 1024


async def _post_asgi(app, path: str, chunks, *, headers=(), trickle: bool = False):
    """POST ``path`` straight into the ASGI app; the body comes from ``chunks``.

    Returns (status, seconds until the response started, chunks the app read).
    With ``trickle`` the client sends the first chunk and then nothing more
    until the response is complete (a slow upload).
    """

    import asyncio

    started = time.monotonic()
    responded = asyncio.Event()
    result: dict = {"status": None, "at": None, "read": 0}
    source = iter(chunks)

    pending = next(source, None)

    async def receive():
        nonlocal pending
        if pending is None or (trickle and result["read"] >= 1):
            await responded.wait()  # the client sends nothing more
            return {"type": "http.disconnect"}
        chunk, pending = pending, next(source, None)
        result["read"] += 1
        more = trickle or pending is not None
        return {"type": "http.request", "body": chunk, "more_body": more}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"], result["at"] = message["status"], time.monotonic() - started
        elif message["type"] == "http.response.body" and not message.get("more_body"):
            responded.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"), *headers],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return result["status"], result["at"], result["read"]


def _huge_chunks():
    payload = b'{"operator_id": "' + b"x" * (CHUNK - 18) + b'"'
    for _ in range(TEN_MB // CHUNK):
        yield payload


@pytest.mark.parametrize("path", STOP_ROUTES)
def test_a_10_mb_body_never_delays_a_stop_or_gets_buffered(client: TestClient, path: str):
    import asyncio

    manager = get_manager()
    _armed(manager)

    status, after_s, read = asyncio.run(_post_asgi(client.app, path, _huge_chunks()))

    assert status == 200
    assert after_s < 2.0
    # At most the cap plus one chunk is ever read: memory stays bounded.
    assert read <= 1 + 4096 // CHUNK + 1, f"{read} chunks of {CHUNK} bytes read"
    stopped = manager.status()
    assert stopped["estopped"] if path.endswith("estop") else stopped["stop_state"] == "halted"


@pytest.mark.parametrize("path", STOP_ROUTES)
def test_a_slow_body_never_delays_a_stop(client: TestClient, path: str):
    import asyncio

    manager = get_manager()
    _armed(manager)
    first = [json.dumps({"operator_id": OPERATOR})[:10].encode()]  # and then nothing

    status, after_s, _ = asyncio.run(_post_asgi(client.app, path, first, trickle=True))

    assert status == 200
    assert after_s < 1.0, f"the stop waited {after_s:.2f} s on its body"
    stopped = manager.status()
    if path.endswith("estop"):
        assert stopped["estopped"] is True
    else:
        assert stopped["stop_state"] == "halted"


def test_a_declared_huge_body_is_not_read_at_all(client: TestClient):
    import asyncio

    manager = get_manager()
    _armed(manager)
    headers = [(b"content-length", str(TEN_MB).encode())]

    status, _, read = asyncio.run(
        _post_asgi(client.app, "/api/robot/halt", _huge_chunks(), headers=headers)
    )

    assert status == 200 and read == 0
    assert manager.status()["halt"]["reason"] == "operator halt"


def test_an_unparseably_large_content_length_cannot_prevent_halt(client: TestClient):
    """Python's integer digit cap must not turn a malformed body into a failed stop."""

    import asyncio

    manager = get_manager()
    _armed(manager)
    headers = [(b"content-length", b"9" * 4301)]

    status, _, read = asyncio.run(
        _post_asgi(client.app, "/api/robot/halt", [b"{}"], headers=headers)
    )

    assert status == 200 and read == 0
    assert manager.status()["stop_state"] == "halted"


def test_a_halt_body_over_the_cap_is_ignored_and_the_halt_still_happens(client: TestClient):
    _armed(get_manager())
    body = json.dumps({"operator_id": OPERATOR, "reason": "r" * 5000})

    response = client.post("/api/robot/halt", content=body)

    assert response.status_code == 200, response.text
    assert get_manager().status()["halt"]["reason"] == "operator halt"


def test_a_small_halt_body_is_still_read_in_chunks(client: TestClient):
    import asyncio

    manager = get_manager()
    _armed(manager)
    body = json.dumps({"operator_id": OPERATOR, "reason": "split body"}).encode()

    status, _, read = asyncio.run(
        _post_asgi(client.app, "/api/robot/halt", [body[:7], body[7:20], body[20:]])
    )

    assert status == 200 and read == 3
    assert manager.status()["halt"]["reason"] == "split body"


def test_halt_route_reports_a_failed_hold_as_503(client: TestClient, monkeypatch):
    manager = get_manager()
    _armed(manager)
    manager.follower().bus.fail_io = True

    response = client.post("/api/robot/halt")

    assert response.status_code == 503, response.text
    assert "cut physical power" in response.json()["detail"]


def test_halt_route_never_queues_behind_the_request_workers(client: TestClient):
    from so101.services.routers import robot as robot_router

    assert robot_router._HALT_EXECUTOR._thread_name_prefix == "so101-halt"


# --- POST /api/robot/collapse ----------------------------------------------------------


def test_collapse_route_is_origin_guarded(client: TestClient):
    _armed(get_manager())

    response = client.post("/api/robot/collapse", headers={"Origin": EVIL})

    assert response.status_code == 403
    assert get_manager().status()["stop_state"] == "none"
    assert get_manager().control_state()["armed"] is True


def test_collapse_route_starts_the_descent(client: TestClient):
    manager = get_manager()
    _armed(manager)

    response = client.post("/api/robot/collapse", json={"reason": "end of shift"})

    assert response.status_code == 200, response.text
    assert "collapsing" in response.json()["detail"]
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    events = get_store().list_events(limit=50)
    start = next(event for event in events if event["type"] == "safety.collapse.start")
    assert start["detail"]["reason"] == "end of shift"


def test_collapse_route_validates_its_body(client: TestClient):
    response = client.post("/api/robot/collapse", json={"reason": "x" * 201})
    assert response.status_code == 422
    assert client.post("/api/robot/collapse").status_code == 200  # no body is fine


# --- status and telemetry ---------------------------------------------------------------


def test_status_and_telemetry_validate_against_the_declared_models(client: TestClient):
    manager = get_manager()
    _armed(manager)
    manager.request_stop(StopKind.HALT, reason="model check")

    status = manager.status()
    frame = manager.telemetry()

    assert RobotStatus(**status).backend == "mock"
    Telemetry(**frame)
    assert set(Telemetry.model_fields) >= {
        "observation_live",
        "torque_state",
        "stop_state",
        "halt",
        "collapse",
    }
    json.dumps(frame)  # every frame goes out as plain JSON over the WebSocket
    body = client.get("/api/robot/status").json()
    assert body["torque_state"] == "on" and body["stop_state"] == "halted"
    assert set(body["halt"]) == {"since", "reason", "hold_expires_at", "server_time"}
    assert body["supervision"]["supervised"] is True


def test_every_telemetry_frame_carries_the_stop_fields(client: TestClient):
    with client.websocket_connect("/ws/telemetry") as ws:
        frame = ws.receive_json()

    for key in ("torque_state", "stop_state", "halt", "collapse", "observation_live"):
        assert key in frame
    assert frame["torque_state"] == "off" and frame["stop_state"] == "none"


def test_torque_state_aggregates_unconfirmed_over_on_over_off(client: TestClient):
    manager = get_manager()
    assert manager.status()["torque_state"] == "off"
    _armed(manager)
    assert manager.status()["torque_state"] == "on"
    with manager._lock:
        manager._torque_unconfirmed = True
    assert manager.status()["torque_state"] == "unconfirmed"

"""The arm-first and tracking execution contract (C9).

- The client's sent_at bounds the command that actually executes, also after
  arming (safety-B7, which is sim-B4).
- Execution without sent_at is a 422.
- A null wrist.z holds the elbow at its engage-time value (sim-B5 backend half).
- The audit throttle is keyed on the live lease, as a bounded LRU with a TTL
  (sim-B20).

Reports: docs/audit/v0.1/safety.md and docs/audit/v0.1/sim.md. Converts safety
repros 05(a) and 07(c) and the sim tracking_repro.py.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import get_manager  # noqa: E402
from so101.services.routers import tracking  # noqa: E402
from so101.services.store import get_store  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(tracking, "_AUDIT", tracking._AuditThrottle())
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _iso(offset_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


def _lease(client: TestClient, source: str = "camera") -> str:
    response = client.post("/api/control/lease", json={"source": source, "ttl_seconds": 30})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _frame(lease_id: str, sequence: int, **fields) -> dict:
    body = {
        "source": "camera",
        "confidence": 0.9,
        "execute": True,
        "lease_id": lease_id,
        "sequence": sequence,
        "ttl_ms": 300,
        "deadman": True,
        "intent": {"normalized_x": 0.2, "pinch": 0.1},
    }
    body.update(fields)
    return body


def _command(lease_id: str, sequence: int, sent_at: str) -> dict:
    return {
        "lease_id": lease_id,
        "source": "ui",
        "sequence": sequence,
        "sent_at": sent_at,
        "ttl_ms": 500,
        "deadman": True,
        "targets": {"shoulder_pan": 10.0},
    }


def _slow_connect(monkeypatch, seconds: float) -> list[dict]:
    """Stand-in for real port open, handshake and configure latency."""

    follower = get_manager().follower()
    connect = follower.connect
    calls: list[dict] = []

    def slow(**kwargs):
        calls.append(kwargs)
        time.sleep(seconds)
        return connect(**kwargs)

    monkeypatch.setattr(follower, "connect", slow)
    return calls


def _sent_at_spy(monkeypatch) -> list[datetime]:
    manager = get_manager()
    send = manager.send_external_command
    seen: list[datetime] = []

    def spy(**kwargs):
        seen.append(kwargs["sent_at"])
        return send(**kwargs)

    monkeypatch.setattr(manager, "send_external_command", spy)
    return seen


def _park_elbow(degrees: float) -> None:
    bus = get_manager().follower().bus
    bus._pos["elbow_flex"] = degrees
    bus._goal["elbow_flex"] = degrees


def _tracking_events() -> list[dict]:
    return [e for e in get_store().list_events(limit=500) if e["type"].startswith("tracking.")]


# --- safety-B7 / sim-B4: the client's timestamp survives arming -----------


@pytest.mark.parametrize("path", ["/api/control/arm-command", "/api/tracking/arm-execute"])
def test_safety_b7_repro_05a_a_frame_that_went_stale_while_arming_never_executes(
    client: TestClient, monkeypatch, path: str
):
    manual = path.endswith("arm-command")
    lease_id = _lease(client, "ui" if manual else "camera")
    connects = _slow_connect(monkeypatch, 0.6)
    sent_at = _iso(-0.1)
    body = (
        _command(lease_id, 1, sent_at)
        if manual
        else _frame(lease_id, 1, sent_at=sent_at, ttl_ms=500)
    )

    response = client.post(path, json=body)

    assert connects, "the frame was fresh when the request arrived, so arming started"
    assert response.status_code == 409, response.text
    assert "stale" in response.json()["detail"]
    assert client.get("/api/control/state").json()["armed"] is False
    # The owner-scoped cleanup is a Halt (design doc §5): the arm holds where
    # arming found it; the stale frame never moved it.
    status = get_manager().status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"


def test_safety_b7_arm_command_hands_the_client_timestamp_to_the_command(
    client: TestClient, monkeypatch
):
    seen = _sent_at_spy(monkeypatch)
    lease_id = _lease(client, "ui")
    sent_at = datetime.now(timezone.utc) - timedelta(milliseconds=50)

    response = client.post(
        "/api/control/arm-command", json=_command(lease_id, 1, sent_at.isoformat())
    )

    assert response.status_code == 200, response.text
    assert seen == [sent_at]


def test_safety_b7_tracked_frames_hand_the_client_timestamp_to_the_command(
    client: TestClient, monkeypatch
):
    seen = _sent_at_spy(monkeypatch)
    lease_id = _lease(client)
    first = datetime.now(timezone.utc) - timedelta(milliseconds=40)

    armed = client.post(
        "/api/tracking/arm-execute", json=_frame(lease_id, 1, sent_at=first.isoformat())
    )
    second = datetime.now(timezone.utc) - timedelta(milliseconds=20)
    tracked = client.post(
        "/api/tracking/preview", json=_frame(lease_id, 2, sent_at=second.isoformat())
    )

    assert armed.status_code == 200, armed.text
    assert tracked.status_code == 200, tracked.text
    assert seen == [first, second]


# --- C9: sent_at is required to execute ------------------------------------


@pytest.mark.parametrize(
    "path, execute",
    [
        ("/api/tracking/preview", True),
        ("/api/tracking/arm-execute", True),
        ("/api/tracking/arm-execute", False),
    ],
)
def test_c9_tracking_execution_without_sent_at_is_422(client: TestClient, path: str, execute: bool):
    lease_id = _lease(client)
    if path.endswith("preview"):
        armed = client.post("/api/tracking/arm-execute", json=_frame(lease_id, 1, sent_at=_iso()))
        assert armed.status_code == 200, armed.text

    response = client.post(path, json=_frame(lease_id, 2, execute=execute))

    assert response.status_code == 422, response.text
    assert "sent_at" in response.json()["detail"]
    if not path.endswith("preview"):
        assert client.get("/api/control/state").json()["armed"] is False


def test_c9_sim_repro_a_stale_frame_without_sent_at_cannot_slip_through(client: TestClient):
    lease_id = _lease(client)
    armed = client.post("/api/tracking/arm-execute", json=_frame(lease_id, 1, sent_at=_iso()))
    assert armed.status_code == 200, armed.text

    stale = client.post("/api/tracking/preview", json=_frame(lease_id, 2, sent_at=_iso(-10)))
    omitted = client.post("/api/tracking/preview", json=_frame(lease_id, 3))

    assert stale.status_code == 409 and "stale" in stale.json()["detail"]
    assert omitted.status_code == 422


def test_c9_a_low_confidence_execute_frame_still_needs_sent_at(client: TestClient):
    lease_id = _lease(client)

    response = client.post("/api/tracking/preview", json=_frame(lease_id, 1, confidence=0.1))

    assert response.status_code == 422


def test_c9_preview_without_execution_needs_no_sent_at(client: TestClient):
    response = client.post(
        "/api/tracking/preview", json={"source": "camera", "confidence": 0.9, "wrist": {"x": 0.2}}
    )

    assert response.status_code == 200, response.text
    assert response.json()["accepted"] is True


# --- C9 / sim-B5: null depth holds the elbow ---------------------------------


@pytest.mark.parametrize("wrist", [{"x": 0.2, "z": None}, [0.2, 0.0, None]], ids=["dict", "list"])
def test_c9_null_wrist_z_holds_the_elbow_at_its_engage_value(client: TestClient, wrist):
    _park_elbow(40.0)
    lease_id = _lease(client)

    first = client.post(
        "/api/tracking/arm-execute",
        json=_frame(lease_id, 1, sent_at=_iso(), intent={}, wrist=wrist),
    )
    following = client.post(
        "/api/tracking/preview", json=_frame(lease_id, 2, sent_at=_iso(), intent={}, wrist=wrist)
    )

    for response in (first, following):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["desired_targets"]["elbow_flex"] == pytest.approx(40.0)
        assert body["commanded_targets"]["elbow_flex"] == pytest.approx(40.0)


def test_c9_null_z_returns_to_the_engage_value_not_the_last_command(client: TestClient):
    _park_elbow(40.0)
    lease_id = _lease(client)
    armed = client.post(
        "/api/tracking/arm-execute",
        json=_frame(lease_id, 1, sent_at=_iso(), intent={}, wrist={"x": 0.2, "z": None}),
    )
    assert armed.status_code == 200, armed.text
    moved = client.post(
        "/api/tracking/preview",
        json=_frame(lease_id, 2, sent_at=_iso(), intent={}, wrist={"x": 0.2, "z": 0.0}),
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["commanded_targets"]["elbow_flex"] < 40.0

    held = client.post(
        "/api/tracking/preview",
        json=_frame(lease_id, 3, sent_at=_iso(), intent={}, wrist={"x": 0.2, "z": None}),
    )

    assert held.status_code == 200, held.text
    assert held.json()["desired_targets"]["elbow_flex"] == pytest.approx(40.0)


def test_c9_null_z_without_an_armed_session_targets_no_elbow(client: TestClient):
    response = client.post(
        "/api/tracking/preview",
        json={"source": "camera", "confidence": 0.9, "wrist": {"x": 0.2, "z": None}},
    )

    assert response.status_code == 200, response.text
    assert "elbow_flex" not in response.json()["desired_targets"]


# --- sim-B20: the audit throttle ----------------------------------------------


def test_sim_b20_rotating_sources_cannot_flood_the_audit_log(client: TestClient):
    before = len(_tracking_events())

    for index in range(300):
        client.post("/api/tracking/preview", json={"source": f"s{index}", "confidence": 0.1})

    assert len(_tracking_events()) - before <= 1


def test_sim_b20_the_audit_throttle_is_keyed_on_the_live_lease(client: TestClient):
    def low_confidence(lease_id: str) -> None:
        response = client.post(
            "/api/tracking/preview",
            json={"source": "camera", "confidence": 0.1, "lease_id": lease_id},
        )
        assert response.status_code == 200, response.text

    before = len(_tracking_events())
    first = _lease(client)
    low_confidence(first)
    low_confidence(first)
    assert len(_tracking_events()) - before == 1

    assert client.delete(f"/api/control/lease/{first}").status_code == 200
    low_confidence(_lease(client))
    assert len(_tracking_events()) - before == 2

    # Made-up lease ids share the caller's bucket instead of minting new keys.
    low_confidence("made-up-1")
    low_confidence("made-up-2")
    assert len(_tracking_events()) - before == 3


def test_sim_b20_the_throttle_is_a_bounded_lru_with_a_ttl():
    now = [0.0]
    throttle = tracking._AuditThrottle(window_s=5.0, ttl_s=60.0, capacity=2, clock=lambda: now[0])

    assert throttle.should_audit("a")
    assert throttle.should_audit("b")
    assert not throttle.should_audit("a")  # inside its window
    assert throttle.should_audit("c")  # full: the least recent entry, "a", goes
    assert throttle.should_audit("a")  # so "a" counts as new; "b" goes
    assert not throttle.should_audit("c")
    assert len(throttle) == 2

    now[0] = 61.0
    assert throttle.should_audit("c")  # both entries outlived the TTL
    assert len(throttle) == 1
    now[0] = 63.0
    assert not throttle.should_audit("c")
    assert throttle.should_audit("c", force=True)
    now[0] = 68.5
    assert throttle.should_audit("c")

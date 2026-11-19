"""C4 lease secrecy (safety-B5, docs/audit/v0.1/safety.md).

The lease id is the only credential for arm, observe and command, so only the
client that created the lease ever sees it: POST /api/control/lease and the
lease heartbeat return it, and nothing else does. Events carry a short,
non-reversible lease_tag instead. Converts safety repro 10.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import get_manager, lease_tag  # noqa: E402

PUBLIC_LEASE_KEYS = {"source", "expires_at", "ttl_seconds", "server_time"}


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _command(lease_id: str, source: str, sequence: int, pan: float) -> dict:
    return {
        "lease_id": lease_id,
        "source": source,
        "sequence": sequence,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "ttl_ms": 500,
        "deadman": True,
        "targets": {"shoulder_pan": pan},
    }


def _lease_events(client: TestClient) -> list[dict]:
    events = client.get("/api/events").json()["events"]
    return [event for event in events if event["type"].startswith("control.")]


def test_safety_b5_repro_10_no_reader_can_copy_the_lease_id(client: TestClient):
    lease = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 30})
    assert lease.status_code == 201, lease.text
    lease_id = lease.json()["id"]
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    first = client.post("/api/control/command", json=_command(lease_id, "ui", 1, 2.0))
    assert first.status_code == 200, first.text

    # A second tab or script reads everything an unauthenticated reader can.
    state = client.get("/api/control/state").json()
    assert set(state["lease"]) == PUBLIC_LEASE_KEYS
    assert state["lease"]["source"] == "ui"
    for path in ("/api/control/state", "/api/robot/status", "/api/events", "/api/diagnostics"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert lease_id not in response.text, f"{path} leaks the lease id"

    # The repro's hijack needs the id; a guessed one gets nowhere.
    hijack = client.post("/api/control/command", json=_command("copied", "ui", 10**12, 15.0))
    assert hijack.status_code == 409
    # And the owner keeps control.
    owner = client.post("/api/control/command", json=_command(lease_id, "ui", 2, 2.0))
    assert owner.status_code == 200, owner.text


def test_c4_state_lease_has_exactly_the_public_fields_and_no_id(client: TestClient):
    lease_id = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5}).json()[
        "id"
    ]

    lease = client.get("/api/control/state").json()["lease"]

    assert set(lease) == PUBLIC_LEASE_KEYS
    assert lease["ttl_seconds"] == 5
    assert lease_id not in json.dumps(get_manager().control_state(), default=str)


def test_c4_arm_and_disarm_answers_do_not_echo_the_id(client: TestClient):
    lease_id = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5}).json()[
        "id"
    ]

    armed = client.post("/api/control/arm", json={"lease_id": lease_id})
    disarmed = client.post("/api/control/disarm", json={"lease_id": lease_id})

    for response in (armed, disarmed):
        assert response.status_code == 200, response.text
        assert lease_id not in response.text
        assert set(response.json()["lease"]) == PUBLIC_LEASE_KEYS


def test_c4_only_create_and_heartbeat_return_the_id(client: TestClient):
    created = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5})
    assert created.status_code == 201
    body = created.json()
    assert set(body) == PUBLIC_LEASE_KEYS | {"id"}

    heartbeat = client.post(f"/api/control/lease/{body['id']}/heartbeat")

    assert heartbeat.status_code == 200, heartbeat.text
    assert heartbeat.json()["id"] == body["id"]
    assert set(heartbeat.json()) == PUBLIC_LEASE_KEYS | {"id"}


def test_c4_events_carry_a_lease_tag_and_never_the_id(client: TestClient):
    lease_id = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5}).json()[
        "id"
    ]
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    assert client.delete(f"/api/control/lease/{lease_id}").status_code == 200

    events = _lease_events(client)
    by_type = {event["type"]: event for event in events}
    for kind in ("control.lease_acquired", "control.armed", "control.lease_released"):
        assert by_type[kind]["detail"]["lease_tag"] == lease_tag(lease_id), kind
    assert lease_id not in json.dumps(events)
    assert all("lease_id" not in event["detail"] for event in events)


def test_c4_an_expired_lease_is_reported_by_tag(client: TestClient):
    manager = get_manager()
    lease_id = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 5}).json()[
        "id"
    ]
    with manager._lock:
        assert manager._lease is not None
        manager._lease.deadline = time.monotonic() - 1
    manager._watchdog_tick()

    assert client.get("/api/control/state").json()["safety_state"] == "lease_expired"

    expired = [e for e in _lease_events(client) if e["type"] == "control.lease_expired"]
    assert len(expired) == 1
    assert expired[0]["detail"] == {"lease_tag": lease_tag(lease_id)}


def test_c4_the_tag_is_short_stable_and_not_part_of_the_id():
    first, second = "f084724e88684b6cac7328815e6afacf", "0" * 32

    tag = lease_tag(first)

    assert tag == lease_tag(first)
    assert len(tag) == 12 and int(tag, 16) >= 0
    assert tag not in first
    assert lease_tag(second) != tag

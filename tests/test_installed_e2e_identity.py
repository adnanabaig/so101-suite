"""Ephemeral installed-E2E identity binds writes to one non-real server."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from so101 import __version__, config
from so101.services import app as app_module
from so101.services.app import create_app
from so101.services.guard import E2E_REJECTED_DETAIL, E2E_REQUIRED_DETAIL
from so101.services.robot_manager import get_manager, reset_manager

TOKEN = "test-only-ephemeral-token"
HEADER = "X-SO101-E2E-Token"


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _set_token(monkeypatch, token: str = TOKEN, *, backend: str = "mock") -> None:
    monkeypatch.setenv("SO101_E2E_TOKEN", token)
    monkeypatch.setenv("SO101_BACKEND", backend)
    monkeypatch.setenv("SO101_MOCK", "1" if backend == "mock" else "0")
    config.get_settings.cache_clear()
    reset_manager()


def test_token_unset_has_no_effect_on_production_requests(client):
    identity = client.get("/api/e2e/identity")
    assert identity.status_code == 404
    assert identity.json() == {"detail": "Not Found"}
    heartbeat = client.post(
        "/api/robot/operator-heartbeat",
        json={"operator_id": "ordinary-operator"},
    )
    assert heartbeat.status_code == 200


def test_identity_route_disables_fastapi_response_model_construction():
    route = next(route for route in create_app().routes if route.path == "/api/e2e/identity")
    assert route.response_model is None


def test_any_e2e_header_is_rejected_when_the_token_is_unset(client):
    response = client.post(
        "/api/robot/operator-heartbeat",
        headers={HEADER: TOKEN},
        json={"operator_id": "unexpected-test-client"},
    )
    assert response.status_code == 403
    assert response.json() == {"detail": E2E_REJECTED_DETAIL}
    assert get_manager().guard.estopped is False


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_configured_e2e_rejects_every_tokenless_mutation(monkeypatch, backend):
    _set_token(monkeypatch, backend=backend)
    client = TestClient(create_app())
    heartbeat = client.post(
        "/api/robot/operator-heartbeat",
        json={"operator_id": "tokenless"},
    )
    assert heartbeat.status_code == 403
    assert heartbeat.json() == {"detail": E2E_REQUIRED_DETAIL}


@pytest.mark.parametrize("path", ["/api/robot/estop", "/api/robot/stop", "/api/robot/halt"])
@pytest.mark.parametrize(
    "headers",
    [{}, {HEADER: "wrong"}],
    ids=["tokenless", "wrong-token"],
)
def test_configured_e2e_never_blocks_a_stop(monkeypatch, path, headers):
    _set_token(monkeypatch)
    with TestClient(create_app()) as client:
        response = client.post(path, headers=headers)
    assert response.status_code == 200, response.text


def test_identity_requires_the_one_configured_token(monkeypatch):
    _set_token(monkeypatch)
    with TestClient(create_app()) as client:
        missing = client.get("/api/e2e/identity")
        wrong = client.get("/api/e2e/identity", headers={HEADER: "wrong"})
        duplicate = client.get(
            "/api/e2e/identity",
            headers=[(HEADER, TOKEN), (HEADER, TOKEN)],
        )
        assert missing.status_code == 403
        assert missing.json() == {"detail": E2E_REQUIRED_DETAIL}
        for rejected in (wrong, duplicate):
            assert rejected.status_code == 403
            assert rejected.json() == {"detail": E2E_REJECTED_DETAIL}

        accepted = client.get("/api/e2e/identity", headers={HEADER: TOKEN})
        assert accepted.status_code == 200
        assert accepted.json() == {
            "ok": True,
            "kind": "so101.installed_e2e",
            "schema_version": 1,
            "backend": "mock",
            "mock": True,
            "version": __version__,
        }
        tokenless = client.post(
            "/api/robot/operator-heartbeat",
            json={"operator_id": "missing-token"},
        )
        assert tokenless.status_code == 403
        assert tokenless.json() == {"detail": E2E_REQUIRED_DETAIL}
        heartbeat = client.post(
            "/api/robot/operator-heartbeat",
            headers={HEADER: TOKEN},
            json={"operator_id": "e2e"},
        )
        assert heartbeat.status_code == 200


def test_identity_is_side_effect_free_and_does_not_initialize_the_robot(monkeypatch):
    _set_token(monkeypatch)

    def opened_robot():
        raise AssertionError("identity endpoint initialized the robot manager")

    monkeypatch.setattr(app_module, "get_manager", opened_robot)
    client = TestClient(create_app())  # no lifespan; identity itself must stay read-only
    response = client.get("/api/e2e/identity", headers={HEADER: TOKEN})
    assert response.status_code == 200
    assert response.json()["kind"] == "so101.installed_e2e"


def test_token_capability_identifies_the_sim_backend(monkeypatch):
    _set_token(monkeypatch, backend="sim")
    client = TestClient(create_app())  # middleware rejects before lifespan/device access
    identity = client.get("/api/e2e/identity", headers={HEADER: TOKEN})
    assert identity.status_code == 200
    assert identity.json() == {
        "ok": True,
        "kind": "so101.installed_e2e",
        "schema_version": 1,
        "backend": "sim",
        "mock": False,
        "version": __version__,
    }


def test_e2e_token_configuration_is_refused_on_the_real_backend(monkeypatch):
    _set_token(monkeypatch, backend="real")
    with pytest.raises(ValidationError, match="SO101_E2E_TOKEN"):
        config.get_settings()


def test_wrong_token_rejects_websocket_before_accept(monkeypatch):
    _set_token(monkeypatch)
    with TestClient(create_app()) as client:
        with pytest.raises(WebSocketDisconnect) as stopped:
            with client.websocket_connect("/ws/telemetry", headers={HEADER: "wrong"}) as ws:
                ws.receive_json()
    assert stopped.value.code == 1008


def test_token_never_appears_in_settings_responses_or_guard_logs(monkeypatch, caplog):
    _set_token(monkeypatch)
    settings = config.get_settings()
    assert TOKEN not in repr(settings)
    assert TOKEN not in str(settings.model_dump())
    with caplog.at_level(logging.WARNING, logger="so101.services.guard"):
        with TestClient(create_app()) as client:
            response = client.get("/api/e2e/identity", headers={HEADER: TOKEN + "-wrong"})
    assert response.status_code == 403
    assert TOKEN not in response.text
    assert TOKEN not in caplog.text

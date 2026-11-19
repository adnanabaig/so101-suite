"""Non-finite and overflowing numbers in any request body return 422, never 500.

safety-B11 (docs/audit/v0.1/safety.md, safety repros 07/07b) and sim-B19
(docs/audit/v0.1/sim.md). Endpoints from several owners are covered on purpose:
the fix is global (strict JSON parsing plus exception handlers), not per route.
"""

from __future__ import annotations

import inspect
import json
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from so101.services.app import create_app
from so101.services.strict_json import (
    JSON_BODY_MAX_BYTES,
    JSON_BODY_TOO_LARGE_MSG,
    NON_FINITE_MSG,
    StrictJSONMiddleware,
    install_numeric_error_handlers,
)

JSON = {"content-type": "application/json"}
COMMAND = '{"lease_id":"x","source":"ui","sequence":1,"ttl_ms":500,"deadman":true,'

CMD = "/api/control/command"
TARGET0 = '"targets":{"shoulder_pan":0}}'
EVAL = '{"model_path":"m","operator_id":"op","max_seconds":'
CASES = {
    "command-sent_at-1e300": (CMD, COMMAND + '"sent_at":1e300,' + TARGET0),
    "command-sent_at--1e300": (CMD, COMMAND + '"sent_at":-1e300,' + TARGET0),
    "command-sent_at-Infinity": (CMD, COMMAND + '"sent_at":Infinity,' + TARGET0),
    "command-sent_at-NaN": (CMD, COMMAND + '"sent_at":NaN,' + TARGET0),
    "command-target-NaN": (CMD, COMMAND + '"sent_at":1,"targets":{"shoulder_pan":NaN}}'),
    "command-target-1e309": (CMD, COMMAND + '"sent_at":1,"targets":{"shoulder_pan":1e309}}'),
    "command-bare-NaN": (CMD, "NaN"),
    "heartbeat-unknown-field-NaN": (
        "/api/robot/operator-heartbeat",
        '{"operator_id":"op","x":NaN}',
    ),
    "tracking-confidence-NaN": (
        "/api/tracking/preview",
        '{"source":"camera","confidence":NaN,"intent":{"normalized_x":0.5}}',
    ),
    "sim-target-NaN": ("/api/simulation/targets", '{"targets":{"shoulder_pan":NaN}}'),
    "sim-play-rate-Infinity": ("/api/simulation/play", '{"rate":Infinity}'),
    "sim-step--Infinity": ("/api/simulation/step", '{"steps":1,"targets":{"gripper":-Infinity}}'),
    "eval-start-Infinity": ("/api/eval/start", EVAL + "Infinity}"),
    "eval-start-huge-int": ("/api/eval/start", EVAL + "9" * 400 + "}"),
    "agent-constraint-NaN": (
        "/api/agent/proposals",
        '{"objective":"open the gripper","constraints":{"max_duration_seconds":NaN}}',
    ),
}


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.mark.parametrize(("path", "body"), list(CASES.values()), ids=list(CASES))
def test_safety_b11_sim_b19_non_finite_bodies_are_422(client, path, body):
    response = client.post(path, content=body, headers=JSON)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "content_type",
    ["application/json; charset=utf-8", "application/merge-patch+json", "Application/JSON"],
)
def test_sim_b19_every_json_content_type_is_strict(client, content_type):
    response = client.post(
        "/api/simulation/targets",
        content='{"targets":{"shoulder_pan":NaN}}',
        headers={"content-type": content_type},
    )
    assert response.status_code == 422, response.text


def test_safety_b11_non_finite_detail_keeps_validation_shape(client):
    response = client.post(
        "/api/simulation/targets", content='{"targets":{"shoulder_pan":NaN}}', headers=JSON
    )
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert detail[0]["type"] == "json_invalid"
    assert detail[0]["loc"] == ["body"]
    assert "non-finite" in detail[0]["msg"]


def test_safety_b11_overflow_error_handler_returns_422(client):
    """sent_at=1e300 is finite JSON; datetime.fromtimestamp raises OverflowError."""
    response = client.post(
        "/api/control/command",
        content=COMMAND + '"sent_at":1e300,"targets":{"shoulder_pan":0}}',
        headers=JSON,
    )
    assert response.status_code == 422
    assert "out of range" in str(response.json()["detail"])


def test_valid_json_bodies_still_reach_handlers(client):
    response = client.post("/api/robot/operator-heartbeat", json={"operator_id": "op"})
    assert response.status_code == 200
    response = client.post(
        "/api/simulation/targets", json={"targets": {"shoulder_pan": 1.5, "gripper": 40}}
    )
    assert response.status_code == 200, response.text


def test_malformed_json_still_reports_json_invalid(client):
    response = client.post(
        "/api/robot/operator-heartbeat", content='{"operator_id": ', headers=JSON
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


def test_validation_errors_never_echo_non_finite_floats():
    """Defense in depth: a validation error whose input holds inf still renders."""
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError

    from so101.services.strict_json import install_numeric_error_handlers

    app = FastAPI()
    install_numeric_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise RequestValidationError(
            [{"type": "finite_number", "loc": ("body", "x"), "msg": "bad", "input": float("inf")}]
        )

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 422
    assert response.json()["detail"][0]["input"] == "inf"


# --- bodies with no Content-Type (gate review) -------------------------------------


class _Point(BaseModel):
    x: float


def _lenient_app() -> FastAPI:
    """An app that parses a body with no Content-Type as JSON.

    FastAPI does that when strict_content_type is off. Versions without that
    setting always do, and pyproject still allows them.
    """
    lenient = (
        {"strict_content_type": False}
        if "strict_content_type" in inspect.signature(FastAPI).parameters
        else {}
    )
    app = FastAPI(**lenient)
    app.add_middleware(StrictJSONMiddleware)
    install_numeric_error_handlers(app)

    @app.post("/point")
    def point(body: _Point) -> dict:
        return {"finite": math.isfinite(body.x)}

    return app


@pytest.mark.parametrize("body", ['{"x":NaN}', '{"x":-Infinity}', '{"x":1e309}', "NaN"])
def test_a_body_with_no_content_type_is_scanned_too(body):
    """Gate review: only declared JSON bodies were scanned, so a non-finite
    body sent with no Content-Type reached a lenient FastAPI app as NaN."""
    response = TestClient(_lenient_app()).post("/point", content=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["msg"] == NON_FINITE_MSG


def test_a_body_with_no_content_type_is_otherwise_untouched():
    client = TestClient(_lenient_app())
    assert client.post("/point", content='{"x":1.5}').json() == {"finite": True}
    not_json = client.post("/point", content="x=1")
    assert not_json.status_code == 422
    assert not_json.json()["detail"][0]["msg"] != NON_FINITE_MSG  # FastAPI's own error


def test_the_app_scans_a_body_with_no_content_type(client):
    response = client.post("/api/simulation/targets", content='{"targets":{"shoulder_pan":NaN}}')
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["msg"] == NON_FINITE_MSG


# --- bounded middleware reads ------------------------------------------------------


def _asgi_scope(*headers: tuple[bytes, bytes]) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/point",
        "raw_path": b"/point",
        "query_string": b"",
        "root_path": "",
        "headers": list(headers),
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }


async def _call_asgi(app, messages, *, headers):
    sent = []
    consumed = 0

    async def receive():
        nonlocal consumed
        index = consumed
        consumed += 1
        if index < len(messages):
            return messages[index]
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(_asgi_scope(*headers), receive, send)
    return consumed, sent


def _response_status(messages) -> int:
    return next(
        message["status"] for message in messages if message["type"] == "http.response.start"
    )


def _response_json(messages):
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return json.loads(body)


async def test_oversized_declared_json_body_is_rejected_without_reading_it():
    async def downstream(_scope, _receive, _send):
        raise AssertionError("oversized request reached the downstream app")

    middleware = StrictJSONMiddleware(downstream)
    consumed, sent = await _call_asgi(
        middleware,
        [],
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", str(JSON_BODY_MAX_BYTES + 1).encode()),
        ],
    )

    assert consumed == 0
    assert _response_status(sent) == 413
    assert _response_json(sent) == {"detail": JSON_BODY_TOO_LARGE_MSG}


async def test_chunked_json_body_stops_reading_at_the_cumulative_limit():
    async def downstream(_scope, _receive, _send):
        raise AssertionError("oversized request reached the downstream app")

    messages = [
        {"type": "http.request", "body": b"x" * JSON_BODY_MAX_BYTES, "more_body": True},
        {"type": "http.request", "body": b"y", "more_body": True},
        {"type": "http.request", "body": b"unread", "more_body": False},
    ]

    consumed, sent = await _call_asgi(
        StrictJSONMiddleware(downstream),
        messages,
        headers=[(b"content-type", b"application/json")],
    )

    assert consumed == 2
    assert _response_status(sent) == 413
    assert _response_json(sent) == {"detail": JSON_BODY_TOO_LARGE_MSG}


async def test_json_body_exactly_at_the_limit_is_replayed():
    prefix = b'{"padding":"'
    suffix = b'"}'
    payload = prefix + b"x" * (JSON_BODY_MAX_BYTES - len(prefix) - len(suffix)) + suffix
    assert len(payload) == JSON_BODY_MAX_BYTES
    json.loads(payload)  # The boundary fixture itself is valid JSON.
    seen = []

    async def downstream(_scope, receive, send):
        seen.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    split = JSON_BODY_MAX_BYTES // 2
    messages = [
        {"type": "http.request", "body": payload[:split], "more_body": True},
        {"type": "http.request", "body": payload[split:], "more_body": False},
    ]

    consumed, sent = await _call_asgi(
        StrictJSONMiddleware(downstream),
        messages,
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", str(JSON_BODY_MAX_BYTES).encode()),
        ],
    )

    assert consumed == 2
    assert _response_status(sent) == 204
    assert seen == [{"type": "http.request", "body": payload, "more_body": False}]


async def test_fragmented_non_finite_json_is_still_rejected():
    async def downstream(_scope, _receive, _send):
        raise AssertionError("non-finite JSON reached the downstream app")

    messages = [
        {"type": "http.request", "body": b'{"x":Na', "more_body": True},
        {"type": "http.request", "body": b"N}", "more_body": False},
    ]

    consumed, sent = await _call_asgi(
        StrictJSONMiddleware(downstream),
        messages,
        headers=[(b"content-type", b"application/json")],
    )

    assert consumed == 2
    assert _response_status(sent) == 422
    assert _response_json(sent)["detail"][0]["msg"] == NON_FINITE_MSG


async def test_non_json_body_and_chunking_are_unchanged():
    messages = [
        {"type": "http.request", "body": b"x" * JSON_BODY_MAX_BYTES, "more_body": True},
        {"type": "http.request", "body": b"y", "more_body": False},
    ]
    seen = []

    async def downstream(_scope, receive, send):
        seen.append(await receive())
        seen.append(await receive())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    consumed, sent = await _call_asgi(
        StrictJSONMiddleware(downstream),
        messages,
        headers=[
            (b"content-type", b"text/plain"),
            (b"content-length", str(JSON_BODY_MAX_BYTES + 1).encode()),
        ],
    )

    assert consumed == 2
    assert _response_status(sent) == 204
    assert seen == messages

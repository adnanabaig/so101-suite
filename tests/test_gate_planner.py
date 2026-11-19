"""Planner fixes sim-B12 to sim-B15 (docs/audit/v0.1/sim.md)."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.agents.planner import (  # noqa: E402
    DeterministicPlanner,
    OpenAIResponsesPlanner,
    PlannerResponseError,
    planner_status,
    route_instruction,
)
from so101.config import get_settings  # noqa: E402
from so101.services.app import create_app  # noqa: E402

TOOL = OpenAIResponsesPlanner.TOOL_NAME
UNSUPPORTED_KEYWORDS = {
    "maxLength",
    "minLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}


def _call(skill="open_gripper", rationale="Named release.", duration=3) -> dict:
    return {
        "type": "function_call",
        "name": TOOL,
        "call_id": "call_1",
        "arguments": json.dumps(
            {"skill": skill, "rationale": rationale, "max_duration_seconds": duration}
        ),
    }


def _planner(handler) -> OpenAIResponsesPlanner:
    return OpenAIResponsesPlanner(
        api_key="test-key", model="test-model", transport=httpx.MockTransport(handler)
    )


def _replying(body, *, status_code=200) -> OpenAIResponsesPlanner:
    def handler(_request: httpx.Request) -> httpx.Response:
        if isinstance(body, bytes):
            return httpx.Response(status_code, content=body)
        return httpx.Response(status_code, json=body)

    return _planner(handler)


def _keywords(schema) -> set[str]:
    found: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            found.add(key)
            found |= _keywords(value)
    elif isinstance(schema, list):
        for value in schema:
            found |= _keywords(value)
    return found


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


# --- sim-B12: strict schema and request flags ---------------------------------------


def test_sim_b12_strict_tool_schema_uses_only_supported_keywords():
    tool = OpenAIResponsesPlanner._tool()
    parameters = tool["parameters"]

    assert tool["strict"] is True
    assert not _keywords(parameters) & UNSUPPORTED_KEYWORDS
    assert parameters["additionalProperties"] is False
    assert sorted(parameters["required"]) == sorted(parameters["properties"])


def test_sim_b12_request_allows_one_call_and_is_not_stored():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "output": [_call()]})

    proposal = _planner(handler).propose("open the gripper", {})

    assert proposal.skill == "open_gripper"
    assert captured["parallel_tool_calls"] is False
    assert captured["store"] is False
    assert captured["tool_choice"] == {"type": "function", "name": TOOL}


def test_sim_b12_a_long_rationale_is_truncated_not_rejected():
    planner = _replying({"status": "completed", "output": [_call(rationale="r" * 800)]})

    proposal = planner.propose("open the gripper", {})

    assert proposal.rationale == "r" * 500


# --- sim-B13: "configured" needs the OpenAI provider ---------------------------------


@pytest.mark.parametrize(
    ("provider", "key", "model", "available"),
    [
        ("deterministic", "test-key", "test-model", False),
        ("openai_responses", "", "test-model", False),
        ("openai_responses", "test-key", "", False),
        ("openai_responses", "test-key", "test-model", True),
    ],
)
def test_sim_b13_configured_planner_is_offered_only_when_it_can_run(
    monkeypatch, client, provider, key, model, available
):
    monkeypatch.setenv("SO101_AGENT_PROVIDER", provider)
    monkeypatch.setenv("SO101_OPENAI_API_KEY", key)
    monkeypatch.setenv("SO101_AGENT_MODEL", model)
    get_settings.cache_clear()

    status = planner_status()
    served = client.get("/api/agent/proposals/planner/status").json()

    assert status["configured_provider_available"] is available
    assert served["configured_provider_available"] is available
    assert served["configured_provider"] == ("openai_responses" if available else "none")


# --- sim-B14: malformed responses are a clear 502 ----------------------------------

MALFORMED = {
    "list body": [_call()],
    "string body": "completed",
    "output not a list": {"status": "completed", "output": {"type": "function_call"}},
    "item not an object": {"status": "completed", "output": ["function_call"]},
    "incomplete status": {"status": "incomplete", "output": [_call()]},
    "missing status": {"output": [_call()]},
    "no function call": {"status": "completed", "output": []},
    "two function calls": {"status": "completed", "output": [_call(), _call()]},
    "arguments not an object": {
        "status": "completed",
        "output": [{"type": "function_call", "name": TOOL, "arguments": "[1, 2]"}],
    },
    "unknown skill": {"status": "completed", "output": [_call(skill="raw_joint_write")]},
    "duration out of range": {"status": "completed", "output": [_call(duration=90)]},
    "not JSON": b"<html>gateway error</html>",
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_sim_b14_malformed_responses_raise_a_response_error(case):
    with pytest.raises(PlannerResponseError, match="planner"):
        _replying(MALFORMED[case]).propose("open the gripper", {})


@pytest.mark.parametrize("case", ["list body", "item not an object", "incomplete status"])
def test_sim_b14_malformed_responses_are_502_not_500(monkeypatch, client, case):
    planner = _replying(MALFORMED[case])
    monkeypatch.setattr("so101.services.routers.agent.get_planner", lambda _selection: planner)

    response = client.post(
        "/api/agent/proposals", json={"objective": "open the gripper", "planner": "configured"}
    )

    assert response.status_code == 502, response.text
    assert "malformed" in response.json()["detail"]
    assert client.get("/api/agent/proposals").json() == []


def test_sim_b14_an_unreachable_planner_is_still_503(monkeypatch, client):
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    planner = _planner(unreachable)
    monkeypatch.setattr("so101.services.routers.agent.get_planner", lambda _selection: planner)

    response = client.post(
        "/api/agent/proposals", json={"objective": "open the gripper", "planner": "configured"}
    )

    assert response.status_code == 503
    assert "request failed" in response.json()["detail"]


# --- sim-B15: whole-word routing with a negation guard -------------------------------

ROUTES = {
    "open the gripper": "open_gripper",
    "Please close the gripper.": "close_gripper",
    "go home": "home",
    "park the arm": "home",
    "move to neutral": "home",
    "wave hello": "wrist_wave",
    "open the gripper's jaws": "open_gripper",
    "don't open the gripper": "inspect_scene",
    "Don’t open the gripper": "inspect_scene",
    "do not close the gripper": "inspect_scene",
    "never wave": "inspect_scene",
    "no home move": "inspect_scene",
    "spark": "inspect_scene",
    "check the homework": "inspect_scene",
    "the gripper keeps reopening": "inspect_scene",
    "open and close the gripper": "inspect_scene",
    "open the door": "inspect_scene",
    "inspect the workspace": "inspect_scene",
}


@pytest.mark.parametrize("instruction", sorted(ROUTES))
def test_sim_b15_routing_matches_whole_words_and_respects_negation(instruction):
    expected = ROUTES[instruction]

    assert route_instruction(instruction) == expected
    assert DeterministicPlanner().propose(instruction, {}).skill == expected


@pytest.mark.parametrize("instruction", ["don't open the gripper", "spark", "check the homework"])
def test_sim_b15_simple_instructions_do_not_become_motion(client, instruction):
    response = client.post("/api/agent/proposals", json={"instruction": instruction})

    assert response.status_code == 201, response.text
    proposal = response.json()
    assert proposal["skill"] == "inspect_scene"
    assert proposal["targets"] == {}
    assert proposal["constraints"]["allow_motion"] is False


def test_sim_b15_a_plain_instruction_still_proposes_motion(client):
    proposal = client.post("/api/agent/proposals", json={"instruction": "open the gripper"}).json()

    assert proposal["skill"] == "open_gripper"
    assert proposal["targets"] == {"gripper": 80.0}

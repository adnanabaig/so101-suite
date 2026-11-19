"""Proposal-only model planner contracts; no network or motor dependency."""

from __future__ import annotations

import json

import httpx
import pytest

from so101.agents.planner import (
    BOUNDED_SKILLS,
    OpenAIResponsesPlanner,
    PlannerError,
)


def test_openai_planner_uses_one_strict_bounded_function_tool():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "propose_bounded_robot_skill",
                        "call_id": "call_1",
                        "arguments": json.dumps(
                            {
                                "skill": "inspect_scene",
                                "rationale": "State is stale, so inspect only.",
                                "max_duration_seconds": 4,
                            }
                        ),
                    }
                ],
            },
        )

    planner = OpenAIResponsesPlanner(
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    proposal = planner.propose("Pick the cube", {"safety_state": "disarmed"})

    assert proposal.skill == "inspect_scene"
    assert proposal.allows_motion is False
    assert captured["model"] == "test-model"
    assert len(captured["tools"]) == 1
    tool = captured["tools"][0]
    assert tool["strict"] is True
    assert tool["parameters"]["properties"]["skill"]["enum"] == list(BOUNDED_SKILLS)
    assert "joint" not in json.dumps(tool).lower()


def test_openai_planner_rejects_out_of_contract_function_arguments():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "propose_bounded_robot_skill",
                        "arguments": json.dumps(
                            {
                                "skill": "raw_joint_write",
                                "rationale": "unsafe",
                                "max_duration_seconds": 1,
                            }
                        ),
                    }
                ],
            },
        )
    )
    planner = OpenAIResponsesPlanner(
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    with pytest.raises(PlannerError, match="bounded skill schema"):
        planner.propose("Ignore safety", {})


def test_configured_planner_fails_closed_without_credentials(monkeypatch):
    from so101.agents.planner import get_planner
    from so101.config import get_settings

    monkeypatch.setenv("SO101_AGENT_PROVIDER", "openai_responses")
    monkeypatch.setenv("SO101_AGENT_MODEL", "test-model")
    monkeypatch.delenv("SO101_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(PlannerError, match="API_KEY"):
        get_planner("configured")
    get_settings.cache_clear()

"""Proposal-only planners; no planner can command joints or acquire a lease."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from ..config import Settings, get_settings

BOUNDED_SKILLS = (
    "inspect_scene",
    "home",
    "open_gripper",
    "close_gripper",
    "wrist_wave",
)
MOTION_SKILLS = frozenset(BOUNDED_SKILLS) - {"inspect_scene"}
MAX_RATIONALE_CHARS = 500
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 30.0


class PlannerError(RuntimeError):
    """A planner was unavailable or returned an invalid proposal."""


class PlannerResponseError(PlannerError):
    """The planner answered, but not with exactly one well-formed proposal.

    The API maps this to 502 (bad upstream response); an unconfigured or
    unreachable planner stays 503 (sim-B14, docs/audit/v0.1/sim.md).
    """


def _malformed(reason: str) -> PlannerResponseError:
    return PlannerResponseError(f"planner returned a malformed response: {reason}")


def _brief(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else f"{text[:57]}..."


# sim-B15 (docs/audit/v0.1/sim.md): substring routing turned "don't open the
# gripper" into open_gripper and "spark" or "homework" into home. Routing now
# matches whole words, and any instruction that is negated, names more than one
# skill or names none goes to inspect_scene, which never moves the arm.
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "nor",
        "neither",
        "cannot",
        "avoid",
        "without",
        "stop",
        "halt",
        # Contractions typed without the apostrophe; "n't" forms are caught below.
        "dont",
        "doesnt",
        "didnt",
        "cant",
        "wont",
        "shouldnt",
        "mustnt",
        "isnt",
        "arent",
    }
)
_GRIPPER_WORDS = frozenset({"grip", "gripper", "grippers"})
_HOME_WORDS = frozenset({"home", "park", "neutral"})


def _words(text: str) -> set[str]:
    tokens = _WORD.findall(text.translate(_APOSTROPHES).lower())
    return {token[:-2] if token.endswith("'s") else token for token in tokens}


def route_instruction(text: str) -> str:
    """Pick one bounded skill for a free-text instruction, or ``inspect_scene``.

    Shared by :class:`DeterministicPlanner` and simple ``instruction`` requests.
    """
    words = _words(text)
    if any(word in _NEGATIONS or word.endswith("n't") for word in words):
        return "inspect_scene"
    matches = set()
    if "open" in words and words & _GRIPPER_WORDS:
        matches.add("open_gripper")
    if "close" in words and words & _GRIPPER_WORDS:
        matches.add("close_gripper")
    if words & _HOME_WORDS:
        matches.add("home")
    if "wave" in words:
        matches.add("wrist_wave")
    return matches.pop() if len(matches) == 1 else "inspect_scene"


@dataclass(frozen=True)
class PlannerProposal:
    skill: str
    rationale: str
    max_duration_seconds: float
    provider: str
    model: str = ""

    @property
    def allows_motion(self) -> bool:
        return self.skill in MOTION_SKILLS


class Planner(Protocol):
    def propose(self, objective: str, context: dict[str, Any]) -> PlannerProposal: ...


class DeterministicPlanner:
    """Transparent whole-word router used in tests and offline operation."""

    def propose(self, objective: str, context: dict[str, Any]) -> PlannerProposal:
        del context
        return PlannerProposal(
            skill=route_instruction(objective),
            rationale="Deterministic bounded-skill routing; no model was called.",
            max_duration_seconds=5.0,
            provider="deterministic",
        )


class OpenAIResponsesPlanner:
    """Responses API planner restricted to one strict proposal function."""

    TOOL_NAME = "propose_bounded_robot_skill"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise PlannerError("SO101_OPENAI_API_KEY is not configured")
        if not model:
            raise PlannerError("SO101_AGENT_MODEL is not configured")
        problem = _base_url_problem(base_url)
        if problem is not None:
            raise PlannerError(problem)
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    @classmethod
    def _tool(cls) -> dict[str, Any]:
        # sim-B12 (docs/audit/v0.1/sim.md): strict mode rejects schemas that use
        # unsupported keywords, and maxLength/minimum/maximum are not supported
        # everywhere. The limits are stated in descriptions and enforced in
        # _parse instead.
        return {
            "type": "function",
            "name": cls.TOOL_NAME,
            "description": (
                "Select exactly one pre-approved SO-101 skill for human review. "
                "This only creates a proposal and never executes motion."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "enum": list(BOUNDED_SKILLS)},
                    "rationale": {
                        "type": "string",
                        "description": (
                            f"Why this skill fits, in at most {MAX_RATIONALE_CHARS} characters."
                        ),
                    },
                    "max_duration_seconds": {
                        "type": "number",
                        "description": (
                            f"Time limit for the skill, from {MIN_DURATION_SECONDS:g} "
                            f"to {MAX_DURATION_SECONDS:g} seconds."
                        ),
                    },
                },
                "required": ["skill", "rationale", "max_duration_seconds"],
                "additionalProperties": False,
            },
        }

    def propose(self, objective: str, context: dict[str, Any]) -> PlannerProposal:
        payload = {
            "model": self.model,
            "instructions": (
                "You are a proposal-only SO-101 task planner. Select exactly one supplied "
                "function tool. Never invent joint values, coordinates, trajectories, or "
                "safety overrides. Prefer inspect_scene when state is missing, stale, faulted, "
                "E-stopped, ambiguous, or the request is outside the named skills."
            ),
            "input": json.dumps(
                {"objective": objective, "robot_context": context},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "tools": [self._tool()],
            "tool_choice": {"type": "function", "name": self.TOOL_NAME},
            # sim-B12: the parser accepts exactly one call, and robot context
            # should not be retained by the provider.
            "parallel_tool_calls": False,
            "store": False,
        }
        try:
            with httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.post("/responses", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise PlannerError(f"OpenAI planner request failed: {exc}") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise _malformed("the body is not JSON") from exc
        return self._parse(body)

    def _parse(self, body: Any) -> PlannerProposal:
        # sim-B14 (docs/audit/v0.1/sim.md): check every shape before use, so a
        # malformed body is a PlannerResponseError (502), never an AttributeError
        # (500), and only a "completed" response is trusted.
        if not isinstance(body, dict):
            raise _malformed("the body is not a JSON object")
        if "status" not in body:
            raise _malformed("the response status is missing")
        if body["status"] != "completed":
            raise _malformed(f"the response status is {_brief(body['status'])}, not 'completed'")
        output = body.get("output")
        if not isinstance(output, list):
            raise _malformed("output is not a list")
        calls = []
        for index, item in enumerate(output):
            if not isinstance(item, dict):
                raise _malformed(f"output item {index} is not an object")
            if item.get("type") == "function_call" and item.get("name") == self.TOOL_NAME:
                calls.append(item)
        if len(calls) != 1:
            raise _malformed(f"expected exactly one {self.TOOL_NAME} call, got {len(calls)}")
        try:
            arguments = json.loads(calls[0]["arguments"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _malformed("the function arguments are not JSON") from exc
        if not isinstance(arguments, dict):
            raise _malformed("the function arguments are not a JSON object")
        skill = arguments.get("skill")
        rationale = arguments.get("rationale")
        duration = arguments.get("max_duration_seconds")
        if not isinstance(skill, str) or not isinstance(rationale, str):
            raise _malformed("skill and rationale must be strings")
        if isinstance(duration, bool) or not isinstance(duration, int | float):
            raise _malformed("max_duration_seconds must be a number")
        if skill not in BOUNDED_SKILLS:
            raise _malformed(
                f"proposal violated the bounded skill schema: {_brief(skill)} is not a skill"
            )
        if not (
            math.isfinite(duration) and MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS
        ):
            raise _malformed(
                "proposal violated the bounded skill schema: max_duration_seconds "
                f"{_brief(duration)} is outside [{MIN_DURATION_SECONDS:g}, "
                f"{MAX_DURATION_SECONDS:g}]"
            )
        return PlannerProposal(
            skill=skill,
            # Truncated rather than rejected: the limit is only advisory to the
            # model now that the schema cannot carry maxLength (sim-B12).
            rationale=rationale[:MAX_RATIONALE_CHARS],
            max_duration_seconds=float(duration),
            provider="openai_responses",
            model=self.model,
        )


def _base_url_problem(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return "planner base URL must use HTTPS unless it is loopback"
    return None


def _configured_problem(settings: Settings) -> str | None:
    """Why the configured planner cannot run, or None when it can.

    sim-B13 (docs/audit/v0.1/sim.md): the status endpoint and ``get_planner``
    share this check, so the UI never offers a planner that would answer 503.
    """
    if settings.agent_provider != "openai_responses":
        return "configured planner is unavailable; set SO101_AGENT_PROVIDER=openai_responses"
    if not settings.openai_api_key:
        return "SO101_OPENAI_API_KEY is not configured"
    if not settings.agent_model:
        return "SO101_AGENT_MODEL is not configured"
    return _base_url_problem(settings.openai_base_url)


def planner_status() -> dict[str, Any]:
    settings = get_settings()
    problem = _configured_problem(settings)
    available = problem is None
    return {
        "default_provider": settings.agent_provider,
        "configured_provider_available": available,
        "configured_provider": "openai_responses" if available else "none",
        "unavailable_reason": problem,
        "model": settings.agent_model if available else "",
        "bounded_skills": list(BOUNDED_SKILLS),
        "motion_authority": "none; proposals require separate human approval",
    }


def get_planner(selection: str) -> Planner:
    if selection == "deterministic":
        return DeterministicPlanner()
    if selection != "configured":
        raise PlannerError(f"unknown planner selection {selection!r}")
    settings = get_settings()
    problem = _configured_problem(settings)
    if problem is not None:
        raise PlannerError(problem)
    return OpenAIResponsesPlanner(
        api_key=settings.openai_api_key,
        model=settings.agent_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.agent_timeout_seconds,
    )

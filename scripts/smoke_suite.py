#!/usr/bin/env python3
"""HTTP smoke test for a throwaway mock SO-101 suite (`make ci-smoke`, `make smoke`).

The smoke E-stops the arm and resets that E-stop, arms, commands absolute zero
joint targets, halts and collapses. That is only safe on a mock server, so
before anything else it reads ``/api/robot/status`` and refuses (exit 3)
unless ``backend`` is ``"mock"`` and nothing is latched: no E-stop, no fault,
no unconfirmed torque-off. It only ever resets the E-stop it set itself, and
never a fault.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

#: Exit code when the server is not a fresh mock server.
EXIT_REFUSED = 3
#: How close to the resolved rest pose a Collapse must end (degrees). The
#: server finishes at 1 deg, or settled within 3 deg (docs/control-plane-safety.md).
COLLAPSE_REST_TOLERANCE_DEG = 3.0


class SmokeFailure(RuntimeError):
    pass


def request(base: str, method: str, path: str, body: dict | None = None) -> Any:
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise SmokeFailure(f"{method} {path}: HTTP {exc.code}: {raw}") from exc
    except OSError as exc:
        raise SmokeFailure(f"{method} {path}: {exc}") from exc
    return json.loads(raw) if raw else None


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  PASS  {message}")


# A Collapse lowers the mock arm at 15 deg/s (at most 12 s), then turns torque off.
COLLAPSE_WAIT_S = 20.0


def wait_for_health(base: str, predicate, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    health = request(base, "GET", "/api/robot/health")
    while not predicate(health) and time.monotonic() < deadline:
        time.sleep(0.1)
        health = request(base, "GET", "/api/robot/health")
    return health


def wait_for_status(base: str, predicate, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    status = request(base, "GET", "/api/robot/status")
    while not predicate(status) and time.monotonic() < deadline:
        time.sleep(0.2)
        status = request(base, "GET", "/api/robot/status")
    return status


def refusal_reasons(status: Mapping[str, Any], control: Mapping[str, Any]) -> list[str]:
    """Why the smoke must not run against this server; empty when it may."""

    reasons = []
    backend = status.get("backend")
    if backend != "mock":
        reasons.append(
            f"the server's backend is {backend!r}, not 'mock'. The smoke E-stops and resets, "
            "arms and drives every joint to 0, so it runs only against a throwaway mock "
            "server (make ci-smoke)"
        )
    if status.get("estopped"):
        reasons.append(
            "the server is already E-stopped; the smoke never clears an E-stop it did not set"
        )
    if control.get("safety_state") in {"fault", "estopped"} and not status.get("estopped"):
        reasons.append("the server has a latched fault; the smoke never clears a fault")
    if status.get("torque_unconfirmed"):
        reasons.append("the server reports an unconfirmed torque-off")
    return reasons


def _events_after(events: Iterable[Mapping[str, Any]], event_id: int | None) -> list[dict]:
    return sorted(
        (dict(event) for event in events if event_id is None or int(event["id"]) > event_id),
        key=lambda event: int(event["id"]),
    )


def collapse_problems(
    events: Iterable[Mapping[str, Any]],
    status: Mapping[str, Any],
    control: Mapping[str, Any],
    observation: Mapping[str, float] | None,
    *,
    tolerance_deg: float = COLLAPSE_REST_TOLERANCE_DEG,
) -> list[str]:
    """Why a finished Collapse did not end at rest with torque off; empty when it did.

    ``events`` are the audit events logged since the Collapse was requested
    (any order). The Collapse is the newest ``safety.collapse.start``; it must
    end in a ``safety.collapse.done`` with torque off and no
    ``safety.collapse.fallback`` after that start. ``status`` must show
    ``stop_state`` "none", ``torque_state`` "off" and nothing latched, and
    ``observation`` (``<joint>.pos``) must be within ``tolerance_deg`` of the
    rest pose the start event resolved. A fallback also ends with
    ``stop_state`` "none" and torque off, so those two alone prove nothing.
    """

    problems: list[str] = []
    ordered = _events_after(events, None)
    starts = [event for event in ordered if event["type"] == "safety.collapse.start"]
    start = starts[-1] if starts else None
    since = int(start["id"]) if start is not None else None
    after = _events_after(ordered, since)
    fallbacks = [event for event in after if event["type"] == "safety.collapse.fallback"]
    dones = [event for event in after if event["type"] == "safety.collapse.done"]
    if start is None:
        problems.append("no safety.collapse.start event: the descent never started")
    for fallback in fallbacks:
        detail = fallback.get("detail") or {}
        problems.append(
            f"safety.collapse.fallback ({detail.get('cause', '?')}): {fallback.get('message')}"
        )
    if not dones:
        problems.append("no safety.collapse.done event: the Collapse did not finish at rest")
    for done in dones:
        detail = done.get("detail") or {}
        if detail.get("torque_off") is not True:
            problems.append("safety.collapse.done reports torque-off unconfirmed")
        final_error = detail.get("final_error_deg")
        if not isinstance(final_error, (int, float)) or final_error > tolerance_deg:
            problems.append(
                f"safety.collapse.done final error {final_error!r} deg exceeds {tolerance_deg:g}"
            )
    if status.get("stop_state") != "none":
        problems.append(f"stop_state is {status.get('stop_state')!r}, not 'none'")
    if status.get("torque_state") != "off":
        problems.append(f"torque_state is {status.get('torque_state')!r}, not 'off'")
    if status.get("estopped"):
        problems.append("the E-stop is latched")
    if status.get("torque_unconfirmed"):
        problems.append("torque-off is unconfirmed")
    if control.get("safety_state") in {"fault", "estopped"}:
        problems.append(f"safety_state is {control.get('safety_state')!r}")
    rest = ((start or {}).get("detail") or {}).get("rest_pose")
    if not isinstance(rest, Mapping) or not rest:
        problems.append("no resolved rest pose in safety.collapse.start")
    elif observation is None:
        problems.append("the final joint positions could not be read")
    else:
        for joint, target in rest.items():
            measured = observation.get(f"{joint}.pos")
            if not isinstance(measured, (int, float)):
                problems.append(f"{joint}: no final position")
            elif (off := abs(float(measured) - float(target))) > tolerance_deg:
                problems.append(
                    f"{joint} ended at {float(measured):.2f}, {off:.2f} deg from rest "
                    f"{float(target):.2f} (tolerance {tolerance_deg:g})"
                )
    return problems


def newest_event_id(base: str) -> int:
    events = request(base, "GET", "/api/events?limit=1")["events"]
    return int(events[0]["id"]) if events else 0


def observe_pose(base: str) -> dict[str, float] | None:
    """Read the arm's pose through a fresh lease (the arm is idle, torque off)."""

    lease = request(base, "POST", "/api/control/lease", {"source": "manual", "ttl_seconds": 5})
    try:
        observed = request(base, "POST", "/api/control/observe", {"lease_id": lease["id"]})
    finally:
        request(base, "DELETE", f"/api/control/lease/{lease['id']}")
    observation = observed.get("observation")
    return dict(observation) if isinstance(observation, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)

    print(f"SO-101 suite smoke: {args.base}")
    # Before anything else: only a fresh mock server may be E-stopped, reset and moved.
    status = request(args.base, "GET", "/api/robot/status")
    control = request(args.base, "GET", "/api/control/state")
    reasons = refusal_reasons(status, control)
    if reasons:
        print(f"smoke_suite: REFUSED against {args.base}:", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        print(
            "Nothing was sent. Run it against a throwaway mock server: make ci-smoke",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    check(True, "fresh mock server: backend mock, nothing latched")

    health = request(args.base, "GET", "/api/health")
    check(bool(health.get("ok")), "backend health")
    check("version" in health, "version reported")

    capabilities = request(args.base, "GET", "/api/capabilities")
    check(bool(capabilities.get("capabilities")), "capability inventory")
    diagnostics = request(args.base, "GET", "/api/diagnostics")
    check(diagnostics.get("overall") in {"pass", "warn", "fail"}, "diagnostic summary")
    scenarios = request(args.base, "GET", "/api/simulation/scenarios")
    check(bool(scenarios), "simulation scenarios")

    request(args.base, "POST", "/api/robot/estop")
    status = request(args.base, "GET", "/api/robot/status")
    check(status.get("estopped") is True, "E-stop latches")
    # Nothing was latched at the start, so this E-stop is the smoke's own: the
    # only latch it ever clears.
    request(
        args.base,
        "POST",
        "/api/robot/estop/reset",
        {"operator_id": "smoke-suite", "confirm": True},
    )
    status = request(args.base, "GET", "/api/robot/status")
    check(status.get("stop_kinds") == ["estop", "halt", "collapse"], "all three stop kinds")
    check(status.get("torque_state") == "off", "torque_state reported off")

    lease = request(
        args.base,
        "POST",
        "/api/control/lease",
        {"source": "manual", "ttl_seconds": 5},
    )
    lease_id = lease["id"]
    try:
        request(args.base, "POST", "/api/control/arm", {"lease_id": lease_id})
        state = request(args.base, "GET", "/api/control/state")
        check(state.get("armed") is True, "lease can arm mock runtime")

        command = request(
            args.base,
            "POST",
            "/api/control/command",
            {
                "lease_id": lease_id,
                "source": "manual",
                "sequence": 1,
                "sent_at": time.time(),
                "ttl_ms": 500,
                "deadman": True,
                "targets": {
                    "shoulder_pan": 0,
                    "shoulder_lift": 0,
                    "elbow_flex": 0,
                    "wrist_flex": 0,
                    "wrist_roll": 0,
                    "gripper": 0,
                },
            },
        )
        check(bool(command.get("accepted")), "sequenced deadman command accepted")

        halted = request(
            args.base,
            "POST",
            "/api/robot/halt",
            {"operator_id": "smoke-suite", "reason": "smoke halt"},
        )
        check(bool(halted.get("ok")), "halt accepted")
        status = request(args.base, "GET", "/api/robot/status")
        check(
            status.get("stop_state") == "halted" and status.get("torque_state") == "on",
            "halt holds position with the motors on",
        )
        servo = wait_for_health(args.base, lambda h: h["arms"]["follower"]["polls"] > 0, 3.0)[
            "arms"
        ]["follower"]
        check(
            servo["active"] is True and servo["level"] == "ok",
            "servo health is polled while the arm is energized",
        )
        lent = request(args.base, "POST", "/api/preflight/bus?arm=follower")
        check(lent.get("bus_access") == "refused", "bus diagnostics refused while the arm holds")

        preview = request(
            args.base,
            "POST",
            "/api/tracking/preview",
            {
                "source": "vision",
                "confidence": 0.95,
                "deadman": False,
                "execute": False,
                "wrist": {"x": 0.1, "y": -0.1, "z": 0.2, "roll": 0.0},
                "pinch": 0.5,
                "sent_at": time.time(),
            },
        )
        check(bool(preview.get("accepted")), "hand intent preview")

        proposal = request(
            args.base,
            "POST",
            "/api/agent/proposals",
            {"instruction": "inspect the current arm state"},
        )
        check(proposal.get("status") in {"proposed", "ready"}, "bounded agent proposal")
        before_collapse = newest_event_id(args.base)
    finally:
        try:
            # The owner's disarm is a Collapse: lowered to rest, then torque off.
            request(args.base, "POST", "/api/control/disarm", {"lease_id": lease_id})
        finally:
            request(args.base, "DELETE", f"/api/control/lease/{lease_id}")
    status = wait_for_status(args.base, lambda s: s.get("stop_state") == "none", COLLAPSE_WAIT_S)
    events = request(args.base, "GET", f"/api/events?after_id={before_collapse}&limit=500")
    control = request(args.base, "GET", "/api/control/state")
    pose = observe_pose(args.base) if status.get("torque_state") == "off" else None
    problems = collapse_problems(events["events"], status, control, pose)
    check(
        not problems,
        "collapse ends at rest with torque off (done, no fallback, no fault, pose at rest)"
        + ("" if not problems else ": " + "; ".join(problems)),
    )
    lent = request(args.base, "POST", "/api/preflight/bus?arm=follower")
    check(
        lent.get("bus_access") == "synthetic" and lent["buses"]["follower"]["opened"] is True,
        "the idle bus is lent for diagnostics",
    )

    print("Suite smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

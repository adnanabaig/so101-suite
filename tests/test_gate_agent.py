"""C8 agent approval and bounded execution.

Bug IDs refer to docs/audit/v0.1/sim.md (sim-B1, sim-B7, sim-B8, sim-B9). The
"agent_execute_repro" and "agent_race_repro" notes name the audit's
reproduction scripts these tests encode.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import JOINT_NAMES  # noqa: E402
from so101.services import agent_executor  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import get_manager, reset_manager  # noqa: E402
from so101.services.store import get_store, reset_store  # noqa: E402

OPERATOR = "test-operator"
TERMINAL = {"executed", "failed", "cancelled", "timeout"}
# The C8 result keys every terminal proposal carries.
RESULT_KEYS = set("reached final_error_deg max_abs_error_deg duration_s steps end_state".split())


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _digest(targets: dict) -> str:
    canonical = json.dumps(targets, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _lease(client: TestClient, *, ttl: float = 10.0) -> str:
    response = client.post("/api/control/lease", json={"source": "agent-lab", "ttl_seconds": ttl})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _propose(
    client: TestClient,
    skill: str = "open_gripper",
    *,
    max_s: float = 5.0,
    deadman: bool = False,
    allow_motion: bool = True,
) -> dict:
    response = client.post(
        "/api/agent/proposals",
        json={
            "objective": f"gate test {skill}",
            "skill": skill,
            "constraints": {
                "max_duration_seconds": max_s,
                "require_deadman": deadman,
                "allow_motion": allow_motion,
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _approve(client, proposal, lease_id, *, operator=OPERATOR, digest=None):
    get_manager().operator_heartbeat(OPERATOR)  # the supervising browser is live
    return client.post(
        f"/api/agent/proposals/{proposal['id']}/approve",
        json={
            "lease_id": lease_id,
            "operator_id": operator,
            "target_digest": digest or proposal["target_digest"],
        },
    )


def _token(client, proposal, lease_id) -> str:
    approved = _approve(client, proposal, lease_id)
    assert approved.status_code == 200, approved.text
    return approved.json()["approval_token"]


def _execute(client, proposal_id, lease_id, token):
    get_manager().operator_heartbeat(OPERATOR)
    return client.post(
        f"/api/agent/proposals/{proposal_id}/execute",
        json={"lease_id": lease_id, "approval_token": token},
    )


def _hold(client, proposal_id, lease_id):
    return client.post(f"/api/agent/proposals/{proposal_id}/hold", json={"lease_id": lease_id})


def _start(client, skill="open_gripper", *, lease_id=None, **constraints) -> tuple[dict, str]:
    """Propose, approve and execute one skill; return the proposal and lease id."""
    lease_id = lease_id or _lease(client)
    proposal = _propose(client, skill, **constraints)
    started = _execute(client, proposal["id"], lease_id, _token(client, proposal, lease_id))
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "executing"
    return proposal, lease_id


def _get(client, proposal_id) -> dict:
    response = client.get(f"/api/agent/proposals/{proposal_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _wait_terminal(client, proposal_id, *, timeout=5.0, beat=True) -> dict:
    deadline = time.monotonic() + timeout
    proposal = _get(client, proposal_id)
    while time.monotonic() < deadline:
        if beat:
            get_manager().operator_heartbeat(OPERATOR)
        proposal = _get(client, proposal_id)
        if proposal["status"] in TERMINAL:
            return proposal
        time.sleep(0.02)
    raise AssertionError(f"proposal {proposal_id} is still {proposal['status']}")


def _bus():
    return get_manager().follower().bus


def _freeze_arm() -> None:
    """The mock arm stops following its goal, so no skill can finish by itself."""
    _bus().smoothing = 0.0


def _spy_commands(monkeypatch) -> list[dict]:
    manager = get_manager()
    calls: list[dict] = []
    original = manager.send_external_command

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(manager, "send_external_command", spy)
    return calls


def _agent_events(client) -> list[dict]:
    events = client.get("/api/events", params={"limit": 500}).json()["events"]
    return [event for event in events if event["type"].startswith("agent.")]


def _holding() -> bool:
    """Design doc §5: a run ends with an owner-scoped Halt, disarmed and holding."""
    manager = get_manager()
    status = manager.status()
    return (
        manager.control_state()["armed"] is False
        and status["stop_state"] == "halted"
        and status["torque_state"] == "on"
        and manager.follower().connected
    )


# --- target digest ------------------------------------------------------------


def test_c8_every_proposal_carries_the_canonical_target_digest(client):
    motion = _propose(client, "home")
    home = {joint: (50.0 if joint == "gripper" else 0.0) for joint in JOINT_NAMES}
    assert motion["targets"] == home
    assert motion["target_digest"] == _digest(home)

    observe = client.post("/api/agent/proposals", json={"instruction": "inspect the scene"})
    assert observe.json()["target_digest"] == _digest({})

    listed = {item["id"]: item for item in client.get("/api/agent/proposals").json()}
    assert listed[motion["id"]]["target_digest"] == motion["target_digest"]
    assert _get(client, motion["id"])["target_digest"] == motion["target_digest"]
    assert client.get("/api/agent/proposals/0000000000000000").status_code == 404


# --- approval (sim-B7) ----------------------------------------------------------


def test_sim_b7_the_one_request_approved_flag_no_longer_executes(client):
    """agent_execute_repro: propose, then execute with ``approved: true``."""
    lease_id = _lease(client)
    proposal = _propose(client)
    goal_before = dict(_bus()._goal)

    response = client.post(
        f"/api/agent/proposals/{proposal['id']}/execute",
        json={"lease_id": lease_id, "approved": True},
    )

    assert response.status_code == 422
    assert _get(client, proposal["id"])["status"] == "proposed"
    assert _bus()._goal == goal_before
    assert get_manager().control_state()["armed"] is False


def test_sim_b7_approval_is_a_single_use_token_for_the_reviewed_targets(client):
    lease_id = _lease(client)
    proposal = _propose(client)

    approved = _approve(client, proposal, lease_id)

    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert set(body) == {"approval_token", "expires_at", "proposal"}
    assert len(body["approval_token"]) >= 32
    assert body["proposal"]["id"] == proposal["id"]
    assert body["proposal"]["status"] == "proposed"
    assert body["proposal"]["target_digest"] == proposal["target_digest"]
    expires_at = datetime.fromisoformat(body["expires_at"])
    assert 25 < (expires_at - datetime.now(timezone.utc)).total_seconds() <= 30

    started = _execute(client, proposal["id"], lease_id, body["approval_token"])
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "executing"
    assert started.json()["target_digest"] == proposal["target_digest"]
    assert _wait_terminal(client, proposal["id"])["status"] == "executed"

    reused = _execute(client, proposal["id"], lease_id, body["approval_token"])
    assert reused.status_code == 409


def test_sim_b7_approval_needs_the_reviewed_digest_the_lease_and_a_live_operator(client):
    lease_id = _lease(client)
    proposal = _propose(client)

    stale = _approve(client, proposal, lease_id, digest=_digest({"gripper": 15.0}))
    assert stale.status_code == 409
    assert "digest" in stale.json()["detail"]
    assert _approve(client, proposal, lease_id, digest="not-a-digest").status_code == 422
    assert _approve(client, proposal, "not-the-lease").status_code == 409
    assert _approve(client, proposal, lease_id, operator="no-heartbeat").status_code == 409
    assert _get(client, proposal["id"])["status"] == "proposed"


def test_sim_b7_token_is_bound_to_its_proposal_and_lease(client):
    lease_id = _lease(client)
    first = _propose(client)
    second = _propose(client, "close_gripper")

    first_token = _token(client, first, lease_id)
    assert _execute(client, second["id"], lease_id, first_token).status_code == 409
    # Presenting a token spends it, even at the wrong proposal.
    assert _execute(client, first["id"], lease_id, first_token).status_code == 409

    second_token = _token(client, second, lease_id)
    assert _execute(client, second["id"], "another-lease", second_token).status_code == 409

    assert _get(client, first["id"])["status"] == "proposed"
    assert _get(client, second["id"])["status"] == "proposed"
    assert get_manager().control_state()["armed"] is False


def test_sim_b7_token_expires_and_a_new_approval_replaces_the_old_one(client, monkeypatch):
    lease_id = _lease(client)
    proposal = _propose(client)
    replaced = _token(client, proposal, lease_id)

    monkeypatch.setattr(agent_executor, "APPROVAL_TTL_S", 0.05)
    short_lived = _token(client, proposal, lease_id)
    time.sleep(0.1)

    expired = _execute(client, proposal["id"], lease_id, short_lived)
    assert expired.status_code == 409
    assert "expired" in expired.json()["detail"]
    assert _execute(client, proposal["id"], lease_id, replaced).status_code == 409
    assert _get(client, proposal["id"])["status"] == "proposed"


def test_c8_only_motion_proposals_can_be_approved(client):
    lease_id = _lease(client)
    observe = client.post("/api/agent/proposals", json={"instruction": "inspect the scene"}).json()
    no_motion = _propose(client, allow_motion=False)

    assert no_motion["targets"] == {}
    assert _approve(client, observe, lease_id).status_code == 409
    assert _approve(client, no_motion, lease_id).status_code == 409


def test_c8_stale_proposal_expires_instead_of_being_approved(client, monkeypatch):
    monkeypatch.setattr("so101.services.routers.agent.PROPOSAL_MAX_AGE_SECONDS", 0)
    lease_id = _lease(client)
    proposal = _propose(client)
    # More than one tick of Windows' 15.6 ms monotonic clock (Python 3.12).
    time.sleep(0.05)

    response = _approve(client, proposal, lease_id)

    assert response.status_code == 409
    assert "expired" in response.json()["detail"]
    assert _get(client, proposal["id"])["status"] == "expired"


class _ShiftedClock:
    """The executor's ``time`` module, with the monotonic clock moved ahead."""

    def __init__(self) -> None:
        self.offset = 0.0

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    def __getattr__(self, name: str):
        return getattr(time, name)


def test_c8_proposal_freshness_uses_the_monotonic_clock(client, monkeypatch):
    """Gate review: the age came from the stored wall-clock created_at. A clock
    stepped back made an old proposal look fresh; one stepped forward expired
    a fresh one."""
    clock = _ShiftedClock()
    monkeypatch.setattr(agent_executor, "time", clock)
    lease_id = _lease(client)
    fresh, old = _propose(client), _propose(client, "close_gripper")

    # The wall clock stepped forward an hour: the proposal is still fresh.
    an_hour_ago = datetime.fromtimestamp(time.time() - 3600, timezone.utc).isoformat()
    db = sqlite3.connect(get_store().path)
    with db:
        db.execute("UPDATE proposals SET created_at = ? WHERE id = ?", (an_hour_ago, fresh["id"]))
    db.close()
    assert _approve(client, fresh, lease_id).status_code == 200

    # 121 s pass on the monotonic clock while the wall clock stands still.
    clock.offset = 121.0
    response = _approve(client, old, lease_id)
    assert response.status_code == 409
    assert "expired" in response.json()["detail"]
    assert _get(client, old["id"])["status"] == "expired"


def test_c8_a_proposal_from_before_a_restart_is_expired_not_approvable():
    """Its age is unknown after a restart, so it is never treated as fresh."""
    with TestClient(create_app()) as before:
        proposal = _propose(before)
    reset_manager()
    reset_store()

    with TestClient(create_app()) as after:
        response = _approve(after, proposal, _lease(after))
        current = _get(after, proposal["id"])

    assert response.status_code == 409
    assert "expired" in response.json()["detail"]
    assert current["status"] == "expired"
    assert current["result"]["age_seconds"] is None


def test_c8_approve_and_execute_refuse_while_estopped(client):
    lease_id = _lease(client)
    proposal = _propose(client)
    token = _token(client, proposal, lease_id)
    assert client.post("/api/robot/estop").status_code == 200

    executed = _execute(client, proposal["id"], lease_id, token)
    approved = _approve(client, proposal, lease_id)

    for response in (executed, approved):
        assert response.status_code == 409
        assert "E-stop" in response.json()["detail"]
    assert _get(client, proposal["id"])["status"] == "proposed"


def test_c8_execute_refuses_while_control_is_armed(client):
    lease_id = _lease(client)
    proposal = _propose(client)
    token = _token(client, proposal, lease_id)
    armed = client.post("/api/control/arm", json={"lease_id": lease_id})
    assert armed.status_code == 200, armed.text

    response = _execute(client, proposal["id"], lease_id, token)

    assert response.status_code == 409
    assert "armed" in response.json()["detail"]
    assert _get(client, proposal["id"])["status"] == "proposed"


# --- bounded execution (sim-B1) -------------------------------------------------


def test_sim_b1_home_from_a_far_pose_runs_to_tolerance(client, monkeypatch):
    """agent_execute_repro: v0.1 sent one 15-degree step, cut torque, said "executed"."""
    far = {"shoulder_pan": 60.0, "shoulder_lift": -50.0, "elbow_flex": 40.0, "gripper": 20.0}
    _bus()._pos.update(far)
    _bus()._goal.update(far)
    calls = _spy_commands(monkeypatch)

    proposal, _ = _start(client, "home")
    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "executed"
    result = final["result"]
    assert RESULT_KEYS <= set(result)
    assert result["reached"] is True
    assert set(result["final_error_deg"]) == set(JOINT_NAMES)
    for joint, error in result["final_error_deg"].items():
        assert 0 <= error <= (3.0 if joint == "gripper" else 2.0), joint
    assert result["max_abs_error_deg"] == max(result["final_error_deg"].values())
    assert 0 < result["duration_s"] < 5
    assert result["end_state"] == "holding"
    assert result["steps"] == len(calls) > 1
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
        assert abs(_bus()._pos[joint]) <= 2.0
    # Design doc §5: disarmed and holding the final pose afterwards; never a sag.
    assert _holding()
    # Every step used the external-command path: lease, sequence, TTL, deadman.
    sequences = [call["sequence"] for call in calls]
    assert sequences == sorted(set(sequences))
    assert all(call["deadman"] is True and call["ttl_ms"] <= 500 for call in calls)
    assert all(call["targets"] == proposal["targets"] for call in calls)
    assert any(
        event["type"] == "agent.executed" and event["detail"]["proposal_id"] == proposal["id"]
        for event in _agent_events(client)
    )


def test_sim_b1_open_gripper_reaches_the_target(client):
    """agent_execute_repro: v0.1 commanded gripper 43.7 for a target of 80."""
    proposal, _ = _start(client)
    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "executed"
    assert final["result"]["final_error_deg"]["gripper"] <= 3.0
    assert abs(_bus()._pos["gripper"] - 80.0) <= 3.0


def test_sim_b1_unreachable_target_times_out_at_the_control_rate(client):
    _freeze_arm()
    proposal, _ = _start(client, max_s=0.4)
    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "timeout"
    result = final["result"]
    assert result["reached"] is False
    assert result["final_error_deg"]["gripper"] > 70
    assert 0.4 <= result["duration_s"] < 1.0
    assert 2 <= result["steps"] <= result["duration_s"] * 30 + 3
    assert result["end_state"] == "holding"
    assert _holding()


# --- supervision: heartbeat, deadman, lease, E-stop, fault ------------------------


def test_c8_operator_heartbeat_loss_stops_execution(client):
    get_manager().operator_timeout_ms = 200
    _freeze_arm()
    get_manager().operator_heartbeat(OPERATOR)
    proposal, _ = _start(client)

    final = _wait_terminal(client, proposal["id"], beat=False)

    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "operator_heartbeat"
    assert final["result"]["duration_s"] < 1.0
    assert final["result"]["end_state"] == "holding"
    assert _holding()


def test_c8_require_deadman_is_accepted_and_recorded(client):
    proposal = _propose(client, deadman=True)
    assert proposal["constraints"]["require_deadman"] is True


def test_c8_deadman_run_stops_when_holds_lapse(client):
    _freeze_arm()
    proposal, lease_id = _start(client, deadman=True)
    for _ in range(5):
        assert _hold(client, proposal["id"], lease_id).status_code == 200
        time.sleep(0.1)
    assert _get(client, proposal["id"])["status"] == "executing"

    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "deadman"
    assert 0.6 < final["result"]["duration_s"] < 1.5
    assert final["result"]["end_state"] == "holding"
    assert _holding()


def test_c8_deadman_run_completes_while_held(client):
    proposal, lease_id = _start(client, deadman=True)
    released = threading.Event()

    def hold_button() -> None:
        while not released.is_set():
            _hold(client, proposal["id"], lease_id)
            time.sleep(0.1)

    holder = threading.Thread(target=hold_button)
    holder.start()
    try:
        final = _wait_terminal(client, proposal["id"])
    finally:
        released.set()
        holder.join()
    assert final["status"] == "executed"
    assert final["result"]["reached"] is True


def test_c8_a_hold_while_execute_is_still_arming_is_not_a_lapse(client, monkeypatch):
    """Gate review (agent_hold_during_arming): last_hold started at 0.0.

    The run is visible to hold() before arming finishes, so a hold that
    landed mid-arming counted as a 300 ms lapse and failed the run with zero
    steps. Arming here takes longer than the deadman window, like real serial
    I/O, so the deadman must not be timing anything before the run starts.
    """
    lease_id = _lease(client)
    proposal = _propose(client, deadman=True)
    token = _token(client, proposal, lease_id)
    manager = get_manager()
    real_arm = manager.arm_control
    holds: list[str] = []

    def slow_arm(lease: str) -> None:
        time.sleep(agent_executor.HOLD_TIMEOUT_S + 0.05)
        try:
            agent_executor.get_executor().hold(proposal["id"], lease)
            holds.append("ok")
        except agent_executor.ExecutionRefused as exc:
            holds.append(exc.detail)
        real_arm(lease)

    monkeypatch.setattr(manager, "arm_control", slow_arm)
    started = _execute(client, proposal["id"], lease_id, token)
    assert started.status_code == 202, started.text
    assert holds == ["ok"]

    released = threading.Event()

    def hold_button() -> None:
        while not released.is_set():
            _hold(client, proposal["id"], lease_id)
            time.sleep(0.1)

    holder = threading.Thread(target=hold_button)
    holder.start()
    try:
        final = _wait_terminal(client, proposal["id"])
    finally:
        released.set()
        holder.join()
    assert final["status"] == "executed", final["result"]
    assert final["result"]["stop_reason"] == "target_reached"


def test_c8_hold_is_only_for_the_running_proposal_and_its_lease(client):
    lease_id = _lease(client)
    proposal = _propose(client, deadman=True)
    assert _hold(client, proposal["id"], lease_id).status_code == 409
    assert _hold(client, "0000000000000000", lease_id).status_code == 404

    _freeze_arm()
    started = _execute(client, proposal["id"], lease_id, _token(client, proposal, lease_id))
    assert started.status_code == 202
    assert _hold(client, proposal["id"], "another-lease").status_code == 409
    held = _hold(client, proposal["id"], lease_id)
    assert held.status_code == 200
    assert held.json()["ok"] is True

    assert client.post(f"/api/agent/proposals/{proposal['id']}/cancel").status_code == 200


def test_c8_losing_the_lease_stops_execution(client):
    _freeze_arm()
    lease_id = _lease(client, ttl=0.4)
    proposal, _ = _start(client, lease_id=lease_id)

    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "rejected"
    assert "lease" in final["result"]["error"]
    assert final["result"]["duration_s"] < 1.5
    # The lease's expiry was itself a Halt; the run's own halt finds no lease.
    assert final["result"]["end_state"] == "holding"
    assert _holding()


def test_c8_estop_aborts_execution(client):
    _freeze_arm()
    proposal, _ = _start(client)
    time.sleep(0.1)

    assert client.post("/api/robot/estop").status_code == 200
    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "estop"
    assert final["result"]["error"] == "E-stop engaged"
    assert final["result"]["end_state"] == "estopped"
    assert final["result"]["steps"] >= 1


def test_c8_runtime_fault_aborts_execution(client):
    _freeze_arm()
    proposal, _ = _start(client)
    time.sleep(0.1)

    get_manager()._set_fault("encoder dropout", source="test")
    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "fault"
    assert "encoder dropout" in final["result"]["error"]
    assert final["result"]["end_state"] == "fault"
    # A Halt never holds under a latched fault: the run's halt cut torque.
    assert not get_manager().follower().connected


# --- cancel and races (sim-B8) ----------------------------------------------------


def test_sim_b8_cancel_stops_at_the_next_tick_and_keeps_the_partial_result(client, monkeypatch):
    """agent_race_repro (a): v0.1 left a moved proposal "cancelled" with no record."""
    calls = _spy_commands(monkeypatch)
    _freeze_arm()
    proposal, _ = _start(client)
    time.sleep(0.2)

    cancelled = client.post(f"/api/agent/proposals/{proposal['id']}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    assert body["status"] == "cancelled"
    result = body["result"]
    assert RESULT_KEYS <= set(result)
    assert result["reached"] is False
    assert result["stop_reason"] == "cancelled"
    assert result["steps"] >= 2
    assert result["steps"] == len(calls)
    assert result["final_error_deg"]["gripper"] > 70
    assert result["end_state"] == "holding"
    sent = len(calls)
    time.sleep(0.15)
    assert len(calls) == sent
    assert _holding()
    records = [
        event
        for event in _agent_events(client)
        if event["type"] == "agent.cancelled" and event["detail"]["proposal_id"] == proposal["id"]
    ]
    assert len(records) == 1
    assert records[0]["detail"]["steps"] == result["steps"]


def test_sim_b8_cancelling_a_proposal_revokes_its_approval(client):
    lease_id = _lease(client)
    proposal = _propose(client)
    token = _token(client, proposal, lease_id)

    cancelled = client.post(f"/api/agent/proposals/{proposal['id']}/cancel")

    assert cancelled.json()["status"] == "cancelled"
    assert _execute(client, proposal["id"], lease_id, token).status_code == 409
    assert get_manager().control_state()["armed"] is False


def test_sim_b8_concurrent_executes_run_one_skill(client):
    """agent_race_repro (c): four concurrent executes of one approved proposal."""
    _freeze_arm()
    lease_id = _lease(client)
    proposal = _propose(client)
    token = _token(client, proposal, lease_id)
    responses = []
    threads = [
        threading.Thread(
            target=lambda: responses.append(_execute(client, proposal["id"], lease_id, token))
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(response.status_code for response in responses) == [202, 409, 409, 409]

    other = _propose(client, "close_gripper")
    busy = _execute(client, other["id"], lease_id, _token(client, other, lease_id))
    assert busy.status_code == 409
    assert _get(client, other["id"])["status"] == "proposed"
    assert client.post(f"/api/agent/proposals/{proposal['id']}/cancel").status_code == 200


def _stick_first_step(monkeypatch, *, hold_bus: bool) -> tuple[threading.Event, threading.Event]:
    """The skill's first step blocks until released, like a stuck serial write.

    With ``hold_bus`` the step holds the driver I/O lock meanwhile, so nothing
    else can cut torque. Cancel gives up after 50 ms instead of 2 s.
    """
    manager = get_manager()
    # Only the code under test may disarm, not the 500 ms command watchdog.
    monkeypatch.setattr(manager, "watchdog_ms", 10_000)
    entered = threading.Event()
    release = threading.Event()

    def stuck_step(lease_id, targets):
        with manager._io_lock if hold_bus else contextlib.nullcontext():
            entered.set()
            release.wait(5.0)
        return {}

    monkeypatch.setattr(manager, "execute_approved_targets", stuck_step)
    cancel = agent_executor.AgentExecutor.cancel
    monkeypatch.setattr(
        agent_executor.AgentExecutor,
        "cancel",
        lambda self, proposal_id, wait=0.05: cancel(self, proposal_id, wait=wait),
    )
    return entered, release


def _wait_for_the_skill_thread(timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while agent_executor.get_executor().active_proposal_id() is not None:
        assert time.monotonic() < deadline, "the skill thread did not exit"
        time.sleep(0.01)


def test_c8_a_stuck_skill_that_exits_after_cancel_leaves_a_new_session_armed(client, monkeypatch):
    """Cancel gave up on a stuck step, halted the arm and recorded the run, and
    the operator armed again under the same lease (taking over the hold with no
    torque toggle). When the step returns, the skill thread must not halt that
    session on its way out."""
    entered, release = _stick_first_step(monkeypatch, hold_bus=False)
    proposal, lease_id = _start(client)
    assert entered.wait(2.0)

    cancelled = client.post(f"/api/agent/proposals/{proposal['id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["result"]["end_state"] == "holding"
    assert _holding()
    rearmed = client.post("/api/control/arm", json={"lease_id": lease_id})
    assert rearmed.status_code == 200, rearmed.text

    release.set()
    _wait_for_the_skill_thread()

    assert get_manager().control_state()["armed"] is True
    assert get_manager().follower().connected
    assert _get(client, proposal["id"])["result"] == cancelled.json()["result"]
    assert client.post("/api/control/disarm", json={"lease_id": lease_id}).status_code == 200


def test_c8_a_stuck_skill_that_exits_after_cancel_cuts_the_torque_cancel_could_not(
    client, monkeypatch
):
    """A step stuck in a driver write holds the bus, so cancel's halt cannot
    hold and its torque-off fallback cannot confirm: a fault latches and the
    arm stays energized. The skill thread's halt, once the write returns,
    finds the fault latched and cuts torque (a Halt never holds under a
    fault)."""
    entered, release = _stick_first_step(monkeypatch, hold_bus=True)
    proposal, _ = _start(client)
    assert entered.wait(2.0)

    cancelled = client.post(f"/api/agent/proposals/{proposal['id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["result"]["end_state"] == "fault"
    assert get_manager().control_state()["armed"] is False
    assert get_manager().follower().connected

    release.set()
    _wait_for_the_skill_thread()

    assert not get_manager().follower().connected


# --- failures and restarts (sim-B9) ----------------------------------------------


def test_sim_b9_unexpected_step_error_fails_the_proposal_and_disarms(client, monkeypatch):
    """agent_race_repro (b): v0.1 returned 500 and left the proposal "executing"."""

    def serial_timeout(*args, **kwargs):
        raise RuntimeError("simulated serial timeout")

    monkeypatch.setattr(get_manager(), "execute_approved_targets", serial_timeout)
    proposal, lease_id = _start(client)

    final = _wait_terminal(client, proposal["id"])

    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "error"
    assert "simulated serial timeout" in final["result"]["error"]
    assert final["result"]["end_state"] == "holding"
    assert _holding()
    assert _approve(client, final, lease_id).status_code == 409


def _break_start(patch: pytest.MonkeyPatch, stage: str) -> None:
    """Make AgentExecutor.start() raise a non-ControlError at ``stage``."""
    store_type = type(get_store())
    original = store_type.transition_proposal

    def transition(self, proposal_id, *, expected, status, result=None):
        if status != "executing":
            return original(self, proposal_id, expected=expected, status=status, result=result)
        if stage == "claim":
            raise sqlite3.OperationalError("database is locked")
        original(self, proposal_id, expected=expected, status=status, result=result)
        raise sqlite3.OperationalError("database is locked")  # committed, then the read failed

    def preview(targets):
        raise RuntimeError("simulated preview bug")

    if stage == "preview":
        patch.setattr(get_manager(), "preview_targets", preview)
    else:
        patch.setattr(store_type, "transition_proposal", transition)


@pytest.mark.parametrize("stage", ["preview", "claim", "after_claim"])
def test_sim_b9_an_unexpected_error_before_arming_releases_the_run_slot(client, monkeypatch, stage):
    """Gate review (agent_slot_leak): a store error kept the only run slot.

    Every later execute was refused with 409 until a restart, and cancelling
    the proposal did not free the slot.
    """
    lease_id = _lease(client)
    proposal = _propose(client)
    token = _token(client, proposal, lease_id)
    with monkeypatch.context() as patch:
        _break_start(patch, stage)
        with pytest.raises((RuntimeError, sqlite3.OperationalError)):
            _execute(client, proposal["id"], lease_id, token)

    assert agent_executor.get_executor().active_proposal_id() is None
    assert get_manager().control_state()["armed"] is False
    if stage == "after_claim":
        # No run owns the claimed proposal; cancel ends it as an orphan.
        assert _get(client, proposal["id"])["status"] == "executing"
        cancelled = client.post(f"/api/agent/proposals/{proposal['id']}/cancel")
        assert cancelled.json()["status"] == "cancelled"
        assert cancelled.json()["result"]["error"] == "cancelled with no live run"
    else:
        assert _get(client, proposal["id"])["status"] == "proposed"

    other = _propose(client, "close_gripper")
    started = _execute(client, other["id"], lease_id, _token(client, other, lease_id))
    assert started.status_code == 202, started.text
    assert _wait_terminal(client, other["id"])["status"] == "executed"


def test_sim_b9_proposals_left_executing_fail_on_restart():
    with TestClient(create_app()) as before:
        proposal = _propose(before)
        claimed = get_store().transition_proposal(
            proposal["id"], expected=("proposed",), status="executing", result={}
        )
        assert claimed is not None
    reset_manager()
    reset_store()

    with TestClient(create_app()) as after:
        recovered = _get(after, proposal["id"])
        events = _agent_events(after)

    assert recovered["status"] == "failed"
    assert recovered["result"]["error"] == "interrupted by restart"
    assert recovered["result"]["reached"] is False
    assert any(
        event["type"] == "agent.failed" and event["detail"]["proposal_id"] == proposal["id"]
        for event in events
    )


def test_sim_b9_server_shutdown_ends_an_active_run():
    with TestClient(create_app()) as client:
        get_manager().operator_heartbeat(OPERATOR)
        _freeze_arm()
        proposal, _ = _start(client)
        time.sleep(0.1)

    final = get_store().get_proposal(proposal["id"])
    assert final is not None
    assert final["status"] == "failed"
    assert final["result"]["stop_reason"] == "shutdown"
    # The executor stops first (its run halts and holds); the manager's
    # shutdown Collapse then lowers the arm and turns the motors off.
    assert final["result"]["end_state"] == "holding"
    assert final["result"]["steps"] >= 1
    assert not get_manager().follower().connected


# --- audit privacy ------------------------------------------------------------------


def test_c8_agent_events_carry_no_token_lease_or_operator(client):
    lease_id = _lease(client)
    proposal = _propose(client)
    token = _token(client, proposal, lease_id)
    assert _execute(client, proposal["id"], lease_id, token).status_code == 202
    _wait_terminal(client, proposal["id"])

    events = _agent_events(client)

    assert {"agent.proposed", "agent.approved", "agent.executing", "agent.executed"} <= {
        event["type"] for event in events
    }
    blob = json.dumps(events)
    assert token not in blob
    assert lease_id not in blob
    assert OPERATOR not in blob

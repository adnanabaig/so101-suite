"""Safety and persistence contract tests for the v0.0 control plane."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.config import get_settings  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.store import get_store, reset_store  # noqa: E402
from so101.training.run_manifest import RUN_MANIFEST_NAME  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        from so101.services.robot_manager import get_manager

        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _lease(client: TestClient, source: str = "test", ttl: float = 5.0) -> dict:
    response = client.post("/api/control/lease", json={"source": source, "ttl_seconds": ttl})
    assert response.status_code == 201, response.text
    return response.json()


def _arm(client: TestClient, lease_id: str) -> None:
    response = client.post("/api/control/arm", json={"lease_id": lease_id})
    assert response.status_code == 200, response.text
    assert response.json()["armed"] is True


def _command(
    client: TestClient,
    lease_id: str,
    *,
    source: str = "test",
    sequence: int = 0,
    sent_at: str | float | None = None,
    deadman: bool = True,
) -> object:
    return client.post(
        "/api/control/command",
        json={
            "lease_id": lease_id,
            "source": source,
            "sequence": sequence,
            "sent_at": sent_at or datetime.now(timezone.utc).isoformat(),
            "ttl_ms": 500,
            "deadman": deadman,
            "targets": {"shoulder_pan": 5.0, "gripper": 60.0},
        },
    )


def _wait_for(predicate, timeout_s: float = 1.0) -> bool:
    """Poll against a deadline instead of sleeping a fixed time (devex-B4)."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _eval_model() -> str:
    """A registered mock checkpoint for eval API contract tests."""

    run = get_settings().outputs_dir / "control_plane_eval"
    checkpoint = run / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text('{"type":"act","mock":true}', encoding="utf-8")
    (run / RUN_MANIFEST_NAME).write_text('{"fps":30}', encoding="utf-8")
    return str(checkpoint)


def test_estop_is_latched_across_start_and_arm(client: TestClient):
    assert client.post("/api/robot/estop").status_code == 200
    response = client.post(
        "/api/teleop/start",
        json={"fps": 30, "operator_id": "test-operator"},
    )
    assert response.status_code == 409
    assert client.get("/api/robot/status").json()["estopped"] is True

    lease = _lease(client)
    response = client.post("/api/control/arm", json={"lease_id": lease["id"]})
    assert response.status_code == 409
    assert client.get("/api/control/state").json()["safety_state"] == "estopped"

    # C2: a reset names the operator and confirms.
    reset = client.post(
        "/api/robot/estop/reset", json={"operator_id": "test-operator", "confirm": True}
    )
    assert reset.status_code == 200
    _arm(client, lease["id"])


def test_estop_reports_unconfirmed_torque_off(client: TestClient, monkeypatch):
    from so101.services.robot_manager import get_manager

    manager = get_manager()
    lease = manager.acquire_lease("unconfirmed", 5)
    manager.arm_control(lease["id"])
    # Fault injection replaces the v0.1 monkeypatch of the removed
    # _force_deenergize: the follower bus stops acknowledging torque-off.
    manager.follower().bus.fail_torque_off = True
    response = client.post("/api/robot/estop")
    assert response.status_code == 503
    assert "cut physical power" in response.json()["detail"]
    assert client.get("/api/robot/status").json()["estopped"] is True


def test_legacy_loop_requires_and_tracks_owning_operator(client: TestClient, manager_clock):
    from so101.services.robot_manager import get_manager

    manager = get_manager()
    manager.operator_timeout_ms = 100
    rejected = client.post(
        "/api/teleop/start",
        json={"fps": 60, "operator_id": "no-heartbeat"},
    )
    assert rejected.status_code == 409
    manager.operator_heartbeat("owner-a")
    response = client.post(
        "/api/teleop/start",
        json={"fps": 60, "operator_id": "owner-a"},
    )
    assert response.status_code == 200
    # A passive second tab cannot supervise an operation it did not start.
    manager.operator_heartbeat("other-tab")
    # The loop's pacing sleeps move the injected clock past 100 ms (devex-B4).
    # The watchdog records the event only after the loop has stopped, so wait
    # for both instead of reading the events the moment the mode turns idle.
    assert _wait_for(
        lambda: (
            client.get("/api/robot/status").json()["mode"] == "idle"
            and any(
                event["type"] == "runtime.operator_timeout"
                for event in client.get("/api/events").json()["events"]
            )
        )
    )
    # Lost supervision is a Halt (design doc §5): the loop stopped and the arm
    # holds; the grace period then turns it into a Collapse.
    assert client.get("/api/robot/status").json()["stop_state"] == "halted"


def test_lease_conflict_heartbeat_release_and_expiry(client: TestClient, manager_clock):
    lease = _lease(client, ttl=0.15)
    conflict = client.post("/api/control/lease", json={"source": "other", "ttl_seconds": 5})
    assert conflict.status_code == 409
    assert client.post(f"/api/control/lease/{lease['id']}/heartbeat").status_code == 200
    assert client.delete(f"/api/control/lease/{lease['id']}").status_code == 200

    short = _lease(client, ttl=0.1)
    manager_clock.advance(0.2)
    assert _wait_for(lambda: client.get("/api/control/state").json()["lease"] is None)
    state = client.get("/api/control/state").json()
    assert state["lease"] is None
    assert state["armed"] is False
    assert state["safety_state"] == "lease_expired"
    assert client.post(f"/api/control/lease/{short['id']}/heartbeat").status_code == 409


def test_command_rejects_deadman_stale_and_out_of_order(client: TestClient, manager_clock):
    lease = _lease(client)
    _arm(client, lease["id"])

    no_deadman = _command(client, lease["id"], deadman=False)
    assert no_deadman.status_code == 409

    stale = _command(
        client,
        lease["id"],
        sent_at=(datetime.now(timezone.utc) - timedelta(seconds=2)).timestamp(),
    )
    assert stale.status_code == 409

    accepted = _command(
        client,
        lease["id"],
        sent_at=datetime.now(timezone.utc).timestamp(),
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["ok"] is True

    repeated = _command(client, lease["id"], sequence=0)
    assert repeated.status_code == 409
    wrong_source = _command(client, lease["id"], source="not-owner", sequence=1)
    assert wrong_source.status_code == 409


def test_observe_seeds_manual_ui_without_enabling_torque(client: TestClient):
    lease = _lease(client, source="manual-ui")
    response = client.post(
        "/api/control/observe",
        json={"lease_id": lease["id"]},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["observation"]) == 6
    assert client.get("/api/robot/status").json()["follower_connected"] is False


def test_atomic_first_command_rejects_stale_before_arming(client: TestClient):
    lease = _lease(client, source="manual-ui")
    response = client.post(
        "/api/control/arm-command",
        json={
            "lease_id": lease["id"],
            "source": "manual-ui",
            "sequence": 0,
            "sent_at": (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),
            "ttl_ms": 300,
            "deadman": True,
            "targets": {"shoulder_pan": 20.0},
        },
    )
    assert response.status_code == 409
    assert client.get("/api/control/state").json()["armed"] is False
    assert client.get("/api/robot/status").json()["follower_connected"] is False


def test_disarm_cancels_slow_in_progress_arm(client: TestClient, monkeypatch):
    from so101.services.robot_manager import ControlRejected, StopKind, get_manager

    manager = get_manager()
    lease = manager.acquire_lease("slow-arm", 5)
    follower = manager.follower()
    original_connect = follower.connect
    entered = threading.Event()
    proceed = threading.Event()
    outcome: list[BaseException | str] = []

    def slow_connect(*, enable_torque=True):
        entered.set()
        proceed.wait(timeout=2)
        return original_connect(enable_torque=enable_torque)

    monkeypatch.setattr(follower, "connect", slow_connect)

    def run_arm():
        try:
            manager.arm_control(lease["id"])
            outcome.append("armed")
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run_arm)
    worker.start()
    assert entered.wait(timeout=1)
    revoked = threading.Event()
    revoke = manager._revoke_authority_locked
    stop_errors: list[BaseException] = []

    def observe_revoke(**kwargs):
        result = revoke(**kwargs)
        revoked.set()
        return result

    monkeypatch.setattr(manager, "_revoke_authority_locked", observe_revoke)

    def halt():
        try:
            manager.request_stop(StopKind.HALT, reason="test release", strict=True)
        except BaseException as exc:
            stop_errors.append(exc)

    stopper = threading.Thread(target=halt)
    stopper.start()
    try:
        assert revoked.wait(timeout=1)
    finally:
        proceed.set()
        worker.join(timeout=2)
        stopper.join(timeout=2)
    assert not worker.is_alive() and not stopper.is_alive()
    assert not stop_errors
    assert outcome and isinstance(outcome[0], ControlRejected)
    assert manager.control_state()["armed"] is False
    assert follower.torque_state == "off"


def _holding(client: TestClient) -> None:
    """Design doc §5: a Halt disarms and holds position with the motors on.

    The watchdog disarms first and writes the hold just after, so wait for it.
    """

    assert _wait_for(lambda: client.get("/api/robot/status").json()["stop_state"] == "halted")
    status = client.get("/api/robot/status").json()
    assert status["stop_state"] == "halted"
    assert status["torque_state"] == "on"
    assert status["follower_connected"] is True


def test_command_watchdog_disarms_and_holds(client: TestClient, manager_clock):
    lease = _lease(client)
    _arm(client, lease["id"])
    assert _command(client, lease["id"]).status_code == 200
    manager_clock.advance(0.65)
    assert _wait_for(lambda: client.get("/api/control/state").json()["armed"] is False)
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "watchdog_timeout"
    _holding(client)


def test_arm_without_first_command_times_out_and_holds(client: TestClient, manager_clock):
    lease = _lease(client)
    _arm(client, lease["id"])
    manager_clock.advance(0.65)
    assert _wait_for(lambda: client.get("/api/control/state").json()["armed"] is False)
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "watchdog_timeout"
    _holding(client)


def test_global_soft_stop_disarms_manual_control_immediately(client: TestClient):
    lease = _lease(client)
    _arm(client, lease["id"])
    assert client.get("/api/robot/status").json()["follower_connected"] is True
    assert client.post("/api/robot/stop").status_code == 200
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "disarmed"
    _holding(client)


def test_capabilities_diagnostics_simulation_and_tracking(client: TestClient):
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    payload = capabilities.json()
    assert payload["system_version"]
    assert {item["id"] for item in payload["capabilities"]} >= {
        "control",
        "diagnostics",
        "simulation",
        "tracking",
        "agent",
    }

    diagnostics = client.get("/api/diagnostics")
    assert diagnostics.status_code == 200
    assert diagnostics.json()["checks"]
    assert diagnostics.json()["ports"]

    scenarios = client.get("/api/simulation/scenarios").json()
    assert {scenario["id"] for scenario in scenarios} >= {"tabletop_pick"}
    assert (
        client.post("/api/simulation/reset", json={"scenario_id": "tabletop_pick"}).status_code
        == 200
    )

    low = client.post(
        "/api/tracking/preview",
        json={"source": "camera", "confidence": 0.2, "wrist": {"x": 0.2}},
    ).json()
    assert low["accepted"] is False
    preview = client.post(
        "/api/tracking/preview",
        json={
            "source": "camera",
            "confidence": 0.9,
            "wrist": {"x": 0.25, "y": -0.25},
            "pinch": 0.75,
        },
    )
    assert preview.status_code == 200
    assert preview.json()["accepted"] is True
    assert set(preview.json()["targets"]) >= {"shoulder_pan", "shoulder_lift", "gripper"}


def test_tracking_atomic_first_frame_reports_actual_limited_command(
    client: TestClient, manager_clock
):
    lease = _lease(client, source="browser-mediapipe")
    response = client.post(
        "/api/tracking/arm-execute",
        json={
            "source": "browser-mediapipe",
            "confidence": 0.95,
            "intent": {"normalized_x": 1.0},
            "execute": True,
            "lease_id": lease["id"],
            "sequence": 0,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "ttl_ms": 300,
            "deadman": True,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["desired_targets"]["shoulder_pan"] == 90.0
    assert payload["commanded_targets"]["shoulder_pan"] < 90.0
    assert payload["targets"] == payload["commanded_targets"]
    # C5: disarm is owner-scoped and needs the lease id.
    assert client.post("/api/control/disarm", json={"lease_id": lease["id"]}).status_code == 200


def test_tracking_low_confidence_never_arms(client: TestClient):
    lease = _lease(client, source="browser-mediapipe")
    response = client.post(
        "/api/tracking/arm-execute",
        json={
            "source": "browser-mediapipe",
            "confidence": 0.5,
            "intent": {"normalized_x": 0.1},
            "execute": True,
            "lease_id": lease["id"],
            "sequence": 0,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "ttl_ms": 300,
            "deadman": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["accepted"] is False
    assert client.get("/api/control/state").json()["armed"] is False


def test_agent_proposal_requires_approval_and_armed_lease(client: TestClient):
    proposal = client.post(
        "/api/agent/proposals",
        json={
            "objective": "Open the gripper for inspection",
            "skill": "open_gripper",
            "constraints": {
                "max_duration_seconds": 2,
                "require_deadman": False,
                "allow_motion": True,
            },
        },
    )
    assert proposal.status_code == 201, proposal.text
    proposal_id = proposal.json()["id"]
    assert proposal.json()["status"] == "proposed"

    lease = _lease(client, source="agent-test")
    denied = client.post(
        f"/api/agent/proposals/{proposal_id}/execute",
        json={"lease_id": lease["id"], "approved": False},
    )
    assert denied.status_code == 422
    # C8: approval is a single-use token for the reviewed target digest, and
    # execution runs server-side until the proposal reaches a terminal state.
    approved = client.post(
        f"/api/agent/proposals/{proposal_id}/approve",
        json={
            "lease_id": lease["id"],
            "operator_id": "test-operator",
            "target_digest": proposal.json()["target_digest"],
        },
    )
    assert approved.status_code == 200, approved.text
    started = client.post(
        f"/api/agent/proposals/{proposal_id}/execute",
        json={"lease_id": lease["id"], "approval_token": approved.json()["approval_token"]},
    )
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "executing"
    deadline = time.monotonic() + 5
    status = "executing"
    while status == "executing" and time.monotonic() < deadline:
        time.sleep(0.02)
        client.post("/api/robot/operator-heartbeat", json={"operator_id": "test-operator"})
        status = client.get(f"/api/agent/proposals/{proposal_id}").json()["status"]
    assert status == "executed"

    simple = client.post(
        "/api/agent/proposals", json={"instruction": "inspect the empty workspace"}
    )
    assert simple.status_code == 201
    assert simple.json()["skill"] == "inspect_scene"
    assert simple.json()["targets"] == {}
    assert simple.json()["constraints"]["allow_motion"] is False
    cancelled = client.post(f"/api/agent/proposals/{simple.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_agent_accepts_the_deadman_constraint(client: TestClient):
    # C8 implements require_deadman as server-checked holds, so v0.1's 422
    # for this constraint is gone.
    response = client.post(
        "/api/agent/proposals",
        json={
            "objective": "Open gripper continuously",
            "skill": "open_gripper",
            "constraints": {
                "max_duration_seconds": 2,
                "require_deadman": True,
                "allow_motion": True,
            },
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["constraints"]["require_deadman"] is True


def test_configured_planner_can_only_create_reviewable_bounded_proposal(
    client: TestClient,
    monkeypatch,
):
    from so101.agents.planner import PlannerProposal

    class Planner:
        def propose(self, objective, context):
            assert objective == "Prepare a safe gripper release"
            assert "safety_state" in context
            return PlannerProposal(
                skill="open_gripper",
                rationale="Named release is sufficient.",
                max_duration_seconds=3,
                provider="test_provider",
                model="test-model",
            )

    monkeypatch.setattr(
        "so101.services.routers.agent.get_planner",
        lambda _selection: Planner(),
    )
    response = client.post(
        "/api/agent/proposals",
        json={
            "objective": "Prepare a safe gripper release",
            "planner": "configured",
            "constraints": {
                "max_duration_seconds": 10,
                "require_deadman": False,
                "allow_motion": True,
            },
        },
    )
    assert response.status_code == 201, response.text
    proposal = response.json()
    assert proposal["skill"] == "open_gripper"
    assert proposal["targets"] == {"gripper": 80.0}
    assert proposal["constraints"]["max_duration_seconds"] == 3
    assert proposal["constraints"]["planner"]["provider"] == "test_provider"
    assert proposal["status"] == "proposed"


def test_events_and_proposals_survive_store_reopen(client: TestClient):
    created = client.post(
        "/api/agent/proposals",
        json={"instruction": "inspect the empty workspace"},
    ).json()
    store_path = get_store().path
    reset_store()
    assert get_store().path == store_path
    assert get_store().get_proposal(created["id"])["objective"] == created["objective"]
    events = client.get("/api/events").json()["events"]
    assert any(event["type"] == "agent.proposed" for event in events)


def test_eval_fast_score_completes_and_persists_report(client: TestClient):
    started = client.post(
        "/api/eval/start",
        json={
            "model_path": _eval_model(),
            "n_episodes": 1,
            "max_seconds": 0.1,
            "operator_id": "test-operator",
        },
    )
    assert started.status_code == 200, started.text
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get("/api/eval/state").json()
        if state["phase"] == "awaiting_score":
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"eval did not await score: {state}")

    scored = client.post("/api/eval/score", json={"operator_id": "test-operator", "success": True})
    assert scored.status_code == 200
    while time.monotonic() < deadline:
        state = client.get("/api/eval/state").json()
        if state["phase"] == "done" and state["report_path"]:
            break
        time.sleep(0.01)
    assert state["phase"] == "done"
    assert state["successes"] == 1
    assert state["success_rate"] == 1.0
    assert Path(state["report_path"]).is_file()


def test_eval_requires_explicit_workspace_reset_before_next_rollout(client: TestClient):
    started = client.post(
        "/api/eval/start",
        json={
            "model_path": _eval_model(),
            "n_episodes": 2,
            "max_seconds": 0.05,
            "operator_id": "test-operator",
        },
    )
    assert started.status_code == 200
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get("/api/eval/state").json()
        if state["phase"] == "awaiting_score":
            break
        time.sleep(0.01)
    assert state["episode"] == 1
    scored = client.post("/api/eval/score", json={"operator_id": "test-operator", "success": False})
    assert scored.status_code == 200
    while time.monotonic() < deadline:
        state = client.get("/api/eval/state").json()
        if state["phase"] == "awaiting_reset":
            break
        time.sleep(0.01)
    assert state["phase"] == "awaiting_reset"
    assert client.get("/api/robot/status").json()["mode"] == "idle"
    time.sleep(0.1)
    assert client.get("/api/eval/state").json()["episode"] == 1

    continued = client.post("/api/eval/continue", json={"operator_id": "test-operator"})
    assert continued.status_code == 200
    while time.monotonic() < deadline:
        state = client.get("/api/eval/state").json()
        if state["phase"] == "awaiting_score" and state["episode"] == 2:
            break
        time.sleep(0.01)
    assert state["episode"] == 2
    client.post("/api/eval/stop")


def test_eval_aborts_instead_of_scoring_an_interrupted_rollout(client: TestClient):
    started = client.post(
        "/api/eval/start",
        json={
            "model_path": _eval_model(),
            "n_episodes": 1,
            "max_seconds": 2,
            "operator_id": "test-operator",
        },
    )
    assert started.status_code == 200
    time.sleep(0.1)
    assert client.post("/api/robot/stop").status_code == 200
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get("/api/eval/state").json()
        if state["phase"] == "aborted" and state["report_path"]:
            break
        time.sleep(0.01)
    assert state["phase"] == "aborted"
    assert state["successes"] == 0

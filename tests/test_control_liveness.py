"""What the UI may treat as live (frontend F8, F10, F13; docs/audit/v0.1/frontend.md).

- C6: every /ws/telemetry frame carries ``observation_live``. It is true only
  while an active loop or an armed session sampled the joints within 500 ms.
- C7: GET /api/robot/status carries ``active_model``, the checkpoint path
  driving the current rollout (and so the current eval trial), else null.
- C10: GET /api/video/snapshot?cam=<id> returns one fresh frame with
  ``Cache-Control: no-store``, or 503 JSON when none is fresh.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.policy.server import MockPolicy  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import Mode, get_manager  # noqa: E402

OPERATOR = "test-operator"


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _wait_for(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _lease(client: TestClient) -> str:
    response = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 30})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _live() -> bool:
    return get_manager().telemetry()["observation_live"]


# --- C6: observation_live -------------------------------------------------------


def test_c6_every_telemetry_frame_says_whether_the_observation_is_live(client: TestClient):
    with client.websocket_connect("/ws/telemetry") as ws:
        frames = [ws.receive_json() for _ in range(3)]

    for frame in frames:
        assert frame["observation_live"] is False
        assert frame["mode"] == "idle"


def _collapse_finished() -> bool:
    return get_manager().status()["stop_state"] == "none"


def test_c6_a_running_loop_is_live_and_its_end_is_not(client: TestClient):
    manager = get_manager()
    manager.start_teleop(30, operator_id=OPERATOR)
    try:
        assert _wait_for(_live)
        frame = manager.telemetry()
        assert frame["observation"] and frame["sample_age_ms"] <= 500
    finally:
        manager.soft_stop("test")

    assert manager.mode is Mode.IDLE
    # The loop's end is a Halt (design doc §5): the energized hold keeps being
    # sampled, so its pose stays live until it ends. A Collapse to rest ends it.
    assert manager.status()["stop_state"] == "halted"
    assert manager.request_stop("collapse", reason="test").collapsing
    assert _wait_for(_collapse_finished)
    assert _live() is False


def test_c6_an_armed_session_is_live_until_it_disarms(client: TestClient):
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200

    assert _live() is True

    assert client.post("/api/control/disarm", json={"lease_id": lease_id}).status_code == 200
    # The owner disarm is a Collapse; its descent is sampled (live) until it
    # ends at rest with the motors off.
    assert _wait_for(_collapse_finished)
    assert _live() is False


def test_c6_a_sample_older_than_500_ms_is_not_live(client: TestClient):
    manager = get_manager()
    lease_id = _lease(client)
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200
    with manager._lock:
        manager._snapshot_monotonic -= 0.6

    assert _live() is False


def test_c6_an_observe_read_is_not_live(client: TestClient):
    lease_id = _lease(client)

    observed = client.post("/api/control/observe", json={"lease_id": lease_id})

    assert observed.status_code == 200, observed.text
    frame = get_manager().telemetry()
    assert frame["observation"], "the one-off read is still reported"
    assert frame["observation_live"] is False


# --- C7: active_model ------------------------------------------------------------


class _HeldPolicy(MockPolicy):
    """Keeps the rollout in its first inference until released."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def select_action(self, observation):
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().select_action(observation)


def test_c7_status_names_the_checkpoint_driving_the_rollout(client: TestClient):
    manager = get_manager()
    assert client.get("/api/robot/status").json()["active_model"] is None
    policy = _HeldPolicy()

    manager.start_rollout("models/act-cube", 30, operator_id=OPERATOR, policy=policy)
    try:
        assert policy.entered.wait(timeout=5)
        status = client.get("/api/robot/status").json()
        assert status["mode"] == "rollout"
        assert status["active_model"] == "models/act-cube"
    finally:
        policy.release.set()
        manager.soft_stop("test")

    assert client.get("/api/robot/status").json()["active_model"] is None


def test_c7_teleop_has_no_active_model(client: TestClient):
    manager = get_manager()
    manager.start_teleop(30, operator_id=OPERATOR)
    try:
        assert _wait_for(lambda: manager.mode is Mode.TELEOP)
        assert client.get("/api/robot/status").json()["active_model"] is None
    finally:
        manager.soft_stop("test")


# --- C10: video snapshot ------------------------------------------------------------

_MAGIC = {"image/jpeg": b"\xff\xd8", "image/png": b"\x89PNG"}


def _frame(cam: str, age_s: float = 0.0) -> None:
    manager = get_manager()
    with manager._lock:
        manager._frames = {cam: np.full((8, 8, 3), 120, dtype=np.uint8)}
        manager._frame_times = {cam: time.monotonic() - age_s}


def test_c10_a_fresh_frame_is_one_uncached_image(client: TestClient):
    _frame("wrist")

    response = client.get("/api/video/snapshot", params={"cam": "wrist"})

    assert response.status_code == 200, response.text
    content_type = response.headers["content-type"]
    assert content_type in _MAGIC
    assert response.content.startswith(_MAGIC[content_type])
    assert response.headers["cache-control"] == "no-store"


def test_c10_jpeg_whenever_an_encoder_is_installed(client: TestClient):
    if not any(_importable(name) for name in ("cv2", "PIL")):
        pytest.skip("no JPEG encoder installed; the dependency-free fallback is PNG")
    _frame("wrist")

    response = client.get("/api/video/snapshot", params={"cam": "wrist"})

    assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("cam, age_s", [("wrist", 2.0), ("missing", 0.0)])
def test_c10_no_fresh_frame_is_503_json(client: TestClient, cam: str, age_s: float):
    _frame("wrist", age_s)

    response = client.get("/api/video/snapshot", params={"cam": cam})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert cam in response.json()["detail"]
    assert response.headers["cache-control"] == "no-store"


def test_c10_the_camera_is_required(client: TestClient):
    assert client.get("/api/video/snapshot").status_code == 422


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True

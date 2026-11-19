"""The ``/api/simulation`` contract on both engines, plus the render endpoint.

Every contract test runs twice, with ``SO101_SIM_ENGINE=kinematic`` and with
``SO101_SIM_ENGINE=mujoco``; the MuJoCo half is skipped without the sim extra.
"""

from __future__ import annotations

import importlib.util
import struct
import threading
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.services.app import create_app
from so101.services.routers import simulation as simulation_router
from so101.services.store import get_store
from so101.simulation import capabilities as capabilities_module
from so101.simulation import selection as selection_module
from so101.simulation.selection import MujocoStatus

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
needs_mujoco = pytest.mark.skipif(not HAS_MUJOCO, reason="needs the [sim] extra (mujoco)")
ENGINES = [
    pytest.param("kinematic", id="kinematic"),
    pytest.param("mujoco", id="mujoco", marks=needs_mujoco),
]
BROKEN = MujocoStatus(False, True, "3.0.0", "RuntimeError: gladLoadGL error")
SLACK_S = 0.5  # request overhead on a busy machine; a hung probe used to cost 10 s a call
STATE_KEYS = {
    "backend",
    "backend_kind",
    "physics",
    "truth_model",
    "status",
    "scheduler",
    "playback_rate",
    "fixed_dt_s",
    "fixed_hz",
    "step_count",
    "sim_time_s",
    "generation",
    "updated_at",
    "joint_order",
    "joint_limits_deg",
    "telemetry",
    "kinematics",
    "scene",
    "physics_state",
}


def _use_engine(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    monkeypatch.setenv("SO101_SIM_ENGINE", engine)
    get_settings.cache_clear()
    simulation_router.reset_simulator()


@pytest.fixture(autouse=True)
def _forget_the_engine() -> Iterator[None]:
    yield
    simulation_router.reset_simulator()
    get_settings.cache_clear()


@pytest.fixture(params=ENGINES)
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    _use_engine(monkeypatch, request.param)
    return str(request.param)


@pytest.fixture
def client(engine: str) -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        reset = test_client.post(
            "/api/simulation/reset", json={"scenario_id": "tabletop_pick", "seed": 101}
        )
        assert reset.status_code == 200, reset.text
        yield test_client
        test_client.post("/api/simulation/pause")


def _without_timestamp(state: dict) -> dict:
    return {key: value for key, value in state.items() if key != "updated_at"}


def _png_size(payload: bytes) -> tuple[int, int]:
    assert payload[:8] == b"\x89PNG\r\n\x1a\n" and payload[12:16] == b"IHDR"
    width, height = struct.unpack(">II", payload[16:24])
    return width, height


# -- the shared contract ----------------------------------------------------------------
def test_state_scenarios_and_telemetry_name_the_engine(client: TestClient, engine: str):
    physics = engine == "mujoco"
    state = client.get("/api/simulation/state").json()
    assert STATE_KEYS <= set(state)
    assert (state["backend"], state["physics"]) == (engine, physics)
    assert state["backend_kind"] == ("physics" if physics else "kinematic")
    assert state["hardware_isolated"] is True and state["writes_motor_commands"] is False
    assert list(state["telemetry"]["measured"]) == list(JOINT_NAMES)
    if physics:
        assert state["physics_state"]["engine"] == "mujoco"
        assert set(state["physics_state"]["objects"]) == {"cube"}
    else:
        assert state["physics_state"] is None

    scenarios = client.get("/api/simulation/scenarios").json()
    assert {item["id"] for item in scenarios} == {"tabletop_pick", "bin_sort", "calibration_grid"}
    assert all(item["backend"] == engine and item["physics"] is physics for item in scenarios)
    assert [item["id"] for item in scenarios if item["active"]] == ["tabletop_pick"]

    telemetry = client.get("/api/simulation/telemetry").json()
    assert telemetry["source"] == "simulation"
    assert telemetry["measured"] == state["telemetry"]["measured"]


def test_targets_step_play_and_pause(client: TestClient):
    for bad in ({"shoulder_pan": 110.1}, {"fake_joint": 0.0}):
        rejected = client.post("/api/simulation/targets", json={"targets": bad})
        assert rejected.status_code == 422, bad
    assert client.post("/api/simulation/step", json={"steps": 0}).status_code == 422

    targeted = client.post("/api/simulation/targets", json={"targets": {"shoulder_pan": 20.0}})
    assert targeted.status_code == 200, targeted.text
    assert targeted.json()["telemetry"]["target"]["shoulder_pan"] == 20.0
    stepped = client.post("/api/simulation/step", json={"steps": 30}).json()
    assert stepped["step_count"] == 30
    assert stepped["sim_time_s"] == pytest.approx(0.5)
    assert 0.0 < stepped["telemetry"]["measured"]["shoulder_pan"] < 25.0

    assert client.post("/api/simulation/play", json={"rate": 9.0}).status_code == 422
    assert client.post("/api/simulation/play", json={"rate": 1.0}).json()["status"] == "playing"
    assert client.post("/api/simulation/step", json={"steps": 1}).status_code == 409
    assert client.post("/api/simulation/pause").json()["status"] == "paused"


def test_reset_logs_the_engine_that_ran_it(client: TestClient, engine: str):
    physics = engine == "mujoco"
    response = client.post("/api/simulation/reset", json={"scenario_id": "bin_sort", "seed": 7})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True and "seed 7" in payload["detail"]
    state = payload["state"]
    assert (state["scene"]["scenario_id"], state["step_count"]) == ("bin_sort", 0)
    assert len(state["scene"]["objects"]) == 4
    if physics:
        assert set(state["physics_state"]["objects"]) == {"blue_cube", "orange_cylinder"}
        assert set(state["physics_state"]["markers"]) == {"left_target", "right_target"}

    events = get_store().list_events(limit=20)  # newest first
    event = next(item for item in events if item["type"] == "simulation.reset")
    assert event["message"] == f"reset {engine} scenario bin_sort"
    assert event["detail"]["backend"] == engine
    assert event["detail"]["physics"] is physics
    assert event["detail"]["seed"] == 7

    missing = client.post("/api/simulation/reset", json={"scenario_id": "does_not_exist"})
    assert missing.status_code == 404


def test_preview_never_mutates_state_and_predicts_step(client: TestClient, engine: str):
    target = {"shoulder_pan": 30.0, "elbow_flex": -20.0, "gripper": 80.0}
    before = client.get("/api/simulation/state").json()
    response = client.post(
        "/api/simulation/trajectory/preview",
        json={"target": target, "duration_s": 0.5, "sample_hz": 60.0},
    )
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["backend"] == engine and preview["mutates_state"] is False
    assert preview["reached_tolerance"] == (0.5 if engine == "mujoco" else 1e-6)
    assert [point["step"] for point in preview["points"]] == list(range(31))
    after = client.get("/api/simulation/state").json()
    assert _without_timestamp(after) == _without_timestamp(before)

    stepped = client.post("/api/simulation/step", json={"steps": 30, "targets": target}).json()
    assert stepped["telemetry"]["measured"] == preview["points"][-1]["measured"]


def test_capabilities_name_the_engine_that_runs(client: TestClient, engine: str):
    physics = engine == "mujoco"
    report = client.get("/api/simulation/capabilities").json()
    assert report["active_backend"] == engine and report["physics_active"] is physics
    assert report["adapters"]["mujoco"]["active"] is physics
    assert report["adapters"]["kinematic"]["active"] is not physics
    assert report["adapters"]["mujoco"]["engine_setting"] == engine
    for name, adapter in report["adapters"].items():  # the UI reads these from every adapter
        assert {"active", "kind", "detail"} <= set(adapter), name
    if physics:
        assert report["adapters"]["mujoco"]["substeps_per_tick"] == 4

    items = client.get("/api/capabilities").json()["capabilities"]
    (simulation,) = [item for item in items if item["id"] == "simulation"]
    assert simulation["available"] is True
    assert ("MuJoCo physics" in simulation["detail"]) is physics
    assert ("No physics" in simulation["detail"]) is not physics


# -- the render endpoint ----------------------------------------------------------------
def test_render_needs_the_physics_engine(monkeypatch: pytest.MonkeyPatch):
    _use_engine(monkeypatch, "kinematic")
    with TestClient(create_app()) as client:
        response = client.get("/api/simulation/render")
        assert response.status_code == 503
        assert "MuJoCo engine" in response.json()["detail"]
        assert client.get("/api/simulation/render", params={"format": "gif"}).status_code == 422


@needs_mujoco
def test_render_returns_camera_frames(monkeypatch: pytest.MonkeyPatch):
    from so101.simulation.render import shared_renderer

    _use_engine(monkeypatch, "mujoco")
    with TestClient(create_app()) as client:
        for params in (
            {"camera": "ceiling"},
            {"width": 8},
            {"height": 2000},
            {"format": "gif"},
            {"quality": 5},
        ):
            assert client.get("/api/simulation/render", params=params).status_code == 422, params
        if not shared_renderer().available:
            assert client.get("/api/simulation/render").status_code == 503
            pytest.skip(f"no OpenGL here: {shared_renderer().error}")

        client.post("/api/simulation/step", json={"steps": 3})
        png = client.get(
            "/api/simulation/render",
            params={"camera": "wrist_cam", "width": 64, "height": 48, "format": "png"},
        )
        assert png.status_code == 200, png.text
        assert png.headers["content-type"] == "image/png"
        assert _png_size(png.content) == (64, 48)
        assert png.headers["cache-control"] == "no-store"
        assert png.headers["x-so101-sim-camera"] == "wrist_cam"
        assert png.headers["x-so101-sim-step"] == "3"
        assert png.headers["x-so101-sim-generation"] == "0"
        assert png.headers["x-so101-sim-time-s"] == "0.050000"

        default = client.get("/api/simulation/render")
        assert default.status_code == 200
        if default.headers["content-type"] == "image/jpeg":
            assert default.content[:2] == b"\xff\xd8"
        else:  # no JPEG encoder installed: PNG, never an error
            assert default.headers["content-type"] == "image/png"
            assert _png_size(default.content) == (320, 240)


@needs_mujoco
def test_a_hung_render_probe_never_blocks_capabilities_or_render(monkeypatch: pytest.MonkeyPatch):
    from so101.simulation import render as render_module

    gate = threading.Event()
    monkeypatch.setattr(render_module.FrameRenderer, "_probe", staticmethod(lambda: gate.wait(30)))
    monkeypatch.setattr(render_module, "PROBE_TIMEOUT_S", 0.5)
    hung = render_module.FrameRenderer()
    monkeypatch.setattr(render_module, "_shared", hung)
    _use_engine(monkeypatch, "mujoco")
    try:
        with TestClient(create_app()) as client:
            started = time.perf_counter()
            mujoco_adapter = client.get("/api/simulation/capabilities").json()["adapters"]["mujoco"]
            assert mujoco_adapter["render_status"] == "probing"  # the UI polls this: no wait
            assert mujoco_adapter["render_available"] is None
            assert "still being probed" in mujoco_adapter["detail"]
            response = client.get("/api/simulation/render")  # waits until the probe counts as hung
            assert response.status_code == 503
            assert "the GL probe has not returned" in response.json()["detail"]
            assert time.perf_counter() - started < 0.5 + SLACK_S

            started = time.perf_counter()  # from now on both answer at once
            mujoco_adapter = client.get("/api/simulation/capabilities").json()["adapters"]["mujoco"]
            assert mujoco_adapter["render_available"] is False
            assert mujoco_adapter["render_status"] == (
                "unavailable: render thread did not start within 0.5 s: "
                "the GL probe has not returned"
            )
            assert client.get("/api/simulation/render").status_code == 503
            assert time.perf_counter() - started < SLACK_S
    finally:
        gate.set()
        hung.close()


# -- engine selection ---------------------------------------------------------------------
def test_explicit_mujoco_that_cannot_run_answers_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(selection_module, "mujoco_status", lambda: BROKEN)
    monkeypatch.setattr(capabilities_module, "mujoco_status", lambda: BROKEN)
    _use_engine(monkeypatch, "mujoco")
    with TestClient(create_app()) as client:
        for method, path in (
            ("get", "/api/simulation/state"),
            ("get", "/api/simulation/telemetry"),
            ("get", "/api/simulation/scenarios"),
            ("get", "/api/simulation/render"),
            ("post", "/api/simulation/step"),
            ("post", "/api/simulation/reset"),
        ):
            response = client.request(method, path)
            assert response.status_code == 503, path
            assert "cannot run here" in response.json()["detail"], path

        report = client.get("/api/simulation/capabilities")
        assert report.status_code == 200
        body = report.json()
        assert (body["active_backend"], body["backend_kind"]) == ("none", "none")
        assert body["adapters"]["mujoco"]["available"] is False
        assert body["adapters"]["mujoco"]["error"] == BROKEN.error

        items = client.get("/api/capabilities").json()["capabilities"]
        (simulation,) = [item for item in items if item["id"] == "simulation"]
        assert simulation["available"] is False
        assert simulation["detail"].startswith("Simulation unavailable")
        assert client.get("/api/simulation/model").status_code == 200  # needs no MuJoCo


def test_auto_serves_the_kinematic_engine_when_mujoco_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(selection_module, "mujoco_status", lambda: BROKEN)
    monkeypatch.setattr(capabilities_module, "mujoco_status", lambda: BROKEN)
    _use_engine(monkeypatch, "auto")
    with TestClient(create_app()) as client:
        state = client.get("/api/simulation/state").json()
        assert (state["backend"], state["physics"]) == ("kinematic", False)
        mujoco_adapter = client.get("/api/simulation/capabilities").json()["adapters"]["mujoco"]
        assert mujoco_adapter["engine_setting"] == "auto"
        assert "gladLoadGL" in mujoco_adapter["selection_reason"]
        assert mujoco_adapter["available"] is False and mujoco_adapter["active"] is False

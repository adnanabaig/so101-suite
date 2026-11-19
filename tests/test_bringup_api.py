"""POST /api/preflight, GET /api/preflight/checks and /latest (mock + stubbed real)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.bringup import preflight as pf  # noqa: E402
from so101.bringup.lerobot_surface import LeRobotSurface  # noqa: E402
from so101.bringup.ownership import ServerProbe  # noqa: E402
from so101.config import get_settings  # noqa: E402
from so101.hardware import lerobot_paths  # noqa: E402
from so101.hardware.safety import DEFAULT_LIMITS  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.routers import preflight as preflight_router  # noqa: E402


def no_server(settings: Any) -> ServerProbe:
    return ServerProbe(running=False, url="http://127.0.0.1:8816/api/robot/status", detail="none")


@pytest.fixture(autouse=True)
def _quiet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    monkeypatch.setattr(preflight_router, "_sleep", lambda seconds: None)
    get_settings.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No lifespan: the manager is created lazily by the first request that needs it.
    yield TestClient(create_app())


class StubManager:
    """Just enough RobotManager for run_api_preflight on the real backend."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.guard = SimpleNamespace(limits=dict(DEFAULT_LIMITS))
        self.fault = None

    def status(self) -> dict[str, Any]:
        return {"mode": "idle", "estopped": False, "live_cameras": []}


def test_preflight_route_runs_a_labeled_rehearsal_and_saves_it(client: TestClient) -> None:
    response = client.post("/api/preflight")
    assert response.status_code == 200
    body = response.json()
    assert body["origin"] == "api" and body["backend"] == "mock" and body["rehearsal"] is True
    assert body["bus_access"] == "synthetic" and body["ready_for_hardware"] is False
    assert body["notes"][0].startswith("REHEARSAL")
    keys = {check["key"] for check in body["checks"]}
    assert {"motors.follower", "motors.leader", "rest_pose.follower", "estop"} <= keys
    assert all(check["rehearsal"] for check in body["checks"])

    settings = get_settings()
    assert pf.latest_report_path(settings).exists()
    assert pf.latest_report_path(settings, "api").exists()
    latest = client.get("/api/preflight/latest")
    assert latest.status_code == 200 and latest.json()["created_at"] == body["created_at"]
    missing = client.get("/api/preflight/latest", params={"origin": "cli"})
    assert missing.status_code == 404 and "from the cli" in missing.json()["detail"]


def test_preflight_route_runs_one_arm_without_saving(client: TestClient) -> None:
    body = client.post("/api/preflight", params={"arm": "follower", "save": "false"}).json()
    assert body["arms"] == ["follower"]
    assert not any(check["arm"] == "leader" for check in body["checks"])
    assert not pf.reports_dir(get_settings()).exists()
    response = client.get("/api/preflight/latest")
    assert response.status_code == 404 and "so101 preflight" in response.json()["detail"]


def test_preflight_route_rejects_an_unknown_arm(client: TestClient) -> None:
    assert client.post("/api/preflight", params={"arm": "left"}).status_code == 422
    assert client.get("/api/preflight/latest", params={"origin": "ui"}).status_code == 422


def test_preflight_checks_catalog_is_in_run_order(client: TestClient) -> None:
    body = client.get("/api/preflight/checks").json()
    assert [item["id"] for item in body] == [spec.id for spec in pf.CHECKS]
    by_id = {item["id"]: item for item in body}
    assert by_id["motors"]["needs_bus"] and by_id["motors"]["per_arm"]
    assert not by_id["disk_space"]["needs_bus"]


def test_preflight_route_reports_the_server_estop(client: TestClient) -> None:
    from so101.services.robot_manager import get_manager

    get_manager().estop()
    try:
        body = client.post("/api/preflight", params={"save": "false"}).json()
    finally:
        get_manager().reset_estop(operator_id="test-operator")
    estop = next(check for check in body["checks"] if check["id"] == "estop")
    assert estop["status"] == "warn" and "latched" in estop["message"]
    assert estop["source"] == "server"


def test_preflight_route_on_the_real_backend_defers_bus_checks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    stub = StubManager(get_settings())
    monkeypatch.setattr(preflight_router, "get_manager", lambda: stub)

    body = client.post("/api/preflight").json()
    assert body["backend"] == "real" and body["rehearsal"] is False
    assert body["bus_access"] == "deferred"
    assert body["bus_access_detail"] == pf.WAVE2_BUS_MESSAGE
    motors = [check for check in body["checks"] if check["id"] == "motors"]
    assert motors and all(check["status"] == "skip" for check in motors)
    assert all(check["message"] == pf.WAVE2_BUS_MESSAGE for check in motors)
    assert body["ready_for_hardware"] is False and body["exit_code"] != 0
    saved = pf.load_latest_report(stub.settings, "api")
    assert saved is not None and saved.backend == "real"


def test_an_api_run_does_not_hide_the_last_cli_report(client: TestClient) -> None:
    settings = get_settings()
    cli_report = pf.run_cli_preflight(
        settings,
        server_probe=no_server,
        lerobot_prober=LeRobotSurface,
        sleep=lambda seconds: None,
    )
    api_body = client.post("/api/preflight").json()

    cli = client.get("/api/preflight/latest", params={"origin": "cli"}).json()
    assert cli["origin"] == "cli" and cli["created_at"] == cli_report.created_at
    assert cli["bus_access"] == "synthetic"
    newest = client.get("/api/preflight/latest").json()
    assert newest["origin"] == "api" and newest["created_at"] == api_body["created_at"]


# --- running preflight is a POST behind the Origin guard ---------------------------------

EVIL = "https://evil.example"
RUN_PATHS = ["/api/preflight", "/api/preflight/bus"]


def _bus_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every idle-bus diagnostic the manager is asked for."""

    from so101.services.robot_manager import RobotManager

    reads: list[str] = []
    original = RobotManager.run_idle_bus_diagnostic

    def counted(self, target, *args, **kwargs):
        reads.append(target)
        return original(self, target, *args, **kwargs)

    monkeypatch.setattr(RobotManager, "run_idle_bus_diagnostic", counted)
    return reads


@pytest.mark.parametrize("path", RUN_PATHS)
def test_a_get_to_a_preflight_run_path_is_405_and_runs_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    reads = _bus_reads(monkeypatch)

    response = client.get(path, headers={"Origin": EVIL})

    assert response.status_code == 405, response.text
    assert response.headers["allow"] == "POST"
    assert "POST" in response.json()["detail"]
    assert reads == []
    assert not pf.reports_dir(get_settings()).exists()


@pytest.mark.parametrize("path", RUN_PATHS)
@pytest.mark.parametrize(
    "headers",
    [{"Origin": EVIL}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}],
    ids=["foreign", "null", "cross-site"],
)
def test_a_cross_origin_preflight_run_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str, headers: dict
) -> None:
    reads = _bus_reads(monkeypatch)

    response = client.post(path, headers=headers)

    assert response.status_code == 403, response.text
    assert reads == []
    assert not pf.reports_dir(get_settings()).exists()


def test_a_same_origin_preflight_run_still_works(client: TestClient) -> None:
    response = client.post("/api/preflight", headers={"Origin": "http://testserver"})
    assert response.status_code == 200, response.text
    assert response.json()["bus_access"] == "synthetic"


def test_the_get_405_holds_when_the_ui_fallback_is_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SPA fallback is a catch-all GET; it must not answer the old run paths."""

    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>so101</title>")
    monkeypatch.setenv("SO101_UI_DIR", str(ui))
    get_settings.cache_clear()
    with TestClient(create_app()) as ui_client:
        assert ui_client.get("/").status_code == 200  # the UI is mounted
        for path in RUN_PATHS:
            response = ui_client.get(path)
            assert response.status_code == 405, (path, response.text)


def test_the_cli_asks_a_running_server_with_a_post_and_no_origin() -> None:
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404, json={"detail": "Not Found"})

    probe = ServerProbe(running=True, url="http://127.0.0.1:8816/api/robot/status", detail="")
    pf.fetch_server_bus_facts(probe, pf.ARMS, "mock", transport=httpx.MockTransport(handler))

    (request,) = seen
    assert request.method == "POST" and request.url.path == "/api/preflight/bus"
    assert "origin" not in request.headers and "sec-fetch-site" not in request.headers

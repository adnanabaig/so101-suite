"""Web gate (contract C1): Origin/Host guard, WebSocket Origin check, SPA containment.

Bug IDs refer to docs/audit/v0.1/{safety,frontend,devex}.md. The "repro NN" notes
name the audit's reproduction scripts these tests encode.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from so101 import config
from so101.services import app as app_module
from so101.services.app import create_app
from so101.services.robot_manager import get_manager

EVIL = "http://evil.example"
REJECTED = {"detail": "cross-origin request rejected"}
INDEX = "<!doctype html><title>SO101-INDEX</title>"
SECRET = "TOP-SECRET-TOKEN"


def _settings_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _heartbeat(client: TestClient, method: str = "POST", **kwargs):
    return client.request(method, "/api/robot/operator-heartbeat", **kwargs)


# --- Origin rule on state-changing methods ------------------------------------


def test_safety_b3_cross_origin_estop_reset_rejected_and_latch_holds(client):
    """Repro 01: a foreign page could clear the latched E-stop with a body-less POST."""
    assert client.post("/api/robot/estop").status_code == 200
    for path in ("/api/robot/estop/reset", "/api/robot/fault/reset"):
        response = client.post(path, headers={"Origin": EVIL})
        assert response.status_code == 403
        assert response.json() == REJECTED
    assert get_manager().guard.estopped


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", None],
    ids=["text-plain-simple-request", "no-content-type"],
)
def test_safety_b3_cross_origin_simple_request_bodies_rejected(client, content_type):
    headers = {"Origin": EVIL}
    if content_type:
        headers["Content-Type"] = content_type
    response = _heartbeat(client, content=b'{"operator_id": "evil"}', headers=headers)
    assert response.status_code == 403
    assert response.json() == REJECTED


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": EVIL},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://testserver.evil.example"},
        {"Origin": "http://localhost:5174"},
        {"Origin": "https://testserver"},
        {"Origin": "https://testserver:80"},
        {"Origin": "file://"},
        {"Origin": "http://localhost:5173, http://evil.example"},
    ],
    ids=[
        "foreign",
        "null",
        "fetch-cross-site",
        "allowlisted-but-cross-site",
        "lookalike",
        "wrong-port",
        "wrong-default-port",
        "wrong-scheme-same-port",
        "file",
        "list",
    ],
)
def test_safety_b4_cross_origin_state_change_rejected(client, method, headers):
    response = _heartbeat(client, method, json={"operator_id": "op"}, headers=headers)
    assert response.status_code == 403
    assert response.json() == REJECTED


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://localhost:5173"},
        {"Origin": "http://127.0.0.1:8000"},
        {"Origin": "http://testserver"},
        {"Origin": "http://testserver:80"},
        {"Sec-Fetch-Site": "same-origin"},
        {"Sec-Fetch-Site": "same-site"},
        {"Sec-Fetch-Site": "none"},
        {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site"},
    ],
    ids=[
        "cli-no-origin",
        "vite-dev-origin",
        "allowlisted-api-origin",
        "same-origin",
        "same-origin-default-port",
        "fetch-same-origin",
        "fetch-same-site",
        "fetch-none",
        "allowlisted-same-site",
    ],
)
def test_c1_trusted_or_originless_requests_pass(client, headers):
    response = _heartbeat(client, json={"operator_id": "op"}, headers=headers)
    assert response.status_code == 200, response.text


def test_c1_vite_proxy_shape_passes(client):
    """Vite's changeOrigin proxy rewrites Host but keeps the browser's Origin."""
    headers = {"Host": "localhost:8000", "Origin": "http://localhost:5173"}
    assert _heartbeat(client, json={"operator_id": "op"}, headers=headers).status_code == 200


def test_c1_reads_are_not_origin_checked_and_get_no_cors_grant(client):
    response = client.get("/api/control/state", headers={"Origin": EVIL})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


# --- stop paths are never blocked -----------------------------------------------


@pytest.mark.parametrize("path", ["/api/robot/estop", "/api/robot/stop"])
@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": EVIL},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Host": "rebind.attacker.example"},
    ],
    ids=["foreign", "null", "cross-site", "untrusted-host"],
)
def test_c1_stop_paths_are_exempt(client, path, headers):
    response = client.post(path, headers=headers)
    assert response.status_code == 200, response.text


def test_c1_halt_path_is_exempt_before_it_exists(client):
    """C11 adds POST /api/robot/halt; the guard must already let it through."""
    response = client.post("/api/robot/halt", headers={"Origin": EVIL})
    assert response.status_code != 403


def test_c1_collapse_is_not_exempt(client):
    response = client.post("/api/robot/collapse", headers={"Origin": EVIL})
    assert response.status_code == 403
    assert response.json() == REJECTED


# --- Host allowlist (DNS rebinding) ----------------------------------------------


def test_safety_b4_untrusted_host_rejected(client):
    """Repro 09: any Host header was accepted, the DNS-rebinding precondition."""
    response = client.get("/api/control/state", headers={"Host": "rebind.attacker.example:8771"})
    assert response.status_code == 400
    assert "SO101_ALLOWED_HOSTS" in response.json()["detail"]
    assert "rebind" not in response.text


@pytest.mark.parametrize(
    "host", ["localhost", "localhost:8000", "127.0.0.1:8000", "[::1]:8000", "LocalHost:8000"]
)
def test_c1_default_hosts_allowed(client, host):
    assert client.get("/api/health", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("host", ["", "::1", "[::1", "localhost:http", "localhost:99999"])
def test_c1_malformed_host_rejected(client, host):
    assert client.get("/api/health", headers={"Host": host}).status_code == 400


def test_c1_allowed_hosts_setting_adds_lan_hosts(monkeypatch):
    _settings_env(
        monkeypatch,
        SO101_ALLOWED_HOSTS="testserver, Robot.LAN, 192.168.1.20:8000, [fd00::5], *.quest.lan",
    )
    with TestClient(create_app()) as c:
        for host in (
            "robot.lan:8000",
            "192.168.1.20:8000",
            "[fd00::5]:8000",
            "headset.quest.lan:5173",
        ):
            assert c.get("/api/health", headers={"Host": host}).status_code == 200, host
        for host in ("quest.lan", "quest.lan.evil.example", "evil.example"):
            assert c.get("/api/health", headers={"Host": host}).status_code == 400, host


def test_c1_lan_host_same_origin_write_passes(monkeypatch):
    """A Quest browser loading the built UI from http://192.168.1.20:8000 is same-origin."""
    _settings_env(monkeypatch, SO101_ALLOWED_HOSTS="192.168.1.20")
    with TestClient(create_app()) as c:
        headers = {"Host": "192.168.1.20:8000", "Origin": "http://192.168.1.20:8000"}
        assert _heartbeat(c, json={"operator_id": "op"}, headers=headers).status_code == 200


def test_c1_bind_host_is_trusted_unless_wildcard(monkeypatch):
    _settings_env(monkeypatch, SO101_HOST="192.168.1.20")
    with TestClient(create_app()) as c:
        assert c.get("/api/health", headers={"Host": "192.168.1.20:8000"}).status_code == 200
    _settings_env(monkeypatch, SO101_HOST="0.0.0.0")
    with TestClient(create_app()) as c:
        assert c.get("/api/health", headers={"Host": "0.0.0.0:8000"}).status_code == 400
        assert c.get("/api/health", headers={"Host": "localhost:8000"}).status_code == 200


@pytest.mark.parametrize("value", ["*", "testserver,*", "robot*.lan"])
def test_c1_allowed_hosts_wildcard_refused_at_startup(monkeypatch, value):
    _settings_env(monkeypatch, SO101_ALLOWED_HOSTS=value)
    with pytest.raises(ValueError, match="SO101_ALLOWED_HOSTS"):
        create_app()


# --- WebSocket Origin check before accept ----------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": EVIL},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Host": "rebind.attacker.example"},
    ],
    ids=["foreign", "null", "cross-site", "untrusted-host"],
)
def test_safety_b4_ws_telemetry_rejected_before_accept(client, headers):
    """Repro 09: /ws/telemetry accepted any Origin and streamed live telemetry."""
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws/telemetry", headers=headers) as ws:
            ws.receive_json()
    assert excinfo.value.code == 1008


@pytest.mark.parametrize(
    "headers",
    [{}, {"Origin": "http://localhost:5173"}, {"Origin": "http://testserver"}],
    ids=["no-origin", "allowlisted", "same-origin"],
)
def test_c1_ws_telemetry_trusted_origins_stream(client, headers):
    with client.websocket_connect("/ws/telemetry", headers=headers) as ws:
        frame = ws.receive_json()
    assert isinstance(frame, dict)
    assert "observation" in frame


# --- SPA containment and JSON 404 for backend paths --------------------------------


def _make_ui(root: Path, marker: str = INDEX) -> Path:
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(marker)
    (root / "assets" / "app.js").write_text("console.log('app');")
    (root / "robots.txt").write_text("User-agent: *")
    return root


@pytest.fixture
def ui_dir(tmp_path):
    # Two levels below tmp_path so "../../secret.txt" escapes to the secret.
    root = _make_ui(tmp_path / "site" / "ui")
    (tmp_path / "secret.txt").write_text(SECRET)
    return root


@pytest.fixture
def spa_client(ui_dir, monkeypatch):
    _settings_env(monkeypatch, SO101_UI_DIR=str(ui_dir))
    with TestClient(create_app()) as test_client:
        yield test_client


def test_spa_serves_index_assets_root_files_and_client_routes(spa_client):
    assert spa_client.get("/").text == INDEX
    assert spa_client.get("/assets/app.js").text == "console.log('app');"
    assert spa_client.get("/robots.txt").text == "User-agent: *"
    route = spa_client.get("/teleop/session")
    assert route.status_code == 200
    assert route.text == INDEX


def _traversal_urls(secret: Path) -> list[str]:
    encoded_abs = "%2F".join(secret.parts[1:])
    return [
        "/..%2F..%2Fsecret.txt",
        "/..%2f..%2fsecret.txt",
        "/%2e%2e/%2e%2e/secret.txt",
        "/..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2Fetc%2Fhosts",
        "http://testserver//etc/hosts",
        "/%2Fetc%2Fhosts",
        "http://testserver/" + str(secret),
        "/%2F" + encoded_abs,
        "/link.txt",
    ]


def test_safety_b4_devex_b12_spa_fallback_never_escapes_the_ui_dir(spa_client, ui_dir, tmp_path):
    """Repro 03: `dist / path` served any file on disk (../, %2e%2e, absolute, //)."""
    secret = tmp_path / "secret.txt"
    (ui_dir / "link.txt").symlink_to(secret)
    for url in _traversal_urls(secret):
        response = spa_client.get(url)
        assert response.status_code == 200, url
        assert response.text == INDEX, url
        assert SECRET not in response.text


@pytest.mark.parametrize(
    "path", ["/api/nope", "/api/", "/api", "/ws/nope", "/ws", "/api/does-not-exist/deeper"]
)
def test_f25_devex_b12_unknown_backend_paths_are_json_404(spa_client, path):
    response = spa_client.get(path)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Not Found"}


def test_f25_double_slash_backend_path_is_json_404(spa_client):
    response = spa_client.get("http://testserver//api/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


# --- UI path check runs before any filesystem call (gate review: spa_unc) ----------
#
# On Windows, root / "\\\\host\\share", root / "//host/share" and root / "D:\\x" all
# discard root, and resolve() opens the result: a UNC path sends the operator's
# NTLM hash to the host, and a device name such as COM3 opens the device. A
# cross-site <img> can send these as GETs, which skip the Origin check.

HOSTILE_UI_PATHS = [
    "\\\\host\\share",
    "\\\\host\\share\\x.png",
    "//host/share",
    "//host/share/x.png",
    "/net/host/share/x.png",  # macOS autofs -hosts map
    "/etc/hosts",
    "C:foo",
    "D:\\x",
    "D:/x",
    "assets\\app.js",
    "COM3",
    "com0.js",
    "LPT9",
    "nul.txt",
    "NUL",
    "con",
    "Aux.tar.gz",
    "assets/PRN.css",
    "index.html::$DATA",
    "index.html:Zone.Identifier",
    "..",
    "../secret.txt",
    "assets/../../secret.txt",
    ".",
    "./index.html",
    ".env",
    ".git/config",
    "assets/.hidden.js",
    "",
    "assets/",
    "assets//app.js",
    "index.html\x00.js",
    "index.html ",
    "caf\u00e9.js",
    "COM\u00b9",
]

# Shapes of the files a Vite build emits, plus a nested asset directory.
REAL_UI_FILES = [
    "index.html",
    "robots.txt",
    "manifest.webmanifest",
    "assets/app.js",
    "assets/index-DYxp6-_V.js",
    "assets/index-wv3IbBtf.css",
    "assets/vision_bundle-BGf_GDpS.js",
    "assets/_commonjsHelpers-Cpj98o6Y.js",
    "assets/fonts/inter-latin-400.woff2",
]


def _forbid_filesystem(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def forbidden(name: str):
        def call(*args, **kwargs):
            calls.append(f"{name}{args!r}")
            raise AssertionError(f"{name} ran before the path check")

        return call

    for owner, name in (
        (Path, "resolve"),
        (Path, "is_file"),
        (Path, "stat"),
        (Path, "exists"),
        (os, "stat"),
        (os, "lstat"),
        (os.path, "realpath"),
    ):
        monkeypatch.setattr(owner, name, forbidden(name))


@pytest.mark.parametrize("path", HOSTILE_UI_PATHS)
def test_c1_hostile_ui_paths_are_rejected_before_any_filesystem_call(tmp_path, monkeypatch, path):
    root = _make_ui(tmp_path / "ui").resolve()
    assert not app_module._is_safe_ui_path(path)
    calls: list[str] = []
    with monkeypatch.context() as patched:
        _forbid_filesystem(patched, calls)
        found = app_module._contained_file(root, path)
    assert found is None
    assert calls == []


def test_c1_real_ui_files_pass_the_path_check_and_stay_under_root_on_windows(tmp_path):
    root = tmp_path / "ui"
    windows_root = PureWindowsPath("C:\\Users\\op\\so101\\frontend\\dist")
    for name in REAL_UI_FILES:
        assert app_module._is_safe_ui_path(name), name
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
        joined = windows_root / name
        assert joined.drive == "C:"
        assert joined.is_relative_to(windows_root), name
    resolved_root = root.resolve()
    for name in REAL_UI_FILES:
        assert app_module._contained_file(resolved_root, name) == resolved_root / name


def test_c1_every_file_in_the_built_ui_passes_the_path_check():
    dist = app_module._checkout_ui_dir()
    if not (dist / "index.html").is_file():
        if os.environ.get("BUILT_UI_PATHS_REQUIRED") == "1":
            pytest.fail("frontend/dist is not built after the CI UI build")
        pytest.skip("frontend/dist is not built")
    names = [p.relative_to(dist).as_posix() for p in dist.rglob("*") if p.is_file()]
    assert [name for name in names if not app_module._is_safe_ui_path(name)] == []


def _watch_paths(monkeypatch: pytest.MonkeyPatch, marker: str) -> list[str]:
    """Record filesystem calls on paths containing ``marker``; stub those out.

    Calls on other paths reach the real functions, so the app still serves its
    own files, and a regression is recorded without touching the network.
    """
    touched: list[str] = []
    real_resolve, real_is_file = Path.resolve, Path.is_file
    real_realpath, real_stat = os.path.realpath, os.stat

    def hit(path) -> bool:
        if isinstance(path, int):
            return False
        if marker in os.fsdecode(path):
            touched.append(os.fsdecode(path))
            return True
        return False

    def resolve(self, strict=False):
        return self if hit(self) else real_resolve(self, strict=strict)

    def is_file(self, **kwargs):
        return False if hit(self) else real_is_file(self, **kwargs)

    def realpath(path, *args, **kwargs):
        return os.fspath(path) if hit(path) else real_realpath(path, *args, **kwargs)

    def stat(path, *args, **kwargs):
        if hit(path):
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(os.path, "realpath", realpath)
    monkeypatch.setattr(os, "stat", stat)
    return touched


@pytest.mark.parametrize(
    ("url", "marker"),
    [
        ("/%5C%5Cattacker.example%5Cshare%5Cx.png", "attacker.example"),
        ("/%5c%5cattacker.example%5cshare", "attacker.example"),
        ("http://testserver///attacker.example/share/x.png", "attacker.example"),
        ("http://testserver//net/attacker.example/share/x.png", "attacker.example"),
        ("/D%3A%5Cx", "D:"),
        ("/C%3Afoo", "C:"),
        ("/COM3", "COM3"),
        ("/nul.txt", "nul.txt"),
        ("/index.html%3A%3A%24DATA", "$DATA"),
        ("/.env", ".env"),
    ],
)
def test_c1_spa_fallback_serves_index_without_touching_hostile_paths(
    spa_client, ui_dir, monkeypatch, url, marker
):
    (ui_dir / ".env").write_text(SECRET)
    touched = _watch_paths(monkeypatch, marker)
    response = spa_client.get(url)
    assert response.status_code == 200, url
    assert response.text == INDEX, url
    assert touched == [], url


@pytest.mark.parametrize(
    ("url", "marker"),
    [
        ("/assets/%5C%5Cattacker.example%5Cshare%5Cx.js", "attacker.example"),
        ("/assets/D%3A%5Cx.js", "D:"),
        ("/assets/COM3", "COM3"),
        ("/assets/nul.js", "nul.js"),
        ("/assets/app.js%3A%3A%24DATA", "$DATA"),
        ("/assets/.hidden.js", ".hidden.js"),
    ],
)
def test_c1_assets_mount_refuses_hostile_paths_without_touching_them(
    spa_client, ui_dir, monkeypatch, url, marker
):
    (ui_dir / "assets" / ".hidden.js").write_text(SECRET)
    touched = _watch_paths(monkeypatch, marker)
    response = spa_client.get(url)
    assert response.status_code == 404, url
    assert SECRET not in response.text
    assert touched == [], url


def test_c1_real_ui_files_are_still_served(ui_dir, monkeypatch):
    for name in REAL_UI_FILES:
        target = ui_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name != "index.html":
            target.write_text(f"content of {name}")
    _settings_env(monkeypatch, SO101_UI_DIR=str(ui_dir))
    with TestClient(create_app()) as c:
        for name in REAL_UI_FILES:
            response = c.get(f"/{name}")
            assert response.status_code == 200, name
            expected = INDEX if name == "index.html" else f"content of {name}"
            assert response.text == expected, name


# --- UI directory resolution (devex-B3) --------------------------------------------


@pytest.fixture
def ui_sources(tmp_path, monkeypatch):
    sources = {
        "env": _make_ui(tmp_path / "env", "<title>ENV</title>"),
        "packaged": _make_ui(tmp_path / "pkg", "<title>PACKAGED</title>"),
        "checkout": _make_ui(tmp_path / "checkout", "<title>CHECKOUT</title>"),
    }
    (tmp_path / "secret.txt").write_text(SECRET)
    monkeypatch.setattr(app_module, "_packaged_ui_dir", lambda: sources["packaged"])
    monkeypatch.setattr(app_module, "_checkout_ui_dir", lambda: sources["checkout"])
    return sources


def test_devex_b3_ui_dir_resolution_order(ui_sources, tmp_path, monkeypatch):
    resolve = app_module.resolve_ui_dir
    _settings_env(monkeypatch, SO101_UI_DIR=str(ui_sources["env"]))
    assert resolve(config.get_settings()) == ui_sources["env"].resolve()

    # A configured directory without index.html is skipped, not served.
    _settings_env(monkeypatch, SO101_UI_DIR=str(tmp_path))
    assert resolve(config.get_settings()) == ui_sources["packaged"].resolve()

    monkeypatch.delenv("SO101_UI_DIR")
    config.get_settings.cache_clear()
    assert resolve(config.get_settings()) == ui_sources["packaged"].resolve()

    monkeypatch.setattr(app_module, "_packaged_ui_dir", lambda: None)
    assert resolve(config.get_settings()) == ui_sources["checkout"].resolve()

    (ui_sources["checkout"] / "index.html").unlink()
    assert resolve(config.get_settings()) is None


@pytest.mark.parametrize("source", ["env", "packaged", "checkout"])
def test_devex_b3_every_ui_source_gets_the_same_containment(ui_sources, monkeypatch, source):
    if source == "env":
        _settings_env(monkeypatch, SO101_UI_DIR=str(ui_sources["env"]))
    elif source == "checkout":
        monkeypatch.setattr(app_module, "_packaged_ui_dir", lambda: None)
    marker = (ui_sources[source] / "index.html").read_text()
    with TestClient(create_app()) as c:
        assert c.get("/").text == marker
        assert c.get("/assets/app.js").status_code == 200
        escaped = c.get("/..%2Fsecret.txt")
        assert escaped.text == marker
        assert SECRET not in escaped.text
        assert c.get("/api/nope").status_code == 404
        assert c.get("/ws/nope").json() == {"detail": "Not Found"}


def test_devex_b3_no_ui_serves_json_placeholder(monkeypatch):
    monkeypatch.setattr(app_module, "_packaged_ui_dir", lambda: None)
    monkeypatch.setattr(app_module, "_checkout_ui_dir", lambda: None)
    with TestClient(create_app()) as c:
        body = c.get("/").json()
        assert body["service"] == "so101"
        assert c.get("/api/nope").json() == {"detail": "Not Found"}
        assert c.get("/ws/nope").status_code == 404


def test_devex_b3_packaged_ui_dir_comes_from_importlib_resources(tmp_path, monkeypatch):
    package = tmp_path / "fake_so101_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    _make_ui(package / "web")
    monkeypatch.syspath_prepend(str(tmp_path))
    found = app_module._packaged_ui_dir("fake_so101_pkg")
    assert found is not None
    assert Path(found).resolve() == (package / "web").resolve()
    assert app_module._packaged_ui_dir("so101_package_that_does_not_exist") is None


def test_devex_b3_checkout_ui_dir_is_the_source_tree_frontend_dist():
    repo_root = Path(__file__).resolve().parents[1]
    assert app_module._checkout_ui_dir() == repo_root / "frontend" / "dist"

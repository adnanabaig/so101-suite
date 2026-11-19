"""scripts/wheel_smoke.py: install the wheel in a clean venv, run the CLI, serve /.

The end-to-end tests install a minimal stand-in so101 wheel with no
dependencies, so pip never needs the network. It carries a stand-in `uvicorn`
(http.server) that serves so101/web the way the real app does, or the JSON
placeholder. The real wheel is smoke-tested by `make ci-wheel` and the CI
wheel job.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_CLI = """\
import sys


def app():
    if "--help" in sys.argv:
        print("Usage: so101 [OPTIONS] COMMAND [ARGS]...")
    elif sys.argv[1:] == ["info"]:
        print("so101 stand-in: info")
    else:
        sys.exit(2)
"""


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "wheel_smoke.py"
    spec = importlib.util.spec_from_file_location("wheel_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ws = _load_script()


# Serves so101/web like so101.services.app (or, with SERVE_UI False, always the
# JSON placeholder, the way an app that cannot find the packaged UI would).
FAKE_UVICORN = """\
import http.server, importlib.resources, json, sys

SERVE_UI = {serve_ui}
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
web = importlib.resources.files("so101") / "web"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        asset = web / self.path.lstrip("/")
        if self.path == "/api/health":
            self.reply(200, "application/json", json.dumps({{"ok": True, "mock": True}}).encode())
        elif self.path == "/" and SERVE_UI and (web / "index.html").is_file():
            self.reply(200, "text/html; charset=utf-8", (web / "index.html").read_bytes())
        elif self.path == "/":
            self.reply(200, "application/json", json.dumps({{"ui": "frontend not built"}}).encode())
        elif self.path.startswith("/assets/") and asset.is_file():
            self.reply(200, "text/javascript", asset.read_bytes())
        else:
            self.reply(404, "application/json", b"{{}}")

    def reply(self, status, content_type, body):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""
INDEX = '<!doctype html><script type="module" src="/assets/index-abc123.js"></script>'
ASSET = "console.log('ui')"


def _stamp(files: dict[str, str]) -> str:
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
    return json.dumps({"format": 1, "sources": "0" * 64, "git_head": None, "files": hashes})


def _fake_wheel(dist: Path, *, ui: bool, serves_ui: bool = True) -> Path:
    """A dependency-free so101 wheel with a console script (and so101/web if ``ui``)."""
    dist_info = "so101-0.0.0.dist-info"
    files = {
        "so101/__init__.py": '__version__ = "0.0.0"\n',
        "so101/cli.py": FAKE_CLI,
        "uvicorn/__init__.py": "",
        "uvicorn/__main__.py": FAKE_UVICORN.format(serve_ui=serves_ui),
        f"{dist_info}/METADATA": "Metadata-Version: 2.1\nName: so101\nVersion: 0.0.0\n",
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": "[console_scripts]\nso101 = so101.cli:app\n",
    }
    if ui:
        web = {"index.html": INDEX, "assets/index-abc123.js": ASSET}
        files.update({f"so101/web/{name}": text for name, text in web.items()})
        files["so101/web/so101-ui-stamp.json"] = _stamp(web)
    record = []
    for name, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=")
        record.append(f"{name},sha256={digest.decode()},{len(text.encode())}")
    record.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = "\n".join(record) + "\n"
    dist.mkdir()
    wheel = dist / "so101-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return wheel


def test_the_venv_environment_hides_the_checkout_and_developer_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "src"))
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    monkeypatch.setenv("SO101_FOLLOWER_PORT", "/dev/ttyACM0")
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("PIP_CERT", "/tmp/ca.pem")
    env = ws.clean_env(tmp_path)
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env
    assert "SO101_FOLLOWER_PORT" not in env
    assert env["SO101_MOCK"] == "1"
    assert env["SO101_DATA_DIR"] == str(tmp_path / "data")
    assert env["PIP_CERT"] == "/tmp/ca.pem"  # pip still reaches its index


@pytest.mark.parametrize(
    ("probe", "ui", "problem"),
    [
        ({"file": "VENV/lib/so101/__init__.py", "web": True}, "required", None),
        ({"file": "VENV/lib/so101/__init__.py", "web": False}, "required", "not in the installed"),
        ({"file": "VENV/lib/so101/__init__.py", "web": True}, "absent", "is in the installed"),
        ({"file": "VENV/lib/so101/__init__.py", "web": False}, "absent", None),
        ({"file": "VENV/lib/so101/__init__.py", "web": False}, "auto", None),
        ({"file": "/checkout/src/so101/__init__.py", "web": True}, "auto", "not from the venv"),
    ],
)
def test_probe_problems(tmp_path, probe, ui, problem):
    venv = tmp_path / "venv"
    probe = {**probe, "file": probe["file"].replace("VENV", str(venv))}
    problems = ws.probe_problems(probe, venv, ui)
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problem in problems[0]


def test_a_failing_command_reports_its_output(tmp_path):
    with pytest.raises(ws.SmokeError, match=r"exited 3\nboom"):
        ws._run(
            [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
            cwd=tmp_path,
            env={},
            timeout=60,
        )


@pytest.mark.parametrize("count", [0, 2])
def test_the_dist_dir_must_hold_exactly_one_wheel(tmp_path, capsys, count):
    for index in range(count):
        (tmp_path / f"so101-0.{index}.0-py3-none-any.whl").write_bytes(b"")
    assert ws.main([str(tmp_path)]) == 1
    assert "expected one so101 wheel" in capsys.readouterr().err


def test_installs_the_wheel_in_a_clean_venv_and_runs_the_cli(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "src"))  # must not leak into the venv
    monkeypatch.setattr(ws.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "home"))
    _fake_wheel(tmp_path / "dist", ui=True)
    assert ws.main([str(tmp_path / "dist"), "--ui", "required"]) == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert "ok    so101 --help" in out
    assert "ok    so101 info" in out
    assert "imports from the venv" in out
    assert "bundled UI (so101/web/index.html): yes --ui required" in out
    assert "the installed app serves the built UI at / (index.html and its assets)" in out
    assert not (tmp_path / "home").exists()  # the temp venv is removed


def test_a_wheel_without_a_ui_serves_the_placeholder(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ws.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "home"))
    _fake_wheel(tmp_path / "dist", ui=False)
    assert ws.main([str(tmp_path / "dist"), "--ui", "absent"]) == 0, capsys.readouterr().err
    assert "serves the JSON placeholder at / (no UI)" in capsys.readouterr().out


def test_an_installed_app_that_does_not_serve_its_ui_fails(tmp_path, capsys, monkeypatch):
    # The wheel carries so101/web, but the installed server answers / with
    # the placeholder: the old smoke passed this.
    monkeypatch.setattr(ws.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "home"))
    _fake_wheel(tmp_path / "dist", ui=True, serves_ui=False)
    assert ws.main([str(tmp_path / "dist"), "--ui", "required"]) == 1
    err = capsys.readouterr().err
    assert "GET / returned 200 'application/json', not the built index.html" in err
    assert not (tmp_path / "home").exists()  # the server was stopped and the venv removed


def _web(tmp_path: Path, index: str = INDEX, stamp: str | None = None) -> Path:
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text(index)
    (web / "assets" / "index-abc123.js").write_text(ASSET)
    (web / "so101-ui-stamp.json").write_text(
        stamp or _stamp({"index.html": index, "assets/index-abc123.js": ASSET})
    )
    return web


def _server(pages: dict[str, tuple[int, str, bytes]]):
    return lambda path: pages.get(path, (404, "application/json", b"{}"))


def test_ui_problems_accepts_the_installed_build(tmp_path):
    web = _web(tmp_path)
    pages = {
        "/": (200, "text/html; charset=utf-8", INDEX.encode()),
        "/assets/index-abc123.js": (200, "text/javascript", ASSET.encode()),
    }
    assert ws.ui_problems(web, _server(pages)) == []


@pytest.mark.parametrize(
    ("pages", "problem"),
    [
        ({"/": (200, "text/html", b"<html>other</html>")}, "GET / is not the installed"),
        (
            {"/": (200, "text/html", INDEX.encode())},
            "GET /assets/index-abc123.js returned 404",
        ),
        (
            {
                "/": (200, "text/html", INDEX.encode()),
                "/assets/index-abc123.js": (200, "text/javascript", b"stale"),
            },
            "GET /assets/index-abc123.js is not the installed so101/web/assets/index-abc123.js",
        ),
        ({"/": (503, "application/json", b"{}")}, "GET / returned 503"),
    ],
    ids=["other-index", "asset-missing", "asset-differs", "error"],
)
def test_ui_problems_names_what_is_served_wrong(tmp_path, pages, problem):
    problems = ws.ui_problems(_web(tmp_path), _server(pages))
    assert any(problem in item for item in problems), problems


def test_ui_problems_checks_index_html_against_the_build_stamp(tmp_path):
    web = _web(tmp_path, stamp=_stamp({"index.html": "<html>built</html>"}))
    pages = {
        "/": (200, "text/html", INDEX.encode()),
        "/assets/index-abc123.js": (200, "text/javascript", ASSET.encode()),
    }
    assert ws.ui_problems(web, _server(pages)) == [
        "GET / is not the index.html that so101-ui-stamp.json lists"
    ]


def test_ui_problems_needs_an_asset_reference(tmp_path):
    index = "<!doctype html><p>no scripts</p>"
    web = _web(tmp_path, index=index)
    problems = ws.ui_problems(web, _server({"/": (200, "text/html", index.encode())}))
    assert problems == ["the served index.html references no /assets/ file"]

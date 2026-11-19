"""The installed-artifact E2E harness fails closed before it can touch hardware."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "installed_e2e.py"
    spec = importlib.util.spec_from_file_location("installed_e2e", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e2e = _load_script()

INDEX = b'<!doctype html><script type="module" src="/assets/app.js"></script>'
ASSET = b"console.log('installed')"


def _wheel(
    dist: Path,
    *,
    ui: bool = True,
    stale_index: bool = False,
    stamp_format: int = 1,
    extra_unlisted: bool = False,
    missing_listed: bool = False,
    extra_stamp_field: bool = False,
    filename: str = "so101-0.2.1-py3-none-any.whl",
) -> Path:
    dist.mkdir(exist_ok=True)
    path = dist / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "so101-0.2.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: so101\nVersion: 0.2.1\n",
        )
        if ui:
            files = {
                "index.html": hashlib.sha256(b"stale" if stale_index else INDEX).hexdigest(),
                "assets/app.js": hashlib.sha256(ASSET).hexdigest(),
            }
            if missing_listed:
                files["assets/missing.js"] = "2" * 64
            archive.writestr("so101/web/index.html", INDEX)
            archive.writestr("so101/web/assets/app.js", ASSET)
            if extra_unlisted:
                archive.writestr("so101/web/assets/unlisted.js", b"unlisted")
            stamp = {
                "format": stamp_format,
                "sources": "0" * 64,
                "git_head": "1" * 40,
                "files": files,
            }
            if extra_stamp_field:
                stamp["untrusted"] = True
            archive.writestr("so101/web/so101-ui-stamp.json", json.dumps(stamp))
    return path


def _fresh(backend: str = "mock") -> tuple[dict, dict]:
    return (
        {
            "backend": backend,
            "mode": "idle",
            "estopped": False,
            "torque_unconfirmed": False,
            "stop_state": "none",
            "torque_state": "off",
            "latch_persistence": "saved",
        },
        {"armed": False, "safety_state": "disarmed", "lease": None},
    )


class ScriptedApi:
    def __init__(self, routes: dict[tuple[str, str], list[object]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, object, int]] = []

    def json(self, method, path, body=None, *, expected_status=200):
        self.calls.append((method, path, body, expected_status))
        queue = self.routes.get((method, path))
        if queue:
            return queue.pop(0)
        return {"ok": True}


class ScriptedTelemetry:
    def __init__(self, frames: list[dict]) -> None:
        self.frames = frames
        self.calls: list[dict] = []

    def wait(self, **kwargs):
        self.calls.append(kwargs)
        return self.frames.pop(0)


def test_selects_one_ui_bearing_wheel_and_pins_its_bytes(tmp_path):
    wheel = _wheel(tmp_path / "dist")
    artifact = e2e.select_ui_wheel(wheel.parent)
    assert artifact.path == wheel.resolve()
    assert artifact.version == "0.2.1"
    assert artifact.index_html == INDEX
    assert artifact.assets == {"/assets/app.js": ASSET}
    assert artifact.ui_sources_sha256 == "0" * 64
    assert artifact.ui_git_head == "1" * 40
    assert artifact.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_wheel_is_snapshotted_before_inspection_hashing_and_evidence_copy(tmp_path):
    source = _wheel(tmp_path / "dist")
    original = source.read_bytes()

    selected, snapshot = e2e.snapshot_ui_wheel(source.parent, tmp_path / "private-snapshot")
    source.write_bytes(b"replaced after selection")
    artifact = e2e.select_ui_wheel(snapshot.parent)
    evidence_artifact = e2e.copy_wheel_to_evidence(artifact, tmp_path / "evidence")

    assert selected == source.resolve()
    assert snapshot.read_bytes() == original
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400
    assert evidence_artifact.path.read_bytes() == original
    assert artifact.sha256 == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("count", [0, 2])
def test_artifact_selection_refuses_zero_or_multiple_wheels(tmp_path, count):
    dist = tmp_path / "dist"
    dist.mkdir()
    for index in range(count):
        (dist / f"so101-0.{index}.0-py3-none-any.whl").write_bytes(b"not needed")
    with pytest.raises(e2e.GateError, match="expected exactly one so101 wheel"):
        e2e.select_ui_wheel(dist)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ui": False}, "not UI-bearing"),
        ({"stale_index": True}, "does not match its UI build stamp"),
        ({"stamp_format": 2}, "not format 1"),
        ({"extra_unlisted": True}, "manifest is incomplete"),
        ({"missing_listed": True}, "manifest is incomplete"),
        ({"extra_stamp_field": True}, "unexpected or missing fields"),
    ],
)
def test_artifact_selection_refuses_missing_or_stale_ui(tmp_path, kwargs, message):
    dist = tmp_path / "dist"
    _wheel(dist, **kwargs)
    with pytest.raises(e2e.GateError, match=message):
        e2e.select_ui_wheel(dist)


def test_environment_hides_the_checkout_and_uses_only_sentinel_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "src"))
    monkeypatch.setenv("VIRTUAL_ENV", "/developer/venv")
    monkeypatch.setenv("SO101_BACKEND", "real")
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_FOLLOWER_PORT", "/dev/ttyACM0")
    monkeypatch.setenv("SO101_CAMERAS", "0,rtsp://secret@example/camera")
    monkeypatch.setenv("PIP_CERT", "/tmp/ca.pem")
    monkeypatch.setenv("PIP_INDEX_URL", "https://build-user:build-pass@example.test/simple")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("SERVICE_URL", "https://user:password@example.test")
    monkeypatch.setenv("DEBUG", "pw:api")
    monkeypatch.setenv("PWDEBUG", "1")
    monkeypatch.setenv("NODE_DEBUG", "http")
    monkeypatch.setenv("NODE_OPTIONS", "--require=/tmp/steal-token.js")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/steal-token.so")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    monkeypatch.setenv("UNRELATED_SAFE", "kept")
    root = tmp_path / "mode"

    mock = e2e.clean_env(root, "mock", e2e_token="ephemeral")
    assert mock["SO101_BACKEND"] == "mock" and mock["SO101_MOCK"] == "1"
    assert mock["SO101_CAMERAS"] == "" and mock["SO101_SIM_CAMERAS"] == ""
    assert "NO-HARDWARE-PORTS" in mock["SO101_FOLLOWER_PORT"]
    assert "NO-HARDWARE-PORTS" in mock["SO101_LEADER_PORT"]
    assert mock["SO101_HOME"] == str(root / "home")
    assert mock["SO101_DATASETS_DIR"] == str(root / "datasets")
    assert "PYTHONPATH" not in mock and "VIRTUAL_ENV" not in mock
    assert mock["SO101_E2E_TOKEN"] == "ephemeral"
    assert mock["UNRELATED_SAFE"] == "kept"
    for secret in (
        "PIP_CERT",
        "PIP_INDEX_URL",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "SERVICE_URL",
        "DEBUG",
        "PWDEBUG",
        "NODE_DEBUG",
        "NODE_OPTIONS",
        "LD_PRELOAD",
        "PYTHONINSPECT",
    ):
        assert secret not in mock

    installer = e2e.install_env(tmp_path / "installer-home")
    assert installer["PIP_CERT"] == "/tmp/ca.pem"
    assert installer["PIP_INDEX_URL"].startswith("https://build-user:")
    assert installer["HOME"] == str(tmp_path / "installer-home")
    assert installer["PIP_CONFIG_FILE"] == os.devnull
    assert installer["PIP_NO_INPUT"] == "1"
    assert "GITHUB_TOKEN" not in installer and "AWS_SECRET_ACCESS_KEY" not in installer
    secrets = e2e.sensitive_environment_values()
    assert "build-pass" in secrets
    assert installer["PIP_INDEX_URL"] in secrets

    sim = e2e.clean_env(root, "sim", e2e_token="ephemeral-sim")
    assert sim["SO101_BACKEND"] == "sim" and sim["SO101_MOCK"] == "0"
    assert sim["SO101_SIM_ENGINE"] == "mujoco"
    assert sim["SO101_SIM_REALTIME"] == "false"
    assert sim["SO101_E2E_TOKEN"] == "ephemeral-sim"


def test_environment_builder_refuses_the_real_backend(tmp_path):
    with pytest.raises(e2e.SafetyRefusal, match="use mock or sim"):
        e2e.clean_env(tmp_path, "real")
    with pytest.raises(e2e.SafetyRefusal, match="nonblank ephemeral token"):
        e2e.clean_env(tmp_path, "mock")


def test_install_environment_ignores_home_and_explicit_pip_configs(tmp_path):
    host_home = tmp_path / "host-home"
    legacy = host_home / ".pip" / "pip.conf"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "[global]\nindex-url = https://home-user:home-pass@packages.test/"
        "tenant/home-config-secret-123/simple\n",
        encoding="utf-8",
    )
    xdg = tmp_path / "host-xdg"
    xdg_config = xdg / "pip" / "pip.conf"
    xdg_config.parent.mkdir(parents=True)
    xdg_config.write_text(
        "[global]\nextra-index-url = https://xdg-user:xdg-pass@packages.test/"
        "tenant/xdg-config-secret-456/simple\n",
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit-config-secret-789.ini"
    explicit.write_text(
        "[global]\nfind-links = https://explicit-user:explicit-pass@packages.test/"
        "tenant/explicit-path-secret/simple\n",
        encoding="utf-8",
    )
    allowed_url = "https://allowed-user:allowed-pass@packages.test/tenant/allowed-env-secret/simple"
    base = {
        "HOME": str(host_home),
        "XDG_CONFIG_HOME": str(xdg),
        "PATH": os.environ.get("PATH", ""),
        "PIP_CONFIG_FILE": str(explicit),
        "PIP_INDEX_URL": allowed_url,
        "PIP_TIMEOUT": "999",
    }
    private_home = tmp_path / "private-installer-home"
    private_home.mkdir(mode=0o700)

    env = e2e.install_env(private_home, base)
    assert env["HOME"] == str(private_home)
    assert env["XDG_CONFIG_HOME"] == str(private_home / ".config")
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["PIP_NO_INPUT"] == "1"
    assert env["PIP_INDEX_URL"] == allowed_url
    assert "PIP_TIMEOUT" not in env

    config_probe = """\
import os
from pathlib import Path
paths = [
    Path.home() / ".pip" / "pip.conf",
    Path(os.environ["XDG_CONFIG_HOME"]) / "pip" / "pip.conf",
]
explicit = os.environ.get("PIP_CONFIG_FILE")
if explicit and explicit != os.devnull:
    paths.append(Path(explicit))
for path in paths:
    if path.is_file():
        print(path.read_text(encoding="utf-8"))
print(os.environ.get("PIP_INDEX_URL", ""))
"""
    result = subprocess.run(
        [sys.executable, "-c", config_probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    for forbidden in (
        "home-config-secret-123",
        "xdg-config-secret-456",
        "explicit-config-secret-789",
        "explicit-path-secret",
    ):
        assert forbidden not in output

    sanitized = e2e._sanitize_text(output, e2e.sensitive_environment_values(base))
    evidence = tmp_path / "install.log"
    e2e._write_text(evidence, sanitized)
    retained = evidence.read_text(encoding="utf-8")
    assert "allowed-user" not in retained and "allowed-pass" not in retained
    assert "allowed-env-secret" not in retained
    for forbidden in (
        "home-config-secret-123",
        "xdg-config-secret-456",
        "explicit-config-secret-789",
        "explicit-path-secret",
    ):
        assert forbidden not in retained


@pytest.mark.parametrize(
    ("status_change", "control_change", "problem"),
    [
        ({"backend": "real"}, {}, "server backend is 'real'"),
        ({"estopped": True}, {}, "E-stop state"),
        ({"torque_unconfirmed": True}, {}, "confirmation is not explicitly clear"),
        ({"latch_persistence": "marker"}, {}, "not durably saved"),
        ({}, {"safety_state": "fault"}, "safety_state is 'fault'"),
        ({}, {"lease": {"source": "other"}}, "lease is active"),
        ({"mode": "rollout"}, {}, "not idle"),
    ],
)
def test_backend_guard_refuses_real_latched_or_busy_servers(status_change, control_change, problem):
    status, control = _fresh()
    status.update(status_change)
    control.update(control_change)
    with pytest.raises(e2e.SafetyRefusal, match=problem):
        e2e.assert_safe_backend(status, control, "mock")


def test_http_client_cannot_mutate_until_the_backend_guard_passes(tmp_path):
    api = e2e.Api(
        "http://127.0.0.1:9",
        tmp_path / "http.jsonl",
        expected_backend="mock",
        expected_version="0.2.1",
        e2e_token="ephemeral",
    )
    with pytest.raises(e2e.SafetyRefusal, match="before the server passed"):
        api.raw("POST", "/api/preflight")
    status, control = _fresh("real")
    with pytest.raises(e2e.SafetyRefusal, match="server backend is 'real'"):
        api.authorize_mutations(status, control, "mock")
    assert api._mutations_allowed is False
    status, control = _fresh("mock")
    api.authorize_mutations(status, control, "mock")
    assert api._mutations_allowed is True
    api.lock_mutations()
    assert api._mutations_allowed is False


def test_http_client_reauthorizes_each_mutation(tmp_path, monkeypatch):
    api = e2e.Api(
        "http://127.0.0.1:9",
        tmp_path / "http.jsonl",
        expected_backend="mock",
        expected_version="0.2.1",
        e2e_token="ephemeral",
    )
    status, control = _fresh()
    api.authorize_mutations(status, control, "mock")
    calls: list[str] = []
    monkeypatch.setattr(api, "_reauthorize", lambda: calls.append("identity"))
    monkeypatch.setattr(
        api,
        "_request",
        lambda method, path, body: (200, "application/json", b'{"ok": true}'),
    )
    assert api.json("POST", "/api/one") == {"ok": True}
    assert api.json("DELETE", "/api/two") == {"ok": True}
    assert calls == ["identity", "identity"]


def test_http_client_disables_environment_proxies(tmp_path, monkeypatch):
    captured = []
    opener = object()

    def build_opener(*handlers):
        captured.extend(handlers)
        return opener

    monkeypatch.setattr(e2e.urllib.request, "build_opener", build_opener)
    api = e2e.Api(
        "http://127.0.0.1:9",
        tmp_path / "http.jsonl",
        expected_backend="sim",
        expected_version="0.2.1",
        e2e_token="ephemeral",
    )
    assert api._opener is opener
    assert len(captured) == 1 and captured[0].proxies == {}


def test_backend_guard_refuses_active_training_or_evaluation():
    status, control = _fresh()
    with pytest.raises(e2e.SafetyRefusal, match="training jobs are active"):
        e2e.assert_safe_backend(
            status,
            control,
            "mock",
            jobs=[{"id": "job-1", "status": "running"}],
        )
    with pytest.raises(e2e.SafetyRefusal, match="an evaluation is active"):
        e2e.assert_safe_backend(
            status,
            control,
            "mock",
            eval_state={"active": True, "phase": "rollout"},
        )


def test_wait_for_observes_state_transitions_instead_of_retrying_an_operation():
    states = iter([{"status": "queued"}, {"status": "running"}, {"status": "succeeded"}])
    now = [0.0]

    def pause(seconds: float) -> None:
        now[0] += seconds

    result = e2e.wait_for(
        "job",
        lambda: next(states),
        lambda state: state["status"] == "succeeded",
        1.0,
        interval=0.1,
        clock=lambda: now[0],
        pause=pause,
    )
    assert result == {"status": "succeeded"}
    assert now[0] == pytest.approx(0.2)


def test_wait_for_timeout_reports_the_last_observed_state():
    now = [0.0]

    def pause(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(e2e.GateError, match="last state: {'phase': 'running'}"):
        e2e.wait_for(
            "evaluation",
            lambda: {"phase": "running"},
            lambda _state: False,
            0.2,
            interval=0.1,
            clock=lambda: now[0],
            pause=pause,
        )


def test_wait_for_stops_on_an_observed_terminal_failure():
    with pytest.raises(e2e.GateError, match="ended 'failed': broken"):
        e2e.wait_for(
            "evaluation",
            lambda: {"phase": "failed", "error": "broken"},
            lambda state: state["phase"] == "done",
            10,
            failed=lambda state: (
                f"ended {state['phase']!r}: {state['error']}"
                if state["phase"] == "failed"
                else None
            ),
        )


def test_expected_eval_result_uses_the_wilson_interval():
    assert e2e._wilson(1, 2) == [0.0945, 0.9055]


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_workflow_ledger_requires_every_safety_and_product_stage(backend):
    ledger = e2e.WorkflowLedger(backend)
    for stage in ledger.required:
        ledger.complete(stage)
    ledger.verify_complete()
    assert {
        "record",
        "inspect_label_qc",
        "deadman",
        "watchdog_halt",
        "lease_expiry",
        "train",
        "rollout",
        "halt",
        "collapse",
        "evaluate",
        "estop",
        "energized_shutdown",
    } <= set(ledger.completed)
    if backend == "sim":
        assert {"sim_diagnostics", "sim_calibration_refusal"} <= set(ledger.completed)


def test_workflow_ledger_refuses_skipped_or_out_of_order_stages():
    ledger = e2e.WorkflowLedger("mock")
    ledger.complete("identity")
    with pytest.raises(e2e.GateError, match="out of order"):
        ledger.complete("record")
    with pytest.raises(e2e.GateError, match="incomplete"):
        ledger.verify_complete()


def test_record_and_manual_control_state_machines_reach_the_safe_halt():
    recording = ScriptedApi(
        {
            ("GET", "/api/robot/status"): [
                {"mode": "idle", "stop_state": "halted", "torque_state": "on"},
            ],
            ("POST", "/api/recording/episode/save"): [
                {"episode": 0, "frames": 30},
                {"episode": 1, "frames": 31},
                {"episode": 2, "frames": 29},
            ],
        }
    )
    telemetry = ScriptedTelemetry(
        [
            {"mode": "recording", "sample_seq": 1, "torque_state": "on"},
            {"mode": "recording", "sample_seq": 13, "episode_frames": 12},
            {"mode": "recording", "sample_seq": 25, "episode_frames": 12},
            {"mode": "recording", "sample_seq": 37, "episode_frames": 12},
        ]
    )
    saved = e2e._record_mock_dataset(recording, telemetry, "installed_e2e/mock_reach")
    assert [entry["episode"] for entry in saved] == [0, 1, 2]
    mutations = [path for method, path, _body, _status in recording.calls if method == "POST"]
    assert "/api/recording/start" in mutations
    assert mutations.count("/api/recording/episode/start") == 3
    assert mutations.count("/api/recording/episode/save") == 3
    assert mutations[-1] == "/api/recording/stop"

    manual = ScriptedApi(
        {
            ("GET", "/api/events?limit=1"): [{"events": [{"id": 7}]}],
            ("POST", "/api/control/lease"): [{"id": "lease-1"}],
            ("POST", "/api/control/arm"): [{"armed": True}],
            ("POST", "/api/control/command"): [
                {"detail": "deadman must be held"},
                {"accepted": True},
            ],
            ("GET", "/api/control/state"): [
                {
                    "lease": {"source": "installed-e2e"},
                    "safety_state": "watchdog_timeout",
                    "armed": False,
                },
                {"lease": None, "safety_state": "lease_expired", "armed": False},
            ],
            ("GET", "/api/robot/status"): [
                {"mode": "idle", "stop_state": "halted", "torque_state": "on"},
                {"mode": "idle", "stop_state": "halted", "torque_state": "on"},
            ],
            ("GET", "/api/events?after_id=7&limit=500"): [
                {
                    "events": [
                        {"type": "control.watchdog_timeout"},
                        {"type": "control.lease_expired"},
                    ]
                }
            ],
        }
    )
    proof = e2e._manual_control_contracts(manual)
    assert proof["watchdog_control"]["safety_state"] == "watchdog_timeout"
    assert proof["expired_control"]["safety_state"] == "lease_expired"
    commands = [call for call in manual.calls if call[:2] == ("POST", "/api/control/command")]
    assert [call[2]["deadman"] for call in commands] == [False, True]
    assert commands[0][3] == 409 and commands[1][3] == 200


def test_rollout_eval_and_stop_state_machines_publish_expected_evidence(tmp_path):
    start = {
        "type": "safety.collapse.start",
        "detail": {"distance_deg": 20.0, "reason": "installed E2E Collapse"},
    }
    done = {
        "type": "safety.collapse.done",
        "detail": {
            "duration_s": 1.0,
            "final_error_deg": 0.2,
            "torque_off": True,
            "fault": None,
        },
    }
    rollout = ScriptedApi(
        {
            ("GET", "/api/events?limit=1"): [
                {"events": [{"id": 10}]},
                {"events": [{"id": 20}]},
            ],
            ("GET", "/api/robot/status"): [
                {
                    "mode": "rollout",
                    "torque_state": "on",
                    "fps": 30.0,
                    "active_model": "/outputs/model",
                },
                {"mode": "idle", "stop_state": "halted", "torque_state": "on", "halt": {}},
                {
                    "mode": "idle",
                    "stop_state": "none",
                    "torque_state": "off",
                    "torque_unconfirmed": False,
                    "estopped": False,
                },
            ],
            ("GET", "/api/events?after_id=10&limit=500"): [{"events": [{"type": "safety.halt"}]}],
            ("GET", "/api/events?after_id=20&limit=500"): [{"events": [start, done]}],
            ("GET", "/api/control/state"): [{"safety_state": "disarmed"}],
        }
    )
    model = {"name": "run@last", "path": "/outputs/model"}
    telemetry = ScriptedTelemetry(
        [
            {"sample_seq": 1},
            {"sample_seq": 4, "mode": "rollout", "torque_state": "on"},
        ]
    )
    stops = e2e._rollout_halt_collapse(rollout, telemetry, model)
    assert stops["collapse_events"] == [start, done]

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    report_path = outputs / "eval.json"
    report_path.write_text(
        json.dumps(
            {
                "source": "mock",
                "phase": "done",
                "error": "",
                "requested_episodes": 2,
                "scored": 2,
                "n": 2,
                "successes": 1,
                "success_rate": 0.5,
                "ci95": [0.0945, 0.9055],
                "label": "state-only BC baseline",
                "privileged_state": False,
            }
        ),
        encoding="utf-8",
    )
    evaluation = ScriptedApi(
        {
            ("GET", "/api/eval/state"): [
                {"phase": "awaiting_score", "episode": 1, "active": True},
                {"phase": "awaiting_reset", "scored": 1},
                {"phase": "awaiting_score", "episode": 2, "active": True},
                {"phase": "done", "active": False, "report_path": str(report_path)},
            ]
        }
    )
    report = e2e._evaluate(evaluation, model, "mock", outputs)
    assert report["source"] == "mock" and report["ci95"] == [0.0945, 0.9055]
    eval_posts = [path for method, path, _body, _status in evaluation.calls if method == "POST"]
    assert eval_posts.count("/api/eval/score") == 2
    assert "/api/eval/continue" in eval_posts

    estop = ScriptedApi(
        {
            ("GET", "/api/events?limit=1"): [{"events": [{"id": 30}]}],
            ("GET", "/api/robot/status"): [
                {"mode": "rollout", "torque_state": "on", "fps": 30.0},
                {
                    "mode": "idle",
                    "estopped": True,
                    "torque_state": "off",
                    "torque_unconfirmed": False,
                    "latch_persistence": "saved",
                },
                {"estopped": False, "torque_state": "off", "latch_persistence": "saved"},
            ],
            ("GET", "/api/events?after_id=30&limit=500"): [{"events": [{"type": "safety.estop"}]}],
        }
    )
    estop_proof = e2e._estop(estop, model)
    assert estop_proof["latched_status"]["estopped"] is True


def test_training_state_machine_proves_label_selection_and_registry(tmp_path):
    outputs = tmp_path / "outputs"
    checkpoint = outputs / "run" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "policy_manifest.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "selection": "kept",
                    "episodes": [0, 2],
                    "skipped": [{"episode": 1, "reason": "outcome is discarded"}],
                }
            }
        ),
        encoding="utf-8",
    )
    api = ScriptedApi(
        {
            ("POST", "/api/training/jobs"): [{"id": "job-1"}],
            ("GET", "/api/training/jobs/job-1"): [{"id": "job-1", "status": "succeeded"}],
            ("GET", "/api/training/jobs/job-1/logs"): [{"log_tail": ["saved"]}],
            ("GET", "/api/models"): [
                [
                    {
                        "name": "run@last",
                        "path": str(checkpoint),
                        "dataset": "installed_e2e/mock_reach",
                        "checkpoint": "last",
                        "kind": "numpy_bc_mlp",
                        "privileged_state": False,
                    }
                ]
            ],
        }
    )
    job, model = e2e._train_and_find_model(
        "mock",
        api,
        "installed_e2e/mock_reach",
        outputs,
    )
    assert job["status"] == "succeeded"
    assert model["e2e_episode_selection"]["episodes"] == [0, 2]
    submitted = next(call for call in api.calls if call[:2] == ("POST", "/api/training/jobs"))
    assert submitted[2]["policy"] == "bc_baseline"
    assert submitted[2]["episodes"] == "kept"


def test_sim_diagnostics_calibration_refusal_and_energized_shutdown(tmp_path):
    sim = ScriptedApi(
        {
            ("GET", "/api/diagnostics"): [
                {
                    "checks": [
                        {
                            "id": "runtime.backend",
                            "status": "pass",
                            "detail": "SO101_BACKEND=sim; MuJoCo loaded",
                        }
                    ],
                    "ports": [{"type": "sim", "path": "sim:follower"}],
                }
            ],
            ("POST", "/api/calibration/start"): [
                {"detail": "calibration has no meaning on the sim backend"}
            ],
        }
    )
    proof = e2e._sim_contracts(sim)
    assert proof["calibration_refusal"]["detail"].endswith("sim backend")
    assert (
        next(call for call in sim.calls if call[:2] == ("POST", "/api/calibration/start"))[3] == 409
    )

    data = tmp_path / "data"
    data.mkdir()
    database = data / "control_plane.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE events (id INTEGER, created_at TEXT, type TEXT, level TEXT, "
        "source TEXT, message TEXT, detail_json TEXT)"
    )
    connection.executemany(
        "INSERT INTO events VALUES (?, '', ?, 'info', 'system', '', ?)",
        [
            (
                2,
                "safety.collapse.start",
                json.dumps({"reason": "service shutdown", "distance_deg": 15.0}),
            ),
            (
                3,
                "safety.collapse.done",
                json.dumps({"torque_off": True, "fault": None}),
            ),
        ],
    )
    connection.commit()
    connection.close()
    shutdown = e2e.verify_energized_shutdown(data, 1)
    assert shutdown["done"]["detail"]["torque_off"] is True


def test_recorded_motion_requires_changing_actions_and_measured_positions(tmp_path):
    episode = tmp_path / "episode_000000"
    episode.mkdir()
    rows = []
    for index in range(12):
        rows.append(
            json.dumps(
                {
                    "t": index / 30,
                    "action": [index * 0.2, 5.0, 0.0, 0.0, 0.0, 40.0],
                    "observation.state": [index * 0.1, 2.0, 0.0, 0.0, 0.0, 20.0],
                }
            )
        )
    (episode / "frames.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    evidence = e2e.analyze_recorded_motion(tmp_path)
    assert evidence["frames"] == 12
    assert evidence["max_arm_action_span"] > 0.1
    assert evidence["max_measured_arm_span"] > 0.05

    still = [
        json.dumps({"t": index / 30, "action": [0.0] * 6, "observation.state": [0.0] * 6})
        for index in range(12)
    ]
    (episode / "frames.jsonl").write_text("\n".join(still) + "\n", encoding="utf-8")
    with pytest.raises(e2e.GateError, match="all zero"):
        e2e.analyze_recorded_motion(tmp_path)


def test_evidence_is_private_complete_and_the_wheel_copy_is_never_overwritten(tmp_path):
    source = _wheel(tmp_path / "dist")
    selected = e2e.select_ui_wheel(source.parent)
    evidence = e2e.prepare_evidence(tmp_path / "evidence")
    copied = e2e.copy_wheel_to_evidence(selected, evidence)
    assert copied.path != source
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o700
    assert stat.S_IMODE(copied.path.stat().st_mode) == 0o600
    source.write_bytes(b"changed after selection")
    assert e2e._sha256(copied.path) == selected.sha256
    with pytest.raises(FileExistsError):
        e2e.copy_wheel_to_evidence(selected, evidence)
    e2e._write_text(evidence / "action-evidence.json", "{}\n")
    hashes = e2e.write_evidence_hashes(evidence)
    assert {"wheel/so101-0.2.1-py3-none-any.whl", "action-evidence.json"} <= set(hashes)
    assert stat.S_IMODE((evidence / e2e.HASH_MANIFEST).stat().st_mode) == 0o600


def test_command_cleanup_kills_descendants_that_outlive_their_parent(tmp_path, monkeypatch):
    if os.name != "posix":
        process = type("Process", (), {"pid": 42})()
        monkeypatch.setattr(e2e, "_descendant_pids", lambda _pid: {84})
        monitor = e2e.DescendantMonitor(process)
        assert monitor.finish() == {84}
        return
    pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(0.7)"
    )
    e2e._run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=os.environ,
        log=tmp_path / "process.log",
        timeout=10,
    )
    child_pid = int(pid_file.read_text(encoding="utf-8"))

    def gone() -> bool:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return True
        return False

    assert e2e.wait_for("descendant cleanup", gone, bool, 2.0)


def test_an_unexpected_server_exit_is_always_a_gate_failure(monkeypatch):
    class ExitedServer:
        pid = 999_999
        returncode = 7

        @staticmethod
        def poll():
            return 7

    monkeypatch.setattr(e2e, "_cleanup_process_group", lambda _pid: None)
    with pytest.raises(e2e.GateError, match="unexpectedly with code 7"):
        e2e.stop_process(ExitedServer())


def test_nonzero_graceful_server_exit_is_a_gate_failure(monkeypatch):
    class FailingShutdown:
        pid = 999_998
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 7
            return 7

    monkeypatch.setattr(e2e, "_signal_process_tree", lambda *args, **kwargs: set())
    monkeypatch.setattr(e2e, "_cleanup_process_group", lambda _pid: None)
    monkeypatch.setattr(e2e, "_cleanup_descendants", lambda _pids: None)
    with pytest.raises(e2e.GateError, match="nonzero during graceful shutdown: 7"):
        e2e.stop_process(FailingShutdown())


def test_command_failure_keeps_complete_output_in_the_evidence(tmp_path):
    log = tmp_path / "command.log"
    with pytest.raises(e2e.GateError, match=r"exited 7\nstdout\nstderr"):
        e2e._run(
            [
                sys.executable,
                "-c",
                "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(7)",
            ],
            cwd=tmp_path,
            env=os.environ,
            log=log,
            timeout=30,
        )
    text = log.read_text(encoding="utf-8")
    assert "exit: 7" in text and "stdout" in text and "stderr" in text


def test_ui_hook_exports_only_the_owned_values_and_never_logs_the_token(tmp_path, monkeypatch):
    token = "ephemeral-never-log"
    monkeypatch.setenv("DEBUG", "pw:api")
    monkeypatch.setenv("PWDEBUG", "1")
    monkeypatch.setenv("NODE_DEBUG", "http")
    monkeypatch.setenv("PLAYWRIGHT_OUTPUT_DIR", "/tmp/not-owned")
    monkeypatch.delenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", raising=False)
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    captured = {}

    def run(command, *, cwd, env, log, timeout, redact_values=()):
        captured.update(command=tuple(command), cwd=cwd, env=dict(env), timeout=timeout)
        e2e._write_text(
            log,
            f"{env['SO101_E2E_BASE_URL']}\n{env['PLAYWRIGHT_ARTIFACTS_DIR']}\n"
            f"{bool(env['SO101_E2E_TOKEN'])}\n"
            f"{
                any(k in env for k in ['DEBUG', 'PWDEBUG', 'NODE_DEBUG', 'PLAYWRIGHT_OUTPUT_DIR'])
            }\n"
            'SO101_E2E_BROWSER_METADATA={"schema_version":1,"kind":"so101.e2e_browser",'
            '"source":"playwright_pinned","executable_path":"/pinned/chrome",'
            '"version":"Chromium test"}\n',
        )
        return ""

    monkeypatch.setattr(e2e, "_run", run)
    monkeypatch.setattr(
        e2e,
        "system_browser_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected override")),
    )
    e2e.run_ui_hook(
        base="http://127.0.0.1:8765",
        e2e_token=token,
        expected_version="0.2.2rc5",
        cwd=frontend,
        evidence=tmp_path / "evidence",
    )
    log = (tmp_path / "evidence" / "ui-e2e.log").read_text(encoding="utf-8")
    assert "http://127.0.0.1:8765" in log
    assert str((tmp_path / "evidence" / "playwright").resolve()) in log
    assert token not in log
    assert "True\nFalse" in log
    assert captured["command"] == e2e.UI_E2E_COMMAND
    assert captured["cwd"] == frontend
    assert "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH" not in captured["env"]
    assert captured["env"]["SO101_E2E_EXPECTED_VERSION"] == "0.2.2rc5"
    browser = json.loads((tmp_path / "evidence" / "browser.json").read_text(encoding="utf-8"))
    assert browser["source"] == "playwright_pinned"


def test_accidental_token_in_ui_log_is_redacted_before_failure(tmp_path):
    log = tmp_path / "ui-e2e.log"
    token = "ephemeral-secret"
    e2e._write_text(log, f"debug header {token}\n")
    assert e2e.redact_secret_file(log, token) is True
    assert token not in log.read_text(encoding="utf-8")
    nested = tmp_path / "playwright" / "error.txt"
    e2e._write_text(nested, f"context {token}\n")
    assert e2e.redact_secret_tree(tmp_path, token) == ["playwright/error.txt"]
    e2e.assert_secret_absent(tmp_path, token)


def test_ui_hook_records_an_explicit_system_browser_override(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    executable = str(Path(sys.executable).resolve())
    monkeypatch.setenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", executable)
    monkeypatch.setattr(
        e2e,
        "system_browser_provenance",
        lambda env, *, cwd, evidence: {"executable": executable, "version": "Python browser"},
    )

    def run(command, *, cwd, env, log, timeout, redact_values=()):
        assert tuple(command) == e2e.UI_E2E_COMMAND
        assert env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] == executable
        e2e._write_text(
            log,
            "SO101_E2E_BROWSER_METADATA="
            + json.dumps(
                {
                    "schema_version": 1,
                    "kind": "so101.e2e_browser",
                    "source": "system_override",
                    "executable_path": executable,
                    "version": "Python browser",
                },
                separators=(",", ":"),
            )
            + "\n",
        )
        return ""

    monkeypatch.setattr(e2e, "_run", run)
    evidence = tmp_path / "evidence"
    e2e.run_ui_hook(
        base="http://127.0.0.1:8765",
        e2e_token="x" * 43,
        expected_version="0.2.2rc5",
        cwd=frontend,
        evidence=evidence,
    )
    browser = json.loads((evidence / "browser.json").read_text(encoding="utf-8"))
    assert browser["source"] == "system_override"
    assert browser["executable_path"] == executable


def test_pip_credentials_are_sanitized_from_chained_errors_and_evidence(tmp_path):
    url = "https://build-user:build-pass@example.test/simple"
    values = [url, "build-user", "build-pass"]
    sanitized = e2e._sanitize_text(f"download {url} as build-user/build-pass", values)
    assert "build-user" not in sanitized and "build-pass" not in sanitized
    log = tmp_path / "pip.log"
    e2e._write_text(log, f"index={url}\n")
    redacted = set()
    for value in values:
        redacted.update(e2e.redact_secret_tree(tmp_path, value))
    assert redacted == {"pip.log"}
    assert "build-pass" not in log.read_text(encoding="utf-8")


def test_package_url_components_and_transformed_credentials_are_all_sensitive(tmp_path):
    package_url = (
        "https://build-user:pa%24%24word@example.test/tenant/path-secret-123/"
        "simple?access_token=query%2Bsecret&signature=signed-value#fragment-secret"
    )
    secrets = e2e.sensitive_environment_values({"PIP_EXTRA_INDEX_URL": package_url})
    for value in (
        package_url,
        "build-user",
        "pa%24%24word",
        "pa$$word",
        "path-secret-123",
        "query%2Bsecret",
        "query+secret",
        "signed-value",
        "fragment-secret",
    ):
        assert value in secrets

    exposed = (
        "auth build-user:pa$$word; download /tenant/path-secret-123/simple; "
        "access_token=query+secret signature=signed-value fragment-secret"
    )
    sanitized = e2e._sanitize_text(exposed, secrets)
    for leaked in (
        "build-user",
        "pa$$word",
        "path-secret-123",
        "query+secret",
        "signed-value",
        "fragment-secret",
    ):
        assert leaked not in sanitized

    log = tmp_path / "download.log"
    e2e._write_text(log, exposed)
    redacted: set[str] = set()
    for secret in secrets:
        redacted.update(e2e.redact_secret_tree(tmp_path, secret))
    assert redacted == {"download.log"}
    retained = log.read_text(encoding="utf-8")
    for leaked in (
        "build-user",
        "pa$$word",
        "path-secret-123",
        "query+secret",
        "signed-value",
        "fragment-secret",
    ):
        assert leaked not in retained


def test_package_url_redaction_covers_encoded_decoded_and_case_variants():
    raw_url = "https://operator:RawSecret!@packages.test/simple"
    encoded_url = "https://operator:Encoded%2BSecret@packages.test/simple"
    secrets = e2e.sensitive_environment_values(
        {"PIP_INDEX_URL": raw_url, "PIP_EXTRA_INDEX_URL": encoded_url}
    )
    exposed = "pip echoed rawsecret%21 then ENCODED+SECRET"
    sanitized = e2e._sanitize_text(exposed, secrets)
    assert "rawsecret%21" not in sanitized.lower()
    assert "encoded+secret" not in sanitized.lower()


def test_gate_failure_publishes_evidence_and_removes_temporary_work(tmp_path):
    evidence = tmp_path / "evidence"
    assert e2e.run_gate(tmp_path / "empty-dist", evidence, ("mock",)) == 1
    result = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["release_ready"] is False
    assert "expected exactly one so101 wheel" in result["error"]
    assert result["work_cleanup"]["removed"] is True
    assert not Path(result["work_cleanup"]["path"]).exists()
    assert "GateError" in (evidence / "failure.log").read_text(encoding="utf-8")
    manifest = (evidence / e2e.HASH_MANIFEST).read_text(encoding="utf-8")
    assert "result.json" in manifest and "failure.log" in manifest


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_run_mode_orchestrates_the_complete_workflow_and_shutdown(tmp_path, monkeypatch, backend):
    artifact = e2e.select_ui_wheel(_wheel(tmp_path / "dist").parent)
    work = tmp_path / "work"
    work.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    events: list[str] = []
    api = ScriptedApi(
        {
            ("POST", "/api/preflight?arm=both&save=true"): [
                {
                    "backend": backend,
                    "rehearsal": True,
                    "ready_for_hardware": False,
                    "overall": "warn",
                }
            ]
        }
    )
    telemetry = ScriptedTelemetry([])

    class Server:
        pid = 999_997
        returncode = None

        def poll(self):
            return self.returncode

    class Monitor:
        def __init__(self, process):
            self.process = process

        def finish(self):
            return set()

    monkeypatch.setattr(
        e2e,
        "probe_install",
        lambda *args, **kwargs: events.append("probe") or {"backend": backend},
    )
    monkeypatch.setattr(
        e2e,
        "_run",
        lambda *args, **kwargs: events.append("info") or "",
    )
    monkeypatch.setattr(e2e, "free_port", lambda: 8765)
    monkeypatch.setattr(e2e, "Api", lambda *args, **kwargs: api)
    monkeypatch.setattr(e2e, "TelemetryProbe", lambda *args, **kwargs: telemetry)
    monkeypatch.setattr(e2e.subprocess, "Popen", lambda *args, **kwargs: Server())
    monkeypatch.setattr(e2e, "DescendantMonitor", Monitor)
    monkeypatch.setattr(
        e2e,
        "wait_for_server",
        lambda *args, **kwargs: {"ok": True, "backend": backend},
    )
    monkeypatch.setattr(
        e2e,
        "authorize_fresh_rehearsal",
        lambda *args, **kwargs: _fresh(backend),
    )
    monkeypatch.setattr(e2e, "verify_served_ui", lambda *args: {"/": artifact.sha256})
    monkeypatch.setattr(
        e2e,
        "_sim_contracts",
        lambda _api: events.append("sim-contracts") or {"ok": True},
    )
    dataset_path = work / backend / "datasets" / f"installed_e2e/{backend}_reach"
    monkeypatch.setattr(
        e2e,
        "_dataset_workflow",
        lambda *args, **kwargs: (
            events.append("dataset")
            or (
                f"installed_e2e/{backend}_reach",
                {"path": str(dataset_path), "num_episodes": 3},
                {"overall": "pass"},
                {"frames": 30},
            )
        ),
    )
    monkeypatch.setattr(
        e2e,
        "_manual_control_contracts",
        lambda _api: events.append("manual") or {"ok": True},
    )
    model = {"name": "run@last", "path": str(work / backend / "outputs/model")}
    monkeypatch.setattr(
        e2e,
        "_train_and_find_model",
        lambda *args: events.append("train") or ({"status": "succeeded"}, model),
    )
    monkeypatch.setattr(
        e2e,
        "_rollout_halt_collapse",
        lambda *args: events.append("rollout-stops") or {"ok": True},
    )
    monkeypatch.setattr(
        e2e,
        "_evaluate",
        lambda *args: (
            events.append("evaluate")
            or {
                "source": backend,
                "phase": "done",
                "successes": 1,
                "n": 2,
                "success_rate": 0.5,
                "ci95": [0.0945, 0.9055],
            }
        ),
    )
    monkeypatch.setattr(e2e, "_estop", lambda *args: events.append("estop") or {"ok": True})
    monkeypatch.setattr(
        e2e,
        "_copy_artifacts",
        lambda *args: events.append("copy-artifacts") or {"copied": True},
    )
    monkeypatch.setattr(
        e2e,
        "_energize_for_shutdown",
        lambda *args: events.append("energize") or (42, {"mode": "rollout", "torque_state": "on"}),
    )
    monkeypatch.setattr(
        e2e,
        "stop_process",
        lambda *args, **kwargs: events.append("stop-server") or "terminated (0)",
    )
    monkeypatch.setattr(
        e2e,
        "verify_energized_shutdown",
        lambda *args: events.append("verify-shutdown") or {"done": True},
    )

    def copy_file(source, destination, **kwargs):
        events.append("copy-control-db")
        e2e._write_text(destination, "db")
        return destination

    monkeypatch.setattr(e2e, "_copy_evidence_file", copy_file)
    summary = e2e.run_mode(
        backend,
        python=Path(sys.executable),
        so101=tmp_path / "venv/bin/so101",
        venv=tmp_path / "venv",
        artifact=artifact,
        work=work,
        evidence_root=evidence,
        ledger_secrets=[],
        ui_e2e=False,
        ui_cwd=frontend,
    )
    assert summary["workflow_stages"] == list(e2e.WorkflowLedger(backend).required)
    expected = ["probe", "info"]
    if backend == "sim":
        expected.append("sim-contracts")
    expected += [
        "dataset",
        "manual",
        "train",
        "rollout-stops",
        "evaluate",
        "estop",
        "copy-artifacts",
        "energize",
        "stop-server",
        "verify-shutdown",
        "copy-control-db",
    ]
    assert events == expected


def test_successful_run_gate_installs_the_private_wheel_then_hashes_full_evidence(
    tmp_path, monkeypatch
):
    source = _wheel(tmp_path / "dist")
    evidence = tmp_path / "evidence"
    events: list[tuple[str, object]] = []

    def run(command, *, cwd, env, log, timeout, redact_values=()):
        e2e._write_text(log, "ok\n")
        if command[1:4] == ["-m", "pip", "install"]:
            wheel_arg = next(item for item in command if ".whl" in str(item))
            wheel_path = Path(str(wheel_arg).removesuffix("[sim]"))
            assert wheel_path.is_relative_to(evidence / "wheel")
            assert wheel_path.is_file()
            events.append(("install", wheel_path))
        elif command[-2:] == ["freeze", "--all"]:
            return "so101==0.2.1\n"
        return ""

    monkeypatch.setattr(e2e, "_run", run)

    def mode(backend, **kwargs):
        events.append(("mode", backend))
        assert (evidence / "wheel" / source.name).is_file()
        e2e._write_json(evidence / backend / "summary.json", {"backend": backend})
        return {"backend": backend, "workflow_stages": list(e2e.WorkflowLedger(backend).required)}

    monkeypatch.setattr(e2e, "run_mode", mode)
    assert e2e.run_gate(source.parent, evidence, ("mock",), ui_e2e=False) == 0
    result = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed" and result["release_ready"] is False
    assert [event[0] for event in events] == ["install", "mode"]
    copied = evidence / "wheel" / source.name
    assert copied.is_file() and e2e._sha256(copied) == e2e._sha256(source)
    hashes = (evidence / e2e.HASH_MANIFEST).read_text(encoding="utf-8")
    for retained in ("wheel/" + source.name, "mock/summary.json", "result.json"):
        assert retained in hashes


def test_evidence_directory_must_not_contain_a_stale_run(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(e2e.GateError, match="not empty"):
        e2e.prepare_evidence(evidence)


def test_make_target_is_opt_in_and_not_silently_added_to_ci():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "installed-e2e:" in makefile
    script = (REPO_ROOT / "scripts" / "installed_e2e.py").read_text()
    assert "SO101_E2E_BASE_URL" in script
    assert "--ui-command" not in script and "INSTALLED_E2E_UI_COMMAND" not in makefile
    assert e2e.UI_E2E_COMMAND == ("npm", "run", "test:e2e")
    ci_recipe = makefile.split("ci:", 1)[1].split(".PHONY: ci-backend", 1)[0]
    assert "installed-e2e" not in ci_recipe

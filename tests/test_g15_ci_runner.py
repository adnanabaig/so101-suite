"""Deterministic contracts for the pre-bootstrap G15 CI evidence runner.

These tests use only the standard library.  The command and probe fakes never
run make, Git, Node, hardware code, or a network operation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import signal
import stat
import sys
import tempfile
import unittest
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "c" * 40


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_g15_ci.py"
    spec = importlib.util.spec_from_file_location("run_g15_ci", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


g15 = _load_script()


def _host(**changes: object):
    values = {
        "system": "Linux",
        "architecture": "x86_64",
        "kernel_release": "6.8.0-test",
        "python_version": "3.12.9",
        "python_implementation": "CPython",
        "python_major": 3,
        "python_minor": 12,
        "python_isolated": True,
        "os_release": {
            "ID": "ubuntu",
            "VERSION_ID": "24.04",
            "VERSION_CODENAME": "noble",
        },
        "effective_uid": 1000,
        "system_vendor": "ASUSTeK COMPUTER INC.",
        "product_name": "ROG Zephyrus G15 GA503QR_GA503QR",
    }
    values.update(changes)
    return g15.HostFacts(**values)


def _tools() -> dict[str, object]:
    return {
        name: g15.ToolInfo(
            name=name,
            invocation_path=f"/usr/bin/{name}",
            resolved_path=f"/usr/bin/{name}",
            sha256=(str(index) * 64),
            mode="0755",
            owner_uid=0,
        )
        for index, name in enumerate(g15.REQUIRED_TOOLS, start=1)
    }


class FakeProbe:
    def __init__(
        self,
        *,
        commit: str = EXPECTED,
        status: bytes = b"",
        final_commit: str | None = None,
        final_status: bytes | None = None,
        node: bytes = b"v24.11.1\n",
    ) -> None:
        self.commit = commit
        self.status = status
        self.final_commit = final_commit
        self.final_status = final_status
        self.node = node
        self.calls: list[tuple[str, ...]] = []
        self.commit_calls = 0
        self.status_calls = 0

    def __call__(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> object:
        del cwd, env
        command = tuple(argv)
        self.calls.append(command)
        executable = Path(command[0]).name
        if executable == "node" and command[1:] == ("--version",):
            return g15.ProbeResult(0, self.node)
        if executable == "npm" and command[1:] == ("--version",):
            return g15.ProbeResult(0, b"11.6.1\n")
        if executable == "git" and command[1:] == ("--version",):
            return g15.ProbeResult(0, b"git version 2.45.2\n")
        if executable == "make" and command[1:] == ("--version",):
            return g15.ProbeResult(0, b"GNU Make 4.3\n")
        if executable == "git" and len(command) >= 5 and command[1] == "-C":
            if command[3:5] == ("rev-parse", "--verify"):
                self.commit_calls += 1
                value = (
                    self.final_commit
                    if self.commit_calls > 1 and self.final_commit is not None
                    else self.commit
                )
                return g15.ProbeResult(0, (value + "\n").encode())
            if command[3] == "status":
                self.status_calls += 1
                value = (
                    self.final_status
                    if self.status_calls > 1 and self.final_status is not None
                    else self.status
                )
                return g15.ProbeResult(0, value)
        raise AssertionError(f"unexpected probe: {command}")


class FakeExecutor:
    def __init__(
        self,
        outcomes: Sequence[int | BaseException | str] = (0, 0),
        outputs: Sequence[bytes] = (b"bootstrap output\n", b"ci output\n"),
    ) -> None:
        self.outcomes = list(outcomes)
        self.outputs = list(outputs)
        self.calls: list[tuple[str, tuple[str, ...], dict[str, str]]] = []
        self.npm_userconfigs: list[str] = []

    def __call__(
        self,
        command_id: str,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        relay: object,
        redact_patterns: Sequence[bytes],
        scratch_dir: Path,
        now: object,
        monotonic_ns: object,
    ) -> object:
        del cwd, now, monotonic_ns, redact_patterns, scratch_dir
        index = len(self.calls)
        self.npm_userconfigs.append(Path(env["NPM_CONFIG_USERCONFIG"]).read_text())
        self.calls.append((command_id, tuple(argv), dict(env)))
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        log_path.write_bytes(self.outputs[index])
        log_path.chmod(0o600)
        interrupted = None
        exit_code = int(outcome) if outcome != "sigint" else -signal.SIGINT
        if outcome == "sigint":
            relay.record(signal.SIGINT, forward=False)
            interrupted = "SIGINT"
        return g15.CommandResult(
            command_id=command_id,
            argv=tuple(argv),
            log_path=log_path.name,
            started_at="2026-10-07T12:00:00.000000Z",
            finished_at="2026-10-07T12:00:00.000001Z",
            duration_ns=123,
            exit_code=exit_code,
            interrupted_by=interrupted,
        )


class RunnerFixture:
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="g15-runner-test-")
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "frontend").mkdir()
        self.evidence = self.root / "evidence"
        self.now = lambda: dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.UTC)
        self.monotonic_ns = lambda: 99_000_000_000

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_gate(
        self,
        *,
        probe: FakeProbe | None = None,
        executor: FakeExecutor | None = None,
        host: object | None = None,
        environ: Mapping[str, str] | None = None,
        evidence: Path | None = None,
    ) -> tuple[int, FakeProbe, FakeExecutor]:
        selected_probe = probe or FakeProbe()
        selected_executor = executor or FakeExecutor()
        code = g15.run_gate(
            EXPECTED,
            evidence or self.evidence,
            repo_root=self.repo,
            environ={"PATH": "/usr/bin"} if environ is None else environ,
            host_facts=host or _host(),
            probe=selected_probe,
            executor=selected_executor,
            tool_resolver=_tools,
            now=self.now,
            monotonic_ns=self.monotonic_ns,
            relay=g15.SignalRelay(install_handlers=False),
        )
        return code, selected_probe, selected_executor

    def result(self, evidence: Path | None = None) -> dict[str, object]:
        return json.loads(((evidence or self.evidence) / "result.json").read_text())


class RunnerCase(RunnerFixture, unittest.TestCase):
    def test_success_runs_each_fixed_command_once_and_seals_private_evidence(self) -> None:
        code, probe, executor = self.run_gate()
        self.assertEqual(code, 0)
        self.assertEqual(
            [(name, argv) for name, argv, _env in executor.calls],
            [
                ("bootstrap", ("/usr/bin/make", "bootstrap")),
                ("ci", ("/usr/bin/make", "ci")),
            ],
        )
        self.assertEqual(probe.commit_calls, 2)
        self.assertEqual(probe.status_calls, 2)

        result = self.result()
        self.assertEqual(result["schema"], g15.SCHEMA)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertFalse(result["release_ready"])
        self.assertFalse(result["hardware_touched"])
        self.assertTrue(result["hardware_validation_required"])
        self.assertEqual(result["source"]["expected_commit"], EXPECTED)
        self.assertEqual(result["source"]["actual_commit"], EXPECTED)
        self.assertTrue(result["source"]["unchanged"])
        self.assertEqual([item["duration_ns"] for item in result["commands"]], [123, 123])
        self.assertEqual([item["exit_code"] for item in result["commands"]], [0, 0])
        self.assertTrue(all(item["invocation_count"] == 1 for item in result["commands"]))
        environment = json.loads((self.evidence / "environment.json").read_text())
        self.assertEqual(environment["host"]["system_vendor"], "ASUSTeK COMPUTER INC.")
        self.assertIn("Zephyrus G15", environment["host"]["product_name"])
        self.assertEqual(environment["tools"]["make"]["invocation_path"], "/usr/bin/make")
        self.assertEqual(len(environment["tools"]["make"]["sha256"]), 64)
        self.assertEqual(stat.S_IMODE(self.evidence.stat().st_mode), 0o700)
        for path in self.evidence.iterdir():
            self.assertTrue(path.is_file())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

        manifest = {}
        for line in (self.evidence / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            manifest[name] = digest
        expected_files = {
            "bootstrap.log",
            "ci.log",
            "environment.json",
            "result.json",
        }
        self.assertEqual(set(manifest), expected_files)
        for name, digest in manifest.items():
            self.assertEqual(
                hashlib.sha256((self.evidence / name).read_bytes()).hexdigest(), digest
            )
        self.assertNotIn("SHA256SUMS", manifest)

    def test_bootstrap_failure_preserves_exact_exit_and_never_runs_ci(self) -> None:
        executor = FakeExecutor((17,), (b"first failure\n",))
        code, probe, executor = self.run_gate(executor=executor)
        self.assertEqual(code, 17)
        self.assertEqual([call[0] for call in executor.calls], ["bootstrap"])
        self.assertEqual(probe.commit_calls, 1, "no postflight probe after bootstrap failure")
        self.assertFalse((self.evidence / "ci.log").exists())
        result = self.result()
        self.assertEqual(result["failure"]["stage"], "bootstrap")
        self.assertEqual(result["failure"]["exit_code"], 17)
        self.assertEqual(result["commands"][0]["exit_code"], 17)
        self.assertEqual((self.evidence / "bootstrap.log").read_bytes(), b"first failure\n")

    def test_ci_failure_is_second_and_last_command(self) -> None:
        executor = FakeExecutor((0, 23))
        code, probe, executor = self.run_gate(executor=executor)
        self.assertEqual(code, 23)
        self.assertEqual([call[0] for call in executor.calls], ["bootstrap", "ci"])
        self.assertEqual(probe.commit_calls, 1, "no postflight probe after CI failure")
        result = self.result()
        self.assertEqual(result["failure"]["stage"], "ci")
        self.assertEqual(result["failure"]["exit_code"], 23)

    def test_executor_exception_is_evidence_and_stops_at_first_failure(self) -> None:
        executor = FakeExecutor((OSError("injected spawn error"),))
        code, _probe, executor = self.run_gate(executor=executor)
        self.assertEqual(code, 1)
        self.assertEqual([call[0] for call in executor.calls], ["bootstrap"])
        result = self.result()
        self.assertEqual(result["failure"]["kind"], "spawn_error")
        self.assertIn("injected spawn error", (self.evidence / "bootstrap.log").read_text())

    def test_sigint_is_recorded_and_ci_never_starts(self) -> None:
        executor = FakeExecutor(("sigint",))
        code, _probe, executor = self.run_gate(executor=executor)
        self.assertEqual(code, 130)
        self.assertEqual([call[0] for call in executor.calls], ["bootstrap"])
        result = self.result()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["failure"]["kind"], "signal")
        self.assertEqual(result["commands"][0]["interrupted_by"], "SIGINT")

    def test_known_credential_values_are_redacted_even_across_a_stream_chunk(self) -> None:
        secret = "query-topsecret"
        url = f"https://packages.example.test/simple?download_sig={secret}"
        prefix = b"x" * (64 * 1024 - 2)
        executor = FakeExecutor((0, 0), (prefix + secret.encode() + b"\n", url.encode() + b"\n"))
        code, _probe, _executor = self.run_gate(
            executor=executor,
            environ={"PATH": "/usr/bin", "PIP_INDEX_URL": url},
        )
        self.assertEqual(code, 0)
        for path in self.evidence.iterdir():
            data = path.read_bytes()
            self.assertNotIn(secret.encode(), data)
            self.assertNotIn(url.encode(), data)
        result = self.result()
        self.assertGreater(result["commands"][0]["stdout_stderr"]["redactions_applied"], 0)
        self.assertGreater(result["commands"][1]["stdout_stderr"]["redactions_applied"], 0)

    def test_every_package_url_and_transformed_variant_is_redacted(self) -> None:
        first = "https://first-user:first%2Bpass@one.example/simple?token=Alpha%2520Beta"
        second = "https://second-user:SecondPass@two.example/private-path/simple"
        transformed = (
            b"FIRST+PASS\n"
            b"Alpha Beta\n"
            b"SECOND-USER\n"
            b"https%3A%2F%2Fsecond-user%3ASecondPass%40two.example%2Fprivate-path%2Fsimple\n"
        )
        executor = FakeExecutor((0, 0), (transformed, f"{first}\n{second}\n".encode()))

        code, _probe, _executor = self.run_gate(
            executor=executor,
            environ={"PATH": "/usr/bin", "PIP_EXTRA_INDEX_URL": f"{first}  {second}"},
        )

        self.assertEqual(code, 0)
        retained = b"".join(path.read_bytes() for path in self.evidence.iterdir())
        for secret in (
            b"FIRST+PASS",
            b"Alpha Beta",
            b"SECOND-USER",
            first.encode(),
            second.encode(),
            urllib.parse.quote(second, safe="").encode(),
        ):
            self.assertNotIn(secret, retained)

    def test_masked_package_url_still_redacts_encoded_path_secrets(self) -> None:
        configured = (
            "https://registry-user:registry-pass@packages.example.test/"
            "tenant/Path%252BSecret/simple"
        )
        echoed = (
            b"https://****:****@packages.example.test/tenant/PATH+SECRET/simple\n"
            b"download tenant path%2bsecret\n"
        )
        executor = FakeExecutor((0, 0), (echoed, echoed))

        code, _probe, _executor = self.run_gate(
            executor=executor,
            environ={"PATH": "/usr/bin", "PIP_INDEX_URL": configured},
        )

        self.assertEqual(code, 0)
        retained = b"".join(path.read_bytes() for path in self.evidence.iterdir())
        self.assertNotIn(b"PATH+SECRET", retained)
        self.assertNotIn(b"path%2bsecret", retained.lower())

    def test_child_commands_are_mock_isolated_and_environment_overrides_are_scrubbed(self) -> None:
        executor = FakeExecutor()
        source = {
            "PATH": "/tmp/poisoned-tool-directory",
            "MAKEFLAGS": "-n",
            "MAKE": "true",
            "PYTEST_ADDOPTS": "--ignore=tests",
            "COVERAGE_RCFILE": "/tmp/omit-everything",
            "NODE_OPTIONS": "--require=/tmp/evil.js",
            "NPM_CONFIG_SCRIPT_SHELL": "/tmp/fake-success-shell",
            "FRONTEND_INSTALL": "0",
            "HOME": "/home/attacker",
            "PYTHONUSERBASE": "/tmp/attacker-python",
            "TMPDIR": str(self.evidence),
            "VITE_SECRET": "must-not-reach-the-build",
            "PIP_INDEX_URL": "https://packages.example.test/simple",
            "UNRELATED": "kept",
        }
        code, probe, executor = self.run_gate(executor=executor, environ=source)
        self.assertEqual(code, 0)
        for _name, argv, env in executor.calls:
            self.assertIn(
                argv,
                (("/usr/bin/make", "bootstrap"), ("/usr/bin/make", "ci")),
            )
            self.assertEqual(env["SO101_BACKEND"], "mock")
            self.assertEqual(env["SO101_MOCK"], "1")
            self.assertEqual(env["SO101_CAMERAS"], "")
            self.assertIn("MUST-NOT-OPEN", env["SO101_FOLLOWER_PORT"])
            self.assertNotIn("UNRELATED", env)
            self.assertEqual(env["PATH"], g15.TRUSTED_PATH)
            self.assertEqual(env["NPM_CONFIG_SCRIPT_SHELL"], "/bin/sh")
            self.assertNotEqual(env["HOME"], "/home/attacker")
            self.assertNotEqual(env["TMPDIR"], str(self.evidence))
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(env["PYTHONSAFEPATH"], "1")
            self.assertEqual(env["PIP_INDEX_URL"], "https://packages.example.test/simple")
            for removed in (
                "MAKE",
                "MAKEFLAGS",
                "PYTEST_ADDOPTS",
                "COVERAGE_RCFILE",
                "NODE_OPTIONS",
                "FRONTEND_INSTALL",
                "VITE_SECRET",
                "PYTHONUSERBASE",
            ):
                self.assertNotIn(removed, env)
        self.assertEqual(executor.npm_userconfigs, ["", ""])
        all_argv = [part for call in probe.calls for part in call]
        all_argv += [part for _name, argv, _env in executor.calls for part in argv]
        for forbidden in ("doctor", "preflight", "hil", "calibrate", "find-cameras"):
            self.assertNotIn(forbidden, all_argv)

    def test_result_json_is_the_last_atomic_publication(self) -> None:
        order: list[str] = []
        original = g15._write_atomic_bytes

        def recording(path: Path, payload: bytes) -> None:
            order.append(path.name)
            original(path, payload)

        with mock.patch.object(g15, "_write_atomic_bytes", recording):
            code, _probe, _executor = self.run_gate()
        self.assertEqual(code, 0)
        self.assertEqual(order[-2:], ["SHA256SUMS", "result.json"])

    def test_postflight_source_change_turns_green_commands_red(self) -> None:
        probe = FakeProbe(final_status=b" M tracked.py\0")
        code, _probe, _executor = self.run_gate(probe=probe)
        self.assertEqual(code, 1)
        result = self.result()
        self.assertEqual(result["failure"]["stage"], "postflight")
        self.assertFalse(result["source"]["unchanged"])

    def test_real_executor_publishes_only_redacted_combined_output(self) -> None:
        self.evidence.mkdir(mode=0o700)
        scratch = self.root / "scratch"
        scratch.mkdir(mode=0o700)
        secret = b"registry-password"
        record = g15.execute_command(
            "probe",
            (
                sys.executable,
                "-c",
                "import sys; print('stdout registry-password'); "
                "print('stderr registry-password', file=sys.stderr)",
            ),
            cwd=self.repo,
            env={"PATH": g15.TRUSTED_PATH},
            log_path=self.evidence / "probe.log",
            relay=g15.SignalRelay(install_handlers=False),
            redact_patterns=(secret,),
            scratch_dir=scratch,
        )
        self.assertTrue(record.passed)
        output = (self.evidence / "probe.log").read_bytes()
        self.assertNotIn(secret, output)
        self.assertEqual(output.count(g15.REDACTION), 2)
        self.assertEqual(record.redactions_applied, 2)
        self.assertEqual([path.name for path in self.evidence.iterdir()], ["probe.log"])

    def test_executor_wait_error_terminates_and_reaps_the_process_group(self) -> None:
        self.evidence.mkdir(mode=0o700)

        class BrokenWaitProcess:
            pid = 987_654
            returncode = None

            def __init__(self) -> None:
                self.wait_calls = 0

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, timeout: float) -> int:
                del timeout
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise RuntimeError("injected wait failure")
                self.returncode = -signal.SIGTERM
                return self.returncode

        process = BrokenWaitProcess()
        forwarded: list[tuple[int, int]] = []
        with (
            mock.patch.object(g15.subprocess, "Popen", return_value=process),
            mock.patch.object(
                g15.os, "killpg", side_effect=lambda pid, signum: forwarded.append((pid, signum))
            ),
            self.assertRaisesRegex(RuntimeError, "injected wait failure"),
        ):
            g15.execute_command(
                "probe",
                ("/usr/bin/make", "bootstrap"),
                cwd=self.repo,
                env={"PATH": g15.TRUSTED_PATH},
                log_path=self.evidence / "probe.log",
                relay=g15.SignalRelay(install_handlers=False),
            )
        self.assertEqual(forwarded, [(process.pid, signal.SIGTERM)])
        self.assertEqual(process.wait_calls, 2)
        self.assertTrue((self.evidence / "probe.log").is_file())


class RefusalCase(RunnerFixture, unittest.TestCase):
    def test_user_owned_path_tool_is_never_trusted(self) -> None:
        tool_dir = self.root / "poisoned-bin"
        tool_dir.mkdir()
        fake = tool_dir / "make"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        with self.assertRaisesRegex(g15.GateError, "not root-owned"):
            g15._resolve_system_tool("make", search_path=str(tool_dir))

    def test_any_case_so101_environment_name_refuses_before_every_subprocess(self) -> None:
        for key in ("SO101_BACKEND", "so101_follower_port", "So101_OpenAi_Api_Key"):
            with self.subTest(key=key):
                evidence = self.root / f"evidence-{len(list(self.root.glob('evidence-*')))}"
                probe = FakeProbe()
                executor = FakeExecutor()
                code, _, _ = self.run_gate(
                    probe=probe,
                    executor=executor,
                    environ={"PATH": "/usr/bin", key: "must-not-be-recorded"},
                    evidence=evidence,
                )
                self.assertEqual(code, 2)
                self.assertEqual(probe.calls, [])
                self.assertEqual(executor.calls, [])
                tree = b"".join(path.read_bytes() for path in evidence.iterdir())
                self.assertNotIn(b"must-not-be-recorded", tree)
                result = self.result(evidence)
                self.assertEqual(result["status"], "refused")
                self.assertEqual(result["failure"]["kind"], "so101_environment")

    def test_static_host_guards_refuse_without_a_probe_or_command(self) -> None:
        variants = (
            _host(system="Darwin"),
            _host(architecture="aarch64"),
            _host(system_vendor="Example Computer Corp."),
            _host(product_name="ROG Strix G16 G614"),
            _host(os_release={"ID": "ubuntu", "VERSION_ID": "22.04"}),
            _host(python_minor=13, python_version="3.13.4"),
            _host(python_isolated=False),
            _host(effective_uid=0),
        )
        for index, host in enumerate(variants):
            with self.subTest(index=index):
                probe = FakeProbe()
                executor = FakeExecutor()
                evidence = self.root / f"host-{index}"
                code, _, _ = self.run_gate(
                    probe=probe, executor=executor, host=host, evidence=evidence
                )
                self.assertEqual(code, 2)
                self.assertEqual(probe.calls, [])
                self.assertEqual(executor.calls, [])

    def test_node_must_be_major_24(self) -> None:
        for version in (b"v23.9.0\n", b"v25.0.0\n", b"not-node\n"):
            with self.subTest(version=version):
                count = len(list(self.root.glob("node-*")))
                evidence = self.root / f"node-{version[:2].hex()}-{count}"
                executor = FakeExecutor()
                code, probe, executor = self.run_gate(
                    probe=FakeProbe(node=version), executor=executor, evidence=evidence
                )
                self.assertEqual(code, 2)
                self.assertEqual(probe.calls, [("/usr/bin/node", "--version")])
                self.assertEqual(executor.calls, [])

    def test_wrong_commit_and_dirty_checkout_refuse(self) -> None:
        for index, probe in enumerate(
            (FakeProbe(commit="d" * 40), FakeProbe(status=b"?? secret.env\0"))
        ):
            with self.subTest(index=index):
                evidence = self.root / f"source-{index}"
                executor = FakeExecutor()
                code, _probe, executor = self.run_gate(
                    probe=probe, executor=executor, evidence=evidence
                )
                self.assertEqual(code, 2)
                self.assertEqual(executor.calls, [])
                result = self.result(evidence)
                self.assertEqual(result["source"]["expected_commit"], EXPECTED)
                if index == 0:
                    self.assertEqual(result["source"]["actual_commit"], "d" * 40)
                else:
                    self.assertNotIn("secret.env", json.dumps(result))

    def test_ignored_reusable_state_and_local_env_files_refuse(self) -> None:
        for index, relative in enumerate((".venv", "frontend/node_modules", ".env")):
            with self.subTest(relative=relative):
                path = self.repo / relative
                if path.suffix == ".env" or path.name == ".env":
                    path.write_text("SO101_FOLLOWER_PORT=/dev/real-secret\n")
                else:
                    path.mkdir(parents=True)
                evidence = self.root / f"reused-{index}"
                code, _probe, executor = self.run_gate(evidence=evidence)
                self.assertEqual(code, 2)
                self.assertEqual(executor.calls, [])
                self.assertNotIn("/dev/real-secret", json.dumps(self.result(evidence)))
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()

    def test_evidence_must_be_external_empty_nonsymlink_mode_0700(self) -> None:
        inside = self.repo / "evidence"
        with self.assertRaisesRegex(g15.GateError, "outside"):
            g15.run_gate(EXPECTED, inside, repo_root=self.repo, environ={}, host_facts=_host())

        nonempty = self.root / "nonempty"
        nonempty.mkdir(mode=0o700)
        sentinel = nonempty / "keep.txt"
        sentinel.write_text("do not overwrite")
        with self.assertRaisesRegex(g15.GateError, "not empty"):
            g15.run_gate(EXPECTED, nonempty, repo_root=self.repo, environ={}, host_facts=_host())
        self.assertEqual(sentinel.read_text(), "do not overwrite")

        public = self.root / "public"
        public.mkdir(mode=0o755)
        with self.assertRaisesRegex(g15.GateError, "0700"):
            g15.run_gate(EXPECTED, public, repo_root=self.repo, environ={}, host_facts=_host())

        target = self.root / "target"
        target.mkdir(mode=0o700)
        symlink = self.root / "linked"
        symlink.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(g15.GateError, "symlink"):
            g15.run_gate(EXPECTED, symlink, repo_root=self.repo, environ={}, host_facts=_host())

    def test_expected_commit_must_be_a_full_lowercase_sha1(self) -> None:
        for value in ("abc", "C" * 40, "g" * 40, EXPECTED + "00"):
            with self.subTest(value=value):
                with self.assertRaises(g15.GateError):
                    g15.run_gate(
                        value,
                        self.root / f"bad-{len(value)}-{value[:1]}",
                        repo_root=self.repo,
                        environ={},
                        host_facts=_host(),
                    )


class MakefileContract(unittest.TestCase):
    def test_g15_runner_is_opt_in_and_not_part_of_make_ci(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text()
        self.assertIn("g15-ci-evidence:", makefile)
        self.assertIn("/usr/bin/python3.12 -I scripts/run_g15_ci.py", makefile)
        ci_body = makefile.split("\nci:", 1)[1].split("\n.PHONY:", 1)[0]
        self.assertNotIn("g15-ci-evidence", ci_body)


if __name__ == "__main__":
    unittest.main()

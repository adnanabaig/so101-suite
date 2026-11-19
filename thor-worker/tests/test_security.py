from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import SCRIPTS, WORKER_ROOT

from dependency_probe import DependencyError, build_command, validate_requirements
from runtime_probe import ProbeError, collect, validate_identity


class SecurityBoundaryTests(unittest.TestCase):
    def test_container_is_non_root_and_has_no_package_install(self) -> None:
        dockerfile = (WORKER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("@sha256:" + "0" * 64, dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn("THOR_WORKER_SOURCE_COMMIT=${SOURCE_COMMIT}", dockerfile)
        self.assertIn('io.so101.thor-worker.mode="qualification-only"', dockerfile)
        self.assertIn("THOR_WORKER_QUALIFICATION_ONLY=1", dockerfile)
        self.assertNotIn("pip install", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertNotIn("COPY . ", dockerfile)
        build_helper = (SCRIPTS / "build-container.sh").read_text(encoding="utf-8")
        self.assertIn('status --porcelain -- .', build_helper)
        self.assertIn('--build-arg "SOURCE_COMMIT=$source_commit"', build_helper)

    def test_runtime_wrapper_has_fixed_device_free_boundary(self) -> None:
        for name in ("run-container.sh", "probe-container.sh"):
            with self.subTest(script=name):
                wrapper = (SCRIPTS / name).read_text(encoding="utf-8")
                executable = "\n".join(
                    line
                    for line in wrapper.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
                self.assertIn("--network=none", executable)
                self.assertIn("--read-only", executable)
                self.assertIn("--cap-drop=ALL", executable)
                self.assertIn("no-new-privileges:true", executable)
                self.assertIn('-- "$immutable_image"', executable)
                self.assertNotIn("--device", executable)
                self.assertNotIn("--privileged", executable)
                self.assertNotIn("docker.sock", executable)
                self.assertNotIn("/dev/tty", executable)
                self.assertNotIn("--tty", executable)
        run_wrapper = (SCRIPTS / "run-container.sh").read_text(encoding="utf-8")
        self.assertIn('[ "$#" -eq 3 ]', run_wrapper)

    def test_host_facts_does_not_enumerate_devices_network_or_environment(self) -> None:
        script = (WORKER_ROOT / "collect-host-facts.sh").read_text(encoding="utf-8")
        executable = "\n".join(
            line
            for line in script.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        for prohibited in (
            "lsusb",
            "udevadm",
            "printenv",
            "ifconfig",
            "ip addr",
            "nmcli",
            "/dev/tty",
            "/dev/serial",
        ):
            self.assertNotIn(prohibited, executable)

    def test_worker_requirements_exclude_hardware_packages_and_extras(self) -> None:
        requirements = validate_requirements(WORKER_ROOT / "requirements-worker.txt")
        joined = "\n".join(requirements).casefold()
        for prohibited in ("pyrealsense", "feetech", "pyserial", "so101[robot", "[realsense"):
            self.assertNotIn(prohibited, joined)
        self.assertIn("lerobot[diffusion,pi,smolvla,training]==0.6.1", requirements)

    def test_requirement_validator_rejects_hardware_and_url_inputs(self) -> None:
        cases = ("pyserial>=3.5\n", "so101[robot]==1\n", "thing @ https://example.invalid/x.whl\n")
        for contents in cases:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "requirements.txt"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(DependencyError):
                    validate_requirements(path)

    def test_dependency_command_is_wheel_only_and_non_installing(self) -> None:
        command = build_command(Path("requirements.txt"), Path("report.json"))
        self.assertIn("--dry-run", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertIn("--no-cache-dir", command)
        self.assertNotIn("--user", command)
        self.assertNotIn("--target", command)

    def test_runtime_identity_requires_commit_and_digest(self) -> None:
        validate_identity("a" * 40, f"image:tag@sha256:{'b' * 64}")
        validate_identity("a" * 40, "HOST-NATIVE")
        with self.assertRaises(ProbeError):
            validate_identity("main", "HOST-NATIVE")
        with self.assertRaises(ProbeError):
            validate_identity("a" * 40, "image:latest")
        with self.assertRaises(ProbeError):
            validate_identity("0" * 40, "HOST-NATIVE")
        with self.assertRaises(ProbeError):
            validate_identity("a" * 40, f"image:tag@sha256:{'0' * 64}")
        with mock.patch.dict(os.environ, {"THOR_WORKER_SOURCE_COMMIT": "b" * 40}):
            with self.assertRaisesRegex(ProbeError, "baked"):
                validate_identity("a" * 40, "HOST-NATIVE")

    def test_container_image_helper_binds_tag_to_immutable_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            repo_digest = "b" * 64
            docker.write_text(
                "#!/usr/bin/env bash\n"
                f"printf 'sha256:{'a' * 64}\\n'\n"
                f"printf 'trusted/repo@sha256:{repo_digest}\\n'\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            command = (
                f". {SCRIPTS / 'container-image.sh'}; "
                f"resolve_pinned_image worker:local worker:local@sha256:{'a' * 64}"
            )
            result = subprocess.run(
                ["bash", "-c", command],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"sha256:{'a' * 64}")

    def test_container_image_helper_rejects_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text(
                "#!/usr/bin/env bash\n"
                f"printf 'sha256:{'a' * 64}\\n'\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            command = (
                f". {SCRIPTS / 'container-image.sh'}; "
                f"resolve_pinned_image worker:local worker:local@sha256:{'c' * 64}"
            )
            result = subprocess.run(
                ["bash", "-c", command],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not match", result.stderr)

    def test_container_image_helper_accepts_pushed_repo_digest_with_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            count_file = Path(directory) / "count"
            docker.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'x\\n' >> \"$DOCKER_COUNT_FILE\"\n"
                f"printf 'sha256:{'a' * 64}\\n'\n"
                f"printf 'trusted/repo@sha256:{'b' * 64}\\n'\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            identity = f"trusted/repo:release@sha256:{'b' * 64}"
            command = (
                f". {SCRIPTS / 'container-image.sh'}; "
                f"resolve_pinned_image worker:local {identity}"
            )
            result = subprocess.run(
                ["bash", "-c", command],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PATH": f"{directory}:{os.environ['PATH']}",
                    "DOCKER_COUNT_FILE": str(count_file),
                },
            )
            calls = count_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"sha256:{'a' * 64}")
        self.assertEqual(calls, ["x"])

    def test_runtime_wrapper_rejects_ancestor_mount_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            output_dir = input_dir / "output"
            output_dir.mkdir(parents=True)
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "run-container.sh"),
                    "worker:local",
                    str(input_dir),
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("overlap", result.stderr)

    def test_runtime_wrapper_rejects_device_output_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "run-container.sh"),
                    "worker:local",
                    str(input_dir),
                    "/dev",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("dedicated evidence", result.stderr)

    def test_wrappers_reject_nonempty_evidence_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (output_dir / "sentinel.txt").write_text("preserve", encoding="utf-8")
            run = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "run-container.sh"),
                    "worker:local",
                    str(input_dir),
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            probe = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "probe-container.sh"),
                    "worker:local",
                    "a" * 40,
                    f"worker:local@sha256:{'b' * 64}",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual((output_dir / "sentinel.txt").read_text(), "preserve")
        self.assertEqual(run.returncode, 2)
        self.assertIn("empty evidence", run.stderr)
        self.assertEqual(probe.returncode, 2)
        self.assertIn("empty evidence", probe.stderr)

    def test_non_thor_host_cannot_be_complete_for_thor(self) -> None:
        result = collect(source_commit="a" * 40, container_image="HOST-NATIVE", target="thor")
        if result["host"]["architecture"] != "aarch64":
            self.assertFalse(result["complete_for_target"])
            self.assertIn("host.architecture(aarch64)", result["missing_required"])

    def test_qualification_image_cannot_claim_an_unbaked_wheelhouse_lock(self) -> None:
        image = f"worker:locked@sha256:{'a' * 64}"
        with mock.patch.dict(
            os.environ,
            {"THOR_WORKER_QUALIFICATION_ONLY": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ProbeError, "baked"):
                collect(
                    source_commit="a" * 40,
                    container_image=image,
                    target="thor",
                    wheelhouse_manifest_sha256="b" * 64,
                )

    def test_shell_scripts_parse(self) -> None:
        scripts = [
            WORKER_ROOT / "collect-host-facts.sh",
            SCRIPTS / "build-container.sh",
            SCRIPTS / "container-image.sh",
            SCRIPTS / "probe-container.sh",
            SCRIPTS / "run-container.sh",
        ]
        result = subprocess.run(
            ["bash", "-n", *map(str, scripts)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(hasattr(os, "chmod"), "POSIX executable modes unavailable")
    def test_build_helper_rejects_placeholder_before_docker(self) -> None:
        script = SCRIPTS / "build-container.sh"
        result = subprocess.run(
            ["bash", str(script), f"nvcr.io/nvidia/image:tag@sha256:{'0' * 64}"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("placeholder", result.stderr)

    def test_build_helper_rejects_non_nvidia_base_before_docker(self) -> None:
        script = SCRIPTS / "build-container.sh"
        result = subprocess.run(
            ["bash", str(script), f"example.invalid/image:tag@sha256:{'a' * 64}"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("nvcr.io/nvidia", result.stderr)


if __name__ == "__main__":
    unittest.main()

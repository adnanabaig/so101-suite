from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from schema_subset import SchemaSet
from support import SCRIPTS, WORKER_ROOT, clone, compatibility, dataset_config, write_json

from benchmark_contract import (
    ContractError,
    canonical_json_bytes,
    validate_compatibility,
    validate_config,
    validate_execution_compatibility,
    validate_result,
)
from artifact_manifest import ManifestError, _validate_manifest, create_bundle_manifest
from run_benchmarks import execute


class CliAndSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = SchemaSet(WORKER_ROOT / "schemas")

    def setUp(self) -> None:
        self.locked_environment = mock.patch.dict(
            os.environ,
            {
                "THOR_WORKER_QUALIFICATION_ONLY": "0",
                "THOR_WORKER_WHEELHOUSE_MANIFEST_SHA256": "c" * 64,
            },
            clear=False,
        )
        self.locked_environment.start()
        self.addCleanup(self.locked_environment.stop)

    def test_every_schema_is_valid_json_and_strict_at_top_level(self) -> None:
        schemas = sorted((WORKER_ROOT / "schemas").glob("*.schema.json"))
        self.assertEqual(len(schemas), 5)
        for path in schemas:
            with self.subTest(path=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertTrue(schema["$id"].endswith(f"/{path.name}"))
                self.assertEqual(schema["type"], "object")
                self.assertIs(schema["additionalProperties"], False)
                self.assertTrue(schema["required"])

    def assertSchemaParity(
        self,
        value: object,
        schema_name: str,
        validator: object,
        *,
        accepted: bool,
    ) -> None:
        try:
            validator(value)  # type: ignore[operator]
        except (ContractError, ValueError):
            python_accepted = False
        else:
            python_accepted = True
        schema_accepted = self.schemas.accepts(value, schema_name)
        self.assertEqual(python_accepted, accepted, "unexpected Python validator verdict")
        self.assertEqual(schema_accepted, accepted, "JSON schema/Python invariant drift")

    def test_config_schema_and_python_validator_parity(self) -> None:
        valid = dataset_config()
        self.assertSchemaParity(
            valid, "benchmark-config.schema.json", validate_config, accepted=True
        )
        invalid_values = []
        bad_tolerance = clone(valid)
        bad_tolerance["comparison"]["absolute_tolerance"] = -1
        invalid_values.append(bad_tolerance)
        wrong_order = clone(valid)
        wrong_order["workloads"][0], wrong_order["workloads"][1] = (
            wrong_order["workloads"][1],
            wrong_order["workloads"][0],
        )
        invalid_values.append(wrong_order)
        remote_policy = clone(valid)
        remote_policy["workloads"][2]["parameters"]["policy"] = "remote_arm"
        invalid_values.append(remote_policy)
        empty_component = clone(valid)
        empty_component["workloads"][0]["parameters"] = {
            "mode": "artifact",
            "bytes_per_iteration": None,
            "iterations": 2,
            "artifact_path": "a//b",
            "manifest_sha256": "d" * 64,
        }
        invalid_values.append(empty_component)
        for invalid in invalid_values:
            self.assertSchemaParity(
                invalid,
                "benchmark-config.schema.json",
                validate_config,
                accepted=False,
            )

    def test_artifact_schema_and_python_validator_path_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.bin").write_bytes(b"payload")
            valid = create_bundle_manifest(
                root,
                kind="other",
                name="schema-parity",
                source_commit="a" * 40,
                created_utc="2026-10-07T12:00:00Z",
            )
        self.schemas.validate(valid, "artifact-manifest.schema.json")
        for path in ("../escape", "a//b", "a/"):
            invalid = clone(valid)
            invalid["files"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(ManifestError):
                _validate_manifest(invalid)
            self.assertFalse(
                self.schemas.accepts(invalid, "artifact-manifest.schema.json")
            )

    def test_compatibility_schema_and_python_validator_parity(self) -> None:
        valid = compatibility()
        self.assertSchemaParity(
            valid, "compatibility.schema.json", validate_compatibility, accepted=True
        )
        invalid_values = []
        placeholder = clone(valid)
        placeholder["qualification"]["wheelhouse_manifest_sha256"] = "0" * 64
        invalid_values.append(placeholder)
        false_cuda_with_devices = clone(valid)
        false_cuda_with_devices["runtime"]["torch"]["cuda_available"] = False
        invalid_values.append(false_cuda_with_devices)
        for invalid in invalid_values:
            self.assertSchemaParity(
                invalid,
                "compatibility.schema.json",
                validate_compatibility,
                accepted=False,
            )

    def test_execution_schema_and_python_validator_parity(self) -> None:
        valid = compatibility()

        def execution_validator(value: object) -> object:
            return validate_execution_compatibility(
                value,
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )

        self.assertSchemaParity(
            valid,
            "execution-compatibility.schema.json",
            execution_validator,
            accepted=True,
        )
        wrong_python = clone(valid)
        wrong_python["runtime"]["python"] = "3.11.9"
        self.assertSchemaParity(
            wrong_python,
            "execution-compatibility.schema.json",
            execution_validator,
            accepted=False,
        )
        qualification_only = clone(valid)
        qualification_only["qualification"] = {
            "mode": "qualification-only",
            "wheelhouse_manifest_sha256": None,
        }
        self.assertSchemaParity(
            qualification_only,
            "execution-compatibility.schema.json",
            execution_validator,
            accepted=False,
        )

    def test_result_schema_and_python_validator_parity(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            valid = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        self.assertSchemaParity(
            valid, "benchmark-result.schema.json", validate_result, accepted=True
        )
        invalid_values = []
        missing_workload = clone(valid)
        missing_workload["results"].pop()
        invalid_values.append(missing_workload)
        zero_duration = clone(valid)
        zero_duration["results"][0]["duration_seconds"] = 0
        invalid_values.append(zero_duration)
        missing_metric = clone(valid)
        missing_metric["results"][0]["metrics"].pop("bytes_per_second")
        invalid_values.append(missing_metric)
        extra_metric = clone(valid)
        extra_metric["results"][0]["metrics"]["unexpected"] = 1
        invalid_values.append(extra_metric)
        duplicate = clone(valid)
        duplicate["results"][-1] = clone(duplicate["results"][0])
        invalid_values.append(duplicate)
        wrong_device = clone(valid)
        wrong_device["config"]["workloads"][1]["enabled"] = True
        wrong_device["config_sha256"] = hashlib.sha256(
            canonical_json_bytes(wrong_device["config"])
        ).hexdigest()
        wrong_device["results"][1] = {
            "name": "torch",
            "status": "passed",
            "parameters": wrong_device["config"]["workloads"][1]["parameters"],
            "duration_seconds": 1.0,
            "metrics": {
                "actual_device": "cpu",
                "iterations": 2,
                "operations": 1,
                "operations_per_second": 1.0,
                "seed": 101,
                "deterministic_algorithms": True,
                "tf32_allowed": False,
                "cublas_workspace_config": ":4096:8",
            },
            "correctness": {
                "sample": [1.0],
                "sample_sha256": "e" * 64,
                "sum": 1.0,
                "finite": True,
            },
            "diagnostic": None,
        }
        invalid_values.append(wrong_device)
        for invalid in invalid_values:
            self.assertSchemaParity(
                invalid,
                "benchmark-result.schema.json",
                validate_result,
                accepted=False,
            )

    def test_cli_runs_stdlib_dataset_benchmark_and_validates_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "benchmark-config.json"
            compatibility_path = root / "compatibility.json"
            result_path = root / "benchmark-result.json"
            write_json(config_path, dataset_config())
            write_json(compatibility_path, compatibility())
            run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "run_benchmarks.py"),
                    "--config",
                    str(config_path),
                    "--compatibility",
                    str(compatibility_path),
                    "--expected-source-commit",
                    "a" * 40,
                    "--expected-container-image",
                    compatibility()["container_image"],
                    "--output",
                    str(result_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(result_path.is_file())
            validate = subprocess.run(
                [sys.executable, str(SCRIPTS / "validate_result.py"), str(result_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertRegex(validate.stdout, r"^ok sha256=[0-9a-f]{64}\n$")

    def test_runtime_probe_cli_writes_a_strict_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "compatibility.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "runtime_probe.py"),
                    "--source-commit",
                    "a" * 40,
                    "--container-image",
                    "HOST-NATIVE",
                    "--target",
                    "other",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["source_commit"], "a" * 40)
            self.assertEqual(document["container_image"], "HOST-NATIVE")
            self.assertIn("missing_required", document)
            self.assertEqual(document["qualification"]["mode"], "qualification-only")
            self.schemas.validate(document, "compatibility.schema.json")

    def test_dependency_probe_default_only_prints_a_dry_run(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "dependency_probe.py"), "--print-command"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--only-binary=:all:", result.stdout)
        self.assertNotIn("pyrealsense", result.stdout.casefold())


if __name__ == "__main__":
    unittest.main()

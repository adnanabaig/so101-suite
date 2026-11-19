from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import WORKER_ROOT, clone, compatibility, dataset_config, g15_compatibility

from artifact_manifest import ManifestError, create_bundle_manifest, manifest_sha256
from benchmark_contract import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    compare_benchmark_results,
    read_json,
    validate_compatibility,
    validate_config,
    validate_execution_compatibility,
    validate_result,
)
import run_benchmarks
from run_benchmarks import execute


class ContractTests(unittest.TestCase):
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

    def test_checked_in_smoke_config_is_valid(self) -> None:
        config = validate_config(read_json(WORKER_ROOT / "configs" / "smoke.json"))
        self.assertEqual(
            [item["name"] for item in config["workloads"]],
            ["dataset", "torch", "inference", "sim"],
        )

    def test_exact_compatibility_tuple_is_valid(self) -> None:
        self.assertEqual(validate_compatibility(compatibility()), compatibility())

    def test_completeness_must_match_missing_fields(self) -> None:
        value = compatibility()
        value["missing_required"] = ["runtime.lerobot"]
        with self.assertRaisesRegex(ContractError, "does not match"):
            validate_compatibility(value)

    def test_null_runtime_cannot_claim_complete(self) -> None:
        value = compatibility()
        value["runtime"]["lerobot"] = None
        with self.assertRaisesRegex(ContractError, "does not match"):
            validate_compatibility(value)

    def test_unpinned_container_is_rejected(self) -> None:
        value = compatibility()
        value["container_image"] = "nvcr.io/nvidia/example:latest"
        with self.assertRaisesRegex(ContractError, "digest-pinned"):
            validate_compatibility(value)

    def test_execution_requires_locked_complete_thor_tuple(self) -> None:
        value = compatibility()
        validate_execution_compatibility(
            value,
            expected_source_commit="a" * 40,
            expected_container_image=value["container_image"],
        )
        cases = []
        wrong_target = clone(value)
        wrong_target["target"] = "g15"
        cases.append(wrong_target)
        incomplete = clone(value)
        incomplete["runtime"]["lerobot"] = None
        incomplete["complete_for_target"] = False
        incomplete["missing_required"] = ["runtime.lerobot"]
        cases.append(incomplete)
        wrong_python = clone(value)
        wrong_python["runtime"]["python"] = "3.11.9"
        cases.append(wrong_python)
        future_python = clone(value)
        future_python["runtime"]["python"] = "3.14.0"
        cases.append(future_python)
        wrong_lerobot = clone(value)
        wrong_lerobot["runtime"]["lerobot"] = "0.7.0"
        cases.append(wrong_lerobot)
        no_cuda = clone(value)
        no_cuda["runtime"]["torch"]["cuda_available"] = False
        no_cuda["runtime"]["torch"]["device_names"] = []
        no_cuda["runtime"]["torch"]["compute_capabilities"] = []
        no_cuda["complete_for_target"] = False
        no_cuda["missing_required"] = ["runtime.torch.cuda_available"]
        cases.append(no_cuda)
        no_cuda_build = clone(value)
        no_cuda_build["runtime"]["torch"]["cuda_build"] = None
        cases.append(no_cuda_build)
        bad_capability = clone(value)
        bad_capability["runtime"]["torch"]["compute_capabilities"] = ["unknown"]
        cases.append(bad_capability)
        inconsistent_cudnn = clone(value)
        inconsistent_cudnn["runtime"]["torch"]["cudnn"] = "different"
        cases.append(inconsistent_cudnn)
        qualification_only = clone(value)
        qualification_only["qualification"] = {
            "mode": "qualification-only",
            "wheelhouse_manifest_sha256": None,
        }
        cases.append(qualification_only)
        for invalid in cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ContractError):
                    validate_execution_compatibility(
                        invalid,
                        expected_source_commit="a" * 40,
                        expected_container_image=value["container_image"],
                    )

    def test_execution_requires_external_source_and_image_identity(self) -> None:
        value = compatibility()
        with self.assertRaisesRegex(ContractError, "source"):
            validate_execution_compatibility(
                value,
                expected_source_commit="d" * 40,
                expected_container_image=value["container_image"],
            )
        with self.assertRaisesRegex(ContractError, "image"):
            validate_execution_compatibility(
                value,
                expected_source_commit=value["source_commit"],
                expected_container_image=f"worker@sha256:{'d' * 64}",
            )

    def test_checked_in_qualification_image_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"THOR_WORKER_QUALIFICATION_ONLY": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ContractError, "qualification-only"):
                execute(
                    validate_config(dataset_config()),
                    compatibility(),
                    artifact_base=Path(directory),
                    expected_source_commit="a" * 40,
                    expected_container_image=compatibility()["container_image"],
                )

    def test_unbound_environment_cannot_execute_locked_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            with self.assertRaisesRegex(ContractError, "qualification-only|wheelhouse"):
                execute(
                    validate_config(dataset_config()),
                    compatibility(),
                    artifact_base=Path(directory),
                    expected_source_commit="a" * 40,
                    expected_container_image=compatibility()["container_image"],
                )

    def test_atomic_json_publication_never_replaces_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_bytes(b"sentinel")
            with self.assertRaises(FileExistsError):
                atomic_write_json(output, {"new": "value"})
            self.assertEqual(output.read_bytes(), b"sentinel")

    def test_artifact_benchmark_never_reads_swapped_unverified_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "bundle"
            root.mkdir()
            payload = root / "payload.bin"
            payload.write_bytes(b"verified")
            create_bundle_manifest(
                root,
                kind="dataset",
                name="swap-test",
                source_commit="a" * 40,
                created_utc="2026-10-07T12:00:00Z",
            )
            parameters = {
                "mode": "artifact",
                "bytes_per_iteration": None,
                "iterations": 1,
                "artifact_path": "bundle",
                "manifest_sha256": manifest_sha256(root),
            }
            backup = parent / "original.bin"
            external = parent / "external.bin"
            external.write_bytes(b"unverified")
            real_verify = run_benchmarks.verify_bundle
            calls = 0

            def swap_around_verify(path: Path) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    manifest = real_verify(path)
                    payload.rename(backup)
                    os.symlink(external, payload)
                    return manifest
                payload.unlink()
                backup.rename(payload)
                return real_verify(path)

            try:
                with mock.patch.object(
                    run_benchmarks,
                    "verify_bundle",
                    side_effect=swap_around_verify,
                ):
                    with self.assertRaises(ManifestError):
                        run_benchmarks._artifact_dataset(parameters, parent, 101)
            finally:
                if payload.is_symlink():
                    payload.unlink()
                if backup.exists():
                    backup.rename(payload)

    def test_unknown_config_fields_are_rejected(self) -> None:
        config = dataset_config()
        config["remote_endpoint"] = "https://robot.invalid"
        with self.assertRaisesRegex(ContractError, "extra"):
            validate_config(config)

    def test_invalid_comparison_tolerances_are_rejected(self) -> None:
        for key, value in (
            ("absolute_tolerance", -1.0),
            ("relative_tolerance", float("inf")),
            ("minimum_throughput_ratio", 0.9),
        ):
            config = dataset_config()
            config["comparison"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_config(config)

    def test_remote_or_unknown_inference_policy_is_rejected(self) -> None:
        config = json.loads((WORKER_ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
        inference = next(item for item in config["workloads"] if item["name"] == "inference")
        inference["parameters"]["policy"] = "remote_arm_policy"
        with self.assertRaisesRegex(ContractError, "local synthetic policy"):
            validate_config(config)

    def test_artifact_path_traversal_is_rejected(self) -> None:
        config = dataset_config()
        parameters = config["workloads"][0]["parameters"]
        parameters.update(
            {
                "mode": "artifact",
                "bytes_per_iteration": None,
                "artifact_path": "../secret",
                "manifest_sha256": "a" * 64,
            }
        )
        with self.assertRaisesRegex(ContractError, "unsafe path"):
            validate_config(config)

    def test_nonfinite_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "non-standard"):
                read_json(path)

    def test_stdlib_dataset_result_validates(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            result = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        self.assertEqual(result["results"][0]["status"], "passed")
        self.assertGreater(result["results"][0]["metrics"]["bytes_per_second"], 0)
        self.assertEqual(validate_result(result), result)
        self.assertEqual(result["config"], config)

    def test_result_rejects_nonfinite_metric(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            result = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        changed = clone(result)
        changed["results"][0]["metrics"]["bytes_per_second"] = float("inf")
        with self.assertRaisesRegex(ContractError, "finite JSON scalar"):
            validate_result(changed)

    def test_result_requires_every_configured_workload_exactly_once(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            result = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        missing = clone(result)
        missing["results"].pop()
        with self.assertRaisesRegex(ContractError, "workload"):
            validate_result(missing)
        duplicate = clone(result)
        duplicate["results"][-1] = clone(duplicate["results"][0])
        with self.assertRaisesRegex(ContractError, "workload|duplicated"):
            validate_result(duplicate)

    def test_cuda_result_cannot_claim_cpu_execution(self) -> None:
        config = validate_config(dataset_config())
        config["workloads"][1]["enabled"] = True
        with tempfile.TemporaryDirectory() as directory:
            result = execute(
                validate_config(dataset_config()),
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        result["config"] = config
        result["config_sha256"] = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
        result["results"][1] = {
            "name": "torch",
            "status": "passed",
            "parameters": config["workloads"][1]["parameters"],
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
        with self.assertRaisesRegex(ContractError, "CUDA|cuda"):
            validate_result(result)

    def test_passed_result_requires_positive_duration_and_contract_fields(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            result = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        zero_duration = clone(result)
        zero_duration["results"][0]["duration_seconds"] = 0.0
        with self.assertRaisesRegex(ContractError, "positive|duration"):
            validate_result(zero_duration)
        missing_metric = clone(result)
        missing_metric["results"][0]["metrics"].pop("bytes_per_second")
        with self.assertRaisesRegex(ContractError, "metrics"):
            validate_result(missing_metric)
        missing_correctness = clone(result)
        missing_correctness["results"][0]["correctness"].clear()
        with self.assertRaisesRegex(ContractError, "correctness"):
            validate_result(missing_correctness)

    def test_torch_policy_enforces_determinism_and_disables_tf32(self) -> None:
        configure = getattr(run_benchmarks, "_configure_torch", None)
        self.assertTrue(callable(configure), "runner lacks a deterministic torch policy")

        class Cuda:
            seeded: list[int] = []

            @classmethod
            def manual_seed_all(cls, seed: int) -> None:
                cls.seeded.append(seed)

        class Matmul:
            allow_tf32 = True

        class Cudnn:
            allow_tf32 = True
            benchmark = True
            deterministic = False

        class Backends:
            cuda = type("CudaBackend", (), {"matmul": Matmul()})()
            cudnn = Cudnn()

        class Torch:
            cuda = Cuda()
            backends = Backends()
            manual_seeds: list[int] = []
            deterministic: list[bool] = []
            precision: list[str] = []

            @classmethod
            def manual_seed(cls, seed: int) -> None:
                cls.manual_seeds.append(seed)

            @classmethod
            def use_deterministic_algorithms(cls, enabled: bool) -> None:
                cls.deterministic.append(enabled)

            @classmethod
            def set_float32_matmul_precision(cls, value: str) -> None:
                cls.precision.append(value)

        policy = configure(Torch, 101)
        self.assertEqual(Torch.manual_seeds, [101])
        self.assertEqual(Cuda.seeded, [101])
        self.assertEqual(Torch.deterministic, [True])
        self.assertEqual(Torch.precision, ["highest"])
        self.assertFalse(Backends.cuda.matmul.allow_tf32)
        self.assertFalse(Backends.cudnn.allow_tf32)
        self.assertFalse(Backends.cudnn.benchmark)
        self.assertTrue(Backends.cudnn.deterministic)
        self.assertEqual(policy["cublas_workspace_config"], ":4096:8")

    def test_comparison_uses_numeric_tolerances_and_throughput_ratio(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            reference = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        candidate = clone(reference)
        reference["compatibility"] = g15_compatibility()
        candidate["results"][0]["metrics"]["bytes_per_second"] *= 2.0
        candidate["results"][0]["duration_seconds"] /= 2.0
        comparison = compare_benchmark_results(reference, candidate)
        self.assertTrue(comparison["qualified"])
        candidate["results"][0]["correctness"]["aggregate_sha256"] = "f" * 64
        comparison = compare_benchmark_results(reference, candidate)
        self.assertFalse(comparison["qualified"])

        tensor_config = clone(config)
        tensor_config["workloads"][0]["enabled"] = False
        tensor_config["workloads"][1]["enabled"] = True
        tensor_digest = hashlib.sha256(canonical_json_bytes(tensor_config)).hexdigest()

        def tensor_entry(sample: list[float], duration: float) -> dict[str, object]:
            operations = 2 * 2 * 32**3
            return {
                "name": "torch",
                "status": "passed",
                "parameters": tensor_config["workloads"][1]["parameters"],
                "duration_seconds": duration,
                "metrics": {
                    "actual_device": "cuda",
                    "iterations": 2,
                    "operations": operations,
                    "operations_per_second": operations / duration,
                    "seed": 101,
                    "deterministic_algorithms": True,
                    "tf32_allowed": False,
                    "cublas_workspace_config": ":4096:8",
                },
                "correctness": {
                    "sample": sample,
                    "sample_sha256": hashlib.sha256(
                        canonical_json_bytes(sample)
                    ).hexdigest(),
                    "sum": sum(sample),
                    "finite": True,
                },
                "diagnostic": None,
            }

        for result in (reference, candidate):
            result["config"] = tensor_config
            result["config_sha256"] = tensor_digest
            result["results"][0] = {
                "name": "dataset",
                "status": "disabled",
                "parameters": tensor_config["workloads"][0]["parameters"],
                "duration_seconds": 0.0,
                "metrics": {},
                "correctness": {},
                "diagnostic": None,
            }
        reference["results"][1] = tensor_entry([1.0, -1.0], 1.0)
        candidate["results"][1] = tensor_entry([2.0, -2.0], 0.5)
        self.assertFalse(compare_benchmark_results(reference, candidate)["qualified"])

        # Numeric simulation state uses the explicit absolute/relative tolerances.
        simulated_config = clone(config)
        simulated_config["workloads"][0]["enabled"] = False
        simulated_config["workloads"][3]["enabled"] = True
        for result in (reference, candidate):
            result["config"] = simulated_config
            result["config_sha256"] = hashlib.sha256(
                canonical_json_bytes(simulated_config)
            ).hexdigest()
            result["results"][0] = {
                "name": "dataset",
                "status": "disabled",
                "parameters": simulated_config["workloads"][0]["parameters"],
                "duration_seconds": 0.0,
                "metrics": {},
                "correctness": {},
                "diagnostic": None,
            }
            for index in (1, 2):
                result["results"][index] = {
                    "name": simulated_config["workloads"][index]["name"],
                    "status": "disabled",
                    "parameters": simulated_config["workloads"][index]["parameters"],
                    "duration_seconds": 0.0,
                    "metrics": {},
                    "correctness": {},
                    "diagnostic": None,
                }
            result["results"][3] = {
                "name": "sim",
                "status": "passed",
                "parameters": simulated_config["workloads"][3]["parameters"],
                "duration_seconds": 0.1,
                "metrics": {"steps": 10, "steps_per_second": 100.0, "seed": 101},
                "correctness": {"finite": True, "final_qpos": 0.5, "final_qvel": 0.0},
                "diagnostic": None,
            }
        candidate["results"][3]["metrics"]["steps_per_second"] = 200.0
        candidate["results"][3]["duration_seconds"] = 0.05
        candidate["results"][3]["correctness"]["final_qpos"] += 5e-6
        self.assertTrue(compare_benchmark_results(reference, candidate)["qualified"])
        candidate["results"][3]["correctness"]["final_qpos"] += 1.0
        self.assertFalse(compare_benchmark_results(reference, candidate)["qualified"])

        with self.assertRaisesRegex(ContractError, "target=g15"):
            compare_benchmark_results(candidate, candidate)

    def test_result_rates_and_counts_must_match_config_and_duration(self) -> None:
        config = validate_config(dataset_config())
        with tempfile.TemporaryDirectory() as directory:
            result = execute(
                config,
                compatibility(),
                artifact_base=Path(directory),
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        wrong_rate = clone(result)
        wrong_rate["results"][0]["metrics"]["bytes_per_second"] *= 2.0
        with self.assertRaisesRegex(ContractError, "count and duration"):
            validate_result(wrong_rate)
        wrong_count = clone(result)
        wrong_count["results"][0]["metrics"]["bytes_processed"] += 1
        with self.assertRaisesRegex(ContractError, "differs from config"):
            validate_result(wrong_count)

    def test_artifact_comparison_rejects_fabricated_byte_or_file_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "payload.bin").write_bytes(b"verified artifact bytes")
            create_bundle_manifest(
                bundle,
                kind="dataset",
                name="comparison-artifact",
                source_commit="a" * 40,
                created_utc="2026-10-07T12:00:00Z",
            )
            config = dataset_config()
            config["workloads"][0]["parameters"] = {
                "mode": "artifact",
                "bytes_per_iteration": None,
                "iterations": 1,
                "artifact_path": "bundle",
                "manifest_sha256": manifest_sha256(bundle),
            }
            reference = execute(
                validate_config(config),
                compatibility(),
                artifact_base=root,
                expected_source_commit="a" * 40,
                expected_container_image=compatibility()["container_image"],
            )
        candidate = clone(reference)
        thor_candidate = clone(candidate)
        reference["compatibility"] = g15_compatibility()
        candidate_metrics = candidate["results"][0]["metrics"]
        candidate_metrics["bytes_processed"] *= 2
        candidate_metrics["bytes_per_second"] *= 2
        self.assertFalse(compare_benchmark_results(reference, candidate)["qualified"])
        candidate = clone(thor_candidate)
        candidate["results"][0]["metrics"]["files"] += 1
        candidate["results"][0]["duration_seconds"] /= 2
        candidate["results"][0]["metrics"]["bytes_per_second"] *= 2
        self.assertFalse(compare_benchmark_results(reference, candidate)["qualified"])


if __name__ == "__main__":
    unittest.main()

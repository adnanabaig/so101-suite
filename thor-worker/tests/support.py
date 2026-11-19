from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

WORKER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = WORKER_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def compatibility() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "captured_utc": "2026-10-07T12:00:00Z",
        "target": "thor",
        "source_commit": "a" * 40,
        "container_image": f"nvcr.io/nvidia/example:1@sha256:{'b' * 64}",
        "qualification": {
            "mode": "locked",
            "wheelhouse_manifest_sha256": "c" * 64,
        },
        "host": {
            "architecture": "aarch64",
            "machine_model": "NVIDIA Jetson AGX Thor Developer Kit",
            "kernel": "6.8.0-test",
            "os_release": {
                "id": "ubuntu",
                "pretty_name": "Ubuntu 24.04 LTS",
                "version_id": "24.04",
            },
            "l4t_release": "# R38 (release), REVISION: 1.0",
            "jetpack": "7.0.0",
            "memory_bytes": 128 * 1024**3,
        },
        "runtime": {
            "python": "3.12.8",
            "python_implementation": "CPython",
            "cuda": "13.0",
            "cudnn": "91000",
            "tensorrt": "10.8.0",
            "torch": {
                "version": "2.8.0",
                "cuda_build": "13.0",
                "cuda_available": True,
                "cudnn": "91000",
                "device_names": ["NVIDIA Jetson Thor"],
                "compute_capabilities": ["10.1"],
            },
            "lerobot": "0.6.1",
            "mujoco": "3.3.7",
            "numpy": "2.1.0",
        },
        "complete_for_target": True,
        "missing_required": [],
    }


def g15_compatibility() -> dict[str, Any]:
    value = clone(compatibility())
    value["target"] = "g15"
    value["container_image"] = "HOST-NATIVE"
    value["host"].update(
        {
            "architecture": "x86_64",
            "machine_model": None,
            "l4t_release": None,
            "jetpack": None,
        }
    )
    value["runtime"]["torch"]["device_names"] = ["NVIDIA RTX 3070 Laptop GPU"]
    value["runtime"]["torch"]["compute_capabilities"] = ["8.6"]
    value["complete_for_target"] = True
    value["missing_required"] = []
    return value


def dataset_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_id": "host-test-v1",
        "seed": 101,
        "comparison": {
            "absolute_tolerance": 1e-5,
            "relative_tolerance": 1e-4,
            "minimum_throughput_ratio": 1.5,
        },
        "workloads": [
            {
                "name": "dataset",
                "enabled": True,
                "parameters": {
                    "mode": "synthetic",
                    "bytes_per_iteration": 4096,
                    "iterations": 2,
                    "artifact_path": None,
                    "manifest_sha256": None,
                },
            },
            {
                "name": "torch",
                "enabled": False,
                "parameters": {
                    "device": "cuda",
                    "precision": "float32",
                    "matrix_size": 32,
                    "warmup": 1,
                    "iterations": 2,
                },
            },
            {
                "name": "inference",
                "enabled": False,
                "parameters": {
                    "policy": "synthetic_mlp_v1",
                    "device": "cuda",
                    "precision": "float32",
                    "batch_size": 1,
                    "input_features": 8,
                    "hidden_features": 8,
                    "output_features": 6,
                    "warmup": 1,
                    "iterations": 2,
                },
            },
            {
                "name": "sim",
                "enabled": False,
                "parameters": {
                    "model": "synthetic_pendulum_v1",
                    "steps": 10,
                },
            },
        ],
    }


def clone(value: Any) -> Any:
    return copy.deepcopy(value)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")

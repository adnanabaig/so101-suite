#!/usr/bin/env python3
"""Strict stdlib validation shared by the worker benchmark tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
IMAGE_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
WORKLOADS = frozenset({"dataset", "torch", "inference", "sim"})
PRECISIONS = frozenset({"float32", "float16", "bfloat16"})
DEVICES = frozenset({"cpu", "cuda", "auto"})


class ContractError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ContractError(f"non-standard JSON number is prohibited: {value}")


def read_json(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    if len(raw) > max_bytes:
        raise ContractError(f"JSON file exceeds {max_bytes} bytes: {path}")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def _object(value: object, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{location} must be an object")
    if set(value) != keys:
        raise ContractError(
            f"{location} keys differ (missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)})"
        )
    return value


def _string_or_none(value: object, location: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ContractError(f"{location} must be a string or null")
    return value


def _integer(value: object, location: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContractError(f"{location} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, location: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{location} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise ContractError(f"{location} must be finite and >= {minimum}")
    return converted


def _timestamp(value: object, location: str) -> str:
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        raise ContractError(f"{location} must be YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ContractError(f"{location} is not a real timestamp") from exc
    return value


def _relative_artifact_path(value: object, location: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise ContractError(f"{location} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ContractError(f"{location} contains an unsafe path component")
    if any(any(ord(char) < 32 or ord(char) == 127 for char in part) for part in parts):
        raise ContractError(f"{location} contains a control character")
    return value


def validate_compatibility(value: object) -> dict[str, Any]:
    document = _object(
        value,
        {
            "schema_version",
            "captured_utc",
            "target",
            "source_commit",
            "container_image",
            "qualification",
            "host",
            "runtime",
            "complete_for_target",
            "missing_required",
        },
        "compatibility",
    )
    if document["schema_version"] != 1:
        raise ContractError("compatibility.schema_version must be 1")
    _timestamp(document["captured_utc"], "compatibility.captured_utc")
    if document["target"] not in {"thor", "g15", "other"}:
        raise ContractError("compatibility.target is invalid")
    if not isinstance(document["source_commit"], str) or not COMMIT_RE.fullmatch(
        document["source_commit"]
    ):
        raise ContractError("compatibility.source_commit must be 40 lowercase hex characters")
    if document["source_commit"] == "0" * 40:
        raise ContractError("compatibility.source_commit may not be all zeroes")
    image = document["container_image"]
    if image != "HOST-NATIVE" and (not isinstance(image, str) or not IMAGE_RE.fullmatch(image)):
        raise ContractError("compatibility.container_image must be HOST-NATIVE or digest-pinned")
    if image != "HOST-NATIVE" and image.endswith("0" * 64):
        raise ContractError("compatibility.container_image may not use the placeholder digest")
    qualification = _object(
        document["qualification"],
        {"mode", "wheelhouse_manifest_sha256"},
        "compatibility.qualification",
    )
    if qualification["mode"] not in {"qualification-only", "locked"}:
        raise ContractError("compatibility.qualification.mode is invalid")
    wheelhouse_digest = qualification["wheelhouse_manifest_sha256"]
    if qualification["mode"] == "qualification-only":
        if wheelhouse_digest is not None:
            raise ContractError("qualification-only compatibility may not claim a wheelhouse lock")
    elif (
        not isinstance(wheelhouse_digest, str)
        or not SHA256_RE.fullmatch(wheelhouse_digest)
        or wheelhouse_digest == "0" * 64
    ):
        raise ContractError("locked compatibility requires a non-placeholder wheelhouse digest")
    host = _object(
        document["host"],
        {
            "architecture",
            "machine_model",
            "kernel",
            "os_release",
            "l4t_release",
            "jetpack",
            "memory_bytes",
        },
        "compatibility.host",
    )
    for key in ("architecture", "machine_model", "kernel", "l4t_release", "jetpack"):
        _string_or_none(host[key], f"compatibility.host.{key}")
    if not isinstance(host["os_release"], dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in host["os_release"].items()
    ):
        raise ContractError("compatibility.host.os_release must map strings to strings")
    if host["memory_bytes"] is not None:
        _integer(
            host["memory_bytes"],
            "compatibility.host.memory_bytes",
            minimum=1,
            maximum=2**63 - 1,
        )
    runtime = _object(
        document["runtime"],
        {
            "python",
            "python_implementation",
            "cuda",
            "cudnn",
            "tensorrt",
            "torch",
            "lerobot",
            "mujoco",
            "numpy",
        },
        "compatibility.runtime",
    )
    for key in (
        "python",
        "python_implementation",
        "cuda",
        "cudnn",
        "tensorrt",
        "lerobot",
        "mujoco",
        "numpy",
    ):
        _string_or_none(runtime[key], f"compatibility.runtime.{key}")
    torch = _object(
        runtime["torch"],
        {
            "version",
            "cuda_build",
            "cuda_available",
            "cudnn",
            "device_names",
            "compute_capabilities",
        },
        "compatibility.runtime.torch",
    )
    for key in ("version", "cuda_build", "cudnn"):
        _string_or_none(torch[key], f"compatibility.runtime.torch.{key}")
    if not isinstance(torch["cuda_available"], bool):
        raise ContractError("compatibility.runtime.torch.cuda_available must be boolean")
    for key in ("device_names", "compute_capabilities"):
        if not isinstance(torch[key], list) or not all(
            isinstance(item, str) for item in torch[key]
        ):
            raise ContractError(f"compatibility.runtime.torch.{key} must be a string array")
    if torch["cuda_available"]:
        if not torch["device_names"] or len(torch["device_names"]) != len(
            torch["compute_capabilities"]
        ):
            raise ContractError("CUDA availability requires one capability per named device")
        if any(not name.strip() for name in torch["device_names"]):
            raise ContractError("CUDA device names may not be empty")
        if any(
            re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None
            for capability in torch["compute_capabilities"]
        ):
            raise ContractError("CUDA compute capabilities must use major.minor syntax")
    elif torch["device_names"] or torch["compute_capabilities"]:
        raise ContractError("CUDA-unavailable compatibility may not list CUDA devices")
    if not isinstance(document["complete_for_target"], bool):
        raise ContractError("compatibility.complete_for_target must be boolean")
    missing = document["missing_required"]
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise ContractError("compatibility.missing_required must be a string array")
    if missing != sorted(set(missing)):
        raise ContractError("compatibility.missing_required must be sorted and unique")
    required: dict[str, object] = {
        "host.architecture": host["architecture"],
        "host.kernel": host["kernel"],
        "runtime.python": runtime["python"],
        "runtime.torch": torch["version"],
        "runtime.lerobot": runtime["lerobot"],
    }
    if document["target"] == "thor":
        required.update(
            {
                "host.machine_model": host["machine_model"],
                "host.l4t_release": host["l4t_release"],
                "host.jetpack": host["jetpack"],
                "runtime.cuda": runtime["cuda"],
                "runtime.cudnn": runtime["cudnn"],
                "runtime.tensorrt": runtime["tensorrt"],
            }
        )
        if host["architecture"] != "aarch64":
            required["host.architecture(aarch64)"] = None
        if not host["machine_model"] or "thor" not in str(host["machine_model"]).casefold():
            required["host.machine_model(Jetson Thor)"] = None
        if not torch["cuda_available"]:
            required["runtime.torch.cuda_available"] = None
    elif document["target"] == "g15":
        required.update(
            {
                "runtime.cuda": runtime["cuda"],
                "runtime.cudnn": runtime["cudnn"],
            }
        )
        if host["architecture"] not in {"x86_64", "amd64"}:
            required["host.architecture(x86_64)"] = None
        if not torch["cuda_available"]:
            required["runtime.torch.cuda_available"] = None
    expected_missing = sorted(
        key
        for key, required_value in required.items()
        if required_value is None or required_value == ""
    )
    if missing != expected_missing:
        raise ContractError(
            "compatibility.missing_required does not match the measured compatibility fields"
        )
    if document["complete_for_target"] != (not missing):
        raise ContractError("compatibility completeness disagrees with missing_required")
    return document


def validate_execution_compatibility(
    value: object,
    *,
    expected_source_commit: str,
    expected_container_image: str,
) -> dict[str, Any]:
    document = validate_compatibility(value)
    if document["target"] != "thor":
        raise ContractError("benchmark execution requires target=thor")
    if not document["complete_for_target"] or document["missing_required"]:
        raise ContractError("benchmark execution requires a complete Thor compatibility tuple")
    if document["source_commit"] != expected_source_commit:
        raise ContractError("compatibility source commit differs from the executing worker")
    if document["container_image"] != expected_container_image:
        raise ContractError("compatibility image differs from the inspected immutable image")
    if document["container_image"] == "HOST-NATIVE":
        raise ContractError("Thor benchmark execution requires a digest-pinned container image")
    qualification = document["qualification"]
    if qualification["mode"] != "locked":
        raise ContractError("qualification-only images may not execute promotion benchmarks")
    runtime = document["runtime"]
    python = runtime["python"]
    if not isinstance(python, str) or re.fullmatch(
        r"3\.(?:12|13)\.[0-9]+(?:[+.-][A-Za-z0-9._-]+)?", python
    ) is None:
        raise ContractError("Thor execution requires a measured Python 3.12 or 3.13 runtime")
    if runtime["lerobot"] != "0.6.1":
        raise ContractError("Thor execution requires exactly LeRobot 0.6.1")
    for key in ("cuda", "cudnn", "tensorrt"):
        if not isinstance(runtime[key], str) or not runtime[key].strip():
            raise ContractError(f"Thor execution requires runtime.{key}")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", runtime["cuda"]) is None:
        raise ContractError("Thor execution requires CUDA as an exact numeric version")
    torch = runtime["torch"]
    if not isinstance(torch["version"], str) or re.fullmatch(
        r"[0-9]+\.[0-9]+(?:\.[0-9]+)?[A-Za-z0-9.+_-]*", torch["version"]
    ) is None:
        raise ContractError("Thor execution requires an exact PyTorch version")
    if not isinstance(torch["cuda_build"], str) or re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)+", torch["cuda_build"]
    ) is None:
        raise ContractError("Thor execution requires PyTorch's exact CUDA build")
    if not torch["cuda_available"] or not torch["device_names"]:
        raise ContractError("Thor execution requires an available CUDA device")
    if len(torch["device_names"]) != 1 or len(torch["compute_capabilities"]) != 1:
        raise ContractError("Thor execution requires exactly one measured CUDA device")
    if not isinstance(torch["cudnn"], str) or torch["cudnn"] != runtime["cudnn"]:
        raise ContractError("Thor execution requires consistent PyTorch/runtime cuDNN truth")
    return document


def validate_g15_reference_compatibility(value: object) -> dict[str, Any]:
    document = validate_compatibility(value)
    if document["target"] != "g15":
        raise ContractError("promotion comparison requires a target=g15 reference")
    if not document["complete_for_target"] or document["missing_required"]:
        raise ContractError("promotion comparison requires a complete G15 tuple")
    if document["qualification"]["mode"] != "locked":
        raise ContractError("promotion comparison requires a dependency-locked G15 reference")
    runtime = document["runtime"]
    python = runtime["python"]
    if not isinstance(python, str) or re.fullmatch(
        r"3\.(?:12|13)\.[0-9]+(?:[+.-][A-Za-z0-9._-]+)?", python
    ) is None:
        raise ContractError("G15 reference requires Python 3.12 or 3.13")
    if runtime["lerobot"] != "0.6.1":
        raise ContractError("G15 reference requires exactly LeRobot 0.6.1")
    if not isinstance(runtime["cuda"], str) or re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)+", runtime["cuda"]
    ) is None:
        raise ContractError("G15 reference requires an exact CUDA version")
    torch = runtime["torch"]
    if not isinstance(torch["version"], str) or re.fullmatch(
        r"[0-9]+\.[0-9]+(?:\.[0-9]+)?[A-Za-z0-9.+_-]*", torch["version"]
    ) is None:
        raise ContractError("G15 reference requires an exact PyTorch version")
    if not torch["cuda_available"] or not torch["device_names"]:
        raise ContractError("G15 reference requires an available CUDA device")
    if not isinstance(torch["cuda_build"], str) or re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)+", torch["cuda_build"]
    ) is None:
        raise ContractError("G15 reference requires PyTorch's exact CUDA build")
    if not isinstance(torch["cudnn"], str) or torch["cudnn"] != runtime["cudnn"]:
        raise ContractError("G15 reference requires consistent PyTorch/runtime cuDNN truth")
    return document


def _validate_parameters(name: str, value: object, location: str) -> dict[str, Any]:
    if name == "dataset":
        parameters = _object(
            value,
            {"mode", "bytes_per_iteration", "iterations", "artifact_path", "manifest_sha256"},
            location,
        )
        if parameters["mode"] not in {"synthetic", "artifact"}:
            raise ContractError(f"{location}.mode is invalid")
        _integer(parameters["iterations"], f"{location}.iterations", minimum=1, maximum=10000)
        if parameters["mode"] == "synthetic":
            _integer(
                parameters["bytes_per_iteration"],
                f"{location}.bytes_per_iteration",
                minimum=1,
                maximum=1024**3,
            )
            if parameters["artifact_path"] is not None or parameters["manifest_sha256"] is not None:
                raise ContractError(f"{location}: synthetic mode may not name an artifact")
        else:
            if parameters["bytes_per_iteration"] is not None:
                raise ContractError(f"{location}: artifact mode bytes_per_iteration must be null")
            _relative_artifact_path(parameters["artifact_path"], f"{location}.artifact_path")
            digest = parameters["manifest_sha256"]
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                raise ContractError(
                    f"{location}.manifest_sha256 must be 64 lowercase hex characters"
                )
        return parameters
    if name == "torch":
        parameters = _object(
            value,
            {"device", "precision", "matrix_size", "warmup", "iterations"},
            location,
        )
        if parameters["device"] not in DEVICES or parameters["precision"] not in PRECISIONS:
            raise ContractError(f"{location} has an invalid device or precision")
        _integer(parameters["matrix_size"], f"{location}.matrix_size", minimum=2, maximum=8192)
        _integer(parameters["warmup"], f"{location}.warmup", minimum=0, maximum=1000)
        _integer(parameters["iterations"], f"{location}.iterations", minimum=1, maximum=10000)
        return parameters
    if name == "inference":
        parameters = _object(
            value,
            {
                "policy",
                "device",
                "precision",
                "batch_size",
                "input_features",
                "hidden_features",
                "output_features",
                "warmup",
                "iterations",
            },
            location,
        )
        if parameters["policy"] != "synthetic_mlp_v1":
            raise ContractError(f"{location}.policy must be the local synthetic policy")
        if parameters["device"] not in DEVICES or parameters["precision"] not in PRECISIONS:
            raise ContractError(f"{location} has an invalid device or precision")
        _integer(parameters["batch_size"], f"{location}.batch_size", minimum=1, maximum=4096)
        for key in ("input_features", "hidden_features", "output_features"):
            _integer(parameters[key], f"{location}.{key}", minimum=1, maximum=65536)
        _integer(parameters["warmup"], f"{location}.warmup", minimum=0, maximum=1000)
        _integer(parameters["iterations"], f"{location}.iterations", minimum=1, maximum=100000)
        return parameters
    if name == "sim":
        parameters = _object(value, {"model", "steps"}, location)
        if parameters["model"] != "synthetic_pendulum_v1":
            raise ContractError(f"{location}.model must be synthetic_pendulum_v1")
        _integer(parameters["steps"], f"{location}.steps", minimum=1, maximum=10_000_000)
        return parameters
    raise ContractError(f"unsupported workload: {name!r}")


def validate_config(value: object) -> dict[str, Any]:
    config = _object(
        value,
        {"schema_version", "benchmark_id", "seed", "comparison", "workloads"},
        "config",
    )
    if config["schema_version"] != 1:
        raise ContractError("config.schema_version must be 1")
    if not isinstance(config["benchmark_id"], str) or not NAME_RE.fullmatch(config["benchmark_id"]):
        raise ContractError("config.benchmark_id is invalid")
    _integer(config["seed"], "config.seed", minimum=0, maximum=2**32 - 1)
    comparison = _object(
        config["comparison"],
        {"absolute_tolerance", "relative_tolerance", "minimum_throughput_ratio"},
        "config.comparison",
    )
    absolute = _number(
        comparison["absolute_tolerance"], "config.comparison.absolute_tolerance"
    )
    relative = _number(
        comparison["relative_tolerance"], "config.comparison.relative_tolerance"
    )
    throughput = _number(
        comparison["minimum_throughput_ratio"],
        "config.comparison.minimum_throughput_ratio",
        minimum=1.0,
    )
    if absolute > 1.0 or relative > 1.0 or throughput > 100.0:
        raise ContractError("config.comparison tolerances or throughput ratio are unreasonable")
    workloads = config["workloads"]
    if not isinstance(workloads, list) or len(workloads) != len(WORKLOADS):
        raise ContractError("config.workloads must contain all four canonical workloads")
    names: set[str] = set()
    for index, raw in enumerate(workloads):
        workload = _object(raw, {"name", "enabled", "parameters"}, f"config.workloads[{index}]")
        name = workload["name"]
        if name not in WORKLOADS:
            raise ContractError(f"config.workloads[{index}].name is unsupported")
        if name in names:
            raise ContractError(f"duplicate workload: {name}")
        names.add(name)
        if not isinstance(workload["enabled"], bool):
            raise ContractError(f"config.workloads[{index}].enabled must be boolean")
        _validate_parameters(name, workload["parameters"], f"config.workloads[{index}].parameters")
    canonical_names = ["dataset", "torch", "inference", "sim"]
    if [workload["name"] for workload in workloads] != canonical_names:
        raise ContractError("config.workloads must use canonical dataset/torch/inference/sim order")
    return config


def _json_metrics(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{location} must be an object")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ContractError(f"{location} keys must be strings")
        if item is None or isinstance(item, (str, bool)):
            continue
        if isinstance(item, list) and item and len(item) <= 16 and all(
            isinstance(part, (int, float))
            and not isinstance(part, bool)
            and math.isfinite(float(part))
            for part in item
        ):
            continue
        if (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
        ):
            continue
        raise ContractError(f"{location}.{key} must be a finite JSON scalar")
    return value


def _positive_integer(value: object, location: str) -> int:
    return _integer(value, location, minimum=1, maximum=2**63 - 1)


def _sha256(value: object, location: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _validate_device(requested: str, actual: object, location: str) -> None:
    if actual not in {"cpu", "cuda"}:
        raise ContractError(f"{location}.actual_device must be cpu or cuda")
    if requested == "cuda" and actual != "cuda":
        raise ContractError(f"{location} requested CUDA but did not execute on CUDA")
    if requested == "cpu" and actual != "cpu":
        raise ContractError(f"{location} requested CPU but did not execute on CPU")


def _validate_determinism(metrics: dict[str, Any], seed: int, location: str) -> None:
    _integer(metrics["seed"], f"{location}.seed", minimum=0, maximum=2**32 - 1)
    if metrics["seed"] != seed:
        raise ContractError(f"{location}.seed differs from the benchmark seed")
    if metrics["deterministic_algorithms"] is not True:
        raise ContractError(f"{location} did not enable deterministic algorithms")
    if metrics["tf32_allowed"] is not False:
        raise ContractError(f"{location} must disable TF32 for correctness comparison")
    if metrics["cublas_workspace_config"] != ":4096:8":
        raise ContractError(f"{location} has the wrong deterministic cuBLAS policy")


def _validate_rate(amount: int, rate: float, duration: float, location: str) -> None:
    expected = amount / duration
    if not math.isclose(rate, expected, rel_tol=1e-12, abs_tol=0.0):
        raise ContractError(f"{location} is inconsistent with count and duration")


def _validate_passed_workload(
    entry: dict[str, Any],
    workload: dict[str, Any],
    seed: int,
    duration: float,
    location: str,
) -> None:
    name = entry["name"]
    parameters = workload["parameters"]
    metrics = entry["metrics"]
    correctness = entry["correctness"]
    if name == "dataset":
        metric_keys = {
            "iterations",
            "bytes_processed",
            "bytes_per_second",
            "mode",
            "seed",
        }
        correctness_keys = {"aggregate_sha256"}
        if parameters["mode"] == "artifact":
            metric_keys.add("files")
            correctness_keys.add("dataset_manifest_sha256")
        _object(metrics, metric_keys, f"{location}.metrics")
        _object(correctness, correctness_keys, f"{location}.correctness")
        _positive_integer(metrics["iterations"], f"{location}.metrics.iterations")
        _integer(metrics["seed"], f"{location}.metrics.seed", minimum=0, maximum=2**32 - 1)
        if metrics["iterations"] != parameters["iterations"]:
            raise ContractError(f"{location}.metrics.iterations differs from config")
        if metrics["mode"] != parameters["mode"] or metrics["seed"] != seed:
            raise ContractError(f"{location}.metrics does not identify the configured dataset run")
        bytes_processed = _positive_integer(
            metrics["bytes_processed"], f"{location}.metrics.bytes_processed"
        )
        rate = _number(metrics["bytes_per_second"], f"{location}.metrics.bytes_per_second")
        if rate <= 0.0:
            raise ContractError(f"{location}.metrics.bytes_per_second must be positive")
        if parameters["mode"] == "synthetic":
            expected_bytes = parameters["bytes_per_iteration"] * parameters["iterations"]
            if bytes_processed != expected_bytes:
                raise ContractError(f"{location}.metrics.bytes_processed differs from config")
        elif bytes_processed % parameters["iterations"] != 0:
            raise ContractError(f"{location}.metrics.bytes_processed is not iteration-consistent")
        _validate_rate(bytes_processed, rate, duration, f"{location}.metrics.bytes_per_second")
        _sha256(correctness["aggregate_sha256"], f"{location}.correctness.aggregate_sha256")
        if parameters["mode"] == "artifact":
            _positive_integer(metrics["files"], f"{location}.metrics.files")
            digest = _sha256(
                correctness["dataset_manifest_sha256"],
                f"{location}.correctness.dataset_manifest_sha256",
            )
            if digest != parameters["manifest_sha256"]:
                raise ContractError(f"{location} used a different dataset manifest")
        return

    if name in {"torch", "inference"}:
        amount_key = "operations" if name == "torch" else "samples"
        rate_key = "operations_per_second" if name == "torch" else "samples_per_second"
        _object(
            metrics,
            {
                "actual_device",
                "iterations",
                amount_key,
                rate_key,
                "seed",
                "deterministic_algorithms",
                "tf32_allowed",
                "cublas_workspace_config",
            },
            f"{location}.metrics",
        )
        _object(
            correctness,
            {"sample", "sample_sha256", "sum", "finite"},
            f"{location}.correctness",
        )
        _validate_device(parameters["device"], metrics["actual_device"], location)
        _validate_determinism(metrics, seed, f"{location}.metrics")
        _positive_integer(metrics["iterations"], f"{location}.metrics.iterations")
        if metrics["iterations"] != parameters["iterations"]:
            raise ContractError(f"{location}.metrics.iterations differs from config")
        amount = _positive_integer(metrics[amount_key], f"{location}.metrics.{amount_key}")
        expected_amount = (
            parameters["iterations"] * 2 * parameters["matrix_size"] ** 3
            if name == "torch"
            else parameters["batch_size"] * parameters["iterations"]
        )
        if amount != expected_amount:
            raise ContractError(f"{location}.metrics.{amount_key} differs from config")
        rate = _number(metrics[rate_key], f"{location}.metrics.{rate_key}")
        if rate <= 0.0:
            raise ContractError(f"{location}.metrics.{rate_key} must be positive")
        _validate_rate(amount, rate, duration, f"{location}.metrics.{rate_key}")
        _sha256(correctness["sample_sha256"], f"{location}.correctness.sample_sha256")
        sample = correctness["sample"]
        if not isinstance(sample, list) or not 1 <= len(sample) <= 16 or not all(
            isinstance(part, (int, float))
            and not isinstance(part, bool)
            and math.isfinite(float(part))
            for part in sample
        ):
            raise ContractError(f"{location}.correctness.sample must contain finite numbers")
        expected_sample_digest = hashlib.sha256(canonical_json_bytes(sample)).hexdigest()
        if correctness["sample_sha256"] != expected_sample_digest:
            raise ContractError(f"{location}.correctness sample digest does not match sample")
        _number(correctness["sum"], f"{location}.correctness.sum", minimum=-float("inf"))
        if correctness["finite"] is not True:
            raise ContractError(f"{location}.correctness.finite must be true")
        return

    _object(metrics, {"steps", "steps_per_second", "seed"}, f"{location}.metrics")
    _object(
        correctness,
        {"finite", "final_qpos", "final_qvel"},
        f"{location}.correctness",
    )
    _positive_integer(metrics["steps"], f"{location}.metrics.steps")
    _integer(metrics["seed"], f"{location}.metrics.seed", minimum=0, maximum=2**32 - 1)
    if metrics["steps"] != parameters["steps"] or metrics["seed"] != seed:
        raise ContractError(f"{location}.metrics differs from the configured simulation")
    rate = _number(metrics["steps_per_second"], f"{location}.metrics.steps_per_second")
    if rate <= 0.0:
        raise ContractError(f"{location}.metrics.steps_per_second must be positive")
    _validate_rate(metrics["steps"], rate, duration, f"{location}.metrics.steps_per_second")
    if correctness["finite"] is not True:
        raise ContractError(f"{location}.correctness.finite must be true")
    _number(
        correctness["final_qpos"],
        f"{location}.correctness.final_qpos",
        minimum=-float("inf"),
    )
    _number(
        correctness["final_qvel"],
        f"{location}.correctness.final_qvel",
        minimum=-float("inf"),
    )


def validate_result(value: object) -> dict[str, Any]:
    result = _object(
        value,
        {
            "schema_version",
            "benchmark_id",
            "run_id",
            "started_utc",
            "finished_utc",
            "compatibility",
            "config",
            "config_sha256",
            "seed",
            "results",
        },
        "result",
    )
    if result["schema_version"] != 1:
        raise ContractError("result.schema_version must be 1")
    if not isinstance(result["benchmark_id"], str) or not NAME_RE.fullmatch(result["benchmark_id"]):
        raise ContractError("result.benchmark_id is invalid")
    if not isinstance(result["run_id"], str) or not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        result["run_id"],
    ):
        raise ContractError("result.run_id must be a lowercase UUIDv4")
    started = _timestamp(result["started_utc"], "result.started_utc")
    finished = _timestamp(result["finished_utc"], "result.finished_utc")
    if datetime.strptime(finished, "%Y-%m-%dT%H:%M:%SZ") < datetime.strptime(
        started, "%Y-%m-%dT%H:%M:%SZ"
    ):
        raise ContractError("result.finished_utc precedes result.started_utc")
    validate_compatibility(result["compatibility"])
    config = validate_config(result["config"])
    if not isinstance(result["config_sha256"], str) or not SHA256_RE.fullmatch(
        result["config_sha256"]
    ):
        raise ContractError("result.config_sha256 is invalid")
    expected_config_digest = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    if result["config_sha256"] != expected_config_digest:
        raise ContractError("result.config_sha256 does not match result.config")
    _integer(result["seed"], "result.seed", minimum=0, maximum=2**32 - 1)
    if result["benchmark_id"] != config["benchmark_id"] or result["seed"] != config["seed"]:
        raise ContractError("result identity differs from its embedded config")
    entries = result["results"]
    if not isinstance(entries, list) or len(entries) != len(config["workloads"]):
        raise ContractError("result must contain every configured workload exactly once")
    names: set[str] = set()
    for index, raw in enumerate(entries):
        entry = _object(
            raw,
            {
                "name",
                "status",
                "parameters",
                "duration_seconds",
                "metrics",
                "correctness",
                "diagnostic",
            },
            f"result.results[{index}]",
        )
        name = entry["name"]
        if name not in WORKLOADS or name in names:
            raise ContractError(f"result.results[{index}].name is invalid or duplicated")
        names.add(name)
        configured = config["workloads"][index]
        if name != configured["name"]:
            raise ContractError("result workloads must match the complete canonical config order")
        if entry["parameters"] != configured["parameters"]:
            raise ContractError(f"result.results[{index}].parameters differ from config")
        status = entry["status"]
        if status not in {"passed", "unavailable", "failed", "disabled"}:
            raise ContractError(f"result.results[{index}].status is invalid")
        _validate_parameters(name, entry["parameters"], f"result.results[{index}].parameters")
        duration = _number(
            entry["duration_seconds"], f"result.results[{index}].duration_seconds"
        )
        metrics = _json_metrics(entry["metrics"], f"result.results[{index}].metrics")
        correctness = _json_metrics(
            entry["correctness"], f"result.results[{index}].correctness"
        )
        if entry["diagnostic"] is not None and not isinstance(entry["diagnostic"], str):
            raise ContractError(f"result.results[{index}].diagnostic must be string or null")
        if status == "passed" and entry["diagnostic"] is not None:
            raise ContractError(f"result.results[{index}] passed with a diagnostic")
        if status == "disabled" and (
            duration != 0.0 or metrics or correctness or entry["diagnostic"] is not None
        ):
            raise ContractError(f"result.results[{index}] has invalid disabled-state fields")
        if status in {"unavailable", "failed"} and (
            duration != 0.0
            or metrics
            or correctness
            or not isinstance(entry["diagnostic"], str)
            or not entry["diagnostic"]
        ):
            raise ContractError(f"result.results[{index}] has invalid failure-state fields")
        if configured["enabled"] and status == "disabled":
            raise ContractError(f"result.results[{index}] disabled an enabled workload")
        if not configured["enabled"] and status != "disabled":
            raise ContractError(f"result.results[{index}] ran a disabled workload")
        if status == "passed":
            if duration <= 0.0:
                raise ContractError(f"result.results[{index}] passed without positive duration")
            _validate_passed_workload(
                entry,
                configured,
                result["seed"],
                duration,
                f"result.results[{index}]",
            )
    return result


def compare_benchmark_results(
    reference_value: object, candidate_value: object
) -> dict[str, Any]:
    """Compare two validated runs using the tolerances embedded in their config."""
    reference = validate_result(reference_value)
    candidate = validate_result(candidate_value)
    validate_g15_reference_compatibility(reference["compatibility"])
    validate_execution_compatibility(
        candidate["compatibility"],
        expected_source_commit=candidate["compatibility"]["source_commit"],
        expected_container_image=candidate["compatibility"]["container_image"],
    )
    if reference["config_sha256"] != candidate["config_sha256"]:
        raise ContractError("benchmark comparison requires the same canonical config")
    if reference["compatibility"]["source_commit"] != candidate["compatibility"][
        "source_commit"
    ]:
        raise ContractError("benchmark comparison requires the same source commit")
    comparison = reference["config"]["comparison"]
    absolute = float(comparison["absolute_tolerance"])
    relative = float(comparison["relative_tolerance"])
    minimum_ratio = float(comparison["minimum_throughput_ratio"])
    rate_keys = {
        "dataset": "bytes_per_second",
        "torch": "operations_per_second",
        "inference": "samples_per_second",
        "sim": "steps_per_second",
    }
    outcomes: list[dict[str, Any]] = []
    reference_entries = {entry["name"]: entry for entry in reference["results"]}
    candidate_entries = {entry["name"]: entry for entry in candidate["results"]}
    for workload in reference["config"]["workloads"]:
        if not workload["enabled"]:
            continue
        name = workload["name"]
        baseline = reference_entries[name]
        contender = candidate_entries[name]
        if baseline["status"] != "passed" or contender["status"] != "passed":
            outcomes.append(
                {
                    "name": name,
                    "correctness_equal": False,
                    "throughput_ratio": None,
                    "throughput_qualified": False,
                }
            )
            continue
        if name == "dataset":
            equal = baseline["correctness"]["aggregate_sha256"] == contender[
                "correctness"
            ]["aggregate_sha256"] and baseline["metrics"]["bytes_processed"] == contender[
                "metrics"
            ]["bytes_processed"]
            if workload["parameters"]["mode"] == "artifact":
                equal = (
                    equal
                    and baseline["metrics"]["files"] == contender["metrics"]["files"]
                    and baseline["correctness"]["dataset_manifest_sha256"]
                    == contender["correctness"]["dataset_manifest_sha256"]
                )
        elif name in {"torch", "inference"}:
            baseline_sample = baseline["correctness"]["sample"]
            contender_sample = contender["correctness"]["sample"]
            equal = len(baseline_sample) == len(contender_sample) and all(
                math.isclose(left, right, rel_tol=relative, abs_tol=absolute)
                for left, right in zip(baseline_sample, contender_sample, strict=True)
            ) and math.isclose(
                baseline["correctness"]["sum"],
                contender["correctness"]["sum"],
                rel_tol=relative,
                abs_tol=absolute,
            )
        else:
            equal = all(
                math.isclose(
                    baseline["correctness"][key],
                    contender["correctness"][key],
                    rel_tol=relative,
                    abs_tol=absolute,
                )
                for key in ("final_qpos", "final_qvel")
            )
        rate_key = rate_keys[name]
        ratio = contender["metrics"][rate_key] / baseline["metrics"][rate_key]
        outcomes.append(
            {
                "name": name,
                "correctness_equal": equal,
                "throughput_ratio": ratio,
                "throughput_qualified": ratio >= minimum_ratio,
            }
        )
    return {
        "schema_version": 1,
        "config_sha256": reference["config_sha256"],
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "minimum_throughput_ratio": minimum_ratio,
        "workloads": outcomes,
        "qualified": bool(outcomes)
        and all(
            outcome["correctness_equal"] and outcome["throughput_qualified"]
            for outcome in outcomes
        ),
    }


def atomic_write_json(path: Path, value: object) -> None:
    content = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

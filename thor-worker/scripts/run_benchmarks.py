#!/usr/bin/env python3
"""Run bounded offline worker benchmarks and emit one strict JSON result."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import stat
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from artifact_manifest import manifest_sha256, stream_verified_file, verify_bundle
from benchmark_contract import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    validate_compatibility,
    validate_config,
    validate_execution_compatibility,
    validate_result,
)


class WorkloadUnavailable(RuntimeError):
    pass


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _entry(
    name: str,
    parameters: dict[str, Any],
    *,
    status: str,
    duration: float = 0.0,
    metrics: dict[str, Any] | None = None,
    correctness: dict[str, Any] | None = None,
    diagnostic: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "parameters": parameters,
        "duration_seconds": duration,
        "metrics": metrics or {},
        "correctness": correctness or {},
        "diagnostic": diagnostic,
    }


def _synthetic_dataset(
    parameters: dict[str, Any], seed: int
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    bytes_per_iteration = parameters["bytes_per_iteration"]
    iterations = parameters["iterations"]
    pattern = hashlib.sha256(f"so101-thor-dataset:{seed}".encode()).digest()
    chunk_size = min(bytes_per_iteration, 1024 * 1024)
    chunk = (pattern * ((chunk_size + len(pattern) - 1) // len(pattern)))[:chunk_size]
    aggregate = hashlib.sha256()
    start = time.perf_counter()
    for _ in range(iterations):
        digest = hashlib.sha256()
        remaining = bytes_per_iteration
        while remaining:
            piece = chunk[: min(remaining, len(chunk))]
            digest.update(piece)
            remaining -= len(piece)
        aggregate.update(digest.digest())
    duration = time.perf_counter() - start
    total = bytes_per_iteration * iterations
    return (
        duration,
        {
            "iterations": iterations,
            "bytes_processed": total,
            "bytes_per_second": total / duration if duration else 0.0,
            "mode": "synthetic",
            "seed": seed,
        },
        {"aggregate_sha256": aggregate.hexdigest()},
    )


def _safe_artifact_root(base: Path, relative: str) -> Path:
    base = base.resolve(strict=True)
    current = base
    for component in relative.split("/"):
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"artifact path crosses a symlink: {relative!r}")
    resolved = current.resolve(strict=True)
    if base != resolved and base not in resolved.parents:
        raise ValueError("artifact path escapes the configured artifact base")
    return current


def _artifact_dataset(
    parameters: dict[str, Any], artifact_base: Path, seed: int
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    root = _safe_artifact_root(artifact_base, parameters["artifact_path"])
    manifest = verify_bundle(root)
    expected_manifest = parameters["manifest_sha256"]
    actual_manifest = manifest_sha256(root)
    if actual_manifest != expected_manifest:
        raise ValueError("dataset manifest digest differs from benchmark configuration")
    aggregate = hashlib.sha256()
    total = 0
    start = time.perf_counter()
    for _ in range(parameters["iterations"]):
        for entry in manifest["files"]:
            total += stream_verified_file(root, entry, aggregate.update)
    duration = time.perf_counter() - start
    # Detect mutation or a symlink swap during the timed reads.
    verify_bundle(root)
    return (
        duration,
        {
            "iterations": parameters["iterations"],
            "bytes_processed": total,
            "bytes_per_second": total / duration if duration else 0.0,
            "mode": "artifact",
            "files": len(manifest["files"]),
            "seed": seed,
        },
        {
            "aggregate_sha256": aggregate.hexdigest(),
            "dataset_manifest_sha256": actual_manifest,
        },
    )


def _dataset(
    parameters: dict[str, Any], seed: int, artifact_base: Path
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if parameters["mode"] == "synthetic":
        return _synthetic_dataset(parameters, seed)
    return _artifact_dataset(parameters, artifact_base, seed)


def _torch_module() -> Any:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise WorkloadUnavailable(f"PyTorch unavailable: {exc}") from exc
    return torch


def _configure_torch(torch: Any, seed: int) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "deterministic_algorithms": True,
        "tf32_allowed": False,
        "cublas_workspace_config": ":4096:8",
    }


def _torch_device(torch: Any, requested: str) -> str:
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise WorkloadUnavailable("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return "cuda" if available else "cpu"
    return requested


def _torch_dtype(torch: Any, precision: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]


def _synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _tensor_signature(tensor: Any) -> dict[str, Any]:
    flat = tensor.detach().float().cpu().reshape(-1)
    sample = [round(float(value), 8) for value in flat[:16].tolist()]
    digest = hashlib.sha256(canonical_json_bytes(sample)).hexdigest()
    total = float(flat.sum().item())
    if not math.isfinite(total) or not all(math.isfinite(value) for value in sample):
        raise ValueError("non-finite benchmark output")
    return {"sample": sample, "sample_sha256": digest, "sum": total, "finite": True}


def _torch_benchmark(
    parameters: dict[str, Any], seed: int
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    torch = _torch_module()
    determinism = _configure_torch(torch, seed)
    device = _torch_device(torch, parameters["device"])
    dtype = _torch_dtype(torch, parameters["precision"])
    size = parameters["matrix_size"]
    left = torch.randn((size, size), device=device, dtype=dtype)
    right = torch.randn((size, size), device=device, dtype=dtype)
    output = left @ right
    for _ in range(parameters["warmup"]):
        output = left @ right
    _synchronize(torch, device)
    start = time.perf_counter()
    for _ in range(parameters["iterations"]):
        output = left @ right
    _synchronize(torch, device)
    duration = time.perf_counter() - start
    operations = parameters["iterations"] * 2 * size**3
    metrics = {
        "actual_device": device,
        "iterations": parameters["iterations"],
        "operations": operations,
        "operations_per_second": operations / duration if duration else 0.0,
        **determinism,
    }
    return duration, metrics, _tensor_signature(output)


def _inference(
    parameters: dict[str, Any], seed: int
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    torch = _torch_module()
    determinism = _configure_torch(torch, seed)
    device = _torch_device(torch, parameters["device"])
    dtype = _torch_dtype(torch, parameters["precision"])
    batch = parameters["batch_size"]
    inputs = torch.randn((batch, parameters["input_features"]), device=device, dtype=dtype)
    first = torch.randn(
        (parameters["input_features"], parameters["hidden_features"]),
        device=device,
        dtype=dtype,
    )
    second = torch.randn(
        (parameters["hidden_features"], parameters["output_features"]),
        device=device,
        dtype=dtype,
    )

    def infer() -> Any:
        return torch.tanh(torch.relu(inputs @ first) @ second)

    output = infer()
    for _ in range(parameters["warmup"]):
        output = infer()
    _synchronize(torch, device)
    start = time.perf_counter()
    for _ in range(parameters["iterations"]):
        output = infer()
    _synchronize(torch, device)
    duration = time.perf_counter() - start
    samples = batch * parameters["iterations"]
    metrics = {
        "actual_device": device,
        "iterations": parameters["iterations"],
        "samples": samples,
        "samples_per_second": samples / duration if duration else 0.0,
        **determinism,
    }
    return duration, metrics, _tensor_signature(output)


def _simulation(
    parameters: dict[str, Any], seed: int
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    try:
        import mujoco
    except (ImportError, OSError) as exc:
        raise WorkloadUnavailable(f"MuJoCo unavailable: {exc}") from exc
    xml = """
<mujoco model="worker_synthetic_pendulum">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.03" density="100"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.25 + (seed % 1000) / 4000.0
    start = time.perf_counter()
    for _ in range(parameters["steps"]):
        mujoco.mj_step(model, data)
    duration = time.perf_counter() - start
    qpos = float(data.qpos[0])
    qvel = float(data.qvel[0])
    if not math.isfinite(qpos) or not math.isfinite(qvel):
        raise ValueError("non-finite simulation state")
    return (
        duration,
        {
            "steps": parameters["steps"],
            "steps_per_second": parameters["steps"] / duration if duration else 0.0,
            "seed": seed,
        },
        {"finite": True, "final_qpos": qpos, "final_qvel": qvel},
    )


def run_workload(
    workload: dict[str, Any], seed: int, artifact_base: Path
) -> dict[str, Any]:
    name = workload["name"]
    parameters = workload["parameters"]
    if not workload["enabled"]:
        return _entry(name, parameters, status="disabled")
    try:
        if name == "dataset":
            duration, metrics, correctness = _dataset(parameters, seed, artifact_base)
        elif name == "torch":
            duration, metrics, correctness = _torch_benchmark(parameters, seed)
        elif name == "inference":
            duration, metrics, correctness = _inference(parameters, seed)
        else:
            duration, metrics, correctness = _simulation(parameters, seed)
        return _entry(
            name,
            parameters,
            status="passed",
            duration=duration,
            metrics=metrics,
            correctness=correctness,
        )
    except WorkloadUnavailable as exc:
        return _entry(name, parameters, status="unavailable", diagnostic=str(exc)[:2000])
    except Exception as exc:
        diagnostic = f"{type(exc).__name__}: {exc}"[:2000]
        return _entry(name, parameters, status="failed", diagnostic=diagnostic)


def execute(
    config: dict[str, Any],
    compatibility: dict[str, Any],
    *,
    artifact_base: Path,
    expected_source_commit: str,
    expected_container_image: str,
) -> dict[str, Any]:
    config = validate_config(config)
    compatibility = validate_execution_compatibility(
        compatibility,
        expected_source_commit=expected_source_commit,
        expected_container_image=expected_container_image,
    )
    if os.environ.get("THOR_WORKER_QUALIFICATION_ONLY") != "0":
        raise ContractError("benchmark image is not explicitly locked against qualification-only")
    baked_wheelhouse = os.environ.get("THOR_WORKER_WHEELHOUSE_MANIFEST_SHA256")
    recorded_wheelhouse = compatibility["qualification"]["wheelhouse_manifest_sha256"]
    if baked_wheelhouse != recorded_wheelhouse:
        raise ContractError("compatibility wheelhouse digest differs from the executing image")
    started = _utc_now()
    results = [
        run_workload(workload, config["seed"], artifact_base) for workload in config["workloads"]
    ]
    document = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "run_id": str(uuid.uuid4()),
        "started_utc": started,
        "finished_utc": _utc_now(),
        "compatibility": compatibility,
        "config": config,
        "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "seed": config["seed"],
        "results": results,
    }
    return validate_result(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--artifact-base", type=Path)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-container-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = validate_config(read_json(args.config))
        compatibility = validate_compatibility(read_json(args.compatibility))
        artifact_base = args.artifact_base or args.config.absolute().parent
        expected_source = args.expected_source_commit or os.environ.get(
            "THOR_WORKER_SOURCE_COMMIT"
        )
        if expected_source is None:
            raise ContractError("--expected-source-commit is required outside a worker image")
        result = execute(
            config,
            compatibility,
            artifact_base=artifact_base,
            expected_source_commit=expected_source,
            expected_container_image=args.expected_container_image,
        )
        atomic_write_json(args.output, result)
    except (ContractError, OSError, ValueError) as exc:
        print(f"benchmark contract error: {exc}", file=sys.stderr)
        return 2
    bad = [
        entry["name"]
        for entry in result["results"]
        if entry["status"] in {"unavailable", "failed"}
    ]
    if bad:
        print(f"benchmark incomplete: {', '.join(bad)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

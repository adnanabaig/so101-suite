"""Stateful, hardware-isolated SO-101 simulation API (kinematic or MuJoCo engine).

``SO101_SIM_ENGINE=auto|kinematic|mujoco`` picks the engine on first use; see
:mod:`so101.simulation.selection`. Both engines serve the same contract. The
MuJoCo engine adds ``physics_state`` and camera frames at ``GET /render``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

from ...config import get_settings
from ...simulation import (
    ScenarioNotFound,
    SimulationConflict,
    SimulationValidationError,
)
from ...simulation.capabilities import simulation_capabilities
from ...simulation.kinematics import model_description
from ...simulation.mjcf import so101_model
from ...simulation.scenarios import ScenarioError, ScenarioLibrary
from ...simulation.selection import (
    EngineSelection,
    EngineUnavailable,
    SimulationBackend,
    create_engine,
)
from ..schemas import (
    SimulationCapabilitiesResponse,
    SimulationJointTelemetry,
    SimulationPlayRequest,
    SimulationResetRequest,
    SimulationResetResponse,
    SimulationState,
    SimulationStepRequest,
    SimulationTargetRequest,
    SimulationTrajectoryPreviewRequest,
    SimulationTrajectoryPreviewResponse,
)
from ..store import get_store

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

_library = ScenarioLibrary()

# Backward-compatible catalog constant used by config-integrity checks and the
# v0.0 UI. Runtime state, engine labels and active flags come from the engine.
SCENARIOS = tuple(_library.catalog())

_simulator_lock = threading.Lock()
_simulator: SimulationBackend | None = None
_selection: EngineSelection | None = None


def _ensure_simulator() -> tuple[SimulationBackend | None, EngineSelection]:
    global _simulator, _selection
    with _simulator_lock:
        if _selection is None:
            requested = get_settings().sim_engine
            try:
                _simulator, _selection = create_engine(requested, _library)
            except EngineUnavailable as exc:
                _simulator = None
                _selection = EngineSelection(str(requested), None, str(exc), str(exc))
        return _simulator, _selection


def get_simulator() -> SimulationBackend:
    """Return the process-local simulator (never the hardware manager).

    Built on first use from ``SO101_SIM_ENGINE``. Raises
    :class:`~so101.simulation.selection.EngineUnavailable` when the requested
    engine cannot run here.
    """

    simulator, selection = _ensure_simulator()
    if simulator is None:
        raise EngineUnavailable(selection.reason)
    return simulator


def engine_selection() -> EngineSelection:
    """Which engine serves this API and why (builds the engine on first use)."""

    return _ensure_simulator()[1]


def reset_simulator() -> None:
    """Forget the engine so the next request re-reads the settings (tests, reloads)."""

    global _simulator, _selection
    with _simulator_lock:
        _simulator, _selection = None, None


def _sim() -> SimulationBackend:
    try:
        return get_simulator()
    except EngineUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


def _raise_http(exc: Exception) -> HTTPException:
    if isinstance(exc, ScenarioNotFound):
        return HTTPException(404, str(exc))
    if isinstance(exc, SimulationConflict):
        return HTTPException(409, str(exc))
    return HTTPException(422, str(exc))


ASSET_URL_PREFIX = "/api/simulation/assets"
# Mesh URLs carry no version, so a re-vendored mesh keeps its URL. A max-age
# would let a browser draw the old mesh next to the new /model for that long
# without asking; with no-cache it revalidates every time and gets a 304 (no
# body) by ETag while the mesh is unchanged, like /model.
ASSET_CACHE_CONTROL = "no-cache"
MESH_MEDIA_TYPE = "model/stl"


def _etag(payload: bytes) -> str:
    return '"' + hashlib.sha256(payload).hexdigest()[:32] + '"'


def _not_modified(request: Request, etag: str) -> bool:
    header = request.headers.get("if-none-match", "")
    return any(tag.strip() in (etag, "*") for tag in header.split(",") if tag.strip())


@lru_cache(maxsize=1)
def _model_json() -> tuple[bytes, str]:
    payload = json.dumps(model_description(ASSET_URL_PREFIX), separators=(",", ":")).encode()
    return payload, _etag(payload)


@lru_cache(maxsize=1)
def _asset_allowlist() -> dict[str, Path]:
    """Exact file name -> path for the vendored meshes drawn by visual geoms."""

    model = so101_model()
    root = model.meshdir.resolve()
    allowed: dict[str, Path] = {}
    for file_name in model.visual_mesh_files().values():
        path = (root / file_name).resolve()
        if path.parent == root and path.is_file():
            allowed[path.name] = path
    return allowed


@lru_cache(maxsize=64)
def _asset_bytes(file_name: str) -> tuple[bytes, str]:
    payload = _asset_allowlist()[file_name].read_bytes()
    return payload, _etag(payload)


@router.get("/model")
def model(request: Request) -> Response:
    """The SO-101 kinematic tree from the vendored MJCF. Does not need MuJoCo."""

    payload, etag = _model_json()
    headers = {"Cache-Control": "no-cache", "ETag": etag}
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    return Response(payload, media_type="application/json", headers=headers)


@router.get("/assets/{file_name}")
def asset(file_name: str, request: Request) -> Response:
    """A vendored visual mesh, by exact file name from the allowlist."""

    # Exact-name lookup only: no path joining with user input, so "..",
    # separators, case variants and collision meshes all miss.
    if file_name not in _asset_allowlist():
        raise HTTPException(404, "unknown simulation asset")
    payload, etag = _asset_bytes(file_name)
    headers = {
        "Cache-Control": ASSET_CACHE_CONTROL,
        "ETag": etag,
        "X-Content-Type-Options": "nosniff",
    }
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    return Response(payload, media_type=MESH_MEDIA_TYPE, headers=headers)


@router.get("/capabilities", response_model=SimulationCapabilitiesResponse)
def capabilities() -> SimulationCapabilitiesResponse:
    simulator, selection = _ensure_simulator()
    return SimulationCapabilitiesResponse(**simulation_capabilities(selection, engine=simulator))


@router.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    simulator = _sim()
    active_id = simulator.state()["scene"]["scenario_id"]
    return [{**scenario, "active": scenario["id"] == active_id} for scenario in simulator.catalog()]


@router.get("/state", response_model=SimulationState)
def state() -> SimulationState:
    return SimulationState(**_sim().state())


@router.get("/telemetry", response_model=SimulationJointTelemetry)
def telemetry() -> SimulationJointTelemetry:
    return SimulationJointTelemetry(**_sim().telemetry())


@router.post("/targets", response_model=SimulationState)
def targets(req: SimulationTargetRequest) -> SimulationState:
    simulator = _sim()
    try:
        result = simulator.set_targets(req.targets)
    except (SimulationValidationError, SimulationConflict) as exc:
        raise _raise_http(exc) from exc
    return SimulationState(**result)


@router.post("/step", response_model=SimulationState)
def step(req: SimulationStepRequest | None = None) -> SimulationState:
    req = req or SimulationStepRequest()
    simulator = _sim()
    try:
        result = simulator.step(req.steps, targets=req.targets)
    except (SimulationValidationError, SimulationConflict) as exc:
        raise _raise_http(exc) from exc
    return SimulationState(**result)


@router.post("/play", response_model=SimulationState)
def play(req: SimulationPlayRequest | None = None) -> SimulationState:
    req = req or SimulationPlayRequest()
    simulator = _sim()
    try:
        result = simulator.play(req.rate)
    except SimulationValidationError as exc:
        raise _raise_http(exc) from exc
    return SimulationState(**result)


@router.post("/pause", response_model=SimulationState)
def pause() -> SimulationState:
    return SimulationState(**_sim().pause())


@router.post("/reset", response_model=SimulationResetResponse)
def reset(req: SimulationResetRequest | None = None) -> SimulationResetResponse:
    req = req or SimulationResetRequest()
    simulator = _sim()
    try:
        result = simulator.reset(
            req.scenario_id,
            seed=req.seed,
            randomize=req.randomize,
        )
    except (ScenarioError, SimulationValidationError) as exc:
        raise _raise_http(exc) from exc
    generation = int(result["generation"])
    backend = str(result["backend"])
    get_store().append_event(
        "simulation.reset",
        f"reset {backend} scenario {req.scenario_id}",
        source="simulation",
        detail={
            "scenario_id": req.scenario_id,
            "seed": result["scene"]["seed"],
            "randomized": result["scene"]["randomized"],
            "generation": generation,
            "backend": backend,
            "physics": bool(result["physics"]),
        },
    )
    label = str(result["scene"]["label"])
    return SimulationResetResponse(
        ok=True,
        detail=f"{label} reset (generation {generation}, seed {result['scene']['seed']})",
        state=SimulationState(**result),
    )


@router.post("/trajectory/preview", response_model=SimulationTrajectoryPreviewResponse)
def trajectory_preview(
    req: SimulationTrajectoryPreviewRequest,
) -> SimulationTrajectoryPreviewResponse:
    simulator = _sim()
    try:
        result = simulator.preview_trajectory(
            target=req.target,
            duration_s=req.duration_s,
            waypoints=[waypoint.model_dump() for waypoint in req.waypoints],
            sample_hz=req.sample_hz,
        )
    except SimulationValidationError as exc:
        raise _raise_http(exc) from exc
    return SimulationTrajectoryPreviewResponse(**result)


RENDER_FORMATS = ("jpeg", "jpg", "png")


@router.get("/render")
def render(
    camera: str = "front",
    width: int = 320,
    height: int = 240,
    fmt: str = Query("jpeg", alias="format"),
    quality: int = Query(85, ge=10, le=95),
) -> Response:
    """One camera frame of the live MuJoCo scene: JPEG, or PNG without an encoder.

    503 when the kinematic engine is active or no GL context exists here;
    422 for an unknown camera, size outside 16x16..1280x960, or format.
    """

    simulator = _sim()
    if fmt.lower() not in RENDER_FORMATS:
        raise HTTPException(422, f"unsupported image format {fmt!r}; use jpeg or png")
    render_frame = getattr(simulator, "render_frame", None)
    if render_frame is None:
        active = engine_selection().active
        raise HTTPException(
            503,
            f"camera frames need the MuJoCo engine; the active engine is {active} "
            "(set SO101_SIM_ENGINE=mujoco or auto with the sim extra installed)",
        )
    from ...simulation.render import RenderUnavailable, encode_frame

    try:
        frame, meta = render_frame(camera, width, height)
    except SimulationValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RenderUnavailable as exc:
        raise HTTPException(503, f"rendering is unavailable here: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(503, "the render thread did not answer in time") from exc
    payload, media_type = encode_frame(frame, fmt, quality=quality)
    return Response(
        payload,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-SO101-Sim-Camera": str(meta["camera"]),
            "X-SO101-Sim-Step": str(meta["step_count"]),
            "X-SO101-Sim-Generation": str(meta["generation"]),
            "X-SO101-Sim-Time-S": f"{meta['sim_time_s']:.6f}",
        },
    )

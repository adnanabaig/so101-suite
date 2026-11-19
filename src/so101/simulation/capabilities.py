"""Truthful simulation capability report: what is installed, what runs, and why.

Installed is not the same as active. The report separates three facts:

- whether the ``mujoco`` package is installed (found without importing it);
- whether the physics adapter can run, meaning MuJoCo imports and the vendored
  model compiles (probed only when ``SO101_SIM_ENGINE`` is not ``kinematic``);
- which engine actually serves ``/api/simulation``, and why.

With the engine pinned to ``kinematic``, MuJoCo is never imported, which
keeps the API safe on hosts with a broken native install.
"""

from __future__ import annotations

from typing import Any

from .kinematics import MODEL_ID
from .mjcf import MODEL_COMMIT, MODEL_LICENSE, MODEL_REPO, MODEL_REPO_PATH
from .selection import EngineSelection, mujoco_package, mujoco_status, resolve_selection

KINEMATIC_ADAPTER_DETAIL = (
    "Deterministic rate-limited joint-space motion; forward kinematics from the vendored "
    "SO-101 MJCF. No contacts, gravity or motor dynamics."
)
PHYSICS_ADAPTER_DETAIL = (
    "MuJoCo physics of the vendored Menagerie SO-101: gravity, contacts, STS3215 servo "
    "profile, rigid-body scene objects, physically simulated trajectory preview and "
    "offscreen camera renders."
)
KINEMATIC_SIMULATION_DETAIL = (
    "Hardware-isolated 60 Hz fixed-step kinematics, seeded scenes, telemetry, and "
    "trajectory preview. No physics"
)
PHYSICS_SIMULATION_DETAIL = (
    "Hardware-isolated MuJoCo physics (vendored Menagerie SO-101): 60 Hz ticks of "
    "4 x 1/240 s substeps, gravity, contacts, STS3215 servo profile, seeded rigid-body "
    "scenes, physically simulated trajectory preview, and camera renders."
)


def simulation_detail(selection: EngineSelection) -> str:
    """One line for the ``simulation`` entry of ``/api/capabilities``."""

    if selection.active == "mujoco":
        return PHYSICS_SIMULATION_DETAIL
    if selection.active == "kinematic":
        return f"{KINEMATIC_SIMULATION_DETAIL} ({selection.reason})."
    return f"Simulation unavailable: {selection.reason}"


def render_status() -> tuple[bool | None, str | None]:
    """The shared render thread's ``(available, error)``, without waiting for it.

    Starts the thread on first use. ``available`` is ``None`` while its GL
    probe is still running; the UI polls this, so it never blocks on a hung one.
    """

    try:
        from .render import shared_renderer

        return shared_renderer().status()
    except Exception as exc:  # no GL, no MuJoCo: no rendering
        return False, f"{type(exc).__name__}: {exc}"


def simulation_capabilities(
    selection: EngineSelection | None = None,
    *,
    engine: Any = None,
    probe_render: bool = True,
) -> dict[str, Any]:
    """The ``/api/simulation/capabilities`` payload.

    ``selection`` is the router's actual choice. Without one, the choice is
    resolved from the settings exactly as the router would make it. ``engine``,
    when given, supplies the live physics configuration (timestep, substeps,
    servo mode, cameras).
    """

    if selection is None:
        from ..config import get_settings

        selection = resolve_selection(get_settings().sim_engine)
    installed, version = mujoco_package()
    status = mujoco_status() if selection.requested != "kinematic" else None
    physics = selection.active == "mujoco"
    kinematic = selection.active == "kinematic"

    render_available: bool | None = None
    render_error: str | None = None
    render_state: str | None = None  # None: not probed
    if physics and probe_render:
        render_available, render_error = render_status()
        if render_available is None:
            render_state = "probing"
        else:
            render_state = "available" if render_available else f"unavailable: {render_error}"

    if physics:
        detail = PHYSICS_ADAPTER_DETAIL
        if render_available is False:
            detail += f" Camera rendering is unavailable here: {render_error}."
        elif render_state == "probing":
            detail += " Camera rendering is still being probed."
    elif status is None:
        detail = (
            "Not probed: SO101_SIM_ENGINE=kinematic pins the kinematic engine."
            if installed
            else "MuJoCo is optional and not installed; the kinematic engine is active."
        )
    else:
        detail = selection.reason

    mujoco_adapter: dict[str, Any] = {
        "package_installed": installed,
        "package_version": status.version if status and status.version else version,
        "adapter_implemented": True,
        "available": None if status is None else status.usable,
        "active": physics,
        "kind": "physics",
        "engine_setting": selection.requested,
        "selection_reason": selection.reason,
        "error": None if status is None else (status.error or selection.error),
        "model": {
            "id": MODEL_ID,
            "repo": MODEL_REPO,
            "path": MODEL_REPO_PATH,
            "commit": MODEL_COMMIT,
            "license": MODEL_LICENSE,
            "vendored": True,
        },
        "fixed_hz": None,
        "timestep_s": None,
        "substeps_per_tick": None,
        "servo_mode": None,
        "cameras": None,
        "render_available": render_available,
        "render_error": render_error,
        "render_status": render_state,
        "detail": detail,
    }
    if physics and engine is not None:
        mujoco_adapter.update(
            fixed_hz=1.0 / float(engine.fixed_dt_s),
            timestep_s=getattr(engine, "timestep_s", None),
            substeps_per_tick=getattr(engine, "substeps_per_tick", None),
            servo_mode=getattr(engine, "servo_mode", None),
            cameras=list(getattr(engine, "cameras", ()) or ()) or None,
        )

    return {
        "active_backend": selection.active or "none",
        "backend_kind": "physics" if physics else ("kinematic" if kinematic else "none"),
        "physics_active": physics,
        "fixed_step": True,
        "hardware_isolated": True,
        "writes_motor_commands": False,
        "adapters": {
            "kinematic": {
                "available": True,
                "active": kinematic,
                "kind": "kinematic",
                "model": MODEL_ID,
                "detail": KINEMATIC_ADAPTER_DETAIL,
            },
            "mujoco": mujoco_adapter,
        },
    }

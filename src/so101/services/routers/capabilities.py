"""Discoverable workstation feature surface and maturity."""

from __future__ import annotations

import importlib.util

from fastapi import APIRouter

from ... import __version__
from ...agents import planner_status
from ...config import get_settings
from ...hardware.backend import current_backend, sim_devices_problem
from ...simulation.capabilities import KINEMATIC_SIMULATION_DETAIL, simulation_detail
from ...training.configs import POLICY_PRESETS
from ...training.trainer import lerobot_train_available
from ..schemas import CapabilitiesResponse, Capability
from .simulation import engine_selection

router = APIRouter(tags=["capabilities"])

# The kinematic wording. The live entry comes from ``simulation_detail``, which
# reports the engine that actually serves /api/simulation (MuJoCo or kinematic).
SIMULATION_DETAIL = f"{KINEMATIC_SIMULATION_DETAIL}."
CONTROL_DETAIL = "Exclusive lease, explicit arm, deadman, sequence and watchdog enforcement."
QUEST_DETAIL = (
    "Immersive WebXR hand preview is implemented and runtime-probed in the browser; "
    "WebRTC and XR physical actuation are not enabled."
)
AGENT_DETAIL = "Durable bounded proposals; motion requires an armed lease and explicit approval."


def _numpy_presets() -> list[str]:
    """Training presets that run in-process on numpy alone (no torch, no LeRobot)."""
    return [name for name, preset in POLICY_PRESETS.items() if not preset.requires_torch]


def _sim_capabilities(real_training: bool) -> list[Capability]:
    """control, teleop and training under SO101_BACKEND=sim.

    They rest on the simulated bus (MuJoCo), not on LeRobot: the sim backend
    records without LeRobot, and its expert leader stands in for the leader
    arm. Training is available when LeRobot's trainer or a numpy preset runs.
    """
    problem = sim_devices_problem()
    sim_ok = problem is None
    missing = f"the simulated arm cannot load ({problem}); install the sim extra"
    numpy_presets = _numpy_presets()
    return [
        Capability(
            id="control",
            label="Safe manual command API",
            available=sim_ok,
            maturity="stable",
            detail=(
                f"Simulated arm (SO101_BACKEND=sim). {CONTROL_DETAIL}"
                if sim_ok
                else f"Unavailable: {missing}."
            ),
        ),
        Capability(
            id="teleop",
            label="Leader-arm teleoperation",
            available=sim_ok,
            maturity="stable",
            detail=(
                "The scripted expert stands in for the leader arm and drives the simulated "
                "follower, with telemetry and recording."
                if sim_ok
                else f"Unavailable: {missing}."
            ),
        ),
        Capability(
            id="training",
            label="Policy training",
            available=real_training or bool(numpy_presets),
            maturity="beta",
            detail=(
                "LeRobot training (lerobot-train) is installed."
                if real_training
                else (
                    f"In-process numpy training ({', '.join(numpy_presets)}); install the "
                    "training stack for the LeRobot policies."
                    if numpy_presets
                    else "Unavailable; install the training stack on this host."
                )
            ),
        ),
    ]


@router.get("/api/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    settings = get_settings()
    # The same check the trainer makes: LeRobot's training module is importable
    # by this interpreter (PATH does not matter; it runs with python -m).
    real_training = lerobot_train_available()
    # The robot extra is lerobot[dataset,feetech]: real recording refuses to
    # start without LeRobot's dataset support (datasets + av), not just lerobot.
    robot_stack = settings.mock or all(
        importlib.util.find_spec(module) is not None for module in ("lerobot", "datasets", "av")
    )
    agent_planner = planner_status()
    simulation = engine_selection()
    backend = current_backend(settings)
    items = [
        Capability(
            id="control",
            label="Safe manual command API",
            available=robot_stack,
            maturity="stable",
            detail=(
                f"API contract only. {CONTROL_DETAIL}"
                if robot_stack
                else "Control protocol is online; install LeRobot before real arming."
            ),
        ),
        Capability(
            id="teleop",
            label="Leader-arm teleoperation",
            available=robot_stack,
            maturity="stable",
            detail=(
                "Local leader-to-follower control with telemetry and recording."
                if robot_stack
                else "Unavailable; install the robot/LeRobot dependency stack."
            ),
        ),
        Capability(
            id="data",
            label="Dataset collection",
            available=True,
            maturity="beta",
            detail="Episode capture, dataset inspection, replay, and model workflow.",
        ),
        Capability(
            id="training",
            label="Policy training",
            available=settings.mock or real_training,
            maturity="beta",
            detail=(
                "Deterministic mock trainer is active."
                if settings.mock
                else (
                    "LeRobot training (lerobot-train) is installed."
                    if real_training
                    else "Unavailable; install the training stack on this host."
                )
            ),
        ),
        Capability(
            id="simulation",
            label="Simulation scenarios",
            available=simulation.active is not None,
            maturity="preview",
            detail=simulation_detail(simulation),
        ),
        Capability(
            id="tracking",
            label="Camera and hand tracking",
            available=True,
            maturity="preview",
            detail="Confidence-gated wrist/pinch retarget preview with optional leased execution.",
        ),
        Capability(
            id="quest3",
            label="Quest 3 / WebXR",
            available=True,
            maturity="experimental",
            detail=QUEST_DETAIL,
        ),
        Capability(
            id="agent",
            label="Approval-gated skill executor",
            available=True,
            maturity="preview",
            detail=(
                f"Optional {agent_planner['configured_provider']} planner configured. "
                if agent_planner["configured_provider_available"]
                else "Deterministic named-skill planner active. "
            )
            + AGENT_DETAIL,
        ),
        Capability(
            id="diagnostics",
            label="Diagnostics and audit",
            available=True,
            maturity="stable",
            detail="Runtime, storage, configuration, device, and persistent audit checks.",
        ),
    ]
    if backend == "sim":
        replaced = {item.id: item for item in _sim_capabilities(real_training)}
        items = [replaced.get(item.id, item) for item in items]
    return CapabilitiesResponse(
        system_version=__version__,
        mock=settings.mock,
        backend=backend,
        capabilities=items,
    )

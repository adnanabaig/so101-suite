"""Which engine runs ``/api/simulation``: ``SO101_SIM_ENGINE=auto|kinematic|mujoco``.

- ``kinematic``: the deterministic joint-space engine. MuJoCo is never imported.
- ``mujoco``: the physics engine. If MuJoCo cannot run here, the simulation
  API answers 503 with the reason instead of quietly running something else.
- ``auto`` (default): the physics engine when MuJoCo imports and the vendored
  SO-101 model compiles, otherwise the kinematic engine. The reason is kept
  and reported by the capability endpoints.

The probe runs once per process (:func:`mujoco_status` is cached).
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any, Protocol

from .engine import SimulationEngine, SimulationError
from .scenarios import ScenarioLibrary

ENGINE_CHOICES = ("auto", "kinematic", "mujoco")
INSTALL_HINT = "install the sim extra: pip install 'so101[sim]'"


class EngineUnavailable(SimulationError):
    """The requested simulation engine cannot run in this process."""


class SimulationBackend(Protocol):
    """What ``/api/simulation`` needs from an engine. Both engines satisfy it."""

    library: ScenarioLibrary
    MAX_MANUAL_STEPS: int

    @property
    def fixed_dt_s(self) -> float: ...

    def catalog(self) -> list[dict[str, Any]]: ...

    def state(self) -> dict[str, Any]: ...

    def telemetry(self) -> dict[str, Any]: ...

    def set_targets(self, targets: dict[str, float]) -> dict[str, Any]: ...

    def step(
        self, steps: int = 1, *, targets: dict[str, float] | None = None
    ) -> dict[str, Any]: ...

    def play(self, rate: float = 1.0) -> dict[str, Any]: ...

    def pause(self) -> dict[str, Any]: ...

    def reset(
        self, scenario_id: str, *, seed: int | None = None, randomize: bool = True
    ) -> dict[str, Any]: ...

    def preview_trajectory(
        self,
        *,
        target: dict[str, float] | None = None,
        duration_s: float | None = None,
        waypoints: list[dict[str, Any]] | None = None,
        sample_hz: float = 20.0,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MujocoStatus:
    """Can the physics engine run here? ``usable`` means import and compile both work."""

    usable: bool
    package_installed: bool
    version: str | None
    error: str | None


@dataclass(frozen=True)
class EngineSelection:
    """The engine setting, what actually runs, and why."""

    requested: str
    active: str | None  # "kinematic", "mujoco", or None when nothing can run
    reason: str
    error: str | None = None

    @property
    def physics(self) -> bool:
        return self.active == "mujoco"


def mujoco_package() -> tuple[bool, str | None]:
    """Is the ``mujoco`` distribution installed, and which version? Never imports it."""

    try:
        installed = importlib.util.find_spec("mujoco") is not None
    except (ImportError, ValueError):
        # A partially imported or broken package can leave a module with no spec.
        installed = False
    version: str | None = None
    if installed:
        try:
            version = metadata.version("mujoco")
        except metadata.PackageNotFoundError:
            pass
    return installed, version


@lru_cache(maxsize=1)
def mujoco_status() -> MujocoStatus:
    """Import MuJoCo and compile the vendored SO-101 scene once, and remember the outcome."""

    installed, version = mujoco_package()
    if not installed:
        return MujocoStatus(False, False, None, f"MuJoCo is not installed ({INSTALL_HINT})")
    try:
        import mujoco

        from .world import SceneSpec, compiled_scene

        compiled_scene(SceneSpec())
    except Exception as exc:  # any failure means "no physics here"
        return MujocoStatus(False, True, version, f"{type(exc).__name__}: {exc}")
    return MujocoStatus(True, True, str(mujoco.__version__), None)


def _normalized(requested: str | None) -> str:
    value = (requested or "auto").strip().lower() or "auto"
    if value not in ENGINE_CHOICES:
        raise EngineUnavailable(
            f"unknown simulation engine {requested!r}; use one of {', '.join(ENGINE_CHOICES)}"
        )
    return value


def _auto_reason(status: MujocoStatus) -> str:
    return f"auto: MuJoCo {status.version} imports and the SO-101 model compiles"


def _fallback(requested: str, error: str | None) -> EngineSelection:
    if requested == "mujoco":
        reason = f"SO101_SIM_ENGINE=mujoco, but MuJoCo cannot run here: {error}"
        return EngineSelection(requested, None, reason, error)
    reason = f"auto: MuJoCo is unavailable ({error}), so the kinematic engine runs"
    return EngineSelection(requested, "kinematic", reason, error)


def resolve_selection(requested: str | None) -> EngineSelection:
    """What :func:`create_engine` would choose, without building an engine."""

    value = _normalized(requested)
    if value == "kinematic":
        return EngineSelection(value, "kinematic", "SO101_SIM_ENGINE=kinematic")
    status = mujoco_status()
    if status.usable:
        reason = "SO101_SIM_ENGINE=mujoco" if value == "mujoco" else _auto_reason(status)
        return EngineSelection(value, "mujoco", reason)
    return _fallback(value, status.error)


def create_engine(
    requested: str | None,
    library: ScenarioLibrary | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[SimulationBackend, EngineSelection]:
    """Build the engine for ``requested``. Raises :class:`EngineUnavailable` for an
    explicit ``mujoco`` that cannot run; ``auto`` falls back to kinematic instead."""

    selection = resolve_selection(requested)
    if selection.active == "mujoco":
        try:
            from .physics_engine import PhysicsSimulationEngine

            return PhysicsSimulationEngine(library, clock=clock), selection
        except Exception as exc:  # the initial scene must build too
            selection = _fallback(selection.requested, f"{type(exc).__name__}: {exc}")
    if selection.active is None:
        raise EngineUnavailable(selection.reason)
    return SimulationEngine(library, clock=clock), selection


__all__ = [
    "ENGINE_CHOICES",
    "EngineSelection",
    "EngineUnavailable",
    "MujocoStatus",
    "SimulationBackend",
    "create_engine",
    "mujoco_package",
    "mujoco_status",
    "resolve_selection",
]

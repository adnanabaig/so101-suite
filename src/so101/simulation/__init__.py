"""Hardware-isolated SO-101 scene and kinematic simulation.

The simulation package deliberately has no dependency on ``RobotManager`` or
the motor drivers.  It consumes the same joint-limit constants as the safety
layer, but simulation targets can never reach a hardware command path.
"""

from .engine import (
    SimulationConflict,
    SimulationEngine,
    SimulationError,
    SimulationValidationError,
)
from .scenarios import ScenarioLibrary, ScenarioNotFound

__all__ = [
    "ScenarioLibrary",
    "ScenarioNotFound",
    "SimulationConflict",
    "SimulationEngine",
    "SimulationError",
    "SimulationValidationError",
]

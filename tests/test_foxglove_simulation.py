"""The Foxglove demonstration uses model geometry and bounded simulated motion."""

import importlib.util
import math
from pathlib import Path

import pytest

from so101 import JOINT_NAMES
from so101.simulation.kinematics import JOINT_LIMITS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sim_worker", ROOT / "scripts/foxglove_simulation.py")
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_demo_targets_stay_inside_simulator_ranges():
    for tick in range(1200):
        positions = worker.targets(tick / 10)
        assert set(positions) == set(JOINT_NAMES)
        assert all(
            JOINT_LIMITS[name][0] <= value <= JOINT_LIMITS[name][1]
            for name, value in positions.items()
        )


def test_simulated_scene_moves_without_connecting_real_motors(monkeypatch):
    from so101.hardware.motors import FeetechBus

    monkeypatch.setattr(FeetechBus, "connect", lambda *args: pytest.fail("physical bus opened"))
    model = worker.so101_model()
    engine = worker.PhysicsSimulationEngine()
    parts = worker.model_parts(model)
    first = worker.snapshot(engine, model, parts, 0)
    current = first
    for tick in range(1, 20):
        current = worker.snapshot(engine, model, parts, tick / 20)
    assert first["physical_arm_connected"] is False
    assert current["source"] == "mujoco_simulation"
    assert len(current["models"]) == 18
    assert any(
        abs(current["measured"][name] - first["measured"][name]) > 0.1 for name in JOINT_NAMES
    )
    for item in current["models"]:
        assert Path(item["mesh"]).is_file()
        assert all(math.isfinite(value) for value in item["position"])
        assert sum(value * value for value in item["quaternion_wxyz"]) == pytest.approx(1)

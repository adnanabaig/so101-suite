"""Versioned operator configs must stay aligned with live API contracts."""

from __future__ import annotations

from pathlib import Path

import yaml

from so101.services.routers.agent import _targets_for_skill
from so101.services.routers.simulation import SCENARIOS
from so101.services.routers.tracking import MIN_TRACKING_CONFIDENCE

ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text())


def test_simulation_scenario_config_matches_api_registry():
    configured = {item["id"] for item in _yaml("configs/simulation/scenarios.yaml")["scenarios"]}
    exposed = {item["id"] for item in SCENARIOS}
    assert configured == exposed


def test_agent_skill_config_matches_bounded_backend():
    skills = _yaml("configs/agent/skills.yaml")["skills"]
    for name, specification in skills.items():
        targets = _targets_for_skill(name, bool(specification["mutates_robot"]))
        assert bool(targets) is bool(specification["mutates_robot"])


def test_tracking_confidence_config_matches_gate():
    configured = _yaml("configs/tracking/retargeting.yaml")["confidence_threshold"]
    assert configured == MIN_TRACKING_CONFIDENCE

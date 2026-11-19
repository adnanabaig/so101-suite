"""Halt and Collapse settings (design doc §5): finite, positive, a bounded descent."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from so101.config import Settings


def test_stop_settings_defaults():
    defaults = {name: field.default for name, field in Settings.model_fields.items()}
    assert defaults["halt_hold_seconds"] == 60.0
    assert defaults["supervision_grace_seconds"] == 5.0
    assert defaults["collapse_speed_deg_s"] == 15.0
    assert defaults["collapse_timeout_seconds"] is None
    assert defaults["grip_hold_margin"] == 5.0


def test_stop_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("SO101_HALT_HOLD_SECONDS", "30")
    monkeypatch.setenv("SO101_COLLAPSE_TIMEOUT_SECONDS", "8.5")
    settings = Settings()
    assert settings.halt_hold_seconds == 30.0
    assert settings.collapse_timeout_seconds == 8.5


@pytest.mark.parametrize(
    "name",
    ["halt_hold_seconds", "supervision_grace_seconds", "collapse_speed_deg_s", "grip_hold_margin"],
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_stop_settings_must_be_finite_and_positive(name: str, value: float):
    with pytest.raises(ValidationError, match="finite and positive"):
        Settings(**{name: value})


@pytest.mark.parametrize("value", ["", " ", "auto", "none"])
def test_a_blank_collapse_timeout_is_derived(monkeypatch, value: str):
    monkeypatch.setenv("SO101_COLLAPSE_TIMEOUT_SECONDS", value)
    assert Settings().collapse_timeout_seconds is None


@pytest.mark.parametrize("value", [0.0, -2.0, math.inf, 12.5])
def test_a_collapse_timeout_is_finite_positive_and_at_most_12_s(value: float):
    with pytest.raises(ValidationError):
        Settings(collapse_timeout_seconds=value)
    assert Settings(collapse_timeout_seconds=12.0).collapse_timeout_seconds == 12.0

"""Simulation settings: the backend rule and the validated ``sim_*`` fields."""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from so101.config import (
    SIM_CAMERAS,
    SIM_IMAGE_MAX,
    SIM_IMAGE_MIN,
    Settings,
    get_settings,
    parse_image_size,
)


@pytest.fixture
def env(monkeypatch):
    """Settings from a clean environment plus the given ``SO101_*`` variables."""

    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.delenv("SO101_BACKEND", raising=False)

    def settings(**values: str) -> Settings:
        for key, value in values.items():
            monkeypatch.setenv(f"SO101_{key.upper()}", value)
        get_settings.cache_clear()
        return get_settings()

    return settings


def test_unset_backend_follows_mock(env):
    real = env()
    assert (real.backend, real.mock) == ("real", False)
    mock = env(mock="1")
    assert (mock.backend, mock.mock) == ("mock", True)


@pytest.mark.parametrize(
    ("variables", "backend", "mock"),
    [
        ({"backend": "mock"}, "mock", True),
        ({"backend": "sim"}, "sim", False),
        ({"backend": "real"}, "real", False),
        ({"backend": "mock", "mock": "1"}, "mock", True),
        ({"backend": "sim", "mock": "0"}, "sim", False),
    ],
)
def test_explicit_backend(env, variables, backend, mock):
    settings = env(**variables)
    assert (settings.backend, settings.mock) == (backend, mock)


@pytest.mark.parametrize("backend", ["real", "sim"])
def test_mock_conflicts_with_a_real_or_sim_backend(env, backend):
    with pytest.raises(
        ValidationError, match=f"SO101_MOCK=1 conflicts with SO101_BACKEND={backend}"
    ):
        env(mock="1", backend=backend)


def test_backend_and_engine_are_normalized(env):
    assert env(backend=" SIM ").backend == "sim"
    assert env(backend="").backend == "real"  # blank is unset
    assert env(backend="", mock="1").backend == "mock"
    assert env(sim_engine=" MuJoCo ").sim_engine == "mujoco"
    assert env(sim_engine="").sim_engine == "auto"
    with pytest.raises(ValidationError):
        env(backend="robot")
    with pytest.raises(ValidationError):
        env(sim_engine="bullet")


def test_sim_defaults(env):
    settings = env()
    assert settings.sim_engine == "auto"
    assert settings.sim_task == "pick_lift"
    assert settings.sim_seed == 0
    assert settings.sim_camera_names == ["front", "wrist_cam"]
    assert settings.sim_image_wh == (320, 240)
    assert settings.sim_realtime is True


def test_sim_fields_from_the_environment(env):
    settings = env(
        sim_task="pick_place",
        sim_seed="42",
        sim_cameras=" wrist_cam, front ,wrist_cam",
        sim_image_size=" 640X480 ",
        sim_realtime="false",
    )
    assert settings.sim_task == "pick_place"
    assert settings.sim_seed == 42
    assert settings.sim_camera_names == ["wrist_cam", "front"]  # order kept, deduplicated
    assert settings.sim_image_size == "640x480"
    assert settings.sim_image_wh == (640, 480)
    assert settings.sim_realtime is False
    assert env(sim_cameras="").sim_camera_names == []


@pytest.mark.parametrize(
    "variables",
    [
        {"sim_task": "stack"},
        {"sim_cameras": "front,side"},
        {"sim_image_size": "320"},
        {"sim_image_size": "15x240"},
        {"sim_image_size": "1281x960"},
        {"sim_seed": "zero"},
        {"sim_seed": "-1"},
    ],
)
def test_invalid_sim_fields_are_rejected(env, variables):
    with pytest.raises(ValidationError):
        env(**variables)


def test_a_negative_sim_seed_fails_at_startup_not_at_first_use(env):
    # numpy's generators refuse negative seeds, so -1 used to pass here and
    # crash the first sim episode with a message that never named the setting.
    with pytest.raises(ValidationError, match="sim_seed must be at least 0"):
        env(sim_seed="-1")
    assert env(sim_seed="0").sim_seed == 0


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("320x240", (320, 240)),
        ("64X48", (64, 48)),
        (" 16x16 ", (16, 16)),
        ("1280x960", (1280, 960)),
    ],
)
def test_parse_image_size(text, size):
    assert parse_image_size(text) == size


@pytest.mark.parametrize("text", ["", "320", "320x", "x240", "axb", "64x48x3", "15x16", "1280x961"])
def test_parse_image_size_rejects(text):
    with pytest.raises(ValueError, match="image size"):
        parse_image_size(text)


def test_get_settings_is_still_cached(env):
    first = env()
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings() is not first


def test_settings_constants_mirror_the_simulator():
    pytest.importorskip("mujoco")
    from so101.simulation import render, tasks, world

    assert SIM_CAMERAS == world.CAMERAS
    assert SIM_IMAGE_MIN == render.MIN_SIZE
    assert SIM_IMAGE_MAX == world.OFFSCREEN_SIZE
    sim_task = Settings.model_fields["sim_task"].annotation
    assert typing.get_args(sim_task) == tasks.TASK_NAMES

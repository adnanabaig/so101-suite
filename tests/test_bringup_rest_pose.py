"""Rest pose: parsing, storage, resolution order, clamping and measurement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from so101 import JOINT_NAMES
from so101.config import Settings, get_settings
from so101.hardware import lerobot_paths
from so101.hardware import rest_pose as rp
from so101.hardware.lerobot_paths import lerobot_calibration_file
from so101.simulation.rest_pose import MODEL_REST_POSE

POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -95.0,
    "elbow_flex": 90.0,
    "wrist_flex": 70.0,
    "wrist_roll": 0.0,
    "gripper": 2.0,
}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("SO101_DATA_DIR", str(tmp_path / "data"))
    # Never touch the real LeRobot calibration directory, whether or not LeRobot
    # (whose root is fixed at import time) is installed.
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def _write_lerobot_calibration(settings: Settings, content: str = '{"x": 1}') -> str:
    path = lerobot_calibration_file("follower", settings.follower_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.mark.parametrize(
    "text",
    [
        "0,-95,90,70,0,2",
        "0 -95 90 70 0 2",
        "[0, -95, 90, 70, 0, 2]",
        json.dumps(POSE),
    ],
)
def test_setting_accepts_ordered_numbers_or_an_object(text: str) -> None:
    assert rp.parse_rest_pose_setting(text) == POSE


@pytest.mark.parametrize(
    "text",
    [
        "1,2,3",
        "0,-95,90,70,0,2,9",
        "0,-95,90,70,0,nan",
        "0,-95,90,70,0,abc",
        '{"shoulder_pan": 0}',
        json.dumps({**POSE, "tail": 1.0}),
        json.dumps({**POSE, "gripper": True}),
        "[0, -95",
    ],
)
def test_malformed_settings_are_rejected(text: str) -> None:
    with pytest.raises(rp.RestPoseError):
        rp.parse_rest_pose_setting(text)


def test_settings_validate_rest_pose() -> None:
    assert rp.parse_rest_pose_setting("") is None
    assert Settings(rest_pose=" 0,-95,90,70,0,2 ").rest_pose == "0,-95,90,70,0,2"
    with pytest.raises(ValidationError):
        Settings(rest_pose="1,2,3")


def test_save_writes_the_contract_schema_atomically(settings: Settings) -> None:
    record = rp.save_rest_pose(
        settings,
        POSE,
        follower_id=settings.follower_id,
        calibration_digest="abc",
        backend="real",
        spread=dict.fromkeys(JOINT_NAMES, 0.2),
        samples=10,
    )
    path = rp.rest_pose_path(settings)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema", "joints", "follower_id", "calibration_digest", "recorded_at"):
        assert key in payload
    assert payload["schema"] == 1 and payload["so101_version"]
    assert payload["joints"] == POSE
    assert [p.name for p in path.parent.iterdir()] == ["rest_pose.json"]  # no temp files left
    assert rp.load_recorded_rest_pose(settings) == record
    assert rp.clear_rest_pose(settings) is True
    assert rp.clear_rest_pose(settings) is False
    assert rp.load_recorded_rest_pose(settings) is None


def test_a_matching_recorded_pose_wins(settings: Settings) -> None:
    digest = _write_lerobot_calibration(settings)
    assert rp.follower_calibration_digest(settings) == digest
    rp.save_rest_pose(
        settings, POSE, follower_id=settings.follower_id, calibration_digest=digest, backend="real"
    )
    resolution = rp.resolve_for_settings(settings)
    assert resolution.source == "recorded"
    assert resolution.joints == POSE
    assert resolution.warnings == []


@pytest.mark.parametrize(
    ("follower_id", "digest", "backend", "fragment"),
    [
        ("someone-else", "DIGEST", "real", "recorded for follower"),
        (None, "stale", "real", "calibration changed"),
        (None, "DIGEST", "mock", "mock backend"),
    ],
)
def test_a_mismatched_recorded_pose_is_ignored_with_a_warning(
    settings: Settings, follower_id: str | None, digest: str, backend: str, fragment: str
) -> None:
    current = _write_lerobot_calibration(settings)
    rp.save_rest_pose(
        settings,
        POSE,
        follower_id=follower_id or settings.follower_id,
        calibration_digest=current if digest == "DIGEST" else digest,
        backend=backend,  # type: ignore[arg-type]
    )
    resolution = rp.resolve_for_settings(settings)
    assert resolution.source == "default"
    assert resolution.placeholder is False
    assert resolution.joints == MODEL_REST_POSE
    assert any(fragment in warning for warning in resolution.warnings)


def test_real_backend_needs_a_calibration_file_to_trust_a_recording(settings: Settings) -> None:
    rp.save_rest_pose(
        settings, POSE, follower_id=settings.follower_id, calibration_digest=None, backend="real"
    )
    resolution = rp.resolve_for_settings(settings)
    assert resolution.source == "default"
    assert any("no LeRobot calibration file" in w for w in resolution.warnings)


def test_mock_recordings_resolve_in_mock_mode(settings: Settings) -> None:
    rp.save_rest_pose(
        settings, POSE, follower_id=settings.follower_id, calibration_digest=None, backend="mock"
    )
    resolution = rp.resolve_rest_pose(
        settings, follower_id=settings.follower_id, calibration_digest=None, backend="mock"
    )
    assert resolution.source == "recorded"


def test_setting_is_used_when_nothing_is_recorded(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    rp.rest_pose_path(settings).parent.mkdir(parents=True, exist_ok=True)
    rp.rest_pose_path(settings).write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("SO101_REST_POSE", "0,-95,90,70,0,2")
    get_settings.cache_clear()
    resolution = rp.resolve_for_settings(get_settings())
    assert resolution.source == "setting"
    assert resolution.joints == POSE
    assert any("not valid JSON" in w for w in resolution.warnings)


def test_every_source_is_clamped_to_safety_limits(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_REST_POSE", "0,-150,90,70,0,120")
    get_settings.cache_clear()
    resolution = rp.resolve_for_settings(get_settings())
    assert resolution.joints["shoulder_lift"] == -101.0
    assert resolution.joints["gripper"] == 100.0
    assert resolution.clamped["shoulder_lift"] == (-150.0, -101.0)
    assert sum("clamped" in w for w in resolution.warnings) == 2
    tight = rp.resolve_for_settings(
        get_settings(), limits={"elbow_flex": (-10.0, 10.0), "wrist_flex": (0.0, 5.0)}
    )
    assert tight.joints["elbow_flex"] == 10.0 and tight.joints["wrist_flex"] == 5.0


def test_compact_recorded_rest_fits_but_commands_still_stop_at_the_boundaries(
    settings: Settings,
) -> None:
    from so101.hardware.safety import DEFAULT_LIMITS, SafetyGuard

    digest = _write_lerobot_calibration(settings)
    compact = {**POSE, "shoulder_lift": -98.98901098901099, "elbow_flex": 97.89010989010988}
    rp.save_rest_pose(
        settings,
        compact,
        follower_id=settings.follower_id,
        calibration_digest=digest,
        backend="real",
        spread={joint: 0.0 for joint in JOINT_NAMES},
    )
    resolution = rp.resolve_collapse_rest_pose(settings, limits=DEFAULT_LIMITS, backend="real")
    assert resolution.source == "recorded"
    assert resolution.joints == compact
    assert resolution.clamped == {}
    assert not resolution.warnings
    guard = SafetyGuard()
    bounded = guard.clamp({"shoulder_lift": -105.0, "elbow_flex": 105.0})
    assert bounded == {"shoulder_lift": -101.0, "elbow_flex": 99.0}


def test_default_is_the_model_derived_pose_collapse_uses(settings: Settings) -> None:
    """Every bring-up view resolves exactly as Collapse does (design doc §5)."""
    resolution = rp.resolve_for_settings(settings)
    assert resolution.source == "default" and resolution.placeholder is False
    assert resolution.joints == MODEL_REST_POSE
    assert resolution == rp.resolve_collapse_rest_pose(settings)
    assert not hasattr(rp, "DEFAULT_REST_POSE")  # the mid-range placeholder is gone
    assert not any("placeholder" in w for w in resolution.warnings)
    assert any("so101 rest-pose record" in w for w in resolution.warnings)


def test_bring_up_and_collapse_share_the_clamp(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_REST_POSE", "0,-95,105,-85,0,5")  # elbow past the model's range
    get_settings.cache_clear()
    model = rp.model_joint_limits()
    shown = rp.resolve_for_settings(get_settings())
    assert shown.joints["elbow_flex"] == model["elbow_flex"][1]
    assert any("SafetyGuard ∩ MJCF limits" in w for w in shown.warnings)
    assert shown == rp.resolve_collapse_rest_pose(get_settings())


def test_measure_requires_a_settled_arm() -> None:
    readings = iter([{**POSE, "elbow_flex": 90.0 + 0.1 * (i % 3)} for i in range(10)])
    sleeps: list[float] = []
    mean, spread = rp.measure_rest_pose(lambda: next(readings), sleep=sleeps.append)
    assert len(sleeps) == 9
    assert spread["elbow_flex"] == pytest.approx(0.2)
    assert mean["shoulder_lift"] == -95.0

    moving = iter([{**POSE, "shoulder_lift": -95.0 + i * 0.2} for i in range(10)])
    with pytest.raises(rp.RestPoseUnstableError) as excinfo:
        rp.measure_rest_pose(lambda: next(moving), sleep=lambda _: None)
    assert excinfo.value.spread["shoulder_lift"] == pytest.approx(1.8)

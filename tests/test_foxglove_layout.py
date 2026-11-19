"""Generated desktop layouts distinguish idle observations from control ticks."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.mark.parametrize("dashboard", [True, False])
def test_shipped_layout_uses_the_correct_measured_source(dashboard):
    filename = (
        "foxglove-operations-layout.json" if dashboard else "foxglove-episode-review-layout.json"
    )
    layout = json.loads((ROOT / "config/visualization" / filename).read_text())
    panels = layout["configById"]
    plots = [config for panel, config in panels.items() if panel.startswith("Plot!")]
    joint_plots = [plot for plot in plots if not plot["title"].startswith("Control timing")]
    assert len(joint_plots) == len(JOINTS)
    for joint, plot in zip(JOINTS, joint_plots, strict=True):
        paths = {path["label"]: path["value"] for path in plot["paths"]}
        expected = (
            f"/robot/encoder/joints.measured.{joint}"
            if dashboard
            else f"/robot/control_tick.joints.{joint}.measured"
        )
        assert paths["Follower measured"] == expected
        assert paths["Leader intent"] == f"/robot/control_tick.joints.{joint}.desired"
        assert paths["Accepted command"] == f"/robot/control_tick.joints.{joint}.commanded"
    if dashboard:
        assert any(panel.get("topicPath") == "/robot/encoder/status" for panel in panels.values())
    timing = next(plot for plot in plots if plot["title"].startswith("Control timing"))
    timing_paths = {path["label"]: path["value"] for path in timing["paths"]}
    assert timing_paths["Motor/control step"] == "/robot/control_tick.timing.motor_step_seconds"
    assert timing_paths["Bus lock wait"] == "/robot/control_tick.timing.io_lock_wait_seconds"
    assert timing_paths["Previous pacing deadline lateness"] == (
        "/robot/control_tick.timing.previous_pacing.deadline_lateness_seconds"
    )
    assert "seconds" in timing["title"]
    for topic in ("/recording/episode", "/recording/metadata"):
        assert any(panel.get("topicPath") == topic for panel in panels.values())

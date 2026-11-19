"""Build a reproducible desktop layout with the Foxglove SDK's typed panel settings."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def desktop_layout(typed):
    configs = {}

    def convert(node):
        kind = node["type"]
        if kind == "panel":
            panel_id = f"{node['panelType']}!so101_{len(configs) + 1}"
            config = dict(node.get("config", {}))
            if node.get("title"):
                config["title"] = node["title"]
            configs[panel_id] = config
            return panel_id
        if kind == "tabs":
            panel_id = f"Tab!so101_{len(configs) + 1}"
            configs[panel_id] = {}
            configs[panel_id] = {
                "activeTabIdx": node.get("selectedTabIndex", 0),
                "tabs": [
                    {"title": tab["title"], "layout": convert(tab["content"])}
                    for tab in node["tabs"]
                ],
            }
            return panel_id
        if kind == "split":

            def split(items):
                if len(items) == 1:
                    return convert(items[0]["content"])
                return {
                    "direction": node["direction"],
                    "first": convert(items[0]["content"]),
                    "second": split(items[1:]),
                    "splitPercentage": 100
                    * items[0].get("proportion", 1)
                    / sum(item.get("proportion", 1) for item in items),
                }

            return split(node["items"])
        raise ValueError(f"Unsupported layout node: {kind}")

    tree = convert(typed["content"])
    return {
        "configById": configs,
        "globalVariables": {},
        "userNodes": {},
        "linkedGlobalVariables": [],
        "playbackConfig": {"speed": 1},
        "layout": tree,
    }


def build():
    from foxglove.layouts import (
        BaseRendererSceneUpdateTopicSettings,
        ImageConfig,
        ImageModeConfig,
        ImagePanel,
        Layout,
        PlotConfig,
        PlotPanel,
        PlotSeries,
        RawMessagesConfig,
        RawMessagesPanel,
        SplitContainer,
        SplitItem,
        TabContainer,
        TabItem,
        ThreeDeeCameraState,
        ThreeDeeConfig,
        ThreeDeePanel,
    )

    def raw(topic, title):
        return RawMessagesPanel(
            title=title, config=RawMessagesConfig(topic_path=topic, default_expanded=True)
        )

    cameras = SplitContainer(
        direction="column",
        items=[
            SplitItem(
                content=ImagePanel(
                    title=title,
                    config=ImageConfig(
                        image_mode=ImageModeConfig(image_topic=f"/camera/{name}/image")
                    ),
                )
            )
            for name, title in (
                ("obsbot", "LIVE — OBSBOT Tiny 2 Lite"),
                ("camera_1", "LIVE — icSpring camera 1"),
                ("camera_2", "LIVE — icSpring camera 2"),
                ("realsense", "LIVE — D435i color"),
            )
        ],
    )
    simulation_scene = ThreeDeePanel(
        title="SIMULATION — MuJoCo SO101",
        config=ThreeDeeConfig(
            fixed_frame="sim_base",
            follow_tf="sim_base",
            camera_state=ThreeDeeCameraState(
                distance=0.85, target_offset=(0, 0, 0.18), phi=65, theta_offset=35, perspective=True
            ),
            topics={"/simulation/scene": BaseRendererSceneUpdateTopicSettings(visible=True)},
        ),
    )
    encoder_scene = ThreeDeePanel(
        title="ENCODERS — physical/model alignment needs review",
        config=ThreeDeeConfig(
            fixed_frame="robot_base",
            follow_tf="robot_base",
            camera_state=ThreeDeeCameraState(
                distance=0.85, target_offset=(0, 0, 0.18), phi=65, theta_offset=35, perspective=True
            ),
            topics={"/robot/encoder/scene": BaseRendererSceneUpdateTopicSettings(visible=True)},
        ),
    )
    scene = TabContainer(
        tabs=[
            TabItem(title="Motor encoders", content=encoder_scene),
            TabItem(title="Independent simulation", content=simulation_scene),
        ]
    )
    plots = PlotPanel(
        title="SIMULATED joint positions — degrees / gripper %",
        config=PlotConfig(
            paths=[
                PlotSeries(value=f"/simulation/joints.measured.{name}", label=name, enabled=True)
                for name in (
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                )
            ],
            following_view_width=20,
            show_legend=True,
        ),
    )
    evidence = TabContainer(
        tabs=[
            TabItem(
                title="Computer vision",
                content=ImagePanel(
                    title="OBSBOT motor tags — green; projection uncalibrated",
                    config=ImageConfig(
                        image_mode=ImageModeConfig(image_topic="/camera/obsbot_cv/image")
                    ),
                ),
            )
        ]
        + [
            TabItem(title=title, content=raw(topic, title))
            for topic, title in (
                ("/robot/encoder/status", "Encoder freshness"),
                ("/recording/episodes", "MCAP episodes"),
                ("/simulation/status", "Simulation status"),
                ("/camera/obsbot/status", "OBSBOT status"),
                ("/camera/camera_1/status", "Camera 1 status"),
                ("/camera/camera_2/status", "Camera 2 status"),
                ("/calibration/installed", "Saved calibration"),
                ("/calibration/rest_pose", "Saved folded pose"),
                ("/calibration/shoulder_reference", "Saved reference attempt"),
            )
        ]
    )
    layout = Layout(
        content=SplitContainer(
            direction="column",
            items=[
                SplitItem(
                    proportion=2,
                    content=SplitContainer(
                        direction="row",
                        items=[
                            SplitItem(proportion=2, content=scene),
                            SplitItem(proportion=1, content=cameras),
                        ],
                    ),
                ),
                SplitItem(
                    proportion=1,
                    content=SplitContainer(
                        direction="row",
                        items=[SplitItem(content=plots), SplitItem(content=evidence)],
                    ),
                ),
            ],
        )
    )
    typed = json.loads(layout.to_json())
    (ROOT / "config/visualization/foxglove-arms-layout-source.json").write_text(
        json.dumps(typed, indent=2) + "\n"
    )
    path = ROOT / "config/visualization/foxglove-arms-layout.json"
    path.write_text(json.dumps(desktop_layout(typed), indent=2) + "\n")
    print(path)


def build_episode_review(*, dashboard=False):
    from foxglove.layouts import (
        BaseRendererSceneUpdateTopicSettings,
        ImageConfig,
        ImageModeConfig,
        ImagePanel,
        Layout,
        PlotConfig,
        PlotPanel,
        PlotSeries,
        RawMessagesConfig,
        RawMessagesPanel,
        SplitContainer,
        SplitItem,
        TabContainer,
        TabItem,
        ThreeDeeCameraState,
        ThreeDeeConfig,
        ThreeDeePanel,
    )

    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    cameras = TabContainer(
        tabs=[
            TabItem(
                title=name,
                content=ImagePanel(
                    title=f"{'Live' if dashboard else 'Recorded'} camera — {name}",
                    config=ImageConfig(
                        image_mode=ImageModeConfig(image_topic=f"/camera/{name}/image")
                    ),
                ),
            )
            for name in ("obsbot", "camera_1", "camera_2", "realsense")
        ]
    )
    robot = ThreeDeePanel(
        title=(
            "Live encoder pose — physical alignment must be qualified"
            if dashboard
            else "Recorded joint positions — physical alignment must be qualified"
        ),
        config=ThreeDeeConfig(
            fixed_frame="robot_base",
            follow_tf="robot_base",
            camera_state=ThreeDeeCameraState(
                distance=0.85,
                target_offset=(0, 0, 0.18),
                phi=65,
                theta_offset=35,
                perspective=True,
            ),
            topics={"/robot/encoder/scene": BaseRendererSceneUpdateTopicSettings(visible=True)},
        ),
    )
    plots = TabContainer(
        tabs=[
            TabItem(
                title=joint,
                content=PlotPanel(
                    title=f"{joint} — {'percent' if joint == 'gripper' else 'degrees'}",
                    config=PlotConfig(
                        paths=[
                            PlotSeries(
                                value=(
                                    f"/robot/encoder/joints.measured.{joint}"
                                    if dashboard and field == "measured"
                                    else f"/robot/control_tick.joints.{joint}.{field}"
                                ),
                                label=label,
                                enabled=True,
                            )
                            for field, label in (
                                ("desired", "Leader intent"),
                                ("commanded", "Accepted command"),
                                ("measured", "Follower measured"),
                            )
                        ],
                        show_legend=True,
                        following_view_width=20,
                    ),
                ),
            )
            for joint in joints
        ]
        + [
            TabItem(
                title="Timing",
                content=PlotPanel(
                    title="Control timing — seconds; available in instrumented captures",
                    config=PlotConfig(
                        paths=[
                            PlotSeries(
                                value=f"/robot/control_tick.timing.{field}",
                                label=label,
                                enabled=True,
                            )
                            for field, label in (
                                ("cycle_interval_seconds", "Cycle start interval"),
                                ("io_lock_wait_seconds", "Bus lock wait"),
                                ("motor_step_seconds", "Motor/control step"),
                                ("previous_tail_seconds", "Previous telemetry/recording tail"),
                                (
                                    "previous_pacing.deadline_lateness_seconds",
                                    "Previous pacing deadline lateness",
                                ),
                            )
                        ],
                        show_legend=True,
                        following_view_width=20,
                    ),
                ),
            )
        ]
    )
    evidence = TabContainer(
        tabs=[
            TabItem(
                title=title,
                content=RawMessagesPanel(
                    title=title,
                    config=RawMessagesConfig(topic_path=topic, default_expanded=True),
                ),
            )
            for topic, title in (
                (
                    ("/robot/status", "Robot status"),
                    ("/robot/encoder/status", "Encoder freshness"),
                    ("/control/state", "Stops and ownership"),
                    ("/recording/status", "Recorder and episodes"),
                    ("/diagnostics/preflight", "Preflight report"),
                    ("/robot/control_capture", "Capture continuity"),
                )
                if dashboard
                else ()
            )
            + (
                ("/recording/review", "Episode review"),
                ("/recording/episode", "Matching episode"),
                ("/recording/metadata", "Capture provenance"),
                ("/recording/episodes", "Episode boundaries"),
                ("/robot/health", "Motor health"),
                ("/robot/control_tick", "Control samples"),
                ("/camera/obsbot/status", "Camera timing"),
                ("/calibration/installed", "Follower calibration"),
                ("/calibration/installed_leader", "Leader calibration"),
            )
        ]
    )
    layout = Layout(
        content=SplitContainer(
            direction="column",
            items=[
                SplitItem(
                    proportion=2,
                    content=SplitContainer(
                        direction="row",
                        items=[
                            SplitItem(content=robot),
                            SplitItem(content=cameras),
                        ],
                    ),
                ),
                SplitItem(
                    content=SplitContainer(
                        direction="row",
                        items=[
                            SplitItem(content=plots),
                            SplitItem(content=evidence),
                        ],
                    )
                ),
            ],
        )
    )
    typed = json.loads(layout.to_json())
    filename = (
        "foxglove-operations-layout.json" if dashboard else "foxglove-episode-review-layout.json"
    )
    path = ROOT / "config/visualization" / filename
    path.write_text(json.dumps(desktop_layout(typed), indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    build()
    build_episode_review()
    build_episode_review(dashboard=True)

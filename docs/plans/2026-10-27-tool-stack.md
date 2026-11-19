# Tool stack and integration direction

Current operator direction: the Suite UI is the primary workspace for live
inspection and recorded playback. Foxglove work is deferred; its existing
adapter/layouts remain optional. The table below records the earlier plan,
not a requirement to keep expanding the external viewer.

## Goal

Make SO101 Suite the main workstation for operating the arms, collecting
demonstrations and reviewing experiments. Integrate specialist tools through
shared interfaces, data and robot models. Keep the current teleoperation and
ten-episode qualification goal intact while establishing this architecture.

The user's spoken references to “CI” and “false” are interpreted here as the
Suite UI and Foxglove. Actual continuous integration has a separate testing role.

## Tool responsibilities

| Component | Responsibility |
| --- | --- |
| Suite UI | Daily collection dashboard, configuration, calibration workflow, operator controls and readiness |
| Python backend / RobotManager | Sole motor-bus ownership, validated commands, calibration access, stops, latches and recorder coordination |
| MuJoCo | Primary physics simulator for proposed actions, contact experiments and repeatable simulated evaluation |
| Three.js STL viewer | Render measured encoder poses and proposed simulation poses with explicit source and freshness labels |
| Foxglove | Detailed live inspection, camera/joint/health plots and MCAP playback; optional Suite operator panel reuses the existing backend |
| MCAP on SigmaDrive | Raw activity recordings with camera frames, control samples, status, timestamps and provenance |
| LeRobot | Hardware/dataset integration and later imitation-learning experiments using selected, validated demonstrations |
| OBS / camera SDKs | Camera setup and supporting observation; shared relays own capture for the collection workflow |
| Automated CI | Software regressions, UI checks, simulated tests, recording/replay integrity and packaging |

Foxglove is a visualization/review tool in this architecture. MuJoCo owns the
physics simulation. An encoder-driven rendering shows a measured pose; it is
not an independent physical measurement or a prediction from a physics engine.

## Integration contract

1. Keep one backend motor authority. UI controls, a Foxglove operator panel
   and future reviewed agent commands must use the same guarded command path.
2. Reuse the robot geometry, joint naming and unit conversion across viewers
   and simulation. Version and review physical/model reference corrections.
3. Preserve distinct layers: measured encoder calibration, geometric reference,
   model joint bounds and the operating safety envelope. Do not widen one to
   conceal a contradiction in another.
4. Every observation identifies real, simulated or mock data; measured and
   proposed poses stay distinguishable. Stale data cannot qualify readiness.
5. Record only explicit activities. Keep original timing, calibration identity,
   source metadata and rejected-episode evidence. Export selected data for
   training after validation and human review.
6. Integrate tools where useful: embedded views, panels, shared model assets,
   API/WebSocket adapters and file playback. Integration does not require
   rebuilding each specialist application inside the Suite.
7. Add another simulator only for a concrete unmet experiment requirement.
   Connect it through an adapter and qualify its model and units. Do not run
   multiple physics engines as competing sources of robot truth.

## Sequence

First qualify the present hardware references, rest pose, motion and stops.
Then complete one real demonstration through recording, review and playback;
collect ten accepted episodes once the pilot passes. Use those measurements
to improve model alignment and controller settings. Next establish a small
pick-and-place baseline and evaluate learned behavior on held-out placements.

Use MuJoCo as the primary simulator throughout this sequence. Expand the UI
only where operating these workflows exposes a specific gap. Keep Foxglove
as the deeper inspection surface and the Suite as the everyday entry point.

Passing software CI does not establish physical calibration, safe motion or
task success. Operator hardware qualification remains a separate gate.

Official references: [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html),
[Foxglove data connections](https://docs.foxglove.dev/docs/visualization/connecting),
and [local recording playback](https://docs.foxglove.dev/docs/visualization/connecting/local-data).

# Local Foxglove operations dashboard

The publisher listens at `ws://127.0.0.1:8766`. Run `./foxglove-arms` to open
the installed desktop app with that source. The WebSocket publisher remains
read-only: it advertises no command or service capability and never opens a
motor bus. Starting it does not create an MCAP.

Import `config/visualization/foxglove-operations-layout.json` for the live
camera, encoder model, joint plots and status tabs. A previously imported
layout is a copy; import this version to obtain newly added panels.

| Feed | Source and meaning |
| --- | --- |
| `/camera/obsbot/image` | Existing camera relay, host camera-read timestamps |
| `/camera/camera_1/image`, `/camera/camera_2/image` | Other RGB relays, individually measured rates |
| `/camera/realsense/image` | D435i color relay at 640×480, 30 fps target; depth is not recorded |
| `/robot/encoder/scene` | Vendored STL geometry driven by measured follower values |
| `/robot/encoder/joints` | Current or last-read follower pose and freshness |
| `/robot/control_tick` | Actual control ticks with original timestamps, intent, accepted command and measurements |
| `/robot/control_capture` | Stream identity, last sequence and missing-buffer indicators |
| `/robot/status`, `/control/state` | Cached controller, torque, stop and ownership state |
| `/robot/health` | Existing servo-health samples |
| `/recording/status` | Recorder readiness, activity, cameras and episode annotations |
| `/diagnostics/preflight` | Last saved preflight report; reading it does not rerun preflight |

If no controller loop is sampling, use the Suite's **Digital twin → Physical
arm → Read current pose** with the physical setup confirmed. The manager lends
its bus for reads only; it refuses an energized arm and competing operations.
Foxglove uses those cached follower reads without opening another driver.
Old reads are faded and marked stale. The independent simulation is a separate
source and is not evidence of actual robot motion.

The operations layout plots measured follower positions from
`/robot/encoder/joints`, including supported idle pose reads. Leader intent and
accepted-command traces appear only while the control loop is sampling. Check
the **Encoder freshness** tab before interpreting a held value as current.
The episode review layout keeps all three traces on the recorded control ticks.

## Operator panel

The local extension source is `extensions/so101-operator`. Build it with
`npm run build` in that directory after installing the frontend dependencies.
It registers **SO101 operator** and embeds the existing local Suite UI. This
reuses its preflight gates, operator ownership, safety state and recording
workflow rather than creating another motor-command implementation.

The workstation's local extension directory is
`~/.foxglove-studio/extensions/so101-suite.so101-operator-0.1.0`. Reload Foxglove
after installation, then add the **SO101 operator** panel. It exposes the
Suite's own navigation, settings, calibration, Digital twin and teleop workflow.
Keep it large enough that the stop controls are visible. H and Escape are Suite
shortcuts only when the embedded Suite panel has focus; Foxglove's other panels
do not acquire these shortcuts. Qualify the stop behavior with the operator
before using this workspace for powered operation.

Foxglove's current documentation says local extensions require a developer
seat. If the installed app asks for that entitlement, do not treat a compiled
extension as an activated panel or attempt to bypass the requirement. The
ordinary Suite UI remains the control surface while the built-in Foxglove
panels display the same observations.

This panel's build/DOM checks and the publisher's data-delivery checks are not
proof that its desktop activation, layout import or physical stop checks have
been completed. Those are separate verification steps.

## Recording and refresh

On the G15, run `so101` with no arguments to stop and restart the managed
Suite services, reload their configuration, check readiness and open a fresh
data collection dashboard. It refuses to interrupt active motion, calibration
or recording, or an arm whose torque-off state cannot be confirmed. It does
not clear faults, arm motors or start recording. `so101 status` checks services;
other Suite CLI subcommands are forwarded to this checkout's CLI. Existing
shells that cached the older toolkit command need `hash -r` once.

The recorder waits between explicit activities. Start and save episodes in the
Suite operator panel; calibration/diagnostic activity requests are separate.
Opening a camera or Foxglove view does not start a recording. SigmaDrive is the
raw MCAP destination; uploads remain disabled.

Live control data uses the existing control loop; the dashboard reads its
buffer every 100 ms and preserves each tick's timestamp. Status snapshots run
at approximately one update per second. Manual pose reads target 100 ms and
pause when hidden or when the controller owns the arm. Raising every polling
rate would increase bus contention and duplicate requests without increasing
the physical sample rate.

The collection dashboard's supported pose button starts continuous reads until
**Stop live readings** is pressed. Responses update the displayed angles directly;
cached observations from another view refresh every 250 ms. Actual read rates
depend on bus/request duration; the client keeps one request in flight.

## Qualification limits on this G15

The local publisher has delivered OBSBOT images, encoder scenes, motor/control
status, health, recording state and preflight observations to a test subscriber.
Real follower and leader pose reads have rendered in the Suite STL viewer.
Desktop operator-panel activation and powered hardware operation still require
verification. The shoulder rest-pose/model discrepancy and later leader
wrist-roll overflow remain open; do not clear them by relabeling a dashboard.

On 2026-10-27 the D435i negotiated USB 2.1. Color-only capture at 640×480
measured approximately 30 fps; a combined color/depth attempt timed out.
The color relay is integrated into the same dashboard, Foxglove publisher and
activity-gated recorder as the other cameras. Depth streaming and USB 3
qualification remain pending; color delivery does not qualify depth.

Official references: [local extension development](https://docs.foxglove.dev/docs/extensions/local-development)
and [extension requirements](https://docs.foxglove.dev/docs/extensions).

# Leader teleop and MCAP workflow

Start the configured workstation with `./services up`. This starts observation
and recording services; motor power and motion still require the operator UI.
Open <http://127.0.0.1:8000/teleop>. `./services status` reports availability,
camera freshness and measured frame rates separately from motor readiness.

## Before physical operation

Complete the Overview preflight wizard. Review the calibrated arm, motor IDs,
supply profile, clear workspace, reachable cutoff and supported rest position.
Resolve every required failure/skip before starting. Do not reset a fault to
hide a failed prerequisite. Service availability alone does not qualify motion.

The current G15 still needs a follower rest pose within the effective limits
and resolution of a contradictory leader wrist-roll sweep. Preserve its
existing reference data. Six repeatable sweeps are useful evidence; a later
overflow is evidence to investigate, not a reason to replace measured ranges
with generic factory values.

With the server stopped and the physical setup confirmed, `./record-rest-pose`
records a supported pose. It disables torque; read its printed setup carefully.
The pose must pass preflight without clamping. After starting services, inspect
model correspondence joint by joint before relying on a powered return path.
An arbitrary folded shape is not automatically a validated home pose.

## Demonstrations

1. Open motor mapping and cameras before motion so the 3D view can finish loading.
2. Confirm the MCAP service is ready and the motor stream is live.
3. Start leader teleop. Enter a task and start an MCAP episode.
4. Demonstrate the task, then save the episode. Repeat for another episode.
5. Stop teleop with Halt. The existing hold timer and bounded Collapse apply.

The online MCAP service waits without creating files between activities. Starting
an episode requests capture and waits for the recorder to acknowledge it before
the episode starts. Save records its interval in `data/real/mcap/episodes.json`
and `/recording/episodes`, then closes the activity's MCAP. Long activities rotate
into five-minute segments; an open segment ends in `.mcap.active`. A stopped loop, recorder failure
or server restart marks an unfinished episode interrupted. The originating
operator owns the save action. Recording annotations never grant motion rights.

Supported standalone calibration and diagnostic tools can request capture for
their operation. An explicit timed diagnostic capture also records for its
requested duration. Camera previews, Foxglove viewing and idle teleop do not
start MCAP recording by themselves. Stopping the recording service closes its
current file; new episode capture requires the service to be ready again.

The separate LeRobot dataset recorder remains available for training datasets.
It is a different workflow and may need direct camera ownership; do not start
it while relay processes own those same camera devices. MCAP observation logs
are not automatically synchronized LeRobot training episodes.

## Manual thumbsticks

Release leader teleop and its Halt first. Select Thumbsticks, acquire the
manual lease, and synchronize from the measured pose. Hold the existing
"Hold to move" control while using WASD for shoulder pan/lift and arrows for
elbow/wrist flex. Q/E controls roll; Z/X controls gripper. Targets advance at
most four degrees or gripper units per second through the existing controller.
Releasing the hold Halts; releasing keys centers input. Check each physical
direction during operator bring-up. A stale command, lease or supervision
heartbeat still stops the controller.

## Camera and encoder comparison

The cyan model and numbered pivots use cached encoder telemetry, the vendored
SO-101 body tree, its STL meshes, joint axes and degree/gripper conversions.
An idle or stale sample is labeled accordingly. The mathematical model has
been checked against backend FK; physical zero/frame alignment remains a
separate measurement.

The independent green image overlay detects ArUco 4×4/50 tags numbered 1–6.
Use the printable tag sheet in the UI and place each tag beside its matching
motor, clear of pinch points and travel. Start vision to see detections and
connecting lines on OBSBOT. Missing or ambiguous tags remain missing. Tag
centers do not establish pivot centers, camera calibration or 3D registration.
This view does not command motion or perform automatic calibration.

Foxglove publishes `/robot/encoder/scene` and `/robot/encoder/joints` separately
from the independent `/simulation/*` demo. Import the prepared layout from
`config/visualization/foxglove-arms-layout.json` and use its Motor encoders tab.
An imported desktop layout is a copy; import again after changing this file.

## Home and automatic calibration

The existing Collapse controller returns toward a validated saved rest pose
with capped joint speeds and then turns torque off. It does not certify a
collision-free route from every possible pose. Keep the workspace clear and
qualify the route with the operator.

Saved calibration read-back and preflight are verification, not a new physical
calibration. Autonomous hard-stop seeking is not qualified: current/load unit
scales, contact thresholds, travel bounds and stop behavior need measured
validation first. Do not enable hard-stop seeking as a workaround for the
remaining reference discrepancies.

# Ten-episode teleoperation session on the G15

The target is ten accepted **real** demonstrations with camera playback, motor
measurements and an encoder-driven robot replay. Software fixtures do not count
toward those ten. A data-check pass still requires a person to review task success.

## Current physical prerequisites

Operator decision on 2026-10-27: shoulder operating range -101 to +100 degrees;
elbow upper operating bound +99 degrees, below its installed calibrated maximum
of +99.47 degrees. The model joint and actuator ranges match those adjusted
boundaries. The proposed 95-degree override was removed. A new supported rest
was captured at shoulder -98.99 and elbow +97.89 degrees, replacing the old
-105.05-degree shoulder record. Calibration offsets and encoder ranges remain
unchanged. Software command limits do not limit manual torque-off travel or
prove physical collision clearance. Refresh preflight after loading these limits;
the old October 23 report does not describe the new capture.

The follower has ten accepted calibration sweeps per joint. The leader has an
installed six-sweep baseline, but a later wrist-roll overflow needs resolution.
The current recorded follower rest resolves without clamping under the reviewed
bounds above; the saved October 27 preflight passed its required checks. Keep
the original calibration evidence and passed joint results.

First inspect the current pose in **Operate → Calibration & pose checks** with both
arms supported and torque-off. Compare the encoder-driven model with OBSBOT.
This verifies the visible pose and direction; endpoint repeatability alone does
not establish the model's geometric reference.

First place a stable padded mechanical catch beneath the load-bearing links,
clear of the PCB, cables and joints. With torque off, observe where the arm
settles hands-free. Do not move it to an arbitrary angle simply to pass the
model limit: -95 degrees is not a required calibration or rest target. A joint
holding that position through friction alone is not proof of a reproducible
operating rest. A protective catch below the links and a load-bearing rest
cradle are different fixtures; a cradle must be planned as part of the workspace.

If the reproducible natural rest remains near -105 degrees, retain that evidence
and resolve the physical/model reference mismatch before choosing a replacement
rest. Do not widen limits without review or record an artificial position just
to clear preflight. Only after a stable intended rest and its reference have
been reviewed, stop the pose reader, stop the motor service and capture it:

```bash
systemctl --user stop so101-motor.service
/home/adnan/so101-suite/record-rest-pose
```

Follow the tool's setup instructions, remove hands and let the supported arm
settle before capture. A valid rest capture must have no clamping warning.
If the arm cannot rest stably inside the existing limits, leave the warning
unresolved and review the physical/model reference before changing limits.
The two-pose `check-shoulder-reference` tool remains available for that review;
it requires a measured pivot-to-pivot angle, rather than the plastic link edge.
Do not repeat calibration or install new offsets just to make preflight green.

After the shoulder and leader wrist-roll checks are resolved, verify installed
registers, the rest target and preflight. The operator then checks stops and
joint correspondence through the UI. Confirm model direction and physical
motion before trusting a return-to-rest transition.

## Qualify one pilot before collecting the remaining nine

Leader teleop uses the pinned LeRobot SO101 leader/follower device methods.
During the manager-owned leader loop, the adapter uses LeRobot's native
unrestricted relative-target setting instead of the separate five-degree
window around the present follower position. A small present-position window
can repeatedly clip the target while a loaded joint has not yet moved. This
is a different constraint from limiting how quickly the command changes.

SafetyGuard still enforces the installed position envelope and hard step
ceiling. `SO101_TELEOP_MAX_VELOCITY_DEG_S` additionally caps the leader loop's
command slew at 60 degrees/second by default (gripper: percentage points/second),
or any stricter existing joint velocity limit. This is a commanded-target
limit, not a measurement or guarantee of physical joint speed. The scoped
profile restores the previous driver window and velocity configuration when
the loop exits, including on errors. Manual control and Collapse keep their
existing driver configuration. Calibration and servo gains are unchanged.

After restarting the idle suite with `so101`, label the next episode
`teleop smoothness pilot 2`. Repeat a short, slow shoulder/elbow movement with
pauses, then press Stop Teleop. Compare requested, accepted and measured values
and stationary intervals against the previous pilot before collecting task
demonstrations. A software test or a structurally valid MCAP does not establish
smooth physical following or optimal motor configuration.

Use **Operate** at `http://127.0.0.1:8000/operate` for today's session.
It combines the relay camera feeds, measured follower 3D view, task label,
leader teleop and the episode log. The desktop visual area uses 75% of the
available width; controls, calibration/pose checks and preflight share the
scrollable right rail. Halt and E-stop remain fixed at the top right.
Keyboard/thumbstick controls are an explicit alternative under Tools; they do not mount or
acquire a manual lease until opened. The real dashboard does not offer the
incompatible native direct-camera dataset capture buttons.

Local HTTP previews use the loopback camera relay ports directly. Their
same-site resource policy permits the local dashboard to embed them while
blocking embedding from unrelated sites. This keeps four long-running MJPEG
connections out of the UI origin's HTTP connection pool. Remote or HTTPS
clients use finite snapshot requests through the API. If local relay access
fails (for example in an embedded browser), previews automatically fall back
to that same finite path rather than holding four API connections open.

Idle model updates use cached, supported pose reads from the Digital twin
page; opening the collection dashboard never opens a motor bus or enables
torque. Stale poses and unavailable camera previews remain visibly marked.

Use one repeatable, simple task with an unobstructed camera view. Define its
starting placement and success condition before recording. First qualify a
short motion check at the existing bounded settings; change motor configuration
only when measurements justify it. Endpoint calibration alone does not establish
optimal motor gains or physical model alignment.

1. Start the qualified server and observation services. The recorder should
   show **ready**, with fresh camera feeds. Being online does not arm the robot.
2. Enter the task label, then press **Start Teleop** under the operator's control.
3. The same action opens an MCAP episode automatically. Wait for **Episode
   recording** before demonstrating. Capture-start failure requests Halt and
   displays the error; service readiness alone is not capture acknowledgement.
4. Demonstrate for at least five seconds, then press **Stop Teleop**. It saves
   the episode boundary and requests Halt, including if saving fails. The save
   request has a one-second timeout before Halt is requested. The global Halt
   and E-stop controls act immediately, independently of saving. Using those
   stops or losing capture can leave an interrupted episode requiring review.
5. Review the pilot's video, recorded joint values, health and encoder model.
   Resolve discrepancies before collecting the remaining nine demonstrations.

The recorder creates files only during episodes, supported calibration/diagnostic
activities, or an explicit timed diagnostic. Camera previews can stay live between
episodes without writing MCAPs. Files go to SigmaDrive; upload remains disabled.

Each new segment includes `/recording/metadata`: episode/activity ID, segment ID,
task label, control stream ID, clock conventions and source-code SHA-256 hashes.
Calibration snapshots retain their own provenance. `/recording/episode` carries
the matching episode annotation, including the final saved/interrupted state;
`/recording/episodes` retains the broader episode index with operator IDs removed.
Every `/robot/control_tick` carries its episode/activity ID, sequence, timestamps,
requested rate, measured positions, leader intent, accepted motor command and
the active control profile (joint limits, units, slew, relative-target window,
calibration identities and available driver version). Camera timestamps/rates,
health and stop events remain separate streams linked through episode bounds.
Diagnostic/calibration activities have their own IDs and no episode ID.
Derived episode exports preserve the matching segment metadata even when it
was written before the acknowledged episode start. Human review remains pending
until the operator checks the footage and task outcome.

## Export and inspect

On this workstation, export the selected saved episodes after their raw files close:

```bash
cd /home/adnan/so101-suite
./review-episodes --since 2026-10-27T21:03:26Z --count 10
```

Use `--count 1` for the pilot. The command prints the versioned report and exported
MCAP paths. It preserves source timestamps and calibration snapshots, verifies
exported CRCs and records SHA-256 hashes. It does not replay robot commands.
Historical originals remain intact. Choose a new `--since` time for a later session.

The current acceptance checks require real motor data for all six joints,
continuous sequence numbers, at least 27 Hz toward a 30 Hz control target and no
control gap above 100 ms. Control samples must cover both episode boundaries
within 150 ms. OBSBOT must achieve at least 27 fps, no gap above 150 ms and
boundary coverage within 250 ms, using its host camera-read timestamp. This is
not a hardware exposure timestamp or proof of camera/motor synchronization.
Other camera rates are reported individually. Critical health and safety faults
fail data acceptance. Review tracking errors and clamped commands separately;
passing integrity checks does not prove optimal tuning.

In Foxglove, import
`config/visualization/foxglove-episode-review-layout.json`, then open an exported
MCAP locally. Its camera tabs, joint plots, health and `/robot/encoder/scene`
show the recorded episode. The encoder scene reconstructs measured joints;
it does not prove that the physical geometry is aligned. The separate simulation
demo is not a recording of the arm. Keep the local model asset service available
on port 8767 for this workstation layout.

Mark each episode's task outcome and review any obscured views, wrong directions,
unexpected contact, tracking errors or incomplete actions. Replace rejected
demonstrations until there are ten accepted real episodes. Keep rejected evidence
with its reason rather than silently deleting it.

## Deployment status

Recorder activity gating has been exercised on this G15. The new per-control-tick
capture and episode export/replay flow has passed an isolated software run with
mock motors and real camera feeds. It has not qualified real powered teleoperation.
The updated backend and UI are deployed on this workstation. Leader teleop
uses shared camera capture so the existing relays retain the camera devices;
the native direct-camera recording flow requires an explicit camera handoff.
Configuration and source tests are not proof of powered teleop qualification.

## Workstation verification on 2026-10-27

The consolidated dashboard was built and inspected in workstation Chrome with
four camera images, the complete numbered STL arm, current encoder samples,
preflight gating and the episode controls. All 733 frontend tests passed,
including accessibility and stop-shell checks; type checking, full frontend
lint and the production build passed. Focused backend checks cover shared
camera isolation, relay freshness, episode durability, activity-only recording,
export/replay, model generation and Foxglove delivery. These are software and
observation checks, not a powered hardware qualification.

Measured live color rates were approximately 30 fps for OBSBOT and D435i,
19 fps for camera 1, and 5 fps for camera 2. The latter two are visibly marked
below target. A shorter manual exposure improved camera 2 to roughly 11 fps
but did not qualify 30 fps; its original settings were restored. D435i color
works at 640×480 on the current USB 2.1 link; combined color/depth timed out.
Depth is not part of the present recording stream.

Additional camera diagnostics on 2026-10-27 read the actual UVC mode tables:
camera 1 (32e6:9005) advertises only 640×480 YUYV, including 30 fps; camera 2
(32e6:9221) advertises MJPEG modes from 320×240 through 1920×1080 at 30 fps,
plus 640×480 YUYV at 30 fps. Neither advertises 60 fps. They enumerate on
separate USB 2.0 host buses, so the shared hub is not established as the cause.
Camera 1's MJPEG request falls back to YUYV; a reported 30 fps is a requested
mode, not an observed frame rate.

Five-second relay measurements were about 15 fps and 10.8 fps. Switching
power-line compensation from 50 Hz to 60 Hz did not materially improve either.
Manual 10 ms and 3 ms exposures reached only about 16 fps and 11 fps while
darkening the images substantially. Original automatic exposure, 313 absolute
exposure and 50 Hz compensation settings were restored. An independent direct
capture test, with each tested relay stopped, also failed to sustain 30 fps:
camera 1 reached about 16.6 fps; camera 2 remained about 5–8 fps across MJPEG
640×480, MJPEG 320×240 and YUYV 640×480 during the short tests. These tests
do not identify a unique sensor/firmware, lighting or USB-path cause, but they
show that UI polling and JPEG relay scheduling alone do not explain the deficit.
Relays were restored after testing; no motor operation or episode was started.

Foxglove protocol checks received all four camera streams, encoder scenes and
joint values, robot health/status, control state, recorder state and saved
preflight. The native app is connected, but its imported layout and optional
operator extension still need desktop visual confirmation. Import the updated
operations layout after adding D435i; old imported copies do not update.

Operator work still required:

1. Compare physical motor positions and directions with the encoder model.
2. Establish the follower's stable intended rest and resolve its model/reference
   mismatch before recording a replacement. Preserve the existing calibration.
3. Resolve the leader wrist-roll range contradiction, retaining passed joints.
4. Aim and secure D435i so it sees the work area rather than the ceiling.
5. Rerun preflight, qualify bounded motion and the stops, then save and review
   one pilot episode before collecting the other nine.

# Simulation

The suite has two simulators. Neither opens a serial port or writes a motor
command.

| | Kinematic engine | MuJoCo physics |
|---|---|---|
| Needs | nothing extra | the `sim` extra (`mujoco>=3.3,<4`) |
| Models | joints move towards their targets at a capped speed; forward kinematics; seeded scene objects that never move | the SO-101 with STS3215-like servos, gravity, joint friction, contacts and grasping |
| Used by | `/api/simulation` and the browser twin when MuJoCo is missing or `SO101_SIM_ENGINE=kinematic` | `so101 sim ...` (scripted demos, policy evaluation, camera renders); `/api/simulation` by default when installed; the simulated robot devices |
| Code | `simulation/engine.py`, `scenarios.py` | `simulation/world.py`, `servo.py`, `tasks.py`, `expert.py`, `rollout.py`, `policy.py`, `render.py`, `physics_engine.py`, `selection.py`, `capabilities.py`, `devices.py` |

Both read the same vendored MJCF, so they agree on the arm's geometry (see
[The model](#the-model)). Nothing in `so101.simulation` imports MuJoCo at
import time: the physics modules are imported only by the code paths that need
them, and the rest of the suite runs without the extra.

## Install and first run

```bash
pip install 'so101[sim]'      # adds mujoco>=3.3,<4; the model ships in the wheel
pip install pillow            # optional: JPEG renders (PNG otherwise)

so101 sim info                # MuJoCo version, model, timestep, tasks, renderer
so101 sim render --task pick_place --camera front --out front.png
so101 sim eval --policy expert --task pick_lift --episodes 20
so101 sim demos user/sim_pick_lift --task pick_lift --episodes 50 --cameras front,wrist_cam
```

Without the extra, `so101 sim` commands exit with the install hint and
everything else is unaffected.

Rendering needs an OpenGL context. macOS uses CGL and needs no window or
display. Headless Linux needs `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa`
(CPU); those two paths were not tested in this build environment. If no
context can be created, renders are unavailable and say why
(`so101 sim info` shows the reason), and state-only runs (no `--cameras`)
still work.

Measured with MuJoCo 3.14.0 on an arm64 Mac (macOS 26).

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `SO101_SIM_TASK` | `pick_lift` | default task for `so101 sim`, and the scene of the simulated robot (`reach`, `pick_lift`, `pick_place`) |
| `SO101_SIM_SEED` | `0` | default first seed, 0 or more; episode *i* uses seed + *i* (the simulated robot too) |
| `SO101_SIM_CAMERAS` | `front,wrist_cam` | cameras of the simulated robot backend |
| `SO101_SIM_IMAGE_SIZE` | `320x240` | default frame size, `16x16` to `1280x960` |
| `SO101_SIM_REALTIME` | `true` | pace the simulated robot backend to the wall clock |
| `SO101_SIM_ENGINE` | `auto` | `/api/simulation` engine: `auto`, `kinematic` or `mujoco` |
| `SO101_BACKEND` | follows `SO101_MOCK` | `mock`, `sim` or `real`; `SO101_MOCK=1` with `real` or `sim` is an error |

Invalid values (an unknown task or camera, a size outside the render limits,
a negative seed) fail at startup instead of at first use. The `--seed`
options of `so101 sim render`, `demos` and `eval` refuse negative seeds too.

## The model

`src/so101/simulation/assets/robotstudio_so101/` vendors the MuJoCo Menagerie
`robotstudio_so101` model at commit `4d038b3` (Apache-2.0; `LICENSE` and
provenance next to it). The collision and mass meshes are byte-exact copies.
Visual meshes are decimated (17.4 MB to 2.7 MB); the original mesh conversion
preserved dynamics bitwise over 18,000 steps. The subsequent documented G15
adaptation changes shoulder/elbow operating ranges in the MJCF; that earlier
dynamics comparison does not qualify the changed limits or physical clearance.
`scripts/vendor_so101_model.py` reproduces the base conversion, and
`scripts/adapt_so101_g15_model.py` reproduces the four G15 range substitutions
from the pinned upstream XML. Its `--check` verifies the adapted XML hash.
Use the vendor tool's `--g15-operating-ranges --check` to compare the complete
adapted tree, including its provenance manifest. The standalone adapter also
updates and checks the manifest; a plain vendor `--check` compares the base
conversion instead.

`simulation/mjcf.py` reads the MJCF body tree in pure Python. The kinematic
engine, the scenario telemetry and the browser twin all use it, so their
forward kinematics agree with MuJoCo's `mj_kinematics` to about 1e-15 m.
`tests/fixtures/so101_fk_golden.json` pins that agreement.

The physics scene (`world.py`) adds a table (0.8 x 0.6 x 0.04 m, top at
z = 0), a floor, lights, a `front` camera at (0.62, 0, 0.42) m looking at
(0.16, 0, 0.06) m with a 50 degree vertical field of view, task objects and
visual-only markers. `wrist_cam` is the model's own camera on the gripper.
World frame: the arm base is the origin, +x points forward, +z up.
Quaternions are (w, x, y, z).

## Units and limits

Everything outside the simulator uses suite units, the same as the motor,
safety and API layers:

- the five arm joints are in degrees, zero at the middle of the range (the
  LeRobot `use_degrees=True` zero, which is also the MJCF zero);
- the gripper runs from 0 (closed) to 100 (open). It maps linearly onto the
  actuator's -0.1745..1.7453 rad, so commanding 0 squeezes slightly past jaw
  contact. That squeeze is what makes grip force.

Effective limits are the `SafetyGuard` limits intersected with what the model
can reach (the tighter of each joint range and actuator control range),
rounded to 0.01:

| Joint | Min | Max |
|---|---|---|
| shoulder_pan | -110 | 110 |
| shoulder_lift | -100 | 100 |
| elbow_flex | -96.83 | 96.83 |
| wrist_flex | -95 | 95 |
| wrist_roll | -157.21 | 160 |
| gripper | 0 | 100 |

Velocities are degrees per second (gripper: units per second). Whether the
model's joint signs match a calibrated real arm is a hardware-in-the-loop
check, not an assumption.

## Servo model

The actuators are Menagerie's STS3215 parameters: a position actuator with
kp 998.22 N·m/rad, kv 2.731 and a ±2.94 N·m force range, on joints with
armature 0.028, damping 0.60 and friction loss 0.052 N·m. The gripper's force
range is halved to ±1.47 N·m, matching the `Max_Torque_Limit=500` that LeRobot
writes on the follower's gripper. The integrator is `implicitfast`, as in the
model.

A real STS3215 does not jump to a new goal: it runs a trapezoidal move limited
by its `Acceleration` register and its top speed. `servo.py` reproduces that
move once per physics substep and feeds the result to the actuator:

- acceleration 2232 deg/s²: LeRobot writes `Acceleration=254`, which is
  254 x 100 steps/s² x 360/4096 deg/step;
- top speed 270 deg/s: the 7.4 V STS3215's 0.222 s per 60 degrees, no load.

Without the profile, the stiff actuator (which saturates above about
0.17 degrees of error) would turn every goal change into a full-torque step.
`servo_mode="step"` turns the profile off and sends raw position targets.

Torque off disables actuation, so the arm sags under gravity against joint
damping and friction like unpowered servos. Turning torque back on first sets
every goal to the present position, so the arm does not jump.

## Timestep

The physics runs at 1/240 s, not the model's native 5 ms, so that both
control rates divide exactly: a 30 Hz control step is 8 substeps and the
60 Hz engine tick is 4. At 5 ms the periods are 35 ms (+5 %, 7 substeps) and
15 ms (-10 %, 3 substeps).

`scripts/measure_sim_timestep.py`, 20 episodes per task, measured while other
jobs loaded the machine (speeds are indicative):

| | 5 ms | **1/240 s** | 1/480 s |
|---|---|---|---|
| 30 Hz / 60 Hz period error | +5 % / -10 % | exact | exact |
| hold error; drift from 1 s to 10 s | 0.022°; 0 | 0.022°; 0 | 0.022°; 0 |
| raw 30° step: overshoot, 2 % settle | 3.39°, 0.265 s | 3.70°, 0.271 s | 3.37°, 0.262 s |
| cube on the table: penetration, creep | 0.061 mm, 0 | 0.061 mm, 0 | 0.061 mm, 0 |
| expert pick_lift / pick_place | 20/20, 20/20 | 20/20, 20/20 | 20/20, 20/20 |
| cube slip in the jaws over 1.5 s (mean) | 0.182 mm | 0.184 mm | 0.194 mm |
| jaw penetration (max) | 0.94 mm | 0.94 mm | 0.97 mm |
| contact-free hold, physics steps/s | 144,877 | 159,811 | 166,295 |
| pick_lift / pick_place, x real time | 76 / 46 | 69 / 34 | 48 / 34 |

1/240 s keeps both rates exact for about 10-25 % of the 5 ms speed, with the
same contact and grasp results. 1/480 s buys nothing measurable.

## Tasks

| Task | Instruction | Length | Success | `observation.environment_state` |
|---|---|---|---|---|
| `reach` | move the gripper to the red ball | 6 s | the tool point stays within 2 cm of the ball for 0.3 s | `target_x, target_y, target_z` |
| `pick_lift` | pick up the orange cube and lift it | 10 s | the cube is held at least 5 cm above the table for 0.5 s | `cube_x, cube_y, cube_z, cube_qw, cube_qx, cube_qy, cube_qz` |
| `pick_place` | put the orange cube in the green square | 16 s | the cube was lifted at least 2 cm, and now its centre is over the square, it rests on the table (bottom within 5 mm) and no part of the arm touches it, for 0.3 s | the 7 cube values, then `zone_x, zone_y, zone_z` |

Every task is seeded: the same seed gives the same layout and start pose.

- The arm starts at the ready pose (0, -60, 30, 85, 0, 30) with ±3 degrees of
  jitter per arm joint. Objects settle for 0.3 s before t = 0.
- The red ball (12 mm radius, visual only) is 0.16-0.26 m from the base axis,
  within ±40 degrees of straight ahead and 4-16 cm above the table.
- The 4 cm cube is 0.16-0.25 m out, within ±35 degrees, yawed ±45 degrees.
- The 10 cm green square is 0.15-0.25 m out, within ±40 degrees, and at least
  12 cm from the cube.
- Positions are metres in the world frame; `cube_z` is the cube's centre.

Success is the first time the predicate holds; recorded demos continue for
0.5 s after it so that they end settled.

## Scripted expert

`expert.py` solves damped-least-squares inverse kinematics on the tool site
and runs a phase machine: approach a hover pose 3 cm above the cube with the
jaws aligned to its faces, re-centre on the cube, descend, close, lift, and
for `pick_place` transport, place, release and retreat. It reads only the
observation (joint state and environment state) and its targets go through the
same `SafetyGuard` and servo model as any policy.

`scripts/measure_sim_expert.py --episodes 100`, seeds 0-99:

| Task | Clean expert | Noisy expert (1°, 4 mm) | Mean time to success |
|---|---|---|---|
| reach | 100/100 | 100/100 | 0.87 s |
| pick_lift | 100/100 | 99/100 | 4.95 s |
| pick_place | 100/100 | 99/100 | 8.94 s |

Wilson 95 % interval for 100/100: 0.963-1.0. The noisy expert adds
correlated joint-target noise (1 degree, about 0.3 s correlation, a quarter of
that near contact) and moves the grasp and place points by a random offset
with a 4 mm standard deviation; seed 35 fails both pick tasks. Use the noise options for
varied demonstrations.

## Policy interface

A policy is anything with `reset()` and `select_action(observation)`, the
LeRobot convention. The observation keys are pinned:

- `observation.state`: float32 (6,), suite units, `JOINT_NAMES` order;
- `observation.environment_state`: float32 (k,), the task layout above;
- `observation.images.<camera>`: H x W x 3 uint8, only for requested cameras.

The action is float32 (6,): absolute suite-unit joint targets in
`JOINT_NAMES` order. Each control step (30 Hz) runs observation, policy,
`SafetyGuard.clamp` with an explicit dt, servo, then 8 physics substeps.

`so101 sim eval --policy` takes `expert`, `hold` (hold the start pose) or a
model path. Model formats register a loader instead of editing the CLI:

```python
from so101.simulation.policy import register_policy_loader

register_policy_loader(is_my_checkpoint, load_my_policy)  # (path) -> bool, (path) -> policy
```

Later registrations are tried first; a path nobody recognises is an error. A
policy trained on images needs the same `--cameras` (and `--image-size`) at
evaluation.

## `so101 sim` commands

Runs never wait for the wall clock and are deterministic: the same task, seed
and policy give the same joint states, actions and environment states.

- `so101 sim info [--json]`: versions, model provenance, timestep, tasks,
  cameras, renderer status, current settings.
- `so101 sim render --out FILE [--task] [--seed] [--camera] [--width] [--height]`:
  one frame of a task's seeded start layout. A `.png` name writes PNG; `.jpg`
  or `.jpeg` writes JPEG and needs Pillow or OpenCV.
- `so101 sim demos REPO_ID [--task] [--episodes] [--seed] [--cameras] [--image-size] [--noise-deg] [--waypoint-noise-mm]`:
  records scripted-expert episodes through the same `SafetyGuard` and
  `EpisodeRecorder` as teleop recording. Each frame stores
  `observation.state`, `action`, `desired_action`,
  `observation.environment_state` and the requested
  `observation.images.<camera>` (feature names are the camera names). Every
  episode is labeled `success` or `failure` by the task predicate, with the
  source `auto:sim_predicate`. The run's settings, seeds, provenance and
  per-episode results go to `sim_demos.json` in the dataset directory.
  Appending to a dataset with a different environment-state or camera layout,
  or a different frame size or dtype, is refused. Fallback metadata keeps each
  camera's `camera_shapes` (H, W, C), `camera_dtypes` and
  `camera_depth_keys`; for a dataset recorded before those fields, the layout
  is proved from all published image chunks before another episode is added.
- `so101 sim eval [--policy] [--task] [--episodes] [--seed] [--cameras] [--image-size] [--out]`:
  closed-loop evaluation with a Wilson 95 % interval. The JSON report (default
  `<outputs>/sim_eval/<task>-<policy>-<time>.json`) holds per-episode seeds,
  outcomes and metrics plus provenance.

With LeRobot installed (the default install), `so101 sim demos` records
LeRobot datasets and stores each predicate label in the dataset's
`so101_outcomes.json`; without it, datasets use the suite's fallback format
(`so101-fallback-v2`) with the label in each episode's `episode.json`. Either
way inspection, the Datasets page and the training-data loader read the
labels. If a label cannot be stored, `so101 sim demos` stops with exit code 1
and names the saved, unlabeled episode, since the default `kept` selection
would train on it.

## HTTP API

`/api/simulation` serves the browser twin. It is a sandbox with its own world:
it never touches the robot, the control plane or the
[simulated robot devices](#simulated-robot-devices).

### Engine selection

`SO101_SIM_ENGINE` picks the engine on the first request:

| Setting | Engine |
|---|---|
| `auto` (default) | MuJoCo physics if `mujoco` imports and the vendored model compiles, otherwise the kinematic engine. The reason is reported. |
| `mujoco` | MuJoCo physics. If MuJoCo cannot run here, the engine endpoints answer 503 with the reason; nothing runs in its place. |
| `kinematic` | The kinematic engine. MuJoCo is never imported, so a broken native install cannot affect the API. |

The probe runs once per process. `GET /api/simulation/capabilities` reports
three facts separately:

- whether the `mujoco` package is installed (`adapters.mujoco.package_installed`);
- whether the physics engine can run here (`adapters.mujoco.available`);
- which engine runs (`active_backend`, `physics_active`), with
  `selection_reason` and `error`.

With physics active, it also reports `fixed_hz`, `timestep_s`,
`substeps_per_tick`, `servo_mode`, `cameras`, and whether rendering works
(`render_available`, `render_error`, `render_status`). The report never
waits for the render thread's GL probe, because the UI polls it every 10 s.
The first report starts the probe. Until the probe returns,
`render_available` is `null` and `render_status` is `probing`. After that,
`render_status` is `available` or `unavailable: <reason>`. A probe still
running after 10 s counts as hung (see [Rendering](#rendering)). The
`simulation` entry of `/api/capabilities` describes the engine that actually
runs. It is unavailable only when no engine can run.

### One contract, two truth models

Both engines share the request contract:

- the same target validation: an out-of-range target is rejected, never clamped;
- the same request-synchronized 60 Hz fixed-step scheduler;
- the same step, play, pause and reset rules, with at most 10,000 steps per
  request;
- the same trajectory-preview planner.

What differs is what a tick does. A MuJoCo tick is 4 substeps of 1/240 s with
gravity, contacts and the STS3215 servo profile:

- `telemetry.measured` and `telemetry.velocity` are the simulated joints,
  including gravity sag, contacts and servo lag. `telemetry.target` is the
  exact command.
- Scene objects are free rigid bodies, and `scene.objects[].pose` is their live
  pose. The exception is target zones, which are visual-only markers.
- Reset lets the scene settle for 0.25 s, then restarts the clock at 0. The
  `simulation.reset` event names the engine (`backend`) and its `physics`
  flag.

### `physics_state`

Every response that carries the simulation state has a `physics_state` field.
It is `null` on the kinematic engine. On the MuJoCo engine it holds:

| Field | Meaning |
|---|---|
| `engine`, `mujoco_version`, `model` | provenance |
| `timestep_s`, `substeps_per_tick`, `physics_steps`, `physics_time_s` | the physics clock |
| `dropped_ticks` | ticks skipped by the catch-up limit (next section) |
| `torque_enabled` | whether the actuators are on |
| `servo` | `mode`; `acceleration_deg_s2` (`null` in `step` mode); `max_speed` for each joint in suite units/s; the profile's current `reference` |
| `effort_pct` | signed actuator force as a percentage of the force limit (the servo's "load") |
| `contacts` | `total`, `arm_self`, `arm_table`, `max_penetration_m` |
| `objects.<id>` | `position_m`, `quat_wxyz`, `speed_m_s`, `grasped`, and contact counts with the table, each jaw, the arm and other objects |
| `markers.<id>` | marker positions, metres |

### Catch-up and `dropped_ticks`

Neither engine runs a background thread during play. Each request simulates
the wall time since the previous one, times the playback rate. The MuJoCo
engine simulates at most 2 s of wall time per request (times the rate, so
120 ticks at 1x) and skips the rest. Skipped ticks are counted in
`physics_state.dropped_ticks`. `step_count` and `sim_time_s` count only
simulated ticks.

- A client that polls at least every 2 s never drops a tick.
- With a constant target the scene is at rest long before 2 s anyway.
- Reset sets `dropped_ticks` back to 0.

The kinematic engine has no limit, because its ticks cost almost nothing.

### Trajectory preview

`POST /api/simulation/trajectory/preview` runs the request on a copy of the
live world (`PhysicsWorld.fork`), so it never changes the live state. The copy
gets the same commands at the same ticks as `targets` followed by `step`
(`semantics: "step_targets_like_play"`). Every sampled point therefore equals,
bit for bit, what manual steps or play would produce.

`final_reached` compares the final measured pose with the final target, joint
by joint, within `reached_tolerance` (suite units):

- 0.5 on the MuJoCo engine, where gravity and contacts leave a small
  steady-state error;
- 1e-6 on the kinematic engine, where rate-limited motion lands exactly.

A 1 s preview takes about 8 ms.

### Camera frames

`GET /api/simulation/render?camera=front&width=320&height=240&format=jpeg&quality=85`
returns one frame of the live scene:

- `camera` is `front` or `wrist_cam`.
- The size is 16x16 to 1280x960.
- `format` is `jpeg`, `jpg` or `png`. `quality` (10-95) applies to JPEG.
- JPEG needs Pillow or OpenCV. Without either, the response is PNG, so check
  `Content-Type`.
- The headers `X-SO101-Sim-Camera`, `X-SO101-Sim-Step`,
  `X-SO101-Sim-Generation` and `X-SO101-Sim-Time-S` say which instant the
  frame shows. Frames are sent with `Cache-Control: no-store`.

### Status codes

| Code | When |
|---|---|
| 503 | The requested engine cannot run (`SO101_SIM_ENGINE=mujoco` without a working MuJoCo). This applies to every engine endpoint: `scenarios`, `state`, `telemetry`, `targets`, `step`, `play`, `pause`, `reset`, `trajectory/preview` and `render`. `capabilities`, `model` and `assets` still answer. |
| 503 | `render` while the kinematic engine runs, when no OpenGL context can be created here, when the GL probe has hung, when 8 frame requests are already waiting, or when the render thread does not answer within 10 s. The 10 s covers the wait for the GL probe and for the frame together. |
| 422 | Invalid targets or requests. For `render`: an unknown camera, a size out of range, an unsupported format, or a quality outside 10-95. |
| 409 | A manual `step` while playing (pause first), as before. |
| 404 | An unknown scenario or asset. |

### Model and meshes

`GET /api/simulation/model` returns the SO-101 kinematic tree read from the
vendored MJCF. It needs no MuJoCo and contains:

- the bodies;
- the hinge axes and ranges;
- the visual geoms with their mesh URLs;
- the gripper frame site;
- the map from suite units to MJCF radians, `mjcf_rad = scale * suite + offset`.

It is sent with an `ETag` and `Cache-Control: no-cache`, and answers 304 to a
matching `If-None-Match`.

`GET /api/simulation/assets/{file}` serves a vendored visual STL by exact file
name from an allowlist. Anything else is a 404, including collision meshes
and paths. Meshes are sent as `model/stl` with an `ETag` and
`Cache-Control: no-cache`, and answer 304 to a matching `If-None-Match`. Their
URLs carry no version, so a browser revalidates each mesh on every use and
never draws an old mesh after an upgrade re-vendors the model.

## Simulated robot devices

`simulation/devices.py` puts one MuJoCo SO-101 behind the robot's device
interfaces. The control plane, teleop, recording, rollout and the safety layer
can then run unchanged against the simulation.

| Object | Factory | Stands in for |
|---|---|---|
| `MujocoBus` | `make_sim_bus("follower")` | the follower's `FeetechBus` |
| `SimExpertLeader` | `make_sim_bus("leader")` (any name containing `leader`) | the leader's `FeetechBus`; reports the scripted expert's targets |
| `SimCamera` | `make_sim_camera("front")` or `make_sim_camera("wrist_cam")` | an OpenCV camera in `CameraManager` |

With `SO101_BACKEND=sim` the hardware layer builds these devices instead of
Feetech buses and OpenCV cameras. [Integration points](#integration-points)
lists where they plug in.

Recording from the UI or the API (`/api/recording`) under `sim` works like
`so101 sim demos`:

- each episode start resets the scene with the next seed (`SO101_SIM_SEED`
  for the first episode, then one more per episode) and moves the arm to the
  task's start pose;
- every frame records `observation.environment_state`, laid out as the task's
  `env_state_names`; a dataset whose saved episodes have another layout is
  refused;
- each saved episode is labeled `success` or `failure` by the task predicate,
  with outcome source `auto:sim_predicate`. A human label is never replaced.
  A successful save answer carries `outcome` and `label` (how the label was
  stored). If grading or label persistence fails, the episode remains saved
  but the request fails loudly with its episode index and recovery guidance;
  it must be labeled before training.

The leader is the scripted expert, so these are expert demonstrations driven
through the same control plane, SafetyGuard and recorder as teleop.

### One shared world

All devices share one `SimRuntime`, from `get_sim_runtime()`. It is built on
first use from `SO101_SIM_TASK`, `SO101_SIM_SEED` and `SO101_SIM_REALTIME`.
At power-on the arm is folded at the rest pose with torque off and settles
for 0.5 s; then the clock restarts at 0. The goal register equals the
measured pose, as on a real arm found unpowered. Nothing moves until the
control plane enables torque.

### Follower

`MujocoBus` implements `MotorBus` with the `FeetechBus` semantics:

- It uses suite units and always reports all six joints.
- `read_positions`, `write_positions` and `set_torque` raise
  `RuntimeError("bus not connected")` until `connect()`. (`MockBus` is
  lenient here; this bus is not.)
- `write_positions` takes a partial goal and ignores unknown joints. A
  non-finite value raises `ValueError`, and nothing is applied.
- `write_positions` returns the goal the servo actually tracks, for the
  written joints only. A value is clipped to the effective limits and the
  actuator range. Any other value comes back exactly as written. For example,
  writing 110 to `shoulder_pan` returns 109.99987526, its actuator limit.
- The goal is a register, like Goal_Position. A write with torque off is
  stored, and enabling torque drives the arm to it along the servo profile.
  `SO101Follower.connect` primes the register with the measured pose before
  it enables torque, as it must on hardware.
- `disconnect` disables torque first, as LeRobot's does.

### Time: real time or lockstep

With `SO101_SIM_REALTIME=true` (the default), a daemon thread named
`so101-sim-physics` advances the physics to the wall clock:

- It wakes every 1/120 s. Steps are counted from the instant of `start()`,
  so timer jitter does not accumulate.
- Each wake catches up at most 0.25 s. A longer stall is dropped and counted
  in `status()["dropped_steps"]`.
- The arm tracks the latest goal continuously, the way a servo tracks
  Goal_Position.
- Physics keeps running after every device disconnects, so a de-energized arm
  sags whether or not anyone is reading.

With `SO101_SIM_REALTIME=false` (lockstep), no thread runs. A follower write
with torque on first simulates one control period under the previous goal:
8 substeps of 1/240 s, then the task update and the expert's next target.
Then it latches the new goal. Reads and torque-off writes do not advance
time. Consequences:

- Recorded observation-action pairs are causal: the state read after a write
  is the state the next command acts on.
- A policy loop sees each action one control period later than in real time.
  Evaluate policies with `so101 sim eval`, or in real time.
- The loop must run at 30 Hz. Every write is 1/30 s of simulated time, while
  the `SafetyGuard` rate limit runs on wall time.
- An E-stopped arm does not sag unless someone calls `SimRuntime.advance()`.
- Runs are deterministic: the same seed and the same writes give the same
  states.

### Leader

`SimExpertLeader.read_positions()` returns the scripted expert's current
target. The expert starts when the first leader connects, and restarts at
every episode start. It observes the shared world once per control period.

Teleop with this leader drives the follower through the real
`TeleopController` and `SafetyGuard`, so recording it gives expert
demonstrations. Writes to the leader are accepted as feedback and ignored. If the expert fails, leader reads raise
with its error, and physics keeps running.

### Cameras

`SimCamera` has the interface `CameraManager` uses:

- `read()` returns an H x W x 3 uint8 RGB frame, by default at
  `SO101_SIM_IMAGE_SIZE`;
- `release()`;
- `cam_id`, `width` and `height`.

Frames come from the shared render thread. Each frame is rendered from a
snapshot taken under the runtime lock, so it never shows a half-finished
step.

`make_sim_camera` raises `CameraUnavailableError` the way a missing real
camera does: for an unknown camera, a size out of range, or when no OpenGL
context is available. It waits for the render thread's GL probe no longer
than a frame read does, 5 s.

### Episodes

`on_episode_start(seed=None)`:

- reseeds the task. The default seed is `SO101_SIM_SEED` plus the number of
  episodes started, so the first episode uses the power-on layout;
- moves the arm to the task's start pose;
- keeps the torque state, so a de-energized arm stays de-energized;
- returns the layout, with `instruction` and `episode`.

`on_episode_end()` returns `{"success": bool, "metrics": {...}}`:

- `success` follows the rule of the runner and `so101 sim`: the success
  predicate held at some point in the episode. It is never true after a
  physics fault.
- `metrics` holds the task, instruction, seed, episode, simulated time,
  `success_at_end`, the task's own metrics (such as `first_success_s` and
  `max_lift_m`), the final state, and the fault if there was one. The values
  are JSON-safe, ready for `label_episode`.

### Faults

A fault is either MuJoCo flagging a bad position, velocity or acceleration
(divergence), or a step raising an exception. On a fault the runtime records
it and stops the physics thread. Every later command raises
`RuntimeError("simulation fault: ...")`.

The fault is sticky. To recover, call `reset_sim_runtime()` and reconnect the
devices. MuJoCo also appends the warning to `MUJOCO_LOG.TXT` in the current
directory.

### Integration points

Import `so101.simulation.devices` lazily inside the sim branches, because it
needs the extra. The module docstring has the details.

| Where | What to call |
|---|---|
| `hardware/motors.py:make_bus` | with `backend == "sim"`: `make_sim_bus(name)` |
| `hardware/cameras.py:CameraManager.__init__` and `_open_one` | default `cam_ids` to `settings.sim_camera_names`; open each one with `make_sim_camera(cid)` |
| `services/robot_manager.py:RobotManager.begin_episode` | Wired: under `_io_lock`, `on_episode_start()`, then `follower.guard.seed(follower.read_positions())`. The re-seed is mandatory, because the arm has just moved. |
| `RobotManager.start_recording` and the recording loop | Wired: `environment_state_names` is the task's, and each frame records `follower.environment_state()`. |
| `RobotManager.save_episode` | Wired: the recording loop keeps an environment/predicate snapshot with each successfully written frame; save publishes the episode and its `auto:sim_predicate` outcome under one dataset transaction. |
| `RobotManager._rollout_loop` (optional) | `on_episode_start()` before `follower.connect()`, and `on_episode_end()` at stop. Environment state is wired: `SO101Follower.environment_state()` returns `MujocoBus.environment_state()`, and the policy runner adds it to every observation, so a privileged-state baseline runs under `so101 run` and from the UI. |
| `RobotManager.shutdown` / `reset_manager` | `stop_sim_runtime()` / `reset_sim_runtime()` |
| `services/routers/calibration.py:start`, `cli.py:calibrate` | Refuse the sim backend. The generic calibration path would sample a motionless arm and overwrite the real calibration file. |

## Rendering

Cameras: `front` and `wrist_cam`, any size from 16x16 to 1280x960. One
dedicated thread owns the OpenGL context and renders from a copied snapshot of
the world, so physics never waits for a frame and a frame never shows a
half-finished step. PNG encoding is pure Python; JPEG needs Pillow or OpenCV
and otherwise falls back to PNG.

The render thread draws about 200 frames/s at 640x480 and 215 at 320x240,
shared by every camera, so two cameras read every control period hold
lockstep teleop to about 100 periods/s. The first frame takes about 150 ms:
about 100 ms to start GL, then about 50 ms to make the scene's GL context.
Every frame size renders in that one context, so the first frame at a new
size costs no more than the next (about 5 ms at 320x240). PNG encoding takes
11 ms at 640x480 and 4 ms at 320x240.

Joint states, actions and environment states are bit-exact across runs.
Image bytes are not: the GL driver can draw a few edge pixels one or two grey
levels differently depending on what the context drew before.

The render thread starts with a GL probe, and a stuck GL driver or window
server can hang the probe instead of failing it. No call waits past its own
timeout, which covers the wait for the probe and the frame together. A probe
still running 10 s after the thread started counts as hung: from then on
every call fails at once with that reason. Once a minute, one call waits for
the probe again, within its own timeout. A probe that returns late is used as
soon as it does. At most 8 frame requests wait for the render thread; more
are refused at once. A request whose caller has given up is skipped, so a
burst of abandoned requests does not delay the next one.

## Stops and the rest pose

`simulation/rest_pose.py` holds `MODEL_REST_POSE` (plain data, no MuJoCo
import) and `check_rest_pose`, which tests any pose against the rest-pose
criteria: folded, collision-free when commanded, every joint settling by less
than 3 degrees within 2 s with actuation off, targets at least 5 degrees
(gripper 5 units) inside the limits or at least 2 degrees short of the stop a
joint settles onto, and the gripper nearly closed.

`MODEL_REST_POSE` = (0, -95, 92.8, -85, 0, 5), derived by
`scripts/derive_rest_pose.py`:

| Joint | Target | Limit margin | Settled (torque off, 2 s) | Settle |
|---|---|---|---|---|
| shoulder_pan | 0 | 110 | 0 | 0 |
| shoulder_lift | -95 | 5.00 | -95.37 | 0.37 |
| elbow_flex | 92.8 | 4.03 | 95.30 | 2.65, onto its stop |
| wrist_flex | -85 | 10.00 | -84.96 | 0.04 |
| wrist_roll | 0 | 157.21 | 0 | 0 |
| gripper | 5 | 5.00 | 5 | 0 |

The commanded pose has no contacts; the whole arm stays within 0.166 m of the
base axis and 0.313 m high. In every folded pose gravity loads the elbow with
0.33-0.48 N·m, 6 to 9 times its friction loss, so with actuation off the
forearm always falls until something holds it. Here it comes to rest on the
shoulder bracket at 95.3 degrees, which leaves the elbow target a 1 degree
window (92.3-93.3). Of 20,720 scanned folded poses, 178 pass. The former
bring-up example (0, -95, 90, 70, 0, 0) presses the gripper into the shoulder bracket
and the table in this model, so it fails.

`scripts/measure_sim_stops.py --speed 15` measures the three stop behaviours.
The work pose is (20, -20, 30, 60, 0, 30); zero is all zero with the gripper
at 50.

**E-stop** (actuation cut). Movement after 2 s:

| Start pose | shoulder_lift | elbow_flex | wrist_flex | Other joints |
|---|---|---|---|---|
| zero | 23.7° | 17.7° | 2.2° | 0 |
| work | 10.2° | 11.6° | 0.0° | ≤ 0.1° |
| rest | 0.4° | 2.5° | 0.0° | 0 |

Most of it happens in the first 0.5 s (from zero: 22.2° of the lift's 23.7°).
A test that expects the arm to sag must start away from the rest pose, where
only the elbow moves, onto its stop.

**Halt** (goal set to the present position, torque on). Holding still for
10 s drifts 0.021-0.033 degrees in both servo modes. Halting a joint that is
moving first decelerates it at 2232 deg/s², which is v²/2a, so the joint
keeps going for a while:

| Speed at the halt | 30 deg/s | 90 deg/s | 180 deg/s |
|---|---|---|---|
| overshoot, `sts3215` profile | 0.22° | 1.63° | 6.83° |
| overshoot, `step` | 0.12° | 0.51° | 2.19° |
| final error | ≤ 0.024° | ≤ 0.024° | ≤ 0.018° |

A drift threshold such as "< 1 degree over 10 s" therefore holds with a wide
margin when measured from where the arm settles after the halt, and fails
when measured from the instant of the command for a joint moving faster than
about 70 deg/s (with the `sts3215` profile).

**Collapse** (a 30 Hz goal stream to the rest pose, each joint capped at
15 deg/s, then torque off):

| | from zero | from work |
|---|---|---|
| duration | 6.33 s | 9.67 s |
| peak instantaneous joint speed | 51.5 deg/s | 47.6 deg/s |
| peak speed from 30 Hz position differences | 18.4 (gripper 20.3) | 18.6 (gripper 20.2) |
| peak speed over 0.2 s | 15.9 deg/s | 15.8 deg/s |
| error before torque off | ≤ 0.12° | ≤ 0.11° |
| settle after torque off | elbow 2.48°, others ≤ 0.25° | elbow 2.48°, others ≤ 0.37° |

The goal stream moves each joint in 0.5 degree steps and the servo covers each
step in part of the 33 ms period, so the instantaneous speed reaches about 3x
the cap (up to 56 deg/s in `step` mode) while the average speed stays at it.
Check the cap on 30 Hz position differences with a +40 % margin (worst case
1.35x, the gripper) or over a 0.2 s window with +10 % (worst case 1.06x), not
on instantaneous joint velocity.

## Performance

| | Measured |
|---|---|
| loading and compiling the model, first world in a process | 63 ms |
| physics, contact-free | 190,000 steps/s holding still (about 790x real time at 240 Hz); 50,000 while every joint moves (210x) |
| physics, pick_place scene, holding still | 123,000 steps/s (515x) |
| expert episodes, no cameras | reach 75x, pick_lift 68x, pick_place 42x real time |
| renders | about 200 frames/s at 640x480, 215 at 320x240; PNG encoding 11 ms and 4 ms |
| `so101 sim demos`, no cameras | 16-22x real time |
| `/api/simulation`, MuJoCo engine | 8,700 ticks/s while the arm moves (145x real time at 60 Hz), 31,000 once it has settled (520x); `state` 0.18 ms; a 1 s preview 8 ms; a 320x240 frame 5 ms |
| simulated robot, lockstep teleop | reach 8,300, pick_lift 2,600, pick_place 2,200 control periods/s (275x, 87x, 72x real time) |
| the same on pick_lift, reading two 320x240 cameras every period | 100 control periods/s (3.4x real time), bound by rendering |
| simulated robot, real time | 0.9992-1.0004x the wall clock, no dropped steps, 1-3 % of one core |

A moving arm costs about four times as much as one holding still. While its
motion profile is at rest the servo model hands MuJoCo all the substeps in one
`mj_step` call; while any joint moves, every substep goes through the profile
in Python.

Demo recording is bounded by the recorder, which opens the episode's frame
file once per frame (about 1 ms). The numbers are typical of three runs on the
arm64 Mac while other jobs kept it busy (load average 20-27 on 10 cores), so
an idle machine should do somewhat better.

## Memory

RSS measured on the arm64 Mac, adding one piece at a time in a fresh process:

| Step | RSS |
|---|---|
| Python with `so101.simulation.devices` imported (MuJoCo, numpy) | 65 MB |
| one `SimRuntime` (scene compile, cached template, world) | +86 MB |
| the expert leader (its own model copy for inverse kinematics) | +15 MB |
| two cameras, first frame each, at any sizes (the scene's GL context, uploaded meshes, render state) | +122 MB |
| 60 more frames and 300 lockstep teleop writes | +0.2 MB |

A compiled model is 14.6 MB (`mj_sizeModel`), almost all of it meshes
(18 meshes, 26,142 vertices); `MjData` is 0.2 MB. Every holder of a model
keeps its own copy:

- each `PhysicsWorld`: the simulated robot's world and the `/api/simulation`
  engine's world (the MuJoCo engine added 37 MB);
- the expert's inverse kinematics;
- each render scene state, at most 4 on the render thread;
- `compiled_scene`, which caches at most 8 templates. A randomized seed that
  rescales an object makes a new template, so many resets cycle through the
  cache instead of growing it;
- each trajectory preview, for its duration (about 15 MB, freed afterwards).

The render thread keeps one GL context per scene, whatever frame sizes are
asked for, and at most 2; a third scene closes the least recently used. Each
context holds the scene's uploaded meshes and a multisampled 1280x960
offscreen buffer, about 65-70 MB on this Mac. The render thread therefore
holds at most about 140 MB of contexts plus its 4 scene states. Cycling 9
frame sizes for 1,800 frames kept RSS flat.

## Portability

The simulation code follows the suite's cross-platform rules:

- Text files are read and written as UTF-8 explicitly, whatever the locale.
- Bundled assets (the model, the scenarios) load relative to the package,
  never the working directory. The model ships in the wheel.
- There are no POSIX-only calls and no hard-coded `/tmp` or `/dev` paths.
- Helper scripts are Python.

Platform notes:

- **Rendering.** The code never sets `MUJOCO_GL`. macOS uses CGL and needs no
  display. Linux without a display needs `MUJOCO_GL=egl` (GPU) or
  `MUJOCO_GL=osmesa` (CPU), set before the process starts. Windows uses WGL
  and needs a desktop session. Only macOS was tested. Without a GL context,
  `/render` answers 503 and `make_sim_camera` raises
  `CameraUnavailableError`; everything else runs.
- **Timing** uses `time.monotonic()`. On Windows with Python 3.12 its
  resolution is 15.6 ms (Python 3.13 uses `QueryPerformanceCounter`, about
  1 µs). The real-time physics thread then advances in bursts of up to
  4 substeps instead of 2 per wake. Steps are counted from a fixed anchor, so
  the average rate stays exact.
- **Install path.** MuJoCo opens the vendored MJCF and meshes by file path. A
  Windows install path with non-ASCII characters (for example in the user
  name) was not tested; if the model fails to compile there, install into an
  ASCII path.
- **`MUJOCO_LOG.TXT`.** MuJoCo writes its warnings, for example on
  divergence, to `MUJOCO_LOG.TXT` in the current working directory.

## Boundaries

- **Servo model.** The actuator parameters come from the Menagerie model, a
  community derivation, not an identification of this suite's arms. The
  servo is an ideal PD loop with torque saturation and an acceleration
  profile. Not modelled: gear backlash, the position dead zone, firmware
  P/I/D gains, current, voltage, thermal and overload protection, encoder
  quantization (0.088 degrees per step) and bus latency.
- **Friction** is one Coulomb friction loss plus viscous damping per joint;
  no stiction or Stribeck curve. Object friction (1.0, 0.03, 0.003) and density
  (500 kg/m³) are fixed.
- **Contacts** are MuJoCo's soft contacts on the model's collision meshes:
  jaws penetrate the cube by up to about 1 mm and a held cube slips about
  0.2 mm over 1.5 s.
- **Images are synthetic**: no noise, blur, exposure changes or lens
  distortion, and the `front` camera is not calibrated to a real rig.
- **The expert uses privileged state**: the true cube pose. Its success says
  a task is feasible for this arm, not that a camera-based policy can do it.
- **Sim success is not real success.** Use simulated evaluation as a
  regression and smoke signal. Real acceptance stays a hardware-in-the-loop
  test.
- **Determinism** holds for the same seed, MuJoCo version and machine. Image
  bytes vary by a few grey levels.
- **Rendering** was tested only on macOS (CGL). Linux headless (EGL,
  OSMesa) and Windows (WGL) were not.

## Reproducing the numbers

| Numbers | Command (`PYTHONPATH=src`, with the `sim` extra) |
|---|---|
| Expert success rates | `python scripts/measure_sim_expert.py --episodes 100` (`--noise-deg 1 --waypoint-noise-m 0.004`) |
| Performance, rendering | `python scripts/measure_sim_perf.py` (`--sections`, `--out perf.json`) |
| Timestep table | `python scripts/measure_sim_timestep.py --episodes 20` |
| Stops | `python scripts/measure_sim_stops.py --speed 15` |
| Rest pose | `python scripts/derive_rest_pose.py --check` (scan without `--check`) |
| FK golden fixture | `python scripts/generate_fk_golden.py` |
| Vendored model | `python scripts/vendor_so101_model.py` |

Tests: `tests/test_sim_physics_*.py`. The ones that need MuJoCo skip without
the extra; `test_sim_physics_recorder.py`, `test_sim_physics_preview.py` and
the kinematic half of `test_sim_physics_api.py` always run.
`test_sim_physics_devices.py` covers the simulated robot devices;
`test_sim_physics_engine.py` and `test_sim_physics_api.py` cover the MuJoCo
`/api/simulation` engine, the engine selection and the render endpoint;
`tests/test_simulation_v01.py` covers the kinematic engine.

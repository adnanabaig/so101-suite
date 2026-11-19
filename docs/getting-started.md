# Getting started

A guided walkthrough of the full lifecycle. Everything here works in
**mock mode** (`SO101_MOCK=1`) with no hardware. Real hardware uses the same
UI through a non-reloading Suite server; owned real motion is not run from the
standalone CLI.

## 1. Install

You need Python 3.12 or 3.13 and Node 24 LTS ([Prerequisites](../README.md#prerequisites)).

```bash
make bootstrap          # create .venv + install backend and frontend
# then, on the robot host:
make install-robot      # adds LeRobot, OpenCV, Feetech SDK, PyTorch
source .venv/bin/activate   # puts so101 on PATH
```

## 2. Start the stack

```bash
SO101_MOCK=1 make suite # built UI + API at http://localhost:8000
```

Open the UI. The **Dashboard** shows robot status (mode, connection, e-stop)
and a live video tile. **Digital Twin**, **Quest & Hands**, **Diagnostics**, and
**Agent Lab** exercise their complete mock/preview contracts without hardware.

For live development, run `SO101_MOCK=1 make api-dev` and `make ui` in separate
terminals, or `SO101_MOCK=1 make dev` for both. `api-dev` restarts the server on
every source change, and each restart is a full shutdown: the arm Collapses to
its rest pose with the motors off, and whatever was moving it stops. The E-stop
latch and any fault survive the restart. A real arm would collapse on every
file save, so it refuses to run without mock hardware. With an arm connected,
use `make suite`, `make api` or `so101 serve`.

## 3. Calibrate (real hardware)

UI → **Calibration** → pick an arm → *Start*, then move each joint through its
full range while the progress bar fills. Repeat for follower and leader. In
mock mode this completes instantly with defaults.

## 4. Collect data

UI → **Teleop + Record**:
1. *Start teleop* — the follower mirrors the leader; watch live joints + video.
2. Enter a dataset id (e.g. `you/pick_place`) and *Start recording*.
3. For each demo: *Start episode* → perform the task → *Save episode*.
4. Aim for ~30–50 clean episodes for a simple ACT task.

Leader teleop and recording are internally supervised local loops. Browser,
Quest, camera, and agent motion use a separate short-lived lease, explicit arm,
deadman, sequence, timestamp, and watchdog contract; the UI never mixes the two
ownership modes.

Mock/simulation CLI equivalent (real recording stays in the Suite UI/API):
```bash
so101 record you/pick_place --episodes 30 --episode-seconds 20
```

## 5. Train

UI → **Training** → choose a dataset + policy (ACT is the default), set steps,
*Launch*. Watch the live loss curve and logs. Training is hardware-free and
may use the CLI:
```bash
so101 train you/pick_place --policy act --steps 100000
```

## 6. Serve & evaluate

UI → **Models** → pick a checkpoint → *Start rollout* to run autonomously
(live video + commanded joints), or UI → **Eval** to run N scored rollouts and
get a success rate. Mock/simulation CLI equivalents are:
```bash
so101 run   <model_path> --duration 30
so101 eval  <model_path> --episodes 10
```

## Safety reminder

The red **E-STOP** button (top bar) latches, stops the active loop, and makes a
best-effort torque-off. Reset takes two deliberate clicks and never happens as
a side effect of reconnecting. On real hardware, also keep a hand near the
physical power switch.

Before connecting a real arm, read [Suite v0.0](SUITE_V0.md) and run the
**Diagnostics** checklist. Run the full software gate, `make ci`, before
treating a build as ready.

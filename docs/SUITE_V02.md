# SO-101 Suite v0.2

v0.2 turns the v0.1 workstation into something you can hand to one machine and
run: real physics, three named stops, a hardware bring-up kit, a CI quality bar,
Intel RealSense cameras, and an Ubuntu installer for the target laptop. The
rule from v0.0 still holds: only deterministic host code owns a motor bus or
issues a validated command.

Target platform: ASUS ROG Zephyrus G15 (RTX 3070 Laptop GPU) on Ubuntu 24.04
LTS, with both arms and the RealSense plugged into it. See
[install.md](install.md). macOS and Windows paths exist but are unverified.

## v0.2.1: review fixes

A review of v0.2.0 returned "request changes" (29 findings: 4 high, 15
medium, 10 low), and a follow-up pass found one more high. v0.2.1 fixes every
finding that could be verified; what is left is listed in
[the v0.3 backlog](plans/2026-10-07-v0.3-backlog.md). Each fix has at least one
test that fails on v0.2.0. The ones that matter most on the G15:

- **Standalone commands honour the E-stop.** `so101 teleop`, `record`,
  `replay`, `run`, `eval` and `calibrate` refuse while an E-stop, fault or
  unconfirmed torque-off is saved, or while a server is running. On the real
  backend the server holds canonical, host-global locks for the configured
  motor devices, so a command from another data folder or checkout and the
  server never drive the arm at once. A latch that could not be saved leaves a
  marker, so a restart still comes back faulted.
- **Collapse never speeds up.** A late control step no longer makes the descent
  catch up; each step is capped at the trajectory speed. Arming re-checks for a
  stop under the bus lock before any torque write.
- **The first install finishes.** The launcher is written before an advisory
  `so101 doctor` (some checks pass only after logging out and back in); a
  re-run replaces a PyTorch build that doesn't match the driver; only a fresh
  wheel of the source version installs; `.env` files are owner-only and
  written atomically; `bash install.sh` works from a FAT USB stick.
- **Labels reach training.** Outcome labels are stored for LeRobot datasets
  (real recordings and `so101 sim demos`) in a suite-owned
  `so101_outcomes.json`, and LeRobot training learns only from the selected
  episodes (`kept` by default: success and unlabeled). A label that can't be
  stored stops the run.
- **The HIL runbook tests the stops.** `START_HERE.md` runs the gate,
  verification and stop groups, with a hand on the power switch for the stop
  group. The unreleased G15 candidate below strengthens this ordering and its
  evidence requirements.
- **Simulation is diagnosed as simulation.** `so101 doctor` checks the
  simulated arm under `SO101_BACKEND=sim` and records the backend; a sim or
  mock preflight or calibration never counts for the real arm; recording under
  sim resets the scene, records the object state and labels each episode.
- **Durability.** Training-job records, recordings, the run manifest and
  `sim_demos.json` are written atomically and fsynced; a trainer left running
  by a crashed server is identified and stopped on restart; closing the
  launcher's terminal shuts the server down gracefully.
- **Smaller fixes.** Preflight runs by POST (a website can no longer trigger
  it); stop requests read at most 4 KiB; dataset ids that differ only in case
  or look like sub-routes are refused; dataset locks hold across processes;
  the UI build is stamped and the wheel refuses a stale UI; the documented rest
  pose is collision-free; the smoke script refuses anything but a fresh mock
  server.

## v0.2.2 release candidate 5

The current development branch identifies itself as `0.2.2rc5`; it is not the
tagged v0.2.1 release and is not final v0.2.2. Its generated
`START_HERE.md` establishes the physical workspace, supply identities,
mechanical catch and reachable cutoff before first arm power. It completes
calibration and the rest pose before server bus ownership, then runs gate, stop
and verification in that safety-first order. A missing stop-status field, a
skipped or warned stop probe, or a pre-contract attestation blocks both a
passed run and `hil_evidence=true`; observation cannot replace machine stop
evidence. These
claims remain release-candidate behavior until the full software gate and G15
HIL evidence pass at one uniquely versioned commit.

The RC also closes residual release-review gaps: fallback episodes use a
recoverable two-object publication journal; training terminal states are not
exposed before durable storage; dataset and trainer locks fail closed; the
installer verifies exact source, dependencies and a current UI before using an
immutable wheel copy; custom data homes survive generated launchers; diagnostic
and installed-E2E evidence redact or omit credential-bearing data; simulated
camera discovery cannot touch host devices; privileged policies must match the
active simulation task; built-UI path checks and Python 3.13 are required by
the hosted CI configuration; and every critical stop/reset/recovery function
has a coverage floor. None of
those software claims substitutes for the target evidence below.

RC3 additionally refuses to call torque off after a connect/handshake unless
every known motor confirms the register read-back, persists calibration-time
unconfirmed torque as a fault, always attempts RobotManager shutdown even when
another teardown callback fails, and caps globally parsed JSON request bodies
at 1 MiB. RC4 writes a device-global energized intent
before standalone or RobotManager follower torque-enable, retains it across a
Halt or ungraceful process death, and makes every data directory sharing those
physical ports refuse until aggregate confirmed torque-off or acknowledged
power cut clears it. A failed marker write prevents server torque-enable and
latches a visible fault; mock and sim never create host-hardware intent.
Standalone teleop, replay and policy completion write a fresh-position Halt
and close only the host port instead of silently disabling torque. Safety rows
and fallback markers carry backend plus follower/leader logical identity;
legacy unscoped state migrates conservatively to real hardware, so mock or sim
rehearsals cannot clear a real E-stop, fault or torque uncertainty.

RC5 follows the final RC4 adversarial pass. It keeps stop routes unconditional
even when an installed-E2E token is present, binds G15 software evidence to
ASUS/GA503 DMI identity, rejects unsafe non-finite motion timing, makes real
motion server-only because an exited standalone process cannot enforce the
watchdog, stop listener or timed Halt-to-Collapse contract, retains E-stop and
fault state across data directories,
and bounds ordinary host-global journal growth. Direct Python requirements now
carry compatible upper bounds; the target G15 install evidence must still pin
the resolved environment before final release.

## Added in v0.2

- **Three stops** ([control-plane-safety.md](control-plane-safety.md)):
  - **E-stop** cuts torque on every motor of every bus at once and latches. It
    answers within a second even when every request worker is busy, persists
    across restarts, and needs a deliberate reset (`operator_id` + `confirm`).
  - **Halt** freezes the arm in place with torque on. The gripper never opens
    further. A halt holds for 60 s (`SO101_HALT_HOLD_SECONDS`), or 5 s once the
    operator's browser stops supervising (`SO101_SUPERVISION_GRACE_SECONDS`),
    then becomes a Collapse.
  - **Collapse** lowers the arm to its rest pose at no more than 15 °/s per
    joint (`SO101_COLLAPSE_SPEED_DEG_S`), bounded at 12 s, then turns torque off
    and reads it back. Timeout, stall, bus error or guard rejection fall back to
    an immediate torque cut with a fault.
  - Episode ends, agent skills, deadman release, the command watchdog and lease
    expiry Halt; owner disarm, service shutdown and a critical servo Collapse.
    Nothing de-energizes the arm at the end of an operation any more.
  - Three-state torque truth (`on`, `off`, `unconfirmed`): the suite reports a
    torque-off it could not confirm instead of claiming the arm is safe.
- **MuJoCo physics** ([simulation.md](simulation.md)): `SO101_BACKEND=sim` runs
  the control plane against a physics model of the SO-101 with simulated
  cameras, seeded reach / pick-lift / pick-place tasks, a scripted expert,
  labelled demo recording (`so101 sim demos`, and recording from the UI: each
  episode resets the scene with the next seed, records the object state and
  is labelled by the task predicate) and policy rollouts. Under `sim` the
  suite refuses to open any real serial port or camera.
- **Hardware bring-up kit** ([bring-up.md](bring-up.md)): `so101 preflight`,
  bus ownership checks, a recorded rest pose (`so101 rest-pose record`) and a
  guided hardware-in-the-loop checklist (`so101 hil run`). With a server
  running, preflight borrows the idle bus read-only from the robot manager
  instead of skipping its bus checks.
- **Servo health monitoring:** the server polls temperature, voltage, load,
  current and the status register once or twice a second. A critical reading
  Collapses the arm and latches a fault; warnings show in status and events;
  `GET /api/robot/health` has the readings. Polling waits at most 50 ms for the
  bus and skips rather than delay a control loop.
- **Data, training and evaluation** ([learning.md](learning.md)): missing
  datasets and models fail closed; outcome labels record who set them (human
  labels win); LeRobot training is bounded, runs one job at a time and keeps
  job records across restarts; VLA policies run with their training task and
  fps; evaluation reports carry Wilson 95 % intervals. A numpy-only
  behaviour-cloning baseline trains without PyTorch.
- **Intel RealSense** ([realsense.md](realsense.md)): color streams for live
  view, recording and policy input; opt-in depth recorded as 16-bit depth maps
  (training leaves depth out unless asked); `so101 find-cameras`; USB 3 and
  udev checks. Camera credentials never reach dataset feature names, status,
  events or logs.
- **Operator UI** ([operator-ui.md](operator-ui.md)): E-stop alone at the end of
  the safety bar with hold-to-reset; Halt and Collapse buttons; a countdown when
  an unsupervised Halt is about to Collapse; freshness-aware status that a stale
  WebSocket can't override; owner-scoped deadman and leases; error boundaries;
  keyboard shortcuts (`?`); a first-run setup checklist; an accessibility pass;
  a badge that says whether the arm is real, simulated or mock; Tailwind CSS 4.
- **Security** ([control-plane-security.md](control-plane-security.md)): an
  Origin and Host guard on every state-changing request (stops stay exempt),
  strict JSON and request models that reject NaN and Infinity, lease ids
  returned only to their owner, operator approval for agent skills.
- **Install and handoff** ([install.md](install.md)): `install/install.sh`
  picks the PyTorch build from the NVIDIA driver, sets up udev names for the
  arm adapters (`/dev/so101-follower`, `/dev/so101-leader`, the Linux default
  ports) and the RealSense rules (every sudo step shown and confirmed), and
  adds an app-menu launcher. Settings live in the data folder's `.env`
  (`~/.local/share/so101`), so `so101` behaves the same from any directory.
  `so101 doctor` checks the GPU, rendering, serial permissions, disk space and
  laptop sleep. `make handoff OUT=<dir>` builds the folder to copy to the
  target machine. Python 3.12 and 3.13 are supported.
- **Quality bar** ([../CONTRIBUTING.md](../CONTRIBUTING.md)): `make ci` runs
  Ruff, formatting, mypy, the test suite with per-function coverage floors on
  the control plane, a live HTTP smoke test, the UI's lint, type check, tests,
  build and `npm audit`, and a wheel build that must contain the UI.

## v0.2.1 verification status

The software gate (`make ci`) passes at the tagged v0.2.1 release commit. The
counts below describe that tag, not the unreleased G15 candidate:

- Ruff, formatting and mypy clean.
- 2,473 backend tests pass (15 skipped: they need LeRobot or another OS), with
  17 coverage floors on the control-plane, safety and camera paths.
- 634 UI tests, lint, type checks and a production build; `npm audit` reports 0.
- A live HTTP smoke test of 18 checks against a mock server, including a Halt,
  and an owner Collapse that ends at rest with torque off.
- A wheel that carries the UI and installs into a clean venv.

This build machine is a Mac with no arm, camera, Linux or GPU attached.

Measured in MuJoCo through the robot manager:

| Stop | Result |
| --- | --- |
| E-stop | the elbow sags about 34° in 2 s (motors off, as intended) |
| Halt | 0.02° drift over 10 s on a still arm; about 1° overshoot halting a 47 °/s move, then 0.00° |
| Collapse | 11.2 s to rest, commanded peak 15.0 °/s, within 10 % of the cap over 0.25 s windows, 0.25° from rest at torque-off |

Within each 1/30 s control step the servo reaches its new goal at its own speed,
so instantaneous joint speed briefly peaks near 3× the cap. Capping that needs
the servos' own velocity and acceleration registers, which v0.2 does not set.

## Unverified until run on the target

- Everything on a real SO-101: calibration, the three stops (run the HIL
  checklist), servo health thresholds and how long a health read takes on the
  real bus, torque read-back.
- The Ubuntu install path: apt, udev, dialout, the NVIDIA driver and CUDA build
  selection, the desktop launcher. This build machine had no Linux.
- RealSense on hardware, and depth on stock Ubuntu 24.04 (kernel 6.8, V4L2
  backend, no patched modules).
- Depth through real LeRobot recording and training, and training launched
  as `python -m lerobot.scripts.lerobot_train` (checked against the 0.6.1
  source, not run here).
- MuJoCo rendering on Linux EGL (tested on macOS only).
- macOS and Windows installers.

Send `so101 doctor --json` and the HIL report from the first run back for
triage.

## Explicit boundaries

- A software stop is not a certified safety device. If the host process dies
  (crash, kill, power loss) the STS3215 servos keep holding their last goal;
  only the physical power switch stops them. Keep a hand near it.
- The RX 5700 XT desktop has no supported PyTorch GPU path; it runs PyTorch on
  the CPU (fine for operating, recording and simulation, not for training).

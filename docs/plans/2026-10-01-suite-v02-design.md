# SO-101 Suite v0.2 design

Date: 2026-10-01 · Branch: `v0.2` (baseline `1d3a14a` = v0.1 as delivered)

## Goal

Make the whole workstation loop real without an arm, ready the suite for the
day an arm is attached, and raise the quality bar so regressions are caught by
machines instead of people. Four streams, plus every confirmed audit bug:

1. **Physics simulation** — MuJoCo running the official SO-101 model behind the
   existing contracts, so record → train → eval produces measured success rates
   with no hardware.
2. **Hardware bring-up kit** — guided preflight, servo health telemetry, and an
   HIL checklist runner that records evidence; adapters re-checked against the
   LeRobot 0.6.1 source.
3. **Quality bar + CI** — GitHub Actions release gate, frontend lint/tests, API
   smoke in CI, coverage floors on safety-critical modules.
4. **Operator UX** — first-run onboarding, keyboard shortcuts, accessibility,
   bundle splitting, and a shared component set.

Non-goals: sim-to-real transfer claims, certified safety, cloud/fleet features,
Quest actuation, WebRTC.

## Invariants that do not change

- `RobotManager` is the only owner of a motor bus. Every motion path, including
  simulated ones, goes through leases, arm/deadman rules and `SafetyGuard`.
- Capability truth: `GET /api/capabilities` reports what is actually active.
  Simulation never presents itself as hardware; rehearsal never counts as HIL.
- Real mode fails closed. Nothing silently substitutes mock or sim data.

## 1. Physics simulation

### Model source

MuJoCo Menagerie `robotstudio_so101` at commit `4d038b3` (2026-09-29),
Apache-2.0. Menagerie derived it from The Robot Studio's
`so101_new_calib.xml` and added primitive collision geometry, contact
parameters tuned for gripping, `implicitfast` integration, and a wrist camera.
We prefer it to the raw upstream MJCF because the upstream file collides with
mesh convex hulls, which grip poorly.

The meshes total 17 MB. A reproducible script, `scripts/vendor_so101_model.py`,
copies the MJCF and does the following to the meshes:

- Collision meshes are kept byte-exact.
- Visual meshes are decimated to keep the vendored package under about 3 MB.
- A `PROVENANCE.md` records the source URL, commit, input/output SHA-256s and
  the modifications, as Apache-2.0 requires.

Mass and inertia are explicit `<inertial>` elements, so decimation does not
change dynamics.

### Units and limits (one module, one place)

- Arm joints: suite degrees ↔ MJCF radians. Both use the middle-of-range zero
  ("new calibration") that LeRobot uses with `use_degrees=True`. Sign agreement
  with a calibrated real arm is an HIL check, not an assumption.
- Gripper: suite 0 (closed) to 100 (open) maps linearly to the MJCF range
  −0.1745 to 1.7453 rad. Commanding 0 squeezes slightly past contact, which
  produces grip force.
- Effective limits are the intersection of `SafetyGuard.DEFAULT_LIMITS` and the
  MJCF joint ranges. For example, elbow_flex becomes ±96.8° rather than ±100°.

### Components (`src/so101/simulation/`)

| Piece | Responsibility |
| --- | --- |
| `PhysicsWorld` | Owns `MjModel`/`MjData`. It provides seeded reset, `step`, observation (joints, end-effector pose, object poses, contacts), and torque-off as disabled actuation, so the arm sags as real servos do. |
| Tasks | `reach`, `pick_lift` and `pick_place`. Each has a seeded reset distribution, a success predicate and diagnostics. |
| Scripted expert | Damped-least-squares IK on the `gripperframe` site, plus a phase machine (approach → descend → close → lift → place). It emits suite-unit joint targets at 30 Hz. |
| Renderer | `mujoco.Renderer` on one dedicated GL thread, rendering from a snapshot copy of state. It degrades to "unavailable" if GL is unavailable, and never crashes the API. |
| `PhysicsSimulationEngine` | Serves the existing `/api/simulation` contract (state, telemetry, targets, step, play, pause, reset, trajectory preview) and adds a camera render endpoint. The kinematic engine stays as the fallback. |
| Devices | `MujocoBus` implements the `MotorBus` protocol. `SimCamera` matches the camera interface. `SimExpertLeader` is a leader bus whose "hand" is the scripted expert. |

### Two ways to run physics

1. **Robot backend** (`SO101_BACKEND=sim`). `make_bus` and `CameraManager`
   return the sim devices. A real-time physics thread tracks the latest goal,
   just as a servo tracks its Goal_Position register. Teleop, recording,
   rollout, eval, the UI, the CLI, leases, E-stop and the guard all run
   unchanged. Recording with the expert leader produces demonstrations through
   the same code path as real teleop collection. The world reseeds at each
   episode start through a small optional hook.
2. **Batch** (`so101 sim demos`, `so101 sim eval`). The same world, guard and
   dataset writer, but deterministic and faster than real time, without the
   manager.

`SO101_MOCK=1` keeps its meaning (`mock`). A contradictory `SO101_MOCK=1` plus
`SO101_BACKEND=real` is a configuration error. Mock-only paths, such as
synthetic training and moving mock policies, stay mock-only.

### Data, learning, evaluation

- Sim datasets carry the same features as real captures. They also include a
  documented sim-only `observation.environment_state` feature (object pose),
  which is LeRobot's convention for privileged sim state.
- Episodes are auto-labeled with outcomes from the task predicate.
- A torch-free baseline policy (numpy behavior cloning) closes the loop on this
  Mac and is labeled as a baseline. The LeRobot ACT/Diffusion paths are
  unchanged when the `[train]` extra is installed.
- Sim eval reports the success rate with its seed list and a Wilson 95%
  interval. It is never presented as real-world success.

## 2. Hardware bring-up kit

- **Preflight** (`so101 preflight`, `GET /api/preflight`, UI wizard). Ordered
  checks, each with pass/warn/fail/skip and remediation:
  - environment and the LeRobot 0.6.1 API surface
  - serial ports and permissions
  - motor ping and model number for IDs 1–6
  - supply voltage band (configurable for 7.4 V or 12 V kits)
  - temperatures, calibration identity and ranges
  - torque-off read stability, camera frames, disk space
  - the software E-stop latch

  Mock and sim runs are labeled rehearsals.
- **Servo health.** Low-rate (about 1–2 Hz) sync reads of Present_Temperature,
  Present_Voltage, Present_Load, Present_Current and Status, using the names
  verified in the LeRobot 0.6.1 Feetech table.
  - Raw values plus converted units, with the unit sources documented.
  - Configurable warn and critical thresholds. Critical triggers a soft-stop
    fault.
  - Exposed in WebSocket telemetry and `/api/robot/health`.
  - Mock and sim values are synthetic and carry a `source` field. Sim derives
    load from actuator force.
- **HIL runner** (`so101 hil run|status|report`, UI page). Steps come from the
  existing SUITE_V0 HIL gate. Each has instructions, an automatic probe where
  possible, and operator attestation. Evidence (telemetry, health, an audit
  excerpt, an optional frame) is written under `data/hil/<run>/` with a JSON
  report, a Markdown summary and a SHA-256 manifest. Only real-backend runs can
  be marked as HIL evidence.

## 3. Quality bar + CI

- `.github/workflows/ci.yml` runs these jobs:
  - backend: Ruff, format, mypy, pytest with coverage, sim tests using EGL or
    OSMesa
  - frontend: `npm ci`, lint, test, build, audit
  - API end-to-end smoke against a mock server
  - wheel build and contents check
- `make ci` mirrors the workflow locally. The workflow cannot run on GitHub
  from here because nothing is pushed. That is stated, not glossed over.
- Coverage floors are enforced per file for `safety.py`, `robot_manager.py`,
  the control and lease paths, and the sim unit mapping. They are set from
  measured values and ratcheted at the end.
- Frontend: ESLint flat config (typescript-eslint, react-hooks) and Vitest with
  Testing Library on jsdom. Coverage targets the safety UI: E-stop, two-step
  reset, deadman, freshness, shortcuts and the API client.
- Fix the Ruff `py310` / mypy 3.12 target mismatch.

## 4. Operator UX

- **First-run onboarding** checklist, derived from capabilities, preflight and
  calibration state. Dismissible and persisted.
- **Keyboard shortcuts** with a `?` overlay.
  - `Escape` always triggers E-stop. Our own dialogs therefore do not use
    Escape to close, so the key never has a second meaning in-app. Accidental
    stops cost downtime, never safety.
  - Navigation and jog keys apply only while the deadman is held.
- **Accessibility:** landmarks, labels, visible focus, `aria-live` safety
  announcements, contrast, reduced motion, full keyboard operability.
- **Bundle:** split the 869 KB twin chunk (three, R3F, drei) and keep heavy
  libraries on their routes. No chunk over 500 kB.
- **Shared primitives** (Button, Card, StatusPill, Field, Dialog, Tabs,
  EmptyState, Toast) on Tailwind tokens, adopted page by page.
- **Wave-2 screens:** physics Simulation page (camera view, task, expert demo,
  record, eval results), preflight wizard, servo health panel, HIL runner.

## 5. Stop categories (user decision, 2026-10-01)

Three named stops replace v0.1's single "disarm and de-energize" path, where
the arm dropped under gravity. The names are user-facing.

| Stop | Meaning | IEC 60204-1 analogue |
| --- | --- | --- |
| **E-stop** | Kill all motors now. Latched; deliberate reset (C2). | Category 0 |
| **Halt** | Freeze in place. Motors stay on and hold position. | Category 2 |
| **Collapse** | Slow fall: lower to the rest pose at a capped speed, then motors off. | Category 1 |

### E-stop

`POST /api/robot/estop`. Its shape is unchanged and it is exempt from the
Origin guard.

- It disables torque on every motor on every bus the manager owns before doing
  anything else: the follower, the leader if connected, and sim actuators.
- It never waits for a trajectory.

### Halt

`POST /api/robot/halt {"operator_id"?, "reason"?}`. It is exempt from the
Origin guard because it can only stop motion.

- **Stopping.**
  - Every motion source stops at the next tick: teleop, recording, rollout,
    eval, agent execution, tracking and jog.
  - Command authority is disarmed. Re-arming uses the normal lease → arm flow,
    which must not toggle torque or make the arm jump.
- **What is held.**
  - Arm joints: Goal := Present, freshly read.
  - Gripper: it never opens further, so a held object is not dropped. It keeps
    at most a small squeeze so an existing grasp holds:
    `goal = clamp(last_goal, present − grip_hold_margin, present)`.
- **Motors already off.** Halt never energizes them. It returns ok as a no-op.
- **Hold timer.**
  - A halt holds for at most `SO101_HALT_HOLD_SECONDS` (default 60). When that
    expires, the halt becomes a Collapse.
  - A repeat halt renews the timer, but only if it comes from an operator with
    a fresh supervision heartbeat, in a request that passes the Origin guard.
  - While supervision is lost, the hold lasts only
    `SO101_SUPERVISION_GRACE_SECONDS` (default 5).
- **Fallback.** If positions cannot be read or goals cannot be written within
  the I/O timeout, torque goes off and a fault latches (category 0 fallback).

### Collapse

`POST /api/robot/collapse {"operator_id"?, "reason"?}`. It is subject to the
Origin guard because it commands motion.

- **Descent.**
  - Every motion source stops. The arm then moves from Present to the rest pose
    in joint space.
  - Every joint is capped at `SO101_COLLAPSE_SPEED_DEG_S` (default 15).
  - The descent passes through SafetyGuard joint limits but bypasses the lease,
    because it is a stop, not a command.
  - The gripper follows the Halt grip rule.
- **Bound and finish.** The descent is bounded by
  `SO101_COLLAPSE_TIMEOUT_SECONDS` (default: distance / speed + 2 s, at most
  12 s). It ends with torque off and a read-back (C2 three-state torque truth).
- **Fallback.** Torque goes off immediately, with `fault` or
  `torque_unconfirmed` set per C2, on any of these:
  - timeout
  - stall: no progress for 0.5 s with error above 3°
  - bus error
  - guard rejection
- **Latching.** A collapse is not latched unless a fault triggered it. It is a
  no-op when the motors are already off.

### Precedence

- E-stop preempts everything instantly.
- A halt during a collapse freezes the descent (operator override) and restarts
  the hold timer.
- A collapse during a halt proceeds.
- Nothing starts while E-stopped.

### Rest pose

The rest pose is resolved in this order:

1. The pose recorded during bring-up (`so101 rest-pose record`). It is bound to
   the follower ID and calibration identity.
2. `SO101_REST_POSE`.
3. The default derived from the official model. It must be:
   - folded
   - collision-free
   - at least 5° inside the effective limits
   - gravity-stable: it moves less than 3° in 2 s with actuation off in MuJoCo

The pose is always clamped to the SafetyGuard ∩ MJCF limits. Preflight warns
on the real backend when no pose has been recorded.

### Which stop fires when

| Trigger | Stop |
| --- | --- |
| E-stop button, Escape, `so101 estop`, `make estop` | E-stop |
| Bus I/O failure, hung I/O, torque read-back failure | E-stop-equivalent fault |
| Halt button, `so101 halt`, `make halt`, legacy `POST /api/robot/stop` | Halt |
| Deadman release, command watchdog, lease expiry or release, error-path cleanup | Halt |
| Agent skill completes, is cancelled or times out | Halt; the final pose is held, then the hold timer applies |
| End of teleop, recording, rollout, eval episode or tracking (`soft_stop()`) | Halt |
| Collapse button, `so101 collapse`, `make collapse` | Collapse |
| `POST /api/control/disarm {lease_id}`, owner-scoped ("done, power down") | Collapse |
| Halt hold timer expires, or supervision lost past the grace period | Collapse |
| Service shutdown (SIGTERM) | Collapse, bounded, then torque off |
| Servo health critical | Collapse + fault |

### Visibility

`GET /api/robot/status` and every telemetry frame gain these fields:

- `torque_state: "on" | "off" | "unconfirmed"`
- `stop_state: "none" | "halted" | "collapsing"` (E-stop and fault keep their
  own fields)
- `halt: {since, reason, hold_expires_at, server_time} | null`
- `collapse: {since, reason, progress, eta_s} | null`

Events: `safety.halt`, `safety.collapse.start`, `safety.collapse.done`,
`safety.collapse.fallback`, `safety.estop`.

### Staging

- **Wave 1.**
  - control-hw makes E-stop kill all motors.
  - control-hw routes every stop call site through one dispatcher, tagged with
    the kind it gets in the table above. Those sites keep v0.1 behavior until
    wave 2.
  - The UI, the bring-up kit (rest pose, HIL steps), the sim (model rest pose,
    stop verification hooks) and the Make targets build against this contract
    in parallel.
- **Wave 2.** A dedicated pass implements Halt and Collapse in `RobotManager`
  on the merged control plane. Physics tests then verify the three stops under
  gravity:
  - E-stop sags.
  - Halt drifts less than 1° over 10 s.
  - Collapse stays within its speed cap and ends at rest with actuation off.

### Known limit

If the host process dies (SIGKILL, crash, computer power loss), Feetech STS3215
servos keep holding their last goal with torque on. No software stop covers
that case; only the physical power switch does. HIL verifies the three stops,
and the docs say this plainly.

## Execution

- **Wave 1:** five isolated worktrees (`wt/control-hw`, `wt/data-learn`,
  `wt/sim`, `wt/quality`, `wt/frontend`) with strict file ownership. Shared
  files (`schemas.py`, `cli.py`, `app.py`, `config.py`, `pyproject.toml`,
  `api.ts`) only get additive edits. New routers and CLI sub-apps go in new
  modules.
- **Wave 2:** backend wiring (`SO101_BACKEND`, episode hook, sim eval scorer)
  and the wave-2 screens.
- **Wave 3:** merge, full gate after each merge, independent code review, docs
  (`SUITE_V02.md`, CHANGELOG, README, architecture), version 0.2.0.

Every chunk is committed on `v0.2`. Nothing is pushed.

## Honest boundaries

- Physics is an approximation. The servo gains are a community derivation;
  there is no backlash and friction is simplified. Sim success does not predict
  real success.
- The real arm, Quest actuation and LAN deployment stay HIL-gated. The bring-up
  kit makes that gate runnable and auditable; it does not pass it.

# Architecture

The suite separates intent, deterministic safety, hardware ownership, durable
metadata, and high-volume episode data. Browsers, trackers, policies, and
agents propose intent; only one host-local runtime may write the follower bus.

```text
 React workstation
 overview · teleop · capture · twin · Quest · data · train · eval · doctor · agent
          │ REST                 │ one latest-state WebSocket       │ MJPEG
          ▼                      ▼                                  ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ FastAPI control plane                                                │
 │ capability probe · leases · jobs · sessions · diagnostics · audit    │
 │ tracking retargeter · bounded agent skills · simulation contracts    │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │ canonical, validated command
                                ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ RobotManager — the single hardware owner                             │
 │ arbitration · state · watchdog · E-stop · faults · recording         │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │
            ┌───────────────────┼────────────────────┐
            ▼                   ▼                    ▼
   SafetyGuard + SO-101     Camera broker      Episode recorder
   LeRobot/Feetech driver                       LeRobot or fallback v2
            │                                        │
            ▼                                        ▼
     follower/leader                        datasets + artifacts

 Durable low-rate state: SQLite WAL     High-rate data: filesystem/LeRobot
```

## Runtime ownership and states

`RobotManager` owns follower, leader, cameras, the active loop, the shared
calibration-derived `SafetyGuard`, external lease, watchdog, and latest
telemetry. A lock serializes lifecycle transitions.

Activity is one of `idle`, `teleop`, `recording`, `rollout`, or `manual`.
Safety is tracked separately as `disarmed`, `armed`, `watchdog_timeout`,
`lease_expired`, `estopped`, or `fault`. This prevents a client from treating
“idle” as permission to move.

Legacy local loops—leader teleop, recording, and policy rollout—are mutually
exclusive internally supervised modes. Network/external sources—browser jog,
Quest, camera tracking, and approved agent skills—use the control protocol:

1. Acquire the one exclusive lease for a named source.
2. Heartbeat it before its TTL expires.
3. Explicitly arm while idle, calibrated, fault-free, and not E-stopped.
4. Send a finite joint target with matching source/lease, increasing sequence,
   source timestamp, TTL, and held deadman.
5. The manager bounds it; the shared guard applies calibration, position, and
   dt-based velocity limits; only then does the driver write.
6. A stale stream, expired lease, released deadman or soft stop disarms and
   Halts: the arm holds position, then Collapses to its rest pose when the
   hold timer runs out. A fault or E-stop disarms and de-energizes; E-stop is
   latched until explicit reset (docs/control-plane-safety.md).

The software stop is defense in depth. A reachable physical power stop remains
mandatory for real hardware.

## Hardware boundary

`MotorBus` is the narrow local interface used by follower and leader. Mock mode
uses a deterministic easing model. Real mode uses the pinned LeRobot 0.6.1
SO-101 follower/leader objects, with compatibility paths for older installed
APIs. Canonical state/action order is:

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper
```

Arm joints use calibrated degrees; gripper uses 0 closed to 100 open. Real mode
never changes into mock after an import, connect, calibration, training, or
policy failure. Calibration ranges load before the service creates its shared
guard and reload after successful follower calibration.

## Persistence and episode data

SQLite in WAL mode stores low-volume records that must survive restart:
timestamped audit/safety events and agent proposal state/transitions. Each
operation uses a short-lived connection and atomic state transition.

LeRobot is the production episode backend. The dependency-light fallback v2
exists for mock development and diagnostics:

```text
datasets/v2--<percent-encoded repo id>/   demo/pick -> v2--demo%2Fpick
  meta.json                         atomic manifest
  episode_000000/
    frames.jsonl                    measured, commanded, and desired action
    images/<safe-camera>/chunk_*.npy
  incomplete_<timestamp>/
    INCOMPLETE.json                 never counted as training data
```

RGB frames flush in bounded chunks, so episode length does not determine RAM
use. A per-dataset thread/file lock allocates episode numbers; staging is
renamed atomically and metadata is replaced through an fsynced temporary file.
Stopping an unsaved episode preserves it as incomplete for diagnosis/recovery.

Training runs outside the API control flow. A cancel watcher terminates the
real trainer process group even if stdout is stuck. Synthetic training and
moving mock policies run only when `SO101_MOCK=1`; real mode fails closed.

## Telemetry and media

The backend publishes latest telemetry at roughly 20 Hz. The React tree has a
single reconnecting WebSocket provider with freshness detection; pages consume
that shared state rather than opening competing sockets. Camera monitoring
uses MJPEG for the local v0.0 workstation.

MJPEG is not a production low-latency immersive transport. Quest deployment
should add WebRTC media/data while retaining server-side leases and command
validation. High-rate telemetry is intentionally not written to SQLite; it
belongs in episodes or a bounded metrics backend.

## Digital twin and simulation

The browser loads Three.js/React Three Fiber only on the simulation route. A
hardware-isolated backend advances deterministic 60 Hz joint-space kinematics
with play/pause/step/reset, rate limits, seeded scene objects, approximate FK,
and non-mutating trajectory previews. The UI binds that state to the articulated
SO-101 and can switch back to real WebSocket telemetry. With the `sim` extra,
a MuJoCo engine serves the same `/api/simulation` contract (`SO101_SIM_ENGINE`)
with gravity, contacts and an STS3215 servo model, without changing hardware
ownership; see [simulation](simulation.md).

The twin is an operational visualization, not certified collision or dynamic
truth. The physics uses the vendored Menagerie SO-101 MJCF (middle zero,
pinned commit) with an explicit gripper mapping.

## Quest, camera, and agent pipeline

```text
RGB/depth or OpenXR hands
  -> source timestamp + validity/confidence
  -> temporal filter + calibrated frame
  -> normalized wrist/pinch intent
  -> previewed bounded joint/EEF target
  -> lease + deadman + sequence + TTL
  -> shared safety/driver path
```

The browser can open a real camera, lazy-load MediaPipe Tasks, infer 21 hand
landmarks locally, and derive wrist/pinch intent for preview or held-deadman
camera teleop. Its Quest path manages an actual immersive WebXR session,
optional hand-tracking/local-floor features, 26-anchor hand frames, headset
joint rendering, recentering, and preview-only backend retargeting. Quest
actuation remains disabled pending HIL. Monocular depth remains ambiguous;
metric operation needs depth/stereo, fiducials, VIO, or external tracking.

Agent Lab persists a proposal before execution. The offline router or optional
Responses-API planner can select only a strict named-skill function; neither
has a motor, lease, arm, reset, or execution tool. Free-form language cannot
write joints. Motion additionally requires an owned lease and exact-target
human review: approving the reviewed target digest returns a single-use
`approval_token` (valid 30 s), and execute requires that token (a v0.1 body
with `approved=true` and no token gets 422). The server then runs the skill on a
thread at 30 Hz through the same lease, sequence, TTL, SafetyGuard and audit
path as every external source, until the targets are reached, the time limit
passes or supervision stops it, and ends every run with a Halt scoped to the
run's lease (see [control-plane-security.md](control-plane-security.md)).

## Capability truth and scaling

`GET /api/capabilities` is the user-facing truth source. Features declare
availability and maturity (`stable`, `beta`, `preview`, `experimental`). A
configured port or camera is not reported as proven hardware. Diagnostics keep
remediation beside failed checks; the audit timeline shows who changed state.

Both `GET /api/capabilities` and `GET /api/robot/status` carry `backend`:
`mock`, `sim` (MuJoCo physics) or `real`. Their older `mock` field is false
under sim as well as real, so only `backend` tells a rehearsal from the arm.

For fleet deployment, keep an edge runtime beside USB/cameras and move metadata
to Postgres, media/artifacts to object storage, durable jobs to a queue, and
live video to WebRTC. Motor commands must remain latest-value, non-replayable,
and locally watchdog-protected; they must never become durable queue messages.

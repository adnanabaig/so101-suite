# SO-101 Suite v0.0

This release is the first integrated operator workstation for an SO-101 arm.
It is intentionally honest about maturity: the complete workflow is available
with the deterministic mock backend, while anything that can move physical
hardware remains opt-in and must pass preflight on the exact arm, motor bus,
camera, calibration, and host where it will run.

The software/mock release gate is automated. No physical SO-101 pair or Quest
was attached during this build, so real-arm and headset status remains
**hardware beta / experimental** until the preflight and HIL checks below pass.

## Capability map

| Surface | v0.0 status | What is usable now |
| --- | --- | --- |
| Command center | Supported | Live state, safety state, cameras, control owner, recording and recent events |
| Leader teleoperation | Hardware beta | SO-101 leader to follower through the common safety guard |
| Manual teleoperation | Supported in mock; hardware beta | Joint sliders with press-and-hold deadman, exclusive lease, sequence/TTL validation, measured-relative limits, and watchdog |
| Collection | Supported | Episode lifecycle, camera/joint/action capture and LeRobot-compatible schema |
| Training and models | Supported in mock; adapter-dependent on real stack | ACT, Diffusion and VLA presets, cancellable jobs, checkpoint registry and rollout |
| Evaluation | Supported | Operator-scored trials, progress and reports |
| Digital twin | Supported for visualization | Live articulated arm and scenario metadata selection; no scene-object renderer or physics adapter yet |
| Diagnostics | Supported | Dependency, storage, calibration, ports, cameras, safety and runtime checks with remediation |
| Quest 3 / WebXR | Experimental | Secure-context and immersive-session capability probe only; no pairing/WebRTC/XR actuation yet |
| Camera hand teleop | Experimental | Browser-local MediaPipe landmarks, wrist/pinch preview, explicit lease, confidence gate, and press-and-hold tracked execution |
| Agent contract lab | Experimental | Deterministic named-skill, exact one-step target, explicit approval, audit and cancel; no learned/LLM planner yet |

`GET /api/capabilities` is the source of truth. The UI does not present an
unavailable adapter as connected and never turns simulated success into a
hardware claim.

## The motion invariant

Every interactive source emits intent. Only the host-local robot runtime emits
motor commands:

```text
leader / sliders / Quest / camera / policy / agent
                     |
                     v
       exclusive lease + deadman + sequence + TTL
                     |
                     v
      retarget / named skill / calibrated joint target
                     |
                     v
      finite check -> limits -> dt rate limit -> E-stop
                     |
                     v
              SO-101 follower driver
                     |
                     +--> telemetry + audit + episode data
```

An agent never receives an unrestricted motor-write tool. It can inspect,
propose one of the bounded skills, preview the resulting targets, and request
execution. A human approval and a live control lease are required before the
proposal reaches the same deterministic safety path used by every other input.

## Run the full mock suite

```bash
# from the repo root
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cd frontend && npm install && npm run build && cd ..
SO101_MOCK=1 .venv/bin/python -m uvicorn --factory so101.services.app:create_app --port 8000
```

Open <http://localhost:8000>. The built UI is served by FastAPI. API contracts
and interactive requests are at <http://localhost:8000/docs>.

For two-process development, run `SO101_MOCK=1 make api` and `make ui` in two
terminals. Vite proxies `/api` and `/ws` to the backend.

## Real-arm gate

Do not make the first real-hardware action a large trajectory. Before arming:

1. Put the arm on a stable surface with an unobstructed workspace and keep the
   physical power control reachable.
2. Confirm the follower and leader supplies, Waveshare bus adapters, motor IDs
   1–6, baud rate, firmware compatibility and stable device paths.
3. Install the robot extras in a dedicated environment and explicitly set
   `SO101_MOCK=0`. A requested hardware session fails closed when its adapter
   cannot load or connect.
4. Run the diagnostics page. Resolve every failed blocking check.
5. Calibrate with the current LeRobot calibration convention and verify the
   calibration identity matches the attached device.
6. Start at low speed. Acquire a lease, arm, hold the deadman and jog one joint
   through a small range. Verify sign and measured position before continuing.
7. Test browser disconnect, tracking loss, lease expiry, soft stop and the
   physical stop before collecting data or running a policy.

The software E-stop is a useful independent latch, but it is not a certified
safety function and cannot replace a physical stop or power isolation.

## Quest and hand tracking

Quest Browser exposes WebXR only in a secure context. Use HTTPS with a trusted
certificate and an authenticated reverse proxy on a reachable host; the suite
itself stays loopback-only by default. Then open **Quest & Hands** and run its
capability check. The v0.0 hand contract is deliberately small:

```json
{
  "source": "quest",
  "confidence": 0.97,
  "deadman": true,
  "execute": false,
  "wrist": {"x": 0.1, "y": -0.2, "z": 0.3, "roll": 0.0},
  "pinch": 0.42,
  "sent_at": 0.0
}
```

The server first returns a preview. Execution additionally requires a matching
lease, increasing sequence number, fresh timestamp, deadman, adequate tracking
confidence and an armed safety state. Tracking loss holds/stops; recovery must
be deliberately re-engaged so a stale pose cannot jump the robot.

Monocular RGB does not provide trustworthy metric depth by itself. The camera
pipeline therefore works in normalized/calibrated workspace coordinates and
keeps uncertainty visible. Add calibrated depth, stereo, fiducials or external
tracking before treating it as metric world-space control.

## Data and evaluation contract

Every saved episode should preserve:

- ordered joint names and units, desired and actually commanded actions;
- measured state, camera frame IDs and source/capture timestamps;
- task, scene, outcome and intervention markers;
- calibration, robot, camera, software and schema identities;
- dropped-frame, safety-clamp and incomplete/aborted status.

The fallback recorder keeps development and CI usable without LeRobot. Data is
not called training-ready until schema validation, camera timing, calibration
identity and an explicit episode outcome pass. Evaluation uses the same safety
path as teleoperation and keeps operator outcomes separate from automated
motion-quality metrics.

## Release verification

```bash
make ci
```

The software gate runs lint, branch coverage and its floors/skip budget, the
frontend checks and audit, the live mock smoke, and strict UI-bearing package
checks. `make verify` is only the faster tests/lint/UI-build development loop.
Hardware-in-loop checks are deliberately separate and opt-in; CI must never
move an attached arm.

## Known v0.0 boundaries

- One process owns one follower/leader pair. Run isolated edge instances for
  multiple followers until a fleet supervisor is added.
- The browser twin is for operations and trajectory understanding, not CAD or
  certified collision checking.
- MJPEG remains available for simple local camera monitoring. Low-latency
  immersive operation should use a WebRTC media adapter in a later milestone.
- Quest hand tracking, monocular camera retargeting and agent execution are
  experimental until measured on the target network, camera and arm.
- The SO-101 has five arm degrees of freedom; a six-degree hand pose cannot be
  followed exactly. Production IK should prioritize position and use a soft
  orientation objective.
- Multi-user authentication and remote Internet exposure are out of scope for
  this local v0.0. Bind to a trusted network only; do not expose the API as-is.

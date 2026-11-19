# SO-101 Suite v0.1

v0.1 extends the safety/control foundation with four end-to-end workstation
slices while preserving the same rule: only deterministic host code can own a
motor bus or issue a validated command.

## Added in v0.1

- **Quest WebXR preview:** real `immersive-vr` lifecycle, optional hand tracking
  and local-floor negotiation, 26-anchor hand frames, headset joint rendering,
  recentering, tracking-loss cleanup, and backend retarget preview. Physical XR
  actuation remains disabled until Quest HIL validation.
- **Stateful simulation:** hardware-isolated 60 Hz fixed-step kinematics,
  play/pause/step/reset, bounded joint targets, measured/target/error/velocity
  telemetry, approximate FK, seeded scene randomization, scene objects, and
  non-mutating trajectory previews. MuJoCo is probed truthfully but no physics
  adapter is active.
- **Dataset workstation:** rich episode/incomplete-capture manifests, bounded
  action timelines, read-only replay previews, outcome labels, QC for schema,
  finiteness, timestamps, limits, and camera chunks, plus recoverable
  trash/restore. Preview APIs never actuate hardware.
- **Optional intelligent proposals:** a server-side OpenAI Responses API
  provider can select exactly one strict named-skill function proposal. It has
  no motor, lease, arm, E-stop, or execution tool. Exact targets and provider
  rationale remain subject to the existing human review and one-step approval.

## Verification status

The software/mock gate covers backend tests, Ruff, formatting, mypy, strict
TypeScript/Vite production build, dependency audit, live HTTP smoke, and wheel
contents. Physical SO-101 and Quest acceptance remains separate because no
devices were connected in this build environment.

Before real use, run the exact-pair HIL checks in [SUITE_V0.md](SUITE_V0.md):
calibration/signs, torque transitions and stale-goal hold, manual and hand
deadman release, browser hide/loss, E-stop and physical power isolation,
camera faults, collection/resume, training, and rollout cancellation.

## Explicit boundaries

- Quest is an immersive hand/retarget **preview**, not WebRTC robot video or XR
  actuation.
- Simulation is deterministic kinematics, not collision/contact/gravity truth.
- The model planner is proposal-only and optional; deterministic planning is
  the offline default.
- One service process owns one leader/follower pair.
- LAN/headset access still requires an authenticated HTTPS reverse proxy; the
  suite binds to loopback by default.

# SO-101 Suite: notes for agents

An operations and learning workstation for the LeRobot SO-101 arm: FastAPI +
Typer (`src/so101`), React + Vite (`frontend/`), LeRobot 0.6.1, MuJoCo 3.14,
Feetech STS3215 servos. Version 0.2.2rc5. Read `docs/SUITE_V02.md` first (what it
does, what is verified, what is not), then `CONTRIBUTING.md` (the dev loop).

## Target and state

- Verified bring-up components on an ASUS ROG Zephyrus G15 GA503QR running
  Ubuntu 22.04.5: real follower calibration with ten accepted sweeps per joint,
  installed leader calibration with six initial accepted sweeps per joint,
  three camera feeds, decoded MCAP recordings and focused regression tests.
  See [G15 validation status](docs/validation/g15.md) for scope and limitations.
- Full G15 qualification is still pending: the October 27 preflight resolves
  the old rest-pose skip, but leader wrist-roll range/reference interpretation
  and command clipping remain unresolved,
  and the complete CI/HIL gate has not passed. Ubuntu 24.04 remains the supported
  qualification target. RealSense color capture and recorded playback are
  exercised; depth/IMU integration, GPU training and real trained-policy
  performance have not been verified. macOS and Windows remain unverified.
- Without hardware, use `SO101_MOCK=1` (no physics) or `SO101_BACKEND=sim`
  (MuJoCo; the `sim` extra).

## Verify before you claim

- `make ci` runs every CI job: lint (ruff, format, mypy), the tests with branch
  coverage, `coverage-floors.toml`, the skip budget (`tests/skip-budget.toml`),
  the frontend (lint, vitest, build, `npm audit --audit-level=high`), a live
  HTTP smoke test and a wheel build that must carry and serve the UI.
- Fix a flaky test; never retry it or lower a floor. A new skip needs a budget
  entry with its reason.
- A bug fix comes with a test that fails without the fix.

## Safety rules (physical arm)

- The robot manager (`services/robot_manager.py`) is the only owner of the
  motor buses. Standalone CLI commands go through `bringup/standalone.py`
  (persisted latches, canonical device-global bus locks).
- Three stops, user-decided: E-stop cuts torque on every motor and latches;
  Halt holds in place, then Collapses after 60 s (5 s unsupervised); Collapse
  lowers to the rest pose at no more than 15 °/s, then torque off. Never add a
  path that de-energizes the arm at the end of an operation.
- Fail closed: torque or position you cannot confirm ends in torque off plus a
  visible fault. Keep the lock order (`_io_lock`, then `_lock`) and take timing
  from the manager's injected `_clock`/`_sleep`.
- `docs/control-plane-safety.md` and `docs/control-plane-security.md` describe
  the contracts; keep them true when you change behaviour.

## Conventions

- Commit messages: `<area>: <what changed>` (control, data, sim, ui, install,
  docs, ...). Small, reviewable commits.
- Settings are `SO101_*` (`src/so101/config.py`); every setting has an entry in
  `.env.example`, which a test enforces.
- Publish only when the operator requests it; exclude private settings and recordings.

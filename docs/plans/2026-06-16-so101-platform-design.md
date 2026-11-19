# SO-101 End-to-End Platform — Design

**Date:** 2026-06-16
**Status:** Approved (brainstorming)

## Goal

A complete, runnable platform for the SO-101 robot arm covering the full
robot-learning lifecycle: **teleoperate → collect data → train policy →
serve/evaluate**, driven by a polished web UI and a clean Python core/CLI,
built on top of Hugging Face **LeRobot**.

## Decisions

| Question | Decision |
| --- | --- |
| Hardware | Real SO-101 (leader + follower, Feetech STS3215, USB cameras) |
| Foundation | Build on LeRobot (drivers, `LeRobotDataset`, policies) |
| Depth | Full working core path + clearly-marked scaffolds for extras |
| Stack | FastAPI + WebSocket backend, React + TS + Tailwind frontend |

## Architecture

Four layers, each usable independently:

1. **Core (`src/so101/`)** — pure Python library + `so101` CLI (Typer).
   Hardware abstraction, teleop loop, data recorder, training/serving/eval
   wrappers. Every external dependency (LeRobot, serial, cameras) degrades
   to a deterministic **mock** when absent, so the whole stack runs without
   hardware for development and CI.
2. **Services (`src/so101/services/`)** — FastAPI app. A single
   `RobotManager` owns the serial connection (hardware is not thread-safe /
   not shareable), a `JobRunner` runs training/eval as background
   subprocesses, and a WebSocket hub streams telemetry, camera frames, and
   logs.
3. **Frontend (`frontend/`)** — React + TS + Vite + Tailwind SPA. Talks to
   the backend over REST + WebSocket. Screens: Dashboard, Calibration
   wizard, Teleop+Record, Dataset browser, Training, Models, Eval, Settings.
4. **Infra** — pyproject (uv), Makefile, Docker, configs (pydantic +
   YAML), tests (pytest), CI, docs.

## Data flow

- **Collect:** leader arm → `TeleopController` → follower arm; frames +
  joint states streamed to UI and, when recording, written as a
  `LeRobotDataset` episode.
- **Train:** UI launches a `TrainingJob` → `JobRunner` shells out to
  LeRobot training → metrics streamed back over WS → checkpoints registered.
- **Serve/Eval:** a trained checkpoint is loaded by `PolicyRunner`, which
  closes the loop on the real follower while the UI shows live video and
  success/intervention controls.

## Safety

`SafetyGuard` enforces per-joint position/velocity limits and a software
e-stop on every command path (teleop, replay, policy rollout). The UI has a
persistent red E-STOP button wired to a dedicated endpoint.

## Mock strategy

`hardware/motors.py` exposes a `MotorBus` protocol with two
implementations: `FeetechBus` (wraps LeRobot/`scservo_sdk`) and `MockBus`
(physics-free, returns smoothly-interpolated joint states). Selection is
automatic (real if import + device succeed) and overridable via config
(`SO101_MOCK=1`). The same pattern applies to cameras and LeRobot policies.

## Out of scope (scaffolded only)

Sim-to-real, multi-robot fleets, RL/reward learning, and HF Hub auto-sync
are stubbed with clear `TODO`/`NotImplementedError` markers.

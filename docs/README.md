# Documentation

Start with the [project overview](../README.md) and
[G15 validation status](validation/g15.md).

For a first walkthrough, read [How SO101 works](codebase-tour.md). It maps the
folders, services, command flow and recording storage in reading order.

## Setup and operation

- [Development preview checks](validation/development-preview.md): publication checks and known limits.
- [Foreground session](suite-session.md): terminal lifecycle and two-stage Ctrl+C shutdown.
- [Getting started](getting-started.md): development and application entry points.
- [Hardware setup](hardware-setup.md) and [bring-up](bring-up.md): physical setup and checks.
- [Operator UI](operator-ui.md): control pages and stop behavior.
- [Operate acceptance record](validation/operator-ui-acceptance.md): requested features, current evidence and remaining physical checks.
- [Teleop and MCAP workflow](teleop-workflow.md): demonstrations, thumbsticks and motor mapping.
- [MCAP storage](mcap-storage.md): external volume checks, migration and recording capacity.
- [Manipulation roadmap and research thesis](plans/2026-10-27-manipulation-roadmap.md): qualification gates, episodes, pick-and-place and learning experiments.
- [Camera mounting brief](hardware/camera-mount-brief.md): D435i with existing supports; future CAD requirements.
- [Simulation](simulation.md): hardware-free development and physics.
- [Troubleshooting](troubleshooting.md): diagnostic paths.

## Engineering

- [Architecture](architecture.md): application boundaries.
- [Tool stack and integration direction](plans/2026-10-27-tool-stack.md): Suite UI, MuJoCo, Foxglove, recording and CI responsibilities.
- [Control-plane safety](control-plane-safety.md): bus ownership, limits and stops.
- [Control-plane security](control-plane-security.md): local access and command authority.
- [Contributing](../CONTRIBUTING.md): development loop and verification gates.
- [Installation](install.md): packaging and supported target.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/so101/` | Python application, motor controller, data and simulation |
| `frontend/` | React UI and frontend tests |
| `scripts/` | Development, validation and workstation helpers |
| `config/services/` | User service definitions |
| `config/visualization/` | Foxglove/OBS layouts and viewer dependencies |
| `config/rigs/` | Example workstation profile |
| `configs/` | Robot, camera, tracking and simulation presets |
| `tests/` | Backend and workflow tests |
| `data/`, `datasets/`, `outputs/` | Local operator data, excluded from Git |
| `submission/` | Local GitHub review copy and ZIP, excluded from Git |

Root commands such as `start`, `services` and `calibrate-leader` remain stable
operator entry points. Historical specifications live under `plans/` and
`SUITE_V*.md`; they describe intent and must not be mistaken for current results.

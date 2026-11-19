# SO-101 Frontend

React + TypeScript + Vite + Tailwind dashboard ("mission control") for the
SO-101 robot-arm platform.

## Development

```bash
npm install
npm run dev
```

The dev server runs on Vite's default port (5173) and proxies:

- `/api/*` → `http://localhost:8000` (FastAPI backend)
- `/ws/*` → `ws://localhost:8000` (telemetry WebSocket)

So the backend must be running on port `8000` for the UI to show live data.

## Installed UI end-to-end check

The Playwright check targets a suite that is already installed and serving its
built UI. It has no development-server fallback and cannot be run directly:
the installed-artifact harness supplies an ephemeral token, a fresh isolated
mock server, and a new evidence directory. From the repository root, run:

```bash
make frontend-deps
frontend/node_modules/.bin/playwright install chromium
make installed-e2e \
  INSTALLED_E2E_BACKEND=mock \
  INSTALLED_E2E_UI=1
```

The browser-install command above is the frontend-local executable from the
locked `@playwright/test` version; do not substitute an unpinned `npx`
download. `test:e2e` itself is a fixed, no-argument launcher. It refuses CLI
overrides and is useful only inside `make installed-e2e`, whose ephemeral
identity contract is required before the browser can send a product write.

`SO101_E2E_BASE_URL`, `SO101_E2E_TOKEN`, and
`PLAYWRIGHT_ARTIFACTS_DIR` are harness-owned. The browser sends the token on
every HTTP and WebSocket request. Before any other network request, the suite
checks `/api/e2e/identity`; it then verifies the installed UI build stamp and
the idle/disarmed mock state. A Vite server, real backend, reused workspace,
wrong token, or missing token is rejected before a product write.

The default uses the Chromium revision pinned by Playwright. On a constrained
host that already provides Chrome, explicitly set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome`; record that browser
version with the run because it is no longer Playwright's pinned revision.
The fixed launcher emits a token-free `SO101_E2E_BROWSER_METADATA=…` JSON line
with the selected executable path and reported version; the installed harness
retains it in the evidence log.
The installed harness routes failure screenshots and diagnostics into its
evidence ledger. The directory must be a new or empty dedicated leaf; the
Playwright config refuses broad or nonempty paths before Playwright can clear
them.

The check records one `e2e/installed_ui` episode, labels and trains it, runs
and stops the resulting policy, and writes a one-trial evaluation report.
Cleanup independently stops every loop, cancels active test jobs, Collapses to
rest, and verifies torque-off, latches, leases and terminal states. Failure
screenshots and token-redacted diagnostics are retained. Playwright traces are
disabled because they record custom request headers, including the ephemeral
token, in cleartext.

## Build

```bash
npm run build
```

Output is emitted to `dist/`. In production the FastAPI app serves the built
assets from `dist/` (same origin), which is why the API client uses relative
`/api/...` paths and the telemetry hook connects to `ws://<host>/ws/telemetry`.

## Pages

- **Dashboard** – robot status, joint bars, camera tiles, quick links.
- **Digital Twin** – articulated React Three Fiber SO-101 driven by live or
  stateful fixed-step simulation telemetry, seeded scenes and trajectories.
- **Quest & Hands** – real immersive WebXR lifecycle/hand preview, plus browser
  MediaPipe 21-landmark inference and deadman-gated camera hand teleop.
- **Diagnostics** – readiness checks, connected ports, audit timeline, and a
  downloadable support snapshot.
- **Agent Lab** – deterministic or optional proposal-only model planner,
  inspectable bounded skills, short lease and one-step approval gate.
- **Calibration** – run + monitor follower/leader calibration.
- **Teleop & Record** – start/stop teleop, live joints/cameras, dataset recording.
- **Datasets** – episode inspection, timeline/replay preview, outcomes, QC,
  incomplete captures and recoverable trash/restore.
- **Training** – policy presets, launch jobs, live loss chart + log tail.
- **Models** – list models, start/stop rollouts with live telemetry.
- **Evaluation** – run evals, score episodes, track success rate.
- **Settings** – backend health and configuration notes.

## Stack

- React 18 + React Router 7
- Vite 8 + `@vitejs/plugin-react`
- Tailwind CSS 3 (dark theme)
- Recharts 2 (training loss chart)
- React Three Fiber + Drei + Three.js (lazy-loaded 3D workstation)
- MediaPipe Tasks Vision (lazy-loaded browser-local hand landmarks)

## Safety model

Leader teleop, recording, policy rollout, and evaluation remain internally
supervised single-owner backend loops with explicit Start/Stop controls. The
external control lease is reserved for browser/manual commands, optional
tracking execution, and approved agent motion. Agent execution requires an
owned lease, current telemetry, exact-target review, and an explicit one-step
approval; the backend performs arm/command/disarm atomically.

Quest Browser WebXR and camera access require a trusted HTTPS origin. MediaPipe
WASM and the hand-landmarker model are fetched only when the operator presses
**Track hand**; URLs can be pinned/hosted locally with the variables documented
in `.env.example`. Camera hand motion requires an exclusive lease and a
press-and-hold deadman, and stops on confidence/tracking/visibility/lease loss.
The page does not claim Quest XR actuation or WebRTC connectivity when those
adapters are unavailable.

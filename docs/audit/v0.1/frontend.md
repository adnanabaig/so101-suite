# SO-101 Suite v0.1: Frontend audit report (React 18 / TS / Vite)

**Paths.**
- Frontend paths are relative to `/Users/baiadnan/Documents/so101/frontend/src/`.
- Backend paths are relative to `/Users/baiadnan/Documents/so101/src/so101/`; `app.py`, `robot_manager.py` and `routers/` are under `services/`.
- Evidence paths are relative to `/tmp/so101_audit/frontend/`.

**Scope.**
- All of `frontend/src`: pages, components, hooks and `api.ts`.
- The backend code that serves or feeds the UI: the SPA fallback, `/ws/telemetry`, `/api/control/*` and `/api/robot/*`.

**Method.**
- `tsc --noEmit` is clean (`tsc.log` is empty).
- **jsdom/vitest repros** run on a /tmp copy of the frontend that is byte-identical to the repo (checked with `diff -rq`).
  - Harness: `work/audit/safety_shell.test.tsx`. Output: `vitest_repro.out`.
  - Re-run with `cd /tmp/so101_audit/frontend/work && npx vitest run audit/safety_shell.test.tsx`.
- **Mock backend** (`SO101_MOCK=1`, port 8773), driven by curl and Python.
  - `scripts/disarm_demo.py` produced `disarm_demo.out`.
  - `evidence/*` holds the rest, including an offline OpenAPI contract diff.
- **Production build** in /tmp: `build.log`, `dist_fresh/`.
- **No real browser**, because headless Chrome isn't usable here. Findings that depend on browser network or rendering behaviour are marked PLAUSIBLE.
- Nothing under the repo was modified.

**Severity scale.**
- **CRITICAL:** exploitable now.
- **HIGH:** can defeat or hide a safety control, turn torque off on a live arm, or blank the safety UI.
- **MEDIUM:** shows misleading safety state or invites wrong operator decisions.
- **LOW:** correctness or UX debt.

## 1. Bugs, most severe first

### F1: CRITICAL. The SPA fallback serves any readable file on disk (path traversal)
- **Where:** `app.py:89-92`. This is backend code, but it is the path that serves the frontend.
  ```python
  @app.get("/{path:path}")
  def spa(path: str) -> FileResponse:  # client-side routing fallback
      target = dist / path
      return FileResponse(target if target.is_file() else dist / "index.html")
  ```
- **Defect:** `path` is joined to `dist` with no normalisation and no check that the result stays inside `dist`. Encoded `..` (`%2e%2e`) therefore escapes `frontend/dist`.
- **Failure scenario:** anyone who can reach the port can read any file the service user can read: HF token, `.env`, SSH keys.
  - The Docker images bind `0.0.0.0` (`docker/Dockerfile.suite:18,20`, `docker/Dockerfile.api:13,15`, `docker-compose.yml:26`). On a lab LAN that is an unauthenticated remote file read.
  - The non-Docker default binds 127.0.0.1 (`config.py:38`).
- **Evidence:**
  - `evidence/traversal_pyproject.txt`: `GET /%2e%2e/%2e%2e/pyproject.toml` returns 200 with content-length 1705 and body `[project]…`. The repo's pyproject.toml is exactly 1705 bytes.
  - `evidence/traversal_hosts.txt`: `/%2e%2e/` repeated 6 times, then `etc/hosts`, returns 200 with the contents of `/etc/hosts`.
- **Status:** CONFIRMED
- **Fix:**
  - Use `target = (dist / path).resolve()`.
  - Serve it only if `target.is_relative_to(dist.resolve()) and target.is_file()`.
  - Return a JSON 404 for `api/` and `ws/` paths (this also fixes F25).

### F2: HIGH. E-stop and "Reset E-stop" share one screen slot, so a panicked triple-tap un-latches the E-stop
- **Where:** `components/Layout.tsx:188-220`
  ```tsx
  {tripped ? (
    <button type="button" onClick={() => {
        if (!confirmReset) { setConfirmReset(true); return; }
        void runRoutine("reset", resetEstop);
      }} disabled={routineBusy !== null} ...>
      {routineBusy === "reset" ? "Resetting…" : confirmReset ? "Confirm reset" : "Reset E-stop"}
    </button>
  ) : (
    <button type="button" onClick={() => void fireEstop()} disabled={estopBusy} ...>
  ```
- **Defect:** as soon as the latch appears, the E-stop button is replaced, in the same DOM position, by the two-step reset.
  - The latch appears over WebSocket within about 50 ms, often while the E-stop POST is still in flight. The backend publishes the latch (`guard.estop()` and `safety_state="estopped"`) before its slow bus teardown (`robot_manager.py:357-367`).
  - Taps 2 and 3 of a repeated panic press therefore become "Reset E-stop" and "Confirm reset".
  - The confirm stays armed for 5 s (`Layout.tsx:118-122`).
- **Failure scenario:**
  - The operator hits E-stop three times. The robot latches and immediately un-latches.
  - The header returns to the normal red "E-stop", so the operator believes it is still latched.
  - Rollout, agent execution and teleop starts are allowed again from every station.
  - The backend refuses a reset only while a loop is alive (`robot_manager.py:385-386`), and `estop()` has just stopped the loop (`:368`).
- **Evidence:** `vitest_repro.out` REPRO 1:
  ```
  3 taps on one DOM slot hit ["E-stop","Reset E-stop","Confirm reset"] in 22 ms (no timer advanced); POSTs=["/api/robot/estop","/api/robot/estop/reset"]; backend still latched=false
  ```
- **Status:** CONFIRMED
- **Fix:**
  - Always render the E-stop in a fixed position.
  - Move Reset to a separate control: hold to confirm for at least 1.5 s, or a modal.
  - Keep Reset inert for 2–3 s after the latch first appears.

### F3: HIGH. Every pointer release on /teleop sends a disarm with no lease, which stops other stations' motion and erases latched safety states
- **Where:**
  - `components/ManualControlPanel.tsx:130-139` calls `stop()` (`:69-76`), which calls `hooks/useMotionLease.ts:35-40`, which calls `api.ts:867-868`.
  - On the backend: `routers/control.py:76-81`, then `robot_manager.py:740-765`, then `_deenergize_external`, which calls `follower.disconnect()` (`:921-949`, the call itself at `:943`).
  ```tsx
  useEffect(() => {
    const release = () => void stop();            // no holding / lease check
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    ...
  }, []);
  // useMotionLease.ts:39
  await disarmControl().catch(() => undefined);   // POST /api/control/disarm, no lease, errors swallowed
  ```
- **Defect:**
  - The window listener has no condition, and ManualControlPanel is always mounted on /teleop (`TeleopRecord.tsx:85`). Any click anywhere on that page POSTs `/api/control/disarm`, even with no lease and no motion.
  - The backend honours the request without a lease. The route calls `disarm_control()` with the default `reason="operator"`.
- **Failure scenario (a), another station's session:**
  - Station A holds a manual or XR-tracking lease and is armed.
  - An observer on station B clicks anything on /teleop.
  - A is disarmed mid-motion and the follower disconnects, which turns torque off. An SO-101 holding a pose against gravity drops.
  - A's next command gets 409.
  - The audit log records "manual control disarmed (operator)".
- **Failure scenario (b), a latched watchdog trip:**
  - After a watchdog trip, `safety_state="watchdog_timeout"` blocks loop starts until someone clears it on purpose (`TeleopRecord.tsx:70`, `Models.tsx:71`).
  - A stray click rewrites it to `"disarmed"`. The trip evidence is erased and the deliberate clear step is bypassed.
- **Evidence:**
  - `vitest_repro.out` REPRO 3:
    ```
    requests caused by two unrelated pointer releases: [POST /api/control/disarm, POST /api/control/disarm]
    ```
  - `disarm_demo.out`, against the mock backend:
    ```
    A arm-command 200 (armed, safety_state armed)
    B lease-less POST /api/control/disarm: 200 -> {armed: False, safety_state: disarmed}
    A command #2 (deadman still held): 409 control is disarmed
    after watchdog trip: safety_state watchdog_timeout; stray lease-less disarm: 200 -> safety_state disarmed
    ```
- **Status:** CONFIRMED
- **Fix:**
  - Frontend: `const release = () => { if (holding.current) void stop(); };`. This is the pattern `pages/Connect.tsx:309-319` already uses. Keep the window listener: it is what catches a release outside the button.
  - Backend: require the lease on `/control/disarm`. Global stops remain available through `/robot/stop` and E-stop. Don't overwrite latched safety states.

### F4: HIGH. Stale WebSocket telemetry overrides fresh HTTP status and removes the E-stop button
- **Where:** `Layout.tsx:116` and `hooks/useTelemetry.ts:84-89`
  ```tsx
  const tripped = telemetry?.estopped ?? status?.estopped ?? false;   // Layout.tsx:116
  ws.onclose = () => { ...; setState("closed"); scheduleReconnect(); }; // last telemetry frame never cleared
  ```
- **Defect:** the last WebSocket frame wins forever. `??` only falls back on null or undefined, so the stale frame is used even when `transportStale` is true and the 1.5 s HTTP poll says otherwise.
- **Failure scenario:**
  1. The E-stop is latched.
  2. The WebSocket drops (backend restart, Wi-Fi blip, proxy idle timeout). Backoff grows to 10 s (`useTelemetry.ts:95`), and it never recovers if the WebSocket path is broken.
  3. The latch is reset from another station, or the service restarts un-latched.
  4. The robot is live, but this station shows "Reset E-stop" and no E-stop button.
- **Evidence:** `vitest_repro.out` REPRO 2:
  ```
  after 2 fresh status polls (estopped=false) and 2 control polls: badge="disarmed", E-stop button rendered=false, Reset button rendered=true
  ```
- **Status:** CONFIRMED
- **Fix:** `const tripped = !transportStale && telemetry ? telemetry.estopped : (status?.estopped ?? false);`. Combined with F2, never remove the E-stop based on a condition. The same precedence pattern appears in `pages/Dashboard.tsx:100-102`, `pages/TeleopRecord.tsx:60-63,130` and `pages/Models.tsx:198`.

### F5: HIGH. There is no app-level ErrorBoundary, so one failed chunk or render error blanks the whole UI, E-stop included
- **Where:** `App.tsx:31-56` and `main.tsx:8-16`. The only boundary in the app is `TwinErrorBoundary` (`components/RobotTwin.tsx:50-76`).
  ```tsx
  <Layout>
    <Suspense fallback={<RouteFallback />}>
      <Routes> …12 lazy routes… </Routes>
    </Suspense>
  </Layout>
  ```
- **Defect:** any uncaught error in any route, including a rejected `lazy()` import, unmounts the whole React root. That includes the header and its E-stop.
- **Failure scenarios:**
  - After a redeploy, an open operator tab navigates to a route whose old hashed chunk now returns 404. The result is a white page with no E-stop and no status.
  - Any render exception does the same. One latent trigger: `ManualControlPanel.tsx:42-49` stores `undefined` for a joint missing from the observe response, and `:158` then calls `targets[joint].toFixed(1)`.
- **Evidence:**
  - `vitest_repro.out` REPRO 4: `after chunk failure: container.innerHTML.length=0; E-stop present=false`.
  - The mock server log shows `GET /assets/Old-chunk-deadbeef.js 404`.
- **Status:** CONFIRMED for the blanking. The `toFixed` trigger is PLAUSIBLE; it needs a partial observation.
- **Fix:**
  - Put an ErrorBoundary around `<Suspense>` inside Layout's main area, so the header survives, with a Reload action on `vite:preloadError`.
  - Add a top-level fallback in `main.tsx` that still renders a bare, fetch-based E-stop.

### F6: MEDIUM. The E-stop is disabled while its own request is in flight, for up to 8 s
- **Where:** `Layout.tsx:213-220` (`disabled={estopBusy}`). The default `timeoutMs = 8000` is at `api.ts:660`, and `estop` uses that default (`api.ts:730`).
- **Defect:** the one control that must always work cannot be pressed while its request is pending.
- **Failure scenario:**
  - The first press stalls. The backend E-stop path is slow by design: torque-off, then `stop(join_timeout=0.5)`, a retry, and `cancel_calibration(timeout=0.5)` (`robot_manager.py:367-371`). A saturated connection pool (F13) can also stall it.
  - The button greys out to "Stopping…" for up to 8 s and the operator cannot re-issue the stop. The backend handles repeated E-stops safely.
  - When it fails, the only feedback is fine print (F7).
- **Status:** CONFIRMED (code)
- **Fix:**
  - Never disable the E-stop; every press sends a request.
  - Show the in-flight state next to the button.
  - Use a short timeout with automatic retry.

### F7: MEDIUM. "Cut physical power now" appears as 12 px text next to a button that has just become "Reset E-stop", and the next action clears it
- **Where:**
  - `Layout.tsx:222-226`. The message is cleared at `:133` (runRoutine) and `:148` (fireEstop).
  - Backend: `robot_manager.py:373-381`, returned as HTTP 503 (`routers/robot.py:18-24`), and the 500 from `/robot/stop` (`routers/robot.py:45-49`).
  ```tsx
  {error && (<span className="max-w-48 text-xs font-semibold text-rose-300" title={error}>{error}</span>)}
  ```
- **Defect:** the most critical instructions in the system get the least visual weight. Examples: "E-STOP is latched but software torque-off failed; cut physical power now" and "control loop hung; forced torque-off attempted…".
  - At the same moment the big button turns amber "Reset E-stop" (F2), which invites the wrong action.
  - Soft stop or Confirm reset wipes the message (`setError(null)`) before its own request has even finished.
- **Failure scenario:**
  - The E-stop latches, but torque-off fails.
  - The operator sees a large amber "Reset E-stop" and a small, wrapped red line squeezed into the 80 px header.
  - Physical power is never cut.
- **Status:** CONFIRMED (code)
- **Fix:**
  - Show critical safety errors in a persistent, full-width `role="alert"` banner that must be acknowledged. This covers the E-stop 503, the soft-stop 500 and the disarm "torque-off unconfirmed" error.
  - Never clear it from unrelated actions.

### F8: MEDIUM. A normal idle robot always shows as "stale": permanent amber, an unreachable "ready" state and misleading copy (alarm fatigue)
- **Where:** `hooks/useTelemetry.ts:128-130`, `pages/Dashboard.tsx:112,145-152` and `Layout.tsx:73-82`. The root cause is backend design: the telemetry snapshot is refreshed only during loops and commands.
  ```ts
  const hasObservation = Object.keys(telemetry?.observation ?? {}).length > 0;
  const sampleStale = !hasObservation || sampleAgeMs === null || sampleAgeMs > 6000;
  ```
- **Defect:** in idle the backend streams `observation: {}` with an ever-growing `sample_age_ms`, so `sampleStale` and `stale` are always true.
- **Failure scenario:**
  - The header permanently shows amber "robot sample stale".
  - The Dashboard permanently says "Telemetry is not current. Motion workflows remain unsafe until the stream recovers", although the stream is healthy.
  - The green "Workstation ready" state is reachable only while a loop is running.
  - Operators learn to ignore amber, so a real stale stream during motion looks identical to normal idle.
- **Evidence:** `evidence/ws_idle_sample.txt`. The transport is healthy at 21 frames/s, yet the payload is:
  ```
  {'mode': 'idle', 'estopped': False, 'sample_age_ms': 27699.5, 'recording': False, 'fps': 0.0}
  observation: {}  action: {}
  ```
- **Status:** CONFIRMED
- **Fix:** either:
  - have the backend do a passive low-rate read in idle while the follower is connected, or
  - add an explicit idle/`sample_source` flag so the UI can show neutral "idle, no live sample".

  In both cases keep amber and red for stale data during activity.

### F9: MEDIUM. Agent Lab says "Execution is blocked" while its execute button is enabled
- **Where:** `pages/AgentLab.tsx:198-201` versus `:248-252`; the button is at `:487-494`.
  ```tsx
  const safetyReady = !!leaseId && !transportStale && control?.safety_state === "disarmed";
  ...
  value={stale ? "Stale" : telemetry?.mode ?? "Offline"}
  detail={stale ? "Execution is blocked" : "Live observation available"}
  ```
- **Defect:** the tile uses `stale` (transport or sample stale), but the gate uses only `transportStale`.
- **Failure scenario:** in normal idle (F8) the tile shows red "Stale / Execution is blocked" while "Approve one bounded step" is enabled and executes. This is the page that approves motion.
- **Status:** CONFIRMED (code, plus the F8 idle sample, where `stale=true` and `transportStale=false`)
- **Fix:** drive the tile and the gate from one predicate. Add `!sampleStale` to `safetyReady` if a fresh observation is required; otherwise fix the copy.

### F10: MEDIUM. The Models rollout UI is driven by local, optimistic state instead of the backend
- **Where:** `pages/Models.tsx:27,41-42,54-55,64,140,179,193-196`
  ```tsx
  await startRollout({ model_path: modelPath, fps }); setActivePath(modelPath);  // :41-42
  await stopRollout(); setActivePath(null);                                       // :54-55, the only place it is cleared
  const rollingOut = effectiveMode === "rollout" || activePath !== null;          // :64
  const isActive = activePath === m.path;                                         // :140 (per-row Stop)
  ```
- **Defect:** only this tab's own buttons set or clear `activePath`.
- **Failure scenarios:**
  - **(a)** The backend ends the rollout: E-stop, the 2 s operator timeout after the tab is hidden, a policy error, or the header's Soft stop. The "Live rollout" card and the red per-row Stop stay on screen, and every Start remains disabled until someone presses Stop against an idle backend.
  - **(b)** The page is reloaded or opened during a rollout. `activePath` is null, so no row shows Stop, and the page has no stop control for a live autonomous rollout. The header's Soft stop and E-stop still work.
- **Status:** CONFIRMED (code)
- **Fix:**
  - Derive rollout state and the active model from `/robot/status`, adding the model path to the response.
  - Drop `activePath`.
  - Show a page-level Stop whenever `effectiveMode === "rollout"`.

### F11: MEDIUM. Polled safety data goes stale silently on error, and camera tiles can show a frozen frame as live
- **Where:**
  - `hooks/usePolling.ts:41-44`: on error the old data is kept and only `error` is set.
  - These consumers ignore `error`: `Layout.tsx:110-111` (safety badge and latch fallback), `Models.tsx:21,23` and `Diagnostics.tsx:91`.
  - Camera liveness: `Dashboard.tsx:284` (`active={(status?.live_cameras ?? []).includes(camera)}`) and `components/CameraView.tsx:44-48`.
- **Defect:**
  - The last good status and control values keep rendering with no age marker.
  - CameraView's pulsing green dot is `active && loaded && !errored`, where `active` comes from the same retained poll. Nothing detects a stalled MJPEG stream.
- **Failure scenario:** the backend wedges and HTTP requests time out.
  - The header badge keeps saying "disarmed" or "armed" indefinitely.
  - Cameras keep their last frame with a pulsing green dot.
  - Only the small connection indicator turns red.
- **Status:** CONFIRMED for the badges. PLAUSIBLE for the camera dot (MJPEG end-of-stream behaviour can't be reproduced here).
- **Fix:**
  - Have `usePolling` expose `lastSuccessAt`.
  - Show "unknown (Xs)" after 2 missed polls.
  - Timestamp frames, or poll a snapshot, and grey out a tile after N s without a new frame.

### F12: MEDIUM. One slow lease heartbeat (round trip over 250 ms) drops the lease and turns torque off, with a misleading message
- **Where:** `hooks/useMotionLease.ts:22-29` and `:119-130`
  ```ts
  if (roundTripMs > 250) { throw new Error(`Control latency is ${...} ms; refusing a 300 ms motion TTL.`); }  // updateClock
  ...
  const renewed = await heartbeatControlLease(leaseId);
  updateClock(renewed.server_time, startedAt, Date.now());   // throws, so the catch below runs
  } catch { if (!disposed) { setError("Motion lease heartbeat failed; control was released."); await release(); } }
  ```
- **Defect:** a heartbeat that succeeded, but took 251 ms, is treated as a failed heartbeat.
- **Failure scenario:**
  - Wi-Fi jitter or a busy backend (MJPEG encoding, a dataset scan) occurs during a hold.
  - `release()` (`:42-48`) runs the lease-less disarm and then `release_lease` (`robot_manager.py:546-567`). Both turn torque off.
  - The operator is told the heartbeat failed, not that latency was high.
- **Status:** CONFIRMED (code). How often it happens is PLAUSIBLE.
- **Fix:**
  - In the heartbeat, skip the clock-offset update when the round trip is slow instead of throwing.
  - Apply the round-trip gate only to arming.
  - Report latency separately.

### F13: MEDIUM. MJPEG `<img>` streams can use up the browser's per-host connection pool, so E-stop and heartbeat requests queue behind them
- **Where:**
  - `components/CameraView.tsx:52-54`.
  - The endless multipart stream: `routers/telemetry.py:55-63`.
  - Camera tiles on `Dashboard.tsx:281`, `Models.tsx:211`, `Eval.tsx:235` and `TeleopRecord.tsx:154`, none paused when hidden or offscreen.
- **Defect:** each tile holds one HTTP/1.1 connection forever. Chrome allows 6 per host, shared across tabs, and uvicorn speaks HTTP/1.1.
- **Failure scenario:**
  - Three tabs with two cameras each fill the pool.
  - Every other request from every tab then waits until its 8 s timeout: the E-stop POST, operator and lease heartbeats, and polls.
  - The backend's supervision makes this fail safe (2 s operator timeout, 500 ms command watchdog). But the UI E-stop times out, and sessions die with confusing errors.
- **Status:** PLAUSIBLE (the 6-connection limit is documented Chrome behaviour; not reproduced, because no usable browser was available)
- **Fix:**
  - Pause or unmount streams when `document.hidden` is true or the tile is offscreen (IntersectionObserver).
  - Cap the number of concurrent streams, or move frames to the WebSocket or to HTTP/2.

### F14: MEDIUM. The digital twin shows a zero pose and a "0.0 / 0.0 / 0.0" command target, labelled as live, when it has no data
- **Where:**
  - `components/RobotTwin.tsx:87-93`: a missing joint becomes 0.
  - `components/RobotTwin.tsx:275-282`: `commandedTargets` is built with `rawJoint(action, …)`.
  - `components/RobotTwin.tsx:382-390`: the overlay renders on `showTarget && action`, `{}` counts as true, and `showTarget` defaults to true (`:404`).
  - Labels: `RobotTwin.tsx:407,475` ("Live kinematic twin · {mode}") and `Dashboard.tsx:206-207` ("Articulated from the latest observed joint state").
  - The Dashboard passes `observation`/`action` from telemetry and no stale flag (`:216-221`).
- **Defect:** missing data is drawn as a concrete pose and a concrete command.
- **Failure scenario:**
  - In idle (observation and action both `{}`) or after a WebSocket drop, the Dashboard twin shows the arm at zero pose with a "COMMAND TARGET 0.0 / 0.0 / 0.0" overlay, captioned as live observed state.
  - An operator can misjudge the real arm position before arming.
- **Status:** CONFIRMED (code plus `evidence/ws_idle_sample.txt`)
- **Fix:**
  - Pass `stale`/`hasObservation` into the twin and render a ghosted "no live sample" state.
  - Hide the target overlay when `action` is empty.

### F15: MEDIUM. The three.js twin (869 kB) loads and renders continuously on the landing page
- **Where:** `pages/Dashboard.tsx:29` imports RobotTwin statically. This contradicts `vite.config.ts:22-23`, which says three.js is isolated behind the lazy twin route. `RobotTwin.tsx:411-417` renders `<Canvas shadows …>` with no `frameloop`, so the default `"always"` applies.
- **Defect:** loading "/" downloads and parses the twin chunk, then runs a shadowed WebGL render loop at display refresh rate forever, even while telemetry is idle.
- **Failure scenario:** constant CPU and GPU load on the control workstation alongside the control loops, and a slower first paint on the page that hosts the safety header.
- **Evidence:**
  - `build.log`: `RobotTwin-DyWEWQ_5.js 869.25 kB │ gzip: 231.23 kB`.
  - `dist_fresh/assets/Dashboard-*.js` references `RobotTwin-DyWEWQ_5.js`.
- **Status:** CONFIRMED
- **Fix:** `lazy(() => import("../components/RobotTwin"))` with a placeholder, plus `frameloop="demand"` and `invalidate()` on each new observation.

### F16: MEDIUM. FastAPI validation errors (422) lose their message
- **Where:** `api.ts:691-699`, which only handles a string `detail`:
  ```ts
  if (parsed && typeof parsed.detail === "string")
  ```
- **Defect:** Pydantic 422 responses carry an array `detail`, so the user only sees "POST /api/… failed: 422".
- **Evidence:** `evidence/validation_422.txt`:
  ```
  {"detail":[{"type":"missing","loc":["body","source"],"msg":"Field required","input":{}}]}
  ```
- **Status:** CONFIRMED
- **Fix:** add a branch for the array form:
  ```ts
  else if (Array.isArray(parsed?.detail)) message = parsed.detail.map((d) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ");
  ```

### LOW

**F17. Job status contract drift (CONFIRMED)**
- **Where:**
  - The frontend expects `"completed"`: `api.ts:215-221` and `pages/Training.tsx:31-43`.
  - The backend emits `"succeeded"`: `training/jobs.py:11-16`. `evidence/jobs_status.json` shows the backend value.
- **Effect:** successful jobs render in a neutral grey tone.
- **Fix:** use `"succeeded"`, or generate the TypeScript unions from OpenAPI.

**F18. `usePolling` ignores fetcher parameter changes (CONFIRMED, REPRO 5)**
- **Where:**
  - `hooks/usePolling.ts:26-28,50-71`: the fetcher is held in a ref, and the effect deps are `[run, intervalMs, enabled]`.
  - Affected: `pages/Calibration.tsx:40-43`, and `pages/Training.tsx:229,329`, where `JobDetail` has no `key`.
- **Effect:** after switching arm or job, the previous arm's status or job's logs show until the next tick. REPRO 5: `"Status — leader: data for follower"`.
- **Fix:** add a key/deps argument that clears the data and refetches immediately.

**F19. `refresh()` is dropped while a request is in flight (CONFIRMED)**
- **Where:** `usePolling.ts:33-35`.
- **Effect:** post-action refreshes can show the pre-action state for a full interval (Eval score buttons at `pages/Eval.tsx:35-46,188-201`; Layout after an E-stop at `:137-138,151-152`). The backend returns 409 on double-scoring, so this is cosmetic.
- **Fix:** queue one follow-up refresh, and use the `EvalState` that `scoreEval` already returns (`api.ts:841-842`).

**F20. Training form validation and job handling (CONFIRMED)**
- `Training.tsx:167,178`: an empty field becomes 0 via `Number("")`, and `min={1}` is not enforced.
- `:185`: Start checks only repo and policy, and the backend accepts `steps=0, batch_size=0` (`evidence/training_steps0.txt`).
- `:249-260`: Cancel has no confirmation and reports errors with `alert()`.
- `:322-327`: the loss points array grows without limit.
- **Fix:** validate inputs, confirm before cancel, show errors inline, and cap or downsample the points.

**F21. TeleopRecord gating and stale state (CONFIRMED)**
- `pages/TeleopRecord.tsx:64-72`: Start is gated by the eight conditions in `legacyLoopReady` (`:108`, `:221`), but no reason is shown.
- `:60-63` (`episodeActive`, `episode_frames`) and `:130` (mode badge) read raw telemetry even when the transport is stale, unlike `effectiveMode` (`:58`).
- `:97,204`: an empty FPS field sends 0 to the backend.
- **Fix:** list unmet preconditions, use freshness-aware values, and validate FPS.

**F22. Narrow screens hide safety controls (CONFIRMED)**
- Below 1024 px, the mobile drawer (`Layout.tsx:324-343`, `fixed inset-0 z-50`) covers the z-20 header (`:347`), so stopping takes two taps.
- Below 640 px, the safety badge is hidden (`:166-168`).
- Header overflow at 375 px is PLAUSIBLE.
- **Fix:** keep the header or a floating E-stop above the drawer, and show a compact safety chip at every width.

**F23. Datasets action races (PLAUSIBLE)**
- **Where:** `pages/Datasets.tsx:169-181,235-246,248-261`. These handlers lack the `active` guard that the effect at `:97-124` uses.
- **Effect:** if the selection changes mid-request:
  - Repo A's quality report can show under repo B. The panel at `:669-673` doesn't display the repo name.
  - A late `labelOutcome` response can put A's detail on screen (`:497`). Later outcome clicks would then label B using A's episode numbers.
- **Fix:** capture the repo at call time and drop mismatched results.

**F24. Diagnostics events can be out of order (CONFIRMED)**
- **Where:** `pages/Diagnostics.tsx:95-104`.
- **Effect:** two newest-first lists from different poll moments are concatenated, de-duplicated and sliced to 50 without sorting.
- **Fix:** sort by timestamp after merging.

**F25. Unknown `/api` routes return 200 HTML (CONFIRMED)**
- **Where:** the SPA fallback (`app.py:89-92`) and `api.ts:704-707`.
- **Effect:**
  - `GET /api/nonexistent` returns 200 `text/html`, and so does a GET on the POST-only `/api/robot/estop` (`evidence/spa_api_fallback.txt`).
  - `request()` then returns `undefined`, so typos surface deep inside components.
- **Fix:** return a JSON 404 for `api/*`, and throw on a non-JSON 2xx response other than 204.

**F26. WebSocket stall handling (PLAUSIBLE)**
- **Where:** `hooks/useTelemetry.ts:58-100`.
- **Effect:**
  - The client reconnects only on `onclose`, so a half-open TCP connection is not reconnected; it is only flagged as stale after 2.5 s.
  - `Layout.tsx:95-96` keeps showing the last frame's age, frozen.
- **Fix:** force a reconnect after 5 s of silence, and show the time since the last message.

**F27. Accessibility on safety surfaces (CONFIRMED by grep)**
- No `role="alert"` or `aria-live` for safety state or E-stop errors. The only `aria-live` is at `pages/Simulation.tsx:1013`.
- No keyboard handling anywhere: no Escape for the drawer, no focus trap, no keyboard E-stop.
- The deadman (`ManualControlPanel.tsx:182-197`) is pointer-only, with no alternative and no explanation.
- **Fix:** the F7 banner, Escape handling and a focus trap for the drawer, and a documented E-stop hotkey.

**F28. Polling and render load (CONFIRMED by code)**
- `pages/Simulation.tsx:148` polls every 250 ms even when the tab is hidden.
- `:162`: `backendAvailable` stays true once set, because old data is kept.
- The telemetry context value changes at 20 Hz (`useTelemetry.ts:123-142`), re-rendering whole consumer pages.
- Status and control are polled separately by Layout and by each page, which also worsens F13.
- **Fix:** pause when hidden, use one shared status/control provider, and split the telemetry context or subscribe through selectors.

**Minor, not counted above**
- `api.ts:682` clears the request timeout before the body is read.
- `AgentLab.tsx:153` decides whether to disarm from a status poll up to 2 s old.
- `useOperatorSupervision.ts:16` keeps `healthy=true` while a heartbeat hangs. The backend enforces the timeout, so this is cosmetic.

**Coordination note.** F1, the backend half of F3, and the root cause of F8 are in backend files, so the safety/control-plane audit may report the same root causes.

## 2. Top improvement opportunities, ranked by value for effort

1. **Persistent safety bar (S–M).**
   - One component: an E-stop that is always rendered, never disabled, in a fixed position and above the drawer.
   - A separate hold-to-confirm Reset with a lockout after a latch, a full-width `role="alert"` banner for critical backend errors, and a keyboard E-stop.
   - Closes F2, F4 (the button part), F6, F7, F22 and F27.
   - This is the biggest daily-operator win: the stop control behaves the same in every state.
2. **Lease-scoped disarm and a gated listener (S).**
   - Frontend: a one-line `holding.current` gate.
   - Backend: require `lease_id` on `/control/disarm` and keep latched states.
   - Closes the only path by which an observer can turn torque off on another station's arm (F3).
3. **App ErrorBoundary and chunk reload (S).** The header and E-stop survive any page crash, and a 404 after a deploy becomes a "reload" prompt instead of a white screen (F5).
4. **One freshness-aware safety selector (S).**
   - A `useSafetyView()` hook merges WebSocket and HTTP data by age and returns estopped, mode, recording, armed, source and ageMs, or "unknown" when both are stale.
   - It replaces 6 or more ad-hoc `telemetry?.x ?? status?.x` merges, so every page shows the same state (F4, F10, F21).
5. **"Why is this disabled?" with one-click recovery (S).**
   - Every gated action lists its unmet preconditions with a direct fix. Examples: "external lease held by agent-lab", "safety_state watchdog_timeout: clear in Diagnostics", "operator link unhealthy".
   - This is the largest source of daily friction today: the buttons are silently grey.
6. **An honest idle state (M, backend and frontend).** Add a passive idle observation or an explicit idle flag, so "stale" means stale. "Ready" becomes reachable and alarm fatigue ends (F8, F9, F14).
7. **Server-authoritative operation state (S).** Rollout, eval and recording state come from backend responses, never from local flags. Add the active model path to status (F10, F19).
8. **Commit the safety UI regression tests (S).** The jsdom harness in `/tmp/so101_audit/frontend/work/audit/` already covers F2–F5 and F18 in 5 tests that run in about 150 ms. Turned into "must not happen" assertions, they would guard against these regressions in CI.
9. **usePolling v2, shared polling and camera hygiene (M).** Parameter-aware keys, lastSuccessAt with staleness, pause when hidden, queued refresh, one status/control provider, and capped, paused MJPEG streams with frozen-frame detection (F11, F13, F18, F19, F28).
10. **Render budget (S–M).** Lazy-load the twin with `frameloop="demand"` (F15), and split the 20 Hz telemetry context (F28).

## 3. What is genuinely good and must be preserved

- **The deadman**
  - Pointer capture; release on `pointerup`, `pointercancel` and `lostpointercapture`; `touch-none`; primary-button check (`ManualControlPanel.tsx:182-197`).
  - Blur stops motion, hiding the tab releases the lease, and unmount disarms and releases (`useMotionLease.ts:132-158`).
  - Commands stream at 10 Hz with a 300 ms TTL, a sequence number, and `sent_at` corrected to server time (`ManualControlPanel.tsx:78-127`).
- **Atomic arming and race guards**
  - The first frame is an atomic arm-command (`ManualControlPanel.tsx:111-121`), and generation guards re-disarm if arming races a release (`useMotionLease.ts:77-87`).
  - Generation guards and active flags for async results: the observe generation (`ManualControlPanel.tsx:35-56`), Calibration's generationRef and the Datasets effect's `active` flag.
  - `canSend()` reads refs, so interval callbacks never act on stale lease state.
- **Telemetry design.** A single shared WebSocket with backoff and teardown that is safe under StrictMode (`useTelemetry.ts:39-116`). It separates transport staleness from sample staleness using the server's `sample_age_ms` (`:125-138`). The model is right; some consumers misuse it.
- **Fail-safe supervision.** The operator-supervision heartbeat plus the backend's 2 s operator timeout and 500 ms command watchdog make nearly every frontend failure fail safe.
- **E-stop available before any page loads.** The header is in the main bundle and every page is lazy-loaded.
- **Confirmations and honest empty states**
  - Confirmations on risky actions: rollout start, recoverable dataset trash, the two-step reset (keep it but move it), and clear-fault gated on safety state.
  - Honest empty and unavailable states, CameraView retry, and the WebGL fallback.
- **The listener pattern to copy.** `pages/Connect.tsx:309-319` gates its window listener on a ref; ManualControlPanel should do the same.
- **Type and contract health.** `tsc` is clean, and the API types match OpenAPI almost everywhere. The offline diff (`evidence/contract_diff.txt`) found only the JobStatus drift and the 422 shape; its other differences were artifacts.

# SO-101 Suite v0.1: safety-critical control plane audit

**Scope**
- /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py
- hardware/{safety,motors,follower,leader,calibration,cameras}.py and teleop/controller.py
- services/routers/{control,robot,teleop,calibration,operations,telemetry}.py
- services/{app,store,jobs,schemas}.py
- policy/runner.py (rollout interaction only)
- The eval, tracking and agent routers were followed only where they drive RobotManager motion.

**Method**
- I read the code against the LeRobot 0.6.1 wheel source, downloaded to /tmp/so101_audit/safety/lerobot061.
- I drove a live mock server on 127.0.0.1:8771 and ran in-process TestClient and uvicorn scripts.
- For real mode I used the real FeetechBus, SO101Follower and RobotManager classes with a fake LeRobot SOFollower device. The fake follows the 0.6.1 semantics checked in the wheel source. No hardware was available.

**Integrity**
- I made no writes and no VCS changes under /Users/baiadnan/Documents/so101.
- Commits 1d3a14a (14:08, "Suite v0.1 baseline") and 7b59eda (14:42, "docs: v0.2 design") appeared during the audit. They are not mine. The only working-tree file changed since my first repro (14:25) is docs/plans/2026-10-01-suite-v02-design.md, which is part of 7b59eda.
- All line numbers below were re-checked against 7b59eda. The tree is clean.
- Baseline: `SO101_MOCK=1 pytest` gives 105 passed.
- Scripts ran with PYTHONDONTWRITEBYTECODE=1, and pytest with `-p no:cacheprovider`.

**Repro scripts and their outputs:** /tmp/so101_audit/safety/repro/

---

## 1. Bugs (most severe first)

### B1. CRITICAL: any web page can start autonomous policy motion through the eval workspace-reset gate, even after the operator pressed E-stop

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/routers/eval.py:240-251 (`continue_next`)
- eval.py:310-316: `POST /api/eval/continue`, which takes no body
- eval.py:301-307: `POST /api/eval/score?success=`, query parameter only
- eval.py:110-114: the rollout starts with the session's stored operator_id
- /Users/baiadnan/Documents/so101/src/so101/services/app.py:50-55: there is only a CORS allowlist, with no server-side Origin or CSRF check

**Defect:** The human "workspace reset confirmed" gate is a body-less POST. A cross-origin "simple" request (a form POST, or fetch with `mode: "no-cors"`) therefore executes on the server. CORS only hides the response from the attacker.

**Failure scenario**
1. The operator runs an eval. Between episodes the arm stops, the phase becomes `awaiting_reset`, and the operator reaches into the workspace. They may even press E-stop as a precaution.
2. Any page open in a browser on that machine POSTs `/api/robot/estop/reset`, then `/api/robot/fault/reset`, then `/api/eval/continue`.
3. `start_rollout` runs with the operator's stored id. The operator's own open UI tab keeps that heartbeat fresh, so the supervision check passes.
4. The policy drives the arm.

An E-stop does not abort the eval session; the phase stays `awaiting_reset`. So pressing E-stop does not protect the operator from this chain.

**Evidence (CONFIRMED)**

repro/02_csrf_eval_continue.out:
```
EVIL cross-origin POST /api/eval/continue (no body): 200 rollout
robot mode after evil request: rollout | follower_connected: True | fps: 23.6 ['rollout', 'rollout', 'rollout']
evil also scores via query-only POST: 200
```

repro/13_csrf_chain_defeats_operator_estop.out:
```
operator presses E-stop: (200, {'ok': True, 'detail': 'E-STOP engaged; all motion stopped'})
  status: {'mode': 'idle', 'estopped': True} | eval phase: awaiting_reset
EVIL /api/robot/estop/reset   -> 200 e-stop cleared
EVIL /api/robot/fault/reset   -> 200 runtime fault cleared
EVIL /api/eval/continue       -> 200 rollout
robot after chain: {'mode': 'rollout', 'estopped': False, 'follower_connected': True, 'fps': 25.8}
```

**Fix**
- Add middleware that rejects any non-GET/HEAD/OPTIONS request whose Origin (or `Sec-Fetch-Site: cross-site`) is not allowlisted.
- Make continue and score take a JSON body with operator_id. It must match the session's operator, whose heartbeat must be fresh.
- Abort the eval session on E-stop.

### B2. HIGH: E-stop reports success after a failed torque-off, and the fault can then be cleared without confirmation

**Where**
- /Users/baiadnan/Documents/so101/src/so101/hardware/motors.py:234-243: `FeetechBus.disconnect` sets `self._device = None` in `finally`, even when LeRobot's disconnect raised.
- Every consumer of `connected` then trusts it, all in /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:
  - :367-370, the estop retry
  - :954-955, where `_force_deenergize` returns True without any bus I/O when not connected
  - :925-926, `_deenergize_external`
  - :173-178, `clear_fault`

**Defect:** In LeRobot 0.6.1, `MotorsBus.disconnect(disable_torque=True)` raises from `disable_torque(num_retry=5)` and never reaches `closePort()` (lerobot/motors/motors_bus.py:556-561). The suite still marks the bus disconnected. From then on, every safety check reads "not connected" as "torque-off confirmed".

**Failure scenario**
1. The servo bus stops answering, for example after a USB glitch or a logic-side brownout. The servos keep holding torque.
2. The operator hits E-stop. The first `_force_deenergize` fails and sets the fault.
3. The retry at :369-370 sees `connected == False` and returns True.
4. The API answers 200 "E-STOP engaged; all motion stopped" instead of 503 "cut physical power now".
5. `reset_estop` followed by `clear_fault` returns the system to `disarmed` without any bus I/O.

The only further torque-off attempt is LeRobot's `Robot.__del__` when the dropped handle is garbage-collected (lerobot/robots/robot.py:76-85), and that call swallows its result. The same flaw makes the watchdog, disarm and lease release report success.

**Evidence (CONFIRMED)** — the real FeetechBus, SO101Follower and RobotManager code, with a fake 0.6.1 device; repro/04_failed_torque_off_confirmed.out:
```
estop #1: OK
  FeetechBus.connected = False | LeRobot device still open = True | servo torque = True | fault = forced torque-disable failed (estop): [TxRxResult] There is no status packet!
estop #2: OK  (reported success, no bus I/O performed)
reset_estop: OK, state = fault
clear_fault: OK, state = disarmed | fault = None
FINAL: servo torque still enabled = True | serial port still open = True
```

**Fix**
- `FeetechBus.disconnect` should keep the handle, or set a `torque_unconfirmed` flag, when the implementation raised.
- `_force_deenergize`, `_deenergize_external` and `clear_fault` should treat that state as failure. They should retry `disable_torque` and read back Torque_Enable.
- `clear_fault` should require positive confirmation, or an explicit "physical power was cut" acknowledgement.

### B3. HIGH: any web page can clear the latched E-stop and runtime faults

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/routers/robot.py:27-33: `POST /api/robot/estop/reset`
- robot.py:36-42: `POST /api/robot/fault/reset`
- Neither takes a body.
- /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:383-390: `reset_estop` only checks that no loop is alive.

**Defect:** The "deliberate reset" is the only way to clear the latch. It is a body-less POST with no operator identity, so a cross-origin simple request can perform it.

**Failure scenario:** The operator latches E-stop and walks over to the arm. A background page, or any local process, clears both the latch and the fault. Combined with B1 this restarts motion. On its own, it silently removes the interlock the operator believes is engaged.

**Evidence (CONFIRMED)** — repro/01_csrf_estop_reset.out:
- After E-stop, the state is `"safety_state":"estopped"`.
- A cross-origin POST to /api/robot/estop/reset returns `HTTP/1.1 200 OK {"ok":true,"detail":"e-stop cleared"}`.
- The state becomes `"safety_state":"disarmed"` with `"estopped":false`.
- For contrast, the JSON endpoints are protected. A text/plain body gets 422, no Content-Type gets 422, and a foreign-origin preflight gets 400 "Disallowed CORS origin".

**Fix:** Use the same Origin middleware as in B1. Also require a JSON body (operator_id plus an explicit confirm flag) for estop/reset and fault/reset, so they always trigger a CORS preflight.

### B4. HIGH: the SPA fallback serves any file on disk, any Host header is accepted (enabling DNS rebinding), and the WebSocket has no Origin check

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/app.py:89-92: `target = dist / path; return FileResponse(target if target.is_file() else dist / "index.html")`
- No TrustedHostMiddleware is installed (app.py:47-55).
- /Users/baiadnan/Documents/so101/src/so101/services/routers/telemetry.py:18-20: `await ws.accept()` with no Origin check

**Defect:** Joining `dist` with a path that contains `..` segments or is absolute escapes frontend/dist. Separately, the server answers requests for any Host header.

**Failure scenario**
- Any local process (or a LAN host, if the server is bound more widely) can read arbitrary user files. Examples: ~/.ssh, the project .env (SO101_HF_TOKEN and SO101_OPENAI_API_KEY per .env.example), and the HF token cache.
- A remote page that DNS-rebinds its own hostname to 127.0.0.1 becomes same-origin with the server. It can then read files, read the lease secret (B5), and call every JSON motion endpoint, which bypasses the strict-Content-Type protection noted in B3.
- Unknown /api/* GETs return 200 index.html, which hides client bugs.

**Evidence (CONFIRMED)**

repro/03_spa_traversal.out:
```
../../pyproject.toml                       -> 200 application/octet-stream 1705B
..%2f..%2fpyproject.toml                   -> 200 application/octet-stream 1705B
%2e%2e/%2e%2e/pyproject.toml               -> 200 application/octet-stream 1705B
/Users/baiadnan/.zshrc                     -> 200 application/octet-stream 1092B
%2fetc%2fpasswd                            -> 200 application/octet-stream 9344B
..%2F..%2F..%2F..%2F..%2F..%2Fetc%2Fpasswd -> 200 application/octet-stream 9344B
```

repro/09_ws_origin_and_host.out:
```
WS from Origin http://evil.example accepted; telemetry keys: ['action', ..., 'observation', ...]
GET /api/control/state with Host: rebind.attacker.example:8771 -> 200
GET /api/does-not-exist -> 200 text/html; charset=utf-8
```

**Fix**
- Compute `target = (dist / path).resolve()` and serve it only if `target.is_relative_to(dist.resolve())` and it is a file.
- Return 404 for the `api/` and `ws/` prefixes.
- Add TrustedHostMiddleware with the allowed hosts (127.0.0.1, localhost and the configured host).
- Check the Origin in `telemetry_ws` before calling `accept()`.

### B5. MEDIUM: the exclusive-lease secret is published on unauthenticated GETs, so any client can take over an armed session and lock the owner out

**Where** (all in /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py)
- :67-74: `ControlLease.public()` includes `"id"`
- :586: `control_state()` returns `self._lease.public()`, which is served by GET /api/control/state
- :513, :563, :736: the lease id is written into event details, which GET /api/events exposes (/Users/baiadnan/Documents/so101/src/so101/services/routers/operations.py:255)

**Defect:** The lease id is the only credential for /arm, /command and /observe, yet every reader receives it. By contrast, `operation_owner` is deliberately redacted in status (robot_manager.py:305).

**Failure scenario**
1. The operator is holding the deadman with the arm armed.
2. A second tab, a script, or a rebinding page (B4) reads lease.id and lease.source from /api/control/state.
3. It sends a command with sequence 10^12, and the arm moves.
4. From then on every operator frame is rejected as out-of-order, so the operator loses control while the other client keeps it.

**Evidence (CONFIRMED)** — repro/10_lease_id_exposed_hijack.out:
```
other client GET /api/control/state -> lease: {'id': 'f084724e88684b6cac7328815e6afacf', 'source': 'ui'}
lease id also present in /api/events entries: 2
other client command with copied lease, seq=1e12 -> 200 command accepted: shoulder_pan=7.94
owner's next command seq=2 -> 409 out-of-order command sequence 2; last was 1000000000000
```

**Fix:** Return the id only from POST /lease. `control_state()` and events should expose a redacted view (source, expiry, a short hash). Optionally bind the lease to a per-tab secret header.

### B6. MEDIUM: a failed arm-and-command request disarms whoever currently holds control, and can return 500

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/routers/control.py:122-123: the `except` block calls `manager.disarm_control(reason="atomic first command failed")` with no ownership check, and strict.
- The same shape appears at:
  - /Users/baiadnan/Documents/so101/src/so101/services/routers/tracking.py:150-152
  - /Users/baiadnan/Documents/so101/src/so101/services/routers/agent.py:164-165, where the disarm sits in `finally`
- The strict raise path is robot_manager.py:924-926 and :761-764.

**Defect:** Cleanup for a rejected request is global. A request that failed *because it does not own the lease* still tears down the owner's armed session.

**Failure scenario**
- A stale second tab, or a tracking client whose lease expired, calls /arm-command. The owner's arm de-energizes mid-manipulation: it slumps and drops whatever it holds.
- If a teleop or rollout loop is running, the strict disarm raises inside the `except` block, and the client gets a 500 instead of a 409.

**Evidence (CONFIRMED)** — repro/05_arm_command_freshness_and_cross_disarm.out:
```
owner command: 200 | armed: True | follower_connected: True
other client arm-command: 409 valid exclusive control lease required
owner after other client's failed request -> armed: False | safety: disarmed | follower_connected: False | owner's next command: 409
teleop start: 200
arm-command during teleop -> 500 Internal Server Error
```
The tracking.py and agent.py sites have the identical code shape. I did not reproduce them separately.

**Fix:** Clean up only if this request did the arming. For example, add `manager.disarm_if_owner(req.lease_id, strict=False)`, which compares `self._lease.id` under `_lock`, and then re-raise the original error.

### B7. MEDIUM: the command TTL is not enforced on the arm-first paths, or when sent_at is omitted

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/routers/control.py:111-117: freshness is checked before arming, then the command is sent with `sent_at=datetime.now(timezone.utc)`.
- /Users/baiadnan/Documents/so101/src/so101/services/routers/tracking.py:131-145: the same pattern, plus `req.sent_at or datetime.now(timezone.utc)` when the client omits sent_at. The schema allows that: /Users/baiadnan/Documents/so101/src/so101/services/schemas.py:516 declares `sent_at: datetime | None = None`.

**Defect:** The server substitutes its own clock for the client's timestamp, so ttl_ms no longer bounds the age of the command that executes.

**Failure scenario**
- On real hardware, `arm_control` opens the port, configures six motors, then reads and primes positions before the first command is sent. None of that time is bounded, because the deadman only starts at robot_manager.py:680. A first frame that has outlived its TTL still executes.
- A tracking client that omits sent_at gets no staleness check at all. For example, buffered hand-tracking frames would all execute.
- The shipped UI always sends sent_at, so the omitted-field case affects other clients.
- Motion stays bounded by SafetyGuard (15 deg per step) and by LeRobot's 5-degree relative clamp.

**Evidence (CONFIRMED)**
- repro/05, with 1.2 s of latency injected into `follower.connect`:
  ```
  arm-command with ttl_ms=500 executed when command was 1518 ms old -> 200 armed and accepted first command: shoulder_pan=10.00
  same timestamp on /api/control/command -> 409 stale command (1519 ms old; limit 500 ms)
  ```
- repro/07:
  ```
  preview execute, 2 s stale   -> 409 stale command (2001 ms old; limit 500 ms)
  preview execute, no sent_at  -> 200 preview accepted and bounded command executed {'shoulder_pan': 3.885500003118068}
  ```

**Fix:** Pass `req.sent_at` through to `send_external_command`, which already re-validates it inside the io_lock. Make sent_at required whenever execute is true.

### B8. MEDIUM: the E-stop latch and faults live only in memory, so any restart clears them without a deliberate reset

**Where**
- /Users/baiadnan/Documents/so101/src/so101/hardware/safety.py:48: `estopped: bool = False`
- /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:94 and :127: the constructor sets `self._fault = None` and `self._safety_state = "disarmed"`
- The store persists none of this state.
- `make api` runs uvicorn with `--reload` (/Users/baiadnan/Documents/so101/Makefile:27-28).

**Defect:** The invariant "only a deliberate reset clears the latch" does not survive a process restart.

**Failure scenario**
1. The operator latches E-stop because something is wrong.
2. The service crashes or is restarted, or, under `make api`, reloads because someone saved a source file.
3. The UI comes back `disarmed` with no fault, and arming works immediately. That includes B2's "torque-off failed; cut physical power" fault.

**Evidence (CONFIRMED)** — repro/11_estop_latch_not_persisted.out:
```
before restart: estopped = True | state = estopped | fault = forced torque-disable failed (estop): no status packet
after restart:  estopped = False | state = disarmed | fault = None
arm after restart without any reset -> armed
```

**Fix:** Persist `{estopped, fault, safety_state}` in the existing SQLite store whenever estop or a fault is set, and clear it only in `reset_estop` and `clear_fault`. Restore it in `RobotManager.__init__` before the API starts serving.

### B9. MEDIUM: request handlers wait on `_io_lock` with no timeout, so one wedged driver call can starve the E-stop endpoint

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:814: `with self._io_lock:` in `send_external_command` runs *before* the lease and armed checks at :817-833.
- Other untimed waits in request paths are at :529, :605, :649, :691 and :697. Only the teardown paths at :927 and :956 use timeouts.
- /Users/baiadnan/Documents/so101/src/so101/services/routers/robot.py:18-24: `def estop()` is synchronous, so it runs on the same AnyIO 40-thread pool.

**Defect:** While one bus call is wedged, every queued /api/control/command holds a pool thread indefinitely, even one carrying an invalid lease. Once 40 are blocked, the E-stop handler cannot start at all, and the guard latch is never set.

**Failure scenario**
- The USB-serial adapter wedges mid-write while several tabs or a script keep posting commands.
- The UI E-stop hangs, and so does the `so101 estop` CLI, which also goes over HTTP (/Users/baiadnan/Documents/so101/src/so101/cli.py:272).
- The watchdog thread still raises a fault, but the latch, the lease revocation and the operator feedback never happen.
- The shipped UI keeps only one command in flight per tab, so this needs multiple tabs or a non-UI client.

**Evidence (CONFIRMED mechanism)** — repro/12_estop_threadpool_starvation.out:
```
1 wedged command:   estop -> 503 in 1.05s: E-STOP is latched but software torque-off failed; cut physical power n...
+40 queued commands: estop -> NO RESPONSE after 5.0s (threadpool exhausted)
guard.estopped at that moment: False
```
The first line is the correct behavior; the second shows the starvation.

**Fix**
- Validate the lease and armed state before taking `_io_lock`.
- In request handlers, acquire `_io_lock` with a timeout (for example 250 ms) and return 503 on expiry.
- Make /api/robot/estop an `async def` that latches `guard.estop()` inline and runs teardown on a dedicated executor.

### B10. MEDIUM: a follower resting outside the guard envelope can never be armed or teleoperated

**Where**
- /Users/baiadnan/Documents/so101/src/so101/hardware/follower.py:68-76 (connect) and :92-94 (enable_torque): Goal_Position is primed to the measured pose, then `record_actual_command` runs.
- /Users/baiadnan/Documents/so101/src/so101/hardware/safety.py:123-125: it raises on an out-of-range accepted command.
- The envelope is DEFAULT_LIMITS of ±100 deg for lift, elbow and wrist_flex (safety.py:18-25), intersected with calibrate_timed's symmetric ±half-span (/Users/baiadnan/Documents/so101/src/so101/hardware/motors.py:430-432).

**Defect:** Connecting requires the current pose to be inside the envelope. But the envelope is clipped to ±100 deg, and the arm can rest beyond that.

**Failure scenario**
- The arm rests folded on a mechanical stop with torque off. On any joint whose calibrated sweep exceeds about 2275 ticks, that rest pose reads beyond 100 deg.
- /arm returns 409 and latches a fault.
- /teleop/start returns 200 "teleop started", and then the loop faults.
- The operator must lift the arm into the envelope by hand before every session.

This fails closed, so there is no unsafe motion, but it blocks normal operation. Mock mode starts every joint at 0, so tests never exercise it.

**Evidence (CONFIRMED code path; hardware precondition PLAUSIBLE)** — repro/06_rest_pose_outside_envelope.out:
```
calibrated shoulder_lift half-span for a 2491-tick sweep: +-109.5 deg
guard envelope shoulder_lift: (-100.0, 100.0)
arm -> 409 could not arm follower: driver accepted out-of-range command for shoulder_lift: -109.49450549450549
safety_state: fault | fault: driver accepted out-of-range command for shoulder_lift: -109.49450549450549
teleop start -> 200 {"ok":true,"detail":"teleop started @ 30 fps"}
after teleop attempt: mode: idle | fault: driver accepted out-of-range command for shoulder_lift: -109.49450549450549
```

**Fix:** When the seed pose is outside the envelope, accept the hold-in-place prime. Then allow only commands that move each joint monotonically back toward [lo, hi]. Alternatively, derive the limits from the calibrated range minus a margin.

### B11. LOW: non-finite or overflowing numbers in request bodies return HTTP 500 instead of 422

**Where**
- /Users/baiadnan/Documents/so101/src/so101/services/schemas.py:11-19: `datetime.fromtimestamp` at :18 raises OverflowError, which Pydantic does not convert into a validation error.
- FastAPI's default 422 handler cannot JSON-encode a NaN or Infinity value echoed back in `input`.

**Defect**
- sent_at values of 1e300, -1e300, Infinity or NaN return 500.
- Any non-finite value that fails validation, such as a NaN target, crashes the error response itself.

**Failure scenario:** A client bug that sends NaN gets a 500 rather than a 4xx. This fails closed (no command executes), but clients that treat 5xx as retryable will retry, and diagnostics get muddied.

**Evidence (CONFIRMED)**
- repro/07: all six cases (sent_at 1e300, Infinity, -1e300 and NaN; targets NaN and 1e309) return 500.
- repro/07b:
  ```
  sent_at=1e300 -> raises OverflowError: timestamp out of range for platform time_t (at schemas.py:18)
  targets NaN   -> raises ValueError: Out of range float values are not JSON compliant: nan (at encoder.py:258)
  ```
- The 07b line labelled "estop reset w/ NaN body" is mislabelled. It actually posts a bare NaN to /api/control/command and gets the same ValueError.

**Fix:** In `_coerce_timestamp`, reject non-finite values and values beyond about 1e11 s by raising ValueError. Register a RequestValidationError handler that sanitizes the `input` echo.

### B12. LOW: calling /api/control/observe while armed drops torque, but the state still says armed

**Where:** /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:591-609. It checks `self.mode is not Mode.IDLE` but not `self._armed`, and calls `follower.disconnect()` at :609.

**Defect:** After /arm and before the first command, the mode is still IDLE, so observe is allowed and disconnects the follower.

**Failure scenario:** Any lease holder calls observe between /arm and its first command, for example to re-seed targets. The arm goes limp, but /control/state keeps reporting armed until the watchdog fires. The next command then fails with "arm control again". The shipped UI observes on lease change, which normally happens before arming, so the trigger is a script or another component.

**Evidence (CONFIRMED)** — repro/08_observe_while_armed.out:
```
arm -> 200 | torque: True | connected: True
observe -> 200 | armed: True | safety: armed | follower connected: False | torque: False
next command -> 409 follower is not connected; arm control again
```

**Fix:** Reject observe when `self._armed` with ControlConflict("disarm first"), or read without disconnecting when already armed.

### Checked and not reported as bugs
- **SafetyGuard math is sound** (/Users/baiadnan/Documents/so101/src/so101/hardware/safety.py:129-179). It clamps position first, then rate-limits against the last accepted value. dt is capped at 0.25 s, and the first command after a seed gets the nominal dt.
- **Lock ordering is consistent.** It is always `_io_lock` then `_lock`.
- **Lease and heartbeat bounds are fine.** Lease TTL is bounded to 0.1-60 s, and the operator-heartbeat map is pruned.
- **Accepting a "mock" calibration in real mode** (follower.py:35) can only narrow the envelope, never widen it.
- **An E-stop during a running loop also latches a secondary fault.** This fails closed.

---

## 2. Improvement opportunities (ranked by value/effort)

1. **One cross-origin and host guard (S).**
   - A small middleware rejects state-changing requests whose Origin/Sec-Fetch-Site is not on the allowlist.
   - Add TrustedHostMiddleware, plus an Origin check in `telemetry_ws`.
   - This closes B1, B3, B4's DNS-rebinding half, and the cross-site telemetry read in one place, without touching any handler.
2. **Contain the SPA fallback (S).**
   - Use `resolve()` plus `is_relative_to(dist)`, and return JSON 404s for `api/` and `ws/`.
   - Three lines remove an arbitrary-file-read primitive, and stop unknown API routes from silently returning index.html.
3. **Make the latch durable and authoritative (S).**
   - Persist `{estopped, fault}` in the existing store and restore it at startup.
   - Make E-stop abort an active eval session; in repro 13, eval stayed in `awaiting_reset` through an E-stop.
   - This makes "only a deliberate reset clears it" actually true.
4. **Treat the lease as a bearer secret end-to-end (S).**
   - Return the id only to the acquirer, and redact it in /control/state and events, as `operation_owner` already is.
   - Make error-path cleanup owner-scoped and non-strict.
   - Stop re-stamping sent_at after arming, and require sent_at for every execute=true request.
   - Fixes B5, B6 and B7 with small, local changes.
5. **Three-state torque truth: on, off, unconfirmed (M).**
   - FeetechBus keeps the device handle when LeRobot's disconnect raises.
   - E-stop retries do real `disable_torque` I/O, followed by a Torque_Enable read-back.
   - `clear_fault`, disarm and the watchdog report "unconfirmed" instead of success.
   - This fixes B2 and makes the "cut physical power" messaging trustworthy.
6. **Bounded hardware waits and a dedicated E-stop path (S-M).**
   - Validate the lease before taking `_io_lock`, and use a timed acquire (returning 503) in request handlers.
   - Make /api/robot/estop async, latching inline and tearing down on its own executor.
   - The E-stop endpoint then never competes with hardware handlers for the 40-thread pool (B9).
7. **Fault-injection harness (M).**
   - MockBus read, write and torque calls ignore `connected`, and its disconnect can never fail (motors.py:68-86), so mock mode cannot exhibit B2.
   - Make MockBus honor connected and torque state.
   - Add a fake LeRobot SOFollower fixture (as in repro 04) to test E-stop, disarm and clear_fault under failed torque-off, hung I/O, and loop/watchdog races.
8. **Realistic envelope and rate defaults (M).**
   - Seed outside the envelope with "move-toward-envelope only" semantics (B10).
   - Make the velocity limit per-joint and below the servo's capability. The 450 deg/s default rarely binds, so LeRobot's `max_relative_target=5.0` (motors.py:140) is the effective rate limiter.

---

## 3. What is genuinely good and must be preserved

**E-stop and locking**
- **E-stop latches first.** `guard.estop()` runs before any lock is taken (robot_manager.py:357), so every command path sees it immediately.
- **E-stop never blocks indefinitely on the bus.** It uses a 0.5 s timed `_io_lock` acquire (:956). With a wedged driver it reports 503 "cut physical power now" honestly (repro 12, first case).
- **The lock split is sound.** `_lock` guards state and `_io_lock` guards bus I/O, always taken in the same order.
- **`_authority_epoch` is revalidated after every I/O phase** in `arm_control` (632-738) and `send_external_command` (794-902). If authority was revoked mid-I/O, the follower is de-energized. This handled every race I tried to construct.

**Command path**
- **The sequence is reserved under the lock before any I/O** (robot_manager.py:836). Freshness is re-checked inside the io_lock (:816), so queued requests cannot execute stale frames.
- **The deadman timer starts at arm** (:680), and the watchdog runs in its own thread, independent of the request pool.
- **Motor-side safety is careful.** Before torque is enabled, the guard is seeded from the measured pose and Goal_Position is primed to it (follower.py:66-98). This prevents the stale-goal jump.
- **`record_actual_command` anchors the limiter to what LeRobot actually commanded**, and it rejects non-finite or out-of-range driver output.
- **Non-finite values are rejected at every layer:** the schema, the guard, and the bus.

**Real mode fails closed**
- No mock fallback for hardware (motors.py:144-161, which also enforces 1 Mbps).
- No mock fallback for policies: /Users/baiadnan/Documents/so101/src/so101/policy/server.py:205-213 refuses mock checkpoints when SO101_MOCK=0.
- Reads and writes are refused when LeRobot reports the arm uncalibrated.
- Calibration is never auto-written on a mismatch.
- `calibrate_timed` keeps torque off, can be cancelled, checks a minimum range, and rolls back both the motor registers and the file on failure (motors.py:331-452).
- `finish_calibration` reloads the guard through the same intersection used at startup.

**HTTP and frontend hygiene**
- **JSON endpoints already resist CSRF**, thanks to strict Content-Type plus the CORS allowlist. In repro 01, text/plain got 422, no Content-Type got 422, and a foreign preflight got 400.
- **The frontend keeps a single request in flight** for manual commands and tracking.
- **Operator ids are per tab.**
- **`operation_owner` is redacted** in /api/robot/status.

**Loop teardown and recording**
- If `stop()` times out on the join, it latches a fault and forces a de-energize.
- Incomplete episodes are preserved on abort.
- The control loop holds `_io_lock` per step only, so an E-stop can interleave between steps.
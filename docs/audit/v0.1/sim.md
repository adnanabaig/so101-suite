# SO-101 Suite v0.1: audit of simulation, hand-tracking/WebXR and the LLM agent

**How I checked:**
- Read the code and ran repro scripts against the app (FastAPI TestClient, `SO101_MOCK=1`, data under /tmp).
- Compared against the official SO-ARM100/Simulation/SO101 URDF and MJCF (cloned to `/tmp/so101_audit/sim/SO-ARM100`) and the LeRobot v0.6.1 source (`/tmp/so101_audit/sim/lerobot`).
- Used the OpenAI Structured Outputs docs, fetched with curl on 2026-10-01.

**Baseline:** 105 tests pass, the 21 scoped tests pass, and `tsc --noEmit` exits 0. No repo files were modified.

**Repros** are in `/tmp/so101_audit/sim/repro/`. Run each with `cd /Users/baiadnan/Documents/so101 && .venv/bin/python <script>`, except `mujoco_probe.py`, which runs with `/tmp/mjprobe/bin/python`.

| Script | What it checks |
|---|---|
| `fk_compare.py` | Suite FK against the official model |
| `twin_compare.py` | 3D twin against the backend FK and the official model |
| `agent_execute_repro.py` | What an approved proposal actually commands |
| `agent_race_repro.py` | Cancel race, stuck "executing", concurrent execute |
| `tracking_repro.py` | Freshness bypass, NaN input, audit throttling |
| `engine_repro.py` | Determinism, catch-up, preview, concurrency |
| `mujoco_probe.py` | Official MJCF in MuJoCo 3.14 |

## 1. Bugs (most severe first)

### B1 HIGH: an approved agent action sends one clamped command, then cuts torque, and is marked "executed"

**Where:**
- `src/so101/services/routers/agent.py:154-165`
- `robot_manager.py:904-918`: one command
- `robot_manager.py:839-840`: `send_goal` returns without waiting
- `robot_manager.py:921-949`: `_deenergize_external` calls `follower.disconnect()`, which turns torque off

**Defect:** execute sends exactly one command through SafetyGuard, which allows at most 15° per command. Its `finally` block then de-energizes the follower in the same request.
- `max_duration_seconds` is never used.
- AgentLab never displays `result`; the string doesn't appear in `AgentLab.tsx`.

**Failure scenario:** the operator approves "home" with the arm raised.
- The joints move at most 15°; on hardware probably much less, because torque goes off milliseconds after the goal is written.
- The arm then goes limp and drops, while the UI shows a green "executed".
- "close_gripper" can never hold an object.

**Evidence** (`agent_execute_repro.py`):

| Skill | Target | Commanded |
|---|---|---|
| home (from pan 60 / lift −50 / elbow 40 / gripper 20) | all joints 0, gripper 50 | pan 45, lift −35, elbow 25, gripper 35.0 |
| open_gripper | gripper 80 | gripper 43.7 |

Both returned HTTP 200 with status "executed". Afterwards the follower was disconnected and torque was False.

**Status:** CONFIRMED in mock. The hardware timing is PLAUSIBLE: the mock bus eases toward the goal even with torque off, which hides it.

**Fix:**
- Run the skill as a short server-side loop at control rate, under the watchdog and cancellable, until it is within tolerance or `max_duration_seconds` expires.
- End by disarming logically but keep holding torque instead of disconnecting.
- Return `reached` and `final_error`, and show them in AgentLab.

### B2 HIGH: simulation FK uses the wrong zero pose, inverted signs and incomplete geometry

**Where:** `src/so101/simulation/kinematics.py:45-79` (chain at lines 50-55), limits at lines 16-18, and `engine.kinematics` in `scenarios.yaml`.

**Defect:** compared with LeRobot calibrated degrees on the official `so101_new_calib` model (LeRobot uses `deg2rad`, frame `gripper_frame_link`, no sign flips), the FK:
- uses the old-calibration "horizontal, fully extended" zero;
- inverts the sign of pan, shoulder_lift, elbow_flex and wrist_flex;
- leaves out the gripper (0.1605 m wrist_flex→gripper frame vs 0.06 m), the shoulder-lift axis height (0.1166 m vs 0.08 m) and the 0.0388 m pan offset.

**Failure scenario:** sim telemetry, end-effector positions and preview paths are off by 17-50 cm.
- Positive lift raises the sim tool but lowers the real one.
- Pan +45 goes left in sim and right on hardware.
- At lift −30 the sim reports z = −0.075 m (below the table); the real tool is at 0.373 m.

**Evidence** (`fk_compare.py`; the official URDF and MJCF agree to 0.00 mm):

| Pose | Error vs official |
|---|---|
| Zero: official (0.391, 0, 0.227) vs sim (0.31, 0, 0.08) | 168 mm |
| lift −30 | 449 mm |
| elbow −45 | 496 mm |
| pan ±45 | 496 mm |
| Best yaw+translation fit | mean 189 mm, max 363 mm |

For +10° on shoulder_lift, the official tool moves z −57.6 mm and the sim tool moves z +53.8 mm.

Limits also differ:

| Joint | Suite | Official |
|---|---|---|
| elbow_flex | ±100 | ±96.83 |
| wrist_flex | ±100 | ±95 |
| wrist_roll | ±160 | −157.21 / +162.79 |

**Status:** CONFIRMED.

**Fix:** generate the FK from the official URDF/MJCF (a numpy chain or `mj_kinematics`) and take limits from the MJCF. Add a golden pose-table test.

### B3 HIGH: the 3D twin draws a third arm that matches neither the FK nor the real robot

**Where:** `frontend/src/components/RobotTwin.tsx:284-320` (pitch joints use `rotation.z`), lines 352-362 (link chain), and lines 128-138 (`worldPosition`, `SCENE_SCALE = 8`).

**Defect:**
- At zero the twin points straight up.
- Pitch joints rotate about three.js z, which is backend x, so the arm swings in the backend y-z plane. That is perpendicular to the FK trajectory overlay and the scene objects, which lie in x-z.
- Its reach is about 0.44 m against the FK's 0.31 m.

**Failure scenario:**
- The preview path and the cube sit in a plane the drawn arm can never enter.
- On the "Real robot" source, live telemetry is drawn with the same wrong geometry, so the operator misreads the real arm.

**Evidence** (`twin_compare.py`):
- Twin tip at zero is (0, 0, 0.499): 522 mm from the backend FK and 477 mm from the official model.
- lift +30 moves the tip to y = −0.221, sideways.
- Pan has no effect at zero, because the arm sits on the pan axis.

**Status:** CONFIRMED.

**Fix:** drive the twin from the same kinematic source as the backend (official URDF in three.js, or link poses from the backend). Add a twin-equals-FK test.

### B4 MEDIUM: leaving out `sent_at` disables the TTL check for tracked motion

**Where:** `routers/tracking.py:132-133` (arm-first path) and `tracking.py:141-145` (execute path); both use `req.sent_at or datetime.now()`.

**Defect:** stale or replayed commands pass the `ttl_ms` and watchdog freshness gate if `sent_at` is omitted.

**Failure scenario:** a client stuck behind a network stall, or any script, sends a 10-second-old intent without `sent_at`, and the arm moves. The browser always sends `sent_at`; the API contract does not require it.

**Evidence** (`tracking_repro.py`):
- With `sent_at` 10 s old: 409 "stale command (10001 ms old; limit 300 ms)".
- Same body without `sent_at`: 200 "preview accepted and bounded command executed", on both the execute and the arm-and-execute paths.

**Status:** CONFIRMED.

**Fix:** require `sent_at` whenever `execute` or `arm_first` is set, like `lease_id` and `sequence`.

### B5 MEDIUM: camera tracking always drives the elbow to 0°

**Where:** `frontend/src/lib/handTracking.ts:121` (`z: clamp(-wrist.z*4)`), sent at `Connect.tsx:193`, mapped at `tracking.py:70` (z → elbow_flex × 70).

**Defect:** MediaPipe's z is depth relative to the wrist, so `wrist.z` is about 0. Every tracked command therefore targets elbow 0°.

**Failure scenario:** when tracking is engaged from a folded rest pose (elbow ≈ 90°), the elbow unfolds at up to 15° per command (~225°/s at ~15 Hz) with no operator input. The camera can never control the elbow.

**Evidence:** the MediaPipe Hand Landmarker docs ("depth at the wrist being the origin") plus the code path above.

**Status:** CONFIRMED from spec and code; not run on a camera.

**Fix:** leave z out of the monocular request, or derive depth from palm width relative to a reference captured when the deadman is engaged.

### B6 MEDIUM: the wrist-roll estimate sits on the atan2 branch cut at the natural hand pose

**Where:** `handTracking.ts:110-113` (`atan2(pinkyMcp - indexMcp) / π`), scaled × 140 at `tracking.py:72`.

**Defect:** frames are not mirrored (`Connect.tsx:113`, `facingMode: "user"`, no flip before `detectForVideo`).
- For a right palm facing the camera, dx < 0 and dy ≈ 0, so atan2 is about ±π.
- The wrist_roll target flips between roughly +140° and −140° with a few degrees of tilt or noise.

**Failure scenario:** with the hand held still, the wrist swings up to 280° back and forth at 15° per command.

**Status:** CONFIRMED by geometry and code.

**Fix:** measure roll relative to a neutral captured at engage, unwrap it, and clamp it. Use a consistent mirror convention (see B10).

### B7 MEDIUM: "human approval" is a boolean the client asserts in the same request

**Where:** `agent.py:122-123` and `schemas.py:581-583` (`{lease_id, approved: bool}`).

**Defect:** nothing ties approval to a human, a separate step, or a different principal from the proposer.

**Failure scenario:** any process that can reach the API and take a lease can propose, approve and execute in two requests, including a future automated agent loop.

**Evidence:** `agent_execute_repro.py` does exactly this with no UI and gets 200 "executed".

**Status:** CONFIRMED.

**Fix:** a single-use approval token from a separate endpoint that requires a UI deadman gesture or operator authentication. Bind it to the proposal id and a hash of its targets, give it a short TTL, and reject it when approver equals proposer.

### B8 LOW-MEDIUM: a cancel during execute leaves a moved proposal marked "cancelled", with no agent-level record

**Where:** `agent.py:166-173` and `agent.py:188-199`.

**Defect:** cancel can move a proposal from executing to cancelled between `send_goal` and execute's own transition to executed.
- Execute then returns 409 after the motion was already commanded.
- No `agent.executed` event is written; only the 1 Hz-throttled `control.command` event remains.

**Evidence** (`agent_race_repro.py` (a), with a 0.6 s delay injected):
- cancel returned 200 "cancelled"; execute returned 409 "proposal execution state changed unexpectedly".
- Final status is "cancelled", there are no `agent.executed` events, and the gripper goal moved to 15.0.

**Status:** CONFIRMED, using an injected delay.

**Fix:** if the final transition fails, still persist the commanded result and the event. Alternatively, serialize cancel and execute with a per-proposal lock.

### B9 LOW: a non-control exception leaves a proposal stuck in "executing"

**Where:** `agent.py:150-165`. Only `ControlConflict` and `ControlRejected` from the `try` block are caught. A `ControlRejected` raised by the strict disarm inside `finally` also escapes.

**Evidence** (`agent_race_repro.py` (b)):
- An injected `RuntimeError` gives HTTP 500.
- The proposal stays in "executing" with result `{approved: true}`.
- A retry gets 409; only cancel recovers it.

**Status:** CONFIRMED.

**Fix:** add `except Exception` that moves the proposal to failed, and call disarm with `strict=False` in `finally`, then handle the outcome explicitly.

### B10 LOW: the camera path assumes mirrored frames but processes raw ones

**Where:** `handTracking.ts:90-100` (prefers "right"), `handTracking.ts:119` (x), compared with `QuestWebXRPanel.tsx:114`.

**Defect:**
- MediaPipe's handedness labels assume a mirrored image. With two hands visible, "prefer right" picks the user's left hand.
- On the camera path +x is the user's left; on WebXR +x is the user's right. The same gesture produces opposite pan signs.

**Status:** CONFIRMED from spec and maths. Low impact, since WebXR is preview-only.

**Fix:** use x' = 1 − x (or swap labels) and define one operator-frame convention with a sign test for each path.

### B11 LOW: the tracking confidence gate measures the wrong signal

**Where:** `handTracking.ts:118` sends the handedness classification score as confidence. `QuestWebXRPanel.tsx:112` sends `poseCompleteness`. `tracking.py:19` and `tracking.py:89` gate on 0.72 and trust the client's value.

**Status:**
- Camera path: CONFIRMED (the handedness score is documented as handedness probability).
- WebXR path: PLAUSIBLE (pose completeness ≈ 1 whenever the runtime infers occluded joints).

**Fix:** gate on presence plus temporal-stability signals, and add server-side rate-of-change plausibility checks.

### B12 LOW, PLAUSIBLE (high impact if true): the strict tool schema uses an undocumented keyword

**Where:** `planner.py:114` (`"maxLength": 500` together with `strict: True` at line 109).

**Defect:** the current OpenAI Structured Outputs docs:
- list only `pattern` and `format` as supported string keywords;
- say an unsupported schema under `strict: true` "will receive an error";
- explicitly exclude `maxLength`, `minimum` and `maximum` for fine-tuned models.

**Failure scenario:** with a fine-tuned model (possibly any model), every call gets HTTP 400, then `PlannerError`, then 503. CI can't catch it because `tests/test_agent_planner.py` uses a mock transport.

**Status:** PLAUSIBLE. Confirmed by the docs for fine-tuned models; untested against a base model because no API key was available.

**Fix:**
- Drop `maxLength` (line 170 already enforces it; truncate rather than reject).
- Add `"parallel_tool_calls": False`, since the parser requires exactly one call.
- Add `"store": False`.

### B13 LOW: the configured planner is offered in the UI but always returns 503

**Where:** `planner.py:183` sets configured to key and model only; `AgentLab.tsx:272-273` enables the option from that; `planner.py:200-203` refuses unless the provider is `openai_responses`.

**Defect:** with the default provider "deterministic" and a key and model set, the option is enabled and every request returns 503.

**Status:** CONFIRMED by code path.

**Fix:** also require `agent_provider == "openai_responses"`.

### B14 LOW: malformed planner responses become 500

**Where:** `planner.py:156-160`.

**Defect:**
- A non-dict body or output item raises an `AttributeError`, which becomes 500 instead of `PlannerError` → 503.
- `body["status"]`, such as "incomplete", is ignored.

**Status:** CONFIRMED with a mock transport.

**Fix:** add isinstance checks and treat any status other than "completed" as an error.

### B15 LOW: substring keyword routing turns negations into motion

**Where:** `schemas.py:550-578` and `planner.py:51-58`.

**Defect:**
- "don't open the gripper" routes to open_gripper.
- "spark" routes to home (it contains "park"), and so does "homework".
- Human approval mitigates this, but see B7.

**Status:** CONFIRMED.

**Fix:** match whole words, guard against negation, and send anything ambiguous to inspect_scene.

### B16 LOW: "Preview current target" does not predict Play

**Where:** `engine.py:326-337` (linear ramp over `duration_s`) compared with `engine.py:118-145` (step target at per-joint max velocity, uncoordinated), wired at `Simulation.tsx:254-275`.

**Evidence** (`engine_repro.py` 4b): pan to 30° over 2 s shows 3.75° at 0.25 s in preview. Play reaches 22.5° at 0.25 s and arrives at 0.33 s.

**Status:** CONFIRMED.

**Fix:** have Play execute the previewed trajectory, or have preview simulate step targets.

### B17 LOW: `sample_hz` limits differ between layers

**Where:** `schemas.py:447` allows up to 240; `engine.py:285-288` rejects anything above `fixed_hz` (60) with a different 422. The UI caps it correctly.

**Status:** CONFIRMED.

**Fix:** validate in one place.

### B18 LOW: randomized scale does not re-seat objects on the table

**Where:** `scenarios.py:276-283` scales the size but leaves z unchanged.

**Evidence:** over 2000 seeds the cube bottom ranges from −2.0 mm to +2.0 mm (`engine_repro.py` 6). It penetrates or floats, which becomes an initial "pop" once there is physics.

**Status:** CONFIRMED.

**Fix:** add `z += (scale - 1) * size_z / 2`.

### B19 LOW: NaN or Infinity in JSON returns 500 instead of 422

**Where:** global.

**Defect:** the 422 body echoes the non-finite input and rendering it fails. Nothing reaches motion, because `_number` and the finite validators drop these values.

**Status:** CONFIRMED (`tracking_repro.py` step 7).

**Fix:** a `RequestValidationError` handler that sanitizes `input`.

### B20 LOW: audit throttling is keyed on a client-chosen source

**Where:** `tracking.py:24-33`, which also clears the dict once it passes 256 sources.

**Evidence:** rotating `source` produced 300 events from 300 requests. The clear also resets every other client's throttle.

**Status:** CONFIRMED.

**Fix:** key on the lease or client address and use an LRU with TTL.

### B21 LOW, PLAUSIBLE: WebXR lifecycle gaps (preview-only)

**Where:** `questWebXR.ts:536-608`.

**Defect:**
- `disposed` and `stop()` are not re-checked after `makeXRCompatible`, `updateRenderState` or the reference-space request. The result is a spurious error status, or a renderer allocated after cleanup.
- There is no `XRReferenceSpace` "reset" listener, so a recenter silently invalidates the calibration origin.
- `emulatedPosition` is ignored.

**Status:** PLAUSIBLE; needs a headset to confirm.

**Fix:** check a generation token after each await, clear calibration on "reset", and down-weight emulated poses.

### Motion-path check

Every hardware motion path needs a server-enforced lease:

| Path | What else is enforced |
|---|---|
| Tracked motion | Arm, deadman, matching source, strictly increasing sequence, authority epoch, 500 ms watchdog and a server-side clamp. Freshness is checked only when `sent_at` is present (B4). |
| Agent motion | `approved: true`, which the client asserts (B7). Targets come from a closed server-side skill table; the model never emits joint values. |
| Simulation endpoints | Never touch hardware. |
| Quest WebXR | Never takes a lease. |

Replays and double execution are blocked: four concurrent executes gave one 200 and three 409s (`agent_race_repro.py` (c)).

## 2. Improvements (ranked by value/effort)

1. **One kinematic truth from the official SO-101 assets (S/M).**
   - Vendor `so101_new_calib.{urdf,xml}` and its meshes (Apache-2.0).
   - Generate the backend FK from them; the MJCF zero-pose `gripperframe` (0.3914, 0, 0.2265) matches the URDF.
   - Render the same model in RobotTwin, and share a golden pose table between pytest and frontend tests.
   - Fixes B2 and B3.
2. **Server-side bounded skill executor for agent proposals (M).**
   - A cancellable trajectory at control rate, under SafetyGuard and the watchdog, that holds torque at the end.
   - Report reached and final error, and show commanded against reached in AgentLab.
   - Fixes B1, B8 and B9.
3. **Approval token bound to the proposal (S).** Single-use, short TTL, minted by a separate endpoint that needs a UI gesture or authentication, and bound to the target hash. Makes approval a server-enforced property (B7).
4. **MuJoCo physics backend behind the existing `/api/simulation` contract (M; L if grasp contact needs tuning).**
   - Measured with mujoco 3.14 on this Mac:
     - The official `scene.xml` loads with a 2 ms timestep.
     - It has 6 position actuators (kp 998.22, kv 2.731, forcerange ±2.94/±3.35), and their control ranges equal the official limits.
     - A 30° multi-joint step settles within 1° in 0.24 s, with 0.02° steady-state error under gravity.
     - 3 s simulates in 15 ms, about 200× real time.
   - Integration:
     - Convert LeRobot degrees with `deg2rad`; map the gripper's 0-100 onto −10°..100°.
     - Keep the request-synchronized fixed-step scheduler.
     - Set the timestep to 1/600 s so each 1/60 s tick is exactly 10 substeps.
     - Seeds and step-equals-play carry over. Capabilities already has an `adapter_implemented` slot.
   - Unlocks: gravity, contact and grasp outcomes, realistic actuator lag, and hardware-free tests of agent skills and tracking.
5. **Planner hardening plus a live smoke test (S).**
   - Drop `maxLength`, and add `parallel_tool_calls: false` and `store: false`.
   - Validate the body type and status, and fix `planner_status`.
   - Add an opt-in `pytest -m live_openai` test, because a mock transport can't catch schema rejection (B12-B14).
6. **Rework retargeting in the operator frame (M).**
   - Recenter at engage, so the mapping is relative.
   - Use one mirror convention for camera and WebXR, unwrap roll, and drop or derive the camera's z.
   - Implement the smoothing (`alpha` 0.28) and deadband that `configs/tracking/retargeting.yaml` declares but nothing reads. Both that file and `configs/agent/skills.yaml` are read only by `tests/test_config_assets.py`.
   - Load gains and thresholds from the config.
   - Fixes B5, B6, B10 and B11.
7. **Rendered cameras, simulated LeRobot datasets, and a closed collect→train→eval loop on physics (L).**
   - The official MJCF defines no `<camera>` (ncam = 0). The `_camera` variant has a `wrist_camera` body but no camera element, so add a suite-owned wrapper with front and top cameras plus a wrist camera.
   - Offscreen `mujoco.Renderer` at 640×480 takes 15.7 ms per frame headless on this Mac. Two cameras at 30 fps fit; 320×240 leaves headroom.
   - Write episodes with the same LeRobotDataset features as the real arm (6-D state and action plus images), so the existing train/eval path runs unchanged.
   - The seeded randomization becomes domain randomization, and eval becomes an object-pose success predicate.
   - Unlocks: real success rates, policy regression in CI (ACT, SmolVLA, diffusion), sim-versus-real A/B, and hardware-free demos.
8. **Make Preview and Play share one meaning (S).** Have Play execute the previewed waypoints, or have preview simulate step targets (B16).

## 3. What is genuinely good and must be preserved

- **Engine** (`engine.py`):
  - The closed-form fixed-step advance, with a scheduler that keeps the leftover sub-step time, gives identical results for a single sync, 97 chunked syncs and random chunks.
  - `step(90)` equals 1.5 s of play, and a 30-day catch-up takes 0.04 ms.
  - Locking is careful: reset builds the scene outside the lock, so a bad scenario leaves state intact.
  - Preview is snapshot-then-compute and verified not to change state. Validation rejects non-finite, out-of-limit and boolean values.
  - An 8-thread hammer test raised no exceptions.
- **Scenarios:**
  - Each scenario's RNG is seeded from SHA-256.
  - The same seed gives the same scene across instances and 500 reset/play cycles.
  - Manifest validation is strict.
- **Honest labelling:**
  - The kinematic model is labelled approximate and `physics_active` is False.
  - MuJoCo is probed without being imported.
  - Config comments say to replace the model with a versioned URDF/MJCF before claiming accuracy.
- **Tracked-motion control plane:**
  - Lease, source match, sequence, TTL, deadman, authority epoch, watchdog and server clamp.
  - The first command is validated before arming.
  - The browser stops on pointerup, cancel, blur, visibility change, low confidence, a stale sample (350 ms) or any rejection.
  - In-flight samples are dropped rather than queued.
  - Clock-offset estimation refuses round trips over 250 ms, and generation guards cover tracker start and stop.
- **Agent:**
  - Proposal-only, with a strict enum-bounded tool, forced `tool_choice`, exactly one call, and server-side re-validation.
  - Targets come from a closed server-side table, and the base URL must be HTTPS or loopback.
  - Proposals expire after 120 s, the proposed→executing claim is atomic, and `require_deadman` is rejected rather than faked.
- **WebXR:**
  - Preview-only by construction.
  - `requestSession` is the first await, which preserves user activation.
  - The reference space falls back from local-floor to local to viewer.
  - Cleanup removes every listener, and an epoch guard drops stale previews.

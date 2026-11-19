# Control-plane safety

This page explains how the SO-101 Suite server stops the arm, what it
remembers across a restart, and how it knows whether the motors are still
energized. It covers:

- the stop categories (E-stop, Halt, Collapse) and the single stop dispatcher
- Halt (freeze in place) and Collapse (slow fall to the rest pose) as built:
  the hold timer, supervision, the descent, fallbacks, precedence, settings
- the E-stop path and why it answers even when the server is overloaded
- the safety latches and their persistence
- three-state torque truth, from the manager down to each motor
- recovery after a torque-off that could not be confirmed
- leases, owner-scoped stops and command freshness
- the fault-injection harness that tests all of this against the real classes

Who may send a command at all (the Origin/Host guard, eval gates, agent
approval) is covered in [control-plane-security.md](control-plane-security.md).
The stop contract is §5 of the
[v0.2 design](plans/2026-10-01-suite-v02-design.md). Bug IDs refer to the v0.1
audit reports in `docs/audit/v0.1/` (`safety.md`, `sim.md`, `data.md`,
`devex.md` and `frontend.md`); `safety-B2` is bug B2 in `safety.md`. Contract
IDs (C2, C4 to C7, C9 to C11) are the v0.2 cross-stream API contract.

## Invariants

Every change in this area keeps these:

- `RobotManager` is the only owner of the motor buses. Every bus transfer runs
  while the manager holds its bus lock (`_io_lock`).
- Every motion path goes through a lease, arming, the deadman and
  `SafetyGuard`.
- Real mode fails closed. There is never a mock fallback, and the sim backend
  never opens a real bus (see [The sim backend guard](#the-sim-backend-guard)).
- The E-stop latches before anything else happens.
- The lock order is `_io_lock`, then the state lock `_lock`. `_lock` only ever
  guards short bookkeeping, never I/O, joins or sleeps.
- The authority epoch is checked again after any I/O, and a command's sequence
  number is reserved before its I/O.
- Teardown takes `_io_lock` with a timed acquire, so a wedged driver call can
  delay a stop but never block it forever.
- The watchdog runs on its own thread (`so101-watchdog`), and the servo-health
  poll on another (`so101-health`), so a slow health read never delays
  supervision.
- Diagnostics and calibration borrow a bus only through the manager: bus
  diagnostics (`run_idle_bus_diagnostic`) only while the robot is idle and
  read-only, calibration (`run_calibration`) on the manager's own bus. Both
  run under `_io_lock`.
- No stop de-energizes at the end of an episode, a loop or an agent skill:
  those Halt, and the arm holds (design doc §5). Torque goes off only through
  an E-stop, a Collapse's finish, or a category 0 fallback with a fault.
- Every Halt or Collapse failure (I/O, timeout, stall, guard rejection) ends
  in torque off with a visible fault, or `torque_unconfirmed` when even that
  cannot be confirmed. None reports success.
- All stop timing (hold timer, grace period, descent pacing, stall and
  timeout) reads the injected clock `_clock` and sleeps through `_sleep`.

## Stop categories

| Stop | Meaning | IEC 60204-1 | `StopKind` |
| --- | --- | --- | --- |
| E-stop | Kill all motors now. Latched; only a deliberate reset clears it. | Category 0 | `ESTOP` |
| Halt | Freeze in place, motors on. | Category 2 | `HALT` |
| Collapse | Lower to the rest pose at a capped speed, then motors off. | Category 1 | `COLLAPSE` |

All three are implemented behind the dispatcher; `GET /api/robot/status`
reports `stop_kinds: ["estop", "halt", "collapse"]`. Each has a route, a CLI
command and a Make target:

| Stop | Route | Origin guard | CLI | Make |
| --- | --- | --- | --- | --- |
| E-stop | `POST /api/robot/estop` | exempt | `so101 estop [--url]` | `make estop` |
| Halt | `POST /api/robot/halt {"operator_id"?, "reason"?}` | exempt | `so101 halt [--url]` | `make halt` |
| Collapse | `POST /api/robot/collapse {"operator_id"?, "reason"?}` | guarded (it commands motion) | `so101 collapse [--url]` | `make collapse` |

The stop routes (`/estop`, `/stop`, `/halt`) also skip the strict JSON body
check: a stop is never refused over its body. The halt parses its body
leniently; anything but a JSON object with string fields is ignored, and the
halt still happens. Nor does a stop ever wait on its body: E-stop and the
legacy stop read none, and the halt reads at most 4 KiB within 0.1 s
(`read_stop_body` in `services/guard.py`). A longer body (a declared
`Content-Length` over the cap is not read at all) or a slower one counts as
no body, so the halt uses its default reason and cannot renew the hold
timer, but it halts at once and buffers nothing. Halt answers 200 while the arm holds (or its motors are
off) and 503 when it fell back to a torque cut. Collapse answers 200 once the
descent has started (or the motors were already off), 503 when it fell back
at once, 403 cross-origin and 422 for a bad body. `so101 halt` and
`so101 collapse` print `HALT NOT CONFIRMED by <url> — use \`so101 estop\` or
the power switch` (or `COLLAPSE ...`) to stderr and exit 1 on anything but a
2xx.

### The dispatcher

Every stop goes through one method on `RobotManager`:

```python
request_stop(kind: StopKind, *, reason: str, lease_id: str | None = None,
             strict: bool = False, operator_id: str | None = None,
             origin_verified: bool = False) -> StopResult
```

- `kind` is a `StopKind` or its string value (`"estop"`, `"halt"`,
  `"collapse"`).
- `ESTOP` is always global and ignores `lease_id`.
- `HALT` and `COLLAPSE` with a `lease_id` are owner-scoped. If the id does not
  name the current lease, nothing changes and the result has `applied=False`.
  Without a `lease_id` they are global and also end loops and calibration.
- `operator_id` and `origin_verified` matter only to a repeat Halt: it renews
  the hold timer (see [Hold timer and supervision](#hold-timer-and-supervision)).
- `strict=False` never raises, so error paths can always call it.
  `strict=True` raises `ControlRejected` when the stop was not applied, fell
  back to a torque cut, left the arm unaccounted for, or a loop did not exit.

`StopResult` is a frozen dataclass:

| Field | Meaning |
| --- | --- |
| `applied` | False when the named lease or authority epoch was stale; nothing changed |
| `loop_stopped` | False when a loop did not exit in time (a fault latched) |
| `torque_off` | torque confirmed off afterwards |
| `holding` | a Halt holds position, motors on |
| `collapsing` | a Collapse descent is running (it ends with its own torque-off) |
| `fallback` | why a Halt or Collapse fell back to an immediate torque cut |

`safe` is `holding or collapsing or torque_off`; `ok` is `applied`,
`loop_stopped`, no `fallback`, and `safe`. `estop()` is
`request_stop(ESTOP, strict=True)`, `soft_stop()` is a global `HALT`, and
`servo_health_critical(detail)` is a fault-latching `COLLAPSE`. The deprecated
v0.1 `disarm_control` shim is gone; every caller uses `request_stop`.

### Which stop fires when

| Trigger | Stop | Scope | Where |
| --- | --- | --- | --- |
| `POST /api/robot/estop`, `so101 estop`, `make estop`, the UI E-stop | E-stop | global | `routers/robot.py` `estop` |
| Bus I/O failure, hung I/O, torque read-back failure | E-stop-equivalent fault | global | `_set_fault` + `_cut_torque` at each I/O site |
| `POST /api/robot/halt`, `so101 halt`, `make halt` | Halt | global; renews the timer for a supervising operator | `routers/robot.py` `halt` |
| Legacy `POST /api/robot/stop`, `soft_stop()` | Halt | global | `routers/robot.py` `stop`, `RobotManager.soft_stop` |
| End of teleop, recording, rollout (`POST .../stop` routes call `soft_stop()`) | Halt | global | `routers/teleop.py`, `recording.py`, `models.py` |
| End of an eval episode (`stop()`) | Halt | loops only | `RobotManager.stop`, `routers/eval.py` |
| Command watchdog (no fresh command within `watchdog_ms`) | Halt | authority epoch; shows `watchdog_timeout` | `_watchdog_tick` |
| Deadman released: commands without the deadman are refused, so the watchdog trips | Halt | authority epoch | `_watchdog_tick` |
| Lease expiry | Halt | owner; ends the lease; shows `lease_expired` | `_expire_lease_if_needed` |
| Lease release (`DELETE /api/control/lease/{id}`) | Halt | owner; ends the lease | `release_lease` |
| Error-path cleanup in the control and tracking routes (safety-B6) | Halt | owner, non-strict | `routers/control.py` `arm_command`, `routers/tracking.py` |
| Arming found the bus busy | Halt | authority epoch | `arm_control` |
| Agent skill completes, is cancelled or times out | Halt | owner (the run's lease) | `agent_executor.py` `_stop_motion` |
| Operator heartbeat lost while a loop runs | Halt, then the grace hold | loops only | `_watchdog_tick` |
| `POST /api/robot/collapse`, `so101 collapse`, `make collapse` | Collapse | global | `routers/robot.py` `collapse` |
| `POST /api/control/disarm {lease_id}` ("done, power down") | Collapse | owner, strict | `routers/control.py` `disarm` |
| Halt hold timer expires, or supervision lost past the grace period | Collapse | global, only while that halt is active | `_supervise_stops` |
| Service shutdown (SIGTERM, app lifespan end) | Collapse, bounded, then torque off | global | `RobotManager.shutdown` |
| Servo health critical | Collapse + fault | global | `RobotManager.servo_health_critical` |

Internal triggers pass their scope to `_dispatch_stop` (`epoch`,
`trip_state`, `end_lease`, `loops_only`, `fault`, `expected_halt`). A latched
state (E-stop or fault) always wins over a trip state, and an owner-scoped
stop leaves a trip such as `watchdog_timeout` visible until its own reset
path clears it (C5).

The servo-health poll (`so101-health`, docs/bring-up.md "Servo health
monitor") calls `servo_health_critical` on a critical verdict. It reads under
`_io_lock` with a 50 ms bounded wait and grades after releasing it, so the
Collapse takes the bus at once; it does not dispatch again while that health
Collapse runs, or while an E-stop owns the arm.

## Halt

Category 2: freeze in place, motors on (`_halt_stop`, `_hold_in_place`).

1. **Every motion source stops.** A global halt cancels calibration, freezes a
   running Collapse (see [Precedence](#precedence)) and ends the legacy loop
   at its next tick (3 s join). A loop that ends because a stop asked it to
   leaves the follower energized for the halt to hold; the passive leader is
   released. A loop that failed, or ended under an E-stop or a fault, gets
   category 0 instead.
2. **Authority is disarmed**: armed off, the epoch bumped, the deadman reset.
3. **The hold**, under the bus lock:
   - Arm joints: Goal := Present, freshly read.
   - Gripper: `goal = clamp(last_goal, present − grip_hold_margin, present)`,
     kept inside the guard's gripper range but never above `present`. It
     never opens further, so a held object is not dropped, and keeps at most
     a small squeeze so an existing grasp holds. `last_goal` is the last goal
     the follower wrote.
   - The hold is written straight to the servos: it is not a motion command,
     so the guard's rate limit does not apply.
4. **Motors already off**: an ok no-op. A halt never energizes anything.
5. **Under a latched fault**: a halt never holds; it cuts torque (the fault
   is category 0).

**Fallback (category 0).** If the bus does not come free, the position
cannot be read, the goal cannot be written, or the hold's I/O takes longer
than `stop_io_timeout_s` (0.5 s), torque goes off on every bus and a fault
latches ("Halt could not hold position ..."). If that torque-off cannot be
confirmed either, `torque_unconfirmed` latches (C2). `safety.halt` records it
at `critical` with the cause. The fault latches in memory before the torque
cut or either persistence write, so a live lease cannot re-energize the arm in
the interval after a confirmed cut.

While a halt holds, the watchdog samples the follower every 0.1 s (skipping
when the bus is busy), so telemetry stays live. A read that fails there is a
bus failure: fault and torque-off. A hold whose torque turned out to be off
(a closed port or a cut) clears itself.

**Re-arming** after a halt uses the normal lease → arm flow. Arming finds the
follower energized and takes the hold over with `SO101Follower.resume_hold()`:
it reads the pose and seeds the guard, with no torque write and no goal write,
so the arm neither toggles nor jumps. A new teleop, recording or rollout loop
also takes a hold over (its connect primes Goal_Position from the measured
pose). `POST /api/control/observe` on a halted arm reads the pose and keeps
the hold; it disconnects only an arm it found with torque off. Calibration is
refused while the follower is energized.

### Hold timer and supervision

- A halt holds for at most `SO101_HALT_HOLD_SECONDS` (60), then becomes a
  Collapse with the reason `halt hold expired after 60 s`.
- **Supervision** means at least one operator browser has a fresh heartbeat
  (`POST /api/robot/operator-heartbeat` within `operator_timeout_ms`, 2 s).
  While no browser supervises, the hold lasts only
  `SO101_SUPERVISION_GRACE_SECONDS` (5), counted from when the newest
  heartbeat went stale (or from the halt, if later), and never past the hold
  timer. The Collapse reason then says `supervision lost: ...`. If a browser
  comes back first, the hold timer applies again.
- **Renewal.** A repeat halt keeps the hold it finds. It renews the timer to
  a full `SO101_HALT_HOLD_SECONDS` only when it names an `operator_id` with a
  fresh heartbeat and the request passes the Origin guard (checked inline;
  the route itself is exempt). Anything else is still an ok halt that changes
  nothing.
- The watchdog (`_supervise_stops`, on `so101-watchdog`) checks the timer on
  every pass against the injected clock. A Collapse it starts applies only
  while that same halt is still active, so a re-arm or a renewal in between
  wins.

## Collapse

Category 1: a slow fall to the rest pose, then motors off (`_collapse_stop`,
`_run_descent`).

1. **Every motion source stops** and authority is disarmed, as for a Halt.
2. **Motors already off**: an ok no-op. Not latched, nothing to lower.
3. **The plan**, in the request: a fresh read of Present under the bus lock,
   the guard seeded from it, and the resolved [rest pose](#rest-pose). A bus
   that is already dead, or an arm more than 30° outside the guard envelope,
   fails the request itself (fallback `bus_error` or `guard_rejection`).
4. **The descent** runs on its own thread, `so101-collapse`, so the request
   returns at once and a Halt or an E-stop can preempt it. It interpolates in
   joint space from Present to rest, all joints arriving together: the joint
   with the farthest to go moves at `SO101_COLLAPSE_SPEED_DEG_S` (15), the
   others slower. Each tick (`collapse_rate_hz`, 30 Hz, paced by deadline
   scheduling on the injected clock) goes through the SafetyGuard (limits
   and rate, `dt` = one tick) but not the lease. The gripper follows the Halt
   grip rule for the whole descent.
   **A late tick never catches up.** The SafetyGuard's own rate limit is far
   above the collapse speed, so the descent caps itself. Its trajectory runs
   on its own time, which advances by the real interval since the last tick
   but by at most one period per tick. A tick that waited for the bus (up to
   `stop_io_timeout_s`) or was delayed by a scheduler stall therefore moves
   the setpoint by one period's worth, not by the time it lost. Each joint's
   step is also clamped to its share of the cap (speed / rate, scaled per
   joint like the plan), measured from the goal the servos last accepted, so
   a goal the driver clipped is not made up in one jump either. Delays make
   the descent longer, never faster.
5. **Bound.** `SO101_COLLAPSE_TIMEOUT_SECONDS`, by default distance / speed +
   2 s, at most 12 s. A set value may not exceed 12 s either. The bound runs
   on the real clock: a descent delayed past it (for example by a 2 s stall,
   which uses up the whole default margin) ends in the `timeout` fallback,
   torque off with a fault. That is the fail-safe for a descent that cannot
   keep its pace.
6. **Finish.** At rest (within 1°), or settled within 3° with no progress for
   0.5 s, torque goes off on every bus with read-back (C2). The descent counts
   as done; `safety.collapse.done` records the final error and the torque
   truth. An unconfirmed torque-off latches `torque_unconfirmed` and a fault.

**Fallback** to an immediate torque-off with a latched fault (or
`torque_unconfirmed` when the cut cannot be confirmed), recorded as
`safety.collapse.fallback` with the cause:

| Cause | When |
| --- | --- |
| `timeout` | the bound expired before the arm reached rest |
| `stall` | no progress (0.1° better) for 0.5 s with more than 3° to go |
| `bus_error` | a read or write failed, or the bus did not come free within 0.5 s |
| `guard_rejection` | the SafetyGuard refused a tick, or the plan's seed |
| `no_rest_pose` | no rest pose could be resolved (below) |
| `torque_unconfirmed` | the follower's torque was not known off, but nothing was there to lower |
| `error` | the descent thread could not start, or it raised an unexpected error |

**Latching.** A Collapse latches nothing by itself; after it ends at rest the
arm can be armed again at once. Only a Collapse that a fault triggered (servo
health) leaves that fault latched, and a fallback latches its own.

**No rest pose.** If the resolver raises (for example when the SafetyGuard
and MJCF ranges of a joint do not overlap), there is nowhere safe to lower
to: the arm is de-energized where it stands (a zero-distance descent) and a
fault latches (`no_rest_pose`).

**Backstop.** The watchdog cuts torque itself (cause `timeout`, "stuck in
I/O") if a descent is still running 1 s past its bound, for a descent thread
stuck in a driver call. Service shutdown waits for the descent at most its
bound plus a margin, then does the same, and the fault persists to the next
start.

## Precedence

- **E-stop preempts everything instantly.** The latch cancels a running
  descent lock-free; the descent checks for it under the bus lock before
  every write, so it never writes after the E-stop, and the E-stop's torque
  cut follows. `safety.estop` records `preempted: "halted" | "collapsing"`.
- **A Halt during a Collapse freezes it** (operator override): the explicit,
  global Halt (button, `so101 halt`, `POST /api/robot/halt` and `/stop`,
  `soft_stop`). The descent writes nothing after the freeze; the halt holds
  Goal := Present and starts a fresh hold timer. `safety.halt` records
  `froze_collapse` with the Collapse's reason. Owner- or epoch-scoped halts
  (lease release or expiry, the watchdog, error cleanup, an agent's end) only
  revoke authority and leave the descent running. A Collapse that a fault
  triggered is the fault's reaction: a Halt cannot hold under a fault, so it
  does not freeze one either.
- **A Collapse during a Halt proceeds.** A repeat Collapse joins the running
  one.
- **Nothing starts while E-stopped.** A Halt or Collapse then is an ok no-op
  (the motors are off). While a Collapse runs, arming, starting a loop,
  calibration and the fault reset are refused.

## Rest pose

Collapse resolves its target through the bring-up resolver,
`so101.hardware.rest_pose.resolve_collapse_rest_pose` →
`resolve_rest_pose`:

1. the pose recorded during bring-up (`so101 rest-pose record`), bound to the
   follower id, the backend and the calibration digest;
2. `SO101_REST_POSE`;
3. the model-derived default, `so101.simulation.rest_pose.MODEL_REST_POSE`
   (folded, collision-free, at least 5° inside the limits, gravity-stable;
   plain data, no MuJoCo import).

Every source is clamped to the active SafetyGuard limits intersected with the
MJCF limits (parsed from the vendored model, no MuJoCo import). If the model
limits cannot be read, the clamp uses the SafetyGuard limits alone and the
plan's warnings say so. The bring-up views (`so101 rest-pose show`, preflight,
the HIL runner) resolve through the same function, so they show exactly the
pose Collapse would use; the old mid-range placeholder is gone.

## Visibility

`GET /api/robot/status` and every telemetry frame carry (declared in
`services/schemas.py`):

| Field | Values |
| --- | --- |
| `torque_state` | `"on"`, `"off"` or `"unconfirmed"`, over every bus the manager owns: unconfirmed > on > off; the manager's `torque_unconfirmed` latch counts as unconfirmed |
| `stop_state` | `"none"`, `"halted"` or `"collapsing"` (E-stop and fault keep their own fields) |
| `halt` | `{since, reason, hold_expires_at, server_time}` or null; `hold_expires_at` is when the halt becomes a Collapse given supervision now; compare it with `server_time` |
| `collapse` | `{since, reason, progress, eta_s}` or null; `progress` is 0..1 of the distance to rest |

Status also carries `supervision: {supervised, lost_at, grace_s}` (an
addition to §5 that the HIL supervision probe reads). Status and telemetry
both carry `health: {level, active, arms, age_s, stale, reasons}`, the servo
health summary; `GET /api/robot/health` has the detail (docs/bring-up.md
"Servo health monitor"). Times are ISO-8601 UTC.
`stop_state` leaves `"collapsing"` only after the finish's event is written.
Telemetry's `observation_live` (C6) is also true while a halt's hold or a
descent samples the arm.

Events: `safety.halt` (hold, renewal or fallback), `safety.collapse.start`,
`safety.collapse.done`, `safety.collapse.fallback`, `safety.estop`, and
`safety.servo_health_critical` (with the arm and reasons). The health monitor
also emits `robot.health.warn`, `robot.health.ok` and
`robot.health.config_error`.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `SO101_HALT_HOLD_SECONDS` | 60 | longest hold before a Collapse |
| `SO101_SUPERVISION_GRACE_SECONDS` | 5 | the hold while no operator supervises |
| `SO101_COLLAPSE_SPEED_DEG_S` | 15 | per-joint speed cap of the descent |
| `SO101_COLLAPSE_TIMEOUT_SECONDS` | empty | the descent bound; empty = distance / speed + 2 s, at most 12 |
| `SO101_GRIP_HOLD_MARGIN` | 5 | the grip rule's largest squeeze, 0..100 gripper units |

All must be finite and positive; a set timeout may not exceed 12 s. The grip
margin is 5 % of the gripper's travel: above the 3-unit position tolerance
the agent executor uses for the gripper, so the squeeze is not lost in the
servo's deadband, and far below a full-close command, so a stalled grasp does
not overheat the gripper servo during a minute-long hold.

## The E-stop path

`POST /api/robot/estop` is an `async` handler (safety-B9):

1. It sets the latch inline on the event loop with
   `latch_estop(reason, lock_timeout=0.05)`. The guard flag is lock-free and set
   first, so every motion path rejects from that moment. The bookkeeping that
   needs the state lock (disarm, bump the epoch, drop the lease) is left to the
   teardown if that lock is not free within 50 ms.
2. It starts the teardown, `manager.estop()`, on a dedicated two-thread
   executor (`so101-estop`). The teardown never queues behind request handlers
   on the shared AnyIO thread pool.
3. It waits at most `ESTOP_RESPONSE_TIMEOUT_S` (3.0 s). 200 means torque-off was
   confirmed on every bus. 503 means the latch is set but torque-off is not
   confirmed (yet, or at all). The teardown keeps running after a 503.

A review reproduced the old failure with 45 slow requests holding every worker
thread: the E-stop took 8.99 s. The regression test fills the worker pool with
blocked requests, queues more behind them, and requires the E-stop to answer
in under a second.

The teardown, `_estop_category0`, runs in this order:

1. Latch (idempotent). A running Collapse descent is cancelled lock-free in
   the same step; it never writes again.
2. Torque off on every bus the manager owns, follower and leader, under
   `_io_lock`, before anything that can block. This ends a Halt's hold or a
   Collapse.
3. Persist the latches (after the cut, so a slow disk never delays it).
4. Stop the loop thread (0.5 s join).
5. If the first cut failed and the loop has stopped, retry once, reopening
   ports that closed with torque unconfirmed.
6. Cancel calibration, persist again, emit `safety.estop`.

Timing: one bus with all six motors silent takes about 1.2 s to give up (see
[Torque-off on the bus](#torque-off-on-the-bus)), so two silent arms take about
2.4 s and the retry can push the total past the 3 s bound. The answer is then
503 while the latch holds, which is the honest answer: torque-off was not
confirmed.

`POST /api/robot/estop` is exempt from the Origin guard
([control-plane-security.md](control-plane-security.md)).

### `so101 estop`

`so101 estop [--url URL]` posts to `/api/robot/estop` with a 5 s timeout. The
default URL comes from `SO101_HOST` and `SO101_PORT`; a wildcard bind maps to
`127.0.0.1`. Anything other than a 2xx answer, including a connection error or
a timeout, prints `E-STOP NOT DELIVERED to <url> — use the power switch` to
stderr and exits 1 (devex-B7).

### Requests never wait long for the bus

Request handlers validate their input first, then wait at most 250 ms for the
bus. If it is still held they answer 503 with
`motor bus is busy; <what> not sent (waited 250 ms); retry` (`ControlBusy`,
safety-B9). Nothing was sent, and the client may retry. `ControlConflict` and
`ControlRejected` are 409.

## Safety latches

Three things are latched, and only their own deliberate reset clears them:

- the E-stop (`estopped`)
- a runtime fault (`fault`), set by any bus failure, hung loop or unconfirmed
  torque-off
- `torque_unconfirmed` (see [Three-state torque truth](#three-state-torque-truth))

The store also carries `energized` write-ahead intent. It is not a
fourth operator stop: on restart it is converted to a fault and unconfirmed
torque, and it clears only after confirmed torque-off or an acknowledged
physical power cut.

`safety_state` is derived from them. Trips such as `watchdog_timeout`,
`lease_expired` and `operator_timeout` are shown, but they are not latches.

### Persistence (safety-B8, devex-B2)

The latches are written through to SQLite, in the control store at
`<SO101_DATA_DIR>/control_plane.sqlite3`, table `safety_latches`. Rows are
keyed by backend plus the configured follower and leader logical IDs; a mock
or sim rehearsal therefore cannot load, reset or out-version the real arm's
row even when it shares the data folder:

- Schema v2 stores `backend`, `follower_id` and `leader_id` with every
  snapshot. `ControlStore` rejects an unexpected schema or identity rather
  than guessing.
- The old singleton row had no identity. Migration preserves it as an
  unscoped **real** row, even when the first process after upgrade is mock or
  sim. Every real logical-device pair sees that conservative row until a
  newer deliberate real snapshot supersedes it; a rehearsal never does.
- `energized`, `energized_at` and `energized_by` are the data-folder mirror of
  real-arm write-ahead intent. On restart it becomes a visible fault plus
  `torque_unconfirmed` until recovery confirms torque off.

- Every snapshot carries a version, and the upsert only applies when the new
  version is higher, so a writer that was overtaken cannot roll the latches
  back. The version starts at wall-clock microseconds, so a process that could
  not read the stored version still writes above an earlier process.
- The commit runs with `synchronous=FULL`: a latch must survive a power cut,
  not only a crash. The database uses WAL and a 5 s busy timeout.
- A write that fails fast (an `OSError`, or a SQLite error other than busy or
  locked) is retried after 50 ms and 100 ms. On Windows an antivirus scanner
  can hold a freshly written file for a moment. A busy or locked database is
  not retried; SQLite already waited.
- A failed write never clears the in-memory latch.
- **The fallback marker.** When the store still refuses a snapshot that
  latches something (E-stop, fault, `torque_unconfirmed` or energized intent),
  the manager writes a small identity-bearing marker next to it. Every v2
  backend/device pair, including real pairs, has a disjoint marker name. The
  historical `<SO101_DATA_DIR>/control_plane.latch-unsaved` path is read
  separately and conservatively by real identities only; it is never removed
  by an ordinary identity save. A newer row supersedes its version only for
  that identity, so resetting B cannot erase A's only durable marker. Mock and
  sim never adopt it. Marker JSON carries a schema version, full identity,
  snapshot version, latches and the store error, written by temp-file + fsync
  + rename.
- An unreadable historical marker has no trustworthy version or identity, so
  automatic resets never delete it. It remains fail-closed across restarts.
  After physically isolating/checking the arm and completing deliberate fault
  recovery, the operator must quarantine that exact legacy file manually; do
  not replace or remove an identity-scoped v2 marker.
- If even the marker cannot be written, the latch holds in memory only:
  `safety.latches_unsaved` says so (`marker_written: false`) and
  `GET /api/robot/status` shows `latch_persistence: "unsaved"`. Do not
  restart the server before checking the arm.
- `latch_persistence` in the status is `"saved"` (the store has the newest
  latching snapshot), `"marker"` (the marker stands in for it, so a restart
  comes back faulted) or `"unsaved"`.

On start, before serving anything, the manager restores the row. A crash, a
clean restart and a `--reload` all come back latched:

- `estopped` latches the guard again.
- `torque_unconfirmed` without a fault becomes a fault ("torque-off was
  unconfirmed before the restart").
- `energized` becomes an unconfirmed fault: a dead owner may have left the
  follower holding with torque on, or the marker may conservatively retain an
  E-stop/fault across data directories after confirmed torque-off.
- A store that cannot be read starts the server faulted (fail closed) and emits
  `safety.latches_unreadable`.
- A marker starts the server faulted ("the last safety latch could not be
  saved before the restart ..."), with the marker's E-stop and
  `torque_unconfirmed` restored as well, and `safety.latches_restored` at
  `critical` carries it (`unsaved_marker`). An unreadable marker counts as
  `torque_unconfirmed`. A marker older than the stored row is obsolete and
  is removed. Otherwise the marker stays until a snapshot of the new process
  reaches the store (for example the fault reset's).
- Trips are not restored; they belonged to a lease that died with the old
  process.

### Deliberate resets (C2)

Both resets need the JSON body `{"operator_id": str, "confirm": true}`.
`confirm` must be the literal `true` (not `1` or `"true"`), and `operator_id`
must be 1 to 128 characters and not blank. Anything else is 422.

- `POST /api/robot/estop/reset` clears only the E-stop. A fault or an
  unconfirmed torque-off stays latched. 409 while a loop is active.
- `POST /api/robot/fault/reset` clears the fault. It accepts
  `"power_cut_acknowledged": true` as well; see
  [Recovery](#recovery-after-an-unconfirmed-torque-off).

A disarm or any other stop never overwrites a latched state (C5).

### Commands that run without the server

Commands that open a device without `RobotManager` pass
`so101.bringup.standalone.standalone_access` first:

| Backend | What the gate does |
| --- | --- |
| `mock` | Nothing. There is no arm, and the latches are not read. |
| `sim` | Reads the latches (below). The simulated arm opens no real bus, so no lock. |
| `real` | Refuses while anything answers `GET /api/robot/status` at `SO101_HOST:SO101_PORT`, then takes one host-global lock per canonical configured motor port and reads the data-folder and host-global latches. Calibration and bring-up diagnostics may proceed under that lock. Teleop, record, replay, run and eval are refused: real motion must run through `RobotManager`. |

Reading the latches means opening `<data_dir>/control_plane.sqlite3` through
the server's own `ControlStore`, selecting only the current backend/device
identity and merging its newer fallback marker under the same version rules as
server startup. It refuses while `estopped`, `fault`, `torque_unconfirmed` or
the standalone `energized` intent is set. An unreadable store or active marker
refuses too (fail closed, as the server starts faulted on either). A marker
older than the stored row is obsolete and ignored. No matching row or marker
means no latch was ever saved for that identity. The gate never clears a
latch.

A refusal prints `refused: <what>` and the way out to stderr and exits 3
(`EXIT_REFUSED`, the bring-up kit's code), for example:

```text
refused: the arm's safety latch is set (E-stop; fault: ...), saved <time> in <data_dir>/control_plane.sqlite3. A command run without the server does not clear or skip it
Clear it on purpose through the Suite: start the server (`so101 serve`) and use Reset in the UI, or send POST /api/robot/estop/reset, then POST /api/robot/fault/reset, each with the JSON body {"operator_id": "<your name>", "confirm": true}. Add "power_cut_acknowledged": true to the fault reset only after you cut the arm's power. Then stop the server (it holds the bus) and re-run this command.
```

Only the resets for what is latched are named, in the order the server
accepts them (the fault reset answers 409 while the E-stop is latched).

The gate sits where the arm is opened, so owned library motion is covered too:
`replay_episode` without a caller-owned `follower` and
`PolicyRunner.connect` raise `StandaloneRefused` before connecting on real
hardware. Passing a caller-owned real follower does not bypass the refusal.
Mock and simulation retain their CLI workflows; supported real operation keeps
all motion inside `RobotManager`.

This refusal is deliberate. A raw process cannot keep the software stop
listener, command watchdog, browser supervision grace and Halt-to-Collapse
timer alive after it exits. Real motion therefore goes through the Suite UI or
API. Calibration, preflight and rest-pose bring-up remain bounded standalone
operations under the same device locks.

A teardown that confirms torque off durably clears both records. It checks
torque truth even when the host port was already closed; `on`, `unconfirmed`
or unreadable truth durably saves a fault and `torque_unconfirmed` (using the
fallback marker if SQLite fails) and refreshes the global intent before
releasing the device locks. A partial global write or clear remains fail-closed:
any energized identity token wins, and malformed or mismatched content is
treated as energized. A follower supplied by a library caller remains
caller-owned and is not disconnected or latched.
If neither the local store/marker nor the host-global marker can preserve an
unconfirmed torque result, the live process refuses to release or re-enter the
device locks; cut physical power and end the command process rather than
continuing.

**The server's device locks.** On the real backend the app lifespan resolves
the configured follower and leader ports, takes their host-global locks before
it creates `RobotManager`, and releases them after the manager's shutdown stop.
Existing POSIX devices lock configured/canonical path tokens, `st_rdev` for a
character device (or device+inode for the regular-file test seam), and the
captured node inode. A udev alias, hardlink, duplicate character node and its
current `/dev/tty*` target therefore collide. Replacing or reusing a tty path
does not inherit the old authority: its node fingerprint is revalidated just
before and just after driver open, before configuration, goal or torque I/O.
Resolved devices share a namespace gate and then take independent per-device
locks. A missing/dangling port takes the gate exclusively and cannot be opened
until the owner restarts with the adapter present. Locks are acquired in sorted
identity order so overlapping two-arm configurations cannot deadlock,
while disjoint ports can proceed. Per-device contention names the holder PID,
purpose and start time plus the canonical port and how to stop or wait for it.
Contention on the conservative unresolved-device namespace gate says `another process`:
several resolved owners may legitimately share that gate, so naming one stale
holder would be misleading. Mock and sim servers and commands take no
motor-device lock.

The installer creates one durable shared namespace outside temporary cleanup:
`/var/lib/so101/device-ownership-v2` on Linux and
`/Library/Application Support/SO101/device-ownership-v2` on macOS, root-owned
by the hardware group with mode 3770 (setgid+sticky). Lock records are 0660;
immutable journal records are 0640. Intent and clear events are published as a
new temp file, fsynced, linked/renamed without overwrite, and directory-fsynced;
Suite code never truncates or deletes an active record. After a stable scope is
fully cleared, ordinary history is compacted intent-first, directory-fsynced,
then clear-record-last, so a crash cannot resurrect an intent. Complex alias,
partial-clear and corrupt-recovery history remains immutable. Stable alias descriptors
and intent alias history make both a retargeted alias and its direct current
target see retained intent. A scoped clear retires exact event facets; ordinary
confirmed-off recovery rejects an old endpoint mismatch, while acknowledged
power-cut recovery may clear only current alias facets and leaves the old
endpoint active. Corrupt current-device generations can be safely retired
without clearing unrelated device records.

The real RobotManager writes the same intent, plus its identity-scoped SQLite
mirror, before manual arming, teleop/recording connect or policy-rollout
connect may enable follower torque. The write runs under the manager's
device-global ownership but before it acquires `_io_lock`, so slow storage
cannot delay an E-stop torque cut; authority/stop state is checked again under
`_io_lock` before the torque write. A marker failure prevents enable and
latches a visible fault. Mock and sim never touch this host-hardware state.
Real bus open itself requires an adopted `DeviceBusLock`, and raw torque-enable
additionally requires a freshly published global-intent capability. Calibration and diagnostics do not enable
torque; an unconfirmed cleanup still publishes recovery intent.

Halt deliberately retains intent. A fault-free Collapse or shutdown clears it
only after the manager confirms the relevant buses off. E-stop and fault keep
a conservative follower-scoped host-global sentinel even after that confirmed
cut; their deliberate reset retires it. A failed or unconfirmed pass,
including failed shutdown, retains energized recovery truth. Fault recovery
also clears it after an acknowledged physical power cut, even when a restarted
process has no usable old bus handle.

Any real transition to unconfirmed torque—including follower or leader
calibration teardown and the shutdown's final port close—refreshes the global
marker before device ownership can be released. Pre-RC local-only recovery
state is republished globally during real server startup. If that publication
cannot be made durable, the visible fault names it and `latch_persistence` is
`unsaved`; another data directory must not be treated as covered.

On Ubuntu/macOS this gives cooperative cross-user exclusion to members of the
installed hardware group; a malicious local hardware-group member can open the
serial device directly and remains outside the threat model. Windows uses
stable per-user Local AppData, independent of `TEMP`, and refuses symlink or
reparse-point motor paths; cross-user Windows exclusion is not claimed.

## Three-state torque truth

A closed port does not mean the motors are off. LeRobot 0.6.1's
`disconnect(disable_torque=True)` skips `closePort()` when its torque-off
raises, and Feetech servos keep their torque with no host attached. v0.1 read
"not connected" as "torque off" and reported success after a failed torque-off
(safety-B2).

Torque is tracked at two levels:

- **Manager:** `torque_unconfirmed` in `GET /api/robot/status`. It turns true,
  with a fault, whenever a torque-off pass could not confirm every motor. Only
  a later pass that confirms every bus clears it, or a fault reset with an
  acknowledged power cut. It is persisted with the latches.
- **Bus:** `torque_state` on `FeetechBus` and `MockBus`, one of `"on"`,
  `"off"` or `"unconfirmed"`. It survives the port close. A bus whose port is
  closed but whose `torque_state` is not `"off"` is a stale bus, and the
  manager treats it as possibly energized.

`follower_connected: false` in the status only says the port is closed.

### Torque-off on the bus

`hardware/motors.py` implements one torque-off pass, used by every stop:

1. One `sync_write("Torque_Enable", 0)` broadcast. It has no status packets,
   so a silent servo cannot delay the others.
2. Each motor's `Torque_Enable` is read back.
3. Each motor that did not read 0 gets its own acknowledged
   `disable_torque` (`Torque_Enable`, then `Lock`) with `TORQUE_OFF_RETRIES`
   (2) retries, and is read back again.

LeRobot's own `disable_torque()` stops at the first servo that does not answer
and leaves every later motor energized; this pass always tries every motor.
At LeRobot 0.6.1's packet timeout of about 50 ms per try, six silent motors
take about 1.2 s. A motor counts as off only when its write was acknowledged
and its read-back is 0; a failed `Lock` write leaves it unconfirmed.
Missing or unreadable motor metadata, or a bus without a usable per-motor
`read`, leaves the whole bus unconfirmed even when a disable write was
acknowledged.

The pass backs these three public bus calls; connection setup and cleanup use
it too, as described below:

| Call | Port afterwards | When torque-off is not confirmed |
| --- | --- | --- |
| `set_torque(False)` | open (a halt keeps the port) | raises `TorqueOffUnconfirmed` |
| `torque_off_all()` | open | returns False |
| `disconnect()` | always closed, with `disable_torque=False` | raises `TorqueOffUnconfirmed` after the close |

`TorqueOffUnconfirmed` (a `RuntimeError`) names the bus and the motors that may
still hold torque. In each case `torque_state` becomes `"off"` or
`"unconfirmed"`. `set_torque(True)` marks the bus `"on"` before it writes,
because a partial enable may already have energized some motors.

Other bus rules:

- `release_port()` closes the port with no torque write and sets
  `torque_state` to `"off"`. It is only for use after physical power was cut.
- A successful `connect()` runs the same confirmed torque-off pass before and
  after motor configuration; a write acknowledgement without a read-back of 0
  never becomes `torque_state="off"`.
- A failed `connect()` turns torque off on every motor that still answers and
  always closes the port. If any motor cannot be confirmed off,
  `torque_state` stays `unconfirmed`: a fresh wrapper cannot know whether an
  earlier session left that motor energized.
- Manager-owned calibration checks that surviving state even when the failed
  connect already closed the port and its final `disconnect()` is a no-op; it
  durably latches the fault and `torque_unconfirmed` before reporting failure.
- `disconnect()` on a bus whose port closed under it while torque was not
  known off raises `TorqueOffUnconfirmed` and leaves it `"unconfirmed"`.
- `MockBus` follows the same rules, so the mock tests exercise the same
  contract.
- Connect errors name the fix: `LEROBOT_HOME` is refused before LeRobot is
  imported (LeRobot 0.6.1 raises a bare `ValueError` for it; set
  `HF_LEROBOT_HOME`), a missing LeRobot says
  `pip install -e ".[robot]"`, and a failed handshake ends with
  ``run `so101 preflight` (motors/firmware checks)``.

### Recovery after an unconfirmed torque-off

The manager's torque-off (`_cut_torque`) covers every bus it owns: the live
ones through the pass above (the follower's pass is its `disconnect()`), and
stale closed ones, which count as failures until a later pass confirms them.
Any failure latches a fault and `torque_unconfirmed`; a confirmed pass on
every bus clears `torque_unconfirmed`.

Retries reopen every device named by durable recovery state through a fresh
handle before they try again. A legacy/local unscoped torque flag means both
follower and leader; an exact host-global marker retains its device set. A
follower read-back can therefore never clear a leader-only calibration fault.
Only the bus is opened; a recovery path never writes goal positions. This happens
on:

- the E-stop retry (step 5 above)
- every fault reset that has something to confirm

`POST /api/robot/fault/reset`:

1. 409 while a loop is active or the E-stop is latched.
2. If torque is unconfirmed, a marked port is open, or a bus is stale, it
   retries a real torque-off with read-back on every marked bus, reopening
   closed follower and/or leader ports.
3. If the retry confirms it, the fault clears (200).
4. If not, it answers 409 with the fault still latched, unless the body has
   `"power_cut_acknowledged": true`. The acknowledgement never skips the
   retry; it only overrides its failure.
5. With the acknowledgement, the buses the retry could not confirm are
   released: open ports close with no torque write, and stale ones stop
   reporting `"unconfirmed"`. This needs the bus lock, so it answers 503 with
   the fault still latched if the bus is held. On success the fault and
   `torque_unconfirmed` clear and `safety.power_cut_acknowledged` lists the
   released buses (`ports_released`).

An acknowledged physical cut may overwrite only the marker tokens for the
currently locked configured/canonical identities. This permits recovery after
a udev alias intentionally retargeted, without treating a normal identity
mismatch as safe or erasing the old target's marker.

The release matters for a dead adapter: an unplugged adapter's open handle
stays dead even after a replug, and keeping it would make every later arm fail
until a restart. After the release, the next arm opens a fresh handle.

## The sim backend

`SO101_BACKEND=sim` leaves `settings.mock` false, and `make_bus` used to key
off `settings.mock` only. Under sim it therefore built a real `FeetechBus` on
the configured ports, and arming, teleop, rollout, calibration and HIL runs
moved the real arm while labelled a rehearsal.

`make_bus` now builds the simulated bus under sim
(`so101.simulation.devices.make_sim_bus`): the MuJoCo follower (`MujocoBus`,
STS3215 servo model), and for the leader the scripted expert
(`SimExpertLeader`, no actuators). The configured ports are ignored. Both
report `torque_state`, so the C2 torque truth and every stop work the same on
the sim arm; that is how the physics tests run the three stops through
`RobotManager`. Without the `[sim]` extra, `make_bus` returns a stand-in bus
whose `connect()` fails closed with the install hint; construction never
raises, so the manager and `/api/robot/status` still work. Calibration is
refused under sim (the simulated arm has no encoders, and a calibration would
overwrite the real one). The manager's shutdown stops the sim runtime's
physics thread.

The guard stays as a permanent invariant for any other route to a
`FeetechBus`: `FeetechBus.connect()` refuses under the sim backend, before
LeRobot is imported or any port is opened, with `HardwareUnavailableError`
("SO101_BACKEND=sim: the real Feetech bus on <port> is refused (it would move
the real arm) ..."). A `FeetechBus` never connects under the sim backend.

## Leases and owner-scoped stops

### Lease secrecy (C4, safety-B5)

The lease id is the only credential for arm, observe and command.

- Only `POST /api/control/lease` (201) and
  `POST /api/control/lease/{lease_id}/heartbeat` return it.
- `GET /api/control/state` shows
  `lease: {source, expires_at, ttl_seconds, server_time} | null`, with no id.
- Events never carry lease ids. They carry `lease_tag`: the first 12 hex
  digits of SHA-256 over `"so101-lease:" + id`. The owner can compute it to
  find its own events.

### Owner-scoped disarm (C5, safety-B6)

- `POST /api/control/disarm` needs `{"lease_id": str}`. A missing, wrong or
  expired lease gets 409 and changes nothing. It is an owner-scoped Collapse.
- Global stops remain `POST /api/robot/stop` and the E-stop.
- A failed arm-and-command request cleans up with an owner-scoped, non-strict
  Halt, so it never disarms another owner and never masks the original error.
- A revoked arm puts back the safety state it found, so a watchdog or expiry
  trip stays visible, and an operator timeout never shows over a fault.

Lease expiry is enforced by the watchdog and rechecked by mutating command
paths. `GET /api/robot/status` and `GET /api/control/state` only snapshot the
state they find: neither dispatches the expiry Halt nor performs motor-bus I/O.
This keeps unguarded GET requests side-effect-free while the independent
watchdog retains the safety deadline.

## Commands and arming

### Freshness (safety-B7, C9)

- The arm-first paths (`/api/control/arm-command`, tracking
  `arm-execute`, and tracking `preview` with `execute=true`) pass the client's
  `sent_at` through to the command, so the TTL still holds after arming.
- Tracking execution needs `lease_id`, `sequence` and `sent_at`; a missing one
  is 422.
- A command is stale when it is older than `min(ttl_ms, watchdog_ms)`, and
  refused when `sent_at` is more than 1 s in the future.
- `wrist.z` may be null (monocular tracking has no metric depth); the elbow
  then holds its engage-time value.
- The tracking audit throttle is keyed on the live lease and kept in a bounded
  LRU (5 s window, 60 s TTL, 256 keys), so a burst of new keys cannot reset
  other callers' throttles (sim-B20).

### Arming

- The first-command deadman window starts after torque-enable succeeds, not
  before it (devex-B4).
- Arming checks its authority again each time it gets the bus lock, before
  any bus write: the lease (named and not lapsed), the authority epoch, the
  E-stop, a fault, calibration, the mode and the stop state. A Halt,
  Collapse, global stop, fault or lease lapse that completes while the arm
  waits for the bus therefore gets no torque-enable and no goal write after
  it; the arm answers 409 ("control authority changed ...") and the stop
  keeps the arm. A port the arm had opened with torque still off gets the
  usual torque-off with read-back and closes. A Halt's hold it was about to
  take over stays with the Halt.
- A Halt or Collapse reserves a private stop transition before it joins a
  loop or waits for the bus and releases it only after the result is final.
  Every transition blocks new leases, arming and loop starts; a motion-owning
  transition also rejects external commands, while a loops-only Halt leaves a
  session that already owns the arm alone. Transitions have monotonic tokens
  and semantic precedence: a newer operator/global Halt supersedes an ordinary
  pending Collapse, while owner-, epoch- and loops-only Halts leave that
  Collapse in charge; a fault-triggered Collapse keeps precedence over every
  Halt. The token also reserves a validated lease/epoch-scoped Collapse: a
  later lease release revokes future authority but cannot cancel that reserved
  descent. Only the current serial token can claim precedence; completed newer
  stops and E-stop bookkeeping permanently stale every older token. Halt reads
  torque, writes Goal := Present and publishes
  `stop_state="halted"` during one `_io_lock` ownership, so a re-arm cannot
  slip between the write and the published hold. Publication revalidates the
  token and whether a loops-only stop acquired a new external owner. If a stop
  revokes an arm during torque-enable, the arm's late cleanup defers to that
  Halt/Collapse; category-0 E-stop and fault cleanup still cut torque.
- A follower resting outside the guard envelope by at most 30 degrees arms and
  holds where it is, then may only move back toward the envelope. Further out
  is refused with "move the arm back into range by hand" (safety-B10).
- `POST /api/control/observe` is refused (409) while armed or arming, because
  observe reads with torque off and would drop an armed arm while the state
  still said armed (safety-B12).
- Every `start_*` refuses while the E-stop or a fault is latched (safety-B1,
  manager half).

## Loops, telemetry and time

- **Loop pacing (data-B4).** Teleop, recording and rollout loops use deadline
  scheduling (`_Pacer`): each iteration takes max(work, period) rather than
  work + period, a late iteration is caught up, and a stall of more than one
  period resyncs instead of bursting frames.
- **Loop end.** A loop that ends because a stop asked it to leaves the
  follower energized: the Halt that stopped it holds the arm. Only a failed
  loop, or one that ended under an E-stop or a fault, cuts torque on its way
  out.
- **Live observation (C6).** Every telemetry frame carries
  `observation_live`, true only when the observation was sampled within the
  last 500 ms by an active loop, an armed session, a halt's hold or a
  collapse descent.
- **Active model (C7).** `GET /api/robot/status` has `active_model`, the
  checkpoint path driving the current rollout, or null.
- **Snapshots (C10).** `GET /api/video/snapshot?cam=<id>` returns one fresh
  frame with `Cache-Control: no-store`, or 503 JSON when the camera has no
  frame from the last second.
- **One clock (devex-B4).** Leases, operator heartbeats, the deadman, the
  watchdog, sample and frame ages, and loop pacing all read one injected
  monotonic clock (`_clock`), and the loops sleep through `_sleep`. Tests
  freeze and advance that clock, so the timing tests are deterministic under
  load.

## Fault-injection harness

`tests/test_control_faults.py` runs the production `RobotManager`,
`SO101Follower` and `FeetechBus` in real mode and fakes only LeRobot's
`SOFollower` device. The fake keeps the LeRobot 0.6.1 semantics the safety
code depends on: per-motor transfers wait for a status packet, `sync_write` is
a reply-less broadcast, `disable_torque` stops at the first silent motor, and
`disconnect` skips `closePort()` when its torque-off raises. An `Arm` object
holds each servo's real torque, which the software must report truthfully, and
injects the faults:

- servos that go silent at the start or in the middle of an operation
- an unplugged adapter whose open handle stays dead after a replug
- a wedged transfer that blocks its caller

The harness also enforces the bus-lock invariant: any transfer made without
the manager's bus lock fails the test. Every test runs on the injected clock
with manual supervision, so the deadman and the watchdog act only when the
test says so.

## Known limits

- If the host process dies (SIGKILL, a crash, power loss to the computer), the
  STS3215 servos keep holding their last goal with torque on. No software stop
  covers the dead interval; only the physical power switch does. Real
  standalone and RobotManager paths write local and device-global energized
  intent before torque-enable, so a later owner of either physical port knows
  and comes back faulted even through another data directory. A halted arm is
  energized by design, so the physical switch stays the last resort while the
  owning process is absent.
- The Collapse cap applies to the commanded trajectory (every joint at most
  `speed / 30` per tick, also after a late tick) and to the motion averaged
  over 0.25 s (within 10 %
  in the MuJoCo test). Within one 1/30 s period an STS3215 with LeRobot's
  Acceleration = 254 closes each goal step along its acceleration profile, so
  instantaneous joint speed peaks near 3x the cap (45 deg/s in the physics
  test). Capping that needs the servo's Goal_Velocity register, which the
  motor-bus interface does not expose yet.
- The 12 s bound is shorter than distance / speed + 2 s for any joint more
  than 150° from rest at 15 deg/s. Such a descent has little margin, and one
  more than 180° away times out into the torque-off fallback. The task's
  READY pose (wrist 168° from rest) collapses in 11.2 s.
- Supervision counts any operator browser's heartbeat, not only the one that
  started the work, so an open second tab keeps a halt for the full hold
  timer (60 s), never longer.
- A driver call that wedges during a Halt's own read or write blocks that
  caller (the request, or the watchdog for its own halts) until the driver
  returns; the E-stop stays independent of it. A wedged Collapse descent is
  cut by the watchdog's backstop.
- Calibration holds `_io_lock` for its whole sweep (8 to 60 s). An E-stop
  then cancels the sweep at its next sample, and its torque cut waits at most
  0.5 s for the sweep to roll back and close the port; past that it latches
  `torque_unconfirmed` and answers 503. `so101 calibrate` (the CLI, server
  stopped) still opens its own bus, under the device-global lock and only
  while no latch is set
  ([Commands that run without the server](#commands-that-run-without-the-server)).
- Data-folder safety rows remain identity-scoped, while the host-global journal
  now retains real E-stop and fault latches as well as energized/unknown torque
  state. A different `SO101_DATA_DIR` sharing either physical port therefore
  adopts a blocking fault after the device lock is released. Only the
  deliberate reset clears that journal entry.
- Command freshness compares the client's wall-clock `sent_at` with the
  server's wall clock, so it depends on both clocks being in sync.
- A loop teardown that confirms every bus off does not clear a
  `torque_unconfirmed` flag set earlier, because a bus that was never opened in
  this process proves nothing about a restored flag. The fault reset's retry
  clears it.

## Tests

| File | Covers |
| --- | --- |
| `tests/test_control_stops.py` | the dispatcher, stop kinds and their call sites |
| `tests/test_control_halt.py` | Halt: the hold, the grip rule, fallbacks, the hold timer and supervision, renewal, re-arming, loop ends |
| `tests/test_control_collapse.py` | Collapse: the capped descent, rest, fallbacks, precedence, servo health, shutdown |
| `tests/test_control_stop_routes.py` | the stop routes and their guard and body rules, CLI, settings, status and telemetry fields |
| `tests/test_sim_stops_physics.py` | the three stops under gravity in MuJoCo, through `RobotManager` |
| `tests/test_control_estop.py` | category 0 on every bus, torque off before anything blocks |
| `tests/test_control_estop_robustness.py` | async E-stop, its executor, a blocked worker pool, bus-busy 503 |
| `tests/test_control_latches.py` | latches, persistence, resets, `torque_unconfirmed` |
| `tests/test_control_faults.py` | the fault-injection harness at the LeRobot seam |
| `tests/test_control_motors.py` | the torque-off pass, disconnect, connect cleanup, error text |
| `tests/test_control_sim_guard.py` | sim routing in `make_bus` and the permanent `FeetechBus` sim guard |
| `tests/test_control_lease_secrecy.py` | C4 lease secrecy and `lease_tag` |
| `tests/test_control_owner_scope.py` | C5 owner-scoped disarm and error cleanup |
| `tests/test_control_tracking.py` | C9 `sent_at`, null `wrist.z`, the audit throttle |
| `tests/test_control_arming.py` | the envelope rest tolerance, observe while armed |
| `tests/test_control_starts.py` | `start_*` refusals, preloaded policy, rollout clock |
| `tests/test_control_pacing.py` | deadline scheduling of the loops |
| `tests/test_control_liveness.py` | C6, C7 and C10 |
| `tests/test_control_timing.py` | the injected clock, the deadman after torque-enable |
| `tests/test_control_cli.py` | `so101 estop` and `so101 doctor` exit codes |
| `tests/test_standalone_gate.py` | the standalone gate on every CLI and library path, and the server's device-global locks |
| `tests/test_control_plane.py` | the v0.1 control-plane contract tests |
| `tests/test_robot_health.py` | the server's servo-health poll: activation, the bus lock, WARN, CRITICAL -> Collapse + fault, inactive cases |
| `tests/test_robot_bus_diagnostic.py` | `run_idle_bus_diagnostic`: refusals, the bus lock, read-only lending, E-stop preemption, preflight through it |
| `tests/test_robot_calibration.py` | calibration through the manager, an unconfirmed torque-off as the job's error, `calibrate_timed`'s torque-off pass |

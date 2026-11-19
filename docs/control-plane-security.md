# Control-plane security

This page explains who can send commands to the SO-101 Suite server and what
stops a request that should not move the arm. It covers the v0.2 web gate:

- the Origin/Host guard (contract C1)
- the eval gates (C3)
- agent approval and execution (C8)
- the planner limits

E-stop latches, latch persistence, three-state torque truth, lease secrecy
and the stop categories (E-stop, Halt, Collapse) are covered in
[control-plane-safety.md](control-plane-safety.md).

The bug IDs below refer to the v0.1 audit reports in `docs/audit/v0.1/`
(`safety.md`, `sim.md`, `data.md`, `devex.md` and `frontend.md`).

## Threat model

The server is a single-user workstation service. By default it listens on
`127.0.0.1:8000`, and Docker publishes it on `127.0.0.1:8000` only. Anything
that can send it HTTP can try to move a real arm, so the question is who can
reach it.

**Something we defend against: a hostile web page in the operator's browser.**
Any page the operator opens can send requests to `http://127.0.0.1:8000`. CORS
only stops the page from *reading* the response. The request itself still
runs if it is a "simple" request: a form POST, `fetch(..., {mode: "no-cors"})`
or a `text/plain` body. WebSocket handshakes are not covered by CORS at all.
In v0.1 a page could clear a latched E-stop this way, then resume an eval
rollout (safety-B3, safety-B1), or read live telemetry over `/ws/telemetry`
(safety-B4).

**Something we defend against: DNS rebinding.** A page on `attacker.example`
can make its own hostname resolve to `127.0.0.1`. The browser then treats the
requests as same-origin, so an Origin check alone lets them through. The
`Host` header still says `attacker.example`, which is what the Host allowlist
checks (safety-B4).

**Something we only partly defend against: other machines on the LAN.**
Binding to `0.0.0.0` (needed for a Quest headset) exposes the port to the
whole network. The guard stops browsers on other machines from being used
against the server. It does **not** authenticate a LAN client that sends its
own requests with curl or a script, because such requests carry no `Origin`
header. Against those, the defences are the control lease, the operator
heartbeat and the agent approval token described below. None of them is an
identity check. Only bind to a non-loopback address on a network you trust.

**Out of scope:** user accounts and authentication, TLS, and malicious local
processes (they can open the serial port directly).

## The Origin/Host guard (C1)

`so101/services/guard.py` is a pure ASGI middleware. It runs before CORS,
routing, body parsing and WebSocket `accept()`, so a rejected request never
reaches a handler. Middleware order, from the outside in:

1. ServerErrorMiddleware
2. Guard
3. StrictJSON
4. CORS
5. ExceptionMiddleware
6. Router

### Host check: every HTTP request and every WebSocket handshake

The `Host` header must name this machine. These names are always allowed:

- `localhost`, `127.0.0.1` and `::1`
- the bind address (`SO101_HOST`), unless it is a wildcard such as `0.0.0.0`
- every entry in `SO101_ALLOWED_HOSTS`

`SO101_ALLOWED_HOSTS` is a comma-separated list of hostnames or IP addresses.
Ports are ignored, and IPv6 addresses may be bracketed or bare. An entry like
`*.lab.lan` matches subdomains only, not `lab.lan` itself. A bare `*`, or a
wildcard anywhere else, stops the server at startup: list each host instead.

A request whose Host is missing, duplicated, malformed or not allowed gets:

```
400 {"detail": "untrusted Host header; add the hostname to SO101_ALLOWED_HOSTS"}
```

The guard parses Host itself rather than using Starlette's
TrustedHostMiddleware, because that middleware splits on the first `:` and so
can never match `[::1]:8000`.

### Origin check: state-changing requests and every WebSocket handshake

This check applies to POST, PUT, PATCH and DELETE, and to every WebSocket
handshake. The request is rejected if any of these is true:

- `Sec-Fetch-Site: cross-site` is present
- `Origin` is `null` or malformed
- there is more than one `Origin` header
- `Origin` is neither listed in `SO101_CORS_ORIGINS` nor the same origin as
  `Host` (scheme default ports are taken into account)

HTTP requests are rejected with `403 {"detail": "cross-origin request rejected"}`.
WebSocket handshakes are closed with code 1008 before `accept()`.

A request with neither an `Origin` nor a `Sec-Fetch-Site` header passes, since
it cannot come from a browser page. That covers the CLI, curl, httpx and the
test client.

Notes:

- Only `Sec-Fetch-Site: cross-site` is rejected outright. A page on another
  port of the same host (for example the Vite dev server on `:5173`) is
  "same-site", so its `Origin` must still be listed in `SO101_CORS_ORIGINS`.
- GET, HEAD and OPTIONS are not Origin-checked. CORS keeps their responses
  unreadable to foreign pages. **GET endpoints must never change state**, and
  must not do hardware I/O either: a foreign page can trigger them. That is
  why running preflight is `POST /api/preflight` and `POST /api/preflight/bus`
  (bus reads, a saved report); a GET there answers 405. Robot status and
  control-state GETs only snapshot current state: lease expiry is dispatched
  by the watchdog or a mutating command path, never by either read.
- A `*` entry in `SO101_CORS_ORIGINS` widens CORS for reads only. The guard
  ignores it, so it never allows cross-origin writes.

### Installed-artifact test identity

`SO101_E2E_TOKEN` is empty in every normal installation. The installed-wheel
test harness alone sets a fresh random value on a throwaway mock or sim server and
sends it as `X-SO101-E2E-Token`. With that setting, the side-effect-free
`GET /api/e2e/identity` proves the expected suite version and explicitly says
which non-real backend is active before a test writes anything. The value is secret: responses,
logs and retained evidence never contain it.

Outside the three unconditional stop routes, the guard never ignores this
header. A missing token on the identity endpoint,
a missing server setting, a wrong or duplicated value, or the real backend
gets a 403 before routing. While the setting is active, every state-changing
request and WebSocket also requires the header. A real-backend configuration
with `SO101_E2E_TOKEN` is rejected at startup. A browser that is accidentally
redirected to a real or ordinary server therefore cannot turn its test requests
into robot commands; an E-stop, Halt or legacy stop remains the intentional
exception. When the
setting and header are both absent, all normal routes behave exactly as before
and the identity path is the same JSON 404 as any unknown `/api/*` route. This
nonce binds one test run to one non-real process; it is not production
authentication and must never be configured on an operator server.

### Stopping is never blocked

`POST /api/robot/estop`, `/api/robot/stop` and `/api/robot/halt` skip both
checks. An E-stop must work from any page and any hostname, including one you
forgot to allowlist. The worst a hostile page can do with these paths is stop
the robot. `/api/robot/estop/reset`, `/api/robot/fault/reset` and Collapse are
**not** exempt.

The test-only identity header is ignored on these three stop paths. An E2E
client that reaches a replacement real server can therefore only stop it; it
cannot reset, arm or command it.

Because they are exempt, these routes also never wait on, or buffer, a body
someone else controls. E-stop and the legacy stop read none. The halt reads
at most 4 KiB within 0.1 s (`read_stop_body` in `guard.py`; a declared
`Content-Length` over the cap is not read at all), and a longer or slower
body counts as no body: the halt happens at once with its default reason.

### Configuration recipes

| Setup | What to set |
|---|---|
| Local browser on the same machine (default) | Nothing. |
| `docker compose` | The `ui` container's Vite proxy reaches the API as Host `api`, so `docker-compose.yml` sets `SO101_ALLOWED_HOSTS=api`. |
| Quest headset or another LAN device | Bind to `SO101_HOST=0.0.0.0` (or the LAN IP). Add the name or IP the device uses to `SO101_ALLOWED_HOSTS`, for example `SO101_ALLOWED_HOSTS=192.168.1.20,so101.local`. |
| UI page served from a different origin than the API (for example Vite on `http://192.168.1.20:5173`) | Also add that exact origin (scheme, host and port, with no path) to `SO101_CORS_ORIGINS`. |

When this server serves the built UI itself, the page's `Origin` equals the
`Host`, so no CORS entry is needed.

### Serving the UI

Unknown `/api/*` and `/ws/*` paths return a JSON 404, never `index.html`
(F25, devex-B12). Static files are served only from inside the UI directory:
`../`, encoded dots, absolute paths and symlinks that leave the directory all
fall through to the SPA index (safety-B4 repro 03).

The requested path is checked as a string before any filesystem call. Each
`/`-separated segment must be an ordinary file name: letters, digits, `_`,
`.` and `-`, not starting with a dot. Backslashes, `:` and Windows device
names (`CON`, `PRN`, `AUX`, `NUL`, `COM0`–`COM9`, `LPT0`–`LPT9`, with or
without an extension) are refused. Any other path gets the SPA index, or a 404
under `/assets`. Without this check, Windows would open `\\host\share`,
`//host/share`, `D:\x` or `COM3` while resolving the path. Opening a UNC path
sends the operator's NTLM hash to that host. A cross-site `<img>` can make
these requests, and GETs are not Origin-checked. Dotfiles in the UI directory
are never served.

The UI directory is the first of these that contains `index.html` (devex-B3):

1. `SO101_UI_DIR`
2. the packaged `so101/web`
3. the checkout's `frontend/dist`

## Strict JSON is bounded and malformed numbers return 422

Python's JSON parser accepts `NaN`, `Infinity` and `1e309` (which becomes
infinity). In v0.1 those values, and finite values too large for a
timestamp, crashed request handling with a 500 (safety-B11, sim-B19).

`so101/services/strict_json.py` rejects them for every JSON body with a 422
and a fixed message. It buffers no more than 1 MiB: a declared
`Content-Length` over that bound is rejected before the body is read, and a
streamed body is rejected as soon as a chunk takes the cumulative size over
the bound. Both cases return
`413 {"detail": "JSON body exceeds the 1048576-byte limit"}` and no later
chunks are read. A body with no `Content-Type` is subject to the same bound
and numeric check because some FastAPI versions and settings may parse it as
JSON; an explicitly non-JSON content type passes through unchanged. An
`OverflowError` anywhere in a handler also maps to 422. Because these checks
are global, new routes get the protection automatically. The three stop-path
exemptions described above remain outside this middleware; Halt keeps its
separate 4 KiB, 0.1 s lenient reader.

## Eval gates (C3)

An eval runs a policy on the real arm between operator scores. These rules
close safety-B1:

- **Start.** `POST /api/eval/start` needs an `operator_id`. That operator must
  have a fresh supervision heartbeat. The robot must be idle, with no E-stop,
  no fault and no external control lease. The checkpoint must be listed by
  `GET /api/models`; an unregistered path gets 404. A registered policy that
  cannot be preflighted or loaded fails the request with 422 before motion
  (data-B13). The checkpoint supplies the default task and control rate, and
  one loaded policy instance is reset and reused for every trial. `max_seconds`,
  the per-trial time limit, must be a finite number greater than 0 and at most
  3600. Anything else, including the strings `"NaN"` and `"Infinity"`, gets
  422.
- **Durable mutation order.** Start writes and fsyncs its journal header and
  commits `eval.start_requested` with SQLite `synchronous=FULL` before its
  rollout thread starts; a later readiness race gets a best-effort
  `eval.start_refused`, never a durable event claiming completed motion.
  Score commits `eval.score_requested` before waking the session thread and
  does not acknowledge until the trial line is fsynced; continue commits
  `eval.continue_requested` before it permits the next rollout. Their
  `eval.scored` and `eval.workspace_reset_confirmed` completion events are
  best effort. A monotonic manager safety/authority generation makes even an
  E-stop or fault that is reset during one of those writes abort the session.
  A failed required write answers 503 without that mutation. Stop is the
  exception by design: it stops motion first, then reports any audit-write
  failure as a warning without claiming the stop failed.
- **Score and continue.** `POST /api/eval/score` takes
  `{"operator_id", "success"}` and `POST /api/eval/continue` takes
  `{"operator_id"}`. Only the operator who started the session may call them,
  and only with a heartbeat younger than 2 s.
  - The old query-string and bodyless forms are gone (422).
  - Requiring a JSON body also means a cross-site page cannot send these as
    simple requests, although the guard already rejects such a page.
- **E-stop and faults.** Continue is refused while the E-stop is latched or a
  runtime fault is active. An E-stop or fault during a session aborts it:
  phase `"aborted"`, with `error` naming the cause. Resetting the E-stop never
  resumes the session. A trial cut short this way is recorded unscored and
  interrupted, and a score that arrives after that gets 409.
- **Reports survive.** Stopping the server writes a partial report before
  the robot manager shuts down, and each trial is journaled as it ends
  (data-B7, data-B14). HTTP and CLI reports use the same report model: Wilson
  95% intervals, checkpoint task/rate, the policy implementation's label and
  the actual `real`, `sim` or `mock` backend. The initial journal and every
  trial line are required and fsynced: initial failure refuses before motion;
  trial failure aborts before another trial and the final JSON report retains
  the result plus the error. State remains `phase="finalizing"`, `active=true`
  until report publication and the terminal journal record have either
  succeeded or produced a visible terminal error. If a clean report was
  published but a later terminal-journal failure cannot be written back into
  it, that stale report is moved outside the `eval_*.json` report set.

The operator heartbeat is `POST /api/robot/operator-heartbeat` with
`{"operator_id": ...}`. The UI sends it while the operator's tab is open.

## Agents: proposals, approval and execution (C8)

No planner can move the arm. A planner only creates a **proposal**: one named
skill, plus the exact joint targets the server derives from that skill.
Moving the arm takes three separate steps, each with its own checks:

1. approve
2. execute
3. supervise until the run ends

### Proposals and planners

- **Skills.** The skills are `inspect_scene`, `home`, `open_gripper`,
  `close_gripper` and `wrist_wave`. `inspect_scene` has no targets and can
  never be approved.
- **Deterministic routing.** This router, also used for simple
  `{"instruction": ...}` requests, matches whole words. Text that is negated
  ("don't open the gripper", "never wave", "stop"), names more than one skill,
  or names none becomes `inspect_scene`. So "spark" and "homework" no longer
  mean `home` (sim-B15).
- **Configured planner.** The optional OpenAI Responses planner gets one
  strict function tool, with `parallel_tool_calls: false` and `store: false`.
  Its schema uses only keywords that strict mode supports; the 500-character
  rationale and the 1–30 s duration are enforced when the reply is parsed
  (sim-B12).
- **Planner availability.** The planner is offered as available only when
  `SO101_AGENT_PROVIDER=openai_responses`, a key and a model are set, and the
  base URL is HTTPS or loopback (sim-B13).
- **Planner errors.** A reply that is not one well-formed bounded proposal
  returns 502; this includes any status other than `completed`. An
  unconfigured or unreachable planner returns 503 (sim-B14).
- **Target digest.** Every proposal includes `target_digest`: the SHA-256 of
  its targets as canonical JSON (sorted keys, floats). The UI shows it next
  to the targets the operator reviews.

### Approve

`POST /api/agent/proposals/{id}/approve` with
`{"lease_id", "operator_id", "target_digest"}` returns
`{"approval_token", "expires_at", "proposal"}`. It refuses with 409 unless all
of these hold:

- The proposal is still `proposed` and at most 120 s old. An older proposal is
  marked `expired` instead. The age is measured on the server's monotonic
  clock, so a wall-clock step cannot make an old proposal look fresh. A
  proposal created before the last server restart has no known age, so it is
  marked `expired` too.
- The proposal allows motion and has targets.
- `target_digest` equals the server's digest of the current targets, so the
  operator approves exactly what they reviewed.
- There is no E-stop and no runtime fault.
- The approving operator has a fresh heartbeat, and the lease is live.
  Approval also heartbeats the lease.

The approval token (sim-B7):

- comes from `secrets.token_urlsafe(32)`
- is valid for 30 s
- is bound to the proposal, the reviewed digest, the lease and the approving
  operator
- is held in server memory as a SHA-256 hash only, so a restart invalidates
  every token
- is single use, and is burned on *any* presentation, even one that fails
- is replaced if the proposal is approved again
- is revoked if the proposal is cancelled

Tokens, lease IDs and operator IDs never appear in `agent.*` audit events.

The v0.1 execute body, `{"lease_id", "approved": true}`, now gets a 422:
execution needs a token.

### Execute

`POST /api/agent/proposals/{id}/execute` with `{"lease_id", "approval_token"}`
returns 202 and the proposal in status `executing`. Before starting, it checks
again everything approval checked, plus:

- the token is valid and unexpired, and was issued for this proposal and lease
- the digest still matches
- no other skill is running (one at a time)
- control is not already armed
- the robot is idle

The server then arms control on the request thread, so a refusal is still a
synchronous 409. It runs the skill on a server thread at 30 Hz (sim-B1).

Every tick reads the measured joints, then ends the run on the first of
these that applies:

| Cause | Status | `stop_reason` |
|---|---|---|
| Cancel request | `cancelled` | `cancelled` |
| E-stop | `failed` | `estop` |
| Runtime fault | `failed` | `fault` |
| Operator heartbeat older than 2 s | `failed` | `operator_heartbeat` |
| Deadman lapse: no hold within 300 ms, when `require_deadman` is set | `failed` | `deadman` |
| Every target joint within tolerance (2° for arm joints, 3 units for the gripper) | `executed` | `target_reached` |
| `max_duration_seconds` passed | `timeout` | `max_duration` |
| A refused step, for example a lost lease | `failed` | `rejected` |
| An unexpected error | `failed` | `error` |

Each step that does run goes through the same external-command path as
teleop:

- lease and increasing sequence numbers
- a 500 ms TTL
- the deadman
- the SafetyGuard clamp: at most 15° per step, plus the velocity limits

### What the client must do while a skill runs

- Send the operator heartbeat at least every 2 s.
- **Keep the lease alive**, with `POST /api/control/lease/{lease_id}/heartbeat`
  within the lease TTL. The executor does not heartbeat the lease for you, so
  an abandoned client loses the lease and the run stops.
- If the proposal has `require_deadman`, send
  `POST /api/agent/proposals/{id}/hold` with `{"lease_id"}` at least every
  300 ms. A hold for another proposal or lease, or one that arrives after the
  deadman lapsed, is refused with 409, and a late hold also stops the run.
  The 300 ms window starts when the run starts. A hold sent while execute is
  still arming is accepted, so the client can start holding as soon as it
  sends execute.
- Poll `GET /api/agent/proposals/{id}` until the status is terminal:
  `executed`, `failed`, `cancelled` or `timeout`.

### Cancel, end state, restart and shutdown

- **Cancel.** `POST /api/agent/proposals/{id}/cancel` has no body. On a
  running skill it stops at the next tick and records the partial result. On a
  proposal that has not started, it cancels the proposal and revokes its
  approval (sim-B8).
- **End state.** Every run ends with a Halt scoped to the run's lease
  ([control-plane-safety.md](control-plane-safety.md#halt)): disarmed, with
  the final pose held and the motors on. The hold timer then applies: a
  Collapse to the rest pose after `SO101_HALT_HOLD_SECONDS`, or after
  `SO101_SUPERVISION_GRACE_SECONDS` with no operator supervising. The
  proposal's `result` records:
  - `reached`, `final_error_deg`, `max_abs_error_deg`, `duration_s` and
    `steps`
  - `end_state`: `holding`, `collapsing`, `deenergized`, `estopped`, `fault`
    or `unknown`
  - the status and `stop_reason`, with any `error`
  - the targets and tolerances
  - `started_at` and `finished_at`
- **Restart.** Proposals left `executing` by a crash become `failed` at
  startup: `stop_reason` `restart`, `end_state` `unknown` (sim-B9).
- **Shutdown.** A clean shutdown stops a running skill before the robot
  manager shuts down: `stop_reason` `shutdown`.

### What approval does not prove

`operator_id` is a claim, not an identity. There are no user accounts, so
there is no check that the approver differs from whoever made the proposal.
The real capability is the **lease ID**: whoever holds it can approve and
execute. Keeping it out of public responses and the audit log is covered in
[control-plane-safety.md](control-plane-safety.md).

## Known limitations

- **The end-of-run Halt covers the run's whole lease.** A run ends with
  `request_stop(StopKind.HALT, ..., lease_id=...)`, so it never disarms another
  client's session. It does halt a session armed again under the run's own
  lease before the run's thread exits, for example by an operator who re-arms
  right after a cancel. If cancel gave up on a stuck skill, the thread's late
  exit halts only when control is not armed by then: that Halt finds the
  latched fault and cuts torque with read-back (`AgentExecutor._may_disarm`).
- **Dev CORS origins in every deployment.** The default `SO101_CORS_ORIGINS`
  lists the Vite dev server origins `http://localhost:5173` and
  `http://127.0.0.1:5173`, even where no dev server runs. A page served from
  port 5173 on the browser's machine, such as another project's dev server,
  can read API responses. When the API runs on that machine too, the page can
  also send state-changing requests. Where this server serves the built UI,
  set `SO101_CORS_ORIGINS=` (empty) or list only the origins you use.
- **A restart forgets tokens and proposal ages.** Both live in memory. After a
  restart, approve again; a proposal that was still `proposed` reports
  `expired` when approved, so create and review a fresh one. It is listed as
  `proposed` until someone tries to approve it.
- **Restart recovery window.** Recovery after a restart scans only the 500
  most recently updated proposals.
- **Eval E-stop latency.** An eval notices an E-stop within its 50 ms poll.
  The E-stop itself acts immediately.
- **No LAN authentication.** The guard does not authenticate LAN clients that
  are not browsers (see the threat model).

## Tests

- `tests/test_gate_guard.py`: Host and Origin rules, WebSocket close codes,
  stop-path exemption, SPA containment, UI directory order
- `tests/test_gate_nonfinite.py`: non-finite and overflowing numbers
- `tests/test_gate_eval.py`: eval gates and durable reports
- `tests/test_gate_agent.py`: approval tokens, execution, supervision, cancel,
  restart, shutdown, audit privacy
- `tests/test_gate_planner.py`: planner schema, availability, malformed
  replies, routing

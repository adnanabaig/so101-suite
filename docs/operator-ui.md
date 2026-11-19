# Operator UI

This page describes how the SO-101 Suite web UI (`frontend/`) behaves around safety. It covers:

- the safety bar and the banners below it;
- how the UI decides whether what it shows is current (freshness);
- the keyboard shortcuts, the deadman and the jog keys;
- the first-run setup checklist;
- the accessibility rules every page follows, and the tests that enforce them.

It is written for operators and for anyone who changes the UI.

## Current Operate workspace

Operate places cameras and the encoder view beside one control rail. Start
Teleop opens an MCAP episode; Stop Teleop saves it and Halts. Idle recording
services wait for an episode or another named capture activity.

Inspect live data exposes leader intent, accepted commands, measured joints,
operating bounds, control-phase timings, motor health and full raw fields.
Sample age and paused sampling are distinct from connection freshness.
Camera settings use each relay's capture owner and device-specific saved profiles;
see [camera timing and appearance](hardware/camera-timing-investigation.md).

Review recordings plays finalized MCAP camera streams and recorded encoder poses
on the episode clock. It sends no robot commands. Changing to a motion operation
removes the recorded player. Charts use at most 3,000 display points; reported
control-rate and gap statistics use every raw tick. Missing images and legacy
ticks without embedded episode IDs are explicitly labeled. Full-field snapshots
use the nearest preceding retained message and are not a lossless raw-message
browser. Original data remains in MCAP. Data checks and human acceptance are
separate; saved tuning recordings do not become accepted demonstrations.

Open in Foxglove requests a local desktop launch for the selected/latest saved
episode export, preparing an export if necessary. It neither uploads data nor
replays the physical arm. A successful launch request does not prove the desktop
window rendered successfully.

Home & reference checks reads installed offsets, limits, torque state and encoder
positions through RobotManager after operator support confirmation. It does not
seek end stops or automatically rewrite calibration. Collapse remains the bounded
return to the saved rest/home pose followed by verified torque off. Powered
automatic endpoint discovery is not qualified by manual endpoint samples alone.

What each stop does to the motors, the latches and the leases is the server's
job, described in [control-plane-safety.md](control-plane-safety.md). Who may
send requests at all is described in
[control-plane-security.md](control-plane-security.md).

IDs such as F4 are findings in the v0.1 frontend audit
(`docs/audit/v0.1/frontend.md`). IDs such as C6 are the v0.2 cross-stream API
contracts. The stop contract, C11, is §5 of the
[v0.2 design](plans/2026-10-01-suite-v02-design.md).

## The safety bar

The safety bar sits in the sticky header on every page and at every screen
width (F22). It stays visible and clickable above the navigation drawer and
above dialogs. From left to right it holds:

1. the **state chip**;
2. the **stops**, ordered from least to most drastic;
3. the **E-stop**, which is always last, always in the same place, and alone
   in its slot (F2).

| Stop | How to trigger it | What it does | Shown when |
| --- | --- | --- | --- |
| Collapse | Its button only; there is no key | Lowers the arm to the rest pose at a capped speed, then turns the motors off | The server lists `collapse` in `stop_kinds` |
| Soft stop | Its button | Legacy `POST /api/robot/stop`: returns the robot to idle and disarms manual control | The server does **not** list `halt` |
| Halt | Its button, or `H` | Freezes the arm in place with the motors on. When the hold timer runs out, the server collapses the arm. | The server lists `halt` |
| E-stop | Its button, or `Escape` | Cuts all motor power. The server latches it until a deliberate reset. | Always |

The stops come from `stop_kinds` in `GET /api/robot/status` (C11). A server
that does not send the field is treated as offering E-stop only.

The v0.2 server lists all three, so the bar shows Collapse, Halt and E-stop.
Wave-1 servers sent `["estop"]`, because their Halt and Collapse still behaved
as in v0.1; against one, the bar shows Soft stop and E-stop.

On screens narrower than 640 px, Collapse and Soft stop move into the chip's
popover. Halt and E-stop stay in the bar at every width.

Collapse is drawn in neutral grey and has no key, so it is never mistaken for
E-stop. Halt is amber.

### E-stop

- **It fires on pointer down.** It does not wait for the release. It also fires
  on click, so keyboard and screen-reader activation work. A click that arrives
  within one second of a pointer down that already fired counts as the same
  press, so one press sends one request.
- **It is never disabled**, not even while its own request is in flight (F6).
  A small dot on the button shows that a request is in flight. Pressing again
  sends again.
- **`Escape` triggers it on every page.** This includes text fields, open
  dialogs and the chip popover. The key is caught at the window, before the
  page's own handlers see it. Key repeat is ignored, so holding `Escape` sends
  one E-stop.
- **A 503 means latched, not failed.** The server sets the latch before it
  answers, then waits up to 3 s for torque-off to be confirmed on every bus.
  A 503 says the latch is set but the teardown did not finish cleanly:
  torque-off is not confirmed yet, it failed, or the control thread is hung.
  The UI shows a critical alert, "E-stop latched, torque-off not
  confirmed: cut physical power", with the server's detail verbatim, and never
  calls it a failed E-stop. A later confirmed E-stop or a reset clears it.
- **If the request fails** any other way (no answer, or another error), a
  critical alert says "E-stop request failed" and tells the operator to cut
  physical power if the arm is moving or the motors cannot be confirmed off.

### Resetting an E-stop

While the latch is set, the "E-stop latched. All motors are off." banner offers
**Hold to reset E-stop**:

- **Reset is a hold-to-confirm control.** It needs a continuous press of 1.5 s
  with the pointer, `Space` or `Enter`. Releasing early, cancelling or losing
  focus aborts the hold, and a plain click does nothing.
- **Reset stays locked for 3 s after any E-stop sent from this page.** The same
  lockout also starts when a latch is first seen from any source, such as
  another station or a fault, counted from the newest evidence. While locked,
  the button counts down ("Reset in 3s"). This stops a double tap from
  clearing the stop it just sent (F2).
- **The reset clears only the E-stop.** The request carries the operator id and
  `confirm: true` (C2). A fault or an unconfirmed torque-off stays latched; the
  fault reset in Diagnostics clears those (see
  [Clearing a fault](#clearing-a-fault)). The robot is left disarmed, so
  nothing moves until someone arms it again.
- **If the server refuses the reset,** the latch stays and an alert shows the
  server's reason. For example, the server refuses while a control loop is
  running.

### Clearing a fault

**Clear remediated fault** on Diagnostics appears while a fault is latched or
torque-off is unconfirmed. The reset body is
`{"operator_id": "…", "confirm": true}` (C2).

1. **The first attempt never carries the acknowledgement.** With torque-off
   unconfirmed, the dialog says the server will retry the torque-off and read
   it back on every bus. If that confirms the motors are off, the fault clears.
2. **If the retry still cannot confirm it,** the server answers 409 and names
   `power_cut_acknowledged`. The UI shows that detail verbatim and offers
   **I have cut physical power…**. That opens a dialog whose Clear button stays
   disabled until the operator ticks "I have cut physical power to the arm".
   The second attempt sends `"power_cut_acknowledged": true`.
3. **Any other refusal,** such as a running control loop or a latched E-stop,
   shows its detail with no acknowledge path.

The acknowledgement is never asked for before the server has retried, so
operators are not trained to tick it by reflex.

### The state chip

The chip shows one label, chosen by the first rule that matches:

| Order | Label | Tone | When |
| --- | --- | --- | --- |
| 1 | E-stopped | red | The E-stop latch is set |
| 2 | torque unconfirmed | red | The server could not confirm torque-off (C2) |
| 3 | state unknown | grey, dashed | No source is fresh (see [Freshness](#freshness)) |
| 4 | collapsing | amber | `stop_state` is `collapsing` |
| 5 | halted | amber | `stop_state` is `halted` |
| 6 | control unknown | grey, dashed | The control-state poll has no answer, or keeps failing |
| 7 | fault, estopped, watchdog timeout, lease expired, operator timeout | red | The control plane reports that unsafe state |
| 8 | armed | amber | A motion session is armed |
| 9 | the control state, for example "disarmed" | neutral | Otherwise |

Pressing the chip opens a popover with five rows:

- **State:** the chip label.
- **Control:** the control-plane state.
- **Torque:** `on`, `off`, `unconfirmed` or `unknown`.
- **Source:** which source decided the state. It reads one of:
  - "Live telemetry";
  - "Status poll, 2s ago (telemetry link down)";
  - "No fresh source (status 12s old)";
  - "No response from the robot server yet".
- **Stops:** the stops the server offers.

The popover also links to Diagnostics. Navigating, or pressing outside the
popover, closes it. `Escape` does not close it, because `Escape` is the E-stop.

Screen readers hear the chip as "Safety state: …". Two polite live regions
announce changes:

- the safety state, whenever it is known;
- each stop request and its outcome, such as "E-stop sent from the Escape key."
  followed by "E-stop latched. Motors off."

### Banners

Safety banners are pinned under the header in a region named "Safety notices".
When nothing needs attention, the region is not rendered. When several banners
apply, they stack in this order:

| Banner | When | What it offers |
| --- | --- | --- |
| Request alerts | An E-stop answered 503, or an E-stop, Halt, Collapse, Soft stop or reset request failed | The server's detail and the next step, such as "Use E-stop if the arm must stop now". E-stop, Halt and Soft stop alerts are critical; the others are warnings. Each can be dismissed. |
| Fault: cut physical power if anything moves | `torque_unconfirmed` (C2, F7) | A link to Diagnostics, where the fault reset retries the torque-off first (see [Clearing a fault](#clearing-a-fault)) |
| E-stop latched | The latch is set | Hold to reset E-stop, as described above |
| Halted — holding position | `stop_state` is `halted` and no E-stop | A countdown to the automatic Collapse; while no operator page sends a supervision heartbeat it reads "Unsupervised: collapses in N s" instead. **Keep holding** re-sends Halt, which renews the hold timer. **Collapse now** appears when the server offers Collapse. |
| Collapsing | `stop_state` is `collapsing` and no E-stop | Progress, the time left, and **Halt the descent** when the server offers Halt |
| Address refused | The status poll gets HTTP 400 from the server's Host guard (C1) | The server's detail, verbatim, and how to allow this address |
| Robot state unknown | No source is fresh, after at least one status answer | Says the state shown may be out of date, that the stop buttons still send, and to confirm the arm visually |
| Live link refused | The telemetry socket was refused (C1); see [Link problems](#link-problems) | The server's reason when the browser exposes it, and how to allow this address |

The Halt countdown is measured on the server's clock. It is
`hold_expires_at − server_time` from the same response, minus the client time
that has passed since that response arrived. A wrong clock on the operator's
machine never changes it.

While `supervision.supervised` is false, the hold lasts only the grace period:
`grace_s` from `lost_at` (or from the halt, if later), never past the hold
timer, measured on the same server clock. The banner then shows "Unsupervised:
collapses in N s", and a polite live region announces the switch each way
("Supervised again: back to the hold timer."), not every tick. A server that
does not send `supervision` gets the plain hold countdown.

An E-stop replaces the Halted and Collapsing banners.

Toasts never carry safety state. Stops, latches and faults appear only in the
bar and the banners, which nothing covers.

## The backend badge

Next to the link indicator, the header always shows what commands drive, from
`backend` in the shared status poll:

| Badge | Style | Meaning |
| --- | --- | --- |
| REAL | normal, solid | Commands move the physical arm |
| SIM | dashed, violet | MuJoCo simulation: a simulated arm moves, not the physical one |
| MOCK | dashed, blue | Mock backend: no motors move |

Screen readers hear "Backend: …" with the same sentence the tooltip shows.
Under sim, Overview's follower tile reads "Simulated · MuJoCo simulation: no
physical arm" and Settings labels the follower pill "follower (simulated)".

A server before v0.2 has no `backend`. The badge then reads MOCK when `mock`
is true and otherwise REAL ("Real arm assumed"), because a simulation must
never be assumed where the arm might be real.

## Freshness

The UI reads robot state from two places:

- one WebSocket (`/ws/telemetry`), which streams frames at about 20 Hz;
- one shared HTTP poll of `GET /api/robot/status` every 1.5 s and of
  `GET /api/control/state` every 2 s.

Every page reads from these. No page polls status or control on its own
(F28).

### Two kinds of freshness (C6)

The telemetry link and the joint samples are judged separately.

**Transport freshness** asks whether the link is alive:

- The link counts as down when the socket is not open, or when no valid frame
  has arrived for 2.5 s.
- A socket that has not opened, or has delivered nothing, for 5 s is abandoned
  and replaced (F26). This catches a half-open connection, for example after a
  Wi-Fi roam.
- Reconnects back off from 0.75 s, doubling up to 10 s.
- A malformed frame is ignored and does not drop the link.

**Sample freshness** asks whether anything is reading the joints. Each frame's
`observation_live` field answers it:

| Sample state | Meaning |
| --- | --- |
| live | `observation_live` is true: an active loop or an armed session sampled the joints within the last 500 ms |
| idle | `observation_live` is false: nothing is sampling the joints. This is normal, not a fault. |
| stale | No frame yet. On a server older than C6, also a running loop whose sample is missing or more than 6 s old. |

**An idle robot is neutral, never amber (F8).** In v0.1, every idle robot
showed amber "stale", so operators learned to ignore amber. Amber and red now
mean that something which should be current is not.

The header's link indicator always shows the time since the last frame (F26):

| Indicator | Label | When |
| --- | --- | --- |
| red | "Connecting to robot", "Telemetry link down · reconnecting" or "Live link refused by the server" | The link is down |
| grey | "Idle · no live joint stream" | The link is up and nothing is sampling the joints |
| amber | "Joint samples stale" | The link is up and a loop's samples are overdue |
| green | "Joint stream live", with the frame rate | The link is up and the samples are live |

The same rules drive the Telemetry tile and the twin caption on Overview, for
example:

- "Idle: the last pose read; nothing is sampling the joints"
- "Link stale: the last known pose, not the live arm"

The twin never presents an old pose as live.

### Which source to believe (F4)

- **The E-stop latch** comes from the live telemetry frame while the link is
  fresh, and from HTTP status otherwise. A stale WebSocket never outvotes fresh
  HTTP status.
- **The state is known** when the link is live, or when HTTP status is fresh.
  Status is fresh when its last success is under 5 s old and the poll has not
  failed twice in a row (F11). When neither holds, the chip reads "state
  unknown" and the "Robot state unknown" banner appears.
- **The stop fields** (`torque_state`, `stop_state`, `halt`, `collapse`) come
  from the live frame when it carries them, and from status otherwise. A live
  socket from an older server cannot hide them.
- **The mode** shown on Overview, Teleop & capture, Models and Evaluation (F21)
  comes from the live frame, or from status while status is fresh. Otherwise
  the mode is unknown, and Overview reads "offline". A dropped link never keeps
  showing the last mode as current.

### Polling and render load (F28)

- HTTP polls pause while the browser tab is hidden and refetch as soon as it is
  visible again. The telemetry socket keeps running.
- A poll keyed to something, such as a selected arm or job, drops the previous
  key's data before it renders, and ignores late answers for that key (F18).
- The derived safety view keeps its identity while its content is unchanged.
  Components that read only the safety state, such as the safety bar and
  Manual joint control, therefore re-render when it changes, not on every
  20 Hz frame.

### Link problems

The server's Origin/Host guard (C1) can refuse this page. The UI tells two
cases apart:

- **HTTP refused.** The server answers every request with HTTP 400. The
  "Address refused" banner shows the server's detail exactly as sent.
- **Only the socket refused.** If the socket closes with code 1008, the banner
  shows the server's close reason verbatim. Browsers usually report a refusal
  before the handshake as 1006 instead, which hides the reason. So the UI also
  shows the banner after three sockets in a row fail to open while HTTP still
  answers. A server that answers nothing gets "Robot state unknown" instead,
  because a dead server also fails every socket.

Both banners explain how to allow this address:

- open the UI from the server's own address, or add the page's hostname to
  `SO101_ALLOWED_HOSTS`;
- when the page comes from another port or through a proxy, also add its origin
  to `SO101_CORS_ORIGINS`;
- restart the server after either change.

The guard always lets the E-stop, Halt and legacy stop routes through, so
E-stop, Halt and Soft stop keep working from a refused page. Collapse is not
exempt.

Every API and WebSocket URL is relative to the page's origin. The UI never
names a host or a port.

## Keyboard shortcuts

Press `?`, or the keyboard button in the header, to see this list in the UI.

| Keys | Where they work | What they do |
| --- | --- | --- |
| `Escape` | Everywhere, including text fields and dialogs | E-stop |
| `H` | Outside text fields | Halt, when the server offers it. Otherwise nothing. |
| `Space` or `Enter`, held | On the deadman button | Hold to stream manual commands. Letting go stops motion. |
| `←` `→` | While the deadman is held and motion is armed | Move the selected joint's target by 1, or by 5 with `Shift`, inside the joint's range |
| `↑` `↓` | While the deadman is held and motion is armed | Select the joint to jog. The selection stops at the first and last joint. |
| `G`, then a letter | Outside text fields, with no dialog open | Go to a page (below) |
| `?` | Outside text fields | Show or hide the shortcut list |

Collapse has no key; use its button.

For the page shortcuts, press `G`, let go, then press the letter within 1.5 s:

| Letter | Page | Letter | Page |
| --- | --- | --- | --- |
| `O` | Overview | `R` | Training |
| `T` | Teleop & capture | `M` | Models |
| `W` | Digital twin | `E` | Evaluation |
| `Q` | Quest & hands | `C` | Calibration |
| `D` | Datasets | `X` | Diagnostics |
| `A` | Agent lab | `S` | Settings |

### Rules

- `Escape` is the E-stop everywhere, so no dialog, popover or drawer closes on
  `Escape`.
- **Letter keys are ignored while you type.** This covers text and number
  inputs, text areas, selects and editable content. Buttons, checkboxes,
  radios and sliders are not text fields.
- **Letter shortcuts are also ignored** in these cases:
  - while `Ctrl`, `Alt` or `Super` (`Cmd` on a Mac) is held;
  - during an input-method composition;
  - when the focused control has already used the key itself, such as
    type-ahead in a list.
- Capitals work the same as lower case.
- **No page uses `G` or `H` as its letter.** `G` then `H` halts and does not
  navigate. An unknown letter after `G` ends the sequence.
- Key repeat never repeats a stop or a page change.
- **Nothing navigates while a dialog is open,** and `?` does not open the list
  on top of another dialog.

### Why page changes are allowed at any time

Navigation keys are not tied to the deadman. This replaces the v0.2 design's
earlier rule that navigation keys work only while the deadman is held.

Leaving Teleop & capture unmounts Manual joint control. If the session may be
armed, that sends a stop: a Halt when the server offers it, otherwise an
owner-scoped disarm. It then releases the lease. A page change can therefore
stop the arm, but never move it.

Manual motion also stops in these cases:

- the deadman is released;
- the browser window loses focus;
- the tab is hidden (this also releases the lease);
- a lease heartbeat fails (this releases the lease too).

### The deadman

The deadman is the **Hold to move** button in Manual joint control, on Teleop &
capture.

- **Pressing it** with the primary pointer button, `Space` or `Enter` arms the
  session and sends the first target in one request
  (`POST /api/control/arm-command`). Commands then stream for as long as the
  hold lasts.
- **The pointer is captured,** so releasing the button anywhere in the window
  ends the hold. So do a cancelled pointer and a lost capture. A pointer
  release while no hold is in progress sends nothing (F3).
- **Releasing the key,** moving focus off the button, or leaving the window
  ends a keyboard hold.
- **The jog keys work only during an armed hold.** They change the streamed
  target, never the arm directly, so every step still passes the server's
  limits: the SafetyGuard joint limits, its per-step cap and the follower's
  `max_relative_target` of 5°. The large step (5) is no larger than that cap.
- **Arrows aimed at a text field** are typing, not jogging.
- **Before the session is armed,** the arrows do nothing and keep their normal
  behaviour.

## Dialogs

Every confirmation in the UI is an in-app dialog. The rules:

- **A dialog never closes on `Escape`.** `Escape` sends the E-stop, and the
  dialog stays open so the operator can see what was pending.
- **The dialog sits below the header,** so E-stop and Halt stay visible and
  clickable while it is open.
- **Focus moves into the dialog,** to Cancel by default. Tab cycles inside it,
  and focus returns to the control that opened it.
- **Clicking the backdrop does nothing.** Only the dialog's own buttons decide.
- **No page uses `window.confirm`, `alert` or `prompt`.** A native dialog
  blocks the page's JavaScript while it is open, so `Escape` could not reach the
  E-stop.

## First-run setup checklist

The checklist is the first card on Overview. It shows what a new workstation
still needs. Every item is read from the server, except the stops
acknowledgement, which the operator gives. **Nothing in the checklist runs
anything on the robot.**

| Item | Required | Source | Done when |
| --- | --- | --- | --- |
| Robot backend | yes | `GET /api/capabilities` | The server is a mock or sim rehearsal, or reports that it can drive a real arm |
| Follower calibration | yes | `GET /api/calibration/status/follower` | A calibration is saved, or the server is the mock backend, which uses default ranges. On a real server a file a mock run saved counts as missing, and the server's reason is shown. |
| Leader calibration | no | `GET /api/calibration/status/leader` | As for the follower. Without one it shows "optional": it is needed only to teleoperate with the leader arm. |
| Preflight | yes | `GET /api/preflight/latest` | The last report passed on the real backend, or, on a mock or sim server, a rehearsal passed. The server serves only its own backend's report; a rehearsal never counts on a real server. |
| Know your stops | yes | This browser | The operator pressed "I know the stops" |

Each item shows one of these states:

| State | Meaning |
| --- | --- |
| done | Complete |
| rehearsal | A mock or simulated preflight passed on a mock or sim server. It counts as complete there (the card then reads "Rehearsal setup complete"), but it is not evidence about a real arm. On a real server a rehearsal is "to do". |
| to do | Something to do, with a link or a command |
| problem | The evidence is bad, such as an unreadable calibration file or a failed preflight. Failed checks are listed. |
| optional | Not needed for a first run |
| in progress | A calibration is running; shows its percentage |
| checking | Waiting for the first answer |
| unknown | The server could not be read; shows its error |
| not available | The server has no such API |

The card's heading reads "Set up this workstation" with a count such as
"2 of 4 required steps done". Once every required item is done or a rehearsal,
it reads "Setup complete".

### What the checklist never does

- **It never calls `POST /api/preflight`,** because that request runs the
  checks and saves a report (a GET there answers 405). It reads only the last
  saved report.
- **When no report exists yet,** the item says to run `so101 preflight` in a
  terminal on the workstation. It also shows the server's 404 detail verbatim.
- **Running preflight from this page** arrives with the wave-2 preflight
  wizard. The component already takes a `preflightAction` slot for it.

### Refreshing, hiding and persistence

- **The checklist refreshes every 15 s while it is shown,** and not at all
  once hidden. A preflight run in a terminal shows up on the next refresh.
- **"Hide checklist"** removes the card, moves focus to the page, and shows a
  toast: "Setup checklist hidden. Show it again from Settings." Settings has a
  **Show setup checklist** button.
- **Both answers are stored in this browser's local storage** under versioned
  keys:
  - `so101.onboarding.dismissed.v1` for the hidden checklist;
  - `so101.onboarding.stops.v1` for the stops acknowledgement.

  A future checklist with new items can use new keys and ask again.
- **A change in one tab shows up in every other tab.** If storage is blocked,
  the answers last for the current page load.

### Backend

The Robot backend item reads `backend` (mock, sim or real). Under sim the
follower calibration item is done, because the server refuses to calibrate a
simulated arm, which has no encoders. A server before v0.2 sends only `mock`,
which is false under sim too; against one, the item says "a real or simulated
arm; the server does not say which" and quotes the backend of the last
preflight report when there is one.

## Accessibility

Every page follows these rules. The tests listed at the end check them.

- **Landmarks.** Each page has one banner (the header), one navigation
  landmark named "Main" (the whole sidebar), and one main region named after
  the page. A "Skip to content" link comes first. All visible text sits
  inside a landmark, a labelled section or a live region.
- **Headings.** Each page has exactly one `h1`, inside the main region, and
  heading levels never skip. Card titles are `h2`.
- **Page changes** update the document title ("Datasets · SO-101") and move
  focus to the new page. An unknown address shows a "Page not found" page with
  a link back to Overview.
- **Names.** Every button, link, form control, tab, progress bar and dialog has
  an accessible name. Decorative icons are hidden from assistive technology. A
  control repeated per row names its row, for example "Move lab/pick to
  recoverable trash". A state shown as an icon is also given in words, as in
  the Quest connection steps.
- **Visible focus.** A global `:focus-visible` outline (2 px, cyan) covers
  every control. A control that removes the outline must draw its own focus
  ring. Only programmatic focus targets, such as the main region, a dialog
  panel or a checklist heading, may go without one.
- **Colours** come from `src/styles/palette.css`: Tailwind v3's palette,
  pinned as hex under Tailwind 4 so every colour and contrast ratio stays as
  measured. A class that names an unpinned colour fails `a11yStyles.test.ts`.
- **Contrast.** Secondary text uses the `muted` colour token (`#8796ab`). It
  reads at 4.86:1 or better on every slate surface; `slate-500` does not.
  White text keeps 4.5:1 on its background in every state, including hover and
  press, so danger buttons darken on hover instead of lightening.
- **Reduced motion.** With `prefers-reduced-motion: reduce`, animations and
  transitions are cut to almost nothing, and decorative spinners and pings stop
  (`motion-safe:`). The twin snaps to each new pose instead of easing. The one
  exception is the hold-to-confirm fill (`data-essential-motion`). It shows how
  much longer to hold, so it keeps moving.
- **Status pills** always say their state in words, never colour alone.

## Shared components

New UI should use these components instead of hand-rolled markup:

| Component | Where | Notes |
| --- | --- | --- |
| `Button` | `components/Button.tsx` | Variants: primary, secondary, danger, ghost, success. |
| `Card` | `components/Card.tsx` | Its title is an `h2`. |
| `StatusPill`, `BoolPill` | `components/ui/StatusPill.tsx` | "label value" in words; screen readers hear "label: value". |
| `Field`, `INPUT_CLASS` | `components/ui/Field.tsx` | Wires the label, hint and error to the control. |
| `Dialog`, `ConfirmDialog`, `useConfirm` | `components/ui/Dialog.tsx` | The [dialog rules](#dialogs). `useConfirm` is a promise-based replacement for `window.confirm`. |
| `Tabs` | `components/ui/Tabs.tsx` | WAI-ARIA tabs. Arrow keys move and wrap; Home and End jump to the ends. Only the selected panel is rendered. |
| `EmptyState` | `components/Workstation.tsx` | Says honestly that there is nothing to show yet, and why. |
| `useToast` (the shell mounts `ToastProvider`) | `components/ui/Toast.tsx` | Routine confirmations only, never safety state. Info and success toasts disappear after 6 s; errors stay until dismissed. At most three show at once. |
| `HoldToConfirmButton` | `components/safety/HoldToConfirmButton.tsx` | For irreversible or safety-relevant confirmations. |

## Calibration history

The Calibration page shows the latest 100 attempts, newest first, with time,
arm/device, backend, requested capture duration, pass/fail, affected motors and the exact
error. Expand **View full log** for audit timestamps and available homing/encoder
diagnostics. Records live in the workstation's SQLite audit store and survive
browser refreshes and server restarts. The read-only
`GET /api/calibration/history?limit=100` endpoint accepts limits from 1 to 200
and never opens a motor bus.

Existing audit entries appear too; missing older backend/duration metadata is
shown as not recorded. A saved file followed by an error remains **FAIL**.
An attempt with no completion record is **INCOMPLETE**, or **RUNNING** when it
matches the active server attempt. A storage error is shown explicitly rather
than presented as empty history.

New attempts record requested homing offsets, register read-back, starting
encoder readings and measured ranges when available. Before announcing a sweep,
the worker verifies the homing-offset and reset-limit registers. The UI asks the
operator to hold the midpoint pose during preparation; it asks for a sweep only
after the worker enters that phase. History collection does not initiate a
calibration or motion.

## Known limitations

- **Close code 1008 is usually hidden.** Browsers report a refused socket as
  1006, so the "Live link refused" banner usually comes from the
  three-failed-attempts rule and cannot quote the server's reason.
- **The UI must be served from the root path,** because API URLs are
  root-relative (`/api/...`).
- **Some pages still re-render on every frame.** Pages that read the raw
  telemetry stream re-render at its frame rate. The safety view does not.

## Tests

From `frontend/`, run `npm run lint`, `npm run test` and `npm run build`.

| Area | Tests |
| --- | --- |
| Safety view derivation (F4, C2, C11, Halt countdown, mode) | `src/lib/safetyView.test.ts` |
| Safety bar, E-stop (including the 503 latch), Halt, Collapse, reset lockout, banners, link problems | `src/components/safety/safetyShell.test.tsx` |
| v0.1 audit reproductions (F2, F3, F4, F5, F18) | `src/test/safetyShell.regression.test.tsx` |
| Hold-to-confirm | `src/components/safety/HoldToConfirmButton.test.tsx` |
| Clearing a fault (retry first, then the power-cut acknowledgement) | `src/pages/Diagnostics.test.tsx`, `src/lib/errors.test.ts` |
| Diagnostics export allowlist (no page URLs, port paths or backend free-form check/event text) | `src/pages/Diagnostics.test.tsx` |
| Telemetry link and sample freshness (C6, F8, F26, C1) | `src/hooks/useTelemetry.test.tsx` |
| Polling (keys, hidden tabs, staleness) | `src/hooks/usePolling.test.tsx` |
| Shortcuts, the `?` list, navigating during a hold | `src/components/shortcuts/shortcuts.test.tsx` |
| Deadman | `src/components/DeadmanButton.test.tsx`, `src/components/ManualControlPanel.test.tsx` |
| Jog keys | `src/lib/jog.test.ts`, `src/components/ManualControlPanel.test.tsx` |
| Leases (heartbeat, blur, hidden tab, unmount) | `src/hooks/useMotionLease.test.tsx` |
| Dialogs, and no native dialogs | `src/components/ui/Dialog.test.tsx` |
| Shared components | `src/components/ui/primitives.test.tsx` |
| Shell: drawer, skip link, page focus and title | `src/components/Layout.test.tsx` |
| API client and error text (C1 hints) | `src/api.test.ts`, `src/lib/errors.test.ts` |
| Setup checklist | `src/lib/setupChecklist.test.ts`, `src/components/onboarding/SetupChecklist.test.tsx`, `src/lib/onboarding.test.tsx` |
| Landmarks, headings, names, ids on every page | `src/test/a11y.test.tsx` |
| Contrast, focus styles, motion rules | `src/test/a11yStyles.test.ts`, `src/hooks/usePrefersReducedMotion.test.tsx` |
| Tailwind 4 parity (v3 defaults kept, space-y children) | `src/test/tailwindParity.test.ts` |
| Backend badge and wording (mock, sim, real, older servers) | `src/lib/backend.test.ts`, `src/components/Layout.test.tsx`, `src/pages/Dashboard.test.tsx`, `src/pages/Settings.test.tsx` |

### Live calibration readings

The calibration page polls cached worker telemetry every 250 ms. The worker
publishes raw current readings, initial post-homing readings, running
minimum/maximum and sample counts from its existing reads. GET status never
opens a motor bus, reads registers or waits for the bus I/O lock.

Each row shows requested/read-back homing offsets and range checks. An encoder
extreme outside 0–4095 or a homing mismatch is FAIL immediately. A span below
100 ticks stays Pending while sampling and fails at final validation. Valid so
far means the observed range meets these checks; it does not prove the operator
reached both physical end stops. PASS requires completed calibration. Wrist roll
uses the pinned upstream fixed 0–4095 range; these are not measured extremes.

The phase banner requests movement only during sampling. Stalled worker updates
and failed HTTP polling are marked stale. Final attempted readings freeze after
completion, remain in attempt history diagnostics, and are distinct from
registers restored on rollback. A new attempt clears the prior live snapshot.
Real telemetry requires restarting an older server after the software update;
mock calibration has no raw encoder samples.

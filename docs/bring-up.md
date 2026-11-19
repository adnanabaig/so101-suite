# Hardware bring-up

This guide takes an SO-101 arm pair from "available, with motor power off" to
an auditable hardware-in-the-loop (HIL) record. It covers the four parts of
the bring-up kit:

| Part | Command | API | What it does to the arm |
| --- | --- | --- | --- |
| Preflight | `so101 preflight` | `POST /api/preflight`, `POST /api/preflight/bus` | Pings and reads registers. It never writes a register and never changes torque. |
| Rest pose | `so101 rest-pose record`, `show`, `clear` | none | `record` turns torque **off** on every follower motor, then reads positions. |
| Servo health | read by preflight; polled by a monitor inside the server | `GET /api/robot/health`; a `health` summary in status and telemetry | Reads only. A critical reading makes the server Collapse the arm and latch a fault. |
| HIL runner | `so101 hil run`, `list`, `status`, `report` | `/api/hil/...` | Probes only observe. The operator performs every action. |

The kit makes the real-arm gate in [SUITE_V0.md](SUITE_V0.md) runnable and
auditable. It does not pass the gate: only an operator with a real arm can.

Physical setup (motor IDs, cameras, calibration, first teleop) is in
[hardware-setup.md](hardware-setup.md). Control and recording problems are in
[troubleshooting.md](troubleshooting.md).

## Bring-up order

1. Install the robot extra on the robot computer while both arm motor supplies
   are switched off ([Install](#install)). USB adapters may be attached for
   identification.
2. Before applying arm power, clamp the bases, clear the arm envelope, put a
   padded mechanical catch below the follower, outside the recorded rest-pose
   and Collapse path, and make its physical power cutoff reachable without
   crossing the arm. Confirm the follower and leader supply labels and servo
   variants, and prove the two stable USB aliases by tracing or unplugging one
   cable at a time. A second person should stay at the cutoff during first
   power and motion; never plan to catch the arm by hand.
3. Only after the operator, and the cutoff operator when separate, confirm
   that setup, power the arms. Keep the follower low over the catch and run
   `so101 preflight`. With no server running it opens the buses itself; with a
   Suite server running and the robot idle, the server lends them
   ([Who may open the bus](#who-may-open-the-bus)). Fix every `fail`. Before the
   first calibration, `calibration` fails and `rest_pose` is incomplete; that is
   expected.
4. Calibrate with the server stopped: `so101 calibrate follower`, then
   `so101 calibrate leader`. Calibration is a hands-on sweep and turns torque
   off. Each real CLI command requires an external confirmation before it
   opens the bus: keep the arm low over the mechanical catch, move only the
   intended links away from pinch points, and keep the cutoff operator ready.
5. With the same precautions, record the rest pose: `so101 rest-pose record`.
   It also turns follower torque off. Record it after calibration because a
   later follower calibration invalidates it.
6. Run `so101 preflight --require-hardware`. It must exit 0.
7. Start the server with `SO101_MOCK=0` (`so101 serve`). In a second terminal,
   run `so101 hil run --operator "Your Name" --group gate`.
8. Continue with `--group stop`, then `--group verification`, one run each.
   The runner does not check that the previous group passed: start the next
   group only after the previous run exited 0. Full stop evidence therefore
   precedes data collection, training and policy rollout. The `stop` group is
   the real-arm E-stop, Halt and Collapse test; keep the follower over the
   mechanical catch, hands clear and the power-cutoff operator at the switch
   for the whole group. Its probes measure from the stop fields in
   `/api/robot/status` (see [Stop steps](#stop-steps)).
9. Check and archive the evidence of each run: `so101 hil report RUN_ID`.

The server must still be running and idle during the gate's preflight probes.
`RobotManager` lends a read-only view of each bus under its lock. Never repeat
the standalone calibration or rest-pose commands after the server takes bus
ownership. Keep that server running between HIL steps except for controlled
lifecycle tests named by the catalog: `v01-collection-resume` deliberately
restarts the same build/backend/data folder, and a non-default hold-timer
exercise pauses the open run, confirms torque off, restarts the same server and
resumes the same run id. Release evidence uses the default 60-second hold and
does not restart for `stop-hold-timer`.

Two gate transitions need deliberate recovery before the run continues:

- `gate3-install` starts teleop through the running server with the follower
  adapter unplugged. The start route is asynchronous, so its initial HTTP 200
  only accepts the request. Wait for the operation/audit terminal fault,
  confirm no motion, reconnect, inspect torque, deliberately reset the fault
  and require idle, unlatched preflight before attesting.
- `gate7-stop-tests` cuts physical follower power. Keep it off until the
  session ends, then restore it over the mechanical catch and confirm no
  motion. Deliberately resolve any latch, rerun hardware preflight through the
  idle server, and require idle mode, clear latches and confirmed torque-off
  before starting the stop group.

## Rehearsal and hardware

Every preflight result and every HIL probe records the backend that produced it.

| Backend | Selected by | Behavior |
| --- | --- | --- |
| `real` | `SO101_MOCK=0` (the default) | Opens the configured ports through LeRobot 0.6.1. It fails closed: when LeRobot, the Feetech SDK or a port is missing, the checks fail. Nothing falls back to the mock. |
| `mock` | `SO101_MOCK=1` | A synthetic bus. Results are labeled `rehearsal: true` and the CLI prints REHEARSAL. |
| `sim` | `SO101_BACKEND=sim` | Labeled a rehearsal. Bus, calibration and rest-pose checks skip, and the server's health monitor is off. |

Rehearsal values are plainly synthetic: firmware `0.0`, 5.0 V, 30 °C. A
rehearsal is never hardware evidence:

- `ready_for_hardware` is true only on the real backend after a recorded rest
  pose matches the current follower and calibration without clamping.
- `so101 preflight --require-hardware` exits 4 on a mock or sim run.
- A HIL run is `hil_evidence` only on the real backend ([Evidence](#evidence)).
- The real backend never sees a rehearsal's saved results, although every
  backend shares the data folder. `GET /api/preflight/latest` serves only the
  server backend's own report ([Reports](#reports)). A Suite calibration a
  mock run saved (device `mock`, or an older file holding exactly the default
  limits) counts as missing on the real backend: diagnostics warn, the
  calibration status reports nothing saved, preflight warns, and the
  follower's SafetyGuard keeps its default limits. Recalibrate on the real
  arm.

For example, a real-backend preflight on a computer with no arm and without
the robot extra exits 1. `lerobot`, `serial_ports`, both `calibration` checks
and `cameras` fail. Every bus check skips because bus access is "unavailable":
LeRobot 0.6.1 or the Feetech SDK is not importable.

## Supported platforms

The kit supports Windows 11, Linux and macOS. It lists ports with pyserial's
`comports()`, never with a hard-coded or globbed device path. It identifies
adapters by USB serial number and VID:PID.

Development and testing happened on macOS without an arm attached. The Windows
and Linux behavior follows pyserial 3.5's source and is covered by unit tests
with fake port lists. It is **unverified on real Windows and Linux hosts**:
port naming, permissions, the bus lock (`msvcrt.locking` on Windows,
`fcntl.flock` elsewhere) and the server probe's timing.

## Install

Install on the computer the arms plug into, in a dedicated virtual environment,
with Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`). Run these from the
repository root.

| Shell | Create and activate a virtual environment |
| --- | --- |
| macOS or Linux | `python3 -m venv .venv` then `source .venv/bin/activate` |
| Windows PowerShell | `py -3.12 -m venv .venv` then `.venv\Scripts\Activate.ps1` |

Then install the robot extra:

```text
python -m pip install -e ".[robot]"
```

Use the double quotes as shown. They work in PowerShell, cmd and POSIX shells;
single quotes do not work in cmd. Where `make` is available,
`make install-robot` runs `pip install -e ".[robot,train,dev]"`, which also
installs the training stack and the developer tools.

The `robot` extra pins:

- `lerobot[dataset,feetech]==0.6.1`. The kit checks the installed LeRobot
  against this pin (the `lerobot` preflight check).
- `feetech-servo-sdk>=1.0.0,<2.0.0`, the `scservo_sdk` module LeRobot's Feetech
  bus uses.
- `pyserial>=3.5`. LeRobot's own `feetech` extra also requires
  `pyserial>=3.5,<4.0` (LeRobot 0.6.1 `pyproject.toml:161,170`).
- `opencv-python>=4.9` for cameras.

Before the first run:

- Leave `LEROBOT_HOME` unset. LeRobot 0.6.1 raises `ValueError` on import when
  it is set (`utils/constants.py:61-65`); use `HF_LEROBOT_HOME` instead.
- Leave `SO101_BAUDRATE` unset or at `1000000`. SO-101 servos run at
  1,000,000 baud (LeRobot `feetech.py:44`).
- Set the supply profile when a follower uses the 12 V servo variant:
  `SO101_HEALTH_FOLLOWER_SUPPLY=12v`. The default `7.4v` is the standard
  STS3215 variant, which the SO-101 kit powers from a 5 V supply. The SO-ARM100
  README says the 12 V variant needs "a 12V 5A+ power supply instead of a 5V
  one" and that "the leader arm is always 7.4V for the SO101".

## Serial ports

Each arm has its own USB adapter and port. Find both ports with LeRobot's
`lerobot-find-port`: it lists the ports, asks you to unplug one adapter, and
prints the port that disappeared (LeRobot 0.6.1 `lerobot_find_port.py:36-41`).
LeRobot's own connect error suggests the same tool (`motors_bus.py:535-539`).

| OS | Port names | Notes |
| --- | --- | --- |
| Linux | `/dev/so101-follower` and `/dev/so101-leader` | Stable names that the udev rules from `install/install.sh --follower-serial S --leader-serial S` create (the serial numbers come from `python -m so101.platform_info`; see [install.md](install.md)). Without them: `/dev/ttyACM0` or `/dev/ttyUSB0`, whose number can change after a reboot or re-plug, so preflight warns; the link under `/dev/serial/by-id/` is stable too. |
| Windows | `COM3`, `COM4`, ... | Device Manager lists them under "Ports (COM & LPT)". Windows gives a COM port to one program at a time, so close serial monitors and other tools first. |
| macOS | `/dev/tty.usbmodem...` or `/dev/cu.usbmodem...` | pyserial lists the `cu.` device (`list_ports_osx.py:269-272`) and `lerobot-find-port` prints the `tty.` one. The kit treats the two as one port. |

**Linux permissions.** The device belongs to a group, usually `dialout`
(`uucp` on Arch). `ls -lL /dev/so101-follower` (or `ls -l /dev/ttyACM0`) shows
which. `install/install.sh` adds you to it; by hand, add yourself and start a
new login session:

```text
sudo usermod -aG dialout $USER
```

Then log out and back in. Preflight reports "permission denied" with this fix.

**Set the ports.** On Linux the defaults are `/dev/so101-follower` and
`/dev/so101-leader`, the installer's udev names (`so101.config.default_arm_port`),
so with those rules in place there is nothing to set. Elsewhere the defaults
are the macOS device names of the original development kit's adapters, so
Windows, macOS with other adapters, and Linux without the udev rules set both
ports. Put them in the data folder's `.env`
(`~/.local/share/so101/.env` on Linux, `~/Library/Application Support/so101/.env`
on macOS, `%LOCALAPPDATA%\so101\.env` on Windows, or `$SO101_HOME/.env`), which
every `so101` command reads whatever directory it runs from. A `.env` in the
directory you run `so101` from (a checkout's own) also applies and wins over
the data folder's; `so101 info` lists the files in effect. On Linux, written
out with the installer's udev names (the installer adds the two port lines
itself when you pass the serial numbers):

```dotenv
SO101_MOCK=0
SO101_FOLLOWER_PORT=/dev/so101-follower
SO101_LEADER_PORT=/dev/so101-leader
SO101_FOLLOWER_ID=so101-follower
SO101_LEADER_ID=so101-leader
```

Without the udev rules, use the by-id link
(`SO101_FOLLOWER_PORT=/dev/serial/by-id/usb-...-if00`). On Windows the values
look like `SO101_FOLLOWER_PORT=COM3`; on macOS like
`SO101_FOLLOWER_PORT=/dev/tty.usbmodem585A0077531`. To set one for a single
PowerShell session: `$env:SO101_FOLLOWER_PORT = "COM3"`.

**What preflight reports.** The `serial_ports` check runs on the real backend
only. It fails when:

- the port listing fails
- a configured port does not exist
- the path is not a character device
- you may not open it
- both arms name the same device

It warns when:

- a port exists but pyserial does not list it
- a Linux name can renumber
- the two adapters report the same USB serial number. Cheap adapters
  sometimes do; then neither a by-id link nor a serial-number lookup can tell
  them apart.

Each listed adapter appears with its serial number and VID:PID.

## Preflight

```text
so101 preflight [--arm follower|leader|both] [--json] [--require-hardware] [--save/--no-save]
```

```text
POST /api/preflight?arm=both&save=true     run every check now
POST /api/preflight/bus?arm=both           one read-only pass per arm on a bus the server lends (raw facts)
GET  /api/preflight/checks                 the check catalog, in run order
GET  /api/preflight/latest?origin=any      the server backend's last saved report (any, cli or api)
```

Running preflight is a POST: it reads the motor buses, and `/api/preflight`
writes a report, so it must pass the Origin guard
([control-plane-security.md](control-plane-security.md)), which checks
state-changing methods only. A cross-origin page gets 403; a request with no
`Origin` header (the CLI, curl) passes. `GET /api/preflight` and
`GET /api/preflight/bus` answer 405 and run nothing. The catalog and the saved
report stay GET; they only read files.

The CLI prints:

- a table of status, check, result and remediation
- the bus access mode
- notes
- the counts, `ready_for_hardware` and the exit code
- the path of the saved report

`--json` prints the full report on stdout and sends logs to stderr.

### Checks

Each check returns `pass`, `warn`, `fail` or `skip`, with a message, a
remediation and the measured values. Per-arm checks run once for each arm and
report as `<id>.<arm>`, for example `motors.follower`.

| Check | Scope | Fails when | Warns when | Skips when |
| --- | --- | --- | --- | --- |
| `environment` | host | Python older than 3.12. On the real backend: `SO101_BAUDRATE` is not 1000000, or `LEROBOT_HOME` is set. | the same baud or `LEROBOT_HOME` problem on a rehearsal ("would fail on the real backend") | never |
| `lerobot` | host | On the real backend: LeRobot is missing, is not 0.6.1, does not import, lacks the Feetech SDK, or a symbol, signature or control-table entry differs from the pin. | the same problems on a rehearsal | never |
| `serial_ports` | host | see [Serial ports](#serial-ports) | see [Serial ports](#serial-ports) | mock and sim (not blocking) |
| `motors` | per arm, bus | the bus does not open; an ID from 1 to 6 does not answer; a motor's model number is not 777 (STS3215) | a nonzero Status register (bit meanings unverified) | the bus was not read |
| `firmware` | per arm, bus | a read error, or the motors report different firmware versions. LeRobot's connect refuses mixed firmware (`feetech.py:145-157`). | some motors did not answer | the bus was not read |
| `supply_voltage` | per arm, bus | a reading at or beyond a critical edge of the supply band | a reading at or beyond a warn edge; a reading outside the servo's own `Min_Voltage_Limit`/`Max_Voltage_Limit` | the bus was not read |
| `temperature` | per arm, bus | at or above the critical threshold | at or above the warn threshold | the bus was not read |
| `calibration` | per arm | LeRobot's calibration file is missing or malformed; a motor id is wrong; a range is empty or outside 0..4095; a span is under 100 ticks; the Suite calibration belongs to another device; the motor registers (`Min_Position_Limit`, `Max_Position_Limit`, `Homing_Offset`) differ from the file | a span under 400 ticks; `wrist_roll` is not the full turn 0..4095; a joint rests more than 20 ticks outside its range; no Suite calibration file | the files are fine but the bus was not read; sim (not blocking) |
| `rest_pose` | follower | never | on a rehearsal, no recorded pose matches or the pose was clamped | on real, no recorded pose matches or the pose was clamped (blocking); sim (not blocking) |
| `torque_off_stability` | per arm, bus | torque is on for any motor; more than one of the 10 position reads fails; a position is outside 0..4095 | a joint moves more than 3 ticks while resting; one read was retried | the bus was not read |
| `cameras` | host | a configured camera does not open (a RealSense failure carries its own OS-specific fix) | a frame's mean brightness is below 5 (dark); a RealSense on a USB 2 link, with firmware older than recommended, or with under 10% valid depth pixels | none configured; the server owns the cameras and reports no live frame; sim (not blocking) |
| `disk_space` | host | a data, datasets or outputs directory is not writable, or has under 2 GiB free | under 10 GiB free | never |
| `estop` | server | never | the software E-stop or a runtime fault is latched. Reset them in the UI, or with `POST /api/robot/estop/reset` and `POST /api/robot/fault/reset`, each with the body `{"operator_id": "<your name>", "confirm": true}` ([C2 resets](control-plane-safety.md#deliberate-resets-c2)). | no server is running (not blocking) |

A check that crashes is reported as a `fail` with the exception.

### Status, readiness and exit codes

A `skip` is **blocking** when it leaves hardware readiness unproven, for
example a bus check that could not read the bus. The non-blocking skips are
marked in the table above.

- Overall: `fail` if any check failed; else `incomplete` if a blocking check
  skipped; else `warn` if any warned; else `pass`.
- `ready_for_hardware` is true only on the real backend with an overall
  `pass` or `warn` and a recorded rest pose bound to the current follower and
  calibration that needs no clamping. Missing that evidence makes the report
  `incomplete`, including a partial run that did not check the follower.

| Exit | Meaning |
| --- | --- |
| 0 | Ready. Warnings are allowed. |
| 1 | A check failed. |
| 3 | Incomplete: a blocking check skipped, for example because a server owns the bus. |
| 4 | `--require-hardware` was given and the run was a rehearsal (mock or sim). |

When several apply, the precedence is 4, then 1, then 3, then 0.

### Who may open the bus

A running Suite server owns both serial buses and the cameras. `RobotManager`
is the only owner of the motor buses, so the kit never opens a port the server
holds; the server lends an idle bus instead.

- **CLI, no server.** Before opening a bus, `so101 preflight` and
  `so101 rest-pose record` check that nothing answers `GET /api/robot/status`
  at `SO101_HOST:SO101_PORT`. Anything that answers, including another program
  or a timeout, counts as a running server. They then take an exclusive,
  host-global lock for each canonical serial device they will open, so data
  folder or checkout changes cannot create a second owner. If a lock is held,
  the bus checks skip as **refused** and preflight exits 3 (or 1 when another
  check failed).
- **CLI, Suite server running.** `so101 preflight` asks that server for the
  bus facts (`POST /api/preflight/bus`, no `Origin` header) and grades them
  with the same checks. The server lends each bus as below. A refusal, a
  server on another backend (`SO101_MOCK`/`SO101_BACKEND` differ), a server
  without the route or the method (another build: 404 or 405), or a failed
  request skips the bus checks as **refused**. Something
  that answers on the port but is not a Suite server is refused as before.
  `so101 rest-pose record` still needs the server stopped.
- **API.** `POST /api/preflight` and `POST /api/preflight/bus` borrow each bus
  through `RobotManager.run_idle_bus_diagnostic` (the idle-bus rule below). On
  the mock backend the lent bus is the synthetic one, so a rehearsal meets the
  same refusals as the real arm.

**The idle-bus rule** (`run_idle_bus_diagnostic(target, fn, *, timeout_s,
opener)`):

- The robot must be idle. The manager refuses (HTTP 409 on the routes,
  `bus_access: refused` in a report) while the E-stop is latched, while manual
  control is armed or arming, while a Halt holds the arm, while a Collapse
  lowers it, while teleop, recording or a rollout runs, while calibrating, and
  when the arm to read is energized. A runtime fault does not block a
  diagnostic: reading is how you find a hot or brown-out servo.
- It waits for the manager's bus lock (`_io_lock`) at most `timeout_s` (the API
  waits 1 s); a bus that stays busy answers "motor bus is busy ... retry"
  (HTTP 503 from the manager, `refused` in the report). Once it has the lock it
  checks the state again.
- The diagnostic runs under that lock on a read-only view of the bus: ping,
  read and sync-read only. Any write or torque method raises. A closed port is
  opened as LeRobot's bare `FeetechMotorsBus` with `connect(handshake=False)`
  and closed with `disconnect(disable_torque=False)`, which only closes the
  port. An open one is lent as it is and stays open.
- An E-stop during the diagnostic fails its next read at once, without bus
  I/O, and the result is discarded.

The report's `bus_access` field is one of:

| Mode | Meaning |
| --- | --- |
| `opened` | Read-only access under the canonical device lock (CLI with no server), or lent by RobotManager (the API, or the CLI through a running server). |
| `synthetic` | Mock backend: the synthetic bus, lent by RobotManager when a server runs. |
| `refused` | A device lock is held, RobotManager refused ("Stop teleop, recording or rollout, disarm, and let any Halt or Collapse finish (or reset the E-stop), then retry.") or its bus stayed busy, or the running server could not be asked. |
| `deferred` | API against a manager that cannot lend buses (no `run_idle_bus_diagnostic`); not this build. |
| `unavailable` | Sim, or LeRobot or the Feetech SDK is not importable. |

The read-only bus never writes: it pings and reads, and the HIL runner's
preflight probe now gets real bus checks from the server.

**Everyone takes the canonical device locks on the real backend.** The server
holds them for its whole lifetime (from startup to after its shutdown stop),
and refuses to start while a CLI command holds either one, naming that command.
`so101 calibrate` and the bring-up commands probe for a server and take the
lock, and also refuse while the persisted E-stop, a fault or an unconfirmed
torque-off is latched (exit 3; see "Commands that run without the server" in
[control-plane-safety.md](control-plane-safety.md)). Real `teleop`, `record`,
`replay`, `run` and `eval` are refused even with a clear latch: motion must use
the server so its watchdog and Halt-to-Collapse timer remain alive. A
server on a port other than `SO101_PORT` is invisible to the probe, but the
locks still keep it and a CLI command apart across different data folders and
checkouts. A udev alias and its underlying Linux device path share an identity;
disjoint ports do not block one another. Mock and sim take no device lock.
Calibration from the UI (`POST /api/calibration/start`) runs through the
server's manager, under its internal bus lock.

Before a real server can energize a follower, it writes a persistent
host-global intent for those device identities. It also retains real E-stop
and fault latches there after torque-off, so another data folder or checkout
still refuses after the device lock is released. A deliberate reset clears
the corresponding retained state. Mock and sim never read or write this
hardware state.

### Reports

Every run is saved, unless you pass `--no-save` (CLI) or `save=false` (API), as
`<data_dir>/preflight/<UTC timestamp>-<origin>-<backend>.json`. Copies are kept
per backend, as `latest-<backend>.json`, `latest-<backend>-cli.json` and
`latest-<backend>-api.json` (for example `latest-real.json`). Real, mock and sim
share the data folder, so `GET /api/preflight/latest` serves only the server's
own backend's report: a real-backend server answers 404 rather than show a sim
or mock rehearsal. A `latest.json` from before the per-backend copies is read
only by a server on the backend that wrote it. `SO101_DATA_DIR` defaults to
`./data` under the directory you start from.

`so101 doctor` is the older, non-invasive diagnostics page in CLI form. It does
not open the bus. Preflight is the bring-up check that does.

## Rest pose

The Collapse stop lowers the arm to the rest pose at a capped speed and then
turns the motors off. The pose must be one the arm can stay in without torque.
Collapse, preflight, `so101 rest-pose` and the HIL runner all resolve it with
the same function (`resolve_collapse_rest_pose`), so every view shows the pose
Collapse would use.

**Resolution order.**

1. The pose recorded with `so101 rest-pose record`. It counts only when its
   follower id, backend and calibration digest all match. The digest is the
   sha256 of the follower's LeRobot calibration file, so recalibrating
   invalidates the pose.
2. `SO101_REST_POSE`: six numbers in joint order
   (`shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper`, for
   example `0,-95,92.8,-85,0,5`, the model-derived default below), or a JSON
   object keyed by joint name.
3. The model-derived default, `so101.simulation.rest_pose.MODEL_REST_POSE`:
   folded, collision-free, at least 5 degrees inside the limits and
   gravity-stable in MuJoCo, but not verified on a real arm. A warning says
   so and points at `so101 rest-pose record`.

Every source is clamped to the SafetyGuard limits intersected with the MJCF
limits (SafetyGuard alone, with a warning, if the vendored model cannot be
read). Units are degrees for the five arm joints and 0..100 for the gripper.
On the real backend preflight's `rest_pose` check is a blocking skip until a
pose is recorded and validated for the current follower calibration, so the
report exits 3 and cannot claim hardware readiness. Rehearsals still warn.

**Commands.**

- `so101 rest-pose record [--yes] [--json]`
  - Needs exclusive bus access, so stop the server first. With a server
    running it refuses and exits 3.
  - Before opening the bus it asks you to confirm a padded mechanical catch,
    clear envelope and reachable physical cutoff (`--yes` skips this). It
    turns torque off on every follower motor, then asks you to fold by the
    links away from pinch points, move hands clear, let go and wait for the arm
    to settle. The catch must not touch or obstruct the intended rest pose.
  - It then samples `Present_Position` 10 times, 0.1 s apart, and saves the
    mean to `<data_dir>/rest_pose.json` only if every joint's spread is under
    1 degree (1 unit for the gripper).
  - Exit 1 means nothing was saved: the arm moved, a read failed, teardown
    could not confirm torque off, or the follower has no LeRobot calibration
    file to bind the pose to (run `so101 calibrate follower` first). An
    unconfirmed teardown persists the same fault latch as other standalone
    hardware commands before releasing its device lock. Exit 2: the sim backend
    has no arm.
  - On the mock backend the pose is synthetic, labeled REHEARSAL, and used only
    on the mock backend.
- `so101 rest-pose show [--json]` prints the pose Collapse would use, its
  source, the limits used and any warnings.
- `so101 rest-pose clear [--yes]` deletes the recorded pose. Collapse then
  falls back to `SO101_REST_POSE` or the model-derived default.

## Servo health

The kit reads servo health from the follower and leader buses. Preflight reads
it once per run; a monitor inside the server polls it at 1 to 2 Hz while the
server has a bus open ([Servo health monitor](#servo-health-monitor)).

### Registers and units

Register names, addresses and sizes are **verified** against LeRobot 0.6.1's
`STS_SMS_SERIES_CONTROL_TABLE` (`motors/feetech/tables.py:41-101`). The unit
scales and the Status bit meanings are **unverified**. They are the commonly
quoted STS-series values; LeRobot 0.6.1 does not state them, and they have not
been checked against a Feetech STS3215 manual. Raw values are kept next to the
converted ones.

| Register | Address, size | Converted unit | Scale |
| --- | --- | --- | --- |
| `Present_Temperature` | 63, 1 | °C | 1 °C per LSB (unverified) |
| `Present_Voltage` | 62, 1 | V | 0.1 V per LSB (unverified) |
| `Present_Load` | 60, 2 | % of max torque | 0.1 % per LSB (unverified). Signed: sign bit 10 is verified (`STS_SMS_SERIES_ENCODINGS_TABLE`); which direction is positive is not documented. |
| `Present_Current` | 69, 2 | mA | 6.5 mA per LSB (unverified). LeRobot does not sign-decode it, so it is a magnitude. |
| `Status` | 65, 1 | bit flags | bit 0 voltage, 1 sensor, 2 temperature, 3 current, 4 angle, 5 overload (unverified) |
| `Max_Temperature_Limit` | 13, 1 | °C | 1 °C per LSB (unverified) |
| `Max_Voltage_Limit` | 14, 1 | V | 0.1 V per LSB (unverified) |
| `Min_Voltage_Limit` | 15, 1 | V | 0.1 V per LSB (unverified) |

### Thresholds

All thresholds are Suite defaults, not Feetech specifications, and every one is
a setting. Boundaries are inclusive: a reading equal to a threshold takes that
threshold's level.

| Setting | Default | Meaning |
| --- | --- | --- |
| `SO101_HEALTH_FOLLOWER_SUPPLY`, `SO101_HEALTH_LEADER_SUPPLY` | `7.4v` | Supply profile, `7.4v` or `12v`. It picks the band below. |
| `SO101_HEALTH_SUPPLY_BAND_7V4` | `[4.0, 4.5, 8.0, 8.5]` | Volts: critical low, warn low, warn high, critical high. |
| `SO101_HEALTH_SUPPLY_BAND_12V` | `[9.5, 10.5, 13.0, 14.0]` | The same, for the 12 V variant. |
| `SO101_HEALTH_TEMP_WARN_MARGIN_C` | 15 | Warn at `Max_Temperature_Limit` minus this. |
| `SO101_HEALTH_TEMP_CRITICAL_MARGIN_C` | 5 | Critical at `Max_Temperature_Limit` minus this. |
| `SO101_HEALTH_TEMP_FALLBACK_WARN_C` | 55 | Used when the limit reads outside 40..100 °C or cannot be read. |
| `SO101_HEALTH_TEMP_FALLBACK_CRITICAL_C` | 65 | The same, for critical. |
| `SO101_HEALTH_LOAD_WARN_PCT` | 80 | On the absolute load. |
| `SO101_HEALTH_LOAD_CRITICAL_PCT` | off | |
| `SO101_HEALTH_CURRENT_WARN_MA` | 2000 | |
| `SO101_HEALTH_CURRENT_CRITICAL_MA` | off | |
| `SO101_HEALTH_STATUS_NONZERO_LEVEL` | `warn` | Level for a nonzero Status register: `ok`, `warn` or `critical`. |
| `SO101_HEALTH_POLL_HZ` | 1 | Monitor rate, 1 to 2 Hz. |
| `SO101_HEALTH_MAX_CONSECUTIVE_FAILURES` | 3 | Failed reads in a row before the monitor goes critical. One failed read is a warning. |

Set a band as a JSON array, for example
`SO101_HEALTH_SUPPLY_BAND_7V4=[4.2,4.6,7.9,8.4]`. It must be strictly
increasing. Set an optional threshold to `off` (or `none`, `null` or empty) to
disable it.

### Servo health monitor

The server polls servo health on its own thread, `so101-health`, separate from
the watchdog, so a slow or wedged health read never delays the deadman, the
hold timer or the Collapse backstop.

**What is polled, and when.**

- The follower while the server has its bus open: an armed session, teleop,
  recording or a rollout, a Halt's hold, a Collapse descent. The leader while
  teleop or recording has its bus open. The monitor never opens a port itself.
- The first read comes as soon as the bus opens, so a servo that is already
  hot or browned out shows at arm time. Then one read every
  `1 / SO101_HEALTH_POLL_HZ` seconds of the manager's clock, with no burst
  catch-up after a stall. At most one arm is read per 50 ms pass, so the
  follower's and the leader's reads never run back to back.
- Each session starts a fresh monitor: a read-failure streak never carries
  over from an earlier session.

**Not polled** (`active: false`, with `inactive_reason`): the bus is closed
(motors off); the E-stop is latched (it owns the bus); calibration owns the
bus; the bus has no health registers; the health thresholds are invalid (an
`robot.health.config_error` event says which); the server is shutting down;
and `SO101_BACKEND=sim`. Under sim, monitoring is off on purpose: the
simulated servos have no health registers, and the monitor reports no
synthetic values rather than risk a false critical.

**The bus lock, and a streaming loop.** Every read runs under the manager's
bus lock (`_io_lock`), like every other transfer. The choice is to interleave,
not to skip: a control loop holds the lock only for its own transfers (a few
milliseconds of each period), so a read waits at most `health_io_wait_s`
(50 ms) and slips in between. Only when the bus stays busy that long is the
poll skipped; it is counted (`skipped_busy`, `last_skip`) and retried 0.2 s
later, never counted as a read failure. The verdict and its handlers run after
the lock is released, so a critical verdict's Collapse takes the bus at once.

Measured with `scripts/measure_health_latency.py` on the mock bus (macOS dev
machine, teleop at 30 fps, health at 2 Hz on both arms, 20 s per run):

| Run | Loop wait for the bus lock | Loop period | Health read holding the bus |
| --- | --- | --- | --- |
| Health held | p50 0.008 ms, p99 0.044 ms, max 0.19 ms | p50 33.3 ms | none |
| Health polling (mock read) | p50 0.008 ms, p99 0.041 ms, max 0.68 ms | p50 33.2 ms | p50 0.50 ms, max 2.7 ms |
| Health polling, each read held 15 ms longer | p50 0.007 ms, p99 18.0 ms, max 22.5 ms | p50 33.4 ms, p99 52 ms | p50 21.6 ms, max 24.4 ms |

With the mock read, polling adds about 0.5 ms to the worst loop iteration and
nothing measurable to the rest. The third run models a real bus: one loop
iteration per poll waits up to one health read, and the deadline pacing
catches up, so the frame rate holds. A real STS3215 health read is five sync
reads of six servos; its duration on the G15 has not been measured. Run the
script with `--read-ms` to model other costs.

**Verdicts.** Thresholds and their settings are in [Thresholds](#thresholds).

- **CRITICAL** (temperature, voltage, load or current at a critical threshold,
  a Status register set to critical, or
  `SO101_HEALTH_MAX_CONSECUTIVE_FAILURES` failed reads in a row) calls
  `servo_health_critical`: a Collapse to the rest pose and a latched fault
  ("servo health critical: <arm>: <reasons>"), with the event
  `safety.servo_health_critical`. A critical leader stops the session too. A
  repeat critical reading while that Collapse runs does not dispatch again.
  Only the deliberate fault reset clears the fault.
- **WARN** emits `robot.health.warn` (once per change to warn) and shows in the
  status summary. It never stops anything. One failed read is a warning.
- Back to ok after a warning or critical emits `robot.health.ok`.

**`GET /api/robot/health`** returns what the monitor knows; it does no bus I/O.
Abridged:

```json
{
  "active": true, "level": "ok", "stale": false, "poller_running": true,
  "poll_hz": 1.0, "io_wait_ms": 50.0, "server_time": "2026-10-06T20:12:38Z",
  "arms": {
    "follower": {
      "active": true, "inactive_reason": null, "level": "ok", "source": "mock",
      "polled_at": "...", "sampled_at": "...", "sample_age_s": 0.02,
      "consecutive_failures": 0, "error": null, "reasons": [],
      "motors": {
        "elbow_flex": {
          "level": "ok", "reasons": [],
          "sample": {"temperature_c": 28.0, "voltage_v": 4.9, "load_pct": 10.0,
                     "current_ma": 117.0, "status_raw": 0, "raw": {"...": 0},
                     "limits": {"max_temperature_c": 70.0, "...": 0}}
        }
      },
      "thresholds": {"supply_profile": "7.4v", "voltage_band": [4.0, 4.5, 8.0, 8.5], "...": 0},
      "polls": 1, "read_failures": 0, "skipped_busy": 0, "last_skip": null,
      "last_read_ms": 0.118, "max_read_ms": 0.118, "activations": 1,
      "critical_dispatches": 0
    },
    "leader": {"active": false, "inactive_reason": "the leader bus is closed; teleop and recording open it", "...": 0}
  },
  "unit_provenance": {"Present_Voltage": {"status": "unverified", "...": 0}}
}
```

`level`, `reasons`, `error` and `consecutive_failures` are the last poll's
verdict, kept after polling stops (see `active`); `motors` holds the last full
sample of every servo with its own verdict. The schema is
`ServoHealthResponse` in `services/schemas.py`. The HIL runner stores this
snapshot in every probe bundle.

**Status and telemetry.** `GET /api/robot/status` and every telemetry frame
carry a compact `health` summary (`HealthSummary`):

| Field | Meaning |
| --- | --- |
| `level` | the worst verdict over the arms being polled; null before their first poll or while none is |
| `active` | at least one arm is polled |
| `arms` | `{"follower": level or null, "leader": level or null}`; null while that arm is not polled |
| `age_s` | the age of the oldest last full sample among the polled arms |
| `stale` | that sample is older than three poll periods (reads keep failing or keep being skipped) |
| `reasons` | up to three warn or critical messages |

## HIL runner

The runner walks the operator through the real-arm checks, one step at a time.
Each step shows its source document and quote, instructions and expected
results. It then optionally runs a probe, which records what the server
observes, and collects the operator's outcome and evidence values.

### Transport

| Situation | Where the run happens |
| --- | --- |
| A Suite server answers on `SO101_PORT` | Over HTTP, against that server. |
| No server, mock or sim backend | In the CLI process, as a labeled rehearsal. |
| No server, real backend | Refused: "Start the server with `so101 serve` in another terminal, then re-run this command." |
| Another program answers on `SO101_PORT` | Refused: "Stop the program on that port, or set SO101_PORT to the Suite server's port." |
| `list`, `status` or `report` with no server | Read from the saved runs. |

### Commands

```text
so101 hil run --operator NAME [--group gate|stop|verification]... [--step ID]... [--note TEXT] [--resume RUN_ID]
so101 hil list [--steps] [--json]
so101 hil status [RUN_ID] [--json]
so101 hil report RUN_ID [--json] [--markdown]
```

`so101 hil run`:

- For each step it offers the probe ("Run the probe now?") and lets you repeat
  it.
- It then asks for the outcome: `pass`, `fail`, `skip` or `quit`. It asks for
  each evidence value, offering the probe's measurement as the default.
- A `fail` or `skip` needs a note. A rejected attestation records nothing and
  asks again.
- `quit` asks whether to seal the run. Answering no leaves the run open and
  prints the `--resume` command.
- Prompts go to stderr.

| Command | Exit codes |
| --- | --- |
| `hil run` | 0 passed, 1 a step failed, 2 bad arguments or the run is already finalized, 3 incomplete (stopped early, a step skipped) or refused |
| `hil status` | 0 shown, 2 no such run or no runs |
| `hil report` | 0 the manifest verifies, 1 it does not (a file changed, went missing or was added), 2 not finalized or no such run |

### API

| Method and path | Purpose |
| --- | --- |
| `GET /api/hil/steps` | The step catalog. |
| `GET /api/hil/runs` | Run summaries, newest first. |
| `POST /api/hil/runs` | Start a run. Body: `{operator, note?, step_ids?, groups?}`. Returns 201. |
| `GET /api/hil/runs/{run_id}` | The run. |
| `POST /api/hil/runs/{run_id}/steps/{step_id}/probe` | Run the step's probe. Optional body `{duration_s}`, at most 180. |
| `POST /api/hil/runs/{run_id}/steps/{step_id}/attest` | Body: `{operator, outcome, note, evidence}`. |
| `POST /api/hil/runs/{run_id}/finalize` | Seal the run and write the report and manifest. |
| `GET /api/hil/runs/{run_id}/report` | The run, its Markdown and the manifest check. |

Errors: 404 for an unknown run or step. 409 for a finalized run, a probe
already running on the run, or a report before finalize. 422 with
`{"message", "problems"}` for a rejected attestation or request.

### Steps

Twenty-two steps run in three safety-ordered groups. `gate` comes from the
SUITE_V0 real-arm gate (`SUITE_V0.md:81-97`); `stop` comes from section 5 of
the v0.2 design (`docs/plans/2026-10-01-suite-v02-design.md` at commit
`a62fda0`; that section is not on this branch); and `verification` comes from
the SUITE_V01 exact-pair HIL checks (`SUITE_V01.md:34-37`).

| # | Step id | Title | Probe |
| --- | --- | --- | --- |
| 1 | `gate1-workspace` | Workspace and physical power control | none |
| 2 | `gate2-bus` | Supplies, bus adapters, motor IDs, baud rate, firmware, device paths | preflight: serial_ports, motors, firmware, supply_voltage |
| 3 | `gate3-install` | Robot extras, SO101_MOCK=0, fail closed | preflight: environment, lerobot |
| 4 | `gate4-diagnostics` | Diagnostics: no failed blocking check | diagnostics |
| 5 | `gate5-calibration` | Calibration convention and identity | preflight: calibration, rest_pose |
| 6 | `gate6-first-jog` | First motion: low speed, one joint, small range | telemetry, 20 s |
| 7 | `gate7-stop-tests` | Browser disconnect, tracking loss, lease expiry, soft stop, physical stop | events |
| 8 | `stop-estop` | E-stop: torque off on every motor | estop |
| 9 | `stop-halt` | Halt: hold in place, keep the object | halt_hold, 12 s |
| 10 | `stop-collapse` | Collapse: capped descent to the rest pose, then torque off | collapse |
| 11 | `stop-hold-timer` | Hold-timer expiry triggers Collapse | hold_timer |
| 12 | `stop-supervision-loss` | Supervision loss triggers Collapse after the grace period | supervision_loss, 30 s |
| 13 | `stop-host-kill` | Known limit: the host dies while the arm is energized | none |
| 14 | `v01-calibration-signs` | Calibration and joint signs | telemetry, 30 s |
| 15 | `v01-torque-transitions` | Torque transitions and stale-goal hold | status |
| 16 | `v01-deadman-release` | Manual and hand deadman release | events |
| 17 | `v01-browser-loss` | Browser hide and loss | events |
| 18 | `v01-estop-power` | E-stop and physical power isolation | estop |
| 19 | `v01-camera-faults` | Camera faults | status |
| 20 | `v01-collection-resume` | Collection and resume (at least 2 episodes saved) | events |
| 21 | `v01-training` | Training | none |
| 22 | `v01-rollout-cancel` | Rollout cancellation | events |

`so101 hil list --steps` prints the catalog; `GET /api/hil/steps` returns each
step's instructions, expected results and evidence fields.

### Probes

Probes only read: robot status, telemetry, the audit log, preflight and
diagnostics. They never command motion. A probe blocks for at most 180 s, and
only one probe runs at a time per run. Each probe saves a bundle with:

- the robot status and control state
- telemetry
- the servo health snapshot (`GET /api/robot/health`)
- the last 20 audit events
- an optional camera frame

| Probe | Pass | Warn or skip | Fail |
| --- | --- | --- | --- |
| `status` | not E-stopped and no fault | warn when E-stopped or faulted | never |
| `preflight` | every graded check passes | skip when a graded check skipped as blocking; warn when one warned | a graded check failed |
| `diagnostics` | every check passes | warn when a check warned | a check failed |
| `telemetry` | a joint moved at least 1° in the window | skip when telemetry did not advance; warn when nothing moved | never |
| `events` | an audit event with one of the step's prefixes since the run started | warn when none | never |
| `estop` | latched with `torque_state` "off" | skip when the incompatible server has no `torque_state` | not latched, or `torque_state` is not "off" ("unconfirmed": cut follower power) |
| `halt_hold` | halted for at least 10 s with every arm joint within 1° | skip without the stop fields; warn (and block a pass) when drift could not be measured | not halted at the start, the halt ended early, or drift over 1° |
| `collapse` | meets every Collapse rule below | skip without the stop fields; warn (and block a pass) when speed or final error could not be measured | no Collapse seen in the probe window (press Collapse within 15 s of starting the probe), or any rule broken |
| `hold_timer` | Collapse starts between 1 s early and 3 s late of `hold_expires_at` | skip without the stop fields; warn (and block a pass) when the expiry time is not reported | not halted at the start, the halt cleared, no Collapse, or outside the window |
| `supervision_loss` | Collapse starts within the grace period plus 3 s | skip without the stop fields; warn (and block a pass) when the loss time is not reported | no Collapse, or late |

### Stop steps

The stop probes read `stop_state`, `torque_state`, `halt` and `collapse` (and
`supervision.lost_at` for supervision loss) from `GET /api/robot/status`. A
server without them is not this build; the probes skip, and the operator
records the incompatible server rather than treating observation as equivalent
evidence. A skipped or warned stop probe cannot be attested `pass`, cannot
produce a `passed` run and makes `hil_evidence` false. A warning means a
required machine measurement was absent or ambiguous; an observation or
manually entered number cannot replace it. This includes `stop-estop`: it
requires a reported `torque_state`, not merely an observed limp arm.
Required fields are marked present only after a successful status read that
actually contained them. A probe that crashes before that read is a failed
probe with unproven fields and therefore cannot contribute HIL evidence.

The expectations come from design section 5 at `a62fda0`:

- **E-stop** (category 0). Torque goes off on every motor of every bus and the
  arm drops. `torque_state` must read "off".
- **Halt** (category 2). The motors stay on and hold position, and the gripper
  never opens further, so a held object is not dropped. Pass: the halt holds at
  least 10 s and no arm joint drifts more than 1°.
- **Collapse** (category 1). The arm moves to the rest pose at no more than
  `SO101_COLLAPSE_SPEED_DEG_S` (default 15 °/s), then torque goes off. Pass:
  - every joint's speed, measured over 0.25 s windows, stays at or below the
    cap plus 10 %
  - the descent finishes within its timeout plus 0.2 s. The timeout is
    distance / speed + 2 s, at most 12 s.
  - every joint ends within 3° of the rest pose
  - `torque_state` is "off" at the end
  - no `safety.collapse.fallback` event
- **Hold timer.** A halt lasts at most `SO101_HALT_HOLD_SECONDS` (default 60),
  then becomes a Collapse. Pass: the Collapse starts between 1 s before and 3 s
  after `hold_expires_at`.
- **Supervision loss.** While supervision is lost, a halt lasts only
  `SO101_SUPERVISION_GRACE_SECONDS` (default 5). Pass: the Collapse starts
  within the grace period plus 3 s.
- **Host death** (`stop-host-kill`) is a known limit, below.

### Attestation

- Every attestation names the operator. Evidence values are checked against
  the step's fields: unknown fields are rejected, and so are type errors and
  text over 2000 characters.
- A `pass` needs every required field, and the values must meet the field's
  pass rule, for example `max_drift_deg` at most 1 or `episodes_saved` at
  least 2.
- A `fail` or `skip` needs a note.
- A `pass` is refused when the step's latest probe failed. A measured stop step
  also requires a probe, and its latest probe may not be `skip`. Restore the
  required status fields and probe again, or attest `fail`/`skip`.
- Attesting a step again replaces the current attestation until the run is
  finalized, while retaining the earlier attestation in `attestation_history`.

### Evidence

A run lives in `<data_dir>/hil/<run_id>/`. The run id looks like
`20261002T143000Z-1a2b3c`, so it is a legal file name on every OS.

```text
run.json                                        the run (source of truth)
steps/<step_id>.json                            step definition and record
artifacts/<step_id>/probe-<n>-<kind>.json       probe bundle
artifacts/<step_id>/probe-<n>-samples.jsonl     10 Hz samples (sampled probes)
artifacts/<step_id>/probe-<n>-preflight-*.json  full preflight reports
artifacts/<step_id>/probe-<n>-frame-<cam>.npy   optional camera frame
report.json, report.md                          written by finalize
MANIFEST.sha256                                 written last by finalize
```

Finalizing seals the run. Nothing can change it afterwards.

`run.json`, every probe and every attestation carry the version and SHA-256
digest of the HIL catalog/evidence contract that created them. An open run from
an older build keeps its run id and probe history, but finalization is refused
until every existing attestation is deliberately repeated under the current
contract and every attested measured-stop step has a current probe with the
required status fields. Re-attestation retains the prior attestation in the
step record, and re-probing never erases probe history. Any skipped or warned
stop probe keeps stop evidence incomplete, so start a fresh stop run for
release evidence.

- **Status.** `failed` if any step failed. `passed` if every step passed.
  Otherwise `incomplete`.
- **`hil_evidence`** is true only when all of these hold:
  - the run is on the real backend
  - every probe ran on the real backend
  - at least one step was attested `pass` or `fail`
  - no selected stop step has a skipped or warned probe (and no stop pass lacks a probe)
- **`evidence_notes`** explain a false `hil_evidence`. They also flag what a
  reviewer should read even when it is true: a non-stop pass attested without
  running its probe, a non-stop pass attested on observation because the probe
  could not measure, and steps left unattested.

**Check a run.** `so101 hil report RUN_ID` re-hashes every file against
`MANIFEST.sha256` and reports changed, missing and added files. The manifest
uses the standard `sha256sum` format, so on Linux and macOS you can check it
without the Suite:

```text
cd <data_dir>/hil/<run_id>
sha256sum -c MANIFEST.sha256        # Linux (coreutils)
shasum -a 256 -c MANIFEST.sha256    # macOS
```

On Windows, use `so101 hil report RUN_ID`.

## Known limit: a dead host leaves the arm energized

If the host process dies (killed, crashed, or the computer loses power),
Feetech STS3215 servos keep holding their last goal with torque on. No software
stop covers that case; only the physical power switch does (design section 5,
"Known limit"). Keep the follower's power switch within reach whenever the arm
is energized. The `stop-host-kill` step records where the switch is and that the
operator knows this.

## Still open

Bus lending, the health monitor in the server, the stop fields, the
model-derived rest pose and device-global exclusion in the server and every
bus-opening command have landed. Still open:

- **Simulated health.** Under `SO101_BACKEND=sim` the monitor is off; the
  simulated servos report no health values (design section 2 asks for load
  derived from actuator force).
- **Real read cost.** How long one health read holds the real bus, and so the
  loop impact on the G15, is unmeasured (see
  [Servo health monitor](#servo-health-monitor)).

Other limits:

- The `events` probe sees only the newest 500 audit events.
- Telemetry does not advance while the robot is idle, so the `telemetry`
  probe skips unless a mode is running.
- The firmware check verifies only that all motors on an arm match. No
  minimum firmware version is verified.

## Sources

Hardware claims in this guide and in the kit cite these, checked for v0.2:

- LeRobot 0.6.1 source distribution (`lerobot-0.6.1.tar.gz`, sha256
  `2111d67d72d9f6c2379e3b8d3ea41b58855066314d88f9c0893579b7707f04b5`). The
  kit pins per-file hashes in `src/so101/bringup/lerobot_surface.py`.
  `tests/test_bringup_lerobot061_pin.py` always compares the kit's register
  table with its own copy. It re-checks the installed LeRobot when one is
  importable. To check the pin against the archive itself (hashes, tables,
  signatures and cited line numbers, read without running any of it), set
  `SO101_TEST_LEROBOT_SDIST` to the archive's path and run
  `pytest tests/test_bringup_lerobot061_pin.py`.
- pyserial 3.5 source (`serial/tools/list_ports*.py`).
- SO-ARM100 README, commit `5f6d2b876a53a4872e405b991dd925556c9e38a4`, line 73
  (servo variants and supply).
- `docs/plans/2026-10-01-suite-v02-design.md` section 5 at commit `a62fda0`
  (stop categories).

Anything else, notably the register unit scales, the Status bits and the
firmware minimum, is marked unverified where it appears.

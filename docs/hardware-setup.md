# Hardware setup

The SO-101 is a leader/follower pair with five arm joints plus a gripper (six
actuators per arm), driven by Feetech **STS3215** bus servos over USB adapters.

Run the commands below from the repo root after `make install-robot`, with the
venv active (`source .venv/bin/activate` puts `so101` on PATH).

## Bill of materials (per the SO-101 project)

- 2× SO-101 arm kits (leader + follower), 6 STS3215 servos each
- 2× Feetech USB bus-servo adapters (one per arm)
- 1–2× USB cameras (a wrist cam greatly helps manipulation)
- Correct supply for each build (the leader is 7.4V; confirm a follower's motor
  voltage before connecting power); **a reachable power switch**

## Before any arm power

Keep both motor supplies switched off while you establish the physical safety
envelope. Clamp or weigh down both bases, clear people, objects and cables from
the arm's reach, remove the payload and place a padded mechanical catch under
the follower, positioned to arrest a drop without obstructing the recorded
rest pose or Collapse path. Never plan to catch a moving or falling arm by
hand. Put the follower power switch where it can be reached without crossing
the arm.

Confirm the follower and leader supply labels and servo variants, and trace
each USB cable to the correct arm before assigning its stable alias. An
ambiguous arm or supply is a no-go. For first power, calibration, rest-pose
recording and stop tests, use two people where possible: one operates the
Suite and one remains at the physical cutoff. Calibration and rest-pose
recording turn torque off, so keep the follower low over the mechanical catch
and hands outside its envelope.

## 1. Find serial ports

On Linux (the G15), give the installer both adapters' USB serial numbers
(`python -m so101.platform_info` lists them). Its udev rules name the adapters
`/dev/so101-follower` and `/dev/so101-leader`, stable across reboots and
re-plugs ([install.md](install.md)):

```bash
install/install.sh --follower-serial <FOLLOWER_SERIAL> --leader-serial <LEADER_SERIAL>
ls -lL /dev/so101-follower /dev/so101-leader
```

Without the udev names, list the kernel names (they can renumber):

```bash
ls /dev/ttyUSB* /dev/ttyACM*   # Linux
ls /dev/tty.usbmodem*          # macOS
```

Set them in `.env`:

```
SO101_MOCK=0
SO101_FOLLOWER_PORT=/dev/so101-follower
SO101_LEADER_PORT=/dev/so101-leader
SO101_FOLLOWER_ID=so101-follower
SO101_LEADER_ID=so101-leader
SO101_BAUDRATE=1000000
```

On macOS the ports look like `/dev/tty.usbmodemXXXX`; on Windows like `COM3`.

## 2. Assign motor IDs

Each servo ships as id=1. Assign ids **1..6** (shoulder pan, shoulder lift,
elbow, wrist flex, wrist roll, gripper), one motor connected at a time:

```bash
make setup-motors        # prints the motor order + both LeRobot commands (prints only)

# Official tools (run both with the correct ports/ids):
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=so101-follower
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/so101-leader --teleop.id=so101-leader
```

## 3. Cameras

```bash
make find-cameras        # Intel RealSense devices, then OpenCV indices 0..9
# then e.g.
SO101_CAMERAS=0,1                                  # two OpenCV cameras
SO101_CAMERAS=0,wrist=realsense:123622270882+depth # a webcam and a named RealSense with depth
```

An Intel RealSense needs the `realsense` extra and, on Linux, udev rules; see
[realsense.md](realsense.md) for setup, the entry grammar and depth units.

## 4. Calibrate

```bash
so101 calibrate follower
so101 calibrate leader
```

Place the arm at its midpoint before starting, then sweep each joint through
its usable range during the timed UI/CLI calibration. The current adapter
writes homing/range data to the motors and LeRobot's per-device calibration
file; the suite also saves normalized measured ranges and intersects them with
its conservative `SafetyGuard` limits. Wrist roll keeps the official full
encoder range. A successful follower calibration refreshes the live guard.

On the real backend, the CLI requires an external confirmation before it takes
the canonical device lock or opens the bus. Put the arm low over the padded
mechanical catch, keep a cutoff operator ready, and confirm a clear sweep.
Calibration requires intentional hands-on movement: move only the links, keep fingers away from
pinch points and keep the rest of your body outside the sweep envelope. Do not
use `--yes` on first bring-up.

The sweep starts and ends with the suite's torque-off pass (broadcast, then a
retry per motor, then read-back). If a servo cannot be confirmed off at the
start, calibration stops before writing anything. Run `so101 calibrate` with
the server stopped: it refuses while a server answers, while another command
holds the same device lock, or while an E-stop or fault from the server is
still latched (exit 3, with the reset to use). From the UI
(`POST /api/calibration/start`) the sweep runs through the server's
`RobotManager`, under its bus lock; if torque-off cannot be
confirmed at the end, the job fails with "cut physical power to the <arm>
arm" and the server latches a fault (see step 7).

## 5. First teleop

```bash
so101 serve
```

Open the Suite UI and use **Teleop + Record** for a short supervised teleop.
The real-backend `so101 teleop` command is intentionally refused because a
standalone process cannot provide the server's watchdog, software stops,
supervision grace and timed Collapse. The follower should mirror the leader.
If a joint moves the wrong way or hits a limit, re-check ids/calibration before
recording data.

For the first real motion, use a clear workspace and a small single-joint jog.
Verify direction and measured position, then test lease expiry, browser loss,
soft stop, software E-stop, and physical power isolation.

## 6. Continue running the suite

```bash
make suite               # build the UI and serve it with the API at http://localhost:8000
```

`make api` (with `make ui` in a second terminal) and `so101 serve` work too. Do
not use `make api-dev` or `make dev` with an arm connected: they restart the
server on every source change, and each restart Collapses the arm (rest pose,
then motors off) and ends whatever was moving it. The E-stop latch and any fault
survive the restart. They refuse to start unless `SO101_MOCK=1`.

## 7. While the server runs

- **Preflight.** `so101 preflight` and `POST /api/preflight` run the bus checks
  through the server: its `RobotManager` lends each bus only while the robot is
  idle (not armed, no Halt or Collapse, no loop, no calibration, no E-stop),
  read-only, under its bus lock. Otherwise the bus checks skip as `refused`
  ([bring-up.md](bring-up.md#who-may-open-the-bus)).
- **Servo health.** While the server has an arm's bus open (armed, a loop, a
  Halt, a Collapse), it polls that arm's servo health at
  `SO101_HEALTH_POLL_HZ` (1 Hz). A critical reading (temperature, supply
  voltage, consecutive read failures, ...) Collapses the arm to the rest pose
  and latches a fault; a warning shows in the status and the event log.
  `GET /api/robot/health` shows the last sample per servo, the verdicts and the
  thresholds ([bring-up.md](bring-up.md#servo-health-monitor)).
- **Resets.** Both resets need a JSON body naming the operator, after you have
  checked the arm:

  ```bash
  curl -X POST http://127.0.0.1:8000/api/robot/estop/reset \
    -H 'Content-Type: application/json' -d '{"operator_id": "Your Name", "confirm": true}'
  curl -X POST http://127.0.0.1:8000/api/robot/fault/reset \
    -H 'Content-Type: application/json' -d '{"operator_id": "Your Name", "confirm": true}'
  ```

  The fault reset retries an unconfirmed torque-off first. If it still cannot
  be confirmed it answers 409; cut the arm's power, then add
  `"power_cut_acknowledged": true` to the body
  ([control-plane-safety.md](control-plane-safety.md#deliberate-resets-c2)).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Real driver unavailable | install the pinned `[robot]` extra; real mode fails closed and never substitutes `MockBus` |
| Port not found | wrong `SO101_*_PORT`; check `ls -lL /dev/so101-follower /dev/so101-leader` (Linux udev names) or `ls /dev/tty*` |
| Arm collapsed with "servo health critical" | read `GET /api/robot/health` for the servo and reason (hot, low supply, silent servo), fix it, then reset the fault |
| Jittery / wrong joint | motor id mismatch — re-run `lerobot-setup-motors` for that arm (`make setup-motors` prints the command) |
| No video in UI | wrong `SO101_CAMERAS`; run `make find-cameras` (RealSense: [realsense.md](realsense.md#troubleshooting)) |
| Command rejected as uncalibrated | calibrate the exact device ID before real motion |
| Arm jumps or moves with wrong sign | stop immediately; verify assembly, IDs, midpoint calibration and device identity |

See [troubleshooting.md](troubleshooting.md) for control rejection, camera,
training, and support-snapshot runbooks.

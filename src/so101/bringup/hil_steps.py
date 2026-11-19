"""The HIL step catalog.

Every step quotes its source verbatim and cites it by file and line:

* ``gate*``: the "Real-arm gate" in docs/SUITE_V0.md (lines 77-97).
* ``v01-*``: the exact-pair HIL checks listed in docs/SUITE_V01.md (lines 34-37).
* ``stop-*``: the stop categories in §5 of
  docs/plans/2026-10-01-suite-v02-design.md at commit a62fda0.

Instructions and expected results are the bring-up kit's wording. Probes only
observe (status, telemetry, the audit log, preflight, diagnostics); they never
command motion. The operator triggers every action and attests the outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .. import JOINT_NAMES
from .schemas import EvidenceField, HilGroup, HilStepDefinition

SUITE_V0 = "docs/SUITE_V0.md"
SUITE_V01 = "docs/SUITE_V01.md"
DESIGN = "docs/plans/2026-10-01-suite-v02-design.md"
DESIGN_COMMIT = "a62fda0"

#: /api/robot/status fields the stop probes read. A server without them is an
#: incompatible older build, so those probes skip rather than invent evidence.
STOP_STATUS_FIELDS = ("stop_state", "torque_state", "halt", "collapse")

#: Pass bounds that are constants of the design rather than settings.
HALT_MAX_DRIFT_DEG = 1.0
HALT_MIN_HOLD_S = 10.0
COLLAPSE_SETTLE_DEG = 3.0


def _flag(name: str, label: str, *, expect: bool | None = True, help: str = "") -> EvidenceField:
    return EvidenceField(name=name, label=label, kind="bool", pass_equals=expect, help=help)


def _number(
    name: str,
    label: str,
    unit: str,
    *,
    pass_min: float | None = None,
    pass_max: float | None = None,
    required: bool = True,
    help: str = "",
) -> EvidenceField:
    return EvidenceField(
        name=name,
        label=label,
        kind="number",
        unit=unit,
        pass_min=pass_min,
        pass_max=pass_max,
        required=required,
        help=help,
    )


def _text(name: str, label: str, *, required: bool = True, help: str = "") -> EvidenceField:
    return EvidenceField(name=name, label=label, kind="text", required=required, help=help)


def _choice(
    name: str, label: str, choices: Iterable[str], *, expect: str | None = None, help: str = ""
) -> EvidenceField:
    return EvidenceField(
        name=name, label=label, kind="choice", choices=list(choices), pass_equals=expect, help=help
    )


def _requires(*fields: str) -> list[str]:
    return [f"status.{field}" for field in fields]


_GATE_SOURCE = f"{SUITE_V0} (Real-arm gate, item {{item}}), lines {{lines}}"
_V01_SOURCE = f"{SUITE_V01}:{{lines}} (exact-pair HIL checks: {{item}})"
_DESIGN_SOURCE = f"{DESIGN} §5 at {DESIGN_COMMIT}, lines {{lines}}"

STOP_PROBE_NOTE = (
    "Reads stop_state, torque_state, halt and collapse from /api/robot/status. A server "
    "without those fields is not this build; the probe skips and the operator records why."
)

_HIL_STEPS: tuple[HilStepDefinition, ...] = (
    # --- Real-arm gate (SUITE_V0.md) ---------------------------------------
    HilStepDefinition(
        id="gate1-workspace",
        group="gate",
        order=1,
        title="Workspace and physical power control",
        source=_GATE_SOURCE.format(item=1, lines="81-82"),
        quote=(
            "Put the arm on a stable surface with an unobstructed workspace and keep the "
            "physical power control reachable."
        ),
        instructions=[
            "Before the first arm power-up, clamp or weigh down the follower base on a "
            "stable table.",
            "Clear everything (objects, cables, people) out of the arm's reach.",
            "Place a padded mechanical catch under the follower, positioned to arrest a drop "
            "without obstructing the recorded rest pose or Collapse path. Do not plan to "
            "catch a falling or moving arm by hand.",
            "Put the follower's power switch or plug where the operator can reach it without "
            "reaching across the arm, and write down where it is.",
            "A second person is recommended for first power and motion: one operates the "
            "Suite and one stays at the physical power control.",
        ],
        expected_results=[
            "The base does not shift when the arm is pushed gently by hand.",
            "Nothing is inside the arm's reach.",
            "The operator can cut follower power from the operator position.",
        ],
        evidence_fields=[
            _flag("stable_surface", "Base is stable and secured"),
            _flag("workspace_clear", "Workspace is clear"),
            _text("power_control_location", "Where the follower power control is"),
            _flag("power_control_reachable", "Power control reachable from the operator position"),
        ],
    ),
    HilStepDefinition(
        id="gate2-bus",
        group="gate",
        order=2,
        title="Supplies, bus adapters, motor IDs, baud rate, firmware, device paths",
        source=_GATE_SOURCE.format(item=2, lines="83-84"),
        quote=(
            "Confirm the follower and leader supplies, Waveshare bus adapters, motor IDs "
            "1–6, baud rate, firmware compatibility and stable device paths."
        ),
        instructions=[
            "Keep the server running and the robot idle, then start this probe. Under its bus "
            "lock, the manager lends each bus to preflight through a read-only view; the "
            "probe never opens a competing motor-bus owner.",
            "Re-confirm the supply identities that were checked before first power: compare "
            "each supply's label with SO101_HEALTH_FOLLOWER_SUPPLY and "
            "SO101_HEALTH_LEADER_SUPPLY. Do not continue if a supply or arm is ambiguous.",
            "Use stable port names: on Windows the COMx name Device Manager shows under "
            "Ports (COM & LPT), on Linux the /dev/serial/by-id/... link, on macOS "
            "/dev/tty.usbmodem... or /dev/cu.usbmodem.... The serial_ports check shows each "
            "adapter's USB serial number and VID:PID; note them with the ports.",
        ],
        expected_results=[
            "serial_ports, motors, firmware and supply_voltage pass for both arms in a "
            "real-backend preflight.",
            "IDs 1-6 answer on each bus at 1,000,000 baud with model number 777 (STS3215).",
        ],
        evidence_fields=[
            _number("follower_supply_v", "Follower supply voltage (measured or label)", "V"),
            _number("leader_supply_v", "Leader supply voltage (measured or label)", "V"),
            _text("follower_port", "Follower port (COMx or device path)"),
            _text("leader_port", "Leader port (COMx or device path)"),
            _flag("ids_1_to_6_answer", "IDs 1-6 answer on both buses"),
            _flag("firmware_matches", "Firmware versions match within each bus"),
        ],
        probe="preflight",
        probe_checks=["serial_ports", "motors", "firmware", "supply_voltage"],
        probe_help="Grades serial_ports, motors, firmware and supply_voltage from preflight.",
    ),
    HilStepDefinition(
        id="gate3-install",
        group="gate",
        order=3,
        title="Robot extras, SO101_MOCK=0, fail closed",
        source=_GATE_SOURCE.format(item=3, lines="85-87"),
        quote=(
            "Install the robot extras in a dedicated environment and explicitly set "
            "`SO101_MOCK=0`. A requested hardware session fails closed when its adapter "
            "cannot load or connect."
        ),
        instructions=[
            "Install the robot extras into a dedicated virtual environment on the robot "
            'computer: `pip install -e ".[robot]"`, or `make install-robot` where make is '
            "available (docs/bring-up.md, Install).",
            "Start the server with SO101_MOCK=0.",
            "Fail-closed check: with the follower over its mechanical catch, unplug its USB "
            "cable and request teleop through the Suite UI or the server's teleop route. "
            "The start request is asynchronous: HTTP 200/accepted is not a pass. Watch the "
            "operation and audit events until a terminal fault names the adapter, and confirm "
            "there was no motion.",
            "Do not substitute `so101 teleop`: real standalone motion is disabled, and its "
            "refusal does not prove adapter failure. Use the server request above.",
            "Reconnect the USB cable, inspect the physical arm and torque status, then "
            "deliberately reset any resulting fault only after torque-off is confirmed. Run "
            "preflight and confirm the server is idle with no latch before continuing.",
        ],
        expected_results=[
            "environment and lerobot pass on the real backend: LeRobot 0.6.1 and the Feetech "
            "SDK import, and LEROBOT_HOME is unset.",
            "With the adapter unplugged, the hardware session fails with an error; nothing "
            "falls back to the mock backend and nothing moves.",
            "After the adapter is restored, preflight passes and the server is idle with no "
            "E-stop, fault or unconfirmed-torque latch.",
        ],
        evidence_fields=[
            _text("lerobot_version", "Installed LeRobot version"),
            _flag("mock_disabled", "Server runs with SO101_MOCK=0"),
            _flag(
                "unplugged_session_refused",
                "Terminal adapter failure observed after the unplugged start",
            ),
            _flag("no_motion_on_failed_start", "No motion during the failed start"),
            _flag("post_test_recovery_idle", "Server idle after deliberate recovery"),
            _flag("post_test_latches_clear", "Safety latches clear after deliberate recovery"),
        ],
        probe="preflight",
        probe_checks=["environment", "lerobot"],
        probe_help="Grades environment and lerobot from preflight.",
    ),
    HilStepDefinition(
        id="gate4-diagnostics",
        group="gate",
        order=4,
        title="Diagnostics: no failed blocking check",
        source=_GATE_SOURCE.format(item=4, lines="88"),
        quote="Run the diagnostics page. Resolve every failed blocking check.",
        instructions=[
            "Open the Diagnostics page (or GET /api/diagnostics).",
            "Fix each failed check and reload until none fail.",
        ],
        expected_results=["GET /api/diagnostics reports no failed check."],
        evidence_fields=[
            _number("failed_checks", "Failed diagnostics checks", "checks", pass_max=0),
        ],
        probe="diagnostics",
        probe_help="Runs the server's diagnostics and fails on any failed check.",
    ),
    HilStepDefinition(
        id="gate5-calibration",
        group="gate",
        order=5,
        title="Calibration convention and identity",
        source=_GATE_SOURCE.format(item=5, lines="89-90"),
        quote=(
            "Calibrate with the current LeRobot calibration convention and verify the "
            "calibration identity matches the attached device."
        ),
        instructions=[
            "Before starting this HIL run, stop the server and calibrate both arms with "
            "`so101 calibrate follower` and `so101 calibrate leader`.",
            "Before each calibration command, place that arm at its midpoint while torque is "
            "off and before the command opens its bus.",
            "Before starting this HIL run, record the Collapse target with "
            "`so101 rest-pose record`, then require `so101 preflight --require-hardware` "
            "to pass.",
            "Calibration and rest-pose recording turn torque off. Use a padded mechanical "
            "catch and have a second person at the power control. During the prompted "
            "hands-on calibration sweep, move only the links, keep fingers away from pinch "
            "points and keep the rest of your body outside the sweep envelope.",
            "If those prerequisites were not completed, attest fail and end this run; do not "
            "stop or restart the server during this step. Only catalog steps that explicitly "
            "test lifecycle recovery may restart it.",
            "During this HIL step leave the server running and the robot idle; start the probe "
            "to verify the saved calibration and rest pose through the manager.",
        ],
        expected_results=[
            "calibration passes for both arms: the LeRobot file exists for the configured id "
            "and its homing offsets and position limits match the motors.",
            "rest_pose passes: a recorded pose bound to this follower id and calibration.",
        ],
        evidence_fields=[
            _text("follower_id", "Follower id"),
            _text("leader_id", "Leader id"),
            _flag("identity_matches", "Calibration identity matches the attached arms"),
            _flag("rest_pose_recorded", "Rest pose recorded for this calibration"),
        ],
        probe="preflight",
        probe_checks=["calibration", "rest_pose"],
        probe_help="Grades calibration and rest_pose from preflight.",
    ),
    HilStepDefinition(
        id="gate6-first-jog",
        group="gate",
        order=6,
        title="First motion: low speed, one joint, small range",
        source=_GATE_SOURCE.format(item=6, lines="91-92"),
        quote=(
            "Start at low speed. Acquire a lease, arm, hold the deadman and jog one joint "
            "through a small range. Verify sign and measured position before continuing."
        ),
        instructions=[
            "Start the probe (it records telemetry for 20 s).",
            "Acquire a lease, arm, hold the deadman and jog one joint a few degrees each way "
            "at low speed.",
            "Release the deadman.",
        ],
        expected_results=[
            "The jogged joint moves in the commanded direction; no other joint moves.",
            "Its measured position changes by about the commanded amount.",
        ],
        evidence_fields=[
            _choice("joint", "Jogged joint", JOINT_NAMES),
            _number("commanded_delta_deg", "Commanded change", "deg"),
            _number("measured_delta_deg", "Measured change", "deg"),
            _flag("sign_correct", "Moved in the commanded direction"),
            _flag("other_joints_still", "No other joint moved"),
        ],
        probe="telemetry",
        probe_window_s=20.0,
        probe_help="Records joint positions at 10 Hz and reports each joint's range.",
    ),
    HilStepDefinition(
        id="gate7-stop-tests",
        group="gate",
        order=7,
        title="Browser disconnect, tracking loss, lease expiry, soft stop, physical stop",
        source=_GATE_SOURCE.format(item=7, lines="93-94"),
        quote=(
            "Test browser disconnect, tracking loss, lease expiry, soft stop and the physical "
            "stop before collecting data or running a policy."
        ),
        instructions=[
            "Browser disconnect: during slow teleop, close the browser tab.",
            "Tracking loss: during hand tracking, move the hand out of view.",
            "Lease expiry: arm, then stop sending commands until the lease expires.",
            "Soft stop: during teleop, press Stop.",
            "Physical stop: position the follower over its padded mechanical catch, keep "
            "hands clear, then have the power-control operator cut follower power.",
            "Keep power off until the active session has ended. With the server disarmed and "
            "the catch and cutoff ready, restore power and confirm the arm does not move.",
            "Inspect status and deliberately clear any resulting safety latches only after "
            "torque-off is confirmed. Run `so101 preflight --require-hardware` through the "
            "idle server.",
            "Before starting the stop group, require idle mode, no E-stop/fault/unconfirmed-"
            "torque latch and torque_state off.",
            "Then probe: it collects the audit events logged since the run started.",
        ],
        expected_results=[
            "Each software stop ends motion and is logged (runtime.operator_timeout, "
            "control.lease_expired, control.watchdog_timeout, runtime.soft_stop).",
            "Cutting power makes the arm go limp.",
            "Restoring power causes no motion; deliberate recovery leaves the server idle, "
            "unlatched and with torque off for the stop group.",
        ],
        evidence_fields=[
            _flag("browser_disconnect_stops", "Browser disconnect stops motion"),
            _flag("tracking_loss_stops", "Tracking loss stops motion"),
            _flag("lease_expiry_disarms", "Lease expiry disarms"),
            _flag("soft_stop_idles", "Soft stop returns to idle"),
            _flag("physical_stop_cuts_power", "Physical stop cuts power"),
            _flag("no_motion_on_power_restore", "No motion when follower power is restored"),
            _flag("recovery_idle", "Server is idle after deliberate recovery"),
            _flag("latches_clear", "E-stop, fault and unconfirmed-torque latches are clear"),
            _flag("torque_off_before_stop_group", "Torque is confirmed off before stop HIL"),
        ],
        probe="events",
        probe_events=[
            "runtime.operator_timeout",
            "control.lease_expired",
            "control.watchdog_timeout",
            "runtime.soft_stop",
            "tracking.",
            "safety.",
        ],
        probe_help="Collects the audit events logged since the run started.",
    ),
    # --- exact-pair HIL checks (SUITE_V01.md) -----------------------------------
    HilStepDefinition(
        id="v01-calibration-signs",
        group="verification",
        order=8,
        title="Calibration and joint signs",
        source=_V01_SOURCE.format(lines="35", item="calibration/signs"),
        quote="calibration/signs",
        instructions=[
            "Start teleop with the exact follower and leader pair.",
            "Start the probe (30 s) and move each leader joint in turn a little positive, "
            "then negative.",
        ],
        expected_results=[
            "Every follower joint follows its leader joint in the same direction.",
            "The follower gripper opens and closes with the leader's.",
        ],
        evidence_fields=[
            _flag("all_signs_match", "Every joint follows in the same direction"),
            _text("mismatched_joints", "Joints that moved the wrong way", required=False),
        ],
        probe="telemetry",
        probe_window_s=30.0,
        probe_help="Records joint positions at 10 Hz and reports each joint's range.",
    ),
    HilStepDefinition(
        id="v01-torque-transitions",
        group="verification",
        order=9,
        title="Torque transitions and stale-goal hold",
        source=_V01_SOURCE.format(lines="35", item="torque transitions and stale-goal hold"),
        quote="torque transitions and stale-goal hold",
        instructions=[
            "Arm with the arm at rest: it must not jump when torque comes on.",
            "Disarm and watch what the arm does (wave 1: torque off; wave 2: Halt holds).",
            "Stale goal: arm, send one jog, then send nothing until the watchdog disarms. "
            "The arm must never move toward an old goal.",
            "Probe to record the server state.",
        ],
        expected_results=[
            "No jump when torque turns on.",
            "After the watchdog fires the arm does not chase a stale goal.",
        ],
        evidence_fields=[
            _flag("no_jump_on_arm", "No jump when arming"),
            _flag("stale_goal_held", "No motion toward a stale goal"),
        ],
        probe="status",
        probe_help="Records the robot status, control state and torque_state if reported.",
    ),
    HilStepDefinition(
        id="v01-deadman-release",
        group="verification",
        order=10,
        title="Manual and hand deadman release",
        source=_V01_SOURCE.format(lines="35-36", item="manual and hand deadman release"),
        quote="manual and hand deadman release",
        instructions=[
            "Manual: jog with the deadman held and release it mid-move.",
            "Hand: during hand tracking, release the hand deadman mid-move.",
            "Then probe: it collects the audit events logged since the run started.",
        ],
        expected_results=[
            "Releasing either deadman stops motion at once (wave 2: the arm Halts in place).",
        ],
        evidence_fields=[
            _flag("manual_release_stops", "Manual deadman release stops motion"),
            _flag("hand_release_stops", "Hand deadman release stops motion"),
        ],
        probe="events",
        probe_events=["control.watchdog_timeout", "control.disarmed", "safety.halt", "tracking."],
        probe_help="Collects the audit events logged since the run started.",
    ),
    HilStepDefinition(
        id="v01-browser-loss",
        group="verification",
        order=11,
        title="Browser hide and loss",
        source=_V01_SOURCE.format(lines="36", item="browser hide/loss"),
        quote="browser hide/loss",
        instructions=[
            "During slow teleop, switch the operator browser to another tab (hide).",
            "Start teleop again and close the tab (loss).",
            "Then probe: it collects the audit events logged since the run started.",
        ],
        expected_results=[
            "Hiding or losing the operator browser stops motion; the operator heartbeat "
            "timeout is 2 s (RobotManager.operator_timeout_ms = 2000).",
        ],
        evidence_fields=[
            _flag("hide_stops", "Hiding the tab stops motion"),
            _flag("close_stops", "Closing the tab stops motion"),
            _number("seconds_to_stop", "Time to stop after closing", "s", required=False),
        ],
        probe="events",
        probe_events=["runtime.operator_timeout", "runtime.soft_stop", "safety."],
        probe_help="Collects the audit events logged since the run started.",
    ),
    HilStepDefinition(
        id="v01-estop-power",
        group="verification",
        order=12,
        title="E-stop and physical power isolation",
        source=_V01_SOURCE.format(lines="36", item="E-stop and physical power isolation"),
        quote="E-stop and physical power isolation",
        instructions=[
            "Position the follower low over its padded mechanical catch. During slow teleop "
            "press E-stop and probe while it is latched; keep hands clear.",
            "Reset the E-stop deliberately.",
            "With the follower over the mechanical catch again, have the power-control "
            "operator cut follower power while it holds a pose. Restore power and record "
            "what the Suite reports.",
        ],
        expected_results=[
            "E-stop stops motion and disables torque.",
            "Cutting power makes the arm limp; nothing moves when power returns until the "
            "operator arms again.",
        ],
        evidence_fields=[
            _flag("estop_stops_motion", "E-stop stops motion"),
            _flag("power_cut_goes_limp", "Power cut makes the arm limp"),
            _flag("no_motion_on_power_restore", "No motion when power returns"),
        ],
        probe="estop",
        probe_help="Checks the E-stop latch, and torque_state when the server reports it.",
    ),
    HilStepDefinition(
        id="v01-camera-faults",
        group="verification",
        order=13,
        title="Camera faults",
        source=_V01_SOURCE.format(lines="37", item="camera faults"),
        quote="camera faults",
        instructions=[
            "With cameras configured, unplug one camera while idle and probe.",
            "Start a recording, unplug a camera, and note what the Suite does.",
            "Plug it back in.",
        ],
        expected_results=[
            "A missing camera drops out of live_cameras and is reported to the operator.",
            "A recording does not silently continue without a configured camera.",
        ],
        evidence_fields=[
            _flag("unplug_reported", "Unplugged camera is reported"),
            _text("recording_behavior", "What the recording did"),
        ],
        probe="status",
        probe_help="Records configured vs live cameras and frame ages.",
    ),
    HilStepDefinition(
        id="v01-collection-resume",
        group="verification",
        order=14,
        title="Collection and resume",
        source=_V01_SOURCE.format(lines="37", item="collection/resume"),
        quote="collection/resume",
        instructions=[
            "Record two short episodes.",
            "This is the only HIL step that requires a server restart. Put the follower over "
            "its mechanical catch with the cutoff operator ready, start a third episode, then "
            "stop the server gracefully with Ctrl-C.",
            "Wait for the shutdown Collapse, confirmed torque-off and process exit. Restart "
            "the same build, backend and data folder; never switch to a different server to "
            "make the run continue.",
            "Leave the HIL CLI waiting while the server restarts. If its request fails, rerun "
            "the printed `so101 hil run --operator ... --resume RUN_ID` command; do not start "
            "or seal a replacement run.",
            "Confirm the restarted server is idle and any latch is deliberately resolved, "
            "resume recording into the same dataset, then probe.",
        ],
        expected_results=[
            "Saved episodes are intact; the interrupted episode is preserved or discarded "
            "with an audit event, never half-written.",
        ],
        evidence_fields=[
            _number("episodes_saved", "Episodes saved", "episodes", pass_min=2),
            _flag("resume_ok", "Recording resumed into the same dataset"),
            _text("dataset", "Dataset name"),
        ],
        probe="events",
        probe_events=["recording."],
        probe_help="Collects the audit events logged since the run started.",
    ),
    HilStepDefinition(
        id="v01-training",
        group="verification",
        order=15,
        title="Training",
        source=_V01_SOURCE.format(lines="37", item="training"),
        quote="training",
        instructions=["Train a short run on the collected dataset (Training page)."],
        expected_results=["The job finishes, or is cancelled cleanly, and writes a checkpoint."],
        evidence_fields=[
            _text("job_id", "Training job id"),
            _flag("checkpoint_written", "Checkpoint written"),
        ],
    ),
    HilStepDefinition(
        id="v01-rollout-cancel",
        group="verification",
        order=16,
        title="Rollout cancellation",
        source=_V01_SOURCE.format(lines="37", item="rollout cancellation"),
        quote="rollout cancellation",
        instructions=[
            "Start a policy rollout at low speed with the workspace clear.",
            "Cancel it mid-motion, then probe.",
        ],
        expected_results=["Cancelling stops motion promptly (wave 2: the arm Halts)."],
        evidence_fields=[
            _flag("cancel_stops_motion", "Cancel stops motion"),
            _number("seconds_to_stop", "Time to stop", "s", required=False),
        ],
        probe="events",
        probe_events=["eval.", "runtime.soft_stop", "safety."],
        probe_help="Collects the audit events logged since the run started.",
    ),
    # --- stop categories (design §5) ---------------------------------------------
    HilStepDefinition(
        id="stop-estop",
        group="stop",
        order=17,
        title="E-stop: torque off on every motor",
        source=_DESIGN_SOURCE.format(lines="178, 187-189") + f"; {SUITE_V0}:96-97",
        quote=(
            "It disables torque on every motor on every bus the manager owns before doing "
            "anything else: the follower, the leader if connected, and sim actuators."
        ),
        instructions=[
            "Position the follower low over a padded mechanical catch and keep hands clear: "
            "it drops when torque goes off.",
            "With the arm holding a pose, press E-stop (UI button, Escape or `so101 estop`).",
            "Probe while the E-stop is latched, then reset it deliberately.",
        ],
        expected_results=[
            "The arm drops; nothing holds it.",
            "Torque_Enable reads 0 on every motor of every connected bus.",
            'Status shows estopped=true and torque_state="off".',
        ],
        evidence_fields=[
            _flag("arm_dropped", "Arm dropped (no holding torque)"),
            _flag("torque_off_all_motors", "Torque_Enable = 0 on every motor"),
            _choice(
                "torque_state", "torque_state shown", ("on", "off", "unconfirmed"), expect="off"
            ),
        ],
        probe="estop",
        requires=_requires("torque_state"),
        probe_help='Checks estopped=true and torque_state="off".',
    ),
    HilStepDefinition(
        id="stop-halt",
        group="stop",
        order=18,
        title="Halt: hold in place, keep the object",
        source=_DESIGN_SOURCE.format(lines="179, 203, 308"),
        quote=(
            "Freeze in place. Motors stay on and hold position. … Gripper: it never opens "
            "further, so a held object is not dropped. … Halt drifts less than 1° over 10 s."
        ),
        instructions=[
            "Put a light object in the gripper, arm, and move away from the rest pose.",
            "Press Halt and start the probe at once (it samples for 12 s).",
            "Do not touch the arm while the probe runs.",
        ],
        expected_results=[
            'stop_state stays "halted" for the whole window.',
            f"No arm joint drifts more than {HALT_MAX_DRIFT_DEG:g}° over at least "
            f"{HALT_MIN_HOLD_S:g} s.",
            "The held object is not released.",
        ],
        evidence_fields=[
            _flag("object_released", "Held object was released", expect=False),
            _number("max_drift_deg", "Largest joint drift", "deg", pass_max=HALT_MAX_DRIFT_DEG),
            _number("hold_duration_s", "Hold observed for", "s", pass_min=HALT_MIN_HOLD_S),
        ],
        probe="halt_hold",
        probe_window_s=12.0,
        requires=_requires("stop_state", "halt"),
        probe_help=STOP_PROBE_NOTE,
    ),
    HilStepDefinition(
        id="stop-collapse",
        group="stop",
        order=19,
        title="Collapse: capped descent to the rest pose, then torque off",
        source=_DESIGN_SOURCE.format(lines="180, 222-231"),
        quote=(
            "Slow fall: lower to the rest pose at a capped speed, then motors off. … Every "
            "joint is capped at `SO101_COLLAPSE_SPEED_DEG_S` (default 15)."
        ),
        instructions=[
            "Confirm gate5-calibration already verified the recorded rest pose before this "
            "server/HIL run. If it did not, attest fail and end the run; never invoke the "
            "standalone rest-pose command while the server is running.",
            "Arm, move away from the rest pose, and keep the space under the arm clear.",
            "Start the probe, then press Collapse within 15 s. Keep hands clear until torque "
            "is off.",
        ],
        expected_results=[
            "No arm joint moves faster than SO101_COLLAPSE_SPEED_DEG_S (default 15 °/s).",
            f"The arm ends within {COLLAPSE_SETTLE_DEG:g}° of the rest pose.",
            'Torque is off at the end (torque_state="off").',
            "The descent ends within SO101_COLLAPSE_TIMEOUT_SECONDS (at most 12 s).",
        ],
        evidence_fields=[
            _flag("speed_within_cap", "Speed stayed within the cap"),
            _number("max_speed_deg_s", "Fastest joint speed", "deg/s"),
            _number(
                "final_error_deg",
                "Largest final error vs the rest pose",
                "deg",
                pass_max=COLLAPSE_SETTLE_DEG,
            ),
            _flag("torque_off_at_end", "Torque off at the end"),
            _flag("within_timeout", "Finished within the timeout"),
            _number("duration_s", "Descent duration", "s"),
        ],
        probe="collapse",
        requires=_requires("stop_state", "torque_state", "collapse"),
        probe_help=STOP_PROBE_NOTE,
    ),
    HilStepDefinition(
        id="stop-hold-timer",
        group="stop",
        order=20,
        title="Hold-timer expiry triggers Collapse",
        source=_DESIGN_SOURCE.format(lines="208-209, 277"),
        quote=(
            "A halt holds for at most `SO101_HALT_HOLD_SECONDS` (default 60). When that "
            "expires, the halt becomes a Collapse."
        ),
        instructions=[
            "For release evidence, keep the default 60-second hold and do not restart the "
            "server during this step.",
            "For a deliberately shortened non-release exercise, first pause this run without "
            "sealing it, confirm torque off, stop the server gracefully, set "
            "SO101_HALT_HOLD_SECONDS, restart the same build/backend/data folder and resume "
            "the same RUN_ID. Never change the setting while the arm is energized.",
            "Arm, move away from the rest pose, press Halt and start the probe at once.",
            "Do not renew the halt. Wait.",
        ],
        expected_results=[
            'stop_state goes from "halted" to "collapsing" when the hold expires (probe '
            "tolerance: 1 s early to 3 s late).",
            "The Collapse then completes as in stop-collapse.",
        ],
        evidence_fields=[
            _flag("collapse_started", "Collapse started when the hold expired"),
            _number("collapse_started_after_s", "Collapse started after the halt by", "s"),
            _number("hold_seconds_setting", "SO101_HALT_HOLD_SECONDS in effect", "s"),
        ],
        probe="hold_timer",
        requires=_requires("stop_state", "halt"),
        probe_help=STOP_PROBE_NOTE,
    ),
    HilStepDefinition(
        id="stop-supervision-loss",
        group="stop",
        order=21,
        title="Supervision loss triggers Collapse after the grace period",
        source=_DESIGN_SOURCE.format(lines="212-213, 277"),
        quote=(
            "While supervision is lost, the hold lasts only `SO101_SUPERVISION_GRACE_SECONDS` "
            "(default 5)."
        ),
        instructions=[
            "Arm from the browser UI (it sends the supervision heartbeat), move away from the "
            "rest pose and press Halt.",
            "Start this probe from the CLI or a second device, then close the supervising "
            "browser tab and start a stopwatch.",
            "Stop the stopwatch when the arm starts to descend.",
        ],
        expected_results=[
            "A Collapse starts within SO101_SUPERVISION_GRACE_SECONDS (default 5 s) of losing "
            "supervision (probe tolerance: +3 s).",
            "It completes as in stop-collapse.",
        ],
        evidence_fields=[
            _flag("collapse_started", "Collapse started"),
            _flag("within_grace", "Started within the grace period"),
            _number("seconds_after_loss", "Stopwatch: loss to descent", "s"),
        ],
        probe="supervision_loss",
        probe_window_s=30.0,
        requires=_requires("stop_state", "collapse"),
        probe_help=STOP_PROBE_NOTE,
    ),
    HilStepDefinition(
        id="stop-host-kill",
        group="stop",
        order=22,
        title="Known limit: the host dies while the arm is energized",
        source=_DESIGN_SOURCE.format(lines="313-316"),
        quote=(
            "If the host process dies (SIGKILL, crash, computer power loss), Feetech STS3215 "
            "servos keep holding their last goal with torque on. No software stop covers that "
            "case; only the physical power switch does."
        ),
        instructions=[
            "Do not simulate a host crash unless the follower is low over a padded mechanical "
            "catch, hands are clear and the power switch is within reach.",
            "Show the operator the follower's power switch and write down where it is.",
            "The operator acknowledges: if the Suite or the computer dies while the arm holds "
            "a pose, keep hands out of its envelope, rely on the mechanical catch and cut "
            "power at that switch.",
        ],
        expected_results=[
            "The operator can name and reach the follower power switch.",
            "The operator acknowledges that no software stop covers a dead host.",
        ],
        evidence_fields=[
            _text("power_switch_location", "Where the follower power switch is"),
            _flag("operator_can_reach_switch", "Operator can reach the switch"),
            _flag("acknowledged", "Operator acknowledges the limit"),
        ],
        known_limit=True,
    ),
)

# Safety evidence precedes data collection, training and policy rollout. Keep
# the source definitions grouped by their source documents above, then expose
# one deterministic safety-first catalog and renumber it for reports.
GROUPS: tuple[HilGroup, ...] = ("gate", "stop", "verification")
HIL_STEPS: tuple[HilStepDefinition, ...] = tuple(
    step.model_copy(update={"order": order})
    for order, step in enumerate(
        sorted(_HIL_STEPS, key=lambda item: (GROUPS.index(item.group), item.order)), start=1
    )
)
STEPS_BY_ID: dict[str, HilStepDefinition] = {step.id: step for step in HIL_STEPS}

# Bump this when an open run must be re-attested under a changed safety or
# evidence contract. The digest prevents a forgotten bump from making two
# different catalogs look compatible.
HIL_CATALOG_VERSION = 3
HIL_CATALOG_DIGEST = hashlib.sha256(
    json.dumps(
        [step.model_dump(mode="json") for step in HIL_STEPS],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def step_catalog() -> list[HilStepDefinition]:
    return [step.model_copy(deep=True) for step in HIL_STEPS]


def select_steps(
    step_ids: Iterable[str] | None = None, groups: Iterable[str] | None = None
) -> list[str]:
    """Catalog-ordered ids for a run. No filter selects every step.

    Raises ``ValueError`` naming any unknown step id or group.
    """
    wanted_ids = list(step_ids or [])
    wanted_groups = list(groups or [])
    unknown = [step_id for step_id in wanted_ids if step_id not in STEPS_BY_ID]
    unknown += [f"group {group}" for group in wanted_groups if group not in GROUPS]
    if unknown:
        raise ValueError("unknown HIL step or group: " + ", ".join(unknown))
    if not wanted_ids and not wanted_groups:
        return [step.id for step in HIL_STEPS]
    return [step.id for step in HIL_STEPS if step.id in wanted_ids or step.group in wanted_groups]

import { describe, expect, test } from "vitest";
import type { ControlState, ModelSummary, RobotStatus } from "../api";
import {
  initialRunFields,
  parseRunOverrides,
  policyLoopBlockers,
  type PolicyLoopInputs,
} from "./policyRun";
import { deriveSafetyView } from "./safetyView";

const SMOLVLA: ModelSummary = {
  name: "smolvla-cube",
  path: "/models/smolvla-cube",
  policy_type: "smolvla",
  is_mock: false,
  checkpoint: "020000",
  task: "Pick up the blue cube",
  fps: 15,
};
const ACT: ModelSummary = { ...SMOLVLA, name: "act", path: "/models/act", policy_type: "act", task: "ignored" };
const OLD_SERVER: ModelSummary = { name: "m", path: "/m", policy_type: "pi0", is_mock: true, checkpoint: "" };
const CUSTOM_VLA: ModelSummary = {
  ...SMOLVLA,
  name: "custom-vla",
  path: "/models/custom-vla",
  policy_type: "lab_policy",
  needs_task: true,
};

describe("run overrides (data audit)", () => {
  test("prefill: training fps for every model, task only for language policies", () => {
    expect(initialRunFields(SMOLVLA)).toEqual({ fps: "15", task: "Pick up the blue cube" });
    expect(initialRunFields(ACT)).toEqual({ fps: "15", task: "" });
    expect(initialRunFields(CUSTOM_VLA)).toEqual({ fps: "15", task: "Pick up the blue cube" });
    expect(initialRunFields({ ...SMOLVLA, needs_task: false })).toEqual({ fps: "15", task: "" });
    expect(initialRunFields({ ...SMOLVLA, policy_type: "PI0" })).toEqual({ fps: "15", task: "Pick up the blue cube" });
    expect(initialRunFields({ ...OLD_SERVER, policy_type: "pi05", task: "Fold it" })).toEqual({
      fps: "",
      task: "Fold it",
    });
    // Older servers send neither field: everything blank means server defaults.
    expect(initialRunFields(OLD_SERVER)).toEqual({ fps: "", task: "" });
    expect(initialRunFields(undefined)).toEqual({ fps: "", task: "" });
  });

  test("blank fields are omitted so the server keeps its defaults", () => {
    expect(parseRunOverrides(SMOLVLA, { fps: " ", task: "" })).toEqual({ ok: true, overrides: {} });
  });

  test("valid values are sent trimmed; a task is never sent for other policies", () => {
    expect(parseRunOverrides(SMOLVLA, { fps: " 30 ", task: "  stack the cups " })).toEqual({
      ok: true,
      overrides: { fps: 30, task: "stack the cups" },
    });
    expect(parseRunOverrides(ACT, { fps: "30", task: "stack the cups" })).toEqual({
      ok: true,
      overrides: { fps: 30 },
    });
    expect(parseRunOverrides(CUSTOM_VLA, { fps: "", task: " inspect the bin " })).toEqual({
      ok: true,
      overrides: { task: "inspect the bin" },
    });
  });

  test("fps must be a whole number from 1 to 240", () => {
    for (const fps of ["0", "241", "12.5", "abc", "-3"]) {
      expect(parseRunOverrides(ACT, { fps, task: "" })).toMatchObject({ ok: false, field: "fps" });
    }
    expect(parseRunOverrides(ACT, { fps: "240", task: "" })).toEqual({ ok: true, overrides: { fps: 240 } });
  });

  test("a task longer than 1000 characters is rejected before sending", () => {
    expect(parseRunOverrides(SMOLVLA, { fps: "", task: "x".repeat(1001) })).toMatchObject({
      ok: false,
      field: "task",
    });
    expect(parseRunOverrides(SMOLVLA, { fps: "", task: "x".repeat(1000) })).toMatchObject({ ok: true });
  });

  test("with task overrides off a task is never sent or validated", () => {
    expect(parseRunOverrides(SMOLVLA, { fps: "15", task: "Pick up the red cube" }, { task: false })).toEqual({
      ok: true,
      overrides: { fps: 15 },
    });
    expect(parseRunOverrides(SMOLVLA, { fps: "", task: "x".repeat(1001) }, { task: false })).toEqual({
      ok: true,
      overrides: {},
    });
    expect(parseRunOverrides(SMOLVLA, { fps: "0", task: "" }, { task: false })).toMatchObject({ ok: false, field: "fps" });
  });
});

const NOW = 1_000_000;

function inputs(
  overrides: {
    status?: Partial<RobotStatus>;
    control?: Partial<ControlState> | null;
    linkLive?: boolean;
    supervised?: boolean;
  } = {},
): PolicyLoopInputs {
  const control: ControlState | null =
    overrides.control === null
      ? null
      : {
          armed: false,
          safety_state: "disarmed",
          lease: null,
          watchdog_ms: 500,
          last_command_at: null,
          ...overrides.control,
        };
  const status = {
    mode: "idle",
    follower_connected: true,
    leader_connected: true,
    estopped: false,
    mock: true,
    cameras: [],
    live_cameras: [],
    camera_frame_age_ms: {},
    operation_owner: null,
    fps: 0,
    ...overrides.status,
  } as RobotStatus;
  return {
    linkLive: overrides.linkLive ?? true,
    supervised: overrides.supervised ?? true,
    control,
    view: deriveSafetyView({
      telemetry: null,
      transportStale: true,
      telemetryAt: null,
      status,
      statusAt: NOW,
      statusStale: false,
      control,
      controlStale: false,
      now: NOW,
    }),
  };
}

describe("policyLoopBlockers", () => {
  test("a clear, disarmed, supervised robot has no blockers", () => {
    expect(policyLoopBlockers(inputs())).toEqual([]);
  });

  test("each precondition has its own reason", () => {
    expect(policyLoopBlockers(inputs({ linkLive: false }))).toEqual([
      "No live telemetry link to the robot server.",
    ]);
    expect(policyLoopBlockers(inputs({ supervised: false }))[0]).toMatch(/heartbeat/);
    expect(policyLoopBlockers(inputs({ control: null }))).toEqual([
      "The control-plane state is unknown.",
    ]);
    expect(
      policyLoopBlockers(
        inputs({
          control: {
            lease: { source: "agent-lab", expires_at: "", ttl_seconds: 10, server_time: "" },
          },
        }),
      ),
    ).toEqual(["A control lease (agent-lab) owns the arm. Release it first."]);
    expect(policyLoopBlockers(inputs({ control: { armed: true } }))).toEqual([
      "The control plane is armed.",
    ]);
  });

  test("E-stop and torque-unconfirmed come first and are not repeated as a safety state", () => {
    const reasons = policyLoopBlockers(
      inputs({ status: { estopped: true, torque_unconfirmed: true }, control: { safety_state: "estopped" } }),
    );
    expect(reasons).toHaveLength(2);
    expect(reasons[0]).toMatch(/E-stop is latched/);
    expect(reasons[1]).toMatch(/Torque-off is unconfirmed/);
  });

  test("halted and collapsing block with their own explanation", () => {
    const halted = policyLoopBlockers(
      inputs({ status: { stop_state: "halted" }, control: { armed: true, safety_state: "halted" } }),
    );
    expect(halted).toEqual([
      "The arm is halted and holding position. Collapse it or let the hold end first.",
    ]);
    expect(policyLoopBlockers(inputs({ status: { stop_state: "collapsing" } }))[0]).toMatch(/collapsing/);
  });

  test("a busy server mode or an active evaluation blocks a new loop", () => {
    expect(policyLoopBlockers({ ...inputs(), mode: "idle" })).toEqual([]);
    expect(policyLoopBlockers({ ...inputs(), mode: "teleop" })).toEqual([
      "The robot is busy in teleop mode. End that session first.",
    ]);
    expect(policyLoopBlockers({ ...inputs(), mode: "rollout" })).toEqual([
      "A policy rollout is already running. Stop it first.",
    ]);
    // An eval drives the arm in rollout mode: name the eval, not a rollout.
    expect(policyLoopBlockers({ ...inputs(), mode: "rollout", evalActive: true })).toEqual([
      "An evaluation is in progress. Finish or stop it on the Evaluation page first.",
    ]);
  });

  test("a latched non-disarmed state is named", () => {
    expect(policyLoopBlockers(inputs({ control: { safety_state: "watchdog_timeout" } }))).toEqual([
      'The safety state is "watchdog timeout"; it must be disarmed.',
    ]);
  });
});

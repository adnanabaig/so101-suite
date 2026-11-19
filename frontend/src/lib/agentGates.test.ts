import { describe, expect, it } from "vitest";
import type { ControlState } from "../api";
import {
  agentExecutionBlockers,
  endStateHoldsTorque,
  endStateLabel,
  isCancellableStatus,
  isExecutableStatus,
  isTerminalStatus,
  proposesMotion,
  requiresDeadman,
} from "./agentGates";
import type { SafetyView } from "./safetyView";

function view(overrides: Partial<SafetyView> = {}): SafetyView {
  return {
    tripped: false,
    source: "status",
    known: true,
    safetyState: "disarmed",
    armed: false,
    torqueUnconfirmed: false,
    torqueState: "unknown",
    stopKinds: ["estop"],
    canHalt: false,
    canCollapse: false,
    stopState: "none",
    halt: null,
    collapse: null,
    supervision: null,
    chipLabel: "Ready",
    tone: "neutral",
    ...overrides,
  };
}

function control(overrides: Partial<ControlState> = {}): ControlState {
  return {
    armed: false,
    safety_state: "disarmed",
    lease: null,
    watchdog_ms: 500,
    last_command_at: null,
    ...overrides,
  };
}

function lease(source: string): ControlState["lease"] {
  return { source, expires_at: "", ttl_seconds: 10, server_time: "" };
}

const ready = {
  leaseOwned: true,
  linkLive: true,
  supervised: true,
  view: view(),
  control: control({ lease: lease("agent-lab") }),
};

describe("agentExecutionBlockers (F9)", () => {
  it("is empty when this tab owns the lease and the arm is disarmed", () => {
    expect(agentExecutionBlockers(ready)).toEqual([]);
  });

  it("does not block on an idle joint stream (F8): only the link and heartbeat count", () => {
    expect(agentExecutionBlockers({ ...ready, linkLive: false })).toEqual([
      "No live telemetry link to the robot server.",
    ]);
    expect(agentExecutionBlockers({ ...ready, supervised: false })[0]).toMatch(/operator heartbeat/);
  });

  it("asks for the lease, or names the station that holds it", () => {
    expect(agentExecutionBlockers({ ...ready, leaseOwned: false, control: control() })).toEqual([
      "Acquire the Agent Lab lease first.",
    ]);
    expect(
      agentExecutionBlockers({
        ...ready,
        leaseOwned: false,
        control: control({ lease: lease("manual-ui") }),
      }),
    ).toEqual(["A control lease (manual-ui) owns the arm. Release it first."]);
  });

  it("puts the latches first and does not repeat them as a safety state", () => {
    const reasons = agentExecutionBlockers({
      ...ready,
      view: view({ tripped: true, torqueUnconfirmed: true, safetyState: "estopped" }),
    });
    expect(reasons[0]).toMatch(/^E-stop is latched/);
    expect(reasons[1]).toMatch(/^Torque-off is unconfirmed/);
    expect(reasons.some((reason) => reason.includes("must be disarmed"))).toBe(false);
  });

  it("reports C11 stop states once, without armed/safety-state noise", () => {
    const reasons = agentExecutionBlockers({
      ...ready,
      view: view({ stopState: "halted", armed: true, safetyState: "halted" }),
    });
    expect(reasons).toEqual([
      "The arm is halted and holding position. Collapse it or let the hold end first.",
    ]);
  });

  it("blocks on an unknown control plane, an armed plane, or a latched state", () => {
    expect(agentExecutionBlockers({ ...ready, view: view({ safetyState: "unknown" }) })).toEqual([
      "The control-plane state is unknown.",
    ]);
    expect(agentExecutionBlockers({ ...ready, view: view({ armed: true, safetyState: "armed" }) })).toEqual([
      "The control plane is armed.",
      'The safety state is "armed"; it must be disarmed.',
    ]);
    expect(agentExecutionBlockers({ ...ready, view: view({ safetyState: "watchdog_timeout" }) })).toEqual([
      'The safety state is "watchdog timeout"; it must be disarmed.',
    ]);
  });
});

describe("proposal states", () => {
  it("classifies statuses", () => {
    expect(isExecutableStatus("proposed")).toBe(true);
    expect(isExecutableStatus("executing")).toBe(false);
    expect(isCancellableStatus("executing")).toBe(true);
    expect(isCancellableStatus("executed")).toBe(false);
    for (const status of ["executed", "failed", "cancelled", "timeout", "expired"]) {
      expect(isTerminalStatus(status)).toBe(true);
    }
    expect(isTerminalStatus("executing")).toBe(false);
  });

  it("knows motion and deadman proposals", () => {
    expect(proposesMotion({ id: "a", targets: { gripper: 80 } })).toBe(true);
    expect(proposesMotion({ id: "a", targets: {} })).toBe(false);
    expect(proposesMotion({ id: "a", targets: { gripper: 80 }, constraints: { allow_motion: false } })).toBe(false);
    expect(requiresDeadman({ id: "a", constraints: { require_deadman: true } })).toBe(true);
    expect(requiresDeadman({ id: "a" })).toBe(false);
  });
});

describe("endStateLabel (C11)", () => {
  it("labels the known end states", () => {
    expect(endStateLabel("halted")).toBe("Halted · holding position, motors on");
    expect(endStateLabel("deenergized")).toBe("Motors off");
    expect(endStateLabel("estopped")).toBe("E-stopped");
    expect(endStateLabel("fault")).toBe("Fault");
  });

  it("renders an unknown end state verbatim and a missing one honestly", () => {
    expect(endStateLabel("parked_on_rest")).toBe("parked_on_rest");
    expect(endStateLabel(undefined)).toBe("Not reported");
    expect(endStateLabel("")).toBe("Not reported");
  });

  it("flags the end state that leaves torque on", () => {
    expect(endStateHoldsTorque("halted")).toBe(true);
    expect(endStateHoldsTorque("deenergized")).toBe(false);
  });
});

// Agent Lab rules (C8): when a motion proposal may be approved and executed,
// which proposal states allow which action, and how a result's end state
// reads. One reason list feeds both the approve/execute controls and the
// Robot context tile, so the tile can never say "ready" while the gate
// refuses (F9).
import { AGENT_TERMINAL_STATUSES, type AgentProposal, type ControlState } from "../api";
import { isStopping, safetyBlockers } from "./policyRun";
import type { SafetyView } from "./safetyView";

export interface AgentGateInputs {
  /** This tab holds the Agent Lab lease. */
  leaseOwned: boolean;
  /** Telemetry frames are arriving (transport freshness, not joint samples). */
  linkLive: boolean;
  /** This tab's operator heartbeat is reaching the server. */
  supervised: boolean;
  view: SafetyView;
  control: ControlState | null;
}

/**
 * Why agent motion cannot be approved or started right now, in the order an
 * operator should fix them. Empty means the server should accept it. An idle
 * joint stream is not a reason (F8): the server samples the arm itself.
 */
export function agentExecutionBlockers({
  leaseOwned,
  linkLive,
  supervised,
  view,
  control,
}: AgentGateInputs): string[] {
  const reasons = safetyBlockers({ linkLive, supervised, view });
  const stopping = isStopping(view);
  if (!leaseOwned) {
    reasons.push(
      control?.lease
        ? `A control lease (${control.lease.source}) owns the arm. Release it first.`
        : "Acquire the Agent Lab lease first.",
    );
  }
  if (view.safetyState === "unknown") {
    reasons.push("The control-plane state is unknown.");
  } else {
    if (view.armed && !stopping) reasons.push("The control plane is armed.");
    if (!view.tripped && !stopping && view.safetyState !== "disarmed") {
      reasons.push(`The safety state is "${view.safetyState.replace(/_/g, " ")}"; it must be disarmed.`);
    }
  }
  return reasons;
}

const EXECUTABLE = new Set(["proposed", "pending", "ready", "approved"]);
const CANCELLABLE = new Set([...EXECUTABLE, "executing"]);
// Not C8 statuses, but earlier servers have ended proposals with them.
const ALSO_TERMINAL = new Set(["rejected", "completed"]);

export function proposalStatus(proposal: AgentProposal): string {
  return proposal.status ?? "proposed";
}

/** Proposes physical motion: it has targets and was allowed to move. */
export function proposesMotion(proposal: AgentProposal): boolean {
  return (
    Object.keys(proposal.targets ?? {}).length > 0 && proposal.constraints?.allow_motion !== false
  );
}

export function isExecutableStatus(status: string): boolean {
  return EXECUTABLE.has(status);
}

export function isCancellableStatus(status: string): boolean {
  return CANCELLABLE.has(status);
}

export function isTerminalStatus(status: string): boolean {
  return AGENT_TERMINAL_STATUSES.includes(status) || ALSO_TERMINAL.has(status);
}

/** C8 `require_deadman`: execution runs only while the operator holds. */
export function requiresDeadman(proposal: AgentProposal): boolean {
  return proposal.constraints?.require_deadman === true;
}

const END_STATE_LABELS: Record<string, string> = {
  halted: "Halted · holding position, motors on",
  deenergized: "Motors off",
  estopped: "E-stopped",
  fault: "Fault",
  unknown: "Motor state unknown",
};

/**
 * C11 end states, in operator words. A value this UI does not know is shown
 * verbatim rather than guessed at.
 */
export function endStateLabel(endState: unknown): string {
  if (typeof endState !== "string" || endState === "") return "Not reported";
  return END_STATE_LABELS[endState] ?? endState;
}

/** Torque may still be on after this end state. */
export function endStateHoldsTorque(endState: unknown): boolean {
  return endState === "halted";
}

/** The server could not confirm whether the motors are on. */
export function endStateUnknown(endState: unknown): boolean {
  return endState === "unknown";
}

const STOP_REASON_LABELS: Record<string, string> = {
  target_reached: "Reached the target within tolerance",
  max_duration: "Ran out of time before reaching the target",
  cancelled: "Cancelled",
  deadman: "Hold-to-run released (no hold within 300 ms)",
  operator_heartbeat: "This tab's operator heartbeat went stale",
  estop: "E-stop engaged",
  fault: "Runtime fault",
  rejected: "The control plane refused a motion command",
  arm: "The arm could not be armed",
  error: "Server error",
  shutdown: "Server shut down",
  restart: "Server restarted mid-run",
};

/**
 * C8 `stop_reason` in operator words. Unknown reasons are shown verbatim; the
 * server's own `error` string is always shown next to it, unedited.
 */
export function stopReasonLabel(reason: unknown): string | null {
  if (typeof reason !== "string" || reason === "") return null;
  return STOP_REASON_LABELS[reason] ?? reason;
}

/** C8 units: the gripper is in gripper units (0-100), every other joint in degrees. */
export function jointUnit(joint: string): string {
  return joint === "gripper" ? " gripper units" : "°";
}

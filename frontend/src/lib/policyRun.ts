// Shared rules for starting an autonomous policy loop (Models rollout and
// Evaluation): the optional Task/FPS overrides and the start preconditions.
import { modelNeedsTask, type ControlState, type ModelSummary } from "../api";
import type { SafetyView } from "./safetyView";

export const MIN_POLICY_FPS = 1;
export const MAX_POLICY_FPS = 240;
export const MAX_TASK_LENGTH = 1000;

/** Text of the optional inputs. Blank means "use the server's default". */
export interface RunFields {
  fps: string;
  task: string;
}

export const EMPTY_RUN_FIELDS: RunFields = { fps: "", task: "" };

/**
 * Prefill from the checkpoint's own metadata when the server reports it (data
 * audit): its training fps, and for language policies (smolvla, pi0) the task
 * it was trained with.
 */
export function initialRunFields(model: ModelSummary | null | undefined): RunFields {
  const fps =
    typeof model?.fps === "number" && Number.isFinite(model.fps) ? String(model.fps) : "";
  const task = modelNeedsTask(model) && typeof model?.task === "string" ? model.task : "";
  return { fps, task };
}

export interface RunOverrides {
  fps?: number;
  task?: string;
}

export type RunOverridesResult =
  | { ok: true; overrides: RunOverrides }
  | { ok: false; field: "fps" | "task"; error: string };

export interface RunOverrideOptions {
  /**
   * Whether this server path accepts a task override. A read-only caller can
   * set this false to omit and skip validation of the task field.
   */
  task?: boolean;
}

/**
 * Validate the optional fields. Blank fields are omitted from the request so
 * the server keeps its defaults (training fps else 30, training task). A task
 * is only sent for language policies, and only where `options.task` allows.
 */
export function parseRunOverrides(
  model: ModelSummary | null | undefined,
  fields: RunFields,
  options: RunOverrideOptions = {},
): RunOverridesResult {
  const overrides: RunOverrides = {};
  const fpsText = fields.fps.trim();
  if (fpsText) {
    const fps = Number(fpsText);
    if (!Number.isInteger(fps) || fps < MIN_POLICY_FPS || fps > MAX_POLICY_FPS) {
      return {
        ok: false,
        field: "fps",
        error: `FPS must be a whole number from ${MIN_POLICY_FPS} to ${MAX_POLICY_FPS}, or blank for the server's default.`,
      };
    }
    overrides.fps = fps;
  }
  if (options.task !== false && modelNeedsTask(model)) {
    const task = fields.task.trim();
    if (task.length > MAX_TASK_LENGTH) {
      return {
        ok: false,
        field: "task",
        error: `The task must be at most ${MAX_TASK_LENGTH} characters (it has ${task.length}).`,
      };
    }
    if (task) overrides.task = task;
  }
  return { ok: true, overrides };
}

export interface PolicyLoopInputs {
  /** A fresh telemetry frame is arriving (transport, not sample freshness). */
  linkLive: boolean;
  /** This tab's operator heartbeat is reaching the server. */
  supervised: boolean;
  view: SafetyView;
  control: ControlState | null;
  /** Server mode (telemetry when live, else status); undefined when unknown. */
  mode?: string;
  /** An evaluation session is active (it drives the arm in "rollout" mode). */
  evalActive?: boolean;
}

/**
 * Reasons shared by every motion-starting action: the stop latches, torque
 * truth (C2), the live link, this tab's operator heartbeat and the C11 stop
 * states, in the order an operator should fix them.
 */
export function safetyBlockers({
  linkLive,
  supervised,
  view,
}: Pick<PolicyLoopInputs, "linkLive" | "supervised" | "view">): string[] {
  const reasons: string[] = [];
  if (view.tripped) {
    reasons.push("E-stop is latched. Inspect the workspace, then reset it from the banner.");
  }
  if (view.torqueUnconfirmed) {
    reasons.push("Torque-off is unconfirmed. Cut physical power, then clear the fault on Diagnostics.");
  }
  if (!linkLive) reasons.push("No live telemetry link to the robot server.");
  if (!supervised) {
    reasons.push("This tab's operator heartbeat is not reaching the server. Keep this page open and visible.");
  }
  if (view.stopState === "halted") {
    reasons.push("The arm is halted and holding position. Collapse it or let the hold end first.");
  } else if (view.stopState === "collapsing") {
    reasons.push("The arm is collapsing to rest. Wait for it to finish.");
  }
  return reasons;
}

/** C11 stop in progress: armed/safety-state reasons would only repeat it. */
export function isStopping(view: SafetyView): boolean {
  return view.stopState === "halted" || view.stopState === "collapsing";
}

/**
 * Why an internally supervised policy loop cannot start right now, in the
 * order an operator should fix them. Empty means the server should accept it.
 * The idle joint sample (F8) is deliberately not a reason: the loop connects
 * and samples the follower itself.
 */
export function policyLoopBlockers({
  linkLive,
  supervised,
  view,
  control,
  mode,
  evalActive = false,
}: PolicyLoopInputs): string[] {
  const reasons = safetyBlockers({ linkLive, supervised, view });
  const stopping = isStopping(view);
  if (evalActive) {
    reasons.push("An evaluation is in progress. Finish or stop it on the Evaluation page first.");
  } else if (mode === "rollout") {
    reasons.push("A policy rollout is already running. Stop it first.");
  } else if (mode !== undefined && mode !== "idle") {
    reasons.push(`The robot is busy in ${mode} mode. End that session first.`);
  }
  if (view.safetyState === "unknown") {
    reasons.push("The control-plane state is unknown.");
  } else {
    if (control?.lease) {
      reasons.push(`A control lease (${control.lease.source}) owns the arm. Release it first.`);
    }
    if (view.armed && !stopping) reasons.push("The control plane is armed.");
    if (!view.tripped && !stopping && view.safetyState !== "disarmed") {
      reasons.push(`The safety state is "${view.safetyState.replace(/_/g, " ")}"; it must be disarmed.`);
    }
  }
  return reasons;
}

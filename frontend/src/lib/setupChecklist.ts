import {
  ApiError,
  isMissingRoute,
  type CalibrationStatus,
  type CapabilitiesResponse,
  type PreflightCheckResult,
  type PreflightReport,
} from "../api";
import type { BackendInfo } from "./backend";
import { errorMessage, joinSentences } from "./errors";

// The first-run setup checklist, derived only from what the server reports:
// capabilities, saved calibration and the last preflight report. Nothing here
// is inferred from UI state, and no item runs anything on the server.

export type SetupItemId =
  | "backend"
  | "calibration.follower"
  | "calibration.leader"
  | "preflight"
  | "stops";

/**
 * How an item stands. Only `done` counts as complete, and `rehearsal` too
 * when the server itself is a rehearsal backend (mock or sim).
 */
export type SetupItemState =
  | "done"
  | "rehearsal"
  | "todo"
  | "problem"
  | "optional"
  | "running"
  | "checking"
  | "unknown"
  | "unavailable";

export interface SetupLink {
  to: string;
  label: string;
}

export interface SetupItem {
  id: SetupItemId;
  title: string;
  state: SetupItemState;
  /** What the operator needs to know, in our words. */
  summary: string;
  /** Server text, shown exactly as sent: a detail, a capability note, check messages. */
  details: string[];
  /** When the evidence was made: a saved calibration or a preflight run. */
  when?: { label: string; iso: string };
  /** Required items decide "setup complete"; optional ones only inform. */
  required: boolean;
  link?: SetupLink;
}

/** One fetched resource, as usePolling reports it: the last data wins over a later error. */
export interface Fetched<T> {
  data: T | null;
  error: Error | null;
}

export interface SetupInputs {
  capabilities: Fetched<CapabilitiesResponse>;
  /** From capabilities, else robot status (backendInfo); null while neither has answered. */
  backend: BackendInfo | null;
  follower: Fetched<CalibrationStatus>;
  leader: Fetched<CalibrationStatus>;
  preflight: Fetched<PreflightReport>;
  stopsAcknowledged: boolean;
  canHalt: boolean;
  canCollapse: boolean;
}

/** Mock and sim servers are rehearsals; an inferred or reported real arm is not. */
export function isRehearsalBackend(backend: BackendInfo | null): boolean {
  return backend?.kind === "mock" || backend?.kind === "sim";
}

/**
 * Whether an item counts toward "setup complete". A rehearsal (a mock or sim
 * run) counts only on a mock or sim server: nothing a rehearsal saved is
 * evidence for the physical arm, though both share one data folder.
 */
export function isComplete(item: SetupItem, backend: BackendInfo | null): boolean {
  if (item.state === "rehearsal") return isRehearsalBackend(backend);
  return item.state === "done";
}

/** The prefix of the calibration status message for an unreadable saved file. */
export const INVALID_CALIBRATION_PREFIX = "invalid saved calibration";

/** At most this many check lines per item; the rest are counted. */
export const MAX_CHECK_LINES = 4;

const CALIBRATION_LINK: SetupLink = { to: "/calibration", label: "Open Calibration" };

function backendItem(
  capabilities: Fetched<CapabilitiesResponse>,
  backend: BackendInfo | null,
  report: PreflightReport | null,
): SetupItem {
  const base = { id: "backend" as const, title: "Robot backend", required: true, details: [] };
  const data = capabilities.data;
  if (!data) {
    if (capabilities.error) {
      return {
        ...base,
        state: "unknown",
        summary: "Could not read the server's capabilities.",
        details: [errorMessage(capabilities.error)],
      };
    }
    return { ...base, state: "checking", summary: "Reading the server's capabilities." };
  }
  if (backend?.kind === "mock" || (!backend && data.mock)) {
    return {
      ...base,
      state: "done",
      summary: "Mock rehearsal: no real motors move. Practise here; nothing on this backend is evidence about a real arm.",
    };
  }
  if (backend?.kind === "sim") {
    return {
      ...base,
      state: "done",
      summary:
        "MuJoCo simulation: commands move a simulated arm, not the physical one. Practise here; nothing on this backend is evidence about a real arm.",
    };
  }
  const control = data.capabilities.find((capability) => capability.id === "control");
  if (!control) {
    return { ...base, state: "unknown", summary: "This server does not report whether it can drive an arm." };
  }
  if (!control.available) {
    return {
      ...base,
      state: "todo",
      summary: "The robot software is not installed, so this server cannot drive an arm.",
      details: [control.detail],
    };
  }
  if (backend?.reported) {
    return { ...base, state: "done", summary: "A real arm: commands move the physical motors." };
  }
  // A server before v0.2 has no `backend`, and its sim backend reports
  // mock=false, so a real arm and the MuJoCo sim look the same here. A
  // preflight report does record which backend it ran on.
  return {
    ...base,
    state: "done",
    summary:
      report && report.backend !== "mock"
        ? `A real or simulated arm; the server does not say which. The last preflight ran on the ${report.backend} backend.`
        : "A real or simulated arm; the server does not say which. A preflight report names its backend.",
  };
}

function calibrationItem(
  arm: "follower" | "leader",
  fetched: Fetched<CalibrationStatus>,
  backend: BackendInfo | null,
): SetupItem {
  const base = {
    id: `calibration.${arm}` as const,
    title: arm === "follower" ? "Follower calibration" : "Leader calibration",
    required: arm === "follower",
    details: [],
  };
  const status = fetched.data;
  if (!status) {
    if (fetched.error) {
      if (isMissingRoute(fetched.error)) {
        return { ...base, state: "unavailable", summary: "This server has no calibration status API." };
      }
      return {
        ...base,
        state: "unknown",
        summary: `Could not read the ${arm}'s calibration.`,
        details: [errorMessage(fetched.error)],
      };
    }
    return { ...base, state: "checking", summary: "Reading the saved calibration." };
  }
  if (status.in_progress) {
    return {
      ...base,
      state: "running",
      summary: `Calibrating now: ${Math.round(status.fraction * 100)}%.`,
      details: status.message ? [status.message] : [],
    };
  }
  if (status.has_saved) {
    return {
      ...base,
      state: "done",
      summary: status.device_id ? `Saved for ${status.device_id}.` : "Saved.",
      when:
        status.created_at !== null
          ? { label: "Calibrated", iso: new Date(status.created_at * 1000).toISOString() }
          : undefined,
    };
  }
  if (status.message.startsWith(INVALID_CALIBRATION_PREFIX)) {
    return {
      ...base,
      state: "problem",
      summary: "The saved calibration file cannot be used. Calibrate again.",
      details: [status.message],
      link: CALIBRATION_LINK,
    };
  }
  if (backend === null) {
    return { ...base, state: "checking", summary: "Reading the server's backend." };
  }
  if (backend.kind === "mock") {
    return {
      ...base,
      state: "done",
      summary: "No saved file; the mock backend uses default ranges. Calibrate before using a real arm.",
    };
  }
  if (backend.kind === "sim") {
    return {
      ...base,
      state: "done",
      summary: "Not needed: the simulated arm has no encoders to calibrate. Calibrate before using a real arm.",
    };
  }
  // The server says why when there is something to say, e.g. that the saved
  // file was made on the mock backend and so counts as missing here.
  const details = status.message ? [status.message] : [];
  if (arm === "leader") {
    return {
      ...base,
      state: "optional",
      summary: "Not calibrated. Needed only to teleoperate with the leader arm.",
      details,
      link: CALIBRATION_LINK,
    };
  }
  return {
    ...base,
    state: "todo",
    summary: "Not calibrated. Calibrate the follower before any real motion.",
    details,
    link: CALIBRATION_LINK,
  };
}

function checkLine(check: PreflightCheckResult): string {
  const title = check.arm ? `${check.title} (${check.arm})` : check.title;
  return `${title}: ${check.remediation ? joinSentences(check.message, check.remediation) : check.message}`;
}

/** "Title (arm): message remediation" for each matching check, capped. */
export function checkLines(checks: readonly PreflightCheckResult[]): string[] {
  const lines = checks.slice(0, MAX_CHECK_LINES).map(checkLine);
  const more = checks.length - MAX_CHECK_LINES;
  if (more > 0) lines.push(`and ${more} more.`);
  return lines;
}

function preflightItem(fetched: Fetched<PreflightReport>, backend: BackendInfo | null): SetupItem {
  const base = { id: "preflight" as const, title: "Preflight", required: true, details: [] };
  const report = fetched.data;
  if (!report) {
    const error = fetched.error;
    if (!error) return { ...base, state: "checking", summary: "Reading the last preflight report." };
    if (isMissingRoute(error)) {
      return { ...base, state: "unavailable", summary: "This server has no preflight API, so no report can be shown." };
    }
    if (error instanceof ApiError && error.status === 404) {
      return {
        ...base,
        state: "todo",
        summary: "No preflight report yet. Run `so101 preflight` in a terminal on the workstation.",
        details: [error.message],
      };
    }
    return {
      ...base,
      state: "unknown",
      summary: "Could not read the last preflight report.",
      details: [errorMessage(error)],
    };
  }
  const when = {
    label: report.origin === "cli" ? "Ran with so101 preflight" : "Ran on the server",
    iso: report.created_at,
  };
  if (report.rehearsal && !isRehearsalBackend(backend)) {
    // A real (or not yet known) server: a mock or sim run says nothing about this arm.
    if (backend === null) return { ...base, state: "checking", summary: "Reading the server's backend.", when };
    return {
      ...base,
      state: "todo",
      summary: `The last preflight was a ${report.backend} rehearsal, which is not evidence about this arm. Run \`so101 preflight\` on the real backend.`,
      when,
    };
  }
  const warnings = checkLines(report.checks.filter((check) => check.status === "warn"));
  if (report.ready_for_hardware) {
    return {
      ...base,
      state: "done",
      summary: report.overall === "warn" ? "Passed with warnings on the real backend." : "Passed on the real backend.",
      details: warnings,
      when,
    };
  }
  if (report.rehearsal && (report.overall === "pass" || report.overall === "warn")) {
    return {
      ...base,
      state: "rehearsal",
      summary: `Rehearsal passed on the ${report.backend} backend. A rehearsal is not evidence about a real arm.`,
      details: warnings,
      when,
    };
  }
  const subject = report.rehearsal ? `The ${report.backend} rehearsal` : "Preflight";
  if (report.overall === "fail") {
    return {
      ...base,
      state: "problem",
      summary: `${subject} failed.`,
      details: checkLines(report.checks.filter((check) => check.status === "fail")),
      when,
    };
  }
  return {
    ...base,
    state: "todo",
    summary: `${subject} is incomplete: a required check was skipped.`,
    details: checkLines(report.checks.filter((check) => check.status === "skip" && check.blocking)),
    when,
  };
}

function stopsItem(acknowledged: boolean, canHalt: boolean, canCollapse: boolean): SetupItem {
  const halt = canHalt
    ? "H is Halt: the arm freezes and the motors hold."
    : "This server has no Halt, so H does nothing.";
  const collapse = canCollapse ? " Collapse has no key: its button lowers the arm to rest, then turns the motors off." : "";
  return {
    id: "stops",
    title: "Know your stops",
    required: true,
    details: [],
    state: acknowledged ? "done" : "todo",
    summary: `Escape is E-stop on every page, even in a field or a dialog: motor power is cut. ${halt}${collapse}`,
  };
}

/** The checklist, in the order an operator works through it. */
export function setupItems(inputs: SetupInputs): SetupItem[] {
  return [
    backendItem(inputs.capabilities, inputs.backend, inputs.preflight.data),
    calibrationItem("follower", inputs.follower, inputs.backend),
    calibrationItem("leader", inputs.leader, inputs.backend),
    preflightItem(inputs.preflight, inputs.backend),
    stopsItem(inputs.stopsAcknowledged, inputs.canHalt, inputs.canCollapse),
  ];
}

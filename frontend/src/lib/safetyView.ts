// Pure derivation of what the safety bar shows. Kept free of React so every
// rule (F4 precedence, C2 torque truth, C11 stop states) is unit-testable.
import type {
  CollapseInfo,
  ControlState,
  HaltInfo,
  RobotStatus,
  StopFields,
  StopKind,
  SupervisionInfo,
  Telemetry,
} from "../api";

/** HTTP status older than this no longer counts as fresh. */
export const STATUS_FRESH_MS = 5000;

const UNSAFE_STATES = new Set([
  "fault",
  "estopped",
  "watchdog_timeout",
  "lease_expired",
  "operator_timeout",
]);

export type SafetyTone = "danger" | "warn" | "neutral" | "unknown";

export interface SafetySources {
  telemetry: Telemetry | null;
  transportStale: boolean;
  /** Client time the latest telemetry frame arrived. */
  telemetryAt: number | null;
  status: RobotStatus | null;
  /** Client time of the last successful status response. */
  statusAt: number | null;
  /** The status poller has failed repeatedly. */
  statusStale: boolean;
  control: ControlState | null;
  controlStale: boolean;
  now: number;
}

/** Halt details plus the client time they were received (for countdowns). */
export interface ReceivedHalt {
  info: HaltInfo;
  receivedAt: number;
}

export interface SafetyView {
  /** The E-stop latch as the operator should see it (F4). */
  tripped: boolean;
  /** Which source decided `tripped`. */
  source: "telemetry" | "status" | "none";
  /** At least one source is fresh enough to trust. */
  known: boolean;
  /** Server safety state (control plane) or "unknown". */
  safetyState: string;
  armed: boolean;
  /** C2: torque-off could not be confirmed. Physical power must be cut. */
  torqueUnconfirmed: boolean;
  /** "on" | "off" | "unconfirmed" | "unknown" (absent field). */
  torqueState: string;
  /** C11: always includes "estop". */
  stopKinds: StopKind[];
  canHalt: boolean;
  canCollapse: boolean;
  /** "none" | "halted" | "collapsing", or an unknown server string verbatim. */
  stopState: string;
  halt: ReceivedHalt | null;
  collapse: CollapseInfo | null;
  /** Null when the server does not report it (older servers). */
  supervision: SupervisionInfo | null;
  chipLabel: string;
  tone: SafetyTone;
}

const KNOWN_STOP_KINDS: readonly StopKind[] = ["estop", "halt", "collapse"];

/** C11: absent means the server only implements E-stop. */
export function stopKindsOf(status: Pick<RobotStatus, "stop_kinds"> | null): StopKind[] {
  const advertised = status?.stop_kinds;
  if (!Array.isArray(advertised)) return ["estop"];
  const kinds = KNOWN_STOP_KINDS.filter((kind) => advertised.includes(kind));
  return kinds.includes("estop") ? kinds : ["estop", ...kinds];
}

type StopField = keyof StopFields;

/**
 * C11 fields come from a fresh telemetry frame when it carries them, otherwise
 * from HTTP status, so a live socket on an older server cannot hide them.
 */
function pick<K extends StopField>(
  live: Telemetry | null,
  liveAt: number | null,
  status: RobotStatus | null,
  statusAt: number | null,
  field: K,
): { value: StopFields[K]; at: number | null } | null {
  if (live && live[field] !== undefined) return { value: live[field], at: liveAt };
  if (status && status[field] !== undefined) return { value: status[field], at: statusAt };
  return null;
}

/**
 * Milliseconds left on a halt's hold timer, measured on the server clock:
 * `hold_expires_at - server_time` at receipt, minus client time elapsed since.
 * Client clock skew never enters the result.
 */
export function haltRemainingMs(halt: ReceivedHalt, now: number): number | null {
  const expires = Date.parse(halt.info.hold_expires_at);
  const serverNow = Date.parse(halt.info.server_time);
  if (Number.isNaN(expires) || Number.isNaN(serverNow)) return null;
  return Math.max(0, expires - serverNow - Math.max(0, now - halt.receivedAt));
}

/**
 * Milliseconds until an unsupervised halt becomes a Collapse, as the server
 * computes it: the grace period from `lost_at` (or from the halt, if later),
 * never past the hold timer. Measured on the server clock via the halt's
 * `server_time`. Null while supervised, or when the server does not report
 * supervision.
 */
export function unsupervisedRemainingMs(
  halt: ReceivedHalt,
  supervision: SupervisionInfo | null | undefined,
  now: number,
): number | null {
  if (!supervision || supervision.supervised !== false) return null;
  const hold = haltRemainingMs(halt, now);
  const serverNow = Date.parse(halt.info.server_time);
  const since = Date.parse(halt.info.since);
  const lost = supervision.lost_at ? Date.parse(supervision.lost_at) : Number.NaN;
  const from = Math.max(Number.isNaN(lost) ? -Infinity : lost, Number.isNaN(since) ? -Infinity : since);
  if (Number.isNaN(serverNow) || !Number.isFinite(from) || !Number.isFinite(supervision.grace_s)) return hold;
  const left = Math.max(0, from + supervision.grace_s * 1000 - serverNow - Math.max(0, now - halt.receivedAt));
  return hold === null ? left : Math.min(left, hold);
}

/** "Unsupervised: collapses in N s", rounded up so it never reads 0 early. */
export function unsupervisedText(ms: number): string {
  return `Unsupervised: collapses in ${Math.ceil(ms / 1000)} s`;
}

/** Collapse progress as a 0-100 percentage (tolerates a 0-100 server value). */
export function collapsePercent(collapse: CollapseInfo): number | null {
  if (typeof collapse.progress !== "number" || Number.isNaN(collapse.progress)) return null;
  const percent = collapse.progress > 1 ? collapse.progress : collapse.progress * 100;
  return Math.min(100, Math.max(0, percent));
}

export function deriveSafetyView(sources: SafetySources): SafetyView {
  const { telemetry, transportStale, status, control } = sources;

  // F4: a stale WebSocket must never outvote fresh HTTP status.
  const tripped =
    !transportStale && telemetry ? telemetry.estopped : (status?.estopped ?? false);
  const source: SafetyView["source"] =
    !transportStale && telemetry ? "telemetry" : status ? "status" : "none";

  const statusFresh =
    status !== null &&
    !sources.statusStale &&
    sources.statusAt !== null &&
    sources.now - sources.statusAt <= STATUS_FRESH_MS;
  const live = !transportStale && telemetry ? telemetry : null;
  const known = live !== null || statusFresh;

  const torque = pick(live, sources.telemetryAt, status, sources.statusAt, "torque_state");
  const torqueState = typeof torque?.value === "string" ? torque.value : "unknown";
  const torqueUnconfirmed = status?.torque_unconfirmed === true || torqueState === "unconfirmed";

  const stopKinds = stopKindsOf(status);
  const stopStateField = pick(live, sources.telemetryAt, status, sources.statusAt, "stop_state");
  const stopState = typeof stopStateField?.value === "string" ? stopStateField.value : "none";

  const haltField = pick(live, sources.telemetryAt, status, sources.statusAt, "halt");
  const halt: ReceivedHalt | null =
    haltField?.value && haltField.at !== null
      ? { info: haltField.value, receivedAt: haltField.at }
      : null;
  const collapseField = pick(live, sources.telemetryAt, status, sources.statusAt, "collapse");
  const supervisionField = pick(live, sources.telemetryAt, status, sources.statusAt, "supervision");
  const collapse = collapseField?.value ?? null;
  const supervision = supervisionField?.value ?? null;

  const controlKnown = control !== null && !sources.controlStale;
  const safetyState = controlKnown ? control.safety_state : "unknown";
  const armed = controlKnown ? control.armed : false;

  let chipLabel: string;
  let tone: SafetyTone;
  if (tripped) {
    chipLabel = "E-stopped";
    tone = "danger";
  } else if (torqueUnconfirmed) {
    chipLabel = "torque unconfirmed";
    tone = "danger";
  } else if (!known) {
    chipLabel = "state unknown";
    tone = "unknown";
  } else if (stopState === "collapsing") {
    chipLabel = "collapsing";
    tone = "warn";
  } else if (stopState === "halted") {
    chipLabel = "halted";
    tone = "warn";
  } else if (!controlKnown) {
    chipLabel = "control unknown";
    tone = "unknown";
  } else if (UNSAFE_STATES.has(safetyState)) {
    chipLabel = safetyState.replace(/_/g, " ");
    tone = "danger";
  } else if (armed) {
    chipLabel = safetyState === "disarmed" ? "armed" : safetyState.replace(/_/g, " ");
    tone = "warn";
  } else {
    chipLabel = safetyState.replace(/_/g, " ");
    tone = "neutral";
  }

  return {
    tripped,
    source,
    known,
    safetyState,
    armed,
    torqueUnconfirmed,
    torqueState,
    stopKinds,
    canHalt: stopKinds.includes("halt"),
    canCollapse: stopKinds.includes("collapse"),
    stopState,
    halt,
    collapse,
    supervision,
    chipLabel,
    tone,
  };
}

/**
 * The server's mode as far as the operator can trust it (F21): the live
 * telemetry frame, else HTTP status while it is fresh (`view.known` without a
 * live link), else undefined. A dropped link or a failing status poll must
 * not keep presenting the last mode as current.
 */
export function currentMode(
  telemetry: Pick<Telemetry, "mode"> | null,
  linkLive: boolean,
  view: Pick<SafetyView, "known">,
  status: Pick<RobotStatus, "mode"> | null,
): string | undefined {
  if (linkLive && telemetry) return telemetry.mode;
  return view.known ? status?.mode : undefined;
}

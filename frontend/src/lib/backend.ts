// What drives the arm, as the server reports it (`backend` in robot status,
// health and capabilities): mock, sim (MuJoCo physics) or real. Only "real"
// moves the physical arm. Servers before v0.2 send only `mock`, which is
// false under sim as well, so for them the UI falls back to the old reading:
// mock, else a real arm (never assume a simulation that might be a real arm).
import type { RobotBackend } from "../api";

export interface BackendInfo {
  kind: RobotBackend;
  /** False when inferred from `mock` on an older server. */
  reported: boolean;
}

export function backendInfo(source: { backend?: unknown; mock?: unknown } | null | undefined): BackendInfo | null {
  if (!source) return null;
  const { backend, mock } = source;
  if (backend === "mock" || backend === "sim" || backend === "real") return { kind: backend, reported: true };
  if (mock === true) return { kind: "mock", reported: false };
  if (mock === false) return { kind: "real", reported: false };
  return null;
}

/** The badge text. */
export const BACKEND_BADGE: Record<RobotBackend, string> = { real: "REAL", sim: "SIM", mock: "MOCK" };

/** One sentence on what a command does on this backend. */
export function backendDescription({ kind, reported }: BackendInfo): string {
  if (kind === "sim") return "Simulation (MuJoCo): commands move a simulated arm, not the physical one";
  if (kind === "mock") return "Mock backend: no motors move";
  return reported
    ? "Real arm: commands move the physical motors"
    : "Real arm assumed: this server does not say whether it is simulated";
}

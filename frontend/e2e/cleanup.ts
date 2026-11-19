import type { APIResponse } from "@playwright/test";
import {
  E2E_DATASET,
  explicitMockStatus,
  json,
  object,
  requireInstalledE2EIdentity,
  type E2EApi,
  type JsonObject,
} from "./contract";
import { dispatchInstalledMutation } from "./mutation-gate";

const ACTIVE_JOBS = new Set(["pending", "queued", "running"]);
export const TERMINAL_JOBS = new Set(["succeeded", "failed", "cancelled", "interrupted", "corrupt"]);
const TERMINAL_EVAL = new Set(["idle", "done", "aborted", "failed"]);

export interface CleanupJob {
  id: string;
  repo_id: string;
  status: string;
}

export interface CleanupSnapshot {
  status: JsonObject;
  control: JsonObject;
  evaluation: JsonObject;
  jobs: CleanupJob[];
}

function asError(reason: unknown, label: string): Error {
  return reason instanceof Error ? new Error(`${label}: ${reason.message}`) : new Error(`${label} failed`);
}

async function requireOk(response: APIResponse, label: string): Promise<void> {
  if (!response.ok()) throw new Error(`${label} returned HTTP ${response.status()}`);
}

export async function cleanupJobs(api: E2EApi): Promise<CleanupJob[]> {
  const value = await json(await api.get("/api/training/jobs"), "cleanup training jobs");
  if (!Array.isArray(value)) throw new Error("cleanup training jobs response was not an array");
  return value.map((entry, index) => {
    const item = object(entry, `cleanup training job ${index}`);
    if (
      typeof item.id !== "string" ||
      typeof item.repo_id !== "string" ||
      typeof item.status !== "string"
    ) {
      throw new Error(`cleanup training job ${index} is malformed`);
    }
    return { id: item.id, repo_id: item.repo_id, status: item.status };
  });
}

/**
 * Every cleanup command is isolated: a 500 is recorded and the remaining
 * stop/cancel/collapse operations still run. No command is sent unless the
 * token-protected identity endpoint proves this is the mock harness now.
 */
export async function runBestEffortCleanup(api: E2EApi): Promise<Error[]> {
  try {
    await requireInstalledE2EIdentity(api);
  } catch (reason) {
    return [asError(reason, "cleanup authorization")];
  }

  const errors: Error[] = [];
  let foundJobs: CleanupJob[] = [];
  try {
    foundJobs = await cleanupJobs(api);
  } catch (reason) {
    errors.push(asError(reason, "list training jobs"));
  }

  const attempt = async (label: string, operation: () => Promise<APIResponse>) => {
    try {
      await requireOk(await dispatchInstalledMutation(api, operation), label);
    } catch (reason) {
      errors.push(asError(reason, label));
    }
  };

  await attempt("stop evaluation", () => api.post("/api/eval/stop"));
  await attempt("stop rollout", () => api.post("/api/models/rollout/stop"));
  await attempt("stop recording", () => api.post("/api/recording/stop"));
  for (const job of foundJobs) {
    if (job.repo_id === E2E_DATASET && ACTIVE_JOBS.has(job.status)) {
      await attempt(`cancel training job ${job.id}`, () =>
        api.post(`/api/training/jobs/${encodeURIComponent(job.id)}/cancel`),
      );
    }
  }
  // Stop endpoints Halt. Collapse must be last so the handoff is at rest with
  // confirmed torque off even when an earlier operation returned an error.
  await attempt("collapse to rest", () =>
    api.post("/api/robot/collapse", {
      data: { operator_id: "installed-ui-e2e-cleanup", reason: "installed UI E2E cleanup" },
    }),
  );
  return errors;
}

export async function readCleanupSnapshot(api: E2EApi): Promise<CleanupSnapshot> {
  await requireInstalledE2EIdentity(api);
  const status = await explicitMockStatus(api);
  const control = object(
    await json(await api.get("/api/control/state"), "cleanup control state"),
    "cleanup control state",
  );
  const evaluation = object(
    await json(await api.get("/api/eval/state"), "cleanup evaluation state"),
    "cleanup evaluation state",
  );
  return { status, control, evaluation, jobs: await cleanupJobs(api) };
}

export function cleanupProblems(snapshot: CleanupSnapshot): string[] {
  const { status, control, evaluation } = snapshot;
  const problems: string[] = [];
  if (status.backend !== "mock") problems.push("backend is not mock");
  if (status.mode !== "idle" || status.operation_owner !== null) problems.push("robot is not idle");
  if (status.stop_state !== "none") problems.push("a stop is still in progress");
  if (status.torque_state !== "off" || status.torque_unconfirmed !== false) {
    problems.push("torque-off is not confirmed");
  }
  if (status.estopped !== false || status.latch_persistence !== "saved") {
    problems.push("a latch is set or not durably saved");
  }
  if (control.armed !== false || control.safety_state !== "disarmed" || control.lease !== null) {
    problems.push("control is armed, tripped, or leased");
  }
  if (evaluation.active !== false || !TERMINAL_EVAL.has(String(evaluation.phase))) {
    problems.push("evaluation is not terminal");
  }
  const nonterminal = snapshot.jobs.filter((job) => !TERMINAL_JOBS.has(job.status));
  if (nonterminal.length) problems.push(`training jobs are not terminal: ${nonterminal.map((job) => job.id).join(", ")}`);
  return problems;
}

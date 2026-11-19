import { createHash } from "node:crypto";
import process from "node:process";
import type { APIRequestContext, APIResponse } from "@playwright/test";
import { E2E_VERSION_ENV, requiredExpectedSuiteVersion } from "./config";

export const E2E_DATASET = "e2e/installed_ui";
export const IDENTITY_PATH = "/api/e2e/identity";
const STAMP_PATH = "/so101-ui-stamp.json";
const SHA256 = /^[a-f0-9]{64}$/;
const MAX_STAMP_FILES = 512;

export type JsonObject = Record<string, unknown>;
export type E2EApi = Pick<APIRequestContext, "get" | "post">;

export interface InstalledE2EIdentity extends JsonObject {
  ok: true;
  kind: "so101.installed_e2e";
  schema_version: 1;
  backend: "mock";
  mock: true;
  version: string;
}

export function object(value: unknown, label: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} did not return a JSON object`);
  }
  return value as JsonObject;
}

export async function json(response: APIResponse, label: string): Promise<unknown> {
  if (!response.ok()) throw new Error(`${label} returned HTTP ${response.status()}`);
  try {
    return await response.json();
  } catch {
    throw new Error(`${label} did not return JSON`);
  }
}

/** The first request of every run and the authorization check before a browser write. */
export async function requireInstalledE2EIdentity(api: E2EApi): Promise<InstalledE2EIdentity> {
  const expectedVersion = requiredExpectedSuiteVersion(process.env[E2E_VERSION_ENV]);
  const response = await api.get(IDENTITY_PATH);
  const identity = object(await json(response, "installed E2E identity"), "installed E2E identity");
  if (
    identity.ok !== true ||
    identity.kind !== "so101.installed_e2e" ||
    identity.schema_version !== 1 ||
    identity.backend !== "mock" ||
    identity.mock !== true ||
    identity.version !== expectedVersion
  ) {
    throw new Error("installed E2E identity did not prove a token-authorized mock harness");
  }
  return identity as InstalledE2EIdentity;
}

function publicPath(path: string): string {
  return `/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function digest(body: Buffer): string {
  return createHash("sha256").update(body).digest("hex");
}

/**
 * Prove the target serves a complete production build. Vite has no stamp, and
 * its SPA fallback is HTML, so it is rejected before status/control reads.
 */
export async function verifyInstalledUiStamp(api: E2EApi): Promise<void> {
  const stampResponse = await api.get(STAMP_PATH);
  const stamp = object(await json(stampResponse, "installed UI stamp"), "installed UI stamp");
  const files = object(stamp.files, "installed UI stamp files");
  const entries = Object.entries(files).sort(([left], [right]) => left.localeCompare(right));
  if (
    stamp.format !== 1 ||
    typeof stamp.sources !== "string" ||
    !SHA256.test(stamp.sources) ||
    entries.length < 2 ||
    entries.length > MAX_STAMP_FILES ||
    !entries.some(([path]) => path.startsWith("assets/"))
  ) {
    throw new Error("installed UI stamp is not a bounded production-build manifest");
  }

  for (const [path, expected] of entries) {
    if (
      typeof expected !== "string" ||
      !SHA256.test(expected) ||
      path.startsWith("/") ||
      path.split("/").some((part) => !part || part === "." || part === "..")
    ) {
      throw new Error("installed UI stamp contains an invalid file entry");
    }
    const response = await api.get(path === "index.html" ? "/" : publicPath(path));
    if (!response.ok()) throw new Error(`installed UI file ${path} returned HTTP ${response.status()}`);
    if (digest(await response.body()) !== expected) {
      throw new Error(`installed UI file ${path} does not match its build stamp`);
    }
  }
  if (!("index.html" in files)) throw new Error("installed UI stamp does not contain index.html");
}

export async function explicitMockStatus(api: E2EApi): Promise<JsonObject> {
  const status = object(
    await json(await api.get("/api/robot/status"), "robot status"),
    "robot status",
  );
  if (status.backend !== "mock") {
    throw new Error(
      `refusing installed UI E2E mutations: robot status must report backend=mock, got ${JSON.stringify(status.backend)}`,
    );
  }
  return status;
}

/** Read-only freshness checks; caller must establish identity and UI stamp first. */
export async function assertFreshMockSuite(api: E2EApi): Promise<void> {
  const status = await explicitMockStatus(api);
  const health = object(await json(await api.get("/api/health"), "health"), "health");
  if (health.backend !== "mock") {
    throw new Error(
      `health disagrees with robot status: expected backend=mock, got ${JSON.stringify(health.backend)}`,
    );
  }
  if (
    status.mode !== "idle" ||
    status.operation_owner !== null ||
    status.estopped !== false ||
    status.torque_unconfirmed !== false ||
    status.torque_state !== "off" ||
    status.stop_state !== "none" ||
    status.latch_persistence !== "saved"
  ) {
    throw new Error(`mock suite is not idle, torque-off, and durably unlatched: ${JSON.stringify(status)}`);
  }

  const control = object(
    await json(await api.get("/api/control/state"), "control state"),
    "control state",
  );
  // A lease is a blocker even when its wall-clock expires_at is in the past:
  // only the server may expire authority and publish the resulting stop.
  if (control.armed !== false || control.safety_state !== "disarmed" || control.lease !== null) {
    throw new Error(`mock control plane is not freshly disarmed: ${JSON.stringify(control)}`);
  }

  const evaluation = object(
    await json(await api.get("/api/eval/state"), "evaluation state"),
    "evaluation state",
  );
  if (evaluation.active !== false) throw new Error("an evaluation is already active");

  for (const [path, label] of [
    ["/api/training/jobs", "training jobs"],
    ["/api/datasets", "datasets"],
    ["/api/models", "models"],
  ] as const) {
    const items = await json(await api.get(path), label);
    if (!Array.isArray(items)) throw new Error(`${label} did not return a JSON array`);
    if (items.length) {
      throw new Error(`${label} are not empty; the installed harness must provide a fresh workspace`);
    }
  }
}

/** Identity must be the first request; Vite/stamp and freshness checks follow. */
export async function verifyFreshInstalledSuite(api: E2EApi): Promise<void> {
  await requireInstalledE2EIdentity(api);
  await verifyInstalledUiStamp(api);
  await assertFreshMockSuite(api);
}

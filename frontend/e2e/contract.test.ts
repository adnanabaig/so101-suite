import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { APIResponse, FullConfig } from "@playwright/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { redactTokenArtifacts } from "./artifact-audit";
import { runBestEffortCleanup } from "./cleanup";
import {
  artifactsDirectory,
  assertSafeDiagnosticsEnv,
  e2eHeaders,
  E2E_TOKEN_HEADER,
  E2E_VERSION_ENV,
  requiredExpectedSuiteVersion,
  requiredE2EToken,
} from "./config";
import {
  E2E_DATASET,
  IDENTITY_PATH,
  requireInstalledE2EIdentity,
  verifyFreshInstalledSuite,
  type E2EApi,
} from "./contract";
import { assertFixedFullConfig, type FixedConfigExpected } from "./full-config";
import { dispatchInstalledMutation } from "./mutation-gate";
import { browserMetadata, fixedChildEnvironment } from "./run-installed-ui.mjs";

const TOKEN = "url_safe_ephemeral_nonce_with_32_chars";
const EXPECTED_VERSION = "0.2.2rc5";
const INDEX = Buffer.from("<!doctype html><div id=\"root\"></div>");
const ASSET = Buffer.from("console.log('installed');");
const digest = (body: Buffer) => createHash("sha256").update(body).digest("hex");
const IDENTITY = {
  ok: true,
  kind: "so101.installed_e2e",
  schema_version: 1,
  backend: "mock",
  mock: true,
  version: EXPECTED_VERSION,
};
const STAMP = {
  format: 1,
  sources: "a".repeat(64),
  git_head: null,
  files: { "assets/app.js": digest(ASSET), "index.html": digest(INDEX) },
};

function response(status: number, value: unknown, raw?: Buffer): APIResponse {
  const body = raw ?? Buffer.from(JSON.stringify(value));
  return {
    ok: () => status >= 200 && status < 300,
    status: () => status,
    json: async () => JSON.parse(body.toString("utf8")) as unknown,
    body: async () => body,
  } as APIResponse;
}

type Handler = (method: "GET" | "POST", path: string) => APIResponse;

class FakeApi {
  readonly requests: Array<{ method: "GET" | "POST"; path: string }> = [];

  constructor(private readonly handler: Handler) {}

  readonly get = async (path: string) => {
    this.requests.push({ method: "GET", path });
    return this.handler("GET", path);
  };

  readonly post = async (path: string) => {
    this.requests.push({ method: "POST", path });
    return this.handler("POST", path);
  };

  asApi(): E2EApi {
    return this as unknown as E2EApi;
  }
}

function freshHandler(overrides: Record<string, APIResponse> = {}): Handler {
  const defaults: Record<string, APIResponse> = {
    [IDENTITY_PATH]: response(200, IDENTITY),
    "/so101-ui-stamp.json": response(200, STAMP),
    "/": response(200, null, INDEX),
    "/assets/app.js": response(200, null, ASSET),
    "/api/robot/status": response(200, {
      backend: "mock",
      mode: "idle",
      operation_owner: null,
      estopped: false,
      torque_unconfirmed: false,
      torque_state: "off",
      stop_state: "none",
      latch_persistence: "saved",
    }),
    "/api/health": response(200, { backend: "mock" }),
    "/api/control/state": response(200, { armed: false, safety_state: "disarmed", lease: null }),
    "/api/eval/state": response(200, { active: false, phase: "idle" }),
    "/api/training/jobs": response(200, []),
    "/api/datasets": response(200, []),
    "/api/models": response(200, []),
  };
  return (_method, path) => overrides[path] ?? defaults[path] ?? response(200, { ok: true });
}

const roots: string[] = [];
beforeEach(() => vi.stubEnv(E2E_VERSION_ENV, EXPECTED_VERSION));
afterEach(() => {
  vi.unstubAllEnvs();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("installed E2E configuration", () => {
  it("requires the harness token and uses exactly its protected header", () => {
    expect(() => requiredE2EToken(undefined)).toThrow(/installed-artifact harness/);
    expect(() => requiredE2EToken("not valid")).toThrow(/URL-safe/);
    expect(e2eHeaders(TOKEN)).toEqual({ [E2E_TOKEN_HEADER]: TOKEN });
    expect(requiredExpectedSuiteVersion(EXPECTED_VERSION)).toBe(EXPECTED_VERSION);
    expect(() => requiredExpectedSuiteVersion(undefined)).toThrow(/artifact harness/);
    expect(() => assertSafeDiagnosticsEnv({ DEBUG: "pw:api" })).toThrow(/text logs/);
  });

  it("refuses a nonempty artifact leaf without removing its sentinel", () => {
    const root = mkdtempSync(join(tmpdir(), "so101-e2e-artifacts-"));
    roots.push(root);
    const target = join(root, "playwright");
    mkdirSync(target, { mode: 0o700 });
    const sentinel = join(target, "KEEP");
    writeFileSync(sentinel, "operator evidence\n");

    expect(() => artifactsDirectory(target)).toThrow(/new or empty/);
    expect(readFileSync(sentinel, "utf8")).toBe("operator evidence\n");
  });

  it("accepts only the harness-created empty owner-only artifact leaf", () => {
    const root = mkdtempSync(join(tmpdir(), "so101-e2e-empty-artifacts-"));
    roots.push(root);
    const target = join(root, "playwright");
    mkdirSync(target, { mode: 0o700 });
    expect(artifactsDirectory(target)).toBe(target);
  });

  it("redacts an accidental token and proves no artifact retains it", () => {
    const root = mkdtempSync(join(tmpdir(), "so101-e2e-redaction-"));
    roots.push(root);
    const diagnostic = join(root, "error-context.md");
    writeFileSync(diagnostic, `request header: ${TOKEN}\n`);

    expect(redactTokenArtifacts(root, TOKEN)).toEqual(["error-context.md"]);
    expect(readFileSync(diagnostic, "utf8")).not.toContain(TOKEN);
    expect(redactTokenArtifacts(root, TOKEN)).toEqual([]);
  });

  it("strips inherited Playwright/debug overrides but keeps harness inputs", () => {
    expect(
      fixedChildEnvironment({
        SO101_E2E_TOKEN: TOKEN,
        SO101_E2E_EXPECTED_VERSION: EXPECTED_VERSION,
        PLAYWRIGHT_ARTIFACTS_DIR: "/evidence/mock/playwright",
        PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH: "/browser",
        PLAYWRIGHT_JSON_OUTPUT_NAME: "/unsafe/report.json",
        PW_TEST_SOURCE_TRANSFORM: "unsafe",
        PWDEBUG: "1",
        DEBUG: "pw:*",
        TEST_WORKER_INDEX: "9",
      }),
    ).toEqual({
      SO101_E2E_TOKEN: TOKEN,
      SO101_E2E_EXPECTED_VERSION: EXPECTED_VERSION,
      SO101_E2E_FIXED_LAUNCH: "1",
      PLAYWRIGHT_ARTIFACTS_DIR: "/evidence/mock/playwright",
      PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH: "/browser",
    });
  });

  it("publishes token-free system browser path and version metadata", async () => {
    const metadata = await browserMetadata({
      SO101_E2E_TOKEN: TOKEN,
      PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH: process.execPath,
      PATH: process.env.PATH,
    });
    expect(metadata).toMatchObject({
      schema_version: 1,
      kind: "so101.e2e_browser",
      source: "system_override",
      executable_path: process.execPath,
    });
    expect(metadata.version).toMatch(/^v\d+/);
    expect(JSON.stringify(metadata)).not.toContain(TOKEN);
  });

  it("documents the Makefile's fixed UI opt-in instead of an arbitrary command", () => {
    const readme = readFileSync(join(__dirname, "..", "README.md"), "utf8");
    const makefile = readFileSync(join(__dirname, "..", "..", "Makefile"), "utf8");
    const contributing = readFileSync(join(__dirname, "..", "..", "CONTRIBUTING.md"), "utf8");
    expect(readme).toContain("INSTALLED_E2E_UI=1");
    expect(readme).not.toContain("INSTALLED_E2E_UI_COMMAND");
    expect(makefile).toContain("INSTALLED_E2E_UI ?= 0");
    expect(makefile).toContain("--ui-e2e");
    expect(contributing).not.toContain("Both rehearsals\ncarry an ephemeral");
    expect(contributing).not.toContain("failure traces and screenshots");
    expect(contributing).toContain("Mock UI requests\ncarry an ephemeral");
  });

  it.each(["--trace=on", "--reporter=blob", "--output=/tmp/out", "--config=other.ts", "--retries=1", "--repeat-each=2"])(
    "fixed launcher rejects the %s override before Playwright starts",
    (flag) => {
      const launcher = join(__dirname, "run-installed-ui.mjs");
      const result = spawnSync(process.execPath, [launcher, flag], { encoding: "utf8" });
      expect(result.status).toBe(2);
      expect(result.stderr).toContain("accepts no command-line overrides");
      expect(result.stdout).not.toContain("SO101_E2E_BROWSER_METADATA");
    },
  );
});

describe("resolved Playwright configuration", () => {
  const configFile = join(__dirname, "..", "playwright.config.ts");
  const auditReporter = join(__dirname, "token-audit-reporter.ts");
  const expected: FixedConfigExpected = {
    argv: [process.execPath, "/frontend/node_modules/@playwright/test/cli.js", "test", "--config", configFile],
    baseURL: "http://127.0.0.1:8000",
    configFile,
    outputDir: "/evidence/mock/playwright",
    token: TOKEN,
  };

  function fullConfig(): FullConfig {
    return {
      configFile,
      argv: [...expected.argv],
      workers: 1,
      fullyParallel: false,
      shard: null,
      reporter: [["line"], [auditReporter]],
      projects: [
        {
          name: "chromium",
          retries: 0,
          repeatEach: 1,
          outputDir: expected.outputDir,
          use: {
            baseURL: expected.baseURL,
            trace: "off",
            screenshot: "only-on-failure",
            video: "off",
            extraHTTPHeaders: { [E2E_TOKEN_HEADER]: TOKEN },
          },
        },
      ],
    } as unknown as FullConfig;
  }

  it("accepts only the fixed effective configuration", () => {
    expect(() => assertFixedFullConfig(fullConfig(), expected)).not.toThrow();
  });

  it.each([
    ["trace", (config: FullConfig): void => { config.projects[0].use.trace = "on"; }],
    ["reporter", (config: FullConfig): void => { config.reporter = [["blob"]]; }],
    ["output", (config: FullConfig): void => { config.projects[0].outputDir = "/tmp/other"; }],
    ["config", (config: FullConfig): void => {
      config.configFile = join(__dirname, "..", "package.json");
    }],
    ["retries", (config: FullConfig): void => { config.projects[0].retries = 1; }],
    ["repeat-each", (config: FullConfig): void => { config.projects[0].repeatEach = 2; }],
  ] as const)("rejects an effective %s override", (_name, change) => {
    const config = fullConfig();
    change(config);
    expect(() => assertFixedFullConfig(config, expected)).toThrow(/effective .* override/);
  });
});

describe("identity-first installed UI gate", () => {
  it.each([
    ["missing stamp", response(404, { detail: "not found" })],
    ["Vite SPA fallback", response(200, null, Buffer.from("<!doctype html><title>Vite</title>"))],
  ])("rejects %s before reading robot status", async (_label, badStamp) => {
    const api = new FakeApi(freshHandler({ "/so101-ui-stamp.json": badStamp }));
    await expect(verifyFreshInstalledSuite(api.asApi())).rejects.toThrow(/installed UI stamp/);
    expect(api.requests).toEqual([
      { method: "GET", path: IDENTITY_PATH },
      { method: "GET", path: "/so101-ui-stamp.json" },
    ]);
  });

  it("rejects an expired real control lease without dispatching a write", async () => {
    const api = new FakeApi(
      freshHandler({
        "/api/control/state": response(200, {
          armed: false,
          safety_state: "disarmed",
          lease: {
            source: "real-arm-browser",
            expires_at: "2000-01-01T00:00:00Z",
            ttl_seconds: 1,
            server_time: "2000-01-01T00:01:00Z",
          },
        }),
      }),
    );
    await expect(verifyFreshInstalledSuite(api.asApi())).rejects.toThrow(/not freshly disarmed/);
    expect(api.requests[0]).toEqual({ method: "GET", path: IDENTITY_PATH });
    expect(api.requests.some(({ method }) => method === "POST")).toBe(false);
  });
});

describe("mutation and cleanup authorization", () => {
  it("blocks zero-dispatch mutation after the URL switches to a real server", async () => {
    let identityCalls = 0;
    const api = new FakeApi((_method, path) => {
      if (path !== IDENTITY_PATH) return response(500, {});
      identityCalls += 1;
      return identityCalls === 1 ? response(200, IDENTITY) : response(403, { detail: "forbidden" });
    });
    await requireInstalledE2EIdentity(api.asApi());
    const dispatch = vi.fn(async () => "sent");
    await expect(dispatchInstalledMutation(api.asApi(), dispatch)).rejects.toThrow(/HTTP 403/);
    expect(dispatch).not.toHaveBeenCalled();
    expect(api.requests.every(({ method }) => method === "GET")).toBe(true);
  });

  it("continues every cleanup step and aggregates a 500", async () => {
    const api = new FakeApi((method, path) => {
      if (method === "GET" && path === IDENTITY_PATH) return response(200, IDENTITY);
      if (method === "GET" && path === "/api/training/jobs") {
        return response(200, [{ id: "job-1", repo_id: E2E_DATASET, status: "running" }]);
      }
      if (method === "POST" && path === "/api/eval/stop") return response(500, {});
      return response(200, { ok: true });
    });
    const errors = await runBestEffortCleanup(api.asApi());
    expect(errors.map((error) => error.message)).toEqual([
      "stop evaluation: stop evaluation returned HTTP 500",
    ]);
    expect(api.requests.filter(({ method }) => method === "POST").map(({ path }) => path)).toEqual([
      "/api/eval/stop",
      "/api/models/rollout/stop",
      "/api/recording/stop",
      "/api/training/jobs/job-1/cancel",
      "/api/robot/collapse",
    ]);
    expect(api.requests.filter(({ path }) => path === IDENTITY_PATH)).toHaveLength(6);
  });

  it("sends no cleanup command when token identity is gone", async () => {
    const api = new FakeApi(() => response(403, { detail: "forbidden" }));
    const errors = await runBestEffortCleanup(api.asApi());
    expect(errors).toHaveLength(1);
    expect(api.requests).toEqual([{ method: "GET", path: IDENTITY_PATH }]);
  });

  it("reauthorizes every cleanup mutation after an identity switch", async () => {
    let identityCalls = 0;
    const api = new FakeApi((method, path) => {
      if (method === "GET" && path === IDENTITY_PATH) {
        identityCalls += 1;
        return identityCalls === 1 ? response(200, IDENTITY) : response(403, { detail: "forbidden" });
      }
      if (method === "GET" && path === "/api/training/jobs") return response(200, []);
      return response(200, { ok: true });
    });
    const errors = await runBestEffortCleanup(api.asApi());
    expect(errors).toHaveLength(4);
    expect(api.requests.some(({ method }) => method === "POST")).toBe(false);
    expect(identityCalls).toBe(5);
  });
});

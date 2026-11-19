import { realpathSync } from "node:fs";
import { resolve } from "node:path";
import type { FullConfig } from "@playwright/test";
import { E2E_TOKEN_HEADER } from "./config";

export interface FixedConfigExpected {
  argv: readonly string[];
  baseURL: string;
  configFile: string;
  outputDir: string;
  token: string;
}

function mode(value: unknown): unknown {
  if (typeof value === "object" && value !== null && "mode" in value) {
    return (value as { mode?: unknown }).mode;
  }
  return value;
}

function reporterName(name: string): string {
  return name.replace(/\\/g, "/");
}

/** Validate resolved settings so Playwright CLI flags cannot weaken the gate. */
export function assertFixedFullConfig(config: FullConfig, expected: FixedConfigExpected): void {
  const fail = (setting: string): never => {
    throw new Error(`installed UI E2E refused an effective ${setting} override`);
  };
  if (!config.configFile || realpathSync(config.configFile) !== realpathSync(expected.configFile)) {
    fail("config file");
  }
  if (
    config.argv.length !== expected.argv.length ||
    config.argv.some((argument, index) => argument !== expected.argv[index])
  ) {
    fail("custom argument");
  }
  if (config.projects.length !== 1) fail("project");
  if (config.workers !== 1 || config.fullyParallel || config.shard !== null) fail("parallelism");

  const project = config.projects[0];
  if (project.name !== "chromium") fail("project");
  if (project.retries !== 0) fail("retries");
  if (project.repeatEach !== 1) fail("repeat-each");
  if (resolve(project.outputDir) !== resolve(expected.outputDir)) fail("output directory");
  if (project.use.baseURL !== expected.baseURL) fail("base URL");
  if (mode(project.use.trace) !== "off") fail("trace");
  if (mode(project.use.screenshot) !== "only-on-failure") fail("screenshot");
  if (mode(project.use.video) !== "off") fail("video");

  const headers = project.use.extraHTTPHeaders;
  if (
    typeof headers !== "object" ||
    headers === null ||
    Array.isArray(headers) ||
    Object.keys(headers).length !== 1 ||
    headers[E2E_TOKEN_HEADER] !== expected.token
  ) {
    fail("request headers");
  }

  const reporters = config.reporter.map(([name]) => reporterName(name));
  const line = reporters.filter((name) => name === "line" || name.endsWith("/line.js"));
  const audit = reporters.filter((name) => name.endsWith("/e2e/token-audit-reporter.ts"));
  if (reporters.length !== 2 || line.length !== 1 || audit.length !== 1) fail("reporter");
}

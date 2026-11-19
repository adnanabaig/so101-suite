import process from "node:process";
import { chmodSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { request, type FullConfig } from "@playwright/test";
import {
  E2E_ARTIFACTS_ENV,
  E2E_TOKEN_ENV,
  E2E_TOKEN_HEADER,
  installedSuiteUrl,
  requiredE2EToken,
} from "./config";
import { verifyFreshInstalledSuite } from "./contract";
import { assertFixedFullConfig } from "./full-config";

export default async function globalSetup(config: FullConfig): Promise<void> {
  const baseURL = installedSuiteUrl(process.env.SO101_E2E_BASE_URL);
  const token = requiredE2EToken(process.env[E2E_TOKEN_ENV]);
  const artifactSetting = process.env[E2E_ARTIFACTS_ENV];
  if (!artifactSetting) throw new Error(`${E2E_ARTIFACTS_ENV} is not configured`);
  // The coordinator already validated the empty 0700 leaf synchronously in
  // playwright.config.ts. Playwright removes it before globalSetup.
  const outputDir = resolve(artifactSetting);
  const configFile = fileURLToPath(new URL("../playwright.config.ts", import.meta.url));
  assertFixedFullConfig(config, {
    argv: [
      process.execPath,
      fileURLToPath(new URL("../node_modules/@playwright/test/cli.js", import.meta.url)),
      "test",
      "--config",
      configFile,
    ],
    baseURL,
    token,
    outputDir,
    configFile,
  });
  // Playwright removes outputDir before globalSetup and would recreate it
  // with the process umask. Restore the already-validated leaf as 0700 before
  // a worker or failure artifact can write into it.
  mkdirSync(outputDir, { mode: 0o700, recursive: true });
  chmodSync(outputDir, 0o700);
  const api = await request.newContext({
    baseURL,
    extraHTTPHeaders: { [E2E_TOKEN_HEADER]: token },
  });
  try {
    // Identity is deliberately the first network request. Everything in this
    // bootstrap is GET-only; refusal happens before browser heartbeats/writes.
    await verifyFreshInstalledSuite(api);
  } finally {
    await api.dispose();
  }
}

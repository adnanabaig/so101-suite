import process from "node:process";
import { defineConfig, devices } from "@playwright/test";
import {
  artifactsDirectory,
  assertSafeDiagnosticsEnv,
  assertFixedLauncher,
  e2eHeaders,
  E2E_ARTIFACTS_ENV,
  E2E_TOKEN_ENV,
  FIXED_LAUNCH_ENV,
  installedSuiteUrl,
  requiredE2EToken,
} from "./e2e/config";

assertFixedLauncher(process.env[FIXED_LAUNCH_ENV]);
assertSafeDiagnosticsEnv(process.env);
const baseURL = installedSuiteUrl(process.env.SO101_E2E_BASE_URL);
const e2eToken = requiredE2EToken(process.env[E2E_TOKEN_ENV]);
const chromiumExecutable = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH?.trim() || undefined;
// The coordinator validates before Playwright clears outputDir. Worker
// processes reload this config after the coordinator has created run files;
// they repeat every path check but may observe that known leaf as nonempty.
const outputDir = artifactsDirectory(process.env[E2E_ARTIFACTS_ENV], {
  allowMissing: process.env.TEST_WORKER_INDEX !== undefined,
  allowNonEmpty: process.env.TEST_WORKER_INDEX !== undefined,
});

export default defineConfig({
  testDir: "./e2e",
  testMatch: "installed-ui.spec.ts",
  globalSetup: "./e2e/global-setup.ts",
  outputDir,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 120_000,
  expect: { timeout: 20_000 },
  reporter: [["line"], ["./e2e/token-audit-reporter.ts"]],
  use: {
    baseURL,
    extraHTTPHeaders: e2eHeaders(e2eToken),
    // Traces store extraHTTPHeaders in cleartext. The ephemeral token must
    // never enter evidence; failure screenshots and redacted diagnostics stay.
    trace: "off",
    screenshot: "only-on-failure",
    video: "off",
    actionTimeout: 20_000,
    navigationTimeout: 30_000,
    launchOptions: chromiumExecutable ? { executablePath: chromiumExecutable } : undefined,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});

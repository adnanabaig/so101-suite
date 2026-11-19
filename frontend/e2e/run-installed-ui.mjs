import { existsSync, statSync } from "node:fs";
import { isAbsolute } from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const FIXED_LAUNCH_ENV = "SO101_E2E_FIXED_LAUNCH";
const ALLOWED_PLAYWRIGHT_ENV = new Set([
  "PLAYWRIGHT_ARTIFACTS_DIR",
  "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH",
]);

export function fixedChildEnvironment(source) {
  const clean = {};
  for (const [name, value] of Object.entries(source)) {
    if (value === undefined) continue;
    if (name === "DEBUG" || name === "PWDEBUG" || name === "NODE_DEBUG") continue;
    if (name === "TEST_WORKER_INDEX" || name === "TEST_PARALLEL_INDEX") continue;
    if (name.startsWith("PW_")) continue;
    if (name.startsWith("PLAYWRIGHT_") && !ALLOWED_PLAYWRIGHT_ENV.has(name)) continue;
    clean[name] = value;
  }
  clean[FIXED_LAUNCH_ENV] = "1";
  return clean;
}

function browserVersion(executablePath, source) {
  const stat = isAbsolute(executablePath) && existsSync(executablePath) ? statSync(executablePath) : null;
  if (!stat?.isFile() || (stat.mode & 0o111) === 0) {
    throw new Error("the configured E2E browser executable is not an absolute executable file");
  }
  const env = fixedChildEnvironment(source);
  delete env.SO101_E2E_TOKEN;
  const result = spawnSync(executablePath, ["--version"], {
    encoding: "utf8",
    env,
    timeout: 10_000,
  });
  const version = `${result.stdout ?? ""}${result.stderr ?? ""}`.trim();
  if (result.status !== 0 || !version || /[\r\n]/.test(version)) {
    throw new Error("could not read the E2E browser version");
  }
  return version;
}

export async function browserMetadata(source) {
  const override = source.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH?.trim();
  let executablePath;
  let sourceKind;
  if (override) {
    executablePath = override;
    sourceKind = "system_override";
  } else {
    const { chromium } = await import("playwright");
    executablePath = chromium.executablePath();
    sourceKind = "playwright_pinned";
  }
  const metadata = {
    schema_version: 1,
    kind: "so101.e2e_browser",
    source: sourceKind,
    executable_path: executablePath,
    version: browserVersion(executablePath, source),
  };
  const serialized = JSON.stringify(metadata);
  const token = source.SO101_E2E_TOKEN;
  if (token && serialized.includes(token)) throw new Error("browser metadata contained protected data");
  return metadata;
}

export async function main(args = process.argv.slice(2), source = process.env) {
  if (args.length) {
    process.stderr.write("installed UI E2E accepts no command-line overrides\n");
    return 2;
  }
  const metadata = await browserMetadata(source);
  process.stdout.write(`SO101_E2E_BROWSER_METADATA=${JSON.stringify(metadata)}\n`);

  const cli = fileURLToPath(new URL("../node_modules/@playwright/test/cli.js", import.meta.url));
  const config = fileURLToPath(new URL("../playwright.config.ts", import.meta.url));
  const result = spawnSync(process.execPath, [cli, "test", "--config", config], {
    env: fixedChildEnvironment(source),
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  return result.status ?? 1;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    process.exitCode = await main();
  } catch (reason) {
    const detail = reason instanceof Error ? reason.message : "launcher failed";
    process.stderr.write(`installed UI E2E launcher: ${detail}\n`);
    process.exitCode = 1;
  }
}

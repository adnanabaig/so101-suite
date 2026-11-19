import { existsSync, lstatSync, readdirSync, realpathSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, parse, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const E2E_TOKEN_HEADER = "X-SO101-E2E-Token";
export const E2E_TOKEN_ENV = "SO101_E2E_TOKEN";
export const E2E_VERSION_ENV = "SO101_E2E_EXPECTED_VERSION";
export const E2E_ARTIFACTS_ENV = "PLAYWRIGHT_ARTIFACTS_DIR";
export const FIXED_LAUNCH_ENV = "SO101_E2E_FIXED_LAUNCH";

const FRONTEND = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function sameOrAncestor(candidate: string, path: string): boolean {
  const rest = relative(candidate, path);
  return rest === "" || (!rest.startsWith(`..${sep}`) && rest !== ".." && !isAbsolute(rest));
}

export function installedSuiteUrl(raw: string | undefined): string {
  if (!raw) {
    throw new Error(
      "SO101_E2E_BASE_URL is required; run this check only through the installed-artifact harness",
    );
  }
  const url = new URL(raw);
  if (!/^https?:$/.test(url.protocol)) throw new Error("SO101_E2E_BASE_URL must use http or https");
  if (url.username || url.password || url.search || url.hash || url.pathname !== "/") {
    throw new Error(
      "SO101_E2E_BASE_URL must be an origin without credentials, a path, query, or fragment",
    );
  }
  return url.origin;
}

export function requiredE2EToken(raw: string | undefined): string {
  if (!raw || !/^[A-Za-z0-9_-]+$/.test(raw) || raw.length < 32 || raw.length > 512) {
    throw new Error(
      `${E2E_TOKEN_ENV} must be the URL-safe nonce supplied by the installed-artifact harness`,
    );
  }
  return raw;
}

export function e2eHeaders(raw: string | undefined): Record<string, string> {
  return { [E2E_TOKEN_HEADER]: requiredE2EToken(raw) };
}

export function requiredExpectedSuiteVersion(raw: string | undefined): string {
  if (!raw || raw.length > 128 || !/^[A-Za-z0-9](?:[A-Za-z0-9._+!-]*[A-Za-z0-9])?$/.test(raw)) {
    throw new Error(
      `${E2E_VERSION_ENV} must be the installed wheel version supplied by the artifact harness`,
    );
  }
  return raw;
}

export function assertSafeDiagnosticsEnv(env: Record<string, string | undefined>): void {
  const unsafe = ["DEBUG", "PWDEBUG", "NODE_DEBUG"].filter((name) => Boolean(env[name]));
  if (unsafe.length) {
    throw new Error(
      `installed UI E2E requires ${unsafe.join(", ")} to be unset so request headers cannot enter text logs`,
    );
  }
}

export function assertFixedLauncher(value: string | undefined): void {
  if (value !== "1") {
    throw new Error("installed UI E2E must run through the fixed npm test:e2e launcher");
  }
}

/**
 * Validate before Playwright gets a chance to clear outputDir. The target must
 * be one new or empty leaf below an existing directory. Refusals do not write,
 * unlink, or traverse a symlink, so a sentinel proves cleanup never ran.
 */
export function artifactsDirectory(
  raw: string | undefined,
  options: {
    frontend?: string;
    home?: string;
    temp?: string;
    allowMissing?: boolean;
    allowNonEmpty?: boolean;
  } = {},
): string {
  if (!raw) {
    throw new Error(
      `${E2E_ARTIFACTS_ENV} is required; the installed-artifact harness owns the evidence directory`,
    );
  }
  if (!isAbsolute(raw)) throw new Error(`${E2E_ARTIFACTS_ENV} must be an absolute path`);

  const target = resolve(raw);
  const frontend = resolve(options.frontend ?? FRONTEND);
  const forbidden = new Set([
    parse(target).root,
    resolve(options.home ?? homedir()),
    resolve(options.temp ?? tmpdir()),
  ]);
  if (forbidden.has(target) || sameOrAncestor(target, frontend)) {
    throw new Error(`${E2E_ARTIFACTS_ENV} must name a dedicated leaf, not a broad directory`);
  }
  if (basename(target) !== "playwright") {
    throw new Error(`${E2E_ARTIFACTS_ENV} must name the harness's dedicated playwright leaf`);
  }

  if (!existsSync(target)) {
    if (options.allowMissing) return target;
    throw new Error(`${E2E_ARTIFACTS_ENV} must be created by the installed-artifact harness`);
  }
  const stat = lstatSync(target);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${E2E_ARTIFACTS_ENV} must be a real directory, not a file or symlink`);
  }
  if ((stat.mode & 0o077) !== 0) {
    throw new Error(`${E2E_ARTIFACTS_ENV} must have owner-only permissions`);
  }
  if (!options.allowNonEmpty && readdirSync(target).length !== 0) {
    throw new Error(`${E2E_ARTIFACTS_ENV} must be new or empty; existing evidence is preserved`);
  }
  return realpathSync(target);
}

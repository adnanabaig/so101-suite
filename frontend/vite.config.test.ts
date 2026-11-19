import { rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ConfigEnv, ProxyOptions, UserConfig } from "vite";
import { afterEach, describe, expect, it, vi } from "vitest";
import config from "./vite.config";

// A mode no real command uses, so a leftover file cannot move the dev proxy.
const MODE = "so101-proxy-test";
const ENV_FILE = join(__dirname, `.env.${MODE}.local`);

function proxyTarget(): string | undefined {
  const resolve = config as (env: ConfigEnv) => UserConfig;
  const proxy = resolve({ mode: MODE, command: "serve" }).server?.proxy ?? {};
  return (proxy["/api"] as ProxyOptions).target as string | undefined;
}

afterEach(() => {
  rmSync(ENV_FILE, { force: true });
  vi.unstubAllEnvs();
});

describe("the Vite dev proxy target", () => {
  it("defaults to the local API", () => {
    vi.stubEnv("VITE_DEV_API_TARGET", undefined);
    expect(proxyTarget()).toBe("http://localhost:8000");
  });

  it("reads VITE_DEV_API_TARGET from frontend/.env files", () => {
    vi.stubEnv("VITE_DEV_API_TARGET", undefined);
    writeFileSync(ENV_FILE, "VITE_DEV_API_TARGET=http://from-dotenv:9001\n");
    expect(proxyTarget()).toBe("http://from-dotenv:9001");
  });

  it("prefers the environment (Docker Compose) over the files", () => {
    writeFileSync(ENV_FILE, "VITE_DEV_API_TARGET=http://from-dotenv:9001\n");
    vi.stubEnv("VITE_DEV_API_TARGET", "http://api:8000");
    expect(proxyTarget()).toBe("http://api:8000");
  });
});

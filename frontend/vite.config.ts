/// <reference types="vitest/config" />
import type { ClientRequest, IncomingMessage } from "node:http";
import { defineConfig, loadEnv, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { so101BuildStamp } from "./build-stamp";

// This directory (frontend/): the build stamp hashes its sources, and the
// dev proxy reads its .env files. Vite's config loader and Vitest both
// provide __dirname (import.meta.url is not a file: URL under Vitest's jsdom).
const FRONTEND = __dirname;
const DEFAULT_DEV_API_TARGET = "http://localhost:8000";

/**
 * changeOrigin rewrites Host to the API target but leaves Origin as this dev
 * page's origin, which the server's C1 guard refuses (403, or WebSocket 1008)
 * unless it is listed in SO101_CORS_ORIGINS. A request whose Origin is this
 * dev server's own address is presented as same-origin with the target. Any
 * other Origin is forwarded unchanged, so a foreign page is still refused.
 * Vite's own allowedHosts check runs first, so DNS-rebinding names never get
 * here.
 */
function presentAsSameOrigin(target: string, proxyReq: ClientRequest, req: IncomingMessage): void {
  const { origin, host } = req.headers;
  if (!origin || !host || proxyReq.headersSent) return;
  if (origin !== `http://${host}` && origin !== `https://${host}`) return;
  proxyReq.setHeader("origin", new URL(target).origin);
}

function devProxy(target: string, ws: boolean): ProxyOptions {
  return {
    target,
    changeOrigin: true,
    ws,
    configure(proxy) {
      proxy.on("proxyReq", (proxyReq, req) => presentAsSameOrigin(target, proxyReq, req));
      if (ws) proxy.on("proxyReqWs", (proxyReq, req) => presentAsSameOrigin(target, proxyReq, req));
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Dev server only. The built UI is served by the robot server itself and
  // calls it with relative URLs, so production has no host or port anywhere.
  // VITE_DEV_API_TARGET comes from the environment (Docker Compose sets it),
  // else from frontend/.env, .env.local, .env.[mode] or .env.[mode].local.
  const devApiTarget =
    loadEnv(mode, FRONTEND, "VITE_DEV_API_TARGET").VITE_DEV_API_TARGET || DEFAULT_DEV_API_TARGET;
  return {
    plugins: [react(), tailwindcss(), so101BuildStamp(FRONTEND)],
    server: {
      proxy: {
        "/api": devProxy(devApiTarget, false),
        "/ws": devProxy(devApiTarget, true),
      },
    },
    build: {
      outDir: "dist",
      // Vite's default 500 kB chunk warning stays on: do not raise it. The 3D
      // stack loads only with the digital twin's lazy scene (F15), split so no
      // chunk crosses the limit. React gets its own group first; otherwise the
      // React Three Fiber group would absorb React (groups include their
      // dependencies), and the entry would have to import that 3D chunk.
      rolldownOptions: {
        output: {
          codeSplitting: {
            groups: [
              { name: "react-vendor", test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/, priority: 50 },
              { name: "three-core", test: /[\\/]node_modules[\\/]three[\\/]build[\\/]three\.core\.js$/, priority: 40 },
              { name: "three", test: /[\\/]node_modules[\\/]three[\\/]/, priority: 30 },
              {
                name: "r3f",
                test: /[\\/]node_modules[\\/](@react-three[\\/]fiber|react-reconciler|scheduler|its-fine|zustand|suspend-react|react-use-measure)[\\/]/,
                priority: 20,
              },
              { name: "drei", test: /[\\/]node_modules[\\/](@react-three[\\/]drei|three-stdlib)[\\/]/, priority: 10 },
            ],
          },
        },
      },
    },
    test: {
      environment: "jsdom",
      include: [
        "src/**/*.test.{ts,tsx}",
        "e2e/**/*.test.ts",
        "build-stamp.test.ts",
        "vite.config.test.ts",
      ],
      setupFiles: ["src/test/setup.ts"],
      restoreMocks: true,
      coverage: {
        provider: "v8",
        reporter: ["text", "json-summary", "html"],
        reportsDirectory: "coverage",
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/vite-env.d.ts", "src/main.tsx"],
      },
    },
  };
});

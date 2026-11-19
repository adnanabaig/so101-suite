import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  Shell,
  type FakeBackend,
} from "../test/harness";
import Settings from "./Settings";

let backend: FakeBackend;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  backend.route("GET", "/api/health", () =>
    jsonResponse({
      ok: true,
      version: "0.2.0",
      mode: "idle",
      mock: true,
      fps: 0,
      follower_connected: true,
      leader_connected: false,
      estopped: false,
      cameras: [],
      live_cameras: [],
      camera_frame_age_ms: {},
      operation_owner: null,
    }),
  );
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderSettings() {
  render(
    <Shell path="/settings">
      <Settings />
    </Shell>,
  );
  await advance(10);
}

describe("Settings", () => {
  test("has one page heading and opens the shortcut list from the page", async () => {
    await renderSettings();
    expect(screen.getByRole("heading", { level: 1, name: "Settings" })).toBeInTheDocument();
    const main = screen.getByRole("main");
    fireEvent.click(within(main).getByRole("button", { name: "Keyboard shortcuts" }));
    await advance(10);
    expect(screen.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeInTheDocument();
  });

  test("a health failure is shown with the server's message", async () => {
    backend.route("GET", "/api/health", () => jsonResponse({ detail: "health probe failed" }, 503));
    await renderSettings();
    expect(screen.getByText("Backend health could not be read")).toBeInTheDocument();
    expect(screen.getByText("health probe failed")).toBeInTheDocument();
  });
});

describe("Settings backend", () => {
  test("health names the backend; a simulated follower says so", async () => {
    backend.route("GET", "/api/health", () =>
      jsonResponse({
        ok: true,
        version: "0.2.0",
        mode: "idle",
        mock: false,
        backend: "sim",
        fps: 0,
        follower_connected: true,
        leader_connected: false,
        estopped: false,
        cameras: [],
        live_cameras: [],
        camera_frame_age_ms: {},
        operation_owner: null,
      }),
    );
    await renderSettings();
    const main = screen.getByRole("main");
    expect(within(main).getByText("sim")).toBeInTheDocument();
    expect(within(main).getByText("follower (simulated)")).toBeInTheDocument();
  });

  test("an older server's health falls back to mock", async () => {
    await renderSettings();
    expect(within(screen.getByRole("main")).getByText("mock", { selector: "span:not(.sr-only)" })).toBeInTheDocument();
  });
});

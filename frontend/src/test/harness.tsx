// Shared test doubles for the operator shell.
//
// Ported from the jsdom harness of the v0.1 frontend audit
// (docs/audit/v0.1/frontend.md): a tiny in-memory backend that mirrors the
// real routers' E-stop semantics (docs/audit/v0.1/safety.md), plus a
// controllable WebSocket double. Tests can add or override routes.
import type { ReactNode } from "react";
import { act } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import Layout from "../components/Layout";
import { TelemetryProvider } from "../hooks/useTelemetry";

export interface RecordedCall {
  method: string;
  url: string;
  path: string;
  body: unknown;
  headers: Record<string, string>;
}

export type RouteHandler = (call: RecordedCall) => Response | Promise<Response>;

export interface FakeBackend {
  estopped: boolean;
  torqueUnconfirmed: boolean;
  armed: boolean;
  safetyState: string;
  mode: string;
  /** Extra fields merged into GET /api/robot/status (e.g. C11 stop fields). */
  statusExtra: Record<string, unknown>;
  calls: RecordedCall[];
  /** The body GET /api/robot/status would return right now. */
  status: () => Record<string, unknown>;
  /** Add or replace a route. Keys are "METHOD /path" (query string ignored). */
  route: (method: string, path: string, handler: RouteHandler) => void;
  /** URLs of every POST so far, in order. */
  posts: () => string[];
  /** Recorded calls to one path (query string ignored). */
  callsTo: (path: string, method?: string) => RecordedCall[];
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function telemetryFrame(
  estopped: boolean,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    t: Date.now() / 1000,
    mode: "idle",
    estopped,
    fps: 0,
    observation: { "shoulder_pan.pos": 0 },
    action: {},
    recording: false,
    recording_saving: false,
    episode_frames: 0,
    sample_age_ms: 100,
    sample_seq: 1,
    observation_live: true,
    ...overrides,
  };
}

function headerRecord(init?: RequestInit): Record<string, string> {
  const out: Record<string, string> = {};
  new Headers(init?.headers).forEach((value, key) => {
    out[key] = value;
  });
  return out;
}

function parseBody(init?: RequestInit): unknown {
  if (typeof init?.body !== "string") return undefined;
  try {
    return JSON.parse(init.body) as unknown;
  } catch {
    return init.body;
  }
}

/** Install a fetch double that serves the safety-relevant routes. */
export function installFakeBackend(): FakeBackend {
  const routes = new Map<string, RouteHandler>();
  const backend: FakeBackend = {
    estopped: false,
    torqueUnconfirmed: false,
    armed: false,
    safetyState: "disarmed",
    mode: "idle",
    statusExtra: {},
    calls: [],
    status: () => status(),
    route: (method, path, handler) => {
      routes.set(`${method.toUpperCase()} ${path}`, handler);
    },
    posts: () => backend.calls.filter((call) => call.method === "POST").map((call) => call.url),
    callsTo: (path, method = "POST") =>
      backend.calls.filter((call) => call.path === path && call.method === method.toUpperCase()),
  };

  const status = () => ({
    mode: backend.mode,
    follower_connected: true,
    leader_connected: true,
    estopped: backend.estopped,
    torque_unconfirmed: backend.torqueUnconfirmed,
    active_model: null,
    mock: true,
    cameras: [],
    live_cameras: [],
    camera_frame_age_ms: {},
    operation_owner: null,
    fps: 0,
    ...backend.statusExtra,
  });
  const control = () => ({
    armed: backend.armed,
    safety_state: backend.estopped ? "estopped" : backend.safetyState,
    lease: null,
    watchdog_ms: 500,
    last_command_at: null,
  });

  backend.route("GET", "/api/robot/status", () => jsonResponse(status()));
  backend.route("GET", "/api/control/state", () => jsonResponse(control()));
  backend.route("POST", "/api/robot/estop", () => {
    backend.estopped = true;
    backend.armed = false;
    FakeWebSocket.broadcast(telemetryFrame(true));
    return jsonResponse({ ok: true, detail: "E-STOP latched" });
  });
  backend.route("POST", "/api/robot/estop/reset", () => {
    backend.estopped = false;
    FakeWebSocket.broadcast(telemetryFrame(false));
    return jsonResponse({ ok: true, detail: "E-stop reset" });
  });
  backend.route("POST", "/api/robot/stop", () => {
    backend.armed = false;
    return jsonResponse({ ok: true, detail: "stopped" });
  });
  backend.route("POST", "/api/robot/halt", () => jsonResponse({ ok: true, detail: "halted" }));
  backend.route("POST", "/api/robot/collapse", () =>
    jsonResponse({ ok: true, detail: "collapsing" }),
  );
  backend.route("POST", "/api/control/disarm", () => {
    backend.armed = false;
    return jsonResponse(control());
  });
  backend.route("POST", "/api/robot/operator-heartbeat", () =>
    jsonResponse({ ok: true, detail: "" }),
  );

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const method = (init?.method ?? "GET").toUpperCase();
    const path = url.split("?")[0];
    const call: RecordedCall = { method, url, path, body: parseBody(init), headers: headerRecord(init) };
    backend.calls.push(call);
    const handler = routes.get(`${method} ${path}`);
    if (handler) return handler(call);
    return jsonResponse({ detail: "Not Found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return backend;
}

type Listener = (event: { data: string }) => void;

/** Controllable WebSocket double: tests push frames and drop the link. */
export class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];
  static autoOpen = true;

  readonly url: string;
  onopen: (() => void) | null = null;
  onmessage: Listener | null = null;
  onerror: (() => void) | null = null;
  onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;
  readyState = FakeWebSocket.CONNECTING;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    if (FakeWebSocket.autoOpen) queueMicrotask(() => this.open());
  }

  static reset(): void {
    FakeWebSocket.instances = [];
    FakeWebSocket.autoOpen = true;
  }

  static latest(): FakeWebSocket {
    const socket = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    if (!socket) throw new Error("no FakeWebSocket has been constructed");
    return socket;
  }

  static broadcast(frame: unknown): void {
    for (const socket of FakeWebSocket.instances) {
      if (socket.readyState === FakeWebSocket.OPEN) socket.push(frame);
    }
  }

  open(): void {
    if (this.readyState !== FakeWebSocket.CONNECTING) return;
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  send(): void {}

  close(code = 1000, reason = ""): void {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code, reason, wasClean: code === 1000 });
  }

  push(frame: unknown): void {
    this.onmessage?.({ data: JSON.stringify(frame) });
  }

  /**
   * Simulate the server or network dropping the socket: 1006 by default, or
   * e.g. 1008 for the C1 guard refusing the handshake.
   */
  drop(code = 1006, reason = ""): void {
    this.close(code, reason);
  }
}

export function installFakeWebSocket(): void {
  FakeWebSocket.reset();
  vi.stubGlobal("WebSocket", FakeWebSocket);
}

/** Let pending promises and zero-delay timers run (real timers only). */
export async function flush(rounds = 5): Promise<void> {
  for (let index = 0; index < rounds; index += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

/** The application shell as main.tsx mounts it, at a chosen route. */
export function Shell({ children = <div />, path = "/teleop" }: { children?: ReactNode; path?: string }) {
  return (
    <MemoryRouter initialEntries={[path]}>
      <TelemetryProvider>
        <Layout>{children}</Layout>
      </TelemetryProvider>
    </MemoryRouter>
  );
}

/** Advance fake timers inside act() so React flushes the resulting updates. */
export async function advance(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

/** A promise the test settles by hand, e.g. to hold a request in flight. */
export function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason: unknown) => void = () => undefined;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Let pending fetches and the renders they cause settle (real timers). */
export async function settle(rounds = 5): Promise<void> {
  await act(async () => {
    await flush(rounds);
  });
}

/** Fake every timer the shell uses, including Date, for clock-driven tests. */
export function fakeShellTimers(): void {
  vi.useFakeTimers({
    toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"],
  });
}

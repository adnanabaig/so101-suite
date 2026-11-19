import { act, render, renderHook, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { Telemetry } from "../api";
import { ConnectionIndicator } from "../components/Layout";
import {
  advance,
  fakeShellTimers,
  FakeWebSocket,
  installFakeBackend,
  installFakeWebSocket,
  Shell,
  telemetryFrame,
} from "../test/harness";
import {
  NEVER_OPENED_HINT_AFTER,
  sampleStateOf,
  SILENCE_RECONNECT_MS,
  TelemetryProvider,
  TRANSPORT_STALE_MS,
  useTelemetry,
  WS_POLICY_VIOLATION,
} from "./useTelemetry";

function frame(overrides: Record<string, unknown> = {}): Telemetry {
  return telemetryFrame(false, overrides) as unknown as Telemetry;
}

beforeEach(() => {
  installFakeWebSocket();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("sampleStateOf (C6)", () => {
  test("observation_live decides when present: idle is not stale (F8)", () => {
    expect(sampleStateOf(frame({ observation_live: true }))).toBe("live");
    expect(sampleStateOf(frame({ observation_live: false, observation: {} }))).toBe("idle");
    expect(sampleStateOf(null)).toBe("stale");
  });

  test("pre-C6 frames are inferred from mode and sample age", () => {
    const legacy = (overrides: Record<string, unknown>) => {
      const value = frame(overrides);
      delete value.observation_live;
      return value;
    };
    expect(sampleStateOf(legacy({ mode: "idle", observation: {} }))).toBe("idle");
    expect(sampleStateOf(legacy({ mode: "teleop", sample_age_ms: 100 }))).toBe("live");
    expect(sampleStateOf(legacy({ mode: "teleop", sample_age_ms: 60_000 }))).toBe("stale");
    expect(sampleStateOf(legacy({ mode: "teleop", observation: {} }))).toBe("stale");
  });
});

describe("TelemetryProvider", () => {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <TelemetryProvider>{children}</TelemetryProvider>
  );

  test("an open socket that goes silent is replaced (F26)", async () => {
    fakeShellTimers();
    const { result } = renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    const first = FakeWebSocket.latest();
    act(() => first.push(frame()));
    expect(result.current.transportStale).toBe(false);

    await advance(TRANSPORT_STALE_MS + 1000);
    // Still open but silent: shown as stale before it is abandoned.
    expect(result.current.transportStale).toBe(true);
    expect(FakeWebSocket.instances).toHaveLength(1);

    await advance(SILENCE_RECONNECT_MS - TRANSPORT_STALE_MS);
    expect(first.readyState).toBe(FakeWebSocket.CLOSED);
    await advance(1000);
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1);
    const second = FakeWebSocket.latest();
    expect(second.readyState).toBe(FakeWebSocket.OPEN);
    act(() => second.push(frame()));
    expect(result.current.transportStale).toBe(false);
  });

  test("a dropped socket reconnects with backoff and frames from the old one are ignored", async () => {
    fakeShellTimers();
    const { result } = renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    const first = FakeWebSocket.latest();
    act(() => first.drop());
    expect(result.current.state).toBe("closed");
    await advance(800);
    const second = FakeWebSocket.latest();
    expect(second).not.toBe(first);
    act(() => first.push(frame({ mode: "teleop" })));
    expect(result.current.telemetry).toBeNull();
  });

  test("the socket URL follows the page's own origin, never a fixed host or port", async () => {
    fakeShellTimers();
    renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    const expected = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/telemetry`;
    expect(FakeWebSocket.latest().url).toBe(expected);
  });

  test("a 1008 close reports the server's reason verbatim until a socket opens (C1)", async () => {
    fakeShellTimers();
    FakeWebSocket.autoOpen = false;
    const { result } = renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    act(() => FakeWebSocket.latest().drop(WS_POLICY_VIOLATION, "cross-origin request rejected"));
    expect(result.current.linkIssue).toEqual({ kind: "refused", detail: "cross-origin request rejected" });

    // Reconnects keep trying; a later plain failure does not hide the refusal.
    await advance(800);
    act(() => FakeWebSocket.latest().drop());
    await advance(1600);
    act(() => FakeWebSocket.latest().drop());
    expect(result.current.linkIssue).toEqual({ kind: "refused", detail: "cross-origin request rejected" });

    await advance(3100);
    act(() => FakeWebSocket.latest().open());
    expect(result.current.linkIssue).toBeNull();
  });

  test("sockets that never open raise the unreachable hint after three tries; an open clears it", async () => {
    fakeShellTimers();
    FakeWebSocket.autoOpen = false;
    const { result } = renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    act(() => FakeWebSocket.latest().drop());
    await advance(800);
    act(() => FakeWebSocket.latest().drop());
    expect(result.current.linkIssue).toBeNull();
    await advance(1600);
    act(() => FakeWebSocket.latest().drop());
    expect(result.current.linkIssue).toEqual({ kind: "unreachable", attempts: NEVER_OPENED_HINT_AFTER });

    await advance(3100);
    act(() => FakeWebSocket.latest().open());
    expect(result.current.linkIssue).toBeNull();
  });

  test("a socket that opened before dropping does not count toward the hint", async () => {
    fakeShellTimers();
    const { result } = renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    for (let index = 0; index < 4; index += 1) {
      act(() => FakeWebSocket.latest().drop());
      await advance(10_100);
    }
    expect(FakeWebSocket.instances.length).toBeGreaterThan(4);
    expect(result.current.linkIssue).toBeNull();
  });

  test("malformed frames are ignored without dropping the link", async () => {
    fakeShellTimers();
    const { result } = renderHook(() => useTelemetry(), { wrapper });
    await advance(10);
    const socket = FakeWebSocket.latest();
    act(() => socket.onmessage?.({ data: "not json" }));
    act(() => socket.push({ hello: "world" }));
    expect(result.current.telemetry).toBeNull();
    expect(socket.readyState).toBe(FakeWebSocket.OPEN);
  });
});

describe("ConnectionIndicator", () => {
  test("an idle robot on a live link is neutral, never amber (C6, F8)", async () => {
    fakeShellTimers();
    installFakeBackend();
    render(<Shell><ConnectionIndicator /></Shell>);
    await advance(10);
    const indicator = screen.getByTestId("connection-indicator");
    expect(indicator).toHaveTextContent("Connecting to robot");
    expect(indicator).toHaveTextContent("no frames yet");

    act(() => FakeWebSocket.latest().push(frame({ observation_live: false, observation: {} })));
    expect(indicator).toHaveTextContent("Idle · no live joint stream");
    expect(indicator.querySelector(".bg-amber-400")).toBeNull();

    act(() => FakeWebSocket.latest().push(frame({ observation_live: true, fps: 30 })));
    expect(indicator).toHaveTextContent("Joint stream live");
  });

  test("shows the time since the last message and flags a dead link (F26)", async () => {
    fakeShellTimers();
    installFakeBackend();
    render(<Shell><ConnectionIndicator /></Shell>);
    await advance(10);
    act(() => FakeWebSocket.latest().push(frame()));
    await advance(2000);
    const indicator = screen.getByTestId("connection-indicator");
    expect(indicator).toHaveTextContent(/last msg \d\.\ds ago/);
    await advance(2000);
    expect(indicator).toHaveTextContent("Telemetry link down · reconnecting");
  });
});

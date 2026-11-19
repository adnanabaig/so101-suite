import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import {
  advance,
  deferred,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  type FakeBackend,
} from "../test/harness";
import { LEASE_HEARTBEAT_MS, useMotionLease } from "./useMotionLease";
import { TelemetryProvider } from "./useTelemetry";

const wrapper = ({ children }: { children: ReactNode }) => (
  <TelemetryProvider>
    <SafetyProvider>{children}</SafetyProvider>
  </TelemetryProvider>
);

let backend: FakeBackend;
/** Simulated network round trip for the next lease and heartbeat calls. */
let roundTripMs = 0;
/** Simulated server clock lead over the client clock. */
let serverLeadMs = 0;

function leaseBody() {
  // The server stamps its time halfway through the round trip.
  const startedAt = Date.now();
  vi.setSystemTime(startedAt + roundTripMs);
  return {
    id: "lease-1",
    source: "manual-ui",
    ttl_seconds: 10,
    server_time: new Date(startedAt + roundTripMs / 2 + serverLeadMs).toISOString(),
  };
}

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  roundTripMs = 0;
  serverLeadMs = 0;
  backend.route("POST", "/api/control/lease", () => jsonResponse(leaseBody()));
  backend.route("POST", "/api/control/lease/lease-1/heartbeat", () => jsonResponse(leaseBody()));
  backend.route("DELETE", "/api/control/lease/lease-1", () => jsonResponse({ ok: true, detail: "" }));
  backend.route("POST", "/api/control/arm", () => {
    backend.armed = true;
    return jsonResponse({ armed: true, safety_state: "armed", lease: null, watchdog_ms: 500, last_command_at: null });
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function mountWithLease() {
  const hook = renderHook(() => useMotionLease("manual-ui"), { wrapper });
  await advance(10);
  await act(async () => {
    await hook.result.current.acquire();
  });
  return hook;
}

describe("useMotionLease clock and latency (F12)", () => {
  test("a slow lease round trip reports latency, does not sync the clock, and refuses to arm", async () => {
    roundTripMs = 400;
    const { result } = await mountWithLease();
    expect(result.current.leaseId).toBe("lease-1");
    expect(result.current.latencyMs).toBe(400);
    expect(result.current.clockSynced).toBe(false);

    let armed: boolean | undefined;
    await act(async () => {
      armed = await result.current.startMotion();
    });
    expect(armed).toBe(false);
    expect(result.current.error).toBe(
      "Control latency is 400 ms. Arming needs under 250 ms because commands expire after 300 ms.",
    );
    expect(backend.callsTo("/api/control/arm")).toHaveLength(0);
  });

  test("a later fast heartbeat syncs the clock and allows arming", async () => {
    roundTripMs = 400;
    const { result } = await mountWithLease();
    roundTripMs = 40;
    await advance(LEASE_HEARTBEAT_MS);
    expect(result.current.latencyMs).toBe(40);
    expect(result.current.clockSynced).toBe(true);
    let armed: boolean | undefined;
    await act(async () => {
      armed = await result.current.startMotion();
    });
    expect(armed).toBe(true);
    expect(result.current.motionActive).toBe(true);
    expect(backend.callsTo("/api/control/arm")[0].body).toEqual({ lease_id: "lease-1" });
  });

  test("a slow heartbeat is reported but keeps the lease; arming waits for a fast one", async () => {
    const { result } = await mountWithLease();
    roundTripMs = 400;
    await advance(LEASE_HEARTBEAT_MS);
    expect(result.current.leaseId).toBe("lease-1");
    expect(result.current.latencyMs).toBe(400);
    expect(backend.callsTo("/api/control/lease/lease-1", "DELETE")).toHaveLength(0);
    let armed: boolean | undefined;
    await act(async () => {
      armed = await result.current.startMotion();
    });
    expect(armed).toBe(false);
    expect(result.current.error).toMatch(/Control latency is 400 ms/);
  });

  test("commands are stamped on the server clock from a fast round trip", async () => {
    serverLeadMs = 3_600_000;
    roundTripMs = 20;
    const { result } = await mountWithLease();
    const now = Date.now();
    expect(Date.parse(result.current.timestampFor(now)) - now).toBe(serverLeadMs);
  });

  test("a failed heartbeat releases the lease and says so", async () => {
    const { result } = await mountWithLease();
    backend.route("POST", "/api/control/lease/lease-1/heartbeat", () =>
      jsonResponse({ detail: "lease expired" }, 409),
    );
    await advance(LEASE_HEARTBEAT_MS);
    expect(result.current.leaseId).toBeNull();
    expect(result.current.error).toBe("Motion lease heartbeat failed; control was released.");
    expect(backend.callsTo("/api/control/lease/lease-1", "DELETE")).toHaveLength(1);
  });
});

describe("useMotionLease stopping (C5, C11)", () => {
  test("deadman release disarms with the lease when the server has no Halt (C5)", async () => {
    const { result } = await mountWithLease();
    await act(async () => {
      await result.current.startMotion();
    });
    await act(async () => {
      await result.current.stopMotion();
    });
    expect(backend.callsTo("/api/control/disarm").map((call) => call.body)).toEqual([
      { lease_id: "lease-1" },
    ]);
    expect(backend.callsTo("/api/robot/halt")).toHaveLength(0);
    expect(result.current.motionActive).toBe(false);
  });

  test("deadman release is a Halt when the server advertises it (C11)", async () => {
    backend.statusExtra = { stop_kinds: ["estop", "halt", "collapse"] };
    const { result } = await mountWithLease();
    await act(async () => {
      await result.current.startMotion();
    });
    await act(async () => {
      await result.current.stopMotion();
    });
    expect(backend.callsTo("/api/robot/halt").map((call) => call.body)).toEqual([
      { operator_id: expect.any(String), reason: "deadman released" },
    ]);
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(0);
  });

  test("nothing is sent when this hook never armed, including on window blur", async () => {
    const { result } = await mountWithLease();
    await act(async () => {
      await result.current.stopMotion();
    });
    act(() => {
      window.dispatchEvent(new Event("blur"));
    });
    await advance(10);
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(0);
    expect(backend.callsTo("/api/robot/halt")).toHaveLength(0);
  });

  test("window blur while armed stops motion", async () => {
    const { result } = await mountWithLease();
    await act(async () => {
      await result.current.startMotion();
    });
    act(() => {
      window.dispatchEvent(new Event("blur"));
    });
    await advance(10);
    expect(result.current.motionActive).toBe(false);
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
  });

  test("a failed stop is reported with the E-stop instruction and retried on the next stop", async () => {
    const { result } = await mountWithLease();
    await act(async () => {
      await result.current.startMotion();
    });
    backend.route("POST", "/api/control/disarm", () => jsonResponse({ detail: "busy" }, 503));
    await act(async () => {
      await result.current.stopMotion();
    });
    expect(result.current.error).toBe("Stopping motion failed: busy. Use E-stop if the arm is moving.");
    await act(async () => {
      await result.current.stopMotion();
    });
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(2);
  });

  test("explicit Disarm always posts the lease (C5)", async () => {
    const { result } = await mountWithLease();
    let done: boolean | undefined;
    await act(async () => {
      done = await result.current.disarm();
    });
    expect(done).toBe(true);
    expect(backend.callsTo("/api/control/disarm").map((call) => call.body)).toEqual([
      { lease_id: "lease-1" },
    ]);
  });

  test("a wrong-lease 409 on Disarm is shown", async () => {
    const { result } = await mountWithLease();
    backend.route("POST", "/api/control/disarm", () =>
      jsonResponse({ detail: "lease is not the active control lease" }, 409),
    );
    await act(async () => {
      await result.current.disarm();
    });
    expect(result.current.error).toBe("Disarm failed: lease is not the active control lease");
  });

  test("a release that races the arm request stops again after the arm lands (F3)", async () => {
    const { result } = await mountWithLease();
    const armGate = deferred<Response>();
    backend.route("POST", "/api/control/arm", () => armGate.promise);
    let armed: Promise<boolean> | undefined;
    act(() => {
      armed = result.current.startMotion();
    });
    await act(async () => {
      await result.current.stopMotion();
    });
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
    await act(async () => {
      armGate.resolve(jsonResponse({ armed: true, safety_state: "armed", lease: null, watchdog_ms: 500, last_command_at: null }));
      expect(await armed).toBe(false);
    });
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(2);
    expect(result.current.motionActive).toBe(false);
  });

  test("a hidden tab releases the lease", async () => {
    let visibility: DocumentVisibilityState = "visible";
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => visibility });
    try {
      const { result } = await mountWithLease();
      await act(async () => {
        await result.current.startMotion();
      });
      visibility = "hidden";
      act(() => {
        document.dispatchEvent(new Event("visibilitychange"));
      });
      await advance(10);
      expect(result.current.leaseId).toBeNull();
      expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
      expect(backend.callsTo("/api/control/lease/lease-1", "DELETE")).toHaveLength(1);
    } finally {
      Reflect.deleteProperty(document, "visibilityState");
    }
  });

  test("unmounting while armed stops and releases", async () => {
    const { result, unmount } = await mountWithLease();
    await act(async () => {
      await result.current.startMotion();
    });
    unmount();
    await advance(10);
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
    expect(backend.callsTo("/api/control/lease/lease-1", "DELETE")).toHaveLength(1);
  });
});

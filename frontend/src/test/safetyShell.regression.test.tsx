// Regression tests for the five v0.1 audit reproductions: F2-F5 and F18 in
// docs/audit/v0.1/frontend.md. The control-plane E-stop semantics they rely
// on are described in docs/audit/v0.1/safety.md.
//
// The audit versions asserted the broken behaviour. These assert the fixed
// behaviour: each one describes what must never happen again.
import { lazy, Suspense, useState } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ManualControlPanel from "../components/ManualControlPanel";
import { usePolling } from "../hooks/usePolling";
import { SafetyDetails } from "../components/safety/SafetyBar";
import {
  advance,
  FakeWebSocket,
  flush,
  installFakeBackend,
  installFakeWebSocket,
  Shell,
  telemetryFrame,
} from "./harness";

beforeEach(() => {
  installFakeWebSocket();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("audit regressions", () => {
  test("F2 (REPRO 1): rapid taps on the E-stop's position never reach a reset", async () => {
    const backend = installFakeBackend();
    render(<Shell />);
    await act(async () => {
      await flush();
    });
    act(() => FakeWebSocket.latest().push(telemetryFrame(false)));

    const estopButton = screen.getByRole("button", { name: "E-stop" });
    const slot = estopButton.parentElement;
    if (!slot) throw new Error("E-stop has no parent");
    const slotIndex = Array.from(slot.children).indexOf(estopButton);
    const tapSameSpot = async () => {
      const element = slot.children[slotIndex] as HTMLElement;
      await act(async () => {
        fireEvent.click(element);
        await flush();
      });
      return element.getAttribute("aria-label") ?? element.textContent;
    };
    const labels = [await tapSameSpot(), await tapSameSpot(), await tapSameSpot()];

    expect(labels).toEqual(["E-stop", "E-stop", "E-stop"]);
    const posts = backend.posts();
    expect(posts.length).toBeGreaterThan(0);
    expect(posts.every((url) => url === "/api/robot/estop")).toBe(true);
    expect(backend.estopped).toBe(true);
  });

  test("F4 (REPRO 2): stale WebSocket telemetry yields to fresh HTTP status", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"] });
    const backend = installFakeBackend();
    backend.estopped = true;
    render(<Shell><SafetyDetails /></Shell>);
    await advance(10);
    act(() => FakeWebSocket.latest().push(telemetryFrame(true)));
    await advance(10);

    // The socket drops and cannot reconnect (wedged WS task, proxy without
    // upgrade, Wi-Fi roam) while another station resets the E-stop.
    FakeWebSocket.autoOpen = false;
    act(() => FakeWebSocket.latest().drop());
    backend.estopped = false;
    const mark = backend.calls.length;
    await advance(3000);

    const fresh = backend.calls.slice(mark);
    expect(fresh.some((call) => call.path === "/api/robot/status")).toBe(true);
    expect(fresh.some((call) => call.path === "/api/control/state")).toBe(true);
    expect(screen.getByTestId("safety-chip")).toHaveTextContent(/disarmed/i);
    expect(screen.getByRole("button", { name: "E-stop" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: /reset e-stop/i })).toBeNull();
  });

  test("F3 (REPRO 3): unrelated pointer releases never send a disarm", async () => {
    const backend = installFakeBackend();
    render(<ManualControlPanel />);
    await act(async () => {
      await flush();
    });
    const before = backend.calls.length;
    await act(async () => {
      window.dispatchEvent(new Event("pointerup"));
      await flush();
    });
    await act(async () => {
      window.dispatchEvent(new Event("pointercancel"));
      await flush();
    });
    const sent = backend.calls.slice(before);
    expect(sent.filter((call) => call.path === "/api/control/disarm")).toEqual([]);
    expect(sent.filter((call) => call.path === "/api/robot/stop")).toEqual([]);
  });

  test("F5 (REPRO 4): a failed lazy route chunk keeps the shell and E-stop alive", async () => {
    installFakeBackend();
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const Broken = lazy(() =>
      Promise.reject(new Error("Failed to fetch dynamically imported module: /assets/Eval-OLDHASH.js")),
    );
    const onError = (event: ErrorEvent) => event.preventDefault();
    window.addEventListener("error", onError);
    try {
      render(
        <Shell>
          <Suspense fallback={<div>Loading workspace</div>}>
            <Broken />
          </Suspense>
        </Shell>,
      );
      await act(async () => {
        await flush();
      });
    } finally {
      window.removeEventListener("error", onError);
    }
    expect(screen.getByRole("button", { name: "E-stop" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /reload/i })).toBeInTheDocument();
  });

  test("F18 (REPRO 5): usePolling never shows the previous parameter's data", async () => {
    const fetchStatus = (arm: string) => Promise.resolve({ arm });
    function Probe({ arm }: { arm: string }) {
      const { data } = usePolling(() => fetchStatus(arm), 1000, { key: arm });
      return (
        <div data-testid="probe">
          Status — {arm}: data for {data?.arm ?? "none"}
        </div>
      );
    }
    function Host() {
      const [arm, setArm] = useState("follower");
      return (
        <>
          <button onClick={() => setArm("leader")}>switch</button>
          <Probe arm={arm} />
        </>
      );
    }
    render(<Host />);
    await act(async () => {
      await flush();
    });
    act(() => {
      fireEvent.click(screen.getByText("switch"));
    });
    expect(screen.getByTestId("probe")).not.toHaveTextContent("data for follower");
    await act(async () => {
      await flush();
    });
    expect(screen.getByTestId("probe")).toHaveTextContent("Status — leader: data for leader");
  });
});

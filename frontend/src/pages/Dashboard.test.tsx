import { act, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import { TelemetryProvider } from "../hooks/useTelemetry";
import {
  FakeWebSocket,
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  telemetryFrame,
  type FakeBackend,
} from "../test/harness";
import Dashboard, { attentionReason, twinSubtitle } from "./Dashboard";

// The WebGL twin is covered by its own tests. Here the stub records the
// freshness the Dashboard hands it.
vi.mock("../components/RobotTwin", () => ({
  default: (props: { stale?: boolean; idle?: boolean }) => (
    <div data-testid="robot-twin" data-stale={String(Boolean(props.stale))} data-idle={String(Boolean(props.idle))} />
  ),
}));

const READY = {
  tripped: false,
  torqueUnconfirmed: false,
  stopState: "none",
  transportStale: false,
  statusKnown: true,
  followerConnected: true,
};

describe("attentionReason", () => {
  test("a released idle torque-off bus is expected; active or unconfirmed disconnections still warn", () => {
    expect(attentionReason({ ...READY, followerConnected: false, mode: "idle", torqueState: "off" })).toBeNull();
    expect(attentionReason({ ...READY, followerConnected: false, mode: "teleop", torqueState: "off" })).toMatch(/not connected/);
    expect(attentionReason({ ...READY, followerConnected: false, mode: "idle", torqueState: "unconfirmed" })).toMatch(/not connected/);
  });
  test("is null when ready; an idle link is not a problem (F8)", () => {
    expect(attentionReason(READY)).toBeNull();
  });

  test("names the most urgent problem first", () => {
    const everything = {
      tripped: true,
      torqueUnconfirmed: true,
      stopState: "halted",
      transportStale: true,
      statusKnown: false,
      followerConnected: false,
    };
    expect(attentionReason(everything)).toMatch(/emergency-stop latch/);
    expect(attentionReason({ ...everything, tripped: false })).toMatch(/Torque-off could not be confirmed/);
    expect(attentionReason({ ...READY, stopState: "halted", transportStale: true })).toMatch(/halted/);
    expect(attentionReason({ ...READY, stopState: "collapsing" })).toMatch(/lowering to its rest pose/);
    expect(attentionReason({ ...READY, transportStale: true, statusKnown: false })).toMatch(/Telemetry is not current/);
    expect(attentionReason({ ...READY, statusKnown: false })).toMatch(/has not been received/);
    expect(attentionReason({ ...READY, followerConnected: false })).toMatch(/follower arm is not connected/);
  });
});

describe("twinSubtitle (F14)", () => {
  test("never calls a missing or old pose live", () => {
    expect(twinSubtitle(false, false, "live")).toBe("No joint sample yet, so the arm is a placeholder");
    expect(twinSubtitle(true, true, "live")).toBe("Link stale: the last known pose, not the live arm");
    expect(twinSubtitle(true, false, "stale")).toBe("Joint samples overdue: the last known pose");
    expect(twinSubtitle(true, false, "idle")).toBe("Idle: the last pose read; nothing is sampling the joints");
    expect(twinSubtitle(true, false, "live")).toBe("Follows the live joint state");
  });

  test("no sample outranks a stale link", () => {
    expect(twinSubtitle(false, true, "stale")).toBe("No joint sample yet, so the arm is a placeholder");
  });
});

let backend: FakeBackend;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  backend.route("GET", "/api/capabilities", () => jsonResponse({ system_version: "test", capabilities: [] }));
  backend.route("GET", "/api/events", () => jsonResponse({ events: [] }));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderDashboard() {
  render(
    <MemoryRouter>
      <TelemetryProvider>
        <SafetyProvider>
          <Dashboard />
        </SafetyProvider>
      </TelemetryProvider>
    </MemoryRouter>,
  );
  await advance(10);
}

describe("Dashboard readiness (F4/F8)", () => {
  test("an idle but live link reads as ready and explains the idle sample", async () => {
    await renderDashboard();
    act(() =>
      FakeWebSocket.broadcast(telemetryFrame(false, { observation: {}, observation_live: false, sample_age_ms: null })),
    );
    await advance(10);
    expect(screen.getByText("Workstation ready for supervised operation")).toBeInTheDocument();
    expect(screen.getByText(/No loop is sampling joints right now/)).toBeInTheDocument();
    expect(screen.getByText("No joint sample yet, so the arm is a placeholder")).toBeInTheDocument();
  });

  test("the twin is told when the pose is only the last one read, and when it is stale", async () => {
    await renderDashboard();
    act(() =>
      FakeWebSocket.broadcast(
        telemetryFrame(false, { observation: { "shoulder_pan.pos": 3 }, observation_live: false, sample_age_ms: 40 }),
      ),
    );
    await advance(10);
    expect(screen.getByTestId("robot-twin")).toHaveAttribute("data-idle", "true");
    expect(screen.getByTestId("robot-twin")).toHaveAttribute("data-stale", "false");
    expect(screen.getByText("Idle: the last pose read; nothing is sampling the joints")).toBeInTheDocument();

    await advance(3000); // the socket goes silent
    expect(screen.getByTestId("robot-twin")).toHaveAttribute("data-stale", "true");
    expect(screen.getByText("Link stale: the last known pose, not the live arm")).toBeInTheDocument();
  });

  test("a stale socket that last said clear cannot outvote a fresh status poll that says latched", async () => {
    await renderDashboard();
    act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
    backend.estopped = true;
    await advance(3000); // the socket goes silent; the status poll keeps running
    expect(screen.getByText("Operator attention required")).toBeInTheDocument();
    expect(screen.getByText(/emergency-stop latch is engaged/)).toBeInTheDocument();
  });

  test("a silent socket is called out instead of showing old values as current", async () => {
    await renderDashboard();
    act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
    await advance(3000);
    expect(screen.getByText(/Telemetry is not current/)).toBeInTheDocument();
  });
});

describe("Dashboard camera wall", () => {
  test("a RealSense with depth shows a color tile and a depth tile, each labelled", async () => {
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => "blob:snapshot");
    backend.route("GET", "/api/video/snapshot", () => new Response("jpeg", { headers: { "content-type": "image/jpeg" } }));
    const color = "realsense:123456789012+depth";
    backend.statusExtra = {
      cameras: ["0", color, `${color}#depth`],
      live_cameras: ["0", color, `${color}#depth`],
    };
    await renderDashboard();
    await advance(10);
    expect(screen.getByText("3/3")).toBeInTheDocument();
    expect(screen.getByText("RealSense 123456789012 depth")).toHaveAttribute("title", `${color}#depth`);
    expect(screen.getByTestId(`camera-${color}#depth`)).not.toHaveAttribute("data-mode", "paused");
    expect(screen.getByText("RealSense 123456789012")).toHaveAttribute("title", color);
    // An OpenCV index is its own label, so there is no tooltip repeating it.
    expect(screen.getByText("0", { selector: "span" })).not.toHaveAttribute("title");
  });
});

describe("Dashboard follower tile", () => {
  test("idle torque-off does not display a motor-bus failure or imply motion qualification", async () => {
    backend.statusExtra = { backend: "real", mock: false, mode: "idle", torque_state: "off", follower_connected: false };
    await renderDashboard();
    act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
    await advance(20);
    expect(screen.getByText("Bus opens when an operation starts")).toBeInTheDocument();
    expect(screen.getByText("Idle · choose an operation")).toBeInTheDocument();
    expect(screen.queryByText(/follower arm is not connected/)).not.toBeInTheDocument();
  });
  test.each([
    ["sim", "Simulated", "MuJoCo simulation: no physical arm"],
    ["mock", "Online", "Mock hardware adapter"],
    ["real", "Online", "Physical motor bus"],
  ])("backend %s: %s, %s", async (kind, value, detail) => {
    backend.statusExtra = { backend: kind, mock: kind === "mock", follower_connected: true };
    await renderDashboard();
    await advance(10);
    expect(screen.getByText(value)).toBeInTheDocument();
    expect(screen.getByText(detail)).toBeInTheDocument();
  });

  test("an older server without backend keeps the mock-based reading", async () => {
    backend.statusExtra = { mock: false, follower_connected: true };
    await renderDashboard();
    await advance(10);
    expect(screen.getByText("Physical motor bus")).toBeInTheDocument();
  });
});

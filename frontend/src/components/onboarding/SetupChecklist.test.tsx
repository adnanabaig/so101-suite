import { fireEvent, render, screen, within } from "@testing-library/react";
import { Navigate, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import Dashboard from "../../pages/Dashboard";
import Settings, { CHECKLIST_SHOWN_TOAST } from "../../pages/Settings";
import { Workspace } from "../../App";
import { WORKSPACES } from "../navigation";
import { CHECKLIST_DISMISSED_KEY, STOPS_ACKNOWLEDGED_KEY } from "../../lib/onboarding";
import {
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  Shell,
  type FakeBackend,
} from "../../test/harness";
import { CHECKLIST_HIDDEN_TOAST, SETUP_POLL_MS } from "./SetupChecklist";

// The WebGL twin has its own tests.
vi.mock("../RobotTwin", () => ({ default: () => <div data-testid="robot-twin" /> }));

const NO_REPORT_DETAIL = "no saved preflight report for the mock backend; run POST /api/preflight or `so101 preflight`";

function capabilities(mock: boolean) {
  return {
    system_version: "0.2.0",
    mock,
    capabilities: [
      { id: "control", label: "Safe manual command API", available: true, maturity: "stable", detail: "API contract only." },
    ],
  };
}

function calibration(arm: string, saved: boolean) {
  return {
    arm,
    in_progress: false,
    fraction: 0,
    message: "",
    has_saved: saved,
    device_id: saved ? `so101_${arm}` : "",
    created_at: saved ? 1_790_000_000 : null,
    ranges: {},
  };
}

function report(overrides: Record<string, unknown> = {}) {
  return {
    kind: "so101.preflight",
    schema_version: 1,
    created_at: "2026-10-01T10:00:00+00:00",
    so101_version: "0.2.0",
    origin: "cli",
    backend: "real",
    rehearsal: false,
    arms: ["follower", "leader"],
    bus_access: "opened",
    bus_access_detail: "",
    overall: "pass",
    ready_for_hardware: true,
    exit_code: 0,
    counts: { pass: 1, warn: 0, fail: 0, skip: 0 },
    checks: [],
    notes: [],
    ...overrides,
  };
}

let backend: FakeBackend;

function serve({
  mock = true,
  saved = false,
  preflight = () => jsonResponse({ detail: NO_REPORT_DETAIL }, 404),
}: {
  mock?: boolean;
  saved?: boolean;
  preflight?: (() => Response) | null;
} = {}) {
  backend.route("GET", "/api/capabilities", () => jsonResponse(capabilities(mock)));
  backend.route("GET", "/api/events", () => jsonResponse({ events: [] }));
  backend.route("GET", "/api/calibration/status/follower", () => jsonResponse(calibration("follower", saved)));
  backend.route("GET", "/api/calibration/status/leader", () => jsonResponse(calibration("leader", saved)));
  if (preflight) backend.route("GET", "/api/preflight/latest", preflight);
}

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function App() {
  return (
    <Shell path="/operate?section=overview">
      <Routes>
        <Route path="/" element={<Navigate replace to="/operate?section=overview" />} />
        <Route path="/operate" element={<Dashboard />} />
        <Route path="/system" element={<Workspace workspace={WORKSPACES[2]} />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Shell>
  );
}

async function renderApp() {
  const view = render(<App />);
  await advance(10);
  return view;
}

const checklist = () =>
  screen.queryByRole("heading", { name: /^(Set up this workstation|Setup complete|Rehearsal setup complete)$/ });
const row = (id: string) => screen.getByTestId(`setup-${id}`);

/** No request ever ran preflight: POST /api/preflight runs every check and saves a report. */
function expectPreflightNeverRun() {
  expect(backend.calls.filter((call) => call.path === "/api/preflight")).toEqual([]);
}

describe("setup checklist on Overview", () => {
  test("a sim server: the backend says simulated, and calibration is not needed", async () => {
    serve({ mock: false, saved: false });
    backend.route("GET", "/api/capabilities", () => jsonResponse({ ...capabilities(false), backend: "sim" }));
    await renderApp();
    expect(row("backend")).toHaveTextContent(/MuJoCo simulation: commands move a simulated arm, not the physical one/);
    expect(row("calibration.follower")).toHaveAttribute("data-state", "done");
    expect(row("calibration.follower")).toHaveTextContent(/Not needed: the simulated arm has no encoders/);
  });

  test("first run on a mock server: what is done, what is next, and the preflight slot", async () => {
    serve();
    await renderApp();
    expect(checklist()).toHaveTextContent("Set up this workstation");
    expect(screen.getByText(/^2 of 4 required steps done/)).toBeInTheDocument();

    expect(row("backend")).toHaveAttribute("data-state", "done");
    expect(row("backend")).toHaveTextContent(/Mock rehearsal: no real motors move/);
    expect(row("calibration.follower")).toHaveAttribute("data-state", "done");
    expect(row("calibration.follower")).toHaveTextContent(/default ranges/);

    const preflight = row("preflight");
    expect(preflight).toHaveAttribute("data-state", "todo");
    expect(within(preflight).getByText("so101 preflight", { selector: "code" })).toBeInTheDocument();
    // The server's detail, exactly as sent.
    expect(within(preflight).getByText(NO_REPORT_DETAIL)).toBeInTheDocument();
    expect(within(preflight).getByText(/comes with the preflight wizard/)).toBeInTheDocument();
    expect(backend.callsTo("/api/preflight/latest", "GET")).toHaveLength(1);
    expectPreflightNeverRun();
  });

  test("confirming the stops persists and keeps focus on that step", async () => {
    serve();
    const view = await renderApp();
    const stops = row("stops");
    expect(stops).toHaveTextContent(/Escape is E-stop on every page/);
    expect(stops).toHaveTextContent(/This server has no Halt/);

    fireEvent.click(within(stops).getByRole("button", { name: "I know the stops" }));
    await advance(10);
    expect(row("stops")).toHaveAttribute("data-state", "done");
    expect(within(row("stops")).queryByRole("button", { name: "I know the stops" })).toBeNull();
    expect(within(row("stops")).getByRole("heading", { name: "Know your stops" })).toHaveFocus();
    expect(screen.getByText(/^3 of 4 required steps done/)).toBeInTheDocument();
    expect(window.localStorage.getItem(STOPS_ACKNOWLEDGED_KEY)).toBe("1");

    view.unmount();
    await renderApp();
    expect(row("stops")).toHaveAttribute("data-state", "done");

    // The full key list is one press away.
    fireEvent.click(within(row("stops")).getByRole("button", { name: "All keyboard shortcuts" }));
    await advance(10);
    expect(screen.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeInTheDocument();
  });

  test("Halt and Collapse are described when the server offers them", async () => {
    serve();
    backend.statusExtra = { stop_kinds: ["estop", "halt", "collapse"] };
    await renderApp();
    expect(row("stops")).toHaveTextContent(/H is Halt: the arm freezes and the motors hold\./);
    expect(row("stops")).toHaveTextContent(/Collapse has no key/);
  });

  test("hiding persists across reloads, stops the polling, and Settings brings it back", async () => {
    serve();
    const view = await renderApp();
    fireEvent.click(screen.getByRole("button", { name: "Hide checklist" }));
    await advance(10);
    expect(checklist()).toBeNull();
    expect(screen.getByRole("main")).toHaveFocus();
    expect(screen.getByTestId("toast-region")).toHaveTextContent(CHECKLIST_HIDDEN_TOAST);
    expect(window.localStorage.getItem(CHECKLIST_DISMISSED_KEY)).toBe("1");

    view.unmount();
    await renderApp();
    expect(checklist()).toBeNull();
    const reads = backend.callsTo("/api/preflight/latest", "GET").length;
    await advance(SETUP_POLL_MS * 2);
    expect(backend.callsTo("/api/preflight/latest", "GET")).toHaveLength(reads);

    fireEvent.click(screen.getByRole("link", { name: "System" }));
    await advance(10);
    fireEvent.click(within(screen.getByRole("navigation", { name: "System views" })).getByRole("link", { name: "Settings" }));
    await advance(10);
    fireEvent.click(screen.getByRole("button", { name: "Show setup checklist" }));
    await advance(10);
    expect(window.localStorage.getItem(CHECKLIST_DISMISSED_KEY)).toBeNull();
    expect(screen.getByTestId("toast-region")).toHaveTextContent(CHECKLIST_SHOWN_TOAST);
    expect(within(screen.getByRole("main")).getByRole("link", { name: "Overview" })).toHaveFocus();

    fireEvent.click(within(screen.getByRole("main")).getByRole("link", { name: "Overview" }));
    await advance(10);
    expect(checklist()).toHaveTextContent("Set up this workstation");
    expectPreflightNeverRun();
  });

  test("a ready real workstation: setup complete, with when each piece of evidence was made", async () => {
    window.localStorage.setItem(STOPS_ACKNOWLEDGED_KEY, "1");
    serve({ mock: false, saved: true, preflight: () => jsonResponse(report()) });
    await renderApp();
    expect(checklist()).toHaveTextContent("Setup complete");
    expect(screen.getByText("Everything a first run needs is in place.")).toBeInTheDocument();
    expect(row("backend")).toHaveTextContent(/The last preflight ran on the real backend\./);
    expect(row("calibration.follower")).toHaveTextContent("Saved for so101_follower.");
    const calibrated = row("calibration.follower").querySelector("time");
    expect(calibrated).toHaveAttribute("dateTime", new Date(1_790_000_000_000).toISOString());
    expect(row("preflight")).toHaveTextContent(/Passed on the real backend\./);
    expect(row("preflight")).toHaveTextContent(/Ran with so101 preflight/);
    expect(row("preflight").querySelector("time")).toHaveAttribute("dateTime", "2026-10-01T10:00:00+00:00");
  });

  test("a preflight run in a terminal shows up on the next refresh", async () => {
    let saved: Record<string, unknown> | null = null;
    serve({ preflight: () => (saved ? jsonResponse(saved) : jsonResponse({ detail: NO_REPORT_DETAIL }, 404)) });
    await renderApp();
    expect(row("preflight")).toHaveAttribute("data-state", "todo");

    saved = report({ backend: "mock", rehearsal: true, ready_for_hardware: false, bus_access: "synthetic" });
    await advance(SETUP_POLL_MS);
    expect(row("preflight")).toHaveAttribute("data-state", "rehearsal");
    expect(row("preflight")).toHaveTextContent(/Rehearsal passed on the mock backend\. A rehearsal is not evidence about a real arm\./);
  });

  test("a real server after a sim preflight: the rehearsal is not counted and setup is not complete", async () => {
    window.localStorage.setItem(STOPS_ACKNOWLEDGED_KEY, "1");
    serve({
      mock: false,
      saved: true,
      preflight: () => jsonResponse(report({ backend: "sim", rehearsal: true, ready_for_hardware: false, bus_access: "unavailable" })),
    });
    backend.route("GET", "/api/capabilities", () => jsonResponse({ ...capabilities(false), backend: "real" }));
    await renderApp();
    expect(row("preflight")).toHaveAttribute("data-state", "todo");
    expect(row("preflight")).toHaveTextContent(/The last preflight was a sim rehearsal, which is not evidence about this arm\./);
    expect(checklist()).toHaveTextContent("Set up this workstation");
    expect(screen.getByText(/^3 of 4 required steps done/)).toBeInTheDocument();
  });

  test("a mock server with every step done: a rehearsal setup, said plainly", async () => {
    window.localStorage.setItem(STOPS_ACKNOWLEDGED_KEY, "1");
    serve({ preflight: () => jsonResponse(report({ backend: "mock", rehearsal: true, ready_for_hardware: false, bus_access: "synthetic" })) });
    await renderApp();
    expect(row("preflight")).toHaveAttribute("data-state", "rehearsal");
    expect(checklist()).toHaveTextContent("Rehearsal setup complete");
    expect(screen.getByText(/in place on the mock backend\. None of it is evidence about a real arm\./)).toBeInTheDocument();
  });

  test("a failed preflight lists the failed checks; an older server without the API says so", async () => {
    serve({
      mock: false,
      preflight: () =>
        jsonResponse(
          report({
            overall: "fail",
            ready_for_hardware: false,
            checks: [
              {
                id: "motors",
                key: "motors.follower",
                arm: "follower",
                title: "Motors",
                status: "fail",
                message: "motor 3 did not answer",
                remediation: "Check the daisy chain.",
                measured: {},
                blocking: true,
                rehearsal: false,
                source: "real",
                references: [],
                duration_ms: 3,
              },
            ],
          }),
        ),
    });
    const view = await renderApp();
    expect(row("preflight")).toHaveAttribute("data-state", "problem");
    expect(within(row("preflight")).getByText("Motors (follower): motor 3 did not answer. Check the daisy chain.")).toBeInTheDocument();
    expect(row("calibration.follower")).toHaveAttribute("data-state", "todo");
    expect(within(row("calibration.follower")).getByRole("link", { name: "Open Calibration" })).toHaveAttribute(
      "href",
      "/calibration",
    );
    view.unmount();

    backend = installFakeBackend();
    serve({ preflight: null });
    await renderApp();
    expect(row("preflight")).toHaveAttribute("data-state", "unavailable");
    expect(row("preflight")).toHaveTextContent(/This server has no preflight API/);
  });
});

import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import TeleopRecord from "./TeleopRecord";
import { MemoryRouter } from "react-router-dom";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import { TelemetryProvider } from "../hooks/useTelemetry";
import { advance, FakeWebSocket, fakeShellTimers, installFakeBackend, installFakeWebSocket, jsonResponse, telemetryFrame, type FakeBackend } from "../test/harness";

vi.mock("../components/RobotTwin", () => ({ default: ({ observation, stale, idle }: { observation?: Record<string, number>; stale?: boolean; idle?: boolean }) =>
  <div data-testid="collection-twin" data-stale={String(stale)} data-idle={String(idle)}>{JSON.stringify(observation)}</div> }));
vi.mock("../components/ManualControlPanel", () => ({ default: () => <div data-testid="manual-tools">Manual tools</div> }));

let backend: FakeBackend;
beforeEach(() => {
  fakeShellTimers(); installFakeWebSocket(); backend = installFakeBackend();
  backend.statusExtra = { backend: "real", mock: false };
  backend.route("GET", "/api/mcap/status", () => jsonResponse({ recorder: { recording: false, service_ready: true }, cameras: {
    obsbot: { live: true, actual_preview_fps: 30 }, camera_1: { live: true }, camera_2: { live: true },
  }, active: null, episodes: [] }));
  backend.route("GET", "/api/calibration/pose/follower", () => jsonResponse({ arm: "follower", backend: "real", complete: true,
    captured_at: (Date.now() - 3000) / 1000,
    observation: Object.fromEntries(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"].map((joint) => [`${joint}.pos`, 42])),
  }));
});

async function open() {
  render(<MemoryRouter><TelemetryProvider><SafetyProvider><TeleopRecord /></SafetyProvider></TelemetryProvider></MemoryRouter>);
  await advance(20);
  act(() => FakeWebSocket.broadcast(telemetryFrame(false, { observation: {}, observation_live: false })));
  await advance(20);
}

test("collection opens cameras, encoder model and one episode workflow without direct recording or motor writes", async () => {
  await open();
  expect(screen.getByRole("heading", { name: "Operate" })).toBeInTheDocument();
  expect(screen.getAllByRole("img")).toHaveLength(3);
  expect(screen.getAllByRole("button", { name: "Start Teleop" })).toHaveLength(1);
  expect(screen.queryByRole("button", { name: "Start recording" })).not.toBeInTheDocument();
  expect(screen.queryByTestId("manual-tools")).not.toBeInTheDocument();
  expect(backend.posts().filter((path) => path !== "/api/robot/operator-heartbeat")).toEqual([]);
});

test("idle dashboard labels a cached pose as sampling paused without pretending it is live or reading a bus", async () => {
  await open();
  expect(screen.getByTestId("collection-twin")).toHaveTextContent('"shoulder_lift.pos":42');
  expect(screen.getByTestId("collection-twin")).toHaveAttribute("data-stale", "false");
  expect(screen.getByTestId("collection-twin")).toHaveAttribute("data-idle", "true");
  expect(screen.getByText("Sampling paused · last read")).toBeVisible();
  expect(backend.callsTo("/api/calibration/pose/follower", "POST")).toHaveLength(0);
});

test("manual tools mount only after an explicit switch and failed camera preview offers retry", async () => {
  await open();
  fireEvent.click(screen.getByRole("button", { name: "Open manual controls" }));
  expect(screen.getByTestId("manual-tools")).toBeVisible();
  fireEvent.error(screen.getByRole("img", { name: "camera_2 arm camera" }));
  expect(screen.getByRole("img", { name: "camera_2 arm camera" })).toHaveAttribute("src", "/api/mcap/cameras/camera_2/snapshot.jpg?frame=0-0");
  fireEvent.error(screen.getByRole("img", { name: "camera_2 arm camera" }));
  expect(screen.getByText("Preview unavailable")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Retry camera preview" }));
  expect(screen.getByRole("img", { name: "camera_2 arm camera" })).toBeVisible();
});

test("a reconnected camera retries its preview automatically without an operator click", async () => {
  await open();
  fireEvent.error(screen.getByRole("img", { name: "camera_2 arm camera" }));
  fireEvent.error(screen.getByRole("img", { name: "camera_2 arm camera" }));
  expect(screen.getByText("Preview unavailable")).toBeVisible();
  await advance(2100);
  expect(screen.getByRole("img", { name: "camera_2 arm camera" })).toBeVisible();
  expect(backend.posts().filter(path => path !== "/api/robot/operator-heartbeat")).toEqual([]);
});

test("snapshot fallback follows measured source cadence instead of refetching duplicate frames at 30 Hz", async () => {
  backend.route("GET", "/api/mcap/status", () => jsonResponse({ recorder: { recording: false, service_ready: true }, active: null, episodes: [], cameras: { camera_2: { live: true, actual_preview_fps: 5 } } }));
  await open();
  fireEvent.error(screen.getByRole("img", { name: "camera_2 arm camera" }));
  const image = screen.getByRole("img", { name: "camera_2 arm camera" });
  fireEvent.load(image);
  await advance(100);
  expect(image.getAttribute("src")).toContain("frame=0-0");
  await advance(120);
  expect(image.getAttribute("src")).toContain("frame=0-1");
});

test("an idle pose read requires support confirmation and confirmed torque off", async () => {
  backend.statusExtra.torque_state = "off";
  backend.route("POST", "/api/calibration/pose/follower", () => jsonResponse({
    arm: "follower", backend: "real", complete: false, observation: {}, captured_at: Date.now() / 1000,
  }));
  await open();
  fireEvent.click(screen.getByText("Read the supported arm here"));
  const read = screen.getByRole("button", { name: "Read current pose" });
  expect(read).toBeDisabled();
  fireEvent.click(screen.getByLabelText("Arm supported, torque off, workspace clear"));
  expect(read).toBeEnabled();
  fireEvent.click(read); await advance(30);
  expect(backend.callsTo("/api/calibration/pose/follower")[0].body).toEqual({ confirm_supported_torque_off: true });
  await advance(400);
  expect(backend.callsTo("/api/calibration/pose/follower").length).toBeGreaterThanOrEqual(3);
  expect(screen.getByText("Motor 2 · shoulder lift:")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Stop live readings" }));
  await advance(20);
  const stoppedAt = backend.callsTo("/api/calibration/pose/follower").length;
  await advance(400);
  expect(backend.callsTo("/api/calibration/pose/follower")).toHaveLength(stoppedAt);
  expect(backend.posts().filter((path) => !["/api/robot/operator-heartbeat", "/api/calibration/pose/follower"].includes(path))).toEqual([]);
});

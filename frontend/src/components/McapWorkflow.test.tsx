import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import McapWorkflow, { cameraPreviewUrl, encoderSampleFresh } from "./McapWorkflow";
import { SafetyProvider } from "./safety/SafetyProvider";
import { TelemetryProvider } from "../hooks/useTelemetry";
import { advance, FakeWebSocket, fakeShellTimers, installFakeBackend, installFakeWebSocket, jsonResponse, telemetryFrame, type FakeBackend } from "../test/harness";
import type { McapEpisode } from "../api";
import { MemoryRouter } from "react-router-dom";

let backend: FakeBackend;
let active: McapEpisode | null;
let episodes: McapEpisode[];
let recording: boolean;

test("a sample arriving between clock ticks stays fresh without hiding expired or future samples", () => {
  expect(encoderSampleFresh(1.05, 1060, 1000)).toBe(true);
  expect(encoderSampleFresh(1.05, 1060, 1500)).toBe(true);
  expect(encoderSampleFresh(1.05, 1060, 1600)).toBe(false);
  expect(encoderSampleFresh(1.1, 1060, 1000)).toBe(false);
});

beforeEach(() => {
  fakeShellTimers(); installFakeWebSocket(); backend = installFakeBackend();
  active = null; episodes = []; recording = true;
  backend.route("GET", "/api/mcap/status", () => jsonResponse({ recorder: { recording, telemetry_fresh: true }, cameras: { obsbot: { live: true } }, active, episodes }));
  backend.route("POST", "/api/teleop/start", () => { backend.mode = "teleop"; return jsonResponse({ ok: true }); });
  backend.route("POST", "/api/teleop/stop", () => { backend.mode = "idle"; return jsonResponse({ ok: true }); });
  backend.route("POST", "/api/mcap/episodes/start", () => {
    active = { id: "episode-1", task: "move cube", source: "leader_teleop", status: "recording", started_at_ns: 1000, starting_segment: "test.mcap" };
    return jsonResponse(active);
  });
  backend.route("POST", "/api/mcap/episodes/save", () => { const saved = { ...active!, status: "saved" as const, ended_at_ns: 2000 }; episodes.push(saved); active = null; return jsonResponse(saved); });
});

async function open() {
  render(<MemoryRouter><TelemetryProvider><SafetyProvider><McapWorkflow /></SafetyProvider></TelemetryProvider></MemoryRouter>);
  await advance(20); act(() => FakeWebSocket.broadcast(telemetryFrame(false))); await advance(20);
}

test("Start and Stop Teleop automatically open and save one MCAP episode", async () => {
  await open();
  expect(screen.getByRole("img", { name: "obsbot arm camera" })).toHaveAttribute("src", cameraPreviewUrl("obsbot"));
  fireEvent.change(screen.getByLabelText("Episode task"), { target: { value: "move cube" } });
  fireEvent.click(screen.getByRole("button", { name: "Start Teleop" })); await advance(30);
  expect(backend.callsTo("/api/mcap/episodes/start")[0].body).toMatchObject({ task: "move cube", operator_id: expect.any(String) });
  fireEvent.click(screen.getByRole("button", { name: "Stop Teleop" })); await advance(30);
  expect(screen.getByText("saved")).toBeInTheDocument();
  expect(backend.callsTo("/api/mcap/episodes/save")[0].body).toMatchObject({ episode_id: "episode-1" });
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
  expect(backend.posts().filter((path) => !path.includes("heartbeat"))).toEqual([
    "/api/teleop/start", "/api/mcap/episodes/start", "/api/mcap/episodes/save", "/api/teleop/stop",
  ]);
});

test("recorder failure blocks new starts while Halt remains available", async () => {
  recording = false; await open();
  expect(screen.getByRole("button", { name: "Start Teleop" })).toBeDisabled();
  expect(screen.getByText("Start the local MCAP recording service.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Stop Teleop" })); await advance(30);
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
});

test("an idle ready recorder allows teleop and an episode without continuous disk recording", async () => {
  backend.route("GET", "/api/mcap/status", () => jsonResponse({
    recorder: { recording: false, service_ready: true, telemetry_fresh: false },
    cameras: { obsbot: { live: true } }, active, episodes,
  }));
  await open();
  expect(screen.getByText("MCAP ready · waiting for activity")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Start Teleop" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Start Teleop" })); await advance(30);
  expect(backend.callsTo("/api/mcap/episodes/start")).toHaveLength(1);
  expect(screen.queryByRole("button", { name: "Start MCAP episode" })).not.toBeInTheDocument();
});

test("local previews use separate relay origins while remote and HTTPS clients retain the proxy", () => {
  expect(cameraPreviewUrl("obsbot", "http://127.0.0.1:8000")).toBe("http://127.0.0.1:8765/stream.mjpg");
  expect(cameraPreviewUrl("realsense", "http://localhost:8000")).toBe("http://localhost:8770/stream.mjpg");
  expect(cameraPreviewUrl("obsbot", "https://robot.example")).toBe("/api/mcap/cameras/obsbot/stream.mjpg");
  expect(cameraPreviewUrl("obsbot", "http://192.168.1.2:8000")).toBe("/api/mcap/cameras/obsbot/stream.mjpg");
});

test("a failed MCAP start Halts teleop and exposes the capture failure", async () => {
  backend.route("POST", "/api/mcap/episodes/start", () => jsonResponse({ detail: "Capture unavailable" }, 409));
  await open();
  fireEvent.click(screen.getByRole("button", { name: "Start Teleop" })); await advance(30);
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
  expect(screen.getByRole("alert")).toHaveTextContent("Capture unavailable");
  expect(episodes).toHaveLength(0);
});

test("a failed episode save still Halts and never claims the episode was saved", async () => {
  await open();
  fireEvent.click(screen.getByRole("button", { name: "Start Teleop" })); await advance(30);
  backend.route("POST", "/api/mcap/episodes/save", () => jsonResponse({ detail: "Disk unavailable" }, 409));
  fireEvent.click(screen.getByRole("button", { name: "Stop Teleop" })); await advance(30);
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
  expect(screen.getByRole("alert")).toHaveTextContent("Disk unavailable");
  expect(episodes).toHaveLength(0);
});

test("Stop during a pending start Halts immediately and again after a late start acknowledgement", async () => {
  let acknowledge!: () => void;
  backend.route("POST", "/api/teleop/start", () => new Promise<Response>((resolve) => {
    acknowledge = () => { backend.mode = "teleop"; resolve(jsonResponse({ ok: true })); };
  }));
  await open();
  fireEvent.click(screen.getByRole("button", { name: "Start Teleop" })); await advance(10);
  fireEvent.click(screen.getByRole("button", { name: "Stop Teleop" })); await advance(10);
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
  acknowledge(); await advance(30);
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(2);
  expect(backend.callsTo("/api/mcap/episodes/start")).toHaveLength(0);
});

test("an unresponsive episode save cannot delay Halt beyond one second", async () => {
  await open();
  fireEvent.click(screen.getByRole("button", { name: "Start Teleop" })); await advance(30);
  const fetch = window.fetch;
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).includes("/api/mcap/episodes/save")) return new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
    return fetch(input, init);
  });
  fireEvent.click(screen.getByRole("button", { name: "Stop Teleop" })); await advance(1100);
  expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
  expect(screen.getByRole("alert")).toBeInTheDocument();
});

test("the fixed visual stage shows all four cameras while the separate control rail owns preflight and recording", async () => {
  backend.statusExtra = { backend: "real", mock: false, torque_state: "off" };
  backend.route("GET", "/api/mcap/status", () => jsonResponse({
    recorder: { recording: false, service_ready: true }, active, episodes,
    cameras: Object.fromEntries(["obsbot", "camera_1", "camera_2", "realsense"].map((name) => [name, { live: true, actual_preview_fps: 30 }])),
  }));
  backend.route("GET", "/api/preflight/latest", () => jsonResponse({
    ready_for_hardware: false, created_at: "2026-10-27T10:00:00Z",
    checks: [
      { key: "rest.follower", title: "Rest pose", status: "fail", blocking: true, message: "Rest pose needs review", remediation: "Record the supported rest." },
      { key: "storage", title: "Storage", status: "pass", blocking: false, message: "Space available" },
    ],
  }));
  render(<MemoryRouter><TelemetryProvider><SafetyProvider><McapWorkflow readinessInitiallyOpen /></SafetyProvider></TelemetryProvider></MemoryRouter>);
  await advance(40);
  const visuals = screen.getByTestId("operate-visuals");
  const controls = screen.getByTestId("operate-controls");
  expect(visuals).toHaveClass("lg:overflow-hidden");
  expect(controls).toHaveClass("lg:overflow-y-auto");
  expect(screen.getByTestId("operate-workflow")).toHaveClass("lg:grid-cols-[minmax(0,3fr)_minmax(280px,1fr)]");
  expect(within(visuals).getAllByRole("img", { name: /arm camera$/ })).toHaveLength(4);
  expect(within(visuals).getByRole("region", { name: "Follower arm" })).toBeInTheDocument();
  expect(within(controls).getByRole("button", { name: "Start Teleop" })).toBeDisabled();
  expect(within(controls).getByRole("button", { name: "Stop Teleop" })).toBeEnabled();
  expect(within(controls).getByText("Rest pose needs review")).toBeInTheDocument();
  expect(within(controls).getByText("Space available")).toBeInTheDocument();
  expect(within(controls).getByText(/^Report:/)).toBeInTheDocument();
  fireEvent.click(within(controls).getByRole("button", { name: "Refresh report" }));
  await advance(20);
  expect(backend.callsTo("/api/preflight", "POST")).toHaveLength(0);
  expect(backend.callsTo("/api/teleop/start")).toHaveLength(0);
  expect(backend.callsTo("/api/calibration/pose/follower", "POST")).toHaveLength(0);
});

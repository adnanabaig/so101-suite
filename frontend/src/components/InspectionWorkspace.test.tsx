import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import InspectionWorkspace from "./InspectionWorkspace";

const state = vi.hoisted(() => ({ mode: "idle", stale: false, timeline: vi.fn(), snapshot: vi.fn(), playback: vi.fn(), post: vi.fn(), get: vi.fn() }));
vi.mock("./safety/SafetyProvider", () => ({ useSafety: () => ({ status: { mode: state.mode }, statusStale: state.stale }) }));
vi.mock("./RobotTwin", () => ({ default: ({ observation }: { observation: unknown }) => <div data-testid="recorded-twin">{JSON.stringify(observation)}</div> }));
vi.mock("recharts", () => ({ ResponsiveContainer: () => <div>Tracking chart</div>, CartesianGrid: () => null, Legend: () => null, Line: () => null, LineChart: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null }));
vi.mock("../api", () => ({
  api: { get: state.get, post: state.post },
  getControlTicks: async () => ({ stream_id: "stream", last_seq: 0, missing_before_buffer: 0, ticks: [] }),
  getMcapReviews: async () => ({ recordings: [{ episode: { id: "a".repeat(32), task: "tuning", status: "saved" }, recording_available: true, review: null }] }),
  getMcapTimeline: state.timeline, getMcapSnapshot: state.snapshot, startMcapPlayback: state.playback,
  getCachedCalibrationPose: async () => null,
}));

async function settle() { await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); }); }
beforeEach(() => {
  state.mode = "idle"; state.stale = false; vi.clearAllMocks();
  state.get.mockResolvedValue({ arms: {} });
  state.snapshot.mockResolvedValue({ record: null });
  state.playback.mockResolvedValue({ playback_id: "recorded-session", delay_seconds: 0.3 });
  state.timeline.mockResolvedValue({ episode: { id: "a".repeat(32), task: "tuning" }, duration_seconds: 2, sample_count: 61, display_sample_count: 61,
    measured_control_hz: 30, max_control_gap_seconds: 0.034, cameras: ["obsbot"], topics: { "/robot/health": 3 }, profile: { limits: { wrist_roll: [-160, 160] } },
    ticks: [{ seconds: 0, timestamp_ns: 1000000000, observation: { "wrist_roll.pos": 159 }, action: { "wrist_roll.pos": 160 }, desired_action: { "wrist_roll.pos": 175 } }],
  });
});

test("recorded playback preserves measured and command distinctions without motor control posts", async () => {
  render(<MemoryRouter><InspectionWorkspace initialTab="recordings" /></MemoryRouter>); await settle();
  fireEvent.change(screen.getByRole("combobox", { name: "Recording" }), { target: { value: "a".repeat(32) } }); await settle();
  expect(state.timeline).toHaveBeenCalledWith("a".repeat(32));
  expect(screen.getByTestId("recorded-twin")).toHaveTextContent('"wrist_roll.pos":159');
  expect(screen.getByText("Position limit")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Play recording" })); await settle();
  expect(state.playback).toHaveBeenCalledWith("a".repeat(32), 0, 1);
  expect(screen.getByRole("img", { name: "Recorded obsbot" }).getAttribute("src")).toContain("playback_id=recorded-session");
  fireEvent.click(screen.getByRole("button", { name: "Pause playback" }));
  expect(state.post).not.toHaveBeenCalled();
});

test("starting motion removes recorded playback and does not turn replay into robot commands", async () => {
  const view = render(<MemoryRouter><InspectionWorkspace initialTab="recordings" /></MemoryRouter>); await settle();
  fireEvent.change(screen.getByRole("combobox", { name: "Recording" }), { target: { value: "a".repeat(32) } }); await settle();
  state.mode = "teleop";
  view.rerender(<MemoryRouter><InspectionWorkspace initialTab="recordings" /></MemoryRouter>);
  expect(screen.queryByTestId("recorded-twin")).not.toBeInTheDocument();
  expect(screen.getByText(/Stop motion before loading/)).toBeVisible();
  expect(state.post).not.toHaveBeenCalled();
});

test("missing camera frame is explicit and live inspection labels paused sampling", async () => {
  render(<MemoryRouter><InspectionWorkspace initialTab="recordings" /></MemoryRouter>); await settle();
  fireEvent.change(screen.getByRole("combobox", { name: "Recording" }), { target: { value: "a".repeat(32) } }); await settle();
  fireEvent.error(screen.getByRole("img", { name: "Recorded obsbot" }));
  expect(screen.getByText(/No recorded frame available/)).toBeVisible();
  fireEvent.click(screen.getByRole("tab", { name: "Live data" })); await settle();
  expect(screen.getByText(/Control sampling paused/)).toBeVisible();
});

test("Foxglove button opens the selected finalized episode and reports a launch request", async () => {
  state.post.mockResolvedValue({ launch_requested: true });
  render(<MemoryRouter><InspectionWorkspace initialTab="recordings" /></MemoryRouter>); await settle();
  fireEvent.change(screen.getByRole("combobox", { name: "Recording" }), { target: { value: "a".repeat(32) } }); await settle();
  fireEvent.click(screen.getByRole("button", { name: "Open this episode in Foxglove" })); await settle();
  expect(state.post).toHaveBeenCalledWith(`/api/mcap/reviews/${"a".repeat(32)}/open-foxglove`, {}, 60000);
  expect(screen.getByText("Foxglove launch requested for aaaaaaaa.")).toBeVisible();
  fireEvent.click(screen.getByRole("tab", { name: "Quality report" }));
  expect(screen.getByText(/Human task acceptance remains separate/)).toBeVisible();
});

test("recorded health never requests a future bucket and fractional end frames stay within the episode", async () => {
  const original = await state.timeline();
  state.timeline.mockResolvedValue({ ...original, duration_seconds: 2.0006 });
  render(<MemoryRouter><InspectionWorkspace initialTab="recordings" /></MemoryRouter>); await settle();
  fireEvent.change(screen.getByRole("combobox", { name: "Recording" }), { target: { value: "a".repeat(32) } }); await settle();
  fireEvent.change(screen.getByRole("slider", { name: "Recorded timeline" }), { target: { value: "0.126" } }); await settle();
  expect(state.snapshot.mock.calls.every(call => call[2] <= 0.126)).toBe(true);
  fireEvent.change(screen.getByRole("slider", { name: "Recorded timeline" }), { target: { value: "2.0006" } }); await settle();
  expect(screen.getByRole("img", { name: "Recorded obsbot" })).toHaveAttribute("src", `/api/mcap/reviews/${"a".repeat(32)}/frames/obsbot?seconds=2.0006`);
});

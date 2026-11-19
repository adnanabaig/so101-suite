import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { getCalibrationHistory, type CalibrationAttempt } from "../api";
import CalibrationHistory from "./CalibrationHistory";

vi.mock("../api", () => ({ getCalibrationHistory: vi.fn() }));

const failed: CalibrationAttempt = {
  id: "failed-attempt", event_id: 2, arm: "follower", backend: "real",
  device_id: "adnan-suite-follower", sample_seconds: 60,
  started_at: "2026-10-18T22:00:00Z", finished_at: "2026-10-18T22:01:00Z",
  result: "fail", saved: false, affected_motors: ["shoulder_pan", "elbow_flex"],
  error: "encoder ranges outside 0..4095: shoulder_pan=-542; elbow_flex=-182",
  diagnostics: { starting_positions: { shoulder_pan: 2047 }, range_mins: { shoulder_pan: -542 } },
  log: [{ timestamp: "2026-10-18T22:01:00Z", level: "error", message: "exact original error <raw>" }],
};

beforeEach(() => {
  vi.mocked(getCalibrationHistory).mockReset();
});

test("renders pass/fail, affected motors, exact errors, and expandable raw diagnostics", async () => {
  const passed: CalibrationAttempt = {
    ...failed, id: "passed-attempt", arm: "leader", result: "pass", saved: true,
    affected_motors: [], error: null, diagnostics: {}, log: [],
  };
  vi.mocked(getCalibrationHistory).mockResolvedValue({ attempts: [failed, passed], limit: 100 });
  render(<CalibrationHistory />);
  const table = await screen.findByRole("table");
  expect(within(table).getByRole("columnheader", { name: "Pass / fail" })).toBeVisible();
  expect(within(table).getByText("FAIL")).toBeVisible();
  expect(within(table).getByText("PASS")).toBeVisible();
  expect(within(table).getByText("shoulder_pan")).toBeVisible();
  expect(within(table).getByText("elbow_flex")).toBeVisible();
  expect(within(table).getByText(failed.error!)).toBeVisible();
  fireEvent.click(within(table).getAllByText("View full log")[0]);
  expect(screen.getByText(/exact original error <raw>/)).toBeVisible();
  expect(screen.getByText(/"starting_positions"/)).toBeVisible();
});

test("distinguishes a saved file from a successful attempt", async () => {
  vi.mocked(getCalibrationHistory).mockResolvedValue({ attempts: [{ ...failed, saved: true }], limit: 100 });
  render(<CalibrationHistory />);
  expect(await screen.findByText("FAIL")).toBeVisible();
  expect(screen.getByText("File saved; attempt still failed.")).toBeVisible();
});

test("displays empty history and reports missing historical metadata honestly", async () => {
  vi.mocked(getCalibrationHistory).mockResolvedValueOnce({ attempts: [], limit: 100 });
  const mounted = render(<CalibrationHistory />);
  expect(await screen.findByText("No calibration attempts recorded yet.")).toBeVisible();
  mounted.unmount();
  vi.mocked(getCalibrationHistory).mockResolvedValue({
    attempts: [{ ...failed, backend: null, sample_seconds: null, device_id: null }], limit: 100,
  });
  render(<CalibrationHistory />);
  expect(await screen.findByText("Backend not recorded")).toBeVisible();
  expect(screen.getByText("Duration not recorded")).toBeVisible();
});

test("a failed history request shows an error instead of saying no attempts exist", async () => {
  vi.mocked(getCalibrationHistory).mockRejectedValue(new Error("storage unavailable"));
  render(<CalibrationHistory />);
  expect(await screen.findByRole("alert")).toHaveTextContent("storage unavailable");
  expect(screen.queryByText("No calibration attempts recorded yet.")).not.toBeInTheDocument();
});

test("refresh fetches persisted history again and shows an incomplete attempt", async () => {
  vi.mocked(getCalibrationHistory).mockResolvedValueOnce({ attempts: [], limit: 100 });
  render(<CalibrationHistory />);
  await screen.findByText("No calibration attempts recorded yet.");
  vi.mocked(getCalibrationHistory).mockResolvedValue({
    attempts: [{ ...failed, result: "incomplete", error: null, finished_at: null }], limit: 100,
  });
  fireEvent.click(screen.getByRole("button", { name: "Refresh history" }));
  expect(await screen.findByText("INCOMPLETE")).toBeVisible();
  expect(screen.getByText("No completion recorded.")).toBeVisible();
});

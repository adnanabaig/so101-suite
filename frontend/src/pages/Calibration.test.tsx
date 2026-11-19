import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { getCalibrationStatus, startCalibration } from "../api";
import Calibration from "./Calibration";

vi.mock("../api", () => ({ getCalibrationStatus: vi.fn(), startCalibration: vi.fn() }));
vi.mock("../components/CalibrationHistory", () => ({
  default: () => <div>Persisted attempt history</div>,
}));

beforeEach(() => {
  vi.mocked(getCalibrationStatus).mockReset();
});

function status(message: string, inProgress = true) {
  vi.mocked(getCalibrationStatus).mockResolvedValue({
    arm: "follower", in_progress: inProgress, fraction: 0.02, message,
    has_saved: false, device_id: "", created_at: null, ranges: {},
  });
}

test("holds the midpoint pose during setup instead of prematurely requesting a sweep", async () => {
  status("starting");
  render(<Calibration />);
  expect(await screen.findByText("Preparing calibration — hold the midpoint pose still")).toBeVisible();
  expect(screen.queryByText("Torque is off — sweep every joint now")).not.toBeInTheDocument();
  expect(screen.getByText("Persisted attempt history")).toBeVisible();
});

test.each([
  "midpoint captured (homing read-back verified); now sweep every joint through its range",
  "sweep every joint through its range",
])("requests a sweep only after midpoint capture: %s", async (message) => {
  status(message);
  render(<Calibration />);
  expect(await screen.findByText("Torque is off — sweep every joint now")).toBeVisible();
});

test("a failed attempt does not continue asking the operator to sweep", async () => {
  status("failed: homing read-back mismatch", false);
  render(<Calibration />);
  expect(await screen.findByText("failed: homing read-back mismatch")).toBeVisible();
  expect(screen.queryByText("Torque is off — sweep every joint now")).not.toBeInTheDocument();
});

test("starts with gripper and resets confirmations when choosing another joint or scope", async () => {
  status("", false);
  render(<Calibration />);
  const joint = await screen.findByRole("combobox", { name: "Calibration joint" });
  expect(joint).toHaveValue("gripper");
  expect(screen.getByRole("option", { name: /gripper · ID 6/ })).toBeVisible();
  const checkboxes = screen.getAllByRole("checkbox");
  checkboxes.forEach((checkbox) => fireEvent.click(checkbox));
  expect(screen.getByRole("button", { name: "Capture gripper baseline" })).toBeEnabled();
  fireEvent.change(joint, { target: { value: "elbow_flex" } });
  expect(screen.getByRole("button", { name: "Capture elbow_flex baseline" })).toBeDisabled();
  expect(screen.getAllByRole("checkbox").every((checkbox) => !(checkbox as HTMLInputElement).checked)).toBe(true);
  fireEvent.change(screen.getByRole("combobox", { name: "Calibration scope" }), { target: { value: "arm" } });
  expect(screen.queryByRole("combobox", { name: "Calibration joint" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Begin calibrated sweep" })).toBeDisabled();
});

test("posts the selected joint after operator confirmations and countdown", async () => {
  status("", false);
  vi.mocked(startCalibration).mockResolvedValue({ ok: true, detail: "started" });
  render(<Calibration />);
  const button = await screen.findByRole("button", { name: "Capture gripper baseline" });
  screen.getAllByRole("checkbox").forEach((checkbox) => fireEvent.click(checkbox));
  vi.useFakeTimers();
  try {
    fireEvent.click(button);
    await vi.advanceTimersByTimeAsync(3000);
    expect(startCalibration).toHaveBeenCalledWith({ arm: "follower", sample_seconds: 12, joint: "gripper" });
  } finally {
    vi.useRealTimers();
  }
});

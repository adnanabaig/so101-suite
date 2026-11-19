import { render, screen, within } from "@testing-library/react";
import { expect, test } from "vitest";
import type { CalibrationLiveStatus } from "../api";
import CalibrationLive from "./CalibrationLive";

function frame(changes: Partial<CalibrationLiveStatus> = {}): CalibrationLiveStatus {
  return {
    phase: "sampling", sequence: 3, sample_count: 2, elapsed_seconds: 1,
    age_seconds: 0.1, frozen: false,
    positions: { shoulder_pan: 1000, shoulder_lift: 4444, elbow_flex: -197, gripper: 2048 },
    starting_positions: { shoulder_pan: 2047, shoulder_lift: 2047, elbow_flex: 2047 },
    range_mins: { shoulder_pan: 745, shoulder_lift: 2006, elbow_flex: -197, gripper: 2047 },
    range_maxes: { shoulder_pan: 3332, shoulder_lift: 4444, elbow_flex: 2064, gripper: 2048 },
    homing_offsets_requested: { shoulder_pan: -162, shoulder_lift: -653, elbow_flex: 109, wrist_roll: -468 },
    homing_readback: Object.fromEntries(
      [["shoulder_pan", -162], ["shoulder_lift", -653], ["elbow_flex", 109], ["wrist_roll", -468]].map(
        ([motor, offset]) => [motor, { homing_offset: Number(offset), range_min: 0, range_max: 4095 }],
      ),
    ),
    ...changes,
  };
}
function row(motor: string) {
  return within(screen.getByRole("rowheader", { name: motor }).closest("tr")!);
}

test("shows per-motor live values, immediate range failures and pending short spans", () => {
  render(<CalibrationLive live={frame()} />);
  expect(row("shoulder_pan").getByText("Valid so far")).toBeVisible();
  expect(row("shoulder_pan").getByText("2587")).toBeVisible();
  expect(row("shoulder_lift").getByText("FAIL")).toBeVisible();
  expect(row("elbow_flex").getByText("FAIL")).toBeVisible();
  expect(row("gripper").getByText("Pending")).toBeVisible();
  expect(row("wrist_roll").getByText("0 (fixed)")).toBeVisible();
  expect(screen.queryByText("PASS")).not.toBeInTheDocument();
});

test("keeps final attempted values visible and does not mark other motors as passed after failure", () => {
  render(<CalibrationLive live={frame({ phase: "failed", frozen: true, error: "range failure" })} />);
  expect(screen.getByText(/Frozen final readings/)).toBeVisible();
  expect(row("shoulder_pan").getByText("Valid so far")).toBeVisible();
  expect(row("gripper").getByText("FAIL")).toBeVisible();
  expect(screen.queryByText("PASS")).not.toBeInTheDocument();
});

test("flags a homing readback mismatch and an attributed homing write failure", () => {
  render(<CalibrationLive live={frame({
    phase: "failed", affected_motors: ["elbow_flex"],
    homing_readback: { shoulder_pan: { homing_offset: 999, range_min: 0, range_max: 4095 } },
    range_mins: {}, range_maxes: {},
  })} />);
  expect(row("shoulder_pan").getByText("Homing read-back mismatch")).toBeVisible();
  expect(row("elbow_flex").getByText("Reported motor failure")).toBeVisible();
});

test("labels stalled worker readings and connection errors", () => {
  render(<CalibrationLive live={frame({ age_seconds: 5 })} error="network lost" />);
  expect(screen.getByRole("alert")).toHaveTextContent("Readings may be stale");
  expect(screen.getByText(/Stale readings/)).toBeVisible();
  expect(row("shoulder_pan").getByText("Stale · Valid so far")).toBeVisible();
});

test("does not invent raw values when backend has no live samples", () => {
  render(<CalibrationLive />);
  expect(screen.getByText(/No live encoder samples/)).toBeVisible();
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});

test("only reports pass after completion with verified homing and adequate span", () => {
  render(<CalibrationLive live={frame({ phase: "complete", frozen: true })} />);
  expect(row("shoulder_pan").getByText("PASS")).toBeVisible();
  expect(row("gripper").getByText("FAIL")).toBeVisible();
});

test("a single wrist-roll baseline shows measured extremes and excludes unrelated joints", () => {
  render(<CalibrationLive live={frame({
    joint: "wrist_roll", baseline_only: true,
    range_mins: { wrist_roll: 1000 }, range_maxes: { wrist_roll: 3000 },
  })} />);
  expect(row("wrist_roll").getByText("1000")).toBeVisible();
  expect(row("wrist_roll").getByText("3000")).toBeVisible();
  expect(screen.queryByRole("rowheader", { name: "gripper" })).not.toBeInTheDocument();
  expect(screen.queryByText("0 (fixed)")).not.toBeInTheDocument();
  expect(screen.getByText(/Joint baselines restore prior calibration/)).toBeVisible();
});

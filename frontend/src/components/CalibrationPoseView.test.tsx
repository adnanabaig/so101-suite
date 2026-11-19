import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { readCalibrationPose, type CalibrationPose } from "../api";
import CalibrationPoseView from "./CalibrationPoseView";
import { TWIN_JOINTS } from "./twin/types";

vi.mock("../api", () => ({ readCalibrationPose: vi.fn(), getCachedCalibrationPose: vi.fn().mockResolvedValue(null) }));
vi.mock("./RobotTwin", () => ({ default: ({ observation }: { observation?: Record<string, number> }) =>
  <div data-testid="pose-model">{observation ? JSON.stringify(observation) : "No physical sample"}</div>,
}));

function sample(): CalibrationPose {
  return { arm: "follower", backend: "real", captured_at: Date.now() / 1000,
    calibration_sha256: "installed-reference", complete: true,
    observation: Object.fromEntries(TWIN_JOINTS.map((name, index) => [`${name}.pos`, index * 10])),
    joints: Object.fromEntries(TWIN_JOINTS.map((name, index) => [name, {
      motor_id: index + 1, ticks: 2000, range_min: 100, range_max: 3900, issues: [],
    }])),
  };
}

beforeEach(() => { vi.mocked(readCalibrationPose).mockReset(); });

test("opening Recalibrate never reads a bus until setup and Read current pose", async () => {
  vi.mocked(readCalibrationPose).mockResolvedValue(sample());
  render(<CalibrationPoseView arm="follower" inProgress={false} onJoint={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Recalibrate" }));
  expect(readCalibrationPose).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Read current pose" })).toBeDisabled();
  expect(screen.getByTestId("pose-model")).toHaveTextContent("No physical sample");
  fireEvent.click(screen.getByLabelText(/Both arms are supported/));
  fireEvent.click(screen.getByRole("button", { name: "Read current pose" }));
  await waitFor(() => expect(readCalibrationPose).toHaveBeenCalledWith("follower"));
  expect(await screen.findByTestId("pose-model")).toHaveTextContent('"elbow_flex.pos":20');
  expect(screen.getByText(/0\/6 visual checks/)).toBeVisible();
  fireEvent.click(screen.getByLabelText("Motor 3 pose and direction match"));
  expect(screen.getByText("Visual match confirmed")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Stop reading pose" }));
  expect(screen.getByLabelText("Motor 3 pose and direction match")).toBeDisabled();
  expect(screen.queryByText("Visual match confirmed")).not.toBeInTheDocument();
});

test("a bad joint prevents a fabricated whole-arm pose and cannot turn green", async () => {
  const data = sample(); data.complete = false;
  data.joints.wrist_roll.issues = ["Current encoder reading is outside the installed range"];
  delete data.observation["wrist_roll.pos"];
  vi.mocked(readCalibrationPose).mockResolvedValue(data);
  const select = vi.fn();
  render(<CalibrationPoseView arm="leader" inProgress={false} onJoint={select} />);
  fireEvent.click(screen.getByRole("button", { name: "Recalibrate" }));
  fireEvent.click(screen.getByLabelText(/Both arms are supported/));
  fireEvent.click(screen.getByRole("button", { name: "Read current pose" }));
  expect(await screen.findByText(/Current encoder reading is outside/)).toBeVisible();
  expect(screen.getByTestId("pose-model")).toHaveTextContent("No physical sample");
  expect(screen.getByLabelText("Motor 5 pose and direction match")).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Check motor 5 endpoints" }));
  expect(select).toHaveBeenCalledWith("wrist_roll");
});

test("follower control samples drive the twin without manual bus reads or automatic green checks", async () => {
  const data = sample();
  const controlPose = { observation: data.observation, capturedAt: data.captured_at, live: true, active: true };
  const props = { arm: "follower" as const, inProgress: false, onJoint: vi.fn(), initiallyOpen: true, controllerActive: true };
  const { rerender } = render(<CalibrationPoseView {...props} controlPose={controlPose} />);
  expect(screen.getByTestId("pose-model")).toHaveTextContent('"elbow_flex.pos":20');
  expect(screen.getByText("Live follower encoders · shared control stream")).toBeVisible();
  fireEvent.click(screen.getByLabelText(/Both arms are supported/));
  expect(screen.getByRole("button", { name: "Read current pose" })).toBeDisabled();
  expect(screen.getByLabelText("Motor 3 pose and direction match")).toBeDisabled();
  expect(readCalibrationPose).not.toHaveBeenCalled();
  rerender(<CalibrationPoseView {...props} controlPose={{ ...controlPose, live: false }} />);
  expect(screen.getByText("Last control pose · live samples unavailable")).toBeVisible();
  expect(screen.getByTestId("pose-model")).toHaveTextContent('"elbow_flex.pos":20');
});

test("follower control samples are never relabeled as leader positions", () => {
  const data = sample();
  render(<CalibrationPoseView arm="leader" inProgress={false} onJoint={vi.fn()} initiallyOpen controllerActive
    controlPose={{ observation: data.observation, capturedAt: data.captured_at, live: true, active: true }} />);
  expect(screen.getByTestId("pose-model")).toHaveTextContent("No physical sample");
  expect(screen.getByText(/control stream contains follower readings only/)).toBeVisible();
  expect(readCalibrationPose).not.toHaveBeenCalled();
});

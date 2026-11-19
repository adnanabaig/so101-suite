import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { useTelemetry } from "../hooks/useTelemetry";
import { useSafety } from "../components/safety/SafetyProvider";
import { TWIN_JOINTS } from "../components/twin/types";
import Simulation from "./Simulation";

vi.mock("../hooks/useTelemetry", () => ({ useTelemetry: vi.fn() }));
vi.mock("../components/safety/SafetyProvider", () => ({ useSafety: vi.fn() }));
vi.mock("../api", () => ({
  getCachedCalibrationPose: vi.fn().mockResolvedValue(null),
  readCalibrationPose: vi.fn(),
}));
vi.mock("../components/RobotTwin", () => ({ default: ({ observation, stale, sourceLabel }: {
  observation?: Record<string, number>; stale?: boolean; sourceLabel?: string;
}) => <div data-testid="physical-twin" data-stale={String(stale)} data-source={sourceLabel}>
  {observation ? JSON.stringify(observation) : "No measured pose"}
</div> }));

let live = true;
let backend = "real";
let observation: Record<string, number>;

beforeEach(() => {
  live = true;
  backend = "real";
  observation = Object.fromEntries(TWIN_JOINTS.map((joint, index) => [`${joint}.pos`, index * 15]));
  vi.mocked(useTelemetry).mockImplementation(() => ({
    telemetry: { t: 100, mode: "teleop", observation },
    hasLiveObservation: live,
  }) as ReturnType<typeof useTelemetry>);
  vi.mocked(useSafety).mockImplementation(() => ({
    status: { backend, mock: backend === "mock" }, statusStale: false,
  }) as ReturnType<typeof useSafety>);
});

test("Digital twin defaults to real measured follower motion and retains a visibly stale pose", () => {
  const { rerender } = render(<Simulation />);
  expect(screen.getByRole("heading", { name: "Digital twin" })).toBeVisible();
  expect(screen.getByTestId("physical-twin")).toHaveTextContent('"elbow_flex.pos":30');
  expect(screen.getByTestId("physical-twin")).toHaveAttribute("data-source", "follower · control encoders");
  expect(screen.getByTestId("physical-twin")).toHaveAttribute("data-stale", "false");
  live = false;
  observation = {};
  rerender(<Simulation />);
  expect(screen.getByTestId("physical-twin")).toHaveTextContent('"elbow_flex.pos":30');
  expect(screen.getByTestId("physical-twin")).toHaveAttribute("data-stale", "true");
  expect(screen.getByText("Last control pose · live samples unavailable")).toBeVisible();
});

test("selecting leader does not reuse the follower control observation", () => {
  render(<Simulation />);
  fireEvent.change(screen.getByLabelText("Visualized arm"), { target: { value: "leader" } });
  expect(screen.getByTestId("physical-twin")).toHaveTextContent("No measured pose");
  expect(screen.getByText(/control stream contains follower readings only/)).toBeVisible();
});

test.each(["mock", "sim"])("%s telemetry cannot populate the physical arm", (value) => {
  backend = value;
  render(<Simulation />);
  expect(screen.getByTestId("physical-twin")).toHaveTextContent("No measured pose");
  expect(screen.queryByText("Live follower encoders · shared control stream")).not.toBeInTheDocument();
});

test("incomplete controller samples cannot fabricate a six-joint physical pose", () => {
  delete observation["wrist_roll.pos"];
  render(<Simulation />);
  expect(screen.getByTestId("physical-twin")).toHaveTextContent("No measured pose");
  expect(screen.getByRole("button", { name: "Read current pose" })).toBeDisabled();
});

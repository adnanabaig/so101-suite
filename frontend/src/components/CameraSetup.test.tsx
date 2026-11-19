import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { getCameraPreviewStatus, startCameraPreview, stopCameraPreview } from "../api";
import CameraSetup from "./CameraSetup";

vi.mock("../api", () => ({ getCameraPreviewStatus: vi.fn(), startCameraPreview: vi.fn(), stopCameraPreview: vi.fn() }));
vi.mock("./safety/SafetyProvider", () => ({ useSafety: () => ({ status: { live_cameras: ["camera_1=0"] } }) }));
vi.mock("./CameraView", () => ({ default: ({ cam, active }: { cam: string; active: boolean }) => <div>{cam} {active ? "live" : "inactive"}</div> }));

const idle = { active: false, error: null, cameras: ["camera_1=0"], devices: {} };
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getCameraPreviewStatus).mockResolvedValue(idle);
  vi.mocked(startCameraPreview).mockResolvedValue({ ...idle, active: true });
  vi.mocked(stopCameraPreview).mockResolvedValue(idle);
});

test("explicit start and stop use camera-only API and show negotiated mode", async () => {
  vi.mocked(startCameraPreview).mockResolvedValue({ ...idle, active: true, devices: { "camera_1=0": { device: { streams: { color: { width: 640, height: 480, fps: 15, fourcc: "YUYV" } } } } } });
  render(<CameraSetup />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Start camera preview" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Start camera preview" }));
  expect(await screen.findByRole("button", { name: "Stop camera preview" })).toBeEnabled();
  expect(screen.getByText(/negotiated 640 × 480 · 15 fps · YUYV/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Stop camera preview" }));
  await waitFor(() => expect(stopCameraPreview).toHaveBeenCalledOnce());
});

test("capture refusal is visible and never claims preview started", async () => {
  vi.mocked(startCameraPreview).mockRejectedValue(new Error("robot is busy"));
  render(<CameraSetup />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Start camera preview" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Start camera preview" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("robot is busy");
  expect(screen.getByRole("button", { name: "Start camera preview" })).toBeEnabled();
});

test("unavailable status disables capture control", async () => {
  vi.mocked(getCameraPreviewStatus).mockRejectedValue(new Error("server unavailable"));
  render(<CameraSetup />);
  expect(await screen.findByRole("alert")).toHaveTextContent("server unavailable");
  expect(screen.getByRole("button", { name: "Start camera preview" })).toBeDisabled();
});

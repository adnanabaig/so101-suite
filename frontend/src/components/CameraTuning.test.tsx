import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import CameraTuning from "./CameraTuning";

const state = vi.hoisted(() => ({ mode: "idle", stale: false, get: vi.fn(), post: vi.fn() }));
vi.mock("../api", () => ({ api: { get: state.get, post: state.post } }));
vi.mock("./safety/SafetyProvider", () => ({ useSafety: () => ({ status: { mode: state.mode }, statusStale: state.stale }) }));
async function settle() { await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); }); }
beforeEach(() => {
  state.mode = "idle"; state.stale = false; vi.clearAllMocks();
  state.get.mockResolvedValue({ live: true, actual_capture_fps: 10, image_quality: {}, camera_profile: { identity: "camera-one", exposure_unit: "100 microseconds", controls: { brightness: { label: "Brightness", value: 0, minimum: 0, maximum: 10, step: 1, writable: true }, exposure_absolute: { label: "Exposure", value: 313, minimum: 1, maximum: 10000, step: 1, writable: false, inactive: true } } } });
  state.post.mockResolvedValue({});
});

test("supported controls show readback and persist only explicit changes for the selected device", async () => {
  render(<CameraTuning />); await settle();
  expect(screen.getByRole("spinbutton", { name: "exposure_absolute" })).toBeDisabled();
  expect(screen.queryByRole("spinbutton", { name: "unsupported_gain" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("spinbutton", { name: "brightness" }), { target: { value: "2" } });
  fireEvent.click(screen.getByRole("button", { name: "Apply and save camera settings" })); await settle();
  expect(state.post).toHaveBeenCalledWith("/api/mcap/cameras/camera_1/controls", { values: { brightness: 2 } });
  expect(screen.getByRole("button", { name: "Apply and save camera settings" })).toBeDisabled();
});

test("teleop or stale workstation status prevents image-setting writes", async () => {
  state.mode = "teleop";
  const view = render(<CameraTuning />); await settle();
  expect(screen.getByRole("spinbutton", { name: "brightness" })).toBeDisabled();
  state.mode = "idle"; state.stale = true;
  view.rerender(<CameraTuning />);
  expect(screen.getByRole("button", { name: "Apply and save camera settings" })).toBeDisabled();
  expect(state.post).not.toHaveBeenCalled();
});

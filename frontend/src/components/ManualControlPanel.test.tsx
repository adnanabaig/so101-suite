import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { TelemetryProvider } from "../hooks/useTelemetry";
import {
  advance,
  deferred,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  type FakeBackend,
  type RecordedCall,
} from "../test/harness";
import ManualControlPanel, { disarmHelpText, readStartPose } from "./ManualControlPanel";
import { SafetyProvider } from "./safety/SafetyProvider";

const FULL_POSE = {
  "shoulder_pan.pos": 1,
  "shoulder_lift.pos": 2,
  "elbow_flex.pos": 3,
  "wrist_flex.pos": 4,
  "wrist_roll.pos": 5,
  "gripper.pos": 40,
};

let backend: FakeBackend;
let observation: Record<string, number>;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  observation = { ...FULL_POSE };
  backend.route("POST", "/api/control/lease", () =>
    jsonResponse({ id: "lease-1", source: "manual-ui", server_time: new Date().toISOString() }),
  );
  backend.route("POST", "/api/control/lease/lease-1/heartbeat", () =>
    jsonResponse({ id: "lease-1", source: "manual-ui", server_time: new Date().toISOString() }),
  );
  backend.route("DELETE", "/api/control/lease/lease-1", () => jsonResponse({ ok: true, detail: "" }));
  backend.route("POST", "/api/control/observe", () =>
    jsonResponse({ observation, captured_at: new Date().toISOString() }),
  );
  const accept = (call: RecordedCall) =>
    jsonResponse({ ok: true, detail: "", accepted: true, targets: (call.body as { targets: unknown }).targets });
  backend.route("POST", "/api/control/arm-command", accept);
  backend.route("POST", "/api/control/command", accept);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderPanel() {
  render(
    <TelemetryProvider>
      <SafetyProvider>
        <ManualControlPanel />
      </SafetyProvider>
    </TelemetryProvider>,
  );
  await advance(10);
  fireEvent.click(screen.getByRole("button", { name: /Acquire manual lease/ }));
  await advance(10);
}

describe("ManualControlPanel", () => {
  test("the keyboard deadman arms atomically, streams, and stops on release", async () => {
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    expect(deadman).toBeEnabled();

    fireEvent.keyDown(deadman, { key: " " });
    await advance(10);
    const [armCommand] = backend.callsTo("/api/control/arm-command");
    expect(armCommand.body).toMatchObject({
      lease_id: "lease-1",
      source: "manual-ui",
      sequence: 0,
      ttl_ms: 300,
      deadman: true,
      targets: { shoulder_pan: 1, shoulder_lift: 2, elbow_flex: 3, wrist_flex: 4, wrist_roll: 5, gripper: 40 },
    });
    expect(typeof (armCommand.body as { sent_at: unknown }).sent_at).toBe("string");
    expect(deadman).toHaveAttribute("aria-pressed", "true");

    // Auto-repeat does not re-arm.
    fireEvent.keyDown(deadman, { key: " ", repeat: true });
    await advance(300);
    expect(backend.callsTo("/api/control/arm-command")).toHaveLength(1);
    const streamed = backend.callsTo("/api/control/command");
    expect(streamed.length).toBeGreaterThanOrEqual(2);
    expect(streamed.map((call) => (call.body as { sequence: number }).sequence)).toEqual(
      streamed.map((_, index) => index + 1),
    );

    fireEvent.keyUp(deadman, { key: " " });
    await advance(10);
    expect(backend.callsTo("/api/control/disarm").map((call) => call.body)).toEqual([
      { lease_id: "lease-1" },
    ]);
    const afterRelease = backend.callsTo("/api/control/command").length;
    await advance(500);
    expect(backend.callsTo("/api/control/command")).toHaveLength(afterRelease);
    expect(deadman).toHaveAttribute("aria-pressed", "false");
  });

  test("a pointer release elsewhere sends nothing when no hold is in progress (F3)", async () => {
    await renderPanel();
    fireEvent.pointerUp(window);
    fireEvent.pointerCancel(window);
    await advance(10);
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(0);
    expect(backend.callsTo("/api/robot/halt")).toHaveLength(0);
    expect(backend.callsTo("/api/robot/stop")).toHaveLength(0);
  });

  test("a pointer release outside the button ends a hold", async () => {
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.pointerDown(deadman, { isPrimary: true, button: 0, pointerId: 1 });
    await advance(150);
    expect(backend.callsTo("/api/control/arm-command")).toHaveLength(1);
    fireEvent.pointerUp(window);
    await advance(10);
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
  });

  test("a missing joint locks motion and names the joint", async () => {
    delete observation["wrist_roll.pos"];
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Start pose incomplete" });
    expect(deadman).toBeDisabled();
    expect(screen.getByText(/did not report wrist_roll/)).toBeInTheDocument();
  });

  test("a rejected command stops motion and shows why", async () => {
    await renderPanel();
    backend.route("POST", "/api/control/command", () => jsonResponse({ detail: "command expired" }, 409));
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.keyDown(deadman, { key: "Enter" });
    await advance(150);
    expect(screen.getByRole("alert")).toHaveTextContent("command expired");
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
  });

  test("Disarm help and stop wording follow the advertised stops (C11)", async () => {
    await renderPanel();
    expect(screen.getByText(disarmHelpText(false))).toBeInTheDocument();
    expect(screen.getByText(/this server turns torque off/)).toBeInTheDocument();
  });

  test("with Halt and Collapse advertised, the copy says the arm holds and lowers to rest", async () => {
    backend.statusExtra = { stop_kinds: ["estop", "halt", "collapse"] };
    await renderPanel();
    expect(screen.getByText(disarmHelpText(true))).toBeInTheDocument();
    expect(screen.getByText(/the arm halts and holds its position/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Disarm" }));
    await advance(10);
    expect(backend.callsTo("/api/control/disarm").map((call) => call.body)).toEqual([
      { lease_id: "lease-1" },
    ]);
  });
});

describe("ManualControlPanel jog keys", () => {
  const slider = (joint: string) => screen.getByRole("slider", { name: `${joint} target` });
  const lastCommand = () => {
    const calls = backend.callsTo("/api/control/command");
    return (calls[calls.length - 1].body as { targets: Record<string, number> }).targets;
  };

  test("arrows do nothing, and keep their default, unless the deadman is held", async () => {
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    // fireEvent returns false only when something called preventDefault.
    expect(fireEvent.keyDown(deadman, { key: "ArrowRight" })).toBe(true);
    expect(fireEvent.keyDown(document.body, { key: "ArrowDown" })).toBe(true);
    await advance(10);
    expect(slider("shoulder_pan")).toHaveValue("1");
    expect(backend.callsTo("/api/control/command")).toHaveLength(0);
  });

  test("while held, Left and Right move the marked joint's streamed target, Shift by 5", async () => {
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    expect(slider("shoulder_pan").closest("label")).toHaveAttribute("data-jog-selected", "true");
    fireEvent.keyDown(deadman, { key: " " });
    await advance(10);
    expect(deadman).toHaveAttribute("aria-pressed", "true");

    expect(fireEvent.keyDown(deadman, { key: "ArrowRight" })).toBe(false);
    await advance(150);
    expect(lastCommand()).toMatchObject({ shoulder_pan: 2, shoulder_lift: 2, gripper: 40 });
    expect(screen.getByTestId("jog-notice")).toHaveTextContent("shoulder_pan target 2.0 deg");

    fireEvent.keyDown(deadman, { key: "ArrowRight", shiftKey: true });
    fireEvent.keyDown(deadman, { key: "ArrowLeft" });
    await advance(150);
    expect(lastCommand().shoulder_pan).toBe(6);
    expect(slider("shoulder_pan")).toHaveValue("6");

    // Down picks the next joint; the jog then moves only that one.
    fireEvent.keyDown(deadman, { key: "ArrowDown" });
    await advance(10);
    expect(screen.getByTestId("jog-notice")).toHaveTextContent("Jogging shoulder_lift");
    expect(slider("shoulder_lift").closest("label")).toHaveAttribute("data-jog-selected", "true");
    fireEvent.keyDown(deadman, { key: "ArrowLeft", shiftKey: true });
    await advance(150);
    expect(lastCommand()).toMatchObject({ shoulder_pan: 6, shoulder_lift: -3 });
    // The arm stays armed through the jog: one arm-command, then a stream.
    expect(backend.callsTo("/api/control/arm-command")).toHaveLength(1);
  });

  test("while the server is still arming, the arrows do nothing yet", async () => {
    const arming = deferred<Response>();
    backend.route("POST", "/api/control/arm-command", () => arming.promise);
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.keyDown(deadman, { key: " " });
    await advance(10);
    expect(deadman).toHaveAttribute("aria-pressed", "false");
    expect(fireEvent.keyDown(deadman, { key: "ArrowRight" })).toBe(true);
    arming.resolve(
      jsonResponse({ ok: true, detail: "", accepted: true, targets: {} }),
    );
    await advance(150);
    expect(deadman).toHaveAttribute("aria-pressed", "true");
    expect(slider("shoulder_pan")).toHaveValue("1");
    expect(lastCommand().shoulder_pan).toBe(1);
  });

  test("targets stop at the joint's limits and selection stops at the ends", async () => {
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.keyDown(deadman, { key: "Enter" });
    await advance(10);
    fireEvent.keyDown(deadman, { key: "ArrowUp" });
    expect(slider("shoulder_pan").closest("label")).toHaveAttribute("data-jog-selected", "true");
    for (let index = 0; index < 8; index += 1) fireEvent.keyDown(deadman, { key: "ArrowDown" });
    expect(slider("gripper").closest("label")).toHaveAttribute("data-jog-selected", "true");
    for (let index = 0; index < 20; index += 1) {
      fireEvent.keyDown(deadman, { key: "ArrowRight", shiftKey: true, repeat: index > 0 });
    }
    await advance(150);
    expect(lastCommand().gripper).toBe(100);
    for (let index = 0; index < 30; index += 1) {
      fireEvent.keyDown(deadman, { key: "ArrowLeft", shiftKey: true, repeat: index > 0 });
    }
    await advance(150);
    expect(lastCommand().gripper).toBe(0);
  });

  test("after release the arrows are free again, and focusing a slider marks its joint", async () => {
    await renderPanel();
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.focus(slider("elbow_flex"));
    expect(slider("elbow_flex").closest("label")).toHaveAttribute("data-jog-selected", "true");
    fireEvent.keyDown(deadman, { key: " " });
    await advance(10);
    fireEvent.keyDown(deadman, { key: "ArrowRight" });
    await advance(150);
    expect(lastCommand().elbow_flex).toBe(4);
    fireEvent.keyUp(deadman, { key: " " });
    await advance(10);
    expect(fireEvent.keyDown(deadman, { key: "ArrowRight" })).toBe(true);
    await advance(10);
    expect(slider("elbow_flex")).toHaveValue("4");
  });

  test("arrows aimed at a text field are typing, not jogging, even during a hold", async () => {
    render(
      <TelemetryProvider>
        <SafetyProvider>
          <input aria-label="Episode notes" />
          <ManualControlPanel />
        </SafetyProvider>
      </TelemetryProvider>,
    );
    await advance(10);
    fireEvent.click(screen.getByRole("button", { name: /Acquire manual lease/ }));
    await advance(10);
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.pointerDown(deadman, { isPrimary: true, button: 0, pointerId: 1 });
    await advance(10);
    expect(deadman).toHaveAttribute("aria-pressed", "true");
    expect(fireEvent.keyDown(screen.getByRole("textbox", { name: "Episode notes" }), { key: "ArrowRight" })).toBe(true);
    await advance(150);
    expect(lastCommand().shoulder_pan).toBe(1);
  });
});

describe("readStartPose", () => {
  test("accepts .pos or bare joint keys and lists what is missing", () => {
    const { targets, missing } = readStartPose({ shoulder_pan: 3, "gripper.pos": 10, elbow_flex: Number.NaN });
    expect(targets).toEqual({ shoulder_pan: 3, gripper: 10 });
    expect(missing).toEqual(["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]);
  });
});

test("thumbsticks stream bounded multi-joint targets only during the existing deadman", async () => {
  const lastCommand = () => {
    const calls = backend.callsTo("/api/control/command");
    return (calls[calls.length - 1].body as { targets: Record<string, number> }).targets;
  };
  await renderPanel();
  fireEvent.click(screen.getByRole("button", { name: "Thumbsticks · WASD + arrows" }));
  fireEvent.keyDown(window, { key: "w" });
  await advance(150);
  expect(backend.callsTo("/api/control/command")).toHaveLength(0);
  const deadman = screen.getByRole("button", { name: "Hold to move" });
  fireEvent.keyDown(deadman, { key: " " }); await advance(20);
  fireEvent.keyDown(window, { key: "w" }); fireEvent.keyDown(window, { key: "ArrowRight" });
  await advance(200);
  const moving = lastCommand();
  expect(moving.shoulder_lift).toBeGreaterThan(2);
  expect(moving.shoulder_lift).toBeLessThanOrEqual(2.81);
  expect(moving.elbow_flex).toBeGreaterThan(3);
  expect(moving.shoulder_pan).toBe(1);
  fireEvent.keyUp(window, { key: "w" }); fireEvent.keyUp(window, { key: "ArrowRight" });
  await advance(200); expect(lastCommand()).toEqual(moving);
  fireEvent.keyUp(deadman, { key: " " }); await advance(20);
  const count = backend.callsTo("/api/control/command").length;
  fireEvent.keyDown(window, { key: "w" }); await advance(200);
  expect(backend.callsTo("/api/control/command")).toHaveLength(count);
  expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
});

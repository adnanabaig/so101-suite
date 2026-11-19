import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import { TelemetryProvider, TRANSPORT_STALE_MS } from "../hooks/useTelemetry";
import {
  FakeWebSocket,
  advance,
  deferred,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  telemetryFrame,
  type FakeBackend,
} from "../test/harness";
import { NativeRecordingWorkspace as TeleopRecord } from "./TeleopRecord";

// Manual jog has its own tests; here it would only add lease traffic.
vi.mock("../components/ManualControlPanel", () => ({
  default: () => <div data-testid="manual-control" />,
}));
vi.mock("../components/McapWorkflow", () => ({
  default: () => <div data-testid="mcap-workflow" />,
}));

const FPS_ERROR = "FPS must be a whole number from 1 to 240.";

let backend: FakeBackend;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  for (const path of [
    "/api/teleop/start",
    "/api/teleop/stop",
    "/api/recording/start",
    "/api/recording/episode/start",
    "/api/recording/stop",
  ]) {
    backend.route("POST", path, () => jsonResponse({ ok: true, detail: "" }));
  }
  backend.route("POST", "/api/recording/episode/save", () => jsonResponse({ episode: 3, frames: 120 }));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderPage(frame: Record<string, unknown> | null = telemetryFrame(false)) {
  render(
    <TelemetryProvider>
      <SafetyProvider>
        <TeleopRecord />
      </SafetyProvider>
    </TelemetryProvider>,
  );
  await advance(10);
  if (frame) act(() => FakeWebSocket.broadcast(frame));
  await advance(10);
}

async function broadcast(frame: Record<string, unknown>) {
  act(() => FakeWebSocket.broadcast(frame));
  await advance(10);
}

function button(name: string): HTMLElement {
  return screen.getByRole("button", { name });
}

/** The reasons listed under a Start button, in order. */
function blockers(title: string): string[] {
  const lead = screen.queryByText(`${title}:`);
  if (!lead) return [];
  const box = lead.parentElement as HTMLElement;
  const first = box.childNodes[1]?.textContent ?? "";
  return [first, ...Array.from(box.querySelectorAll("li"), (item) => item.textContent ?? "")];
}

/** The value shown on a StatusPill, found by its label. */
function badge(label: string): string {
  const labelEl = screen
    .getAllByText(label)
    .find((element) => element.hasAttribute("data-pill-label")) as HTMLElement;
  return labelEl.nextElementSibling?.textContent ?? "";
}

describe("F21: Start teleop lists what blocks it", () => {
  test("an idle, supervised robot starts with the typed FPS", async () => {
    await renderPage();
    expect(blockers("Start teleop is blocked")).toEqual([]);
    expect(badge("mode")).toBe("idle");
    expect(badge("teleop")).toBe("stopped");
    fireEvent.change(screen.getByLabelText("FPS"), { target: { value: "15" } });
    fireEvent.click(button("Start teleop"));
    await advance(10);
    expect(backend.callsTo("/api/teleop/start").map((call) => call.body)).toEqual([
      { fps: 15, operator_id: expect.any(String) },
    ]);
  });

  test("a latched E-stop is the first reason", async () => {
    backend.estopped = true;
    await renderPage(telemetryFrame(true));
    expect(blockers("Start teleop is blocked")[0]).toBe(
      "E-stop is latched. Inspect the workspace, then reset it from the banner.",
    );
    expect(button("Start teleop")).toBeDisabled();
    expect(blockers("Start recording is blocked")[0]).toMatch(/^E-stop is latched/);
  });

  test("a lease, an armed control plane and a busy mode are all listed", async () => {
    backend.route("GET", "/api/control/state", () =>
      jsonResponse({
        armed: true,
        safety_state: "armed",
        lease: { source: "agent", expires_at: "", ttl_seconds: 5, server_time: "" },
        watchdog_ms: 500,
        last_command_at: null,
      }),
    );
    await renderPage(telemetryFrame(false, { mode: "rollout" }));
    expect(blockers("Start teleop is blocked")).toEqual([
      "A policy rollout is already running. Stop it first.",
      "A control lease (agent) owns the arm. Release it first.",
      "The control plane is armed.",
      'The safety state is "armed"; it must be disarmed.',
    ]);
    expect(button("Start teleop")).toBeDisabled();
    expect(button("Start recording")).toBeDisabled();
  });

  test("another session's mode blocks the other loop, and running teleop hides its list", async () => {
    await renderPage(telemetryFrame(false, { mode: "recording" }));
    expect(blockers("Start teleop is blocked")).toEqual([
      "The robot is busy in recording mode. End that session first.",
    ]);

    await broadcast(telemetryFrame(false, { mode: "teleop" }));
    expect(badge("teleop")).toBe("running");
    expect(button("Start teleop")).toBeDisabled();
    expect(blockers("Start teleop is blocked")).toEqual([]);
    expect(blockers("Start recording is blocked")).toContain(
      "The robot is busy in teleop mode. End that session first.",
    );
  });

  test("no telemetry link is a reason even when HTTP status says idle", async () => {
    await renderPage(null);
    expect(blockers("Start teleop is blocked")).toEqual(["No live telemetry link to the robot server."]);
    expect(badge("mode")).toBe("idle");
  });
});

describe("F21: FPS is validated before it is sent", () => {
  test.each(["", "0", "241", "2.5", "-5"])("FPS %j is refused inline and never sent", async (value) => {
    await renderPage();
    const input = screen.getByLabelText("FPS");
    fireEvent.change(input, { target: { value } });
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(input).toHaveAccessibleDescription(FPS_ERROR);
    expect(button("Start teleop")).toBeDisabled();
    expect(blockers("Start teleop is blocked")).toEqual(["The FPS is not valid."]);
    fireEvent.click(button("Start teleop"));
    await advance(10);
    expect(backend.callsTo("/api/teleop/start")).toHaveLength(0);
  });

  test("the bounds themselves are accepted", async () => {
    await renderPage();
    const input = screen.getByLabelText("FPS");
    fireEvent.change(input, { target: { value: "1" } });
    expect(input).not.toHaveAttribute("aria-invalid");
    fireEvent.change(input, { target: { value: "240" } });
    fireEvent.click(button("Start teleop"));
    await advance(10);
    expect(backend.callsTo("/api/teleop/start")[0]?.body).toMatchObject({ fps: 240 });
  });

  test("the recording FPS follows the same rule", async () => {
    await renderPage();
    fireEvent.change(screen.getByLabelText("Dataset repo_id"), { target: { value: "user/demo" } });
    fireEvent.change(screen.getByLabelText("Recording FPS"), { target: { value: "" } });
    expect(screen.getByLabelText("Recording FPS")).toHaveAccessibleDescription(FPS_ERROR);
    expect(blockers("Start recording is blocked")).toEqual(["The recording FPS is not valid."]);
    expect(button("Start recording")).toBeDisabled();
  });
});

describe("F21: Start recording checks the dataset repo_id", () => {
  test("blank asks for one without an inline error", async () => {
    await renderPage();
    expect(blockers("Start recording is blocked")).toEqual(["Enter a dataset repo_id."]);
    expect(screen.getByLabelText("Dataset repo_id")).not.toHaveAttribute("aria-invalid");
    expect(button("Start recording")).toBeDisabled();
  });

  test("a name the server would refuse is explained inline", async () => {
    await renderPage();
    const input = screen.getByLabelText("Dataset repo_id");
    for (const value of ["bad repo", "user//demo", "/demo", "user/.demo", "user/demo/"]) {
      fireEvent.change(input, { target: { value } });
      expect(input).toHaveAttribute("aria-invalid", "true");
      expect(input).toHaveAccessibleDescription(/^Use letters, digits, dots, dashes and underscores/);
      expect(blockers("Start recording is blocked")).toEqual(["The dataset repo_id is not valid."]);
    }
    fireEvent.change(input, { target: { value: "a".repeat(201) } });
    expect(input).toHaveAccessibleDescription("The repo_id must be at most 200 characters (it has 201).");
    expect(button("Start recording")).toBeDisabled();
  });

  test("sends the trimmed fields, and leaves a blank task to the server", async () => {
    await renderPage();
    fireEvent.change(screen.getByLabelText("Dataset repo_id"), { target: { value: "  user/so101_pick.v2  " } });
    fireEvent.change(screen.getByLabelText("Task"), { target: { value: "  Pick up the cube " } });
    fireEvent.change(screen.getByLabelText("Recording FPS"), { target: { value: "15" } });
    expect(blockers("Start recording is blocked")).toEqual([]);
    fireEvent.click(button("Start recording"));
    await advance(10);

    fireEvent.change(screen.getByLabelText("Task"), { target: { value: "   " } });
    fireEvent.click(button("Start recording"));
    await advance(10);
    const bodies = backend.callsTo("/api/recording/start").map((call) => call.body);
    expect(bodies).toEqual([
      { repo_id: "user/so101_pick.v2", task: "Pick up the cube", fps: 15, operator_id: expect.any(String) },
      { repo_id: "user/so101_pick.v2", fps: 15, operator_id: expect.any(String) },
    ]);
  });

  test("a refusal is shown verbatim and the safety state is re-read", async () => {
    backend.route("POST", "/api/recording/start", () =>
      jsonResponse({ detail: "leader arm is not connected" }, 409),
    );
    await renderPage();
    fireEvent.change(screen.getByLabelText("Dataset repo_id"), { target: { value: "user/demo" } });
    const statusPolls = backend.callsTo("/api/robot/status", "GET").length;
    fireEvent.click(button("Start recording"));
    await advance(10);
    expect(screen.getByRole("alert")).toHaveTextContent("leader arm is not connected");
    expect(backend.callsTo("/api/robot/status", "GET").length).toBeGreaterThan(statusPolls);
  });
});

describe("F21: episode state is only shown while it is fresh", () => {
  test("a live recording session drives the episode buttons", async () => {
    backend.mode = "recording";
    await renderPage(telemetryFrame(false, { mode: "recording" }));
    expect(badge("recording")).toBe("ready");
    expect(button("Start episode")).toBeEnabled();
    expect(button("Save episode")).toBeDisabled();
    fireEvent.click(button("Start episode"));
    await advance(10);
    expect(backend.callsTo("/api/recording/episode/start")).toHaveLength(1);

    await broadcast(telemetryFrame(false, { mode: "recording", recording: true, episode_frames: 42 }));
    expect(badge("recording")).toBe("episode active");
    expect(badge("frames")).toBe("42");
    expect(button("Start episode")).toBeDisabled();
    expect(screen.getByText("Save the episode to keep it. Stop recording discards an unsaved episode.")).toBeInTheDocument();

    await broadcast(telemetryFrame(false, { mode: "recording", recording_saving: true, episode_frames: 42 }));
    expect(badge("recording")).toBe("finalizing");
    expect(button("Start episode")).toBeDisabled();
    expect(button("Save episode")).toBeDisabled();
    expect(screen.getByText(/The previous episode is still finalizing/)).toBeInTheDocument();
  });

  test("a dropped link shows the episode as unknown but still lets the operator save", async () => {
    backend.mode = "recording";
    await renderPage(telemetryFrame(false, { mode: "recording", recording: true, episode_frames: 42 }));
    // The link clock ticks once a second.
    await advance(TRANSPORT_STALE_MS + 1100);

    // Fresh HTTP status still knows the mode; only telemetry carries the episode.
    expect(badge("mode")).toBe("recording");
    expect(badge("recording")).toBe("episode unknown");
    expect(badge("frames")).toBe("—");
    expect(
      screen.getByText(/^No live telemetry link, so the episode state is unknown\. Last known: episode active, 42 frames\./),
    ).toBeInTheDocument();
    expect(button("Start episode")).toBeDisabled();
    expect(button("Stop recording")).toBeEnabled();

    fireEvent.click(button("Save episode"));
    await advance(10);
    expect(backend.callsTo("/api/recording/episode/save")).toHaveLength(1);
    expect(screen.getByText("Saved episode #3 with 120 frames.")).toHaveAttribute("role", "status");
  });

  test("with the link down and status failing, the mode is unknown, not the last one seen", async () => {
    await renderPage(telemetryFrame(false, { mode: "teleop" }));
    expect(badge("teleop")).toBe("running");
    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: "down" }, 503));
    await advance(TRANSPORT_STALE_MS + 3500);

    expect(badge("mode")).toBe("unknown");
    expect(badge("teleop")).toBe("unknown");
    expect(badge("recording")).toBe("unknown");
    expect(button("Stop")).toBeEnabled();
    expect(button("Stop recording")).toBeEnabled();
    expect(button("Start teleop")).toBeDisabled();
  });
});

describe("Stops are never blocked", () => {
  test("Stop works while a start is still in flight, and during an E-stop", async () => {
    const pending = deferred<Response>();
    backend.route("POST", "/api/teleop/start", () => pending.promise);
    await renderPage();
    fireEvent.click(button("Start teleop"));
    await advance(10);
    expect(button("Start teleop")).toBeDisabled();
    expect(button("Stop")).toBeEnabled();
    fireEvent.click(button("Stop"));
    await advance(10);
    expect(backend.callsTo("/api/teleop/stop")).toHaveLength(1);
    pending.resolve(jsonResponse({ ok: true, detail: "" }));
    await advance(10);

    backend.estopped = true;
    await broadcast(telemetryFrame(true, { mode: "recording" }));
    expect(button("Stop")).toBeEnabled();
    expect(button("Stop recording")).toBeEnabled();
    fireEvent.click(button("Stop recording"));
    await advance(10);
    expect(backend.callsTo("/api/recording/stop")).toHaveLength(1);
  });
});

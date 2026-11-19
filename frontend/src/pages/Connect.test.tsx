import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { TrackingPreviewResponse } from "../api";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import { TelemetryProvider } from "../hooks/useTelemetry";
import type { HandTrackingSample } from "../lib/handTracking";
import { cameraHand } from "../test/handFixtures";
import {
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  type FakeBackend,
} from "../test/harness";
import Connect from "./Connect";

// The MediaPipe loop has its own tests (handTracking.test.ts); here the page
// receives samples directly so each one is under the test's control.
const tracker = vi.hoisted(() => ({
  onSample: null as ((sample: HandTrackingSample | null) => void) | null,
}));

vi.mock("../lib/handTracking", () => ({
  startBrowserHandTracking: (_video: HTMLVideoElement, onSample: (sample: HandTrackingSample | null) => void) => {
    tracker.onSample = onSample;
    return Promise.resolve({ stop: () => undefined });
  },
}));

vi.mock("../components/QuestWebXRPanel", () => ({
  default: () => <div data-testid="quest-panel" />,
}));

const ACCEPTED: TrackingPreviewResponse = {
  accepted: true,
  reason: "ok",
  targets: {},
  desired_targets: {},
  commanded_targets: {},
  confidence: 1,
};

let backend: FakeBackend;

function handSample(overrides: Partial<HandTrackingSample> = {}): HandTrackingSample {
  return {
    landmarks: [],
    handedness: "right",
    confidence: 1,
    quality: { presence: 1, stability: 1, settle: 1 },
    x: 0.2,
    y: 0.1,
    rollRad: 0,
    pinch: 0.4,
    capturedAt: Date.now(),
    ...overrides,
  };
}

function feed(sample: HandTrackingSample | null) {
  act(() => tracker.onSample?.(sample));
}

type Body = Record<string, unknown> & { wrist?: { roll: number; z: number | null } };

const executeSteps = () =>
  backend
    .callsTo("/api/tracking/preview")
    .map((call) => call.body as Body)
    .filter((body) => body.execute === true);

const deadman = () => screen.getByRole("button", { name: /Hold for tracked motion/ });

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  tracker.onSample = null;
  backend.route("GET", "/api/capabilities", () => jsonResponse({ system_version: "test", capabilities: [] }));
  backend.route("POST", "/api/control/lease", () =>
    jsonResponse({ id: "lease-1", source: "browser-mediapipe", ttl_seconds: 10, server_time: new Date().toISOString() }),
  );
  backend.route("POST", "/api/control/lease/lease-1/heartbeat", () =>
    jsonResponse({ id: "lease-1", source: "browser-mediapipe", ttl_seconds: 10, server_time: new Date().toISOString() }),
  );
  backend.route("DELETE", "/api/control/lease/lease-1", () => jsonResponse({ ok: true, detail: "" }));
  backend.route("POST", "/api/control/disarm", () =>
    jsonResponse({ armed: false, safety_state: "disarmed", lease: null, watchdog_ms: 500, last_command_at: null }),
  );
  backend.route("POST", "/api/tracking/arm-execute", () => jsonResponse(ACCEPTED));
  backend.route("POST", "/api/tracking/preview", () => jsonResponse(ACCEPTED));
  vi.stubGlobal("isSecureContext", true);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: () => Promise.resolve({ getTracks: () => [{ stop: () => undefined }] }) },
  });
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  Reflect.deleteProperty(navigator, "mediaDevices");
});

/** Camera on, tracker running, hand lease held. */
async function renderTracking() {
  render(
    <MemoryRouter>
      <TelemetryProvider>
        <SafetyProvider>
          <Connect />
        </SafetyProvider>
      </TelemetryProvider>
    </MemoryRouter>,
  );
  await advance(10);
  fireEvent.click(screen.getByRole("button", { name: /Enable camera/ }));
  await advance(10);
  fireEvent.click(screen.getByRole("button", { name: /Track hand/ }));
  await advance(10);
  fireEvent.click(screen.getByRole("button", { name: /Acquire hand-teleop lease/ }));
  await advance(10);
  expect(tracker.onSample).not.toBeNull();
}

async function engage() {
  fireEvent.pointerDown(deadman(), { isPrimary: true, button: 0, pointerId: 1 });
  await advance(10);
  expect(screen.getByText("Hand deadman held · bounded motion")).toBeInTheDocument();
}

/** What a screen reader says for an element: its text without the aria-hidden parts. */
function spokenText(element: HTMLElement): string {
  const copy = element.cloneNode(true) as HTMLElement;
  copy.querySelectorAll('[aria-hidden="true"]').forEach((hidden) => hidden.remove());
  return copy.textContent ?? "";
}

describe("Connect accessibility", () => {
  test("the Quest steps are an ordered list that says each step's state in words, not only with a tick", async () => {
    render(
      <MemoryRouter>
        <TelemetryProvider>
          <SafetyProvider>
            <Connect />
          </SafetyProvider>
        </TelemetryProvider>
      </MemoryRouter>,
    );
    await advance(10);
    const list = screen.getByRole("list", { name: "Quest connection steps" });
    expect(list.tagName).toBe("OL");
    const steps = within(list).getAllByRole("listitem");
    expect(steps.map(spokenText)).toEqual([
      "Open this workstation over trusted HTTPS: done",
      "Place Quest and robot host on a reachable network: not checked",
      "Enter immersive-vr in Quest Browser: not checked",
      "Verify backend retargeting capability: not done",
    ]);
    // The tick and the step number are decoration once the state is in words.
    expect(steps[0].querySelector("svg")?.closest('[aria-hidden="true"]')).not.toBeNull();
  });
});

describe("Connect camera teleop (B6/B10/B11)", () => {
  test("the first command of a hold sends roll 0 and later roll unwraps across the branch cut", async () => {
    await renderTracking();
    feed(handSample({ rollRad: 2.8 })); // acquired with the hand turned almost upside down
    feed(handSample({ rollRad: 3.0 }));
    await advance(10);
    expect(screen.getByText(/roll 0\.06/)).toBeInTheDocument(); // previews measure from acquisition
    await engage();

    const [armBody] = backend.callsTo("/api/tracking/arm-execute").map((call) => call.body as Body);
    expect(armBody.wrist).toEqual({ x: 0.2, y: 0.1, z: null, roll: 0 });
    expect(armBody.sent_at).toEqual(expect.any(String));

    // A further 0.18 rad turn crosses atan2's +-pi cut: raw -3.1 rad.
    feed(handSample({ rollRad: -3.1 }));
    await advance(10);
    const [step] = executeSteps();
    expect(step.wrist?.roll).toBeCloseTo((-3.1 - 3.0 + 2 * Math.PI) / Math.PI, 6); // about +0.058, not -1
    expect(step.wrist?.z).toBeNull();
  });

  test("a below-gate quality sample ends the hold, disables the deadman and says why", async () => {
    await renderTracking();
    feed(handSample());
    await advance(10);
    await engage();

    feed(handSample({ confidence: 0.3, quality: { presence: 1, stability: 0.3, settle: 1 } }));
    await advance(10);
    expect(screen.queryByText("Hand deadman held · bounded motion")).toBeNull();
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
    expect(executeSteps()).toHaveLength(0);
    expect(deadman()).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Tracking quality 30% is below the 72% needed to move. Tracking jumped.",
    );
  });

  test("losing the hand ends the hold, and the reacquired hand is the new roll neutral", async () => {
    await renderTracking();
    feed(handSample({ rollRad: 0.4 }));
    await advance(10);
    await engage();

    feed(null);
    await advance(10);
    expect(screen.queryByText("Hand deadman held · bounded motion")).toBeNull();
    expect(backend.callsTo("/api/control/disarm")).toHaveLength(1);
    expect(screen.getByText("Tracking lost · target cleared")).toBeInTheDocument();

    feed(handSample({ rollRad: 2.5 }));
    await advance(10);
    expect(screen.getByText(/roll 0\.00/)).toBeInTheDocument();
    feed(handSample({ rollRad: 2.5 + Math.PI / 4 }));
    await advance(10);
    expect(screen.getByText(/roll 0\.25/)).toBeInTheDocument();
  });

  test("the camera view is mirrored so the operator's right is the screen's right", async () => {
    await renderTracking();
    expect(screen.getByTestId("camera-video")).toHaveClass("-scale-x-100");
    // The operator's right hand at x = +0.4 is at image x 0.3 (unmirrored).
    const landmarks = cameraHand({ at: { x: 0.4, y: 0 } });
    feed(handSample({ landmarks, x: 0.4, y: 0 }));
    await advance(10);
    const wrist = screen.getAllByTestId("hand-landmark")[0];
    expect(parseFloat(wrist.style.left)).toBeCloseTo((1 - landmarks[0].x) * 100, 6);
    expect(parseFloat(wrist.style.left)).toBeGreaterThan(50);
  });
});

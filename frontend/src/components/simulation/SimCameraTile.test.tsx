import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  advance,
  fakeShellTimers,
  installFakeBackend,
  jsonResponse,
  type FakeBackend,
} from "../../test/harness";
import SimCameraTile, { FRAME_INTERVAL_MS, FRAME_RETRY_BASE_MS, FRAME_RETRY_MAX_MS, frameRetryDelay } from "./SimCameraTile";

let backend: FakeBackend;
let step = 42;
let created = 0;
const revoked: string[] = [];
const originalCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");

function frameResponse(camera: string): Response {
  return new Response(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), {
    status: 200,
    headers: {
      "content-type": "image/jpeg",
      "x-so101-sim-camera": camera,
      "x-so101-sim-step": String(step),
      "x-so101-sim-generation": "3",
      "x-so101-sim-time-s": "0.700000",
    },
  });
}

function renderCalls() {
  return backend.callsTo("/api/simulation/render", "GET");
}

function setHidden(hidden: boolean) {
  if (hidden) {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
  } else {
    Reflect.deleteProperty(document, "visibilityState");
  }
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
}

beforeEach(() => {
  fakeShellTimers();
  backend = installFakeBackend();
  step = 42;
  created = 0;
  revoked.length = 0;
  backend.route("GET", "/api/simulation/render", (call) =>
    frameResponse(new URL(call.url, "http://robot.local").searchParams.get("camera") ?? "front"),
  );
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    writable: true,
    value: () => `blob:frame-${++created}`,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    writable: true,
    value: (url: string) => revoked.push(url),
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  Reflect.deleteProperty(document, "visibilityState");
  for (const [name, descriptor] of [
    ["createObjectURL", originalCreate],
    ["revokeObjectURL", originalRevoke],
  ] as const) {
    if (descriptor) Object.defineProperty(URL, name, descriptor);
    else Reflect.deleteProperty(URL, name);
  }
});

describe("SimCameraTile", () => {
  test("asks for one frame per scene change, never more than one per interval", async () => {
    const { rerender } = render(<SimCameraTile cameras={["front"]} sceneVersion="3:42" />);
    await advance(10);
    expect(renderCalls()).toHaveLength(1);
    expect(renderCalls()[0]?.url).toBe(
      "/api/simulation/render?camera=front&width=320&height=240&format=jpeg&quality=80",
    );
    expect(screen.getByRole("img")).toHaveAccessibleName("Simulated front camera at step 42");

    // A paused scene does not change, so polling costs nothing.
    await advance(3000);
    expect(renderCalls()).toHaveLength(1);

    // Three scene changes inside one interval cost one frame.
    step = 45;
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="3:43" />);
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="3:44" />);
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="3:45" />);
    await advance(FRAME_INTERVAL_MS);
    expect(renderCalls()).toHaveLength(2);
    expect(screen.getByRole("img")).toHaveAccessibleName("Simulated front camera at step 45");
    await advance(FRAME_INTERVAL_MS * 4);
    expect(renderCalls()).toHaveLength(2);
  });

  test("frees each replaced frame, and the last one on unmount", async () => {
    const { rerender, unmount } = render(<SimCameraTile cameras={["front"]} sceneVersion="1:1" />);
    await advance(10);
    expect(screen.getByRole("img")).toHaveAttribute("src", "blob:frame-1");
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="1:2" />);
    await advance(FRAME_INTERVAL_MS);
    expect(screen.getByRole("img")).toHaveAttribute("src", "blob:frame-2");
    expect(revoked).toEqual(["blob:frame-1"]);
    unmount();
    expect(revoked).toEqual(["blob:frame-1", "blob:frame-2"]);
  });

  test("switching camera asks for that camera", async () => {
    render(<SimCameraTile cameras={["front", "wrist_cam"]} sceneVersion="1:1" />);
    await advance(10);
    fireEvent.change(screen.getByLabelText("Camera"), { target: { value: "wrist_cam" } });
    await advance(10);
    expect(renderCalls().map((call) => new URL(call.url, "http://robot.local").searchParams.get("camera"))).toEqual([
      "front",
      "wrist_cam",
    ]);
    expect(screen.getByRole("img")).toHaveAccessibleName("Simulated wrist camera at step 42");
  });

  test("a hidden tab asks for nothing, and gets a fresh frame when shown", async () => {
    const { rerender } = render(<SimCameraTile cameras={["front"]} sceneVersion="1:1" />);
    await advance(10);
    setHidden(true);
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="1:2" />);
    await advance(5000);
    expect(renderCalls()).toHaveLength(1);

    setHidden(false);
    await advance(10);
    expect(renderCalls()).toHaveLength(2);
  });

  test("a refused frame shows the server's reason, keeps the last frame, and retries later", async () => {
    const { rerender } = render(<SimCameraTile cameras={["front"]} sceneVersion="1:1" />);
    await advance(10);
    backend.route("GET", "/api/simulation/render", () =>
      jsonResponse({ detail: "rendering is unavailable here: no OpenGL context" }, 503),
    );
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="1:2" />);
    await advance(FRAME_INTERVAL_MS);
    expect(renderCalls()).toHaveLength(2);
    expect(screen.getByRole("status")).toHaveTextContent(
      "Showing the last frame. rendering is unavailable here: no OpenGL context",
    );
    expect(screen.getByRole("img")).toHaveAttribute("src", "blob:frame-1");

    expect(screen.getByRole("status")).toHaveTextContent("Next try in 1 s.");
    await advance(FRAME_RETRY_BASE_MS - 100);
    expect(renderCalls()).toHaveLength(2);
    backend.route("GET", "/api/simulation/render", () => frameResponse("front"));
    await advance(200);
    expect(renderCalls()).toHaveLength(3);
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByRole("img")).toHaveAttribute("src", "blob:frame-2");
  });

  test("repeated 503s back off instead of spinning, and a frame resets the backoff", async () => {
    let answer: "busy" | "frame" = "busy";
    backend.route("GET", "/api/simulation/render", () =>
      answer === "busy"
        ? jsonResponse({ detail: "rendering is unavailable here: the render queue is full (8 requests are already waiting)" }, 503)
        : frameResponse("front"),
    );
    const { rerender } = render(<SimCameraTile cameras={["front"]} sceneVersion="1:1" />);
    await advance(10);
    expect(screen.getByRole("status")).toHaveTextContent(
      "rendering is unavailable here: the render queue is full (8 requests are already waiting)",
    );
    // Attempts at 0, then 1, 2, 4 and 8 s later.
    const attemptsAt: number[] = [];
    for (let elapsed = 0; elapsed <= 15_000; elapsed += 100) {
      if (renderCalls().length > attemptsAt.length) attemptsAt.push(elapsed);
      await advance(100);
    }
    expect(attemptsAt).toEqual([0, 1000, 3000, 7000, 15000]);
    expect(screen.getByRole("status")).toHaveTextContent("Next try in 16 s.");

    answer = "frame";
    await advance(16_000);
    expect(screen.getByRole("img")).toBeInTheDocument();
    expect(screen.queryByRole("status")).toBeNull();
    const afterSuccess = renderCalls().length;

    // The next failure starts again from the base delay.
    answer = "busy";
    rerender(<SimCameraTile cameras={["front"]} sceneVersion="1:2" />);
    await advance(FRAME_INTERVAL_MS);
    expect(renderCalls()).toHaveLength(afterSuccess + 1);
    expect(screen.getByRole("status")).toHaveTextContent("Next try in 1 s.");
  });

  test("the retry delay doubles from the base and is capped", () => {
    expect([1, 2, 3, 4, 5, 6, 7, 20].map(frameRetryDelay)).toEqual([
      FRAME_RETRY_BASE_MS,
      2000,
      4000,
      8000,
      16000,
      FRAME_RETRY_MAX_MS,
      FRAME_RETRY_MAX_MS,
      FRAME_RETRY_MAX_MS,
    ]);
  });

  test("a server without the route says so instead of showing a broken image", async () => {
    backend.route(
      "GET",
      "/api/simulation/render",
      () => new Response("<!doctype html>", { status: 200, headers: { "content-type": "text/html" } }),
    );
    render(<SimCameraTile cameras={["front"]} sceneVersion="1:1" />);
    await advance(10);
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(/returned text\/html instead of an image/);
  });
});

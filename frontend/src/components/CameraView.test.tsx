import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { TelemetryProvider } from "../hooks/useTelemetry";
import { MAX_MJPEG_STREAMS, resetStreamBudget } from "../lib/streamBudget";
import {
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  type FakeBackend,
} from "../test/harness";
import CameraView, { SNAPSHOT_INTERVAL_MS } from "./CameraView";
import { SafetyProvider } from "./safety/SafetyProvider";

let backend: FakeBackend;

function jpeg(): Response {
  return new Response("jpeg-bytes", {
    status: 200,
    headers: { "content-type": "image/jpeg", "cache-control": "no-store" },
  });
}

function renderTiles(cams: string[], active = true) {
  return render(
    <TelemetryProvider>
      <SafetyProvider>
        {cams.map((cam) => (
          <CameraView key={cam} cam={cam} active={active} />
        ))}
      </SafetyProvider>
    </TelemetryProvider>,
  );
}

function mjpegImages(container: HTMLElement) {
  return Array.from(container.querySelectorAll('img[src^="/api/video?cam="]'));
}

function snapshotCalls() {
  return backend.callsTo("/api/video/snapshot", "GET");
}

beforeEach(() => {
  fakeShellTimers();
  resetStreamBudget();
  installFakeWebSocket();
  backend = installFakeBackend();
  backend.route("GET", "/api/video/snapshot", () => jpeg());
  vi.spyOn(URL, "createObjectURL").mockImplementation(() => "blob:snapshot");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("CameraView stream budget (F13)", () => {
  test("a RealSense depth stream is a tile like any other: labelled, encoded and counted", async () => {
    const color = "front=realsense:123456789012+depth";
    const depth = `${color}#depth`;
    const { container } = renderTiles([color, depth, "wrist"]);
    await advance(10);
    expect(within(screen.getByTestId(`camera-${color}`)).getByText("front")).toHaveAttribute("title", color);
    expect(within(screen.getByTestId(`camera-${depth}`)).getByText("front depth")).toHaveAttribute("title", depth);
    // The depth tile takes the second stream slot, so the third tile polls snapshots.
    expect(mjpegImages(container).map((img) => img.getAttribute("src"))).toEqual([
      "/api/video?cam=front%3Drealsense%3A123456789012%2Bdepth&retry=0",
      "/api/video?cam=front%3Drealsense%3A123456789012%2Bdepth%23depth&retry=0",
    ]);
    expect(screen.getByAltText("camera front depth, live")).toBeInTheDocument();
    expect(screen.getByTestId("camera-wrist")).toHaveAttribute("data-mode", "snapshot");
    expect(snapshotCalls().map((call) => call.url)).toEqual(["/api/video/snapshot?cam=wrist"]);
  });

  test("a depth tile waiting for a slot polls its own snapshot, encoded", async () => {
    renderTiles(["front", "wrist", "realsense:123456789012+depth#depth"]);
    await advance(10);
    expect(snapshotCalls().map((call) => call.url)).toEqual([
      "/api/video/snapshot?cam=realsense%3A123456789012%2Bdepth%23depth",
    ]);
    expect(screen.getByAltText("camera RealSense 123456789012 depth, snapshot")).toBeInTheDocument();
    expect(
      within(screen.getByTestId("camera-realsense:123456789012+depth#depth")).getByText("RealSense 123456789012 depth"),
    ).toHaveAttribute("title", "realsense:123456789012+depth#depth");
  });

  test(`at most ${MAX_MJPEG_STREAMS} tiles stream MJPEG; the rest show snapshots`, async () => {
    const { container } = renderTiles(["front", "wrist", "top", "side"]);
    await advance(10);
    const streams = mjpegImages(container);
    expect(streams).toHaveLength(MAX_MJPEG_STREAMS);
    expect(streams.map((img) => img.getAttribute("src"))).toEqual([
      "/api/video?cam=front&retry=0",
      "/api/video?cam=wrist&retry=0",
    ]);
    expect(screen.getByTestId("camera-top")).toHaveAttribute("data-mode", "snapshot");
    expect(screen.getByTestId("camera-side")).toHaveAttribute("data-mode", "snapshot");
    expect(snapshotCalls().map((call) => call.url)).toEqual([
      "/api/video/snapshot?cam=top",
      "/api/video/snapshot?cam=side",
    ]);
    expect(screen.getByAltText("camera top, snapshot")).toHaveAttribute("src", "blob:snapshot");
  });

  test("snapshots refresh once a second and Watch live takes a stream slot", async () => {
    const { container } = renderTiles(["front", "wrist", "top"]);
    await advance(10);
    await advance(SNAPSHOT_INTERVAL_MS);
    expect(snapshotCalls()).toHaveLength(2);

    const top = screen.getByTestId("camera-top");
    fireEvent.click(within(top).getByRole("button", { name: "Watch live" }));
    await advance(10);
    expect(top).toHaveAttribute("data-mode", "stream");
    expect(mjpegImages(container)).toHaveLength(MAX_MJPEG_STREAMS);
    // The newest holder gave up its slot.
    expect(screen.getByTestId("camera-wrist")).toHaveAttribute("data-mode", "snapshot");
  });

  test("a server without the snapshot endpoint is told once, not polled forever", async () => {
    backend.route("GET", "/api/video/snapshot", () => jsonResponse({ detail: "Not Found" }, 404));
    renderTiles(["front", "wrist", "top"]);
    await advance(10);
    expect(screen.getByTestId("camera-top")).toHaveTextContent("this server has no snapshot endpoint");
    await advance(5 * SNAPSHOT_INTERVAL_MS);
    expect(snapshotCalls()).toHaveLength(1);
  });

  test("a 503 from the snapshot endpoint shows its detail and keeps trying (C10)", async () => {
    backend.route("GET", "/api/video/snapshot", () => jsonResponse({ detail: "camera top has no frame yet" }, 503));
    renderTiles(["front", "wrist", "top"]);
    await advance(10);
    expect(screen.getByTestId("camera-top")).toHaveTextContent("camera top has no frame yet");
    await advance(2 * SNAPSHOT_INTERVAL_MS);
    expect(snapshotCalls()).toHaveLength(3);
  });

  test("a hidden tab holds no MJPEG connection", async () => {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
    try {
      const { container } = renderTiles(["front", "wrist"]);
      await advance(10);
      expect(mjpegImages(container)).toHaveLength(0);
      expect(screen.getByTestId("camera-front")).toHaveAttribute("data-mode", "hidden");
      expect(screen.getByTestId("camera-front")).toHaveTextContent("paused while this tab is hidden");
      expect(snapshotCalls()).toHaveLength(0);
    } finally {
      Reflect.deleteProperty(document, "visibilityState");
    }
  });

  test("an off-screen tile pauses and hands its slot to a waiting tile", async () => {
    const observers: { callback: IntersectionObserverCallback; element: Element | null }[] = [];
    class FakeIntersectionObserver {
      element: Element | null = null;
      constructor(public callback: IntersectionObserverCallback) {
        observers.push(this);
      }
      observe(element: Element) {
        this.element = element;
      }
      unobserve() {}
      disconnect() {}
      takeRecords() {
        return [];
      }
    }
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    renderTiles(["front", "wrist", "top"]);
    await advance(10);
    expect(screen.getByTestId("camera-top")).toHaveAttribute("data-mode", "snapshot");

    const front = screen.getByTestId("camera-front");
    const observer = observers.find((entry) => entry.element === front);
    act(() => {
      observer?.callback(
        [{ isIntersecting: false } as IntersectionObserverEntry],
        observer as unknown as IntersectionObserver,
      );
    });
    await advance(10);
    expect(front).toHaveAttribute("data-mode", "offscreen");
    expect(front.querySelector("img")).toBeNull();
    expect(screen.getByTestId("camera-top")).toHaveAttribute("data-mode", "stream");
  });

  test("a camera the server does not report live shows a placeholder", async () => {
    const { container } = renderTiles(["front"], false);
    await advance(10);
    expect(screen.getByTestId("camera-front")).toHaveAttribute("data-mode", "paused");
    expect(mjpegImages(container)).toHaveLength(0);
  });
});

describe("CameraView freshness (F11)", () => {
  test("an old frame is overlaid as stalled, never shown as live", async () => {
    backend.statusExtra = { camera_frame_age_ms: { front: 3000, wrist: 200 } };
    renderTiles(["front", "wrist"]);
    await advance(10);
    const front = screen.getByTestId("camera-front");
    expect(within(front).getByRole("status")).toHaveTextContent("Feed stalled · last frame 3s old");
    expect(front).toHaveTextContent("stalled");
    expect(within(screen.getByTestId("camera-wrist")).queryByRole("status")).toBeNull();
  });

  test("when robot status stops updating, the feed state is unknown", async () => {
    renderTiles(["front"]);
    await advance(10);
    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: "down" }, 503));
    await advance(6000);
    expect(within(screen.getByTestId("camera-front")).getByRole("status")).toHaveTextContent(
      /Feed state unknown · status \d+s old/,
    );
  });
});

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { cameraHand, mediaPipeLabel } from "../test/handFixtures";
import { SETTLE_FRAMES } from "./operatorFrame";
import { INFERENCE_INTERVAL_MS, startBrowserHandTracking, type HandTrackingSample } from "./handTracking";

const mocks = vi.hoisted(() => ({
  createFromOptions: vi.fn(),
  detectForVideo: vi.fn(),
  close: vi.fn(),
}));

vi.mock("@mediapipe/tasks-vision", () => ({
  FilesetResolver: { forVisionTasks: () => Promise.resolve({}) },
  HandLandmarker: { createFromOptions: mocks.createFromOptions },
}));

let frames: FrameRequestCallback[] = [];
let clock = 0;
const video = { readyState: 4, currentTime: 0 } as unknown as HTMLVideoElement & { currentTime: number };

/** Run the next animation frame after the clock and the video both moved on. */
function nextFrame(ms = INFERENCE_INTERVAL_MS + 1) {
  clock += ms;
  video.currentTime += ms / 1000;
  const pending = frames;
  frames = [];
  pending.forEach((callback) => callback(clock));
}

const rightHand = () => ({ landmarks: [cameraHand({ at: { x: 0.4, y: 0 } })], handedness: [mediaPipeLabel("right")] });
const noHand = () => ({ landmarks: [], handedness: [] });

beforeEach(() => {
  frames = [];
  clock = 1000;
  video.currentTime = 0;
  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
    frames.push(callback);
    return frames.length;
  });
  vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => undefined);
  vi.spyOn(performance, "now").mockImplementation(() => clock);
  vi.spyOn(Date, "now").mockImplementation(() => clock);
  mocks.detectForVideo.mockReset();
  mocks.close.mockReset();
  mocks.createFromOptions.mockReset();
  mocks.createFromOptions.mockResolvedValue({ detectForVideo: mocks.detectForVideo, close: mocks.close });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("startBrowserHandTracking", () => {
  test("emits operator-frame samples for the anatomical hand with a quality confidence", async () => {
    const samples: (HandTrackingSample | null)[] = [];
    mocks.detectForVideo.mockImplementation(rightHand);
    await startBrowserHandTracking(video, (sample) => samples.push(sample));

    nextFrame();
    expect(samples).toHaveLength(1);
    expect(samples[0]?.handedness).toBe("right"); // labelled "Left" by MediaPipe
    expect(samples[0]?.x).toBeCloseTo(0.4, 6); // the operator's right
    expect(samples[0]?.confidence).toBeCloseTo(1 / SETTLE_FRAMES, 12);

    for (let frame = 1; frame < SETTLE_FRAMES; frame += 1) nextFrame();
    expect(samples[SETTLE_FRAMES - 1]?.confidence).toBe(1);
  });

  test("losing the hand emits null and the next hand settles from scratch", async () => {
    const samples: (HandTrackingSample | null)[] = [];
    mocks.detectForVideo.mockImplementation(rightHand);
    await startBrowserHandTracking(video, (sample) => samples.push(sample));
    for (let frame = 0; frame < SETTLE_FRAMES; frame += 1) nextFrame();
    expect(samples[samples.length - 1]).not.toBeNull();

    mocks.detectForVideo.mockImplementation(noHand);
    nextFrame();
    expect(samples[samples.length - 1]).toBeNull();

    mocks.detectForVideo.mockImplementation(rightHand);
    nextFrame();
    expect(samples[samples.length - 1]?.confidence).toBeCloseTo(1 / SETTLE_FRAMES, 12);
  });

  test("inference is throttled and skips frames the video has not advanced", async () => {
    mocks.detectForVideo.mockImplementation(rightHand);
    await startBrowserHandTracking(video, () => undefined);
    nextFrame();
    expect(mocks.detectForVideo).toHaveBeenCalledTimes(1);
    nextFrame(10);
    expect(mocks.detectForVideo).toHaveBeenCalledTimes(1);
    nextFrame(INFERENCE_INTERVAL_MS);
    expect(mocks.detectForVideo).toHaveBeenCalledTimes(2);
    clock += INFERENCE_INTERVAL_MS + 1; // time passes but the video shows the same frame
    frames.splice(0).forEach((callback) => callback(clock));
    expect(mocks.detectForVideo).toHaveBeenCalledTimes(2);
  });

  test("a detection error closes the model, clears the sample and stops the loop", async () => {
    const onSample = vi.fn();
    const onError = vi.fn();
    const failure = new Error("GPU context lost");
    mocks.detectForVideo.mockImplementation(() => {
      throw failure;
    });
    await startBrowserHandTracking(video, onSample, onError);
    nextFrame();
    expect(onSample).toHaveBeenCalledWith(null);
    expect(onError).toHaveBeenCalledWith(failure);
    expect(mocks.close).toHaveBeenCalledTimes(1);
    expect(frames).toHaveLength(0);
  });

  test("falls back to the CPU delegate, and stop() closes the model once", async () => {
    mocks.createFromOptions
      .mockRejectedValueOnce(new Error("no WebGL"))
      .mockResolvedValueOnce({ detectForVideo: mocks.detectForVideo, close: mocks.close });
    const tracker = await startBrowserHandTracking(video, () => undefined);
    const delegates = mocks.createFromOptions.mock.calls.map(
      (call) => (call[1] as { baseOptions: { delegate: string } }).baseOptions.delegate,
    );
    expect(delegates).toEqual(["GPU", "CPU"]);
    tracker.stop();
    tracker.stop();
    expect(mocks.close).toHaveBeenCalledTimes(1);
  });
});

import {
  cameraSampleFromResult,
  type CameraQualityState,
  type HandTrackingSample,
} from "./operatorFrame";

export type { HandTrackingSample } from "./operatorFrame";

const TASKS_VERSION = "1.0.1";
const DEFAULT_WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`;
const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";
/** Inference is throttled to about 15 Hz. */
export const INFERENCE_INTERVAL_MS = 66;

export interface RunningHandTracker {
  stop: () => void;
}

/**
 * Run MediaPipe on the (unmirrored) camera video and emit operator-frame
 * samples (operatorFrame.ts): the anatomical hand (B10), a tracking-quality
 * confidence (B11) and a raw roll the consumer references at engage (B6).
 * Emits null whenever no complete hand is visible.
 */
export async function startBrowserHandTracking(
  video: HTMLVideoElement,
  onSample: (sample: HandTrackingSample | null) => void,
  onError: (error: unknown) => void = () => undefined,
): Promise<RunningHandTracker> {
  const { FilesetResolver, HandLandmarker } = await import(
    "@mediapipe/tasks-vision"
  );
  const wasmUrl = import.meta.env.VITE_MEDIAPIPE_WASM_URL || DEFAULT_WASM_URL;
  const modelUrl = import.meta.env.VITE_HAND_LANDMARKER_MODEL_URL || DEFAULT_MODEL_URL;
  const vision = await FilesetResolver.forVisionTasks(wasmUrl);
  const create = (delegate: "GPU" | "CPU") =>
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelUrl, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });

  let landmarker;
  try {
    landmarker = await create("GPU");
  } catch {
    landmarker = await create("CPU");
  }

  let stopped = false;
  let animationFrame = 0;
  let lastVideoTime = -1;
  let lastInferenceAt = 0;
  let quality: CameraQualityState | null = null;

  const tick = () => {
    if (stopped) return;
    const now = performance.now();
    if (
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.currentTime !== lastVideoTime &&
      now - lastInferenceAt >= INFERENCE_INTERVAL_MS
    ) {
      lastVideoTime = video.currentTime;
      lastInferenceAt = now;
      let result;
      try {
        result = landmarker.detectForVideo(video, now);
      } catch (error) {
        stopped = true;
        landmarker.close();
        onSample(null);
        onError(error);
        return;
      }
      const next = cameraSampleFromResult(result, Date.now(), quality);
      quality = next.state;
      onSample(next.sample);
    }
    animationFrame = window.requestAnimationFrame(tick);
  };
  animationFrame = window.requestAnimationFrame(tick);

  return {
    stop: () => {
      if (stopped) return;
      stopped = true;
      window.cancelAnimationFrame(animationFrame);
      landmarker.close();
    },
  };
}

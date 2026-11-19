import type { TrackingExecuteRequest, TrackingPreviewRequest, TrackingWrist } from "../api";

/** Source tag for browser camera (MediaPipe) tracking. */
export const CAMERA_TRACKING_SOURCE = "browser-mediapipe";
/** C9 command lifetime: the server drops a command older than this. */
export const TRACKING_TTL_MS = 300;
/**
 * The server's tracking gate (tracking.py MIN_TRACKING_CONFIDENCE). The client
 * applies the same value so the deadman is only offered, and motion only
 * continues, while the tracking-quality signal (B11) clears it.
 */
export const MIN_TRACKING_CONFIDENCE = 0.72;

/**
 * The parts of a camera sample a request needs. `x`/`y` are in the operator
 * frame (operatorFrame.ts, B10), `roll` is relative to the engage-time neutral
 * in -1..1 (B6), and `confidence` is the tracking-quality signal (B11).
 */
export interface CameraPose {
  x: number;
  y: number;
  roll: number;
  pinch: number;
  confidence: number;
}

/**
 * B5/C9: a monocular camera has no metric depth, so `z` is sent as null and
 * the server holds the elbow at its engage-time value. Pose values go in
 * `wrist`, never in `intent`: the server merges wrist into intent with intent
 * winning, so a stray `intent.z` would override the null.
 */
export function cameraWrist(pose: Pick<CameraPose, "x" | "y" | "roll">): TrackingWrist {
  return { x: pose.x, y: pose.y, z: null, roll: pose.roll };
}

/** Preview-only request; `sentAt` is server time (useMotionLease.timestampFor). */
export function cameraPreviewRequest(
  pose: CameraPose,
  sentAt: string,
  capturedAt: number,
): TrackingPreviewRequest {
  return {
    source: CAMERA_TRACKING_SOURCE,
    confidence: pose.confidence,
    wrist: cameraWrist(pose),
    pinch: pose.pinch,
    intent: { preview_only: true, captured_at: capturedAt },
    sent_at: sentAt,
  };
}

/** C9 execute request: lease, sequence, TTL, deadman and `sent_at` are required. */
export function cameraExecuteRequest(
  pose: CameraPose,
  command: { leaseId: string; sequence: number; sentAt: string },
): TrackingExecuteRequest {
  return {
    source: CAMERA_TRACKING_SOURCE,
    confidence: pose.confidence,
    wrist: cameraWrist(pose),
    pinch: pose.pinch,
    execute: true,
    lease_id: command.leaseId,
    sequence: command.sequence,
    sent_at: command.sentAt,
    ttl_ms: TRACKING_TTL_MS,
    deadman: true,
  };
}

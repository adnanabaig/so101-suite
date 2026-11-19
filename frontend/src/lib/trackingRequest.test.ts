import { describe, expect, test } from "vitest";
import { cameraExecuteRequest, cameraPreviewRequest, cameraWrist, TRACKING_TTL_MS } from "./trackingRequest";

const pose = { x: 0.25, y: -0.5, roll: 0.1, pinch: 0.8, confidence: 0.9 };
const SENT_AT = "2026-10-01T12:00:00.000Z";

describe("camera tracking requests (C9, B5)", () => {
  test("monocular wrist depth is null, never a guessed number", () => {
    expect(cameraWrist(pose)).toEqual({ x: 0.25, y: -0.5, z: null, roll: 0.1 });
  });

  test("an execute request carries every C9 field and keeps pose out of intent", () => {
    const request = cameraExecuteRequest(pose, { leaseId: "lease-1", sequence: 7, sentAt: SENT_AT });
    expect(request).toEqual({
      source: "browser-mediapipe",
      confidence: 0.9,
      wrist: { x: 0.25, y: -0.5, z: null, roll: 0.1 },
      pinch: 0.8,
      execute: true,
      lease_id: "lease-1",
      sequence: 7,
      sent_at: SENT_AT,
      ttl_ms: TRACKING_TTL_MS,
      deadman: true,
    });
    // The server merges wrist into intent with intent winning; no intent.z may exist.
    expect(request.intent).toBeUndefined();
    expect(JSON.parse(JSON.stringify(request)).wrist.z).toBeNull();
  });

  test("a preview request is stamped and marked preview-only", () => {
    const request = cameraPreviewRequest(pose, SENT_AT, 1234);
    expect(request.sent_at).toBe(SENT_AT);
    expect(request.execute).toBeUndefined();
    expect(request.intent).toEqual({ preview_only: true, captured_at: 1234 });
    expect(request.wrist?.z).toBeNull();
  });
});

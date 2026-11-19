import { describe, expect, test } from "vitest";
import {
  IMPLAUSIBLE_SPEED_PALMS_PER_S,
  KEY_POINT_MISSING_CAP,
  REACQUIRE_GAP_MS,
  SETTLE_FRAMES,
  anatomicalHandFromMediaPipe,
  cameraOperatorPoint,
  cameraQualityHint,
  cameraRollRad,
  cameraSampleFromResult,
  engageRoll,
  landmarkPresence,
  mirroredDisplayPoint,
  nextCameraQuality,
  preferredHandIndex,
  questOperatorWrist,
  questTrackingConfidence,
  relativeRoll,
  updateRoll,
  wrapAngle,
  type CameraQualityState,
} from "./operatorFrame";
import { MIN_TRACKING_CONFIDENCE } from "./trackingRequest";
import { cameraHand, mediaPipeLabel, type HandPose } from "../test/handFixtures";

function sampleOf(poses: HandPose[], capturedAt = 1000, previous: CameraQualityState | null = null) {
  return cameraSampleFromResult(
    {
      landmarks: poses.map((pose) => cameraHand(pose)),
      handedness: poses.map((pose) => mediaPipeLabel(pose.side ?? "right")),
    },
    capturedAt,
    previous,
  );
}

/** Feed `frames` steady frames 66 ms apart and return the last result. */
function steadyFrames(pose: HandPose, frames: number, start = 1000) {
  let state: CameraQualityState | null = null;
  let last = sampleOf([pose], start, null);
  for (let frame = 0; frame < frames; frame += 1) {
    last = sampleOf([pose], start + frame * 66, state);
    state = last.state;
  }
  return last;
}

describe("one operator frame for camera and Quest (B10)", () => {
  test("MediaPipe labels assume a mirrored input, so raw-frame labels are swapped", () => {
    expect(anatomicalHandFromMediaPipe("Left")).toBe("right");
    expect(anatomicalHandFromMediaPipe("Right")).toBe("left");
    expect(anatomicalHandFromMediaPipe(" right ")).toBe("left");
    expect(anatomicalHandFromMediaPipe(undefined)).toBe("unknown");
    expect(anatomicalHandFromMediaPipe("")).toBe("unknown");
  });

  test("moving the hand to the operator's right is +x on both paths", () => {
    // Camera: the operator's right is the left of the raw image.
    const right = sampleOf([{ at: { x: 0.4, y: 0 } }]).sample;
    const left = sampleOf([{ at: { x: -0.4, y: 0 } }]).sample;
    expect(cameraHand({ at: { x: 0.4, y: 0 } })[0].x).toBeLessThan(0.5);
    expect(right?.x).toBeCloseTo(0.4, 6);
    expect(left?.x).toBeCloseTo(-0.4, 6);
    // Quest: WebXR +x is already the user's right.
    expect(questOperatorWrist({ x: 0.1, y: 0, z: 0 }).normalized_x).toBeCloseTo(0.4, 6);
    expect(questOperatorWrist({ x: -0.1, y: 0, z: 0 }).normalized_x).toBeCloseTo(-0.4, 6);
  });

  test("up is +y and reaching forward is +z on both paths", () => {
    expect(cameraOperatorPoint({ x: 0.5, y: 0.2 }).y).toBeCloseTo(0.6, 6);
    expect(questOperatorWrist({ x: 0, y: 0.05, z: 0 }).normalized_y).toBeCloseTo(0.2, 6);
    // WebXR -z is forward; the camera sends no depth at all (B5).
    expect(questOperatorWrist({ x: 0, y: 0, z: -0.1 }).z).toBeCloseTo(0.5, 6);
    expect(questOperatorWrist({ x: 2, y: -2, z: -2 })).toEqual({ normalized_x: 1, normalized_y: -1, z: 1 });
  });

  test("with two hands in view the operator's right hand is tracked", () => {
    const result = sampleOf([
      { side: "left", at: { x: -0.4, y: 0 } },
      { side: "right", at: { x: 0.4, y: 0 } },
    ]);
    expect(result.sample?.handedness).toBe("right");
    expect(result.sample?.x).toBeCloseTo(0.4, 6);
  });

  test("a tracked hand stays tracked while it is visible, even when the right hand appears", () => {
    const first = sampleOf([{ side: "left", at: { x: -0.4, y: 0 } }]);
    expect(first.sample?.handedness).toBe("left");
    const both = sampleOf(
      [
        { side: "right", at: { x: 0.4, y: 0 } },
        { side: "left", at: { x: -0.4, y: 0 } },
      ],
      1066,
      first.state,
    );
    expect(both.sample?.handedness).toBe("left");
  });

  test("preferredHandIndex: the preferred hand, then the right hand, else none", () => {
    expect(preferredHandIndex(["left", "right"], "left")).toBe(0);
    expect(preferredHandIndex(["left", "right"])).toBe(1);
    expect(preferredHandIndex(["left", "unknown"], "right")).toBe(-1);
    expect(preferredHandIndex(["unknown", "left"], "unknown")).toBe(-1);
  });

  test("the mirrored preview draws landmarks where the operator-frame crosshair points", () => {
    const sample = sampleOf([{ at: { x: 0.3, y: -0.2 } }]).sample;
    if (!sample) throw new Error("expected a sample");
    const wrist = mirroredDisplayPoint(sample.landmarks[0]);
    expect(wrist.left).toBeCloseTo((sample.x + 1) / 2, 6);
    expect(wrist.top).toBeCloseTo((1 - sample.y) / 2, 6);
    expect(mirroredDisplayPoint({ x: 0.2, y: 0.3 })).toEqual({ left: 0.8, top: 0.3 });
  });
});

describe("engage-referenced, unwrapped roll (B6)", () => {
  test("wrapAngle maps into (-π, π]", () => {
    expect(wrapAngle(0)).toBe(0);
    expect(wrapAngle(Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(wrapAngle(-Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(wrapAngle((3 * Math.PI) / 2)).toBeCloseTo(-Math.PI / 2, 12);
    expect(wrapAngle(-5 * Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(wrapAngle(7)).toBeCloseTo(7 - 2 * Math.PI, 12);
  });

  test("a right hand reads about 0 and a left hand sits on the branch cut", () => {
    const right = cameraHand({ side: "right" });
    const left = cameraHand({ side: "left" });
    expect(cameraRollRad(right[5], right[17])).toBeCloseTo(0, 6);
    expect(Math.abs(cameraRollRad(left[5], left[17]))).toBeCloseTo(Math.PI, 6);
    const tilted = cameraHand({ side: "right", rollRad: 0.3 });
    expect(cameraRollRad(tilted[5], tilted[17])).toBeCloseTo(0.3, 6);
  });

  test("the neutral captured at engage reads 0, whatever the raw angle", () => {
    const reference = engageRoll(2.9);
    expect(relativeRoll(reference)).toBe(0);
    expect(updateRoll(reference, 2.9).relative).toBe(0);
    expect(updateRoll(reference, 2.9 + 0.2).relative).toBeCloseTo(0.2 / Math.PI, 12);
  });

  test("crossing the atan2 branch cut moves a few degrees instead of flipping", () => {
    // A left hand held still with a few degrees of jitter around ±π.
    let reference = engageRoll(Math.PI - 0.03);
    const raw = [-Math.PI + 0.02, Math.PI - 0.05, -Math.PI + 0.04, Math.PI - 0.01];
    for (const value of raw) {
      const next = updateRoll(reference, value);
      reference = next.reference;
      expect(Math.abs(next.relative)).toBeLessThan(0.1 / Math.PI);
    }
    // The old absolute mapping would have swung between about +1 and -1 here.
    expect(Math.abs(raw[0] / Math.PI - raw[1] / Math.PI)).toBeGreaterThan(1.9);
  });

  test("rotation accumulates past ±π, saturates at ±1 and comes back", () => {
    let reference = engageRoll(0);
    let relative = 0;
    for (let step = 1; step <= 10; step += 1) {
      ({ reference, relative } = updateRoll(reference, wrapAngle(step * 0.4)));
    }
    expect(reference.unwrapped).toBeCloseTo(4, 9);
    expect(relative).toBe(1);
    for (let step = 9; step >= 0; step -= 1) {
      ({ reference, relative } = updateRoll(reference, wrapAngle(step * 0.4)));
    }
    expect(relative).toBeCloseTo(0, 9);
  });

  test("a non-finite reading keeps the last value", () => {
    const reference = updateRoll(engageRoll(0), 0.5).reference;
    const next = updateRoll(reference, Number.NaN);
    expect(next.reference).toBe(reference);
    expect(next.relative).toBeCloseTo(0.5 / Math.PI, 12);
    expect(relativeRoll(engageRoll(Number.NaN))).toBe(0);
  });
});

describe("camera tracking quality (B11)", () => {
  test("presence is the fraction of landmarks inside the image", () => {
    expect(landmarkPresence(cameraHand())).toBe(1);
    // Fingertips (non-key) past the top edge.
    const cropped = cameraHand().map((point, index) =>
      [12, 16, 20].includes(index) ? { ...point, y: -0.02 } : point,
    );
    expect(landmarkPresence(cropped)).toBeCloseTo(18 / 21, 12);
    expect(landmarkPresence(cameraHand().slice(0, 20))).toBe(0);
  });

  test("an extrapolated key landmark caps presence below the gate", () => {
    const wristOut = cameraHand().map((point, index) => (index === 0 ? { ...point, y: 1.04 } : point));
    expect(landmarkPresence(wristOut)).toBe(KEY_POINT_MISSING_CAP);
    expect(KEY_POINT_MISSING_CAP).toBeLessThan(MIN_TRACKING_CONFIDENCE);
    const nanThumb = cameraHand().map((point, index) => (index === 4 ? { ...point, x: Number.NaN } : point));
    expect(landmarkPresence(nanThumb)).toBe(KEY_POINT_MISSING_CAP);
  });

  test("a steady, fully visible hand settles above the gate over a few frames", () => {
    const confidences: number[] = [];
    let state: CameraQualityState | null = null;
    for (let frame = 0; frame < SETTLE_FRAMES + 1; frame += 1) {
      const result = sampleOf([{}], 1000 + frame * 66, state);
      state = result.state;
      confidences.push(result.sample?.confidence ?? -1);
    }
    expect(confidences[0]).toBeCloseTo(1 / SETTLE_FRAMES, 12);
    expect(confidences[1]).toBeLessThan(MIN_TRACKING_CONFIDENCE);
    expect(confidences[SETTLE_FRAMES - 1]).toBe(1);
    expect(confidences[SETTLE_FRAMES]).toBe(1);
  });

  test("the handedness score never changes the confidence", () => {
    const landmarks = [cameraHand()];
    const confident = cameraSampleFromResult(
      { landmarks, handedness: [mediaPipeLabel("right", 0.99)] },
      1000,
      null,
    );
    const unsure = cameraSampleFromResult({ landmarks, handedness: [mediaPipeLabel("right", 0.51)] }, 1000, null);
    expect(unsure.sample?.confidence).toBe(confident.sample?.confidence);
  });

  test("a jump to the other hand drops confidence to zero and restarts settling", () => {
    const settled = steadyFrames({ at: { x: -0.4, y: 0 } }, SETTLE_FRAMES);
    expect(settled.sample?.confidence).toBe(1);
    // Same label, 0.8 operator units (4 palm widths) in one 66 ms frame.
    const jumped = sampleOf([{ at: { x: 0.4, y: 0 } }], 1000 + SETTLE_FRAMES * 66, settled.state);
    expect(jumped.sample?.quality.stability).toBe(0);
    expect(jumped.sample?.confidence).toBe(0);
    const after = sampleOf([{ at: { x: 0.4, y: 0 } }], 1000 + (SETTLE_FRAMES + 1) * 66, jumped.state);
    expect(after.sample?.quality.stability).toBe(1);
    expect(after.sample?.confidence).toBeCloseTo(2 / SETTLE_FRAMES, 12);
  });

  test("ordinary hand motion stays steady; very fast motion is penalised", () => {
    const base: CameraQualityState = {
      hand: "right",
      wrist: { x: 0.5, y: 0.5 },
      palmWidth: 0.1,
      capturedAt: 1000,
      streak: SETTLE_FRAMES,
    };
    const moved = (palmsPerSecond: number) =>
      nextCameraQuality(base, {
        hand: "right",
        wrist: { x: 0.5 + palmsPerSecond * 0.1 * 0.066, y: 0.5 },
        palmWidth: 0.1,
        capturedAt: 1066,
        presence: 1,
      });
    expect(moved(8).confidence).toBe(1);
    expect(moved(20).quality.stability).toBeCloseTo(0.5, 6);
    expect(moved(20).confidence).toBeLessThan(MIN_TRACKING_CONFIDENCE);
    expect(moved(IMPLAUSIBLE_SPEED_PALMS_PER_S + 1).confidence).toBe(0);
  });

  test("a gap or a different hand restarts settling instead of measuring a jump", () => {
    const state: CameraQualityState = {
      hand: "right",
      wrist: { x: 0.5, y: 0.5 },
      palmWidth: 0.1,
      capturedAt: 1000,
      streak: 10,
    };
    const far = { wrist: { x: 0.9, y: 0.5 }, palmWidth: 0.1, presence: 1 };
    const gap = nextCameraQuality(state, { ...far, hand: "right", capturedAt: 1000 + REACQUIRE_GAP_MS + 1 });
    expect(gap.quality).toEqual({ presence: 1, stability: 1, settle: 1 / SETTLE_FRAMES });
    const other = nextCameraQuality(state, { ...far, hand: "left", capturedAt: 1066 });
    expect(other.state.streak).toBe(1);
    expect(other.confidence).toBeCloseTo(1 / SETTLE_FRAMES, 12);
  });

  test("no hand means no sample and a reset quality state", () => {
    expect(cameraSampleFromResult({ landmarks: [], handedness: [] }, 1000, null)).toEqual({
      sample: null,
      state: null,
    });
    const partial = cameraSampleFromResult(
      { landmarks: [cameraHand().slice(0, 10)], handedness: [mediaPipeLabel("right")] },
      1000,
      null,
    );
    expect(partial).toEqual({ sample: null, state: null });
  });

  test("the hint names the limiting factor", () => {
    expect(cameraQualityHint({ presence: 0.5, stability: 1, settle: 1 })).toMatch(/inside the camera view/);
    expect(cameraQualityHint({ presence: 1, stability: 0.2, settle: 1 })).toMatch(/Tracking jumped/);
    expect(cameraQualityHint({ presence: 1, stability: 1, settle: 0.34 })).toMatch(/settling/);
  });

  test("pinch is 1 with the fingertips together and 0 with them spread", () => {
    const hand = cameraHand();
    const together = hand.map((point, index) => (index === 4 ? { ...hand[8] } : point));
    const result = cameraSampleFromResult({ landmarks: [together], handedness: [mediaPipeLabel("right")] }, 1, null);
    expect(result.sample?.pinch).toBe(1);
    expect(sampleOf([{}]).sample?.pinch).toBe(0);
  });
});

describe("Quest tracking quality (B11)", () => {
  test("inferred joints are down-weighted and an inferred wrist cannot pass the gate", () => {
    expect(questTrackingConfidence({ tracked: 25, emulated: 0, total: 25, wrist: "tracked" })).toBe(1);
    expect(questTrackingConfidence({ tracked: 15, emulated: 10, total: 25, wrist: "tracked" })).toBeCloseTo(0.8, 12);
    expect(questTrackingConfidence({ tracked: 24, emulated: 1, total: 25, wrist: "emulated" })).toBe(
      KEY_POINT_MISSING_CAP,
    );
    expect(questTrackingConfidence({ tracked: 24, emulated: 0, total: 25, wrist: "missing" })).toBe(
      KEY_POINT_MISSING_CAP,
    );
    expect(questTrackingConfidence({ tracked: 0, emulated: 0, total: 0, wrist: "missing" })).toBe(0);
  });
});

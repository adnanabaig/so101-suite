import type { LandmarkPoint } from "../lib/operatorFrame";

/**
 * Hand-local landmark offsets in palm widths for a RIGHT hand held up with the
 * palm toward the camera, in the operator frame (+x operator's right, +y up):
 * the index knuckle is toward the body midline (-x), the pinky outward (+x).
 */
const RIGHT_HAND_OFFSETS: readonly [number, number][] = [
  [0, -1], // 0 wrist
  [-0.4, -0.8],
  [-0.7, -0.55],
  [-0.9, -0.3],
  [-1, -0.05], // 4 thumb tip
  [-0.5, 0], // 5 index MCP
  [-0.55, 0.4],
  [-0.58, 0.65],
  [-0.6, 0.9], // 8 index tip
  [-0.17, 0.02],
  [-0.17, 0.45],
  [-0.17, 0.75],
  [-0.17, 1],
  [0.17, 0],
  [0.19, 0.4],
  [0.2, 0.68],
  [0.21, 0.9],
  [0.5, 0], // 17 pinky MCP
  [0.55, 0.25],
  [0.58, 0.45],
  [0.6, 0.62],
];

export interface HandPose {
  /** Hand centre in the operator frame (-1..1). */
  at?: { x: number; y: number };
  side?: "left" | "right";
  /** Counter-clockwise in the operator (selfie) view. */
  rollRad?: number;
  /** Palm width in operator-frame units. */
  palm?: number;
}

/** Landmarks as MediaPipe reports them on a raw, unmirrored front-camera frame. */
export function cameraHand({ at = { x: 0, y: 0 }, side = "right", rollRad = 0, palm = 0.2 }: HandPose = {}): LandmarkPoint[] {
  const cos = Math.cos(rollRad);
  const sin = Math.sin(rollRad);
  return RIGHT_HAND_OFFSETS.map(([ox, oy]) => {
    const localX = side === "right" ? ox : -ox;
    const opX = at.x + palm * (localX * cos - oy * sin);
    const opY = at.y + palm * (localX * sin + oy * cos);
    return { x: 0.5 - opX / 2, y: 0.5 - opY / 2, z: 0 };
  });
}

/** MediaPipe labels raw frames as if mirrored: the right hand is called "Left". */
export function mediaPipeLabel(side: "left" | "right", score = 0.9) {
  return [{ categoryName: side === "right" ? "Left" : "Right", score }];
}

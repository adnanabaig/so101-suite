// Pure pose maths for the digital twin. Like ./types, nothing here imports
// three.js, so the light shell and the tests can use it without loading the
// WebGL chunk.
import { hasJointData, jointValue } from "./types";

/**
 * How much the drawn pose can be trusted.
 * - live: joints sampled continuously, and the sample is fresh.
 * - idle: nothing samples the joints (C6), so the pose is the last one read.
 *   Neutral rather than a warning (F8), but not live either: the arm may have
 *   been moved by hand since.
 * - stale: the link is down or a loop's samples are overdue.
 * - none: no joint data at all. The arm is drawn as a ghost in a rest pose
 *   that says nothing about where the real arm is (F14).
 */
export type TwinPresence = "live" | "idle" | "stale" | "none";

export function twinPresence(
  observation: Readonly<Record<string, number>> | undefined,
  freshness: { stale?: boolean; idle?: boolean },
): TwinPresence {
  if (!hasJointData(observation)) return "none";
  if (freshness.stale) return "stale";
  return freshness.idle ? "idle" : "live";
}

/** Material opacity for each presence: the less current, the fainter. */
export const PRESENCE_OPACITY: Readonly<Record<TwinPresence, number>> = {
  live: 1,
  idle: 0.7,
  stale: 0.5,
  none: 0.22,
};

/**
 * Drawing targets: arm joints in radians, the gripper as a 0-1 opening.
 * null means the sample does not carry that joint, and the joint stays where
 * it is drawn. Missing data never drives a joint to 0 (F14).
 */
export interface ArmTargets {
  pan: number | null;
  lift: number | null;
  elbow: number | null;
  wristFlex: number | null;
  wristRoll: number | null;
  gripper: number | null;
}

const DEG_TO_RAD = Math.PI / 180;

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * Backend and LeRobot contracts give every arm joint in calibrated degrees.
 * Guessing radians from the magnitude would make small degree values spin.
 */
export function jointAngle(degrees: number | null): number | null {
  return degrees === null ? null : clamp(degrees, -180, 180) * DEG_TO_RAD;
}

export function armTargets(observation: Readonly<Record<string, number>> | undefined): ArmTargets {
  const gripper = jointValue(observation, "gripper");
  return {
    pan: jointAngle(jointValue(observation, "shoulder_pan")),
    lift: jointAngle(jointValue(observation, "shoulder_lift")),
    elbow: jointAngle(jointValue(observation, "elbow_flex")),
    wristFlex: jointAngle(jointValue(observation, "wrist_flex")),
    wristRoll: jointAngle(jointValue(observation, "wrist_roll")),
    gripper: gripper === null ? null : clamp(Math.abs(gripper) / 100, 0, 1),
  };
}

/** Easing rate: about 95 % of the way to a new pose in a quarter second. */
const SMOOTHING_PER_SECOND = 12;
/**
 * Longest frame the easing will account for. With an on-demand render loop
 * the first frame after an idle spell reports the whole idle time; capping it
 * keeps that frame from jumping straight to the new pose.
 */
const MAX_FRAME_SECONDS = 0.05;

/** Fraction of the remaining distance to cover in one frame; 1 snaps. */
export function blendFactor(deltaSeconds: number, reducedMotion: boolean): number {
  if (reducedMotion) return 1;
  const dt = clamp(Number.isFinite(deltaSeconds) ? deltaSeconds : 0, 0, MAX_FRAME_SECONDS);
  return 1 - Math.exp(-dt * SMOOTHING_PER_SECOND);
}

/** Closer than this (radians, or opening fraction) counts as arrived. */
export const SETTLE_EPSILON = 1e-4;

export interface Step {
  value: number;
  /** True while the value has not reached its target: render another frame. */
  moving: boolean;
}

/** One easing step toward `target`. A null target holds the current value. */
export function stepToward(current: number, target: number | null, blend: number): Step {
  if (target === null) return { value: current, moving: false };
  const next = current + (target - current) * blend;
  if (Math.abs(target - next) <= SETTLE_EPSILON) return { value: target, moving: false };
  return { value: next, moving: true };
}

/** Half the finger gap for a 0-1 opening, in scene units. */
export function fingerOffset(opening: number): number {
  return 0.055 + opening * 0.09;
}

// Shared by the light twin shell (RobotTwin) and the lazily loaded WebGL
// scene (TwinScene). Nothing here may import three.js: the shell is part of
// page chunks and must stay small (F15).

export interface RobotTwinScenarioObject {
  id: string;
  shape: string;
  size_m: readonly number[];
  pose: readonly number[];
  color?: string;
  role?: string;
}

export interface RobotTwinScenario {
  scenario_id: string;
  label?: string;
  seed?: number;
  randomized?: boolean;
  surface?: {
    size_m: readonly number[];
    height_m?: number;
    color?: string;
  };
  objects?: readonly RobotTwinScenarioObject[];
}

export interface RobotTwinTrajectoryPoint {
  position_m: readonly number[];
}

export const TWIN_JOINTS = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
] as const;

export type TwinJoint = (typeof TWIN_JOINTS)[number];

/**
 * A joint's value (degrees; the gripper in 0-100 gripper units), or null when
 * the sample does not carry it. Missing data is never turned into 0 (F14).
 */
export function jointValue(
  values: Readonly<Record<string, number>> | undefined,
  joint: TwinJoint,
): number | null {
  if (!values) return null;
  const value = values[`${joint}.pos`] ?? values[joint];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** True when at least one twin joint has a finite value. `{}` has none. */
export function hasJointData(values: Readonly<Record<string, number>> | undefined): boolean {
  return TWIN_JOINTS.some((joint) => jointValue(values, joint) !== null);
}

export const TWIN_JOINT_LABELS: Readonly<Record<TwinJoint, string>> = {
  shoulder_pan: "Shoulder pan",
  shoulder_lift: "Shoulder lift",
  elbow_flex: "Elbow flex",
  wrist_flex: "Wrist flex",
  wrist_roll: "Wrist roll",
  gripper: "Gripper (0-100)",
};

/** Arm joints in degrees, the gripper in its 0-100 units, and a dash when the joint is missing. */
export function formatJointValue(joint: TwinJoint, value: number | null): string {
  if (value === null) return "—";
  return joint === "gripper" ? value.toFixed(1) : `${value.toFixed(1)}°`;
}

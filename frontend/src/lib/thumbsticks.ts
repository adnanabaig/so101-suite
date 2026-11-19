export interface StickAxes {
  leftX: number;
  leftY: number;
  rightX: number;
  rightY: number;
  roll: number;
  grip: number;
}

export const CENTERED_STICKS: StickAxes = { leftX: 0, leftY: 0, rightX: 0, rightY: 0, roll: 0, grip: 0 };
export const STICK_KEYS = new Set(["w", "a", "s", "d", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "q", "e", "z", "x"]);

export function keyboardAxes(keys: ReadonlySet<string>): StickAxes {
  const direction = (positive: string, negative: string) => Number(keys.has(positive)) - Number(keys.has(negative));
  return {
    leftX: direction("d", "a"), leftY: direction("w", "s"),
    rightX: direction("ArrowRight", "ArrowLeft"), rightY: direction("ArrowUp", "ArrowDown"),
    roll: direction("e", "q"), grip: direction("x", "z"),
  };
}

export function stepThumbsticks(
  current: Readonly<Record<string, number | undefined>>,
  axes: StickAxes,
  elapsed: number,
  limits: Readonly<Record<string, readonly [number, number]>>,
): Record<string, number | undefined> {
  const next = { ...current };
  const mapping = { shoulder_pan: axes.leftX, shoulder_lift: axes.leftY,
    elbow_flex: axes.rightX, wrist_flex: axes.rightY, wrist_roll: axes.roll, gripper: axes.grip };
  const duration = Number.isFinite(elapsed) ? Math.max(0, Math.min(0.1, elapsed)) : 0;
  for (const [joint, input] of Object.entries(mapping)) {
    const value = current[joint];
    const bounds = limits[joint];
    if (value === undefined || !Number.isFinite(value) || !bounds || !Number.isFinite(input) || Math.abs(input) < 0.12) continue;
    const delta = Math.max(-1, Math.min(1, input)) * 4 * duration;
    if ((value < bounds[0] && delta <= 0) || (value > bounds[1] && delta >= 0)) continue;
    next[joint] = Math.max(Math.min(bounds[0], value), Math.min(Math.max(bounds[1], value), value + delta));
  }
  return next;
}

import { describe, expect, test } from "vitest";
import { CENTERED_STICKS, keyboardAxes, stepThumbsticks } from "./thumbsticks";

const limits = { shoulder_pan: [-110, 110] as const, shoulder_lift: [-100, 100] as const,
  elbow_flex: [-100, 100] as const, wrist_flex: [-100, 100] as const,
  wrist_roll: [-160, 160] as const, gripper: [0, 100] as const };

describe("thumbstick target generation", () => {
  test("keyboard directions oppose, cancel and control the labeled joints", () => {
    expect(keyboardAxes(new Set(["w", "s", "ArrowRight", "e", "z"]))).toEqual({
      ...CENTERED_STICKS, rightX: 1, roll: 1, grip: -1,
    });
    expect(stepThumbsticks({ elbow_flex: 10, wrist_roll: 2, gripper: 40 },
      keyboardAxes(new Set(["ArrowRight", "e", "z"])), 0.1, limits)).toEqual({
        elbow_flex: 10.4, wrist_roll: 2.4, gripper: 39.6,
      });
  });
  test("a delayed tick cannot produce a catch-up jump", () => {
    expect(stepThumbsticks({ shoulder_pan: 1 }, { ...CENTERED_STICKS, leftX: 100 }, 20, limits)).toEqual({ shoulder_pan: 1.4 });
    expect(stepThumbsticks({ shoulder_pan: 1 }, { ...CENTERED_STICKS, leftX: 1 }, Number.NaN, limits)).toEqual({ shoulder_pan: 1 });
  });
  test("unknown joints are not guessed and seeded out-of-range poses only move inward", () => {
    expect(stepThumbsticks({}, keyboardAxes(new Set(["w"])), 0.1, limits)).toEqual({});
    expect(stepThumbsticks({ shoulder_lift: -105 }, keyboardAxes(new Set(["s"])), 0.1, limits)).toEqual({ shoulder_lift: -105 });
    expect(stepThumbsticks({ shoulder_lift: -105 }, keyboardAxes(new Set(["w"])), 0.1, limits)).toEqual({ shoulder_lift: -104.6 });
    expect(stepThumbsticks({ shoulder_lift: 99.9 }, keyboardAxes(new Set(["w"])), 0.1, limits)).toEqual({ shoulder_lift: 100 });
  });
});

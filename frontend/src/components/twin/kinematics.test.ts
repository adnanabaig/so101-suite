import { describe, expect, test } from "vitest";
import {
  armTargets,
  blendFactor,
  fingerOffset,
  jointAngle,
  PRESENCE_OPACITY,
  SETTLE_EPSILON,
  stepToward,
  twinPresence,
} from "./kinematics";
import { formatJointValue, hasJointData, jointValue } from "./types";

describe("joint values (F14)", () => {
  test("prefer the .pos key, fall back to the bare joint name", () => {
    expect(jointValue({ "elbow_flex.pos": 12, elbow_flex: 99 }, "elbow_flex")).toBe(12);
    expect(jointValue({ elbow_flex: 7 }, "elbow_flex")).toBe(7);
  });

  test("missing and non-finite values are null, never 0", () => {
    expect(jointValue(undefined, "gripper")).toBeNull();
    expect(jointValue({}, "gripper")).toBeNull();
    expect(jointValue({ "gripper.pos": Number.NaN }, "gripper")).toBeNull();
    expect(jointValue({ "gripper.pos": Number.POSITIVE_INFINITY }, "gripper")).toBeNull();
    expect(jointValue({ "gripper.pos": 0 }, "gripper")).toBe(0);
  });

  test("an empty or joint-less record has no joint data", () => {
    expect(hasJointData(undefined)).toBe(false);
    expect(hasJointData({})).toBe(false);
    expect(hasJointData({ "unrelated.pos": 3 })).toBe(false);
    expect(hasJointData({ "wrist_roll.pos": 0 })).toBe(true);
  });

  test("format: degrees for arm joints, plain units for the gripper, a dash when missing", () => {
    expect(formatJointValue("shoulder_pan", 12.345)).toBe("12.3°");
    expect(formatJointValue("gripper", 80)).toBe("80.0");
    expect(formatJointValue("elbow_flex", null)).toBe("—");
  });
});

describe("twinPresence", () => {
  test("no joint data is none, whatever the freshness", () => {
    expect(twinPresence(undefined, {})).toBe("none");
    expect(twinPresence({}, { stale: true, idle: true })).toBe("none");
  });

  test("stale outranks idle; otherwise live", () => {
    const sample = { "shoulder_pan.pos": 1 };
    expect(twinPresence(sample, { stale: true, idle: true })).toBe("stale");
    expect(twinPresence(sample, { idle: true })).toBe("idle");
    expect(twinPresence(sample, {})).toBe("live");
  });

  test("the less current the pose, the fainter it is drawn", () => {
    expect(PRESENCE_OPACITY.live).toBe(1);
    expect(PRESENCE_OPACITY.idle).toBeLessThan(PRESENCE_OPACITY.live);
    expect(PRESENCE_OPACITY.stale).toBeLessThan(PRESENCE_OPACITY.idle);
    expect(PRESENCE_OPACITY.none).toBeLessThan(PRESENCE_OPACITY.stale);
  });
});

describe("armTargets", () => {
  test("degrees become radians, clamped to ±180°", () => {
    expect(jointAngle(90)).toBeCloseTo(Math.PI / 2);
    expect(jointAngle(400)).toBeCloseTo(Math.PI);
    expect(jointAngle(-400)).toBeCloseTo(-Math.PI);
    expect(jointAngle(null)).toBeNull();
  });

  test("small degree values are not mistaken for radians", () => {
    expect(armTargets({ "shoulder_pan.pos": 2 }).pan).toBeCloseTo((2 * Math.PI) / 180);
  });

  test("the gripper is a 0-1 opening; missing joints stay null", () => {
    const targets = armTargets({ "gripper.pos": 150, "elbow_flex.pos": 30 });
    expect(targets.gripper).toBe(1);
    expect(targets.elbow).toBeCloseTo(Math.PI / 6);
    expect(targets.pan).toBeNull();
    expect(targets.lift).toBeNull();
    expect(targets.wristFlex).toBeNull();
    expect(targets.wristRoll).toBeNull();
    expect(armTargets({ "gripper.pos": -40 }).gripper).toBeCloseTo(0.4);
  });
});

describe("easing", () => {
  test("a missing target holds the drawn value instead of driving it to 0", () => {
    expect(stepToward(0.8, null, 0.5)).toEqual({ value: 0.8, moving: false });
  });

  test("eases toward the target and settles exactly on it", () => {
    let value = 0;
    let frames = 0;
    let moving = true;
    while (moving && frames < 500) {
      const step = stepToward(value, 1, blendFactor(1 / 60, false));
      value = step.value;
      moving = step.moving;
      frames += 1;
    }
    expect(value).toBe(1);
    expect(frames).toBeGreaterThan(5);
    expect(frames).toBeLessThan(120);
  });

  test("reduced motion snaps in one frame", () => {
    expect(blendFactor(1 / 60, true)).toBe(1);
    expect(stepToward(0, 1, blendFactor(1 / 60, true))).toEqual({ value: 1, moving: false });
  });

  test("a long idle gap does not make the first frame jump", () => {
    expect(blendFactor(5, false)).toBe(blendFactor(0.05, false));
    expect(blendFactor(5, false)).toBeLessThan(0.5);
    expect(blendFactor(Number.NaN, false)).toBe(0);
    expect(blendFactor(-1, false)).toBe(0);
  });

  test("values within the settle distance count as arrived", () => {
    expect(stepToward(1 - SETTLE_EPSILON / 2, 1, 0.1)).toEqual({ value: 1, moving: false });
  });

  test("finger offset grows with the opening", () => {
    expect(fingerOffset(0)).toBeCloseTo(0.055);
    expect(fingerOffset(1)).toBeCloseTo(0.145);
  });
});

import { describe, expect, test } from "vitest";
import { JOG_STEP, JOG_STEP_LARGE, jogActionForKey, jogTarget, nextJogJoint } from "./jog";

describe("jog", () => {
  test("Left and Right step the target, Shift by the large step; Up and Down pick the joint", () => {
    expect(jogActionForKey("ArrowRight", false)).toEqual({ kind: "step", delta: JOG_STEP });
    expect(jogActionForKey("ArrowLeft", true)).toEqual({ kind: "step", delta: -JOG_STEP_LARGE });
    expect(jogActionForKey("ArrowUp", false)).toEqual({ kind: "select", delta: -1 });
    expect(jogActionForKey("ArrowDown", true)).toEqual({ kind: "select", delta: 1 });
    expect(jogActionForKey("h", false)).toBeNull();
    expect(jogActionForKey(" ", false)).toBeNull();
  });

  test("the large step is no larger than the follower's 5-degree max_relative_target", () => {
    expect(JOG_STEP_LARGE).toBeLessThanOrEqual(5);
    expect(JOG_STEP).toBeLessThan(JOG_STEP_LARGE);
  });

  test("targets clamp to the joint range; selection stops at the ends", () => {
    expect(jogTarget(108, 5, -110, 110)).toBe(110);
    expect(jogTarget(-108, -5, -110, 110)).toBe(-110);
    expect(jogTarget(3, -1, -110, 110)).toBe(2);
    expect(nextJogJoint(0, -1, 6)).toBe(0);
    expect(nextJogJoint(5, 1, 6)).toBe(5);
    expect(nextJogJoint(2, 1, 6)).toBe(3);
  });
});

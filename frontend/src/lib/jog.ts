// Keyboard jog for Manual joint control. Jog keys act only while the deadman
// is held and motion is armed; they move the streamed target, never the arm
// directly, so every step still passes the server's limits (SafetyGuard joint
// limits and per-step cap, and the follower's max_relative_target).

/** One arrow press moves the selected joint's target by this much. */
export const JOG_STEP = 1;
/** With Shift held. No larger than the follower's 5-degree max_relative_target. */
export const JOG_STEP_LARGE = 5;

export type JogAction =
  | { kind: "step"; delta: number }
  | { kind: "select"; delta: -1 | 1 };

/** What an arrow key does while jogging, or null for any other key. */
export function jogActionForKey(key: string, shiftKey: boolean): JogAction | null {
  const step = shiftKey ? JOG_STEP_LARGE : JOG_STEP;
  switch (key) {
    case "ArrowRight":
      return { kind: "step", delta: step };
    case "ArrowLeft":
      return { kind: "step", delta: -step };
    case "ArrowUp":
      return { kind: "select", delta: -1 };
    case "ArrowDown":
      return { kind: "select", delta: 1 };
    default:
      return null;
  }
}

/** The jogged target, kept inside the joint's range (the UI and SafetyGuard share it). */
export function jogTarget(value: number, delta: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value + delta));
}

/** The next joint up or down the list; it stops at the ends instead of wrapping. */
export function nextJogJoint(index: number, delta: -1 | 1, count: number): number {
  return Math.min(count - 1, Math.max(0, index + delta));
}

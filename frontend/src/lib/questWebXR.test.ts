import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { deferred } from "../test/harness";
import {
  QuestWebXRSession,
  WEBXR_HAND_JOINTS,
  type QuestXRHandFrame,
  type QuestXRSessionStatus,
  type QuestXRVector3,
} from "./questWebXR";
import { MIN_TRACKING_CONFIDENCE } from "./trackingRequest";

// A scripted WebXR runtime: just enough of XRSession, XRFrame, XRWebGLLayer and
// WebGL for the controller, with gates to pause start() at each await (B21).

type GatePoint = "requestSession" | "makeXRCompatible" | "updateRenderState" | "requestReferenceSpace";

interface FakeJointSpace {
  owner: FakeHand;
  name: string;
}

class FakeHand {
  wrist: QuestXRVector3 = { x: 0.1, y: 1.2, z: -0.3 };
  readonly emulated = new Set<string>();
  readonly missing = new Set<string>();
  throwOnPose = false;
  readonly source: XRInputSource;

  constructor(readonly handedness: XRHandedness) {
    const hand = new Map<string, FakeJointSpace>(WEBXR_HAND_JOINTS.map((name) => [name, { owner: this, name }]));
    this.source = { handedness, hand, targetRayMode: "tracked-pointer", profiles: [] } as unknown as XRInputSource;
  }

  pose(name: string): XRJointPose | undefined {
    if (this.throwOnPose) throw new Error("joint pose lookup failed");
    if (this.missing.has(name)) return undefined;
    const offset = WEBXR_HAND_JOINTS.indexOf(name as (typeof WEBXR_HAND_JOINTS)[number]) * 0.004;
    return {
      transform: {
        position: { x: this.wrist.x + offset, y: this.wrist.y + offset, z: this.wrist.z, w: 1 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
      radius: 0.008,
      emulatedPosition: this.emulated.has(name),
    } as unknown as XRJointPose;
  }
}

class FakeXRSession extends EventTarget {
  inputSources: XRInputSource[] = [];
  visibilityState: XRVisibilityState = "visible";
  readonly enabledFeatures = ["hand-tracking"];
  readonly frameRate = 72;
  readonly referenceSpace = new EventTarget();
  ended = false;
  endCalls = 0;
  endError: Error | null = null;
  private frames = new Map<number, XRFrameRequestCallback>();
  private nextHandle = 1;

  constructor(private readonly env: FakeXREnvironment) {
    super();
  }

  readonly updateRenderState = vi.fn(async () => {
    await this.env.gate("updateRenderState");
  });

  readonly requestReferenceSpace = vi.fn(async () => {
    await this.env.gate("requestReferenceSpace");
    return this.referenceSpace;
  });

  requestAnimationFrame(callback: XRFrameRequestCallback): number {
    const handle = this.nextHandle++;
    this.frames.set(handle, callback);
    return handle;
  }

  cancelAnimationFrame(handle: number): void {
    this.frames.delete(handle);
  }

  async end(): Promise<void> {
    this.endCalls += 1;
    if (this.endError) throw this.endError;
    if (this.ended) throw new DOMException("The session has already ended.", "InvalidStateError");
    this.ended = true;
    this.dispatchEvent(new Event("end"));
  }

  /** Run the pending animation frame callbacks once. */
  step(time: number): void {
    const pending = Array.from(this.frames.values());
    this.frames.clear();
    const frame = {
      session: this,
      getJointPose: (space: FakeJointSpace) => space.owner.pose(space.name),
      // No viewer pose: the renderer returns before touching WebGL.
      getViewerPose: () => null,
    } as unknown as XRFrame;
    pending.forEach((callback) => callback(time, frame));
  }

  setHands(...hands: FakeHand[]): void {
    this.inputSources = hands.map((hand) => hand.source);
  }

  changeInputSources(removed: XRInputSource[], added: XRInputSource[] = []): void {
    this.inputSources = [...this.inputSources.filter((source) => !removed.includes(source)), ...added];
    this.dispatchEvent(Object.assign(new Event("inputsourceschange"), { removed, added, session: this }));
  }
}

class FakeXREnvironment {
  readonly gates = new Map<GatePoint, Promise<unknown>>();
  readonly session = new FakeXRSession(this);
  readonly gl = fakeGl(this);
  readonly canvas = { width: 300, height: 150, getContext: () => this.gl } as unknown as HTMLCanvasElement;
  readonly requestSession = vi.fn(async () => {
    await this.gate("requestSession");
    return this.session as unknown as XRSession;
  });

  async gate(point: GatePoint): Promise<void> {
    await this.gates.get(point);
  }
}

interface FakeGl {
  makeXRCompatible: ReturnType<typeof vi.fn>;
  createProgram: ReturnType<typeof vi.fn>;
  deleteProgram: ReturnType<typeof vi.fn>;
}

function fakeGl(env: FakeXREnvironment): FakeGl & WebGL2RenderingContext {
  const known: Record<string, unknown> = {
    makeXRCompatible: vi.fn(async () => {
      await env.gate("makeXRCompatible");
    }),
    createShader: () => ({}),
    createProgram: vi.fn(() => ({})),
    createBuffer: () => ({}),
    deleteProgram: vi.fn(),
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    getAttribLocation: () => 0,
    getUniformLocation: () => ({}),
  };
  return new Proxy(known, {
    get(target, property) {
      if (typeof property !== "string") return undefined;
      if (property in target) return target[property];
      // WebGL enums (COMPILE_STATUS, FRAMEBUFFER, ...) and no-op calls.
      return /^[A-Z0-9_]+$/.test(property) ? 0 : () => undefined;
    },
  }) as unknown as FakeGl & WebGL2RenderingContext;
}

class FakeXRWebGLLayer {
  readonly framebuffer = null;
  getViewport() {
    return null;
  }
}

let env: FakeXREnvironment;
let statuses: QuestXRSessionStatus[];
let handFrames: (QuestXRHandFrame | null)[];
let clock: number;

function controller(): QuestWebXRSession {
  return new QuestWebXRSession(env.canvas, {
    onStatus: (status) => statuses.push(status),
    onHands: (frame) => handFrames.push(frame),
  });
}

const flush = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const lastStatus = () => statuses[statuses.length - 1];
const lastHands = () => handFrames[handFrames.length - 1];
const phases = () => statuses.map((status) => status.phase);

/** Advance one rendered frame (the controller emits hands at most every 50 ms). */
function frame(): QuestXRHandFrame | null | undefined {
  clock += 100;
  const before = handFrames.length;
  env.session.step(clock);
  return handFrames.length > before ? lastHands() : undefined;
}

async function activeController(...hands: FakeHand[]): Promise<QuestWebXRSession> {
  env.session.setHands(...hands);
  const xr = controller();
  await xr.start();
  expect(lastStatus().phase).toBe("active");
  return xr;
}

beforeEach(() => {
  env = new FakeXREnvironment();
  statuses = [];
  handFrames = [];
  clock = 0;
  vi.stubGlobal("isSecureContext", true);
  vi.stubGlobal("XRWebGLLayer", FakeXRWebGLLayer);
  Object.defineProperty(navigator, "xr", {
    configurable: true,
    value: { requestSession: env.requestSession, isSessionSupported: async () => true },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  Reflect.deleteProperty(navigator, "xr");
});

describe("QuestWebXRSession start lifecycle (B21)", () => {
  test("a clean start allocates one renderer and reports active", async () => {
    expect(window.isSecureContext).toBe(true);
    await activeController(new FakeHand("right"));
    expect(env.gl.createProgram).toHaveBeenCalledTimes(1);
    expect(lastStatus()).toMatchObject({ referenceSpace: "local-floor", handTrackingNegotiated: true });
  });

  test.each(["makeXRCompatible", "updateRenderState", "requestReferenceSpace"] as const)(
    "a session the runtime ends during %s never comes back as active or error",
    async (point) => {
      const gate = deferred<void>();
      env.gates.set(point, gate.promise);
      const xr = controller();
      const started = xr.start();
      await flush();
      expect(lastStatus().phase).toBe("starting");

      await env.session.end(); // the user took the headset off
      expect(lastStatus()).toMatchObject({ phase: "idle", message: "The immersive session ended." });

      gate.resolve();
      await expect(started).resolves.toBeUndefined();
      expect(phases()).not.toContain("active");
      expect(phases()).not.toContain("error");
      expect(env.gl.createProgram).not.toHaveBeenCalled();
      expect(lastStatus().phase).toBe("idle");
    },
  );

  test("a late failure from a session that already ended is not reported as an error", async () => {
    const gate = deferred<void>();
    env.gates.set("requestReferenceSpace", gate.promise);
    const started = controller().start();
    await flush();
    await env.session.end();
    gate.reject(new DOMException("The session has ended.", "InvalidStateError"));
    await expect(started).resolves.toBeUndefined();
    expect(phases()).not.toContain("error");
    expect(env.gl.createProgram).not.toHaveBeenCalled();
  });

  test("stop() during startup ends the session and the start never resumes", async () => {
    const gate = deferred<void>();
    env.gates.set("updateRenderState", gate.promise);
    const xr = controller();
    const started = xr.start();
    await flush();
    await xr.stop();
    expect(env.session.ended).toBe(true);
    expect(lastStatus().phase).toBe("idle");
    gate.resolve();
    await started;
    expect(phases()).not.toContain("active");
    expect(env.gl.createProgram).not.toHaveBeenCalled();
  });

  test("dispose() while the headset prompt is open ends the late session quietly", async () => {
    const gate = deferred<void>();
    env.gates.set("requestSession", gate.promise);
    const xr = controller();
    const started = xr.start();
    await flush();
    xr.dispose();
    const reported = statuses.length;

    gate.resolve();
    await expect(started).resolves.toBeUndefined();
    expect(env.session.ended).toBe(true);
    expect(env.session.endCalls).toBe(1);
    expect(statuses).toHaveLength(reported); // a disposed controller reports nothing
    expect(env.gl.createProgram).not.toHaveBeenCalled();
  });

  test("a real startup failure is still reported", async () => {
    env.gl.makeXRCompatible.mockRejectedValueOnce(new Error("GPU is not XR compatible"));
    await expect(controller().start()).rejects.toThrow("GPU is not XR compatible");
    expect(lastStatus()).toMatchObject({ phase: "error", message: "GPU is not XR compatible" });
    expect(env.session.ended).toBe(true);
  });

  test("stop() resolves even when the runtime cannot end the session", async () => {
    const xr = await activeController(new FakeHand("right"));
    env.session.endError = new Error("runtime crashed");
    await expect(xr.stop()).resolves.toBeUndefined();
    expect(lastStatus()).toMatchObject({ phase: "idle", calibration: null });
    expect(env.gl.deleteProgram).toHaveBeenCalledTimes(1);
  });

  test("a frame-loop failure keeps the error status after the session ends", async () => {
    const hand = new FakeHand("right");
    await activeController(hand);
    hand.throwOnPose = true;
    frame();
    await flush();
    expect(env.session.ended).toBe(true);
    expect(lastStatus()).toMatchObject({ phase: "error", message: "XR preview stopped: joint pose lookup failed" });
  });
});

describe("QuestWebXRSession tracking state (B21)", () => {
  test("a reference-space reset clears the calibration and the last hand frame", async () => {
    const xr = await activeController(new FakeHand("right"));
    frame();
    expect(xr.recenterPreview()).toBe(true);
    expect(lastStatus().calibration).not.toBeNull();
    expect(frame()?.primaryHand.relativeWrist).toEqual({ x: 0, y: 0, z: 0 });

    env.session.referenceSpace.dispatchEvent(new Event("reset"));
    expect(lastStatus().calibration).toBeNull();
    expect(lastStatus().message).toMatch(/recentered its origin/);
    expect(lastHands()).toBeNull();
    expect(xr.recenterPreview()).toBe(false); // the old frame is gone
    expect(frame()?.primaryHand.relativeWrist).toBeNull();
  });

  test("the reset listener is removed when the session ends", async () => {
    const xr = await activeController(new FakeHand("right"));
    frame();
    xr.recenterPreview();
    await xr.stop();
    const reported = statuses.length;
    env.session.referenceSpace.dispatchEvent(new Event("reset"));
    expect(statuses).toHaveLength(reported);
  });

  test("removing a hand source clears the hands at once; removing a controller does not", async () => {
    const right = new FakeHand("right");
    const controllerSource = { handedness: "left", targetRayMode: "tracked-pointer", profiles: [] } as unknown as XRInputSource;
    const xr = await activeController(right);
    env.session.inputSources.push(controllerSource);
    expect(frame()).not.toBeNull();

    env.session.changeInputSources([controllerSource]);
    expect(lastHands()).not.toBeNull();

    env.session.changeInputSources([right.source]);
    expect(lastHands()).toBeNull();
    expect(xr.recenterPreview()).toBe(false);
  });

  test("the calibration survives a brief hand loss", async () => {
    const hand = new FakeHand("right");
    const xr = await activeController(hand);
    frame();
    xr.recenterPreview();
    WEBXR_HAND_JOINTS.forEach((name) => hand.missing.add(name));
    expect(frame()).toBeNull();
    expect(lastStatus().calibration).not.toBeNull();

    hand.missing.clear();
    hand.wrist = { ...hand.wrist, x: hand.wrist.x + 0.05 };
    expect(frame()?.primaryHand.relativeWrist?.x).toBeCloseTo(0.05, 9);
  });

  test("a hidden session clears the hands and emits none until visible", async () => {
    await activeController(new FakeHand("right"));
    expect(frame()).not.toBeNull();
    env.session.visibilityState = "hidden";
    env.session.dispatchEvent(new Event("visibilitychange"));
    expect(lastStatus().visibility).toBe("hidden");
    expect(lastHands()).toBeNull();
    expect(frame()).toBeUndefined();

    env.session.visibilityState = "visible";
    env.session.dispatchEvent(new Event("visibilitychange"));
    expect(frame()).not.toBeNull();
  });
});

describe("QuestWebXRSession quality and hand choice (B10/B11)", () => {
  test("inferred joints count half and an inferred or missing wrist caps quality below the gate", async () => {
    const hand = new FakeHand("right");
    await activeController(hand);
    expect(frame()?.primaryHand).toMatchObject({ trackingConfidence: 1, emulatedJointCount: 0 });

    WEBXR_HAND_JOINTS.slice(15).forEach((name) => hand.emulated.add(name));
    const partly = frame()?.primaryHand;
    expect(partly?.emulatedJointCount).toBe(10);
    expect(partly?.trackingConfidence).toBeCloseTo((15 + 0.5 * 10) / 25, 9);
    expect(partly?.poseCompleteness).toBe(1); // completeness alone would have hidden it

    hand.emulated.clear();
    hand.emulated.add("wrist");
    expect(frame()?.primaryHand.trackingConfidence).toBeLessThan(MIN_TRACKING_CONFIDENCE);

    hand.emulated.clear();
    hand.missing.add("wrist");
    const noWrist = frame()?.primaryHand;
    expect(noWrist?.wrist).toBeNull();
    expect(noWrist?.trackingConfidence).toBeLessThan(MIN_TRACKING_CONFIDENCE);
  });

  test("the primary hand is the calibrated hand, else the right hand", async () => {
    const left = new FakeHand("left");
    const right = new FakeHand("right");
    const xr = await activeController(left, right);
    expect(frame()?.primaryHand.handedness).toBe("right");

    env.session.setHands(left);
    frame();
    xr.recenterPreview(); // calibrates the left wrist
    env.session.setHands(left, right);
    const both = frame();
    expect(both?.primaryHand.handedness).toBe("left");
    expect(both?.primaryHand.relativeWrist).not.toBeNull();
  });
});

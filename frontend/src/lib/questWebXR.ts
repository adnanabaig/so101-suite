import { preferredHandIndex, questTrackingConfidence } from "./operatorFrame";

export const WEBXR_HAND_JOINTS = [
  "wrist",
  "thumb-metacarpal",
  "thumb-phalanx-proximal",
  "thumb-phalanx-distal",
  "thumb-tip",
  "index-finger-metacarpal",
  "index-finger-phalanx-proximal",
  "index-finger-phalanx-intermediate",
  "index-finger-phalanx-distal",
  "index-finger-tip",
  "middle-finger-metacarpal",
  "middle-finger-phalanx-proximal",
  "middle-finger-phalanx-intermediate",
  "middle-finger-phalanx-distal",
  "middle-finger-tip",
  "ring-finger-metacarpal",
  "ring-finger-phalanx-proximal",
  "ring-finger-phalanx-intermediate",
  "ring-finger-phalanx-distal",
  "ring-finger-tip",
  "pinky-finger-metacarpal",
  "pinky-finger-phalanx-proximal",
  "pinky-finger-phalanx-intermediate",
  "pinky-finger-phalanx-distal",
  "pinky-finger-tip",
] as const satisfies readonly XRHandJoint[];

export const QUEST_CANONICAL_JOINT_COUNT = 26;

export type QuestCanonicalJointName = "palm" | XRHandJoint;
export type QuestXRPhase =
  | "idle"
  | "starting"
  | "active"
  | "ending"
  | "error";

export interface QuestXRVector3 {
  x: number;
  y: number;
  z: number;
}

export interface QuestXRQuaternion extends QuestXRVector3 {
  w: number;
}

export interface QuestXRJoint {
  name: QuestCanonicalJointName;
  position: QuestXRVector3;
  orientation: QuestXRQuaternion;
  radius: number | null;
  derived: boolean;
  /** The runtime inferred this pose (`XRPose.emulatedPosition`) instead of tracking it. */
  emulated: boolean;
}

export interface QuestXRHandSample {
  handedness: XRHandedness;
  joints: QuestXRJoint[];
  /** Native WebXR poses only; the derived palm is excluded. */
  observedJointCount: number;
  /** Fraction of the 25 standard WebXR joint poses available in this frame. */
  poseCompleteness: number;
  /** Native poses the runtime inferred (`emulatedPosition`) rather than tracked. */
  emulatedJointCount: number;
  /**
   * B11: the quality signal sent as `confidence`. Completeness with inferred
   * joints down-weighted; a missing or inferred wrist caps it below the gate.
   */
  trackingConfidence: number;
  wrist: QuestXRJoint | null;
  thumbTip: QuestXRJoint | null;
  indexTip: QuestXRJoint | null;
  pinchDistanceM: number | null;
  pinch: number;
  relativeWrist: QuestXRVector3 | null;
}

export interface QuestXRHandFrame {
  capturedAt: number;
  hands: QuestXRHandSample[];
  primaryHand: QuestXRHandSample;
}

export interface QuestXRProbe {
  secureContext: boolean;
  apiAvailable: boolean;
  immersiveVrSupported: boolean;
  detail: string;
}

export interface QuestXRCalibration {
  handedness: XRHandedness;
  origin: QuestXRVector3;
  capturedAt: number;
}

export interface QuestXRSessionStatus {
  phase: QuestXRPhase;
  message: string;
  visibility: XRVisibilityState | null;
  referenceSpace: XRReferenceSpaceType | null;
  handTrackingNegotiated: boolean | null;
  frameRate: number | null;
  startedAt: number | null;
  calibration: QuestXRCalibration | null;
}

export interface QuestWebXRCallbacks {
  onStatus: (status: QuestXRSessionStatus) => void;
  /** Null is emitted immediately when no hand pose is available. */
  onHands: (frame: QuestXRHandFrame | null) => void;
}

const INITIAL_STATUS: QuestXRSessionStatus = {
  phase: "idle",
  message: "Ready to request a headset session.",
  visibility: null,
  referenceSpace: null,
  handTrackingNegotiated: null,
  frameRate: null,
  startedAt: null,
  calibration: null,
};

function clamp(value: number, minimum = 0, maximum = 1): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function copyVector(value: DOMPointReadOnly): QuestXRVector3 {
  return { x: value.x, y: value.y, z: value.z };
}

function copyQuaternion(value: DOMPointReadOnly): QuestXRQuaternion {
  return { x: value.x, y: value.y, z: value.z, w: value.w };
}

function distance(a: QuestXRVector3, b: QuestXRVector3): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function errorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "The immersive session was denied or did not begin from a headset gesture.";
    }
    if (error.name === "NotSupportedError") {
      return "This browser/runtime does not support the requested immersive VR session.";
    }
    if (error.name === "InvalidStateError") {
      return "Another immersive session is already active.";
    }
  }
  return error instanceof Error ? error.message : String(error);
}

export async function probeQuestWebXR(): Promise<QuestXRProbe> {
  if (!window.isSecureContext) {
    return {
      secureContext: false,
      apiAvailable: "xr" in navigator,
      immersiveVrSupported: false,
      detail: "WebXR immersive sessions require a trusted HTTPS origin.",
    };
  }
  if (!navigator.xr) {
    return {
      secureContext: true,
      apiAvailable: false,
      immersiveVrSupported: false,
      detail: "navigator.xr is not exposed by this browser.",
    };
  }
  try {
    const supported = await navigator.xr.isSessionSupported("immersive-vr");
    return {
      secureContext: true,
      apiAvailable: true,
      immersiveVrSupported: supported,
      detail: supported
        ? "An immersive VR runtime is available. Hand tracking is negotiated on entry."
        : "No immersive VR device/runtime is available to this browser.",
    };
  } catch (error) {
    return {
      secureContext: true,
      apiAvailable: true,
      immersiveVrSupported: false,
      detail: errorMessage(error),
    };
  }
}

function makeJoint(
  name: QuestCanonicalJointName,
  pose: XRJointPose,
): QuestXRJoint {
  return {
    name,
    position: copyVector(pose.transform.position),
    orientation: copyQuaternion(pose.transform.orientation),
    radius: typeof pose.radius === "number" ? pose.radius : null,
    derived: false,
    emulated: pose.emulatedPosition === true,
  };
}

function derivePalm(joints: ReadonlyMap<string, QuestXRJoint>): QuestXRJoint | null {
  const wrist = joints.get("wrist");
  const index = joints.get("index-finger-metacarpal");
  const middle = joints.get("middle-finger-metacarpal");
  const pinky = joints.get("pinky-finger-metacarpal");
  if (!wrist || !index || !middle || !pinky) return null;
  return {
    name: "palm",
    position: {
      x: (wrist.position.x + index.position.x + middle.position.x + pinky.position.x) / 4,
      y: (wrist.position.y + index.position.y + middle.position.y + pinky.position.y) / 4,
      z: (wrist.position.z + index.position.z + middle.position.z + pinky.position.z) / 4,
    },
    orientation: { ...wrist.orientation },
    radius: null,
    derived: true,
    emulated: [wrist, index, middle, pinky].some((joint) => joint.emulated),
  };
}

function extractHand(
  frame: XRFrame,
  referenceSpace: XRReferenceSpace,
  source: XRInputSource,
  calibration: QuestXRCalibration | null,
): QuestXRHandSample | null {
  if (!source.hand || !frame.getJointPose) return null;
  const hand = source.hand as ReadonlyMap<string, XRJointSpace>;
  const observed = new Map<string, QuestXRJoint>();

  for (const name of WEBXR_HAND_JOINTS) {
    const jointSpace = hand.get(name);
    if (!jointSpace) continue;
    const pose = frame.getJointPose(jointSpace, referenceSpace);
    if (pose) observed.set(name, makeJoint(name, pose));
  }

  // OpenXR exposes a 26th palm joint. The current WebXR hand-input spec exposes
  // 25 named joints, but some runtimes surface palm as an extension. Preserve it
  // when present; otherwise derive a clearly marked preview-only palm anchor.
  const runtimePalmSpace = hand.get("palm");
  let palm: QuestXRJoint | null = null;
  if (runtimePalmSpace) {
    const pose = frame.getJointPose(runtimePalmSpace, referenceSpace);
    if (pose) palm = makeJoint("palm", pose);
  }
  palm ??= derivePalm(observed);

  const wrist = observed.get("wrist") ?? null;
  const thumbTip = observed.get("thumb-tip") ?? null;
  const indexTip = observed.get("index-finger-tip") ?? null;
  if (!wrist && observed.size === 0) return null;

  const pinchDistanceM =
    thumbTip && indexTip ? distance(thumbTip.position, indexTip.position) : null;
  const pinch =
    pinchDistanceM === null ? 0 : 1 - clamp((pinchDistanceM - 0.018) / 0.065);
  const joints = [
    ...(palm ? [palm] : []),
    ...WEBXR_HAND_JOINTS.flatMap((name) => {
      const joint = observed.get(name);
      return joint ? [joint] : [];
    }),
  ];
  const relativeWrist =
    wrist && calibration && calibration.handedness === source.handedness
      ? {
          x: wrist.position.x - calibration.origin.x,
          y: wrist.position.y - calibration.origin.y,
          z: wrist.position.z - calibration.origin.z,
        }
      : null;

  const emulatedJointCount = Array.from(observed.values()).filter((joint) => joint.emulated).length;

  return {
    handedness: source.handedness,
    joints,
    observedJointCount: observed.size + (palm && !palm.derived ? 1 : 0),
    poseCompleteness: observed.size / WEBXR_HAND_JOINTS.length,
    emulatedJointCount,
    trackingConfidence: questTrackingConfidence({
      tracked: observed.size - emulatedJointCount,
      emulated: emulatedJointCount,
      total: WEBXR_HAND_JOINTS.length,
      wrist: !wrist ? "missing" : wrist.emulated ? "emulated" : "tracked",
    }),
    wrist,
    thumbTip,
    indexTip,
    pinchDistanceM,
    pinch,
    relativeWrist,
  };
}

function extractHands(
  frame: XRFrame,
  referenceSpace: XRReferenceSpace,
  calibration: QuestXRCalibration | null,
): QuestXRHandFrame | null {
  const hands: QuestXRHandSample[] = [];
  for (const source of frame.session.inputSources) {
    const hand = extractHand(frame, referenceSpace, source, calibration);
    if (hand) hands.push(hand);
  }
  if (hands.length === 0) return null;
  // B10: the calibrated hand, else the operator's right hand, else the first.
  const preferred = preferredHandIndex(
    hands.map((hand) => (hand.handedness === "none" ? "unknown" : hand.handedness)),
    calibration && calibration.handedness !== "none" ? calibration.handedness : null,
  );
  const primaryHand = hands[preferred >= 0 ? preferred : 0];
  return { capturedAt: Date.now(), hands, primaryHand };
}

type XRCompatibleGL = WebGLRenderingContext | WebGL2RenderingContext;

function compileShader(
  gl: XRCompatibleGL,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not allocate a WebGL shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || "unknown shader compile error";
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

/** Minimal headset renderer: joint points only, with no robot actuation path. */
class QuestHandPointRenderer {
  private readonly program: WebGLProgram;
  private readonly buffer: WebGLBuffer;
  private readonly positionAttribute: number;
  private readonly projectionUniform: WebGLUniformLocation;
  private readonly viewUniform: WebGLUniformLocation;
  private readonly colorUniform: WebGLUniformLocation;

  constructor(private readonly gl: XRCompatibleGL) {
    const vertex = compileShader(
      gl,
      gl.VERTEX_SHADER,
      `
        attribute vec3 a_position;
        uniform mat4 u_projection;
        uniform mat4 u_view;
        void main() {
          gl_Position = u_projection * u_view * vec4(a_position, 1.0);
          gl_PointSize = 10.0;
        }
      `,
    );
    const fragment = compileShader(
      gl,
      gl.FRAGMENT_SHADER,
      `
        precision mediump float;
        uniform vec4 u_color;
        void main() {
          vec2 point = gl_PointCoord - vec2(0.5);
          if (dot(point, point) > 0.25) discard;
          gl_FragColor = u_color;
        }
      `,
    );
    const program = gl.createProgram();
    if (!program) throw new Error("Could not allocate a WebGL program.");
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) || "unknown program link error";
      gl.deleteProgram(program);
      throw new Error(log);
    }
    const buffer = gl.createBuffer();
    if (!buffer) {
      gl.deleteProgram(program);
      throw new Error("Could not allocate a WebGL hand buffer.");
    }
    const positionAttribute = gl.getAttribLocation(program, "a_position");
    const projectionUniform = gl.getUniformLocation(program, "u_projection");
    const viewUniform = gl.getUniformLocation(program, "u_view");
    const colorUniform = gl.getUniformLocation(program, "u_color");
    if (
      positionAttribute < 0 ||
      !projectionUniform ||
      !viewUniform ||
      !colorUniform
    ) {
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      throw new Error("The WebGL hand-preview shader is incomplete.");
    }
    this.program = program;
    this.buffer = buffer;
    this.positionAttribute = positionAttribute;
    this.projectionUniform = projectionUniform;
    this.viewUniform = viewUniform;
    this.colorUniform = colorUniform;
  }

  render(
    frame: XRFrame,
    referenceSpace: XRReferenceSpace,
    layer: XRWebGLLayer,
    hands: readonly QuestXRHandSample[],
    calibration: QuestXRCalibration | null,
  ): void {
    const viewer = frame.getViewerPose(referenceSpace);
    if (!viewer) return;
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clearColor(0.015, 0.025, 0.055, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.enableVertexAttribArray(this.positionAttribute);
    gl.vertexAttribPointer(this.positionAttribute, 3, gl.FLOAT, false, 0, 0);

    for (const view of viewer.views) {
      const viewport = layer.getViewport(view);
      if (!viewport) continue;
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.uniformMatrix4fv(this.projectionUniform, false, view.projectionMatrix);
      gl.uniformMatrix4fv(this.viewUniform, false, view.transform.inverse.matrix);

      for (const hand of hands) {
        const positions = new Float32Array(hand.joints.length * 3);
        hand.joints.forEach((joint, index) => {
          positions[index * 3] = joint.position.x;
          positions[index * 3 + 1] = joint.position.y;
          positions[index * 3 + 2] = joint.position.z;
        });
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
        if (hand.handedness === "left") {
          gl.uniform4f(this.colorUniform, 0.16, 0.82, 0.98, 1);
        } else {
          gl.uniform4f(this.colorUniform, 0.98, 0.72, 0.2, 1);
        }
        gl.drawArrays(gl.POINTS, 0, hand.joints.length);
      }

      if (calibration) {
        const origin = new Float32Array([
          calibration.origin.x,
          calibration.origin.y,
          calibration.origin.z,
        ]);
        gl.bufferData(gl.ARRAY_BUFFER, origin, gl.DYNAMIC_DRAW);
        gl.uniform4f(this.colorUniform, 0.68, 0.35, 0.98, 1);
        gl.drawArrays(gl.POINTS, 0, 1);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  dispose(): void {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteProgram(this.program);
  }
}

/** End a session, ignoring the error a session that already ended throws. */
async function endQuietly(session: XRSession): Promise<void> {
  try {
    await session.end();
  } catch {
    // Local cleanup is authoritative even if the runtime cannot end cleanly.
  }
}

async function requestBestReferenceSpace(
  session: XRSession,
): Promise<{ space: XRReferenceSpace; type: XRReferenceSpaceType }> {
  const candidates: readonly XRReferenceSpaceType[] = ["local-floor", "local", "viewer"];
  let lastError: unknown = null;
  for (const type of candidates) {
    try {
      const space = await session.requestReferenceSpace(type);
      return { space, type };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError ?? new Error("No usable WebXR reference space is available.");
}

/**
 * Owns one immersive session and every resource attached to it. This controller
 * intentionally exposes hand pose preview only; it never acquires a robot lease
 * and never calls a motion endpoint.
 */
export class QuestWebXRSession {
  private status: QuestXRSessionStatus = { ...INITIAL_STATUS };
  private session: XRSession | null = null;
  private referenceSpace: XRReferenceSpace | null = null;
  private layer: XRWebGLLayer | null = null;
  private renderer: QuestHandPointRenderer | null = null;
  private gl: XRCompatibleGL | null = null;
  private frameRequest: number | null = null;
  private latestFrame: QuestXRHandFrame | null = null;
  private lastSampleEmitAt = 0;
  private hadTracking = false;
  private disposed = false;
  /**
   * Bumped by every cleanup (session end, stop, failure, dispose). start()
   * re-checks it after each await so a session that ended meanwhile never
   * comes back as "active" or leaves a renderer behind (B21).
   */
  private generation = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly callbacks: QuestWebXRCallbacks,
  ) {
    callbacks.onStatus(this.snapshot());
  }

  private snapshot(): QuestXRSessionStatus {
    return {
      ...this.status,
      calibration: this.status.calibration
        ? {
            ...this.status.calibration,
            origin: { ...this.status.calibration.origin },
          }
        : null,
    };
  }

  private updateStatus(patch: Partial<QuestXRSessionStatus>): void {
    this.status = { ...this.status, ...patch };
    if (!this.disposed) this.callbacks.onStatus(this.snapshot());
  }

  private readonly onSessionEnd = () => {
    this.cleanup("The immersive session ended.", "idle");
  };

  private readonly onVisibilityChange = () => {
    const visibility = this.session?.visibilityState ?? null;
    this.updateStatus({ visibility });
    if (visibility !== "visible") this.clearTrackedHands();
  };

  private readonly onInputSourcesChange = (event: XRInputSourcesChangeEvent) => {
    // A removed hand must not leave its last pose on screen or in latestFrame
    // (which recenter reads). Hands still present come back on the next frame.
    if (
      event.removed.some((source) => source.hand) ||
      !this.session ||
      !Array.from(this.session.inputSources).some((source) => source.hand)
    ) {
      this.clearTrackedHands();
    }
  };

  /**
   * The runtime moved the reference-space origin (for example the user held
   * the Meta button to recenter). The calibration origin and the last frame
   * are in the old frame, so neither is valid any more (B21).
   */
  private readonly onReferenceReset = () => {
    this.clearTrackedHands();
    this.updateStatus({
      calibration: null,
      message:
        "The headset recentered its origin, so the preview origin was cleared. Recenter from a neutral wrist pose.",
    });
  };

  async start(): Promise<void> {
    if (this.session || this.status.phase === "starting") return;
    if (!window.isSecureContext) {
      const message = "A trusted HTTPS context is required for immersive WebXR.";
      this.updateStatus({ phase: "error", message });
      throw new Error(message);
    }
    const xr = navigator.xr;
    if (!xr) {
      const message = "navigator.xr is unavailable in this browser.";
      this.updateStatus({ phase: "error", message });
      throw new Error(message);
    }

    this.disposed = false;
    const generation = ++this.generation;
    const superseded = () => this.disposed || generation !== this.generation;
    this.updateStatus({ phase: "starting", message: "Waiting for headset approval…" });
    let requestedSession: XRSession | null = null;
    try {
      // Keep requestSession as the first asynchronous operation so the browser's
      // transient user activation is still valid.
      requestedSession = await xr.requestSession("immersive-vr", {
        optionalFeatures: ["hand-tracking", "local-floor"],
      });
      if (superseded()) {
        await endQuietly(requestedSession);
        return;
      }
      this.session = requestedSession;
      requestedSession.addEventListener("end", this.onSessionEnd);
      requestedSession.addEventListener("visibilitychange", this.onVisibilityChange);
      requestedSession.addEventListener("inputsourceschange", this.onInputSourcesChange);

      const contextOptions: WebGLContextAttributes = {
        alpha: false,
        antialias: true,
        depth: true,
        stencil: false,
        xrCompatible: true,
      };
      const gl =
        this.canvas.getContext("webgl2", contextOptions) ??
        this.canvas.getContext("webgl", contextOptions);
      if (!gl) throw new Error("Could not create an XR-compatible WebGL context.");
      await gl.makeXRCompatible();
      if (superseded()) return await endQuietly(requestedSession);
      const layer = new XRWebGLLayer(requestedSession, gl, {
        alpha: false,
        antialias: true,
        depth: true,
        stencil: false,
      });
      await requestedSession.updateRenderState({
        baseLayer: layer,
        depthNear: 0.03,
        depthFar: 20,
      });
      if (superseded()) return await endQuietly(requestedSession);
      const reference = await requestBestReferenceSpace(requestedSession);
      if (superseded()) return await endQuietly(requestedSession);

      reference.space.addEventListener("reset", this.onReferenceReset);
      this.gl = gl;
      this.layer = layer;
      this.referenceSpace = reference.space;
      this.renderer = new QuestHandPointRenderer(gl);
      this.updateStatus({
        phase: "active",
        message: "Immersive preview active. Physical motion is disabled.",
        visibility: requestedSession.visibilityState,
        referenceSpace: reference.type,
        handTrackingNegotiated:
          requestedSession.enabledFeatures?.includes("hand-tracking") ?? null,
        frameRate: requestedSession.frameRate ?? null,
        startedAt: Date.now(),
        calibration: null,
      });
      this.frameRequest = requestedSession.requestAnimationFrame(this.onXRFrame);
    } catch (error) {
      if (superseded()) {
        // Already cleaned up (the session ended, or stop/dispose ran): a late
        // failure from the dead session is not a new error to report.
        if (requestedSession) await endQuietly(requestedSession);
        return;
      }
      const message = errorMessage(error);
      if (requestedSession) await endQuietly(requestedSession);
      this.cleanup(message, "error");
      throw error;
    }
  }

  private readonly onXRFrame: XRFrameRequestCallback = (time, frame) => {
    const session = this.session;
    const referenceSpace = this.referenceSpace;
    const layer = this.layer;
    const renderer = this.renderer;
    if (!session || !referenceSpace || !layer || !renderer) return;
    this.frameRequest = session.requestAnimationFrame(this.onXRFrame);
    try {
      if (session.visibilityState !== "visible") {
        renderer.render(frame, referenceSpace, layer, [], this.status.calibration);
        if (this.hadTracking) this.clearTrackedHands();
        return;
      }
      const hands = extractHands(frame, referenceSpace, this.status.calibration);
      renderer.render(
        frame,
        referenceSpace,
        layer,
        hands?.hands ?? [],
        this.status.calibration,
      );
      this.latestFrame = hands;
      if (!hands) {
        if (this.hadTracking) this.clearTrackedHands();
        return;
      }
      this.hadTracking = true;
      if (time - this.lastSampleEmitAt >= 50) {
        this.lastSampleEmitAt = time;
        this.callbacks.onHands(hands);
      }
    } catch (error) {
      void this.fail(error);
    }
  };

  private clearTrackedHands(): void {
    this.latestFrame = null;
    this.hadTracking = false;
    this.lastSampleEmitAt = 0;
    if (!this.disposed) this.callbacks.onHands(null);
  }

  recenterPreview(): boolean {
    const hand = this.latestFrame?.primaryHand;
    if (!hand?.wrist) return false;
    const calibration: QuestXRCalibration = {
      handedness: hand.handedness,
      origin: { ...hand.wrist.position },
      capturedAt: Date.now(),
    };
    this.updateStatus({
      calibration,
      message: `Preview origin captured from the ${hand.handedness} wrist.`,
    });
    return true;
  }

  clearCalibration(): void {
    this.updateStatus({
      calibration: null,
      message: "Preview calibration cleared; runtime coordinates are shown.",
    });
  }

  async stop(): Promise<void> {
    const session = this.session;
    if (!session) {
      this.cleanup("Ready to request a headset session.", "idle");
      return;
    }
    this.updateStatus({ phase: "ending", message: "Ending immersive session…" });
    await endQuietly(session);
    if (this.session === session) {
      this.cleanup("The immersive session ended.", "idle");
    }
  }

  private async fail(error: unknown): Promise<void> {
    const session = this.session;
    if (!session) return;
    // Clean up first: that detaches the "end" listener, so the error stays
    // the reported status, and a session started later is never torn down.
    this.cleanup(`XR preview stopped: ${errorMessage(error)}`, "error");
    await endQuietly(session);
  }

  private cleanup(message: string, phase: "idle" | "error"): void {
    this.generation += 1;
    const session = this.session;
    if (session && this.frameRequest !== null) {
      session.cancelAnimationFrame(this.frameRequest);
    }
    if (session) {
      session.removeEventListener("end", this.onSessionEnd);
      session.removeEventListener("visibilitychange", this.onVisibilityChange);
      session.removeEventListener("inputsourceschange", this.onInputSourcesChange);
    }
    this.referenceSpace?.removeEventListener("reset", this.onReferenceReset);
    this.renderer?.dispose();
    this.gl?.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.renderer = null;
    this.referenceSpace = null;
    this.layer = null;
    this.gl = null;
    this.session = null;
    this.frameRequest = null;
    this.canvas.width = 1;
    this.canvas.height = 1;
    this.clearTrackedHands();
    this.updateStatus({
      ...INITIAL_STATUS,
      phase,
      message,
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const session = this.session;
    this.cleanup("XR preview disposed.", "idle");
    if (session) void session.end().catch(() => undefined);
  }
}

/**
 * One operator frame for every hand-tracking path (B10).
 *
 *   +x    the operator's right
 *   +y    up
 *   +z    away from the operator (reach). A monocular camera has no z (B5).
 *   roll  counter-clockwise positive as the operator sees their own hand in a
 *         mirror, which is the selfie view the Connect page shows
 *
 * Camera (MediaPipe on an unmirrored front camera, `facingMode: "user"`): the
 * frame is what the camera sees, so the operator's right is the image's left
 * and image y grows downward. MediaPipe's handedness labels assume a mirrored
 * (selfie) input, so on these frames each label names the other hand and is
 * swapped to get the anatomical hand.
 *
 * Quest (WebXR): reference spaces are right-handed with +x to the user's
 * right, +y up and -z forward, and `XRInputSource.handedness` is anatomical.
 * Only depth is negated.
 *
 * Both paths track the operator's right hand unless the operator calibrated
 * another one. The server maps x to shoulder_pan, so the same gesture now
 * produces the same pan sign on both paths.
 */

/** The 21 MediaPipe hand landmarks; only these indices are read here. */
export const WRIST = 0;
export const THUMB_TIP = 4;
export const INDEX_MCP = 5;
export const INDEX_TIP = 8;
export const PINKY_MCP = 17;
const HAND_LANDMARK_COUNT = 21;
/** The landmarks that position, roll and pinch are computed from. */
const KEY_LANDMARKS: readonly number[] = [WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP];
/**
 * Quality ceiling when a key landmark (or the Quest wrist) is not really
 * observed. It sits below the 0.72 tracking gate, so a pose built from
 * extrapolated or inferred key points can never drive the arm.
 */
export const KEY_POINT_MISSING_CAP = 0.5;

export type AnatomicalHand = "left" | "right" | "unknown";

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface HandednessCategory {
  categoryName: string;
  score: number;
}

export function clampUnit(value: number): number {
  return Math.min(1, Math.max(-1, value));
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function distance(a: LandmarkPoint, b: LandmarkPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** Undo MediaPipe's mirrored-input assumption: on raw frames "Left" is the right hand. */
export function anatomicalHandFromMediaPipe(label: string | undefined): AnatomicalHand {
  const normalized = label?.trim().toLowerCase();
  if (normalized === "left") return "right";
  if (normalized === "right") return "left";
  return "unknown";
}

/**
 * Index of the hand to track: the `preferred` hand, else the operator's right
 * hand, else -1 so the caller can apply its own fallback.
 */
export function preferredHandIndex(
  hands: readonly AnatomicalHand[],
  preferred: AnatomicalHand | null = null,
): number {
  if (preferred && preferred !== "unknown") {
    const match = hands.indexOf(preferred);
    if (match >= 0) return match;
  }
  return hands.indexOf("right");
}

/** Raw image point (0..1, unmirrored) to the operator frame (-1..1). */
export function cameraOperatorPoint(point: { x: number; y: number }): { x: number; y: number } {
  return { x: clampUnit((0.5 - point.x) * 2), y: clampUnit((0.5 - point.y) * 2) };
}

/** Where a raw landmark sits on the mirrored (selfie) preview, as box fractions. */
export function mirroredDisplayPoint(point: { x: number; y: number }): { left: number; top: number } {
  return { left: 1 - point.x, top: point.y };
}

/**
 * Raw roll of the index-to-pinky knuckle line in the operator frame, in
 * radians. A right hand held up with the palm toward the camera reads about
 * 0; a left hand reads about ±π, which is why consumers must reference and
 * unwrap it (B6) instead of using it directly.
 */
export function cameraRollRad(indexMcp: LandmarkPoint, pinkyMcp: LandmarkPoint): number {
  return Math.atan2(-(pinkyMcp.y - indexMcp.y), -(pinkyMcp.x - indexMcp.x));
}

/** Wrap an angle into (-π, π]. */
export function wrapAngle(rad: number): number {
  const turn = 2 * Math.PI;
  const wrapped = rad - turn * Math.floor((rad + Math.PI) / turn);
  return wrapped <= -Math.PI ? wrapped + turn : wrapped;
}

/**
 * Roll referenced to a neutral orientation and unwrapped (B6). The neutral is
 * captured when the deadman engages (and when tracking starts or reacquires a
 * hand, for previews). Each update adds the wrapped frame-to-frame change, so
 * crossing the atan2 branch cut moves the value by a few degrees instead of
 * flipping it by a full turn.
 */
export interface RollReference {
  /** Raw roll at capture. */
  readonly neutral: number;
  /** The previous raw roll, for unwrapping. */
  readonly last: number;
  /** Continuous roll since capture. */
  readonly unwrapped: number;
}

/** Relative roll that maps to ±1; beyond it the value saturates. */
export const ROLL_RANGE_RAD = Math.PI;

export function engageRoll(rawRad: number): RollReference {
  const raw = Number.isFinite(rawRad) ? rawRad : 0;
  return { neutral: raw, last: raw, unwrapped: raw };
}

/** Relative roll in [-1, 1]: 0 at the neutral, ±1 at ±ROLL_RANGE_RAD. */
export function relativeRoll(reference: RollReference): number {
  return clampUnit((reference.unwrapped - reference.neutral) / ROLL_RANGE_RAD);
}

export function updateRoll(
  reference: RollReference,
  rawRad: number,
): { reference: RollReference; relative: number } {
  if (!Number.isFinite(rawRad)) return { reference, relative: relativeRoll(reference) };
  const next: RollReference = {
    neutral: reference.neutral,
    last: rawRad,
    unwrapped: reference.unwrapped + wrapAngle(rawRad - reference.last),
  };
  return { reference: next, relative: relativeRoll(next) };
}

/* ------------------------------------------------------------------------ */
/* Camera tracking quality (B11)                                             */
/* ------------------------------------------------------------------------ */

/**
 * Fraction of the 21 landmarks that lie inside the image. MediaPipe
 * extrapolates landmarks it cannot see, and its hand landmarks carry no
 * per-point visibility, so being inside the frame is the presence signal. If
 * a key landmark is outside, the value is capped below the tracking gate.
 */
export function landmarkPresence(landmarks: readonly LandmarkPoint[]): number {
  if (landmarks.length < HAND_LANDMARK_COUNT) return 0;
  let inside = 0;
  let keyMissing = false;
  for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
    const point = landmarks[index];
    const ok =
      Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      point.x >= 0 &&
      point.x <= 1 &&
      point.y >= 0 &&
      point.y <= 1;
    if (ok) inside += 1;
    else if (KEY_LANDMARKS.includes(index)) keyMissing = true;
  }
  const fraction = inside / HAND_LANDMARK_COUNT;
  return keyMissing ? Math.min(fraction, KEY_POINT_MISSING_CAP) : fraction;
}

/** Wrist speed (palm widths per second) up to which tracking counts as fully steady. */
export const STEADY_SPEED_PALMS_PER_S = 10;
/** Wrist speed at which a frame-to-frame jump is treated as a glitch or a hand swap. */
export const IMPLAUSIBLE_SPEED_PALMS_PER_S = 30;
/** Consecutive plausible frames before confidence can reach its full value. */
export const SETTLE_FRAMES = 3;
/** A gap longer than this means tracking restarted, not that the hand moved. */
export const REACQUIRE_GAP_MS = 250;
const MIN_FRAME_INTERVAL_S = 1 / 60;

export interface CameraQuality {
  /** Landmark presence (0..1). */
  presence: number;
  /** Temporal stability (0..1): 1 when the wrist moves plausibly. */
  stability: number;
  /** Settling (0..1): ramps over SETTLE_FRAMES after (re)acquisition. */
  settle: number;
}

export interface CameraQualityState {
  hand: AnatomicalHand;
  wrist: { x: number; y: number };
  palmWidth: number;
  capturedAt: number;
  streak: number;
}

export interface CameraQualityObservation {
  hand: AnatomicalHand;
  /** Raw image coordinates of the wrist. */
  wrist: { x: number; y: number };
  palmWidth: number;
  capturedAt: number;
  presence: number;
}

/**
 * Advance the per-tracker quality state. Confidence is presence x stability
 * x settle. A new hand, a long gap or an implausible jump restarts settling,
 * so the 0.72 gate stays shut for a few frames after any discontinuity.
 */
export function nextCameraQuality(
  previous: CameraQualityState | null,
  observation: CameraQualityObservation,
): { state: CameraQualityState; quality: CameraQuality; confidence: number } {
  let stability = 1;
  let streak = 1;
  const continuous =
    previous !== null &&
    previous.hand === observation.hand &&
    observation.capturedAt - previous.capturedAt <= REACQUIRE_GAP_MS;
  if (previous && continuous) {
    const seconds = Math.max(
      (observation.capturedAt - previous.capturedAt) / 1000,
      MIN_FRAME_INTERVAL_S,
    );
    const palm = Math.max((previous.palmWidth + observation.palmWidth) / 2, Number.EPSILON);
    const travelled = Math.hypot(
      observation.wrist.x - previous.wrist.x,
      observation.wrist.y - previous.wrist.y,
    );
    const speed = travelled / palm / seconds;
    stability =
      1 -
      clamp01(
        (speed - STEADY_SPEED_PALMS_PER_S) /
          (IMPLAUSIBLE_SPEED_PALMS_PER_S - STEADY_SPEED_PALMS_PER_S),
      );
    streak = stability > 0 ? previous.streak + 1 : 1;
  }
  const settle = Math.min(1, streak / SETTLE_FRAMES);
  const presence = clamp01(observation.presence);
  return {
    state: {
      hand: observation.hand,
      wrist: { ...observation.wrist },
      palmWidth: observation.palmWidth,
      capturedAt: observation.capturedAt,
      streak,
    },
    quality: { presence, stability, settle },
    confidence: presence * stability * settle,
  };
}

/** One operator-facing sentence for whichever quality factor is limiting. */
export function cameraQualityHint(quality: CameraQuality): string {
  const worst = Math.min(quality.presence, quality.stability, quality.settle);
  if (worst === quality.presence) return "Keep the whole hand, wrist included, inside the camera view.";
  if (worst === quality.stability) return "Tracking jumped. Move more slowly, with only one hand in view.";
  return "Tracking is settling. Hold the hand steady for a moment.";
}

/* ------------------------------------------------------------------------ */
/* Camera sample                                                             */
/* ------------------------------------------------------------------------ */

export interface HandTrackingSample {
  /** Raw MediaPipe landmarks in image coordinates (unmirrored). */
  landmarks: LandmarkPoint[];
  /** The anatomical hand, after undoing MediaPipe's mirrored labels. */
  handedness: AnatomicalHand;
  /** B11: tracking quality (presence x stability x settle), never the handedness score. */
  confidence: number;
  quality: CameraQuality;
  /** Operator frame, -1..1. */
  x: number;
  y: number;
  /** Raw knuckle-line roll in radians; reference and unwrap it before use (B6). */
  rollRad: number;
  pinch: number;
  capturedAt: number;
}

export interface CameraFrameResult {
  landmarks: readonly (readonly LandmarkPoint[])[];
  handedness: readonly (readonly HandednessCategory[])[];
}

/**
 * Turn one MediaPipe result into an operator-frame sample. Returns a null
 * sample (and a reset quality state) when no complete hand is visible.
 */
export function cameraSampleFromResult(
  result: CameraFrameResult,
  capturedAt: number,
  previous: CameraQualityState | null,
): { sample: HandTrackingSample | null; state: CameraQualityState | null } {
  const count = result.landmarks.length;
  if (count === 0) return { sample: null, state: null };
  const hands = Array.from({ length: count }, (_, index) =>
    anatomicalHandFromMediaPipe(result.handedness[index]?.[0]?.categoryName),
  );
  let selected = preferredHandIndex(hands, previous?.hand ?? null);
  if (selected < 0) {
    // Neither the previous nor a right hand: take the most certain label.
    selected = 0;
    for (let index = 1; index < count; index += 1) {
      const score = result.handedness[index]?.[0]?.score ?? 0;
      if (score > (result.handedness[selected]?.[0]?.score ?? 0)) selected = index;
    }
  }
  const landmarks = result.landmarks[selected];
  if (landmarks.length < HAND_LANDMARK_COUNT) return { sample: null, state: null };

  const wrist = landmarks[WRIST];
  const indexMcp = landmarks[INDEX_MCP];
  const pinkyMcp = landmarks[PINKY_MCP];
  const palmWidth = Math.max(distance(indexMcp, pinkyMcp), 0.025);
  const pinchRatio = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / palmWidth;
  const point = cameraOperatorPoint(wrist);
  const next = nextCameraQuality(previous, {
    hand: hands[selected],
    wrist: { x: wrist.x, y: wrist.y },
    palmWidth,
    capturedAt,
    presence: landmarkPresence(landmarks),
  });
  return {
    sample: {
      landmarks: landmarks.map((landmark) => ({ x: landmark.x, y: landmark.y, z: landmark.z })),
      handedness: hands[selected],
      confidence: next.confidence,
      quality: next.quality,
      x: point.x,
      y: point.y,
      rollRad: cameraRollRad(indexMcp, pinkyMcp),
      pinch: clamp01(1 - (pinchRatio - 0.18) / 0.72),
      capturedAt,
    },
    state: next.state,
  };
}

/* ------------------------------------------------------------------------ */
/* Quest                                                                     */
/* ------------------------------------------------------------------------ */

/** Wrist travel from the calibrated origin that maps to ±1. */
export const QUEST_WRIST_RANGE_M = 0.25;
/** Forward reach from the calibrated origin that maps to ±1. */
export const QUEST_DEPTH_RANGE_M = 0.2;

/** Calibrated wrist offset (metres, WebXR axes) to operator-frame preview intent. */
export function questOperatorWrist(relative: { x: number; y: number; z: number }): {
  normalized_x: number;
  normalized_y: number;
  z: number;
} {
  return {
    normalized_x: clampUnit(relative.x / QUEST_WRIST_RANGE_M),
    normalized_y: clampUnit(relative.y / QUEST_WRIST_RANGE_M),
    z: clampUnit(-relative.z / QUEST_DEPTH_RANGE_M),
  };
}

/** Weight of a runtime-inferred (`emulatedPosition`) joint in the Quest quality signal. */
export const EMULATED_JOINT_WEIGHT = 0.5;

/**
 * B11 for Quest: pose completeness with inferred joints down-weighted. Quest
 * runtimes fill in occluded joints, so plain completeness reads about 1 even
 * for a badly occluded hand. A missing or inferred wrist caps the value below
 * the gate, because the wrist is what the retarget reads.
 */
export function questTrackingConfidence(counts: {
  tracked: number;
  emulated: number;
  total: number;
  wrist: "tracked" | "emulated" | "missing";
}): number {
  if (counts.total <= 0) return 0;
  const score = clamp01((counts.tracked + EMULATED_JOINT_WEIGHT * counts.emulated) / counts.total);
  return counts.wrist === "tracked" ? score : Math.min(score, KEY_POINT_MISSING_CAP);
}

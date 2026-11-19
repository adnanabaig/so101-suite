// Field rules for the leader-arm loops on Teleop & Record (F21): the loop FPS
// and the dataset repo_id, checked the way the server checks them so a start
// is never sent with a value it must refuse (an empty FPS used to send 0).
import { MAX_POLICY_FPS, MIN_POLICY_FPS } from "./policyRun";

/** The robot manager refuses other rates ("fps must be between 1 and 240"). */
export const MIN_LOOP_FPS = MIN_POLICY_FPS;
export const MAX_LOOP_FPS = MAX_POLICY_FPS;

/** StartRecordingRequest.repo_id in services/schemas.py. */
export const MAX_REPO_ID_LENGTH = 200;
export const REPO_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*(?:\/[A-Za-z0-9][A-Za-z0-9._-]*)*$/;

export type LoopFpsResult = { ok: true; fps: number } | { ok: false; error: string };

/** A loop rate is required here: a whole number from 1 to 240. */
export function parseLoopFps(text: string): LoopFpsResult {
  const trimmed = text.trim();
  const fps = Number(trimmed);
  if (trimmed === "" || !Number.isInteger(fps) || fps < MIN_LOOP_FPS || fps > MAX_LOOP_FPS) {
    return {
      ok: false,
      error: `FPS must be a whole number from ${MIN_LOOP_FPS} to ${MAX_LOOP_FPS}.`,
    };
  }
  return { ok: true, fps };
}

export type RepoIdResult =
  | { ok: true; repoId: string }
  | { ok: false; blank: boolean; error: string };

/** Surrounding whitespace is dropped; everything else must match the server. */
export function parseRepoId(text: string): RepoIdResult {
  const repoId = text.trim();
  if (repoId === "") return { ok: false, blank: true, error: "Enter a dataset repo_id." };
  if (repoId.length > MAX_REPO_ID_LENGTH) {
    return {
      ok: false,
      blank: false,
      error: `The repo_id must be at most ${MAX_REPO_ID_LENGTH} characters (it has ${repoId.length}).`,
    };
  }
  if (!REPO_ID_PATTERN.test(repoId)) {
    return {
      ok: false,
      blank: false,
      error:
        "Use letters, digits, dots, dashes and underscores, with a single slash between parts, for example user/so101_pick_place.",
    };
  }
  return { ok: true, repoId };
}

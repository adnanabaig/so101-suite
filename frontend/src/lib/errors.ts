import { ApiError } from "../api";

/** The C1 Origin guard's 403 detail. */
export const CROSS_ORIGIN_DETAIL = "cross-origin request rejected";

export function isCrossOriginRejection(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    error.status === 403 &&
    error.message.toLowerCase().includes(CROSS_ORIGIN_DETAIL)
  );
}

/**
 * C2: the fault reset's 409 when the server's torque-off retry still cannot
 * confirm the motors are off. Its detail names the acknowledgement field; any
 * other 409 (a loop running, the E-stop latched) has nothing to acknowledge.
 */
export function needsPowerCutAcknowledgement(error: unknown): boolean {
  return error instanceof ApiError && error.status === 409 && error.message.includes("power_cut_acknowledged");
}

/** What to do about a C1 Origin rejection; appended after the server's detail. */
export const CROSS_ORIGIN_HINT =
  "The robot server does not accept changes from this page's origin. Open the operator UI from the " +
  "server's own address, or add this page's origin (scheme, host and port) to SO101_CORS_ORIGINS on the " +
  "server and restart it. E-stop, Halt and Soft stop still work from any page.";

/**
 * Operator-facing text for a failed request. Server details are shown exactly
 * as sent; the C1 rejection also gets a hint, because "cross-origin request
 * rejected" alone does not tell anyone what to do.
 */
export function errorMessage(error: unknown): string {
  if (isCrossOriginRejection(error)) {
    const detail = (error as ApiError).message.replace(/\.$/, "");
    return `${detail}. ${CROSS_ORIGIN_HINT}`;
  }
  if (error instanceof Error) return error.message;
  return String(error);
}

/**
 * `first` followed by the sentence `next`. Server details rarely end in a full
 * stop ("E-stop is engaged; reset it before agent motion"), so one is added
 * when missing; the detail itself is never otherwise changed.
 */
export function joinSentences(first: string, next: string): string {
  const text = first.trimEnd();
  if (!text) return next;
  return /[.!?…]["'”’)\]]*$/.test(text) ? `${text} ${next}` : `${text}. ${next}`;
}

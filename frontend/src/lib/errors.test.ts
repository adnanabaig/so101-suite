import { describe, expect, test } from "vitest";
import { ApiError } from "../api";
import {
  CROSS_ORIGIN_DETAIL,
  CROSS_ORIGIN_HINT,
  errorMessage,
  isCrossOriginRejection,
  joinSentences,
  needsPowerCutAcknowledgement,
} from "./errors";
import { isEditableTarget } from "./keyboard";

describe("C1 cross-origin rejection", () => {
  test("the Origin guard's 403 detail is shown as sent, followed by a hint", () => {
    const rejection = new ApiError(403, CROSS_ORIGIN_DETAIL, JSON.stringify({ detail: CROSS_ORIGIN_DETAIL }));
    expect(isCrossOriginRejection(rejection)).toBe(true);
    const text = errorMessage(rejection);
    expect(text.startsWith("cross-origin request rejected. ")).toBe(true);
    expect(text).toBe(`cross-origin request rejected. ${CROSS_ORIGIN_HINT}`);
    expect(text).toMatch(/server's own address/);
    expect(text).toMatch(/SO101_CORS_ORIGINS/);
    // The guard exempts exactly the E-stop, Halt and legacy stop routes.
    expect(text).toMatch(/E-stop, Halt and Soft stop still work from any page/);
  });

  test("the Host guard's 400 detail passes through verbatim", () => {
    const detail = "untrusted Host header; add the hostname to SO101_ALLOWED_HOSTS";
    expect(isCrossOriginRejection(new ApiError(400, detail, ""))).toBe(false);
    expect(errorMessage(new ApiError(400, detail, ""))).toBe(detail);
  });

  test("other 403s and errors pass through unchanged", () => {
    expect(isCrossOriginRejection(new ApiError(403, "lease owned by another operator", ""))).toBe(false);
    expect(errorMessage(new ApiError(409, "lease expired", ""))).toBe("lease expired");
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage("plain")).toBe("plain");
  });
});

describe("isEditableTarget", () => {
  test("text inputs, textareas, selects and contenteditable are editable", () => {
    const input = document.createElement("input");
    const textarea = document.createElement("textarea");
    const select = document.createElement("select");
    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");
    const child = document.createElement("span");
    editable.appendChild(child);
    for (const element of [input, textarea, select, editable, child]) {
      expect(isEditableTarget(element)).toBe(true);
    }
  });

  test("buttons, checkboxes, ranges and plain elements are not", () => {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    const range = document.createElement("input");
    range.type = "range";
    const off = document.createElement("div");
    off.setAttribute("contenteditable", "false");
    for (const element of [document.createElement("button"), checkbox, range, off, document.body]) {
      expect(isEditableTarget(element)).toBe(false);
    }
    expect(isEditableTarget(null)).toBe(false);
  });
});

describe("joinSentences", () => {
  test("adds a full stop after a server detail that lacks one, and nothing else", () => {
    expect(joinSentences("E-stop is engaged; reset it before agent motion", "Use Halt.")).toBe(
      "E-stop is engaged; reset it before agent motion. Use Halt.",
    );
    expect(joinSentences("lease expired.", "Use Halt.")).toBe("lease expired. Use Halt.");
    expect(joinSentences("is it moving?", "Use Halt.")).toBe("is it moving? Use Halt.");
    expect(joinSentences('refused ("busy.")', "Use Halt.")).toBe('refused ("busy.") Use Halt.');
    expect(joinSentences("trailing space  ", "Use Halt.")).toBe("trailing space. Use Halt.");
    expect(joinSentences("", "Use Halt.")).toBe("Use Halt.");
  });

  test("the cross-origin message already ends a sentence", () => {
    const rejection = new ApiError(403, CROSS_ORIGIN_DETAIL, "");
    expect(joinSentences(errorMessage(rejection), "Next.")).toBe(`${errorMessage(rejection)} Next.`);
  });
});

describe("C2 fault reset refusals", () => {
  test("only the 409 that names power_cut_acknowledged asks for the power-cut acknowledgement", () => {
    const retryFailed =
      "torque-off is still unconfirmed after a retry; cut physical power to the arm, then reset the fault with power_cut_acknowledged: true";
    expect(needsPowerCutAcknowledgement(new ApiError(409, retryFailed, ""))).toBe(true);
    expect(needsPowerCutAcknowledgement(new ApiError(409, "cannot clear a fault while a control loop is active", ""))).toBe(false);
    // The bus did not come free: still latched, but acknowledging again changes nothing.
    expect(needsPowerCutAcknowledgement(new ApiError(503, retryFailed, ""))).toBe(false);
    expect(needsPowerCutAcknowledgement(new Error(retryFailed))).toBe(false);
  });
});

import { describe, expect, test } from "vitest";
import { MAX_REPO_ID_LENGTH, parseLoopFps, parseRepoId } from "./recording";

describe("parseLoopFps (F21)", () => {
  test("accepts whole numbers from 1 to 240, surrounding spaces allowed", () => {
    expect(parseLoopFps("1")).toEqual({ ok: true, fps: 1 });
    expect(parseLoopFps(" 30 ")).toEqual({ ok: true, fps: 30 });
    expect(parseLoopFps("240")).toEqual({ ok: true, fps: 240 });
  });

  test.each(["", "   ", "0", "-1", "241", "2.5", "abc", "NaN", "Infinity"])("refuses %j", (text) => {
    expect(parseLoopFps(text)).toEqual({ ok: false, error: "FPS must be a whole number from 1 to 240." });
  });
});

describe("parseRepoId (F21)", () => {
  test.each(["local", "user/so101_pick-place.v2", "a/b/c", "A1", "x".repeat(MAX_REPO_ID_LENGTH)])(
    "accepts %j",
    (text) => {
      expect(parseRepoId(text)).toEqual({ ok: true, repoId: text });
    },
  );

  test("trims surrounding whitespace", () => {
    expect(parseRepoId("  user/demo\n")).toEqual({ ok: true, repoId: "user/demo" });
  });

  test("blank is its own case so the page can avoid a premature inline error", () => {
    expect(parseRepoId("  ")).toEqual({ ok: false, blank: true, error: "Enter a dataset repo_id." });
  });

  test.each(["-user/demo", ".hidden", "user/", "/demo", "user//demo", "user/_x", "a b", "ü/demo", "user/demo?"])(
    "refuses %j as the server's pattern would",
    (text) => {
      const result = parseRepoId(text);
      expect(result.ok).toBe(false);
      expect(result).toMatchObject({ blank: false, error: expect.stringMatching(/^Use letters, digits/) });
    },
  );

  test("over 200 characters says how long it is", () => {
    expect(parseRepoId("x".repeat(MAX_REPO_ID_LENGTH + 1))).toEqual({
      ok: false,
      blank: false,
      error: "The repo_id must be at most 200 characters (it has 201).",
    });
  });
});

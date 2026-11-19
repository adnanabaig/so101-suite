import { describe, expect, test } from "vitest";
import { BACKEND_BADGE, backendDescription, backendInfo } from "./backend";

describe("backendInfo", () => {
  test("reads backend when the server sends it, whatever mock says", () => {
    expect(backendInfo({ backend: "real", mock: false })).toEqual({ kind: "real", reported: true });
    // Under sim the server keeps mock=false; only backend tells it apart.
    expect(backendInfo({ backend: "sim", mock: false })).toEqual({ kind: "sim", reported: true });
    expect(backendInfo({ backend: "mock", mock: true })).toEqual({ kind: "mock", reported: true });
  });

  test("an older server without backend falls back to mock: mock, else a real arm", () => {
    expect(backendInfo({ mock: true })).toEqual({ kind: "mock", reported: false });
    expect(backendInfo({ mock: false })).toEqual({ kind: "real", reported: false });
    // An unknown value is not trusted either way.
    expect(backendInfo({ backend: "replay", mock: false })).toEqual({ kind: "real", reported: false });
  });

  test("nothing to read: unknown", () => {
    expect(backendInfo(null)).toBeNull();
    expect(backendInfo(undefined)).toBeNull();
    expect(backendInfo({})).toBeNull();
  });

  test("badge text and descriptions never call a simulated arm real", () => {
    expect(BACKEND_BADGE).toEqual({ real: "REAL", sim: "SIM", mock: "MOCK" });
    expect(backendDescription({ kind: "sim", reported: true })).toMatch(/simulated arm, not the physical one/);
    expect(backendDescription({ kind: "mock", reported: true })).toBe("Mock backend: no motors move");
    expect(backendDescription({ kind: "real", reported: true })).toBe("Real arm: commands move the physical motors");
    expect(backendDescription({ kind: "real", reported: false })).toMatch(/does not say whether it is simulated/);
  });
});

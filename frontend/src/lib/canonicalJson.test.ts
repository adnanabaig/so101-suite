import { describe, expect, test } from "vitest";
import { canonicalJson, checkTargetDigest, pythonFloatRepr, sha256Hex, targetDigest } from "./canonicalJson";

// Reference values were produced by CPython 3 with
//   json.dumps(targets, sort_keys=True, separators=(",", ":"))
// and hashlib.sha256(...).hexdigest().
const TARGETS = { wrist_roll: 35, gripper: 40, shoulder_pan: -12.5, elbow_flex: 0.1 };
const PYTHON_TEXT = '{"elbow_flex":0.1,"gripper":40.0,"shoulder_pan":-12.5,"wrist_roll":35.0}';
const PYTHON_DIGEST = "aff5a28fd0a2b3775c2284933bf45928d164a5d17dfacbb7e07b8deb78164ffa";
const INT_DIGEST = "f729580f401246351d7fa5e75083f68681e5a06b926f5caa9a6f30d41ed249e0"; // {"gripper":40,"wrist_roll":35}

// gate's bounded skills (routers/agent.py _targets_for_skill) and their digests
// from agent_executor.target_digest:
//   hashlib.sha256(json.dumps({j: float(v)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
const GATE_SKILL_DIGESTS: [string, Record<string, number>, string][] = [
  ["open_gripper", { gripper: 80 }, "45f46d2cbe0dd6d435994148e5949187802c7a9f305dae48d3a1eb4813e26821"],
  ["close_gripper", { gripper: 15 }, "fb6bd7c0e3fc8b8bfc24883f0a910872c3fadc3d6c8ef1f9a7e86ee441b9ff64"],
  ["wrist_wave", { wrist_roll: 30 }, "d16f8054143c88fceb4a5aa1bc3b4d7010df2d6a1d50ddf978006ea18bb05487"],
  [
    "home",
    { shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, wrist_flex: 0, wrist_roll: 0, gripper: 50 },
    "64c3e84ceb089003adb6a7b59fc86c609f4ca9cdde0199d171a59ceb472707e8",
  ],
  ["mixed", { wrist_roll: -0.1, gripper: 12.5 }, "105928c4f50c864ca1e9a69d38b40efae7c0a02d0680a2dd4bd0c8e90293ac37"],
];

describe("pythonFloatRepr", () => {
  test("matches Python repr(float) across the formatting boundaries", () => {
    const cases: [number, string][] = [
      [40, "40.0"],
      [0.1, "0.1"],
      [-12.5, "-12.5"],
      [1e-5, "1e-05"],
      [0.0001, "0.0001"],
      [1e16, "1e+16"],
      [1.5e16, "1.5e+16"],
      [123456789012345.6, "123456789012345.6"],
      [1e15, "1000000000000000.0"],
      [-0, "-0.0"],
      [0, "0.0"],
      [2.5e-7, "2.5e-07"],
      [1e100, "1e+100"],
      [100 / 3, "33.333333333333336"],
      [5e-324, "5e-324"],
      [1.7976931348623157e308, "1.7976931348623157e+308"],
      [100, "100.0"],
      [0.5, "0.5"],
    ];
    for (const [value, expected] of cases) expect(pythonFloatRepr(value)).toBe(expected);
  });

  test("non-finite values use Python's spelling", () => {
    expect(pythonFloatRepr(Number.NaN)).toBe("NaN");
    expect(pythonFloatRepr(Infinity)).toBe("Infinity");
    expect(pythonFloatRepr(-Infinity)).toBe("-Infinity");
  });
});

describe("canonicalJson", () => {
  test("sorts keys, drops whitespace and writes floats like Python", () => {
    expect(canonicalJson(TARGETS)).toBe(PYTHON_TEXT);
  });

  test("nested values and strings follow json.dumps(ensure_ascii=True)", () => {
    expect(canonicalJson({ "\u00e9": 1, "a\u2028": "x\u00e9" })).toBe('{"a\\u2028":"x\\u00e9","\\u00e9":1.0}');
    expect(canonicalJson({ b: [1, true, null], a: { d: "q\"t", c: undefined } })).toBe(
      '{"a":{"d":"q\\"t"},"b":[1.0,true,null]}',
    );
  });

  test("integers are written as floats, so 80 and 80.0 are the same targets", () => {
    expect(canonicalJson({ wrist_roll: 35, gripper: 40 })).toBe('{"gripper":40.0,"wrist_roll":35.0}');
  });
});

describe("target digests (C8)", () => {
  test("sha256Hex is lowercase hex", () => {
    expect(sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  test("the digest of the targets equals the server's Python digest", () => {
    expect(targetDigest(TARGETS)).toBe(PYTHON_DIGEST);
  });

  test.each(GATE_SKILL_DIGESTS)("gate's %s targets hash to gate's digest", (_skill, targets, digest) => {
    expect(targetDigest(targets)).toBe(digest);
    expect(checkTargetDigest(targets, digest)).toEqual({ status: "verified", digest });
  });

  test("a matching server digest verifies; case is ignored", () => {
    expect(checkTargetDigest(TARGETS, PYTHON_DIGEST.toUpperCase())).toEqual({
      status: "verified",
      digest: PYTHON_DIGEST,
    });
  });

  test("a digest of integer-rendered targets is a mismatch: gate hashes floats only", () => {
    expect(checkTargetDigest({ gripper: 40, wrist_roll: 35 }, INT_DIGEST)).toEqual({
      status: "mismatch",
      local: targetDigest({ gripper: 40, wrist_roll: 35 }),
      server: INT_DIGEST,
    });
  });

  test("no server digest falls back to the local one", () => {
    expect(checkTargetDigest(TARGETS, undefined)).toEqual({ status: "local-only", digest: PYTHON_DIGEST });
  });

  test("different targets are a mismatch", () => {
    const check = checkTargetDigest({ ...TARGETS, gripper: 41 }, PYTHON_DIGEST);
    expect(check.status).toBe("mismatch");
  });
});

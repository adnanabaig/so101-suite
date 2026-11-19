import { createHash } from "node:crypto";
import { describe, expect, test } from "vitest";
import { sha256Hex } from "./sha256";

describe("sha256Hex", () => {
  test("matches the FIPS 180-4 test vectors", () => {
    expect(sha256Hex("")).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect(sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")).toBe(
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
    expect(
      sha256Hex(
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
      ),
    ).toBe("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
  });

  test("agrees with Node's hash at every padding boundary and for multi-byte UTF-8", () => {
    const samples: string[] = [];
    for (let length = 0; length <= 130; length += 1) samples.push("x".repeat(length));
    samples.push("é".repeat(40), "  joint ✓", '{"gripper":40.0,"wrist_roll":35.0}', "a".repeat(10_000));
    for (const text of samples) {
      expect(sha256Hex(text)).toBe(createHash("sha256").update(text, "utf8").digest("hex"));
    }
  });
});

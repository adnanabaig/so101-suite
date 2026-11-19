// Canonical JSON and SHA-256 for C8 target digests.
//
// The server (agent_executor.target_digest) computes `target_digest` as
//   sha256(json.dumps({joint: float(value)}, sort_keys=True, separators=(",", ":")))
// so numbers must be written the way Python writes a float (`40.0`, `1e-05`,
// `1e+16`), not the way JSON.stringify writes them (`40`, `0.00001`,
// `10000000000000000`). `80` and `80.0` therefore hash the same. Strings are
// escaped as Python's default `ensure_ascii=True` does.

import { sha256Hex } from "./sha256";

export { sha256Hex };

function pythonExponent(exponent: number): string {
  const sign = exponent < 0 ? "-" : "+";
  return `e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`;
}

/** Python `repr(float(value))`, e.g. 40 -> "40.0", 0.00001 -> "1e-05". */
export function pythonFloatRepr(value: number): string {
  if (Number.isNaN(value)) return "NaN";
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";

  const sign = value < 0 ? "-" : "";
  // toExponential() without an argument gives the shortest round-trip digits,
  // which is also what Python's repr chooses.
  const [mantissa, exponentText] = Math.abs(value).toExponential().split("e");
  const digits = mantissa.replace(".", "");
  const exponent = Number(exponentText);

  if (exponent < -4 || exponent >= 16) {
    const scientific = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${scientific}${pythonExponent(exponent)}`;
  }
  if (exponent < 0) {
    return `${sign}0.${"0".repeat(-exponent - 1)}${digits}`;
  }
  const integerDigits = exponent + 1;
  if (digits.length <= integerDigits) {
    return `${sign}${digits}${"0".repeat(integerDigits - digits.length)}.0`;
  }
  return `${sign}${digits.slice(0, integerDigits)}.${digits.slice(integerDigits)}`;
}

/** JSON string literal with non-ASCII escaped, as Python's json.dumps does. */
function asciiString(value: string): string {
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, (char) =>
    `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

/** Sorted keys, "," and ":" separators, no whitespace, numbers as Python floats. */
export function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return pythonFloatRepr(value);
  if (typeof value === "string") return asciiString(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      // Python sorts str keys by code point; plain comparison matches for BMP keys.
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, entry]) => `${asciiString(key)}:${canonicalJson(entry)}`);
    return `{${entries.join(",")}}`;
  }
  throw new TypeError(`cannot canonicalise a ${typeof value}`);
}

/** C8: the digest the operator approves, computed from the targets on screen. */
export function targetDigest(targets: Record<string, number>): string {
  return sha256Hex(canonicalJson(targets));
}

export type DigestCheck =
  | { status: "verified"; digest: string }
  | { status: "local-only"; digest: string }
  | { status: "mismatch"; local: string; server: string };

/**
 * Compare the server's `target_digest` with one computed here from the same
 * targets. "verified" when they agree, "local-only" for servers that do not
 * send a digest, and "mismatch" otherwise: the targets shown are not the ones
 * the server would execute, so approval must be refused. There is one
 * rendering only, the server's: a digest of integer-rendered targets would be
 * refused by the server's approve check anyway.
 */
export function checkTargetDigest(
  targets: Record<string, number>,
  serverDigest: string | null | undefined,
): DigestCheck {
  const local = targetDigest(targets);
  if (!serverDigest) return { status: "local-only", digest: local };
  const server = serverDigest.toLowerCase();
  if (server === local) return { status: "verified", digest: local };
  return { status: "mismatch", local, server };
}

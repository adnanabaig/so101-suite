import { existsSync, lstatSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

function replaceAll(body: Buffer, needle: Buffer, replacement: Buffer): { body: Buffer; found: boolean } {
  let start = 0;
  let index = body.indexOf(needle, start);
  if (index < 0) return { body, found: false };
  const clean = Buffer.from(body);
  while (index >= 0) {
    replacement.copy(clean, index);
    start = index + needle.length;
    index = clean.indexOf(needle, start);
  }
  return { body: clean, found: true };
}

/** Redact an accidental token byte-for-byte and report the affected relative paths. */
export function redactTokenArtifacts(root: string, token: string): string[] {
  if (!existsSync(root)) return [];
  const needle = Buffer.from(token);
  const replacement = Buffer.from("*".repeat(needle.length));
  const redacted: string[] = [];

  const walk = (directory: string) => {
    for (const name of readdirSync(directory)) {
      const path = join(directory, name);
      const stat = lstatSync(path);
      if (stat.isSymbolicLink()) throw new Error("Playwright artifacts contain an unexpected symlink");
      if (stat.isDirectory()) {
        walk(path);
        continue;
      }
      if (!stat.isFile()) continue;
      const result = replaceAll(readFileSync(path), needle, replacement);
      if (!result.found) continue;
      writeFileSync(path, result.body, { mode: stat.mode });
      if (readFileSync(path).includes(needle)) {
        throw new Error("an E2E token could not be removed from a Playwright artifact");
      }
      redacted.push(relative(root, path));
    }
  };
  walk(root);
  return redacted.sort();
}

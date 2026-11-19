import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";

// The target is Ubuntu on a laptop, in Chrome or Firefox. Nothing the UI shows
// may assume a Mac: modifier keys are Ctrl, Alt and Shift there.

const SOURCES = Object.entries(
  import.meta.glob<string>("../**/*.{ts,tsx}", { query: "?raw", import: "default", eager: true }),
).filter(([path]) => !/\.test\.tsx?$/.test(path) && !path.includes("/test/"));

describe("no Mac-only assumptions", () => {
  test("no Mac modifier glyphs or Cmd labels", () => {
    expect(SOURCES.length).toBeGreaterThan(50);
    const offenders = SOURCES.flatMap(([path, text]) =>
      text.split("\n").flatMap((line, index) => (/[⌘⌥⌃]|\bCmd\b/.test(line) ? [`${path}:${index + 1}`] : [])),
    );
    expect(offenders).toEqual([]);
  });

  test("no macOS device paths or folders in hints", () => {
    const offenders = SOURCES.flatMap(([path, text]) =>
      text
        .split("\n")
        .flatMap((line, index) => (/\/dev\/(?:cu|tty)\.|usbmodem|\/Users\/|~\/Library/.test(line) ? [`${path}:${index + 1}`] : [])),
    );
    expect(offenders).toEqual([]);
  });

  test("font stacks fall back to generic families a Linux desktop has", () => {
    // Read from disk: Vitest replaces CSS, even ?raw, with "".
    const css = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");
    expect(css).toMatch(/--font-mono:[^;]*\bmonospace;/);
    expect(css).toMatch(/--font-sans:[^;]*\bsystem-ui\b[^;]*\bsans-serif\b/);
    expect(css).toMatch(/font-family:[^;]*\bsystem-ui\b[^;]*\bsans-serif;/);
  });
});

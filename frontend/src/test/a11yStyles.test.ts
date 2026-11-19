import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";

// Contrast, focus and motion rules that jsdom cannot see in a rendered page:
// checked against the stylesheet, the Tailwind theme and the class names in
// the source.

// Read from disk: Vitest replaces CSS modules, even ?raw ones, with "".
// Vitest runs from the frontend directory, where vite.config.ts lives.
const css = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");
const paletteCss = readFileSync(resolve(process.cwd(), "src/styles/palette.css"), "utf8");

/**
 * The colours Tailwind renders: v3's palette, pinned as hex in palette.css
 * (v4's own default palette is OKLCH and would render differently).
 */
const PALETTE: Record<string, Record<string, string> | undefined> = {};
for (const [, family, level, hex] of paletteCss.matchAll(/--color-([a-z]+)-(\d+):\s*(#[0-9a-f]{6});/g)) {
  (PALETTE[family] ??= {})[level] = hex;
}

/** A palette colour that must exist; fails the test rather than skipping it. */
function shade(family: string, value: string): string {
  const hex = PALETTE[family]?.[value];
  if (!hex) throw new Error(`palette.css has no --color-${family}-${value}`);
  return hex;
}

/** WCAG 2 relative luminance of a #rrggbb colour. */
function luminance(hex: string): number {
  const channels = [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16) / 255);
  const [r, g, b] = channels.map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (light + 0.05) / (dark + 0.05);
}

/** Every non-test source file, by path relative to src/. */
const SOURCES = Object.entries(
  import.meta.glob<string>("../**/*.{ts,tsx}", { query: "?raw", import: "default", eager: true }),
)
  .map(([path, text]) => [path.replace(/^\.\.\//, ""), text] as const)
  .filter(([path]) => !/\.test\.tsx?$/.test(path) && !path.startsWith("test/"));


/** [file:line, text] for every source line matching a pattern. */
function linesMatching(pattern: RegExp, sources = SOURCES): Array<[string, string]> {
  return sources.flatMap(([path, text]) =>
    text.split("\n").flatMap((line, index): Array<[string, string]> => (pattern.test(line) ? [[`${path}:${index + 1}`, line.trim()]] : [])),
  );
}

const SURFACES = ["800", "900", "950"].map((value) => shade("slate", value));

describe("palette", () => {
  test("is Tailwind v3's: 22 families of 11 shades, in hex", () => {
    const families = Object.keys(PALETTE);
    expect(families).toHaveLength(22);
    for (const family of families) expect(Object.keys(PALETTE[family] ?? {})).toHaveLength(11);
    // Spot checks against tailwindcss@3.4.19 colors.js (v4's OKLCH values differ).
    expect([shade("slate", "950"), shade("cyan", "300"), shade("red", "600"), shade("rose", "600")]).toEqual([
      "#020617",
      "#67e8f9",
      "#dc2626",
      "#e11d48",
    ]);
    expect(paletteCss).toMatch(/--color-white:\s*#fff;/);
    expect(paletteCss).toMatch(/--color-black:\s*#000;/);
  });

  test("every palette colour a class names is pinned, so none falls back to v4's OKLCH defaults", () => {
    const named = new Set<string>();
    for (const [, text] of SOURCES) {
      for (const [, family, level] of text.matchAll(
        /(?<![\w-])(?:[a-z-]+:)*(?:bg|text|border(?:-[trblxy])?|ring|ring-offset|shadow|from|via|to|divide|outline|fill|stroke|accent|caret|decoration|placeholder)-([a-z]+)-(\d{2,3})(?![\w-])/g,
      )) {
        if (family !== "muted") named.add(`${family}-${level}`);
      }
    }
    expect(named.size).toBeGreaterThan(50);
    const missing = [...named].filter((name) => {
      const [family, level] = name.split("-");
      return !PALETTE[family]?.[level];
    });
    expect(missing).toEqual([]);
  });
});

describe("contrast", () => {
  const muted = /--color-muted:\s*(#[0-9a-fA-F]{6});/.exec(css)?.[1];

  test("the muted text token reads at 4.5:1 or better on every slate surface; slate-500 does not", () => {
    expect(muted).toBeDefined();
    for (const surface of SURFACES) {
      expect(contrast(muted ?? "#000000", surface)).toBeGreaterThanOrEqual(4.5);
    }
    // Why the token exists.
    expect(contrast(shade("slate", "500"), shade("slate", "800"))).toBeLessThan(4.5);
    expect(contrast(shade("slate", "500"), shade("slate", "950"))).toBeLessThan(4.5);
  });

  test("no readable text uses slate-500 or slate-600; text-muted replaces them", () => {
    const offenders = linesMatching(
      /\btext-slate-(500|600)\b/,
      SOURCES,
    );
    expect(offenders).toEqual([]);
    expect(linesMatching(/\btext-muted\b/).length).toBeGreaterThan(50);
  });

  test("white text keeps 4.5:1 on its background, hover and press included", () => {
    const weak = linesMatching(/\btext-white\b/, SOURCES).flatMap(
      ([where, line]) =>
        [...line.matchAll(/(?<![\w-])(?:(?:hover|active):)?bg-([a-z]+)-(\d{2,3})(?![\w/])/g)].flatMap((match) => {
          const background = PALETTE[match[1]]?.[match[2]];
          if (!background) return [];
          const ratio = contrast("#ffffff", background);
          return ratio < 4.5 ? [`${where} ${match[0]} ${ratio.toFixed(2)}:1`] : [];
        }),
    );
    expect(weak).toEqual([]);
  });
});

describe("focus", () => {
  test("keyboard focus is always visible: a global :focus-visible outline", () => {
    expect(css).toMatch(/(^|\n)\s*:focus-visible\s*\{[^}]*outline:\s*2px solid[^}]*\}/);
  });

  test("every control that removes the outline draws its own focus style", () => {
    const sources = SOURCES;
    const missing = sources.flatMap(([path, text]) => {
      const lines = text.split("\n");
      return lines.flatMap((line, index) => {
        if (!/\boutline-(?:none|hidden)\b/.test(line)) return [];
        const nearby = lines.slice(Math.max(0, index - 8), index + 3).join("\n");
        // Its own visible focus style, on this class string or the next line.
        if (/focus-visible:(ring|border)/.test(line + lines[index + 1])) return [];
        // A programmatic focus target (a page, a dialog, a heading) is not a control.
        if (/\bfocus:outline-(?:none|hidden)\b/.test(line) && /tabIndex=\{[^}]*-1/.test(nearby)) return [];
        return [`${path}:${index + 1}`];
      });
    });
    expect(missing).toEqual([]);
  });
});

describe("motion", () => {
  test("reduced motion stops animation and transitions, except essential motion", () => {
    const block = /@media \(prefers-reduced-motion: reduce\)\s*\{([\s\S]*?)\n\}/.exec(css)?.[1] ?? "";
    expect(block).toContain("*:not([data-essential-motion])");
    expect(block).toMatch(/animation-duration:\s*0\.01ms !important/);
    expect(block).toMatch(/animation-iteration-count:\s*1 !important/);
    expect(block).toMatch(/transition-duration:\s*0\.01ms !important/);
  });

  test("decorative animation runs only when motion is welcome (motion-safe:)", () => {
    const unguarded = linesMatching(
      /(?<!motion-safe:)\banimate-(spin|ping|pulse|bounce)\b/,
      SOURCES,
    );
    expect(unguarded).toEqual([]);
  });

  test("only the hold-to-confirm fill claims essential motion", () => {
    expect(linesMatching(/data-essential-motion=/).map(([where]) => where.split(":")[0])).toEqual([
      "components/safety/HoldToConfirmButton.tsx",
    ]);
  });
});

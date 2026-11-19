import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import ts from "typescript";
import { describe, expect, test } from "vitest";

// Guards for the Tailwind 3 -> 4 migration: the v3 defaults index.css keeps,
// and the one layout rule v4 changed that source code can trip over.

const css = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8");

const SOURCES = Object.entries(
  import.meta.glob<string>("../**/*.tsx", { query: "?raw", import: "default", eager: true }),
).filter(([path]) => !/\.test\.tsx$/.test(path));

describe("v3 defaults kept in index.css", () => {
  test("the sans stack is v3's, with system-ui (v4.3's default names fonts instead)", () => {
    expect(css).toMatch(/--font-sans:\s*ui-sans-serif, system-ui, sans-serif,/);
  });

  test("border, placeholder, cursor, outline and form-control defaults", () => {
    const base = css.slice(css.indexOf("@layer base"));
    expect(base).toMatch(/border-color: var\(--color-gray-200, currentcolor\)/);
    expect(base).toMatch(/textarea::placeholder\s*\{\s*color: var\(--color-gray-400\)/);
    expect(base).toMatch(/\[role="button"\]:not\(:disabled\)\s*\{\s*cursor: pointer/);
    expect(base).toMatch(/outline-color: var\(--color-cyan-300\)/);
    expect(base).toMatch(/optgroup\s*\{\s*background-color: revert;\s*border-radius: revert;/);
    expect(base).toMatch(/option\s*\{\s*padding: revert;/);
  });
});

/** Static class text of a JSX element's className, including template pieces. */
function classText(element: ts.JsxElement | ts.JsxSelfClosingElement): string {
  const opening = ts.isJsxElement(element) ? element.openingElement : element;
  const attribute = opening.attributes.properties.find(
    (property): property is ts.JsxAttribute => ts.isJsxAttribute(property) && property.name.getText() === "className",
  );
  if (!attribute?.initializer) return "";
  const parts: string[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isStringLiteralLike(node) || ts.isTemplateLiteralToken(node)) parts.push(node.text);
    ts.forEachChild(node, visit);
  };
  visit(attribute.initializer);
  return parts.join(" ");
}

const SPACED = /(?:^|\s)(?:[a-z0-9-]+:)*(?:space-[xy]-[\d.]+|divide-[xy])(?=\s|$)/;
const OWN_MARGIN = /(?:^|\s)(?:[a-z0-9-]+:)*-?m[ytbxlr]?-[\w.[\]/-]+/;
const OUT_OF_FLOW = /(?:^|\s)(?:[a-z0-9-]+:)*(?:absolute|fixed|sr-only|hidden|contents)(?=\s|$)/;

describe("space-y and divide-y children", () => {
  // v4 spaces every child but the last (v3: every child but the first), at
  // zero specificity. A child's own margin then wins, and an out-of-flow last
  // child leaves the gap under the last visible one.
  test("no child has its own margin, and the last child is in flow", () => {
    const problems: string[] = [];
    let containers = 0;
    for (const [path, text] of SOURCES) {
      const file = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
      const visit = (node: ts.Node) => {
        if (ts.isJsxElement(node) && SPACED.test(classText(node))) {
          containers += 1;
          const children = node.children.filter(
            (child): child is ts.JsxElement | ts.JsxSelfClosingElement =>
              ts.isJsxElement(child) || ts.isJsxSelfClosingElement(child),
          );
          const line = file.getLineAndCharacterOfPosition(node.getStart()).line + 1;
          children.forEach((child, index) => {
            const classes = classText(child);
            if (OWN_MARGIN.test(classes)) problems.push(`${path}:${line} child ${index} has a margin class`);
            // Last in the DOM: nothing but whitespace follows it (an {expression} might).
            const isLast = node.children.indexOf(child) === node.children.length - 1 - trailingText(node);
            if (isLast && OUT_OF_FLOW.test(classes)) problems.push(`${path}:${line} last child is out of flow`);
          });
        }
        ts.forEachChild(node, visit);
      };
      visit(file);
    }
    expect(containers).toBeGreaterThan(40);
    expect(problems).toEqual([]);
  });
});

/** Whitespace-only JSX text after the last real child. */
function trailingText(node: ts.JsxElement): number {
  let count = 0;
  for (let index = node.children.length - 1; index >= 0; index -= 1) {
    const child = node.children[index];
    if (ts.isJsxText(child) && child.getText().trim() === "") count += 1;
    else break;
  }
  return count;
}

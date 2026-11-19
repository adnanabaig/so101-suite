import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  buildStamp,
  distManifest,
  UI_SOURCE_DIRS,
  UI_SOURCE_FILES,
  UI_STAMP,
  uiSourceFiles,
  uiSourcesDigest,
} from "./build-stamp";

// The same tree and known answers as tests/test_hatch_build.py, so the
// stamp this plugin writes is the one hatch_build.py and check_dist.py accept.
const FIXTURE: Record<string, string> = {
  "index.html": "<!doctype html><title>SO-101</title>\n",
  "package.json": '{"name": "so101-frontend"}\n',
  "src/main.tsx": "export {};\n",
  "src/lib/a.ts": "export const a = 1;\n",
  "src/.DS_Store": "junk",
  "public/manifest.webmanifest": "{}\n",
  "README.md": "not an input\n",
  "node_modules/x/index.js": "not an input\n",
  "dist/index.html": "<html></html>\n",
  "dist/assets/index-abc123.js": "console.log(1);\n",
  "dist/so101-ui-stamp.json": "{}\n",
};
const SOURCES_DIGEST = "c778c8d01b2435e5ac63ba58349054c906b9aac296f8be8f4c7fb5513763a2d1";
const MANIFEST = {
  "assets/index-abc123.js": "b603d946eb2b396ca4ecf65c223daff659dbe6f1cfeac235b7c61d3ba6964cae",
  "index.html": "b0693dc92f76e08bf1485b3dd9b514a2e31dfd6f39422a6b60edb722671dc98f",
};

const roots: string[] = [];

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), "so101-stamp-"));
  roots.push(root);
  for (const [path, text] of Object.entries(FIXTURE)) {
    mkdirSync(dirname(join(root, path)), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("the UI build stamp", () => {
  it("hashes exactly the build inputs, as hatch_build.py does", () => {
    const root = fixture();
    expect(uiSourceFiles(root)).toEqual([
      "index.html",
      "package.json",
      "public/manifest.webmanifest",
      "src/lib/a.ts",
      "src/main.tsx",
    ]);
    expect(uiSourcesDigest(root)).toBe(SOURCES_DIGEST);
  });

  it("lists every built file but the stamp in the manifest", () => {
    expect(distManifest(join(fixture(), "dist"))).toEqual(MANIFEST);
  });

  it("changes the digest when a source changes", () => {
    const root = fixture();
    writeFileSync(join(root, "src/main.tsx"), "export const changed = true;\n");
    expect(uiSourcesDigest(root)).not.toBe(SOURCES_DIGEST);
  });

  it("builds the stamp hatch_build.py reads", () => {
    const root = fixture();
    const stamp = buildStamp(root, join(root, "dist"));
    expect(stamp).toMatchObject({ format: 1, sources: SOURCES_DIGEST, files: MANIFEST });
    expect(stamp.git_head === null || /^[0-9a-f]{40}$/.test(stamp.git_head)).toBe(true);
  });

  it("keeps its constants in step with hatch_build.py", () => {
    const hook = readFileSync(join(__dirname, "..", "hatch_build.py"), "utf8");
    expect(hook).toContain(`UI_STAMP = "${UI_STAMP}"`);
    expect(hook).toContain(`UI_SOURCE_DIRS = (${UI_SOURCE_DIRS.map((d) => `"${d}"`).join(", ")})`);
    for (const name of UI_SOURCE_FILES) expect(hook).toContain(`    "${name}",\n`);
  });
});

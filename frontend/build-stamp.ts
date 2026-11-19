/**
 * The UI build stamp: dist/so101-ui-stamp.json, written at the end of every
 * production build (`npm run build`, the so101-ui-stamp plugin in vite.config.ts).
 *
 *     {"format": 1, "sources": DIGEST, "git_head": SHA or null,
 *      "files": {"index.html": SHA256, "assets/...": SHA256, ...}}
 *
 * `files` is the build's manifest: every file in dist except the stamp.
 * `sources` is the digest of the build's inputs: every file under
 * UI_SOURCE_DIRS plus UI_SOURCE_FILES, except .DS_Store, as
 * SHA-256 over "<path>\0<sha256 of the file>\n" sorted by path (POSIX,
 * relative to frontend/). `git_head` is the checkout's commit, null without git.
 *
 * hatch_build.py refuses to package a dist whose stamp is missing, whose
 * manifest does not match its files, or whose digest does not match the
 * sources; scripts/check_dist.py and scripts/make_handoff.py check the built
 * packages the same way. Keep the constants and the digest in step with
 * hatch_build.py: build-stamp.test.ts and tests/test_hatch_build.py pin the
 * same known answer.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Plugin } from "vite";

export const UI_STAMP = "so101-ui-stamp.json";
export const UI_STAMP_FORMAT = 1;
export const UI_SOURCE_DIRS = ["src", "public"];
export const UI_SOURCE_FILES = [
  "index.html",
  "package.json",
  "package-lock.json",
  "vite.config.ts",
  "build-stamp.ts",
  "tsconfig.json",
  "tsconfig.node.json",
];
const SKIP_NAMES = new Set([".DS_Store"]);

export interface BuildStamp {
  format: number;
  sources: string;
  git_head: string | null;
  files: Record<string, string>;
}

function sha256(data: Buffer | string): string {
  return createHash("sha256").update(data).digest("hex");
}

function byPath(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function kind(path: string): "dir" | "file" | "other" {
  try {
    const stat = statSync(path); // follows symlinks, like Python's is_dir()/is_file()
    return stat.isDirectory() ? "dir" : stat.isFile() ? "file" : "other";
  } catch {
    return "other";
  }
}

/** Regular files under `dir`, as `prefix/...` POSIX paths. */
function walk(dir: string, prefix: string, found: string[]): string[] {
  for (const name of readdirSync(dir)) {
    const path = prefix ? `${prefix}/${name}` : name;
    const entry = kind(join(dir, name));
    if (entry === "dir") walk(join(dir, name), path, found);
    else if (entry === "file") found.push(path);
  }
  return found;
}

/** The build's inputs under `frontend`, sorted by path. */
export function uiSourceFiles(frontend: string): string[] {
  const found: string[] = [];
  for (const name of UI_SOURCE_DIRS) {
    if (kind(join(frontend, name)) === "dir") walk(join(frontend, name), name, found);
  }
  for (const name of UI_SOURCE_FILES) {
    if (kind(join(frontend, name)) === "file") found.push(name);
  }
  return found.filter((path) => !SKIP_NAMES.has(path.split("/").pop() ?? "")).sort(byPath);
}

/** The stamp's `sources` digest of the frontend directory. */
export function uiSourcesDigest(frontend: string): string {
  const digest = createHash("sha256");
  for (const path of uiSourceFiles(frontend)) {
    digest.update(`${path}\0${sha256(readFileSync(join(frontend, path)))}\n`);
  }
  return digest.digest("hex");
}

/** SHA-256 of every file under `dist` except the stamp: the build's manifest. */
export function distManifest(dist: string): Record<string, string> {
  const files: Record<string, string> = {};
  for (const path of walk(dist, "", []).sort(byPath)) {
    if (path !== UI_STAMP) files[path] = sha256(readFileSync(join(dist, path)));
  }
  return files;
}

/** The commit checked out at `cwd`, or null (no git, not a checkout). */
export function gitHead(cwd: string): string | null {
  try {
    const head = execFileSync("git", ["rev-parse", "HEAD"], {
      cwd,
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .trim();
    return /^([0-9a-f]{40}|[0-9a-f]{64})$/.test(head) ? head : null;
  } catch {
    return null;
  }
}

export function buildStamp(frontend: string, dist: string): BuildStamp {
  return {
    format: UI_STAMP_FORMAT,
    sources: uiSourcesDigest(frontend),
    git_head: gitHead(frontend),
    files: distManifest(dist),
  };
}

/** Writes the stamp after the bundle and the public files are in the output directory. */
export function so101BuildStamp(frontend: string): Plugin {
  let outDir = resolve(frontend, "dist");
  return {
    name: "so101-ui-stamp",
    apply: "build",
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    writeBundle: {
      sequential: true,
      order: "post",
      handler() {
        const stamp = buildStamp(frontend, outDir);
        writeFileSync(join(outDir, UI_STAMP), `${JSON.stringify(stamp, null, 2)}\n`);
      },
    },
  };
}

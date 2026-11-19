// Camera stream keys, as GET /api/robot/status lists them in `cameras` and
// `live_cameras` (the server's SO101_CAMERAS grammar, docs/realsense.md).
//
// An entry is `[<name>=]<source>`, credentials already stripped by the server,
// and it is also the key of its color stream. A RealSense entry with `+depth`
// adds a second stream, `<entry>#depth`, which the server colorizes before it
// sends frames. Keys can hold `#`, `=`, `:` and `+`, so they always travel
// percent-encoded (see videoUrl and snapshotUrl in api.ts).

/** Appended to an entry to form the key of its depth stream. */
export const DEPTH_SUFFIX = "#depth";

/** A name, as the server's grammar allows one before the first "=". */
const NAMED_ENTRY = /^([A-Za-z][A-Za-z0-9_]{0,31})=(.+)$/s;

// realsense[:<token>[:<token>]][+depth], a token being a mode <W>x<H>[@<fps>]
// or a serial (4-32 letters/digits); see camera_spec.py on the server.
const REALSENSE_SCHEME = "realsense";
const DEPTH_FLAG = "+depth";
const MODE = /^\d+x\d+(?:@\d+)?$/i;
const SERIAL = /^[0-9A-Za-z]{4,32}$/;

/** What the server knows about a camera, should it ever expose it. */
export interface CameraSpecView {
  name?: string | null;
  entry?: string | null;
}

export function isDepthKey(key: string): boolean {
  return key.endsWith(DEPTH_SUFFIX);
}

/** The entry (the color stream key) a stream key belongs to. */
export function baseKey(key: string): string {
  return isDepthKey(key) ? key.slice(0, -DEPTH_SUFFIX.length) : key;
}

/** The `<name>` of a `<name>=<source>` entry, or null. Text before "=" that is not a name is part of the source. */
export function cameraName(entry: string): string | null {
  return NAMED_ENTRY.exec(entry)?.[1] ?? null;
}

/**
 * "RealSense <serial>" for a RealSense source (the serial may be a device
 * name such as "front"), "RealSense" when it means the only attached one.
 * Modes and +depth are left out. Null for anything else, or a source the
 * server's grammar would not accept.
 */
export function realsenseLabel(source: string): string | null {
  const lower = source.toLowerCase();
  if (lower !== REALSENSE_SCHEME && !lower.startsWith(`${REALSENSE_SCHEME}:`) && !lower.startsWith(`${REALSENSE_SCHEME}+`)) {
    return null;
  }
  let body = source.slice(REALSENSE_SCHEME.length);
  if (body.toLowerCase().endsWith(DEPTH_FLAG)) body = body.slice(0, -DEPTH_FLAG.length);
  if (body.includes("+")) return null;
  const tokens = body ? body.slice(1).split(":") : [];
  if (tokens.length > 2 || tokens.some((token) => !MODE.test(token) && !SERIAL.test(token))) return null;
  const serial = tokens.find((token) => !MODE.test(token));
  return serial ? `RealSense ${serial}` : "RealSense";
}

/**
 * What a tile calls a stream: the camera's name; else, for a RealSense,
 * "RealSense <serial>"; else its entry. A depth stream adds " depth". Without
 * a spec everything comes from the key; the tile's tooltip keeps the full key.
 */
export function cameraLabel(key: string, spec?: CameraSpecView): string {
  const entry = spec?.entry || baseKey(key);
  const label = spec?.name || cameraName(entry) || realsenseLabel(entry) || entry;
  return isDepthKey(key) ? `${label} depth` : label;
}

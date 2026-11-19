// MJPEG stream budget (F13).
//
// Each MJPEG <img> holds one HTTP/1.1 connection open for as long as it is
// mounted, and browsers allow six per host. Four camera tiles plus polling
// could starve the API, and that queue would include E-stop requests. At most
// MAX_MJPEG_STREAMS tiles stream at once; the rest poll a single-frame
// snapshot, which holds a connection only briefly.

export const MAX_MJPEG_STREAMS = 2;

type Listener = () => void;

/** Tile ids that want a stream, in grant order. The first N hold slots. */
let queue: string[] = [];
const listeners = new Set<Listener>();

function emit() {
  for (const listener of listeners) listener();
}

export function subscribeStreams(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function requestStream(id: string) {
  if (queue.includes(id)) return;
  queue = [...queue, id];
  emit();
}

export function releaseStream(id: string) {
  if (!queue.includes(id)) return;
  queue = queue.filter((entry) => entry !== id);
  emit();
}

/** Move a waiting tile to the front, taking a slot from the newest holder. */
export function promoteStream(id: string) {
  queue = [id, ...queue.filter((entry) => entry !== id)];
  emit();
}

/**
 * "none" until the tile has asked for a slot. Child effects run before the
 * tile's own request, so a tile must not start a snapshot fetch in that
 * window, or every tile fetches once on mount.
 */
export type StreamSlotState = "none" | "waiting" | "granted";

export function streamSlotState(id: string): StreamSlotState {
  const index = queue.indexOf(id);
  if (index < 0) return "none";
  return index < MAX_MJPEG_STREAMS ? "granted" : "waiting";
}

/** Number of tiles currently granted a live stream. */
export function activeStreamCount(): number {
  return Math.min(queue.length, MAX_MJPEG_STREAMS);
}

/** Test helper. */
export function resetStreamBudget() {
  queue = [];
  emit();
}

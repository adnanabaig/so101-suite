// A camera of the live MuJoCo scene (GET /api/simulation/render). There is
// no stream, so the tile polls, but only while it can be seen and only when
// the scene has moved since the frame on screen: a paused simulation costs
// one render.
import { useEffect, useId, useRef, useState } from "react";
import { getSimulationFrame } from "../../api";
import { useDocumentVisible, useOnScreen } from "../../hooks/useVisibility";

/** At most one frame per this interval while the scene changes. */
export const FRAME_INTERVAL_MS = 500;
/**
 * After a failed request (a 503 when the GL probe is hung or 8 frames are
 * already queued), wait this long, doubling per failure up to the cap.
 */
export const FRAME_RETRY_BASE_MS = 1000;
export const FRAME_RETRY_MAX_MS = 30_000;

/** The wait before the next try after `failures` failures in a row (1 or more). */
export function frameRetryDelay(failures: number): number {
  return Math.min(FRAME_RETRY_BASE_MS * 2 ** Math.max(0, failures - 1), FRAME_RETRY_MAX_MS);
}

const CAMERA_LABELS: Record<string, string> = {
  front: "Front",
  wrist_cam: "Wrist",
};

interface Frame {
  url: string;
  camera: string;
  step: number | null;
}

export default function SimCameraTile({
  cameras,
  sceneVersion,
}: {
  cameras: readonly string[];
  /** Changes whenever the scene may look different, e.g. generation and step. */
  sceneVersion: string;
}) {
  const [chosen, setChosen] = useState(cameras[0] ?? "front");
  const camera = cameras.includes(chosen) ? chosen : (cameras[0] ?? "front");
  const [frame, setFrame] = useState<Frame | null>(null);
  const [error, setError] = useState<{ message: string; retryInMs: number } | null>(null);
  const selectId = useId();

  const rootRef = useRef<HTMLDivElement>(null);
  const onScreen = useOnScreen(rootRef);
  const visible = useDocumentVisible();
  const showing = onScreen && visible;

  // Read by the polling loop, so a new scene version does not restart it.
  const versionRef = useRef(sceneVersion);
  useEffect(() => {
    versionRef.current = sceneVersion;
  });

  // The object URL on screen; revoked when replaced and on unmount.
  const urlRef = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    },
    [],
  );

  useEffect(() => {
    if (!showing) return undefined;
    let stopped = false;
    let timer: number | undefined;
    // Scene version of the frame shown for this camera; a new loop (camera
    // change, tab shown again) always fetches a fresh frame.
    let shownVersion: string | null = null;
    let failures = 0;
    const tick = async () => {
      timer = undefined;
      const wanted = versionRef.current;
      let delay = FRAME_INTERVAL_MS;
      if (wanted !== shownVersion) {
        try {
          const result = await getSimulationFrame({ camera });
          if (stopped) return;
          const url = URL.createObjectURL(result.blob);
          const previous = urlRef.current;
          urlRef.current = url;
          shownVersion = wanted;
          setFrame({ url, camera, step: result.step });
          setError(null);
          failures = 0;
          if (previous) URL.revokeObjectURL(previous);
        } catch (failure) {
          if (stopped) return;
          failures += 1;
          delay = frameRetryDelay(failures);
          setError({ message: failure instanceof Error ? failure.message : String(failure), retryInMs: delay });
        }
      }
      if (!stopped) timer = window.setTimeout(() => void tick(), delay);
    };
    void tick();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [showing, camera]);

  const label = (name: string) => CAMERA_LABELS[name] ?? name;
  const current = frame && frame.camera === camera ? frame : null;

  return (
    <div ref={rootRef}>
      {cameras.length > 1 && (
        <div className="mb-3 flex items-center gap-2 text-xs text-slate-400">
          <label htmlFor={selectId}>Camera</label>
          <select
            id={selectId}
            value={camera}
            onChange={(event) => setChosen(event.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-950/70 px-2 py-1 text-xs text-slate-100 outline-hidden focus-visible:border-cyan-400 focus-visible:ring-2 focus-visible:ring-cyan-400/40"
          >
            {cameras.map((name) => (
              <option key={name} value={name}>
                {label(name)}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-xl border border-white/5 bg-slate-950">
        {current ? (
          <img
            src={current.url}
            alt={`Simulated ${label(current.camera).toLowerCase()} camera${current.step === null ? "" : ` at step ${current.step}`}`}
            className="h-full w-full object-contain"
          />
        ) : (
          <span className="px-4 text-center text-xs text-slate-400">
            {error ? "No frame yet." : showing ? "Rendering a frame…" : "Frames load while this view is visible."}
          </span>
        )}
      </div>
      <p className="mt-2 text-[11px] text-slate-400">
        {current
          ? `Rendered at simulation step ${current.step ?? "unknown"}. New frames follow the scene.`
          : "Rendered on the server from the live physics scene."}
      </p>
      {error && (
        <p role="status" className="mt-2 text-xs leading-relaxed text-amber-200">
          {current ? "Showing the last frame. " : ""}
          {error.message} Next try in {Math.round(error.retryInMs / 1000)} s.
        </p>
      )}
    </div>
  );
}

import { useEffect, useId, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { Video, VideoOff } from "lucide-react";
import { snapshotUrl, videoUrl } from "../api";
import { useNow } from "../hooks/useNow";
import { cameraLabel } from "../lib/cameras";
import { useDocumentVisible, useOnScreen } from "../hooks/useVisibility";
import { STATUS_FRESH_MS } from "../lib/safetyView";
import {
  promoteStream,
  releaseStream,
  requestStream,
  streamSlotState,
  subscribeStreams,
  type StreamSlotState,
} from "../lib/streamBudget";
import { useSafety } from "./safety/SafetyProvider";

/** A frame older than this is shown as stalled, never as live (F11). */
export const FRAME_STALL_MS = 2000;
/** Secondary tiles refresh a single frame at this period (F13). */
export const SNAPSHOT_INTERVAL_MS = 1000;
const FIRST_FRAME_TIMEOUT_MS = 3000;

interface CameraViewProps {
  cam: string;
  /** The server reports this camera live. When false a placeholder is shown. */
  active?: boolean;
  className?: string;
}

type TileMode = "paused" | "hidden" | "offscreen" | "pending" | "stream" | "snapshot";

function useStreamSlot(id: string, wants: boolean): StreamSlotState {
  useEffect(() => {
    if (!wants) return;
    requestStream(id);
    return () => releaseStream(id);
  }, [id, wants]);
  return useSyncExternalStore(
    subscribeStreams,
    () => (wants ? streamSlotState(id) : "none"),
    () => "none",
  );
}

function formatSeconds(ms: number): string {
  return `${Math.max(0, Math.round(ms / 1000))}s`;
}

/**
 * One camera tile. It streams MJPEG only while the tab is visible, the tile is
 * on screen and it holds one of the stream slots; otherwise it pauses or
 * shows a 1 Hz snapshot (F13). Stalls are overlaid so a frozen frame is never
 * presented as live (F11).
 */
export default function CameraView({ cam, active = true, className = "" }: CameraViewProps) {
  const id = useId();
  const tileRef = useRef<HTMLDivElement>(null);
  const pageVisible = useDocumentVisible();
  const onScreen = useOnScreen(tileRef);
  const wantsVideo = active && pageVisible && onScreen;
  const slot = useStreamSlot(`${id}:${cam}`, wantsVideo);
  const [retry, setRetry] = useState(0);
  const { status, statusUpdatedAt } = useSafety();
  const now = useNow(1000);

  const mode: TileMode = !active
    ? "paused"
    : !pageVisible
      ? "hidden"
      : !onScreen
        ? "offscreen"
        : slot === "granted"
          ? "stream"
          : slot === "waiting"
            ? "snapshot"
            : "pending";

  const label = cameraLabel(cam);
  const frameAge = status?.camera_frame_age_ms?.[cam];
  const statusAge = statusUpdatedAt === null ? null : Math.max(0, now - statusUpdatedAt);
  const statusStale = statusAge !== null && statusAge > STATUS_FRESH_MS;
  const frameStalled = typeof frameAge === "number" && frameAge > FRAME_STALL_MS;
  const stalled = mode === "stream" && (frameStalled || statusStale);

  return (
    <div
      ref={tileRef}
      data-testid={`camera-${cam}`}
      data-mode={mode}
      className={`overflow-hidden rounded-xl border border-slate-800 bg-black ${className}`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-slate-800 bg-slate-900/70 px-3 py-1.5">
        <span className="truncate font-mono text-xs text-slate-300" title={cam === label ? undefined : cam}>
          {label}
        </span>
        <span className="flex items-center gap-2 text-[10px] text-muted">
          {mode === "snapshot" && "snapshot"}
          {mode === "stream" && (stalled ? "stalled" : "live")}
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full ${
              mode === "stream" && !stalled
                ? "motion-safe:animate-pulse bg-emerald-400"
                : stalled
                  ? "bg-amber-400"
                  : "bg-slate-600"
            }`}
          />
        </span>
      </div>
      <div className="relative flex aspect-video items-center justify-center">
        {mode === "stream" ? (
          <MjpegStream
            key={`${cam}:${retry}`}
            cam={cam}
            label={label}
            retry={retry}
            onRetry={() => setRetry((value) => value + 1)}
          />
        ) : mode === "snapshot" ? (
          <SnapshotStream key={cam} cam={cam} label={label} onWatchLive={() => promoteStream(`${id}:${cam}`)} />
        ) : (
          <Placeholder
            text={
              mode === "paused"
                ? "no live frames from this camera"
                : mode === "hidden"
                  ? "paused while this tab is hidden"
                  : mode === "pending"
                    ? "starting camera"
                    : "paused while off screen"
            }
          />
        )}
        {stalled && (
          <div
            role="status"
            className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-black/70 text-center text-xs font-semibold text-amber-200"
          >
            <VideoOff aria-hidden="true" className="h-6 w-6" />
            {statusStale
              ? `Feed state unknown · status ${formatSeconds(statusAge ?? 0)} old`
              : `Feed stalled · last frame ${formatSeconds(frameAge ?? 0)} old`}
          </div>
        )}
      </div>
    </div>
  );
}

function Placeholder({ text, children }: { text: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-1 px-3 text-center text-muted">
      <Video aria-hidden="true" className="h-8 w-8" />
      <span className="text-xs">{text}</span>
      {children}
    </div>
  );
}

function MjpegStream({
  cam,
  label,
  retry,
  onRetry,
}: {
  cam: string;
  label: string;
  retry: number;
  onRetry: () => void;
}) {
  const [errored, setErrored] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const firstFrameTimerRef = useRef<number | null>(null);

  useEffect(() => {
    firstFrameTimerRef.current = window.setTimeout(() => setErrored(true), FIRST_FRAME_TIMEOUT_MS);
    const img = imgRef.current;
    return () => {
      if (firstFrameTimerRef.current !== null) {
        window.clearTimeout(firstFrameTimerRef.current);
        firstFrameTimerRef.current = null;
      }
      // Dropping src aborts the multipart request and frees the connection.
      img?.removeAttribute("src");
    };
  }, []);

  if (errored) {
    return (
      <Placeholder text="stream unavailable">
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 rounded-md border border-slate-700 px-2 py-1 text-[10px] text-slate-300 hover:border-slate-600 hover:text-slate-100"
        >
          retry stream
        </button>
      </Placeholder>
    );
  }
  return (
    <img
      ref={imgRef}
      src={`${videoUrl(cam)}&retry=${retry}`}
      alt={`camera ${label}, live`}
      className="h-full w-full object-contain"
      onLoad={() => {
        if (firstFrameTimerRef.current !== null) {
          window.clearTimeout(firstFrameTimerRef.current);
          firstFrameTimerRef.current = null;
        }
      }}
      onError={() => setErrored(true)}
    />
  );
}

async function readDetail(response: Response): Promise<string> {
  if (response.status === 404) return "this server has no snapshot endpoint";
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
      return body.detail;
    }
  } catch {
    // Not JSON; fall through to the status code.
  }
  return `snapshot failed (HTTP ${response.status})`;
}

function SnapshotStream({ cam, label, onWatchLive }: { cam: string; label: string; onWatchLive: () => void }) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [takenAt, setTakenAt] = useState<number | null>(null);
  const now = useNow(1000);

  useEffect(() => {
    let disposed = false;
    let objectUrl: string | null = null;
    let controller: AbortController | null = null;
    let timer: number | null = null;

    const tick = async () => {
      controller?.abort();
      const current = new AbortController();
      controller = current;
      try {
        const response = await fetch(snapshotUrl(cam), { cache: "no-store", signal: current.signal });
        if (!response.ok) {
          const detail = await readDetail(response);
          if (disposed) return;
          setError(detail);
          // A missing endpoint will not appear by retrying every second.
          if (response.status === 404 && timer !== null) {
            window.clearInterval(timer);
            timer = null;
          }
          return;
        }
        const blob = await response.blob();
        if (disposed) return;
        if (typeof URL.createObjectURL === "function") {
          const next = URL.createObjectURL(blob);
          if (objectUrl) URL.revokeObjectURL(objectUrl);
          objectUrl = next;
          setSrc(next);
        }
        setTakenAt(Date.now());
        setError(null);
      } catch (caught) {
        if (disposed || current.signal.aborted) return;
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    };

    void tick();
    timer = window.setInterval(() => void tick(), SNAPSHOT_INTERVAL_MS);
    return () => {
      disposed = true;
      if (timer !== null) window.clearInterval(timer);
      controller?.abort();
      if (objectUrl && typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(objectUrl);
    };
  }, [cam]);

  const age = takenAt === null ? null : Math.max(0, now - takenAt);
  const old = age !== null && age > FRAME_STALL_MS + SNAPSHOT_INTERVAL_MS;
  const watchLive = (
    <button
      type="button"
      onClick={onWatchLive}
      className="rounded-md border border-slate-600 bg-slate-900/80 px-2 py-1 text-[10px] font-semibold text-slate-200 hover:bg-slate-800"
    >
      Watch live
    </button>
  );

  if (!src) {
    return <Placeholder text={error ?? "loading snapshot"}>{watchLive}</Placeholder>;
  }
  return (
    <>
      <img src={src} alt={`camera ${label}, snapshot`} className="h-full w-full object-contain" />
      <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between gap-2">
        <span
          className={`rounded-sm px-1.5 py-0.5 font-mono text-[10px] ${
            old || error ? "bg-amber-500/80 text-slate-950" : "bg-black/60 text-slate-300"
          }`}
        >
          {error ? `${error} · ` : ""}
          {age === null ? "" : `${formatSeconds(age)} ago`}
        </span>
        {watchLive}
      </div>
    </>
  );
}

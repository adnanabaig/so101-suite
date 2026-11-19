import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from "react";
import { RefreshCw, TriangleAlert } from "lucide-react";

/**
 * Messages browsers and Vite use when a code-split chunk cannot be fetched,
 * usually because the server was redeployed and the old hashed file is gone.
 */
const CHUNK_ERROR_PATTERN =
  /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module|Unable to preload CSS|ChunkLoadError|Loading (CSS )?chunk [\w-]+ failed/i;

export function isChunkLoadError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  return error.name === "ChunkLoadError" || CHUNK_ERROR_PATTERN.test(error.message);
}

export function reloadPage() {
  window.location.reload();
}

function RouteErrorPanel({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const chunk = isChunkLoadError(error);
  return (
    <div
      role="alert"
      className="mx-auto mt-10 max-w-xl rounded-2xl border border-amber-400/40 bg-slate-900/80 p-6 text-sm text-slate-200"
    >
      <div className="flex items-center gap-2 text-base font-bold text-amber-100">
        <TriangleAlert aria-hidden="true" className="h-5 w-5" />
        {chunk ? "This page could not be loaded" : "This page stopped working"}
      </div>
      <p className="mt-2 leading-relaxed text-slate-300">
        {chunk
          ? "The operator UI was probably updated on the server, so this browser holds an outdated copy. Reload to get the current version."
          : "Something in this page failed. The safety bar above still works."}{" "}
        E-stop (Escape) and the other stops in the header are unaffected.
      </p>
      <p className="mt-3 break-words rounded-lg bg-slate-950/70 px-3 py-2 font-mono text-xs text-slate-400">
        {error.message || String(error)}
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={reloadPage}
          className="inline-flex h-10 items-center gap-2 rounded-lg border border-cyan-400/40 bg-cyan-500 px-4 text-xs font-bold text-slate-950 hover:bg-cyan-400"
        >
          <RefreshCw aria-hidden="true" className="h-4 w-4" />
          Reload the UI
        </button>
        {!chunk && (
          <button
            type="button"
            onClick={onRetry}
            className="h-10 rounded-lg border border-slate-600 px-4 text-xs font-semibold text-slate-200 hover:bg-slate-800"
          >
            Try again
          </button>
        )}
      </div>
    </div>
  );
}

interface BoundaryProps {
  children: ReactNode;
  /** Changing this (e.g. the route) clears a caught error. */
  resetKey?: string;
}

interface BoundaryState {
  error: Error | null;
}

/**
 * Route-level boundary inside Layout: a crashed or unloadable page is replaced
 * by a recovery panel while the header, safety bar and E-stop keep working (F5).
 */
export class ErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error("Operator UI page crashed", error, info.componentStack);
  }

  componentDidUpdate(previous: BoundaryProps) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error) {
      return <RouteErrorPanel error={this.state.error} onRetry={() => this.setState({ error: null })} />;
    }
    return this.props.children;
  }
}

/**
 * True once Vite reports that preloading a chunk failed (`vite:preloadError`).
 * The default is left alone so the import still rejects into the boundary;
 * this only adds a prompt. The page never reloads by itself, because an
 * unexpected reload mid-operation drops the operator's lease and context.
 */
export function useUpdateAvailable(): boolean {
  const [available, setAvailable] = useState(false);
  useEffect(() => {
    const onPreloadError = () => setAvailable(true);
    window.addEventListener("vite:preloadError", onPreloadError);
    return () => window.removeEventListener("vite:preloadError", onPreloadError);
  }, []);
  return available;
}

export function UpdateBanner() {
  return (
    <div
      role="status"
      className="flex flex-wrap items-center gap-3 rounded-xl border border-cyan-400/40 bg-slate-900/95 px-4 py-3 text-sm text-cyan-50 shadow-lg"
    >
      <RefreshCw aria-hidden="true" className="h-5 w-5 shrink-0" />
      <p className="min-w-0 flex-1 text-xs leading-relaxed">
        <span className="font-bold">A newer operator UI is on the server.</span> Part of this page
        could not load. Reload when the robot is in a safe state.
      </p>
      <button
        type="button"
        onClick={reloadPage}
        className="h-9 shrink-0 rounded-lg border border-cyan-400/40 bg-cyan-500 px-3 text-xs font-bold text-slate-950 hover:bg-cyan-400"
      >
        Reload to update
      </button>
    </div>
  );
}

/** Best-effort E-stop that depends on nothing but fetch. */
function bareEstop() {
  void fetch("/api/robot/estop", { method: "POST" }).catch(() => undefined);
}

function CrashScreen({ error }: { error: Error }) {
  const [sent, setSent] = useState(false);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      if (!event.repeat) {
        bareEstop();
        setSent(true);
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, []);
  return (
    <div role="alert" className="flex min-h-screen items-center justify-center bg-slate-950 p-6 text-slate-100">
      <div className="w-full max-w-lg rounded-2xl border border-rose-400/40 bg-slate-900 p-6">
        <h1 className="text-lg font-bold">The operator UI crashed</h1>
        <p className="mt-2 text-sm text-slate-300">
          This screen cannot show robot state. If the arm may be moving, press E-stop (or Escape)
          and cut physical power if you cannot confirm that it stopped.
        </p>
        <p className="mt-3 break-words rounded-lg bg-slate-950 px-3 py-2 font-mono text-xs text-slate-400">
          {error.message || String(error)}
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <button
            type="button"
            aria-label="E-stop"
            aria-keyshortcuts="Escape"
            onClick={() => {
              bareEstop();
              setSent(true);
            }}
            className="h-12 rounded-xl border border-rose-300/50 bg-rose-600 px-6 text-sm font-black uppercase tracking-[0.14em] text-white hover:bg-rose-700 active:bg-rose-800"
          >
            E-stop
          </button>
          <button
            type="button"
            onClick={reloadPage}
            className="h-12 rounded-xl border border-slate-600 px-5 text-sm font-semibold hover:bg-slate-800"
          >
            Reload the UI
          </button>
        </div>
        <p className="mt-3 text-xs text-slate-400" aria-live="polite">
          {sent ? "E-stop sent. This screen cannot confirm the latch; check the arm." : ""}
        </p>
      </div>
    </div>
  );
}

/** Last-resort boundary around the whole app (main.tsx). */
export class RootErrorBoundary extends Component<{ children: ReactNode }, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): BoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error("Operator UI crashed", error, info.componentStack);
  }

  render() {
    if (this.state.error) return <CrashScreen error={this.state.error} />;
    return this.props.children;
  }
}

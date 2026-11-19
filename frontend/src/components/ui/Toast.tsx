import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { CheckCircle2, Info, X, XCircle } from "lucide-react";

/*
 * Toasts confirm routine actions ("Checklist hidden"). They never carry
 * safety state: stops, latches and faults live in the safety bar and its
 * banners, which nothing may cover. The stack sits bottom-right, below the
 * header and dialogs, and is a polite live region that exists before any
 * toast arrives so screen readers announce each one.
 */

export type ToastTone = "info" | "success" | "error";

export interface ToastOptions {
  tone?: ToastTone;
}

export interface ToastApi {
  /** Show a message. Info and success fade after TOAST_MS; errors stay until dismissed. */
  show: (message: string, options?: ToastOptions) => void;
  dismiss: (id: number) => void;
}

interface ToastItem {
  id: number;
  message: string;
  tone: ToastTone;
}

/** How long an info or success toast stays up. */
export const TOAST_MS = 6000;
/** At most this many toasts at once; the oldest goes first. */
export const MAX_TOASTS = 3;

const INERT_API: ToastApi = { show: () => undefined, dismiss: () => undefined };
const ToastContext = createContext<ToastApi>(INERT_API);

const TONE_STYLES: Record<ToastTone, string> = {
  info: "border-cyan-400/30 text-cyan-50",
  success: "border-emerald-400/30 text-emerald-50",
  error: "border-rose-400/40 text-rose-50",
};

const TONE_ICONS = { info: Info, success: CheckCircle2, error: XCircle } as const;
const TONE_ICON_COLOURS: Record<ToastTone, string> = {
  info: "text-cyan-300",
  success: "text-emerald-300",
  error: "text-rose-300",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(0);
  const timers = useRef(new Map<number, number>());

  const dismiss = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer !== undefined) window.clearTimeout(timer);
    timers.current.delete(id);
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const show = useCallback(
    (message: string, options: ToastOptions = {}) => {
      const tone = options.tone ?? "info";
      const id = ++nextId.current;
      setToasts((current) => {
        // The same message again replaces the old one instead of stacking.
        const kept = current.filter((toast) => toast.message !== message);
        return [...kept, { id, message, tone }].slice(-MAX_TOASTS);
      });
      if (tone !== "error") {
        timers.current.set(
          id,
          window.setTimeout(() => dismiss(id), TOAST_MS),
        );
      }
    },
    [dismiss],
  );

  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const timer of pending.values()) window.clearTimeout(timer);
      pending.clear();
    };
  }, []);

  const api = useMemo<ToastApi>(() => ({ show, dismiss }), [show, dismiss]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        aria-relevant="additions text"
        data-testid="toast-region"
        className="pointer-events-none fixed bottom-4 right-4 z-30 flex w-[min(22rem,calc(100vw-2rem))] flex-col items-stretch gap-2"
      >
        {toasts.map((toast) => {
          const Icon = TONE_ICONS[toast.tone];
          return (
            <div
              key={toast.id}
              data-tone={toast.tone}
              className={`pointer-events-auto flex items-start gap-2.5 rounded-xl border bg-slate-900/95 px-3.5 py-3 text-xs leading-relaxed shadow-xl shadow-black/40 ${TONE_STYLES[toast.tone]}`}
            >
              <Icon aria-hidden="true" className={`mt-0.5 h-4 w-4 shrink-0 ${TONE_ICON_COLOURS[toast.tone]}`} />
              <p className="min-w-0 flex-1">{toast.message}</p>
              <button
                type="button"
                aria-label="Dismiss notification"
                onClick={() => dismiss(toast.id)}
                className="-m-1 shrink-0 rounded-md p-1 text-slate-300 hover:bg-white/10 hover:text-white focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-300"
              >
                <X aria-hidden="true" className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

/** Toasts from the shell; outside a ToastProvider (isolated tests) it does nothing. */
export function useToast(): ToastApi {
  return useContext(ToastContext);
}

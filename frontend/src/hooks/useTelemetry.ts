import {
  createElement,
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { Telemetry } from "../api";

export type WsState = "connecting" | "open" | "closed";

/**
 * Joint-sample state, separate from transport freshness (C6).
 * - live:  joints sampled by an active loop or armed session within 500 ms
 * - idle:  nothing is sampling joints; neutral, never a warning (F8)
 * - stale: a loop should be sampling but the sample is old or missing
 */
export type SampleState = "live" | "idle" | "stale";

/**
 * Why the live link keeps failing, when the browser can tell (C1).
 * - refused:     the server closed the socket with 1008 (policy violation);
 *                `detail` is its close reason, verbatim ("" if none).
 * - unreachable: {@link NEVER_OPENED_HINT_AFTER} sockets in a row closed
 *                without ever opening. Browsers report a guard rejection
 *                before accept() as 1006, so this is the usual symptom.
 */
export type TelemetryLinkIssue =
  | { kind: "refused"; detail: string }
  | { kind: "unreachable"; attempts: number };

export interface TelemetryResult {
  telemetry: Telemetry | null;
  state: WsState;
  /** Set while the link is failing in a way that names a cause; cleared on open. */
  linkIssue: TelemetryLinkIssue | null;
  /** Client time the last valid frame arrived. */
  lastMessageAt: number | null;
  /** No frame for {@link TRANSPORT_STALE_MS} or the socket is not open. */
  transportStale: boolean;
  sampleState: SampleState;
  /** Legacy alias: `sampleState === "stale"`. */
  sampleStale: boolean;
  sampleAgeMs: number | null;
  /** `transportStale || sampleState === "stale"`. Idle is not stale. */
  stale: boolean;
  /** A fresh frame carrying live joint samples. */
  hasLiveObservation: boolean;
}

/** A frame older than this means the link is down for display purposes. */
export const TRANSPORT_STALE_MS = 2500;
/** An open socket that stays silent this long is forced to reconnect (F26). */
export const SILENCE_RECONNECT_MS = 5000;
/** Pre-C6 backends: a sample older than this is stale. */
const LEGACY_SAMPLE_STALE_MS = 6000;
/** C1 guard: WebSocket close code for a refused Origin or Host. */
export const WS_POLICY_VIOLATION = 1008;
/** Consecutive never-opened sockets before the link is reported as refused-or-down. */
export const NEVER_OPENED_HINT_AFTER = 3;

export function sampleStateOf(telemetry: Telemetry | null): SampleState {
  if (!telemetry) return "stale";
  if (typeof telemetry.observation_live === "boolean") {
    return telemetry.observation_live ? "live" : "idle";
  }
  // Pre-C6 backend: infer from the mode and sample age.
  const hasObservation = Object.keys(telemetry.observation ?? {}).length > 0;
  if (telemetry.mode === "idle" && !hasObservation) return "idle";
  if (!hasObservation || Math.max(0, telemetry.sample_age_ms) > LEGACY_SAMPLE_STALE_MS) {
    return telemetry.mode === "idle" ? "idle" : "stale";
  }
  return "live";
}

const TelemetryContext = createContext<TelemetryResult | null>(null);

/** Owns the application's single telemetry WebSocket. */
export function TelemetryProvider({ children }: { children: ReactNode }) {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [state, setState] = useState<WsState>("connecting");
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const [linkIssue, setLinkIssue] = useState<TelemetryLinkIssue | null>(null);
  const [clock, setClock] = useState(() => Date.now());

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<number | null>(null);
  const attemptRef = useRef(0);
  const closedByUnmountRef = useRef(false);
  // Sockets in a row that closed before they ever opened.
  const neverOpenedRef = useRef(0);
  // Last time the current socket showed signs of life (opened or delivered a frame).
  const lastActivityRef = useRef(0);

  useEffect(() => {
    closedByUnmountRef.current = false;

    // Declared before connect() so both closures can reference each other;
    // the timer only calls connect() after this effect body has finished.
    const scheduleReconnect = () => {
      if (closedByUnmountRef.current || reconnectRef.current !== null) return;
      const attempt = attemptRef.current++;
      const delay = Math.min(750 * 2 ** attempt, 10000);
      reconnectRef.current = window.setTimeout(() => {
        reconnectRef.current = null;
        connect();
      }, delay);
    };

    const connect = () => {
      if (closedByUnmountRef.current) return;
      // Same origin as the page, whatever host and port served it.
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${window.location.host}/ws/telemetry`;
      setState("connecting");
      let opened = false;

      let ws: WebSocket;
      try {
        ws = new WebSocket(url);
      } catch {
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;
      lastActivityRef.current = Date.now();

      ws.onopen = () => {
        if (wsRef.current !== ws) return;
        opened = true;
        attemptRef.current = 0;
        neverOpenedRef.current = 0;
        lastActivityRef.current = Date.now();
        setLinkIssue(null);
        setState("open");
      };

      ws.onmessage = (event) => {
        if (wsRef.current !== ws) return;
        try {
          const parsed = JSON.parse(event.data as string) as Telemetry;
          if (
            typeof parsed === "object" &&
            parsed !== null &&
            typeof parsed.t === "number" &&
            typeof parsed.mode === "string"
          ) {
            const now = Date.now();
            lastActivityRef.current = now;
            setTelemetry(parsed);
            setLastMessageAt(now);
          }
        } catch {
          // A malformed frame should not tear down the status stream.
        }
      };

      ws.onerror = () => {
        // Browsers follow an error with close; close schedules reconnect.
      };

      ws.onclose = (event?: Pick<CloseEvent, "code" | "reason">) => {
        if (wsRef.current !== ws) return;
        wsRef.current = null;
        if (closedByUnmountRef.current) return;
        if (event?.code === WS_POLICY_VIOLATION) {
          setLinkIssue({ kind: "refused", detail: event.reason ?? "" });
        } else if (!opened) {
          neverOpenedRef.current += 1;
          const attempts = neverOpenedRef.current;
          if (attempts >= NEVER_OPENED_HINT_AFTER) {
            setLinkIssue((current) =>
              current?.kind === "refused" ? current : { kind: "unreachable", attempts },
            );
          }
        }
        setState("closed");
        scheduleReconnect();
      };
    };

    // F26: a socket can stay "open" while delivering nothing (wedged server
    // task, half-open TCP after a Wi-Fi roam). Abandon it and reconnect.
    const watchdog = window.setInterval(() => {
      const ws = wsRef.current;
      if (!ws || closedByUnmountRef.current) return;
      if (Date.now() - lastActivityRef.current < SILENCE_RECONNECT_MS) return;
      wsRef.current = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      try {
        ws.close();
      } catch {
        // Closing an already-dead socket can throw in some browsers.
      }
      setState("closed");
      scheduleReconnect();
    }, 1000);

    connect();

    return () => {
      closedByUnmountRef.current = true;
      window.clearInterval(watchdog);
      if (reconnectRef.current !== null) {
        window.clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
      if (wsRef.current) {
        const ws = wsRef.current;
        wsRef.current = null;
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const value = useMemo<TelemetryResult>(() => {
    const transportStale =
      state !== "open" || lastMessageAt === null || clock - lastMessageAt > TRANSPORT_STALE_MS;
    const sampleAgeMs = telemetry ? Math.max(0, telemetry.sample_age_ms) : null;
    const sampleState = sampleStateOf(telemetry);
    return {
      telemetry,
      state,
      linkIssue,
      lastMessageAt,
      transportStale,
      sampleState,
      sampleStale: sampleState === "stale",
      sampleAgeMs,
      stale: transportStale || sampleState === "stale",
      hasLiveObservation: !transportStale && sampleState === "live",
    };
  }, [telemetry, state, linkIssue, lastMessageAt, clock]);

  return createElement(TelemetryContext.Provider, { value }, children);
}

export function useTelemetry(): TelemetryResult {
  const value = useContext(TelemetryContext);
  if (!value) {
    throw new Error("useTelemetry must be used inside TelemetryProvider");
  }
  return value;
}

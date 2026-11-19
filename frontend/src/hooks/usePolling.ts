import { useCallback, useEffect, useRef, useState } from "react";

/** Primitive identity of what is being polled (compared with Object.is). */
export type PollingKey = string | number | boolean | null | undefined;

export interface PollingOptions {
  /** Toggle polling on/off (default true). Data is kept while disabled. */
  enabled?: boolean;
  /**
   * Identity of the polled resource, e.g. the selected arm or job id. When it
   * changes, the previous key's data is cleared in the same render and late
   * responses for it are discarded (F18).
   */
  key?: PollingKey;
  /**
   * Skip interval ticks while the document is hidden and refetch as soon as it
   * is visible again (default true, F28).
   */
  pauseWhenHidden?: boolean;
  /** Consecutive failures before `stale` turns true (default 2). */
  staleAfterFailures?: number;
}

export interface PollingResult<T> {
  data: T | null;
  error: Error | null;
  /** True until the first response (success or failure) for the current key. */
  loading: boolean;
  /**
   * Fetch now. A refresh requested while a fetch is in flight runs once more
   * after it, so callers always observe post-mutation data (F19).
   */
  refresh: () => void;
  /** Client time of the last successful response for the current key. */
  lastSuccessAt: number | null;
  /** Consecutive failures since the last success. */
  failures: number;
  /** `failures >= staleAfterFailures`: the data shown may be out of date. */
  stale: boolean;
  /**
   * Adopt data the server just returned from a mutation (e.g. the EvalState a
   * score call answers with) as the current value. A fetch already in flight
   * started before the mutation, so its response is discarded (F19).
   */
  mutate: (data: T) => void;
}

interface PollState<T> {
  key: PollingKey;
  data: T | null;
  error: Error | null;
  settled: boolean;
  lastSuccessAt: number | null;
  failures: number;
}

function initialState<T>(key: PollingKey): PollState<T> {
  return { key, data: null, error: null, settled: false, lastSuccessAt: null, failures: 0 };
}

function normalise(options: boolean | PollingOptions): PollingOptions {
  return typeof options === "boolean" ? { enabled: options } : options;
}

/**
 * Interval polling with keyed results, queued refreshes, hidden-tab pausing
 * and freshness metadata.
 *
 * @param fetcher    async function returning the data; the latest closure is
 *                   always used, so changing its identity never restarts timers
 * @param intervalMs polling interval in milliseconds; 0 fetches once
 * @param options    `enabled` boolean (legacy form) or {@link PollingOptions}
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs = 2000,
  options: boolean | PollingOptions = true,
): PollingResult<T> {
  const { enabled = true, key, pauseWhenHidden = true, staleAfterFailures = 2 } =
    normalise(options);

  const [state, setState] = useState<PollState<T>>(() => initialState<T>(key));
  // F18: a new key clears the previous key's data before anything renders it.
  let current = state;
  if (!Object.is(state.key, key)) {
    current = initialState<T>(key);
    setState(current);
  }

  // Latest fetcher and key, updated before the polling effect runs.
  const fetcherRef = useRef(fetcher);
  const keyRef = useRef(key);
  useEffect(() => {
    fetcherRef.current = fetcher;
    keyRef.current = key;
  });

  const mountedRef = useRef(false);
  const inFlightRef = useRef<{ key: PollingKey; token: number } | null>(null);
  const queuedRef = useRef(false);
  const tokenRef = useRef(0);
  // Bumped by mutate(): responses to fetches started earlier are out of date.
  const epochRef = useRef(0);
  const runRef = useRef<() => void>(() => undefined);

  const run = useCallback(() => {
    if (!mountedRef.current) return;
    const runKey = keyRef.current;
    const inFlight = inFlightRef.current;
    if (inFlight && Object.is(inFlight.key, runKey)) {
      queuedRef.current = true;
      return;
    }
    const token = ++tokenRef.current;
    const epoch = epochRef.current;
    inFlightRef.current = { key: runKey, token };
    queuedRef.current = false;
    const execute = async () => {
      try {
        const result = await fetcherRef.current();
        if (!mountedRef.current || epochRef.current !== epoch) return;
        const receivedAt = Date.now();
        setState((prev) =>
          Object.is(prev.key, runKey)
            ? {
                key: runKey,
                data: result,
                error: null,
                settled: true,
                lastSuccessAt: receivedAt,
                failures: 0,
              }
            : prev,
        );
      } catch (caught) {
        if (!mountedRef.current || epochRef.current !== epoch) return;
        const error = caught instanceof Error ? caught : new Error(String(caught));
        setState((prev) =>
          Object.is(prev.key, runKey)
            ? { ...prev, error, settled: true, failures: prev.failures + 1 }
            : prev,
        );
      } finally {
        if (inFlightRef.current?.token === token) {
          inFlightRef.current = null;
          if (queuedRef.current && mountedRef.current) {
            queuedRef.current = false;
            runRef.current();
          }
        }
      }
    };
    void execute();
  }, []);

  useEffect(() => {
    runRef.current = run;
  }, [run]);

  const mutate = useCallback((data: T) => {
    if (!mountedRef.current) return;
    epochRef.current += 1;
    const mutateKey = keyRef.current;
    const at = Date.now();
    setState((prev) =>
      Object.is(prev.key, mutateKey)
        ? { key: mutateKey, data, error: null, settled: true, lastSuccessAt: at, failures: 0 }
        : prev,
    );
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      inFlightRef.current = null;
      queuedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const visible = () => !pauseWhenHidden || document.visibilityState !== "hidden";
    if (visible()) run();
    if (!intervalMs) return;
    const id = window.setInterval(() => {
      if (visible()) run();
    }, intervalMs);
    const onVisibility = () => {
      if (pauseWhenHidden && document.visibilityState === "visible") run();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [run, intervalMs, enabled, key, pauseWhenHidden]);

  return {
    data: current.data,
    error: current.error,
    loading: enabled && !current.settled,
    refresh: run,
    lastSuccessAt: current.lastSuccessAt,
    failures: current.failures,
    stale: current.failures >= staleAfterFailures,
    mutate,
  };
}

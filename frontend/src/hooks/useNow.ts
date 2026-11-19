import { useEffect, useState } from "react";

/**
 * Wall-clock time that re-renders the caller every `intervalMs`.
 *
 * Components must not call Date.now() during render (the result would change
 * between renders for the same props); they read this ticking value instead.
 */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

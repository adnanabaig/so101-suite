import { useSyncExternalStore } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

function mediaQuery(): MediaQueryList | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return null;
  return window.matchMedia(QUERY);
}

function subscribe(listener: () => void) {
  const query = mediaQuery();
  if (!query || typeof query.addEventListener !== "function") return () => undefined;
  query.addEventListener("change", listener);
  return () => query.removeEventListener("change", listener);
}

/**
 * True while the operating system asks for reduced motion. CSS animations
 * already honour it (index.css); this is for motion driven from script, such
 * as the digital twin easing between poses.
 */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, () => mediaQuery()?.matches ?? false, () => false);
}

import { useEffect, useState, useSyncExternalStore, type RefObject } from "react";

function subscribeVisibility(listener: () => void) {
  document.addEventListener("visibilitychange", listener);
  return () => document.removeEventListener("visibilitychange", listener);
}

/** False while the tab is hidden (background tab, minimised window). */
export function useDocumentVisible(): boolean {
  return useSyncExternalStore(
    subscribeVisibility,
    () => document.visibilityState !== "hidden",
    () => true,
  );
}

/**
 * Whether the element intersects the viewport. Without IntersectionObserver
 * (old browsers, jsdom) everything counts as on screen.
 */
export function useOnScreen(ref: RefObject<Element | null>, rootMargin = "100px"): boolean {
  const [onScreen, setOnScreen] = useState(true);
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[entries.length - 1];
        if (entry) setOnScreen(entry.isIntersecting);
      },
      { rootMargin },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, rootMargin]);
  return onScreen;
}

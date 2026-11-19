import { useCallback, useSyncExternalStore } from "react";

// First-run setup state kept in this browser. Versioned keys, so a future
// checklist with new items can ask again instead of inheriting old answers.

/** The operator hid the setup checklist. Settings can show it again. */
export const CHECKLIST_DISMISSED_KEY = "so101.onboarding.dismissed.v1";
/** The operator confirmed they know E-stop, Halt and Collapse. */
export const STOPS_ACKNOWLEDGED_KEY = "so101.onboarding.stops.v1";

const KEYS = [CHECKLIST_DISMISSED_KEY, STOPS_ACKNOWLEDGED_KEY];

// When storage is blocked, answers last for this page load only.
const memory = new Map<string, boolean>();
const listeners = new Set<() => void>();

function readFlag(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === "1";
  } catch {
    return memory.get(key) ?? false;
  }
}

function writeFlag(key: string, value: boolean): void {
  memory.set(key, value);
  try {
    if (value) window.localStorage.setItem(key, "1");
    else window.localStorage.removeItem(key);
  } catch {
    /* storage blocked or full: the in-memory answer stands */
  }
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Another tab changed an answer.
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || KEYS.includes(event.key)) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** A persisted yes/no answer, shared live by every component that reads it. */
export function usePersistentFlag(key: string): [boolean, (value: boolean) => void] {
  const value = useSyncExternalStore(subscribe, () => readFlag(key));
  const set = useCallback((next: boolean) => writeFlag(key, next), [key]);
  return [value, set];
}

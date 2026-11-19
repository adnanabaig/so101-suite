import { act, renderHook } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { usePrefersReducedMotion } from "./usePrefersReducedMotion";

afterEach(() => {
  vi.unstubAllGlobals();
});

function stubMediaQuery(initial: boolean) {
  const listeners = new Set<() => void>();
  const query = {
    matches: initial,
    media: "(prefers-reduced-motion: reduce)",
    addEventListener: (_type: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener),
  };
  vi.stubGlobal("matchMedia", () => query);
  return {
    listeners,
    set(matches: boolean) {
      query.matches = matches;
      act(() => listeners.forEach((listener) => listener()));
    },
  };
}

test("follows the operating system setting as it changes", () => {
  const media = stubMediaQuery(false);
  const { result, unmount } = renderHook(() => usePrefersReducedMotion());
  expect(result.current).toBe(false);

  media.set(true);
  expect(result.current).toBe(true);
  media.set(false);
  expect(result.current).toBe(false);

  unmount();
  expect(media.listeners.size).toBe(0);
});

test("browsers without matchMedia get full motion", () => {
  vi.stubGlobal("matchMedia", undefined);
  const { result } = renderHook(() => usePrefersReducedMotion());
  expect(result.current).toBe(false);
});

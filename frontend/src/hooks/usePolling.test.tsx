import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { advance, deferred, fakeShellTimers, type Deferred } from "../test/harness";
import { usePolling } from "./usePolling";

let visibility: DocumentVisibilityState = "visible";

function setVisibility(state: DocumentVisibilityState) {
  visibility = state;
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
}

beforeEach(() => {
  fakeShellTimers();
  visibility = "visible";
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => visibility,
  });
});

afterEach(() => {
  Reflect.deleteProperty(document, "visibilityState");
  vi.useRealTimers();
});

describe("usePolling", () => {
  test("records lastSuccessAt and turns stale after repeated failures", async () => {
    let fail = false;
    const fetcher = vi.fn(() => (fail ? Promise.reject(new Error("down")) : Promise.resolve("ok")));
    const { result } = renderHook(() => usePolling(fetcher, 1000));
    expect(result.current.loading).toBe(true);
    await advance(0);
    expect(result.current.data).toBe("ok");
    expect(result.current.loading).toBe(false);
    const firstSuccess = result.current.lastSuccessAt;
    expect(firstSuccess).toBe(Date.now());

    fail = true;
    await advance(1000);
    expect(result.current.failures).toBe(1);
    expect(result.current.stale).toBe(false);
    // The last good data stays visible, but its age is known.
    expect(result.current.data).toBe("ok");
    expect(result.current.lastSuccessAt).toBe(firstSuccess);

    await advance(1000);
    expect(result.current.failures).toBe(2);
    expect(result.current.stale).toBe(true);
    expect(result.current.error?.message).toBe("down");

    fail = false;
    await advance(1000);
    expect(result.current.stale).toBe(false);
    expect(result.current.failures).toBe(0);
    expect(result.current.lastSuccessAt).toBeGreaterThan(firstSuccess ?? 0);
  });

  test("a refresh during an in-flight fetch runs once more afterwards (F19)", async () => {
    const pending: Deferred<number>[] = [];
    const fetcher = vi.fn(() => {
      const next = deferred<number>();
      pending.push(next);
      return next.promise;
    });
    const { result } = renderHook(() => usePolling(fetcher, 0));
    expect(fetcher).toHaveBeenCalledTimes(1);

    // A mutation completes and asks for fresh data while the first read is in flight.
    act(() => {
      result.current.refresh();
      result.current.refresh();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending[0].resolve(1);
      await Promise.resolve();
    });
    await advance(0);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async () => {
      pending[1].resolve(2);
      await Promise.resolve();
    });
    await advance(0);
    expect(result.current.data).toBe(2);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  test("skips ticks while the tab is hidden and refetches when it returns (F28)", async () => {
    const fetcher = vi.fn(() => Promise.resolve("ok"));
    renderHook(() => usePolling(fetcher, 1000));
    await advance(0);
    expect(fetcher).toHaveBeenCalledTimes(1);

    setVisibility("hidden");
    await advance(5000);
    expect(fetcher).toHaveBeenCalledTimes(1);

    setVisibility("visible");
    await advance(0);
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  test("a response for a previous key is discarded (F18)", async () => {
    const pending = new Map<string, Deferred<string>>();
    const fetcher = (key: string) => {
      const next = deferred<string>();
      pending.set(key, next);
      return next.promise;
    };
    const { result, rerender } = renderHook(({ id }) => usePolling(() => fetcher(id), 0, { key: id }), {
      initialProps: { id: "job-1" },
    });
    rerender({ id: "job-2" });
    await act(async () => {
      pending.get("job-1")?.resolve("job-1 data");
      await Promise.resolve();
    });
    await advance(0);
    expect(result.current.data).toBeNull();
    await act(async () => {
      pending.get("job-2")?.resolve("job-2 data");
      await Promise.resolve();
    });
    await advance(0);
    expect(result.current.data).toBe("job-2 data");
  });

  test("mutate adopts a mutation's response and discards an older in-flight read (F19)", async () => {
    const pending: Deferred<string>[] = [];
    const fetcher = vi.fn(() => {
      const next = deferred<string>();
      pending.push(next);
      return next.promise;
    });
    const { result } = renderHook(() => usePolling(fetcher, 1000));
    await act(async () => {
      pending[0].resolve("before");
      await Promise.resolve();
    });
    expect(result.current.data).toBe("before");

    // The next tick starts a read; a mutation answers before that read does.
    await advance(1000);
    expect(fetcher).toHaveBeenCalledTimes(2);
    act(() => result.current.mutate("scored"));
    expect(result.current.data).toBe("scored");
    expect(result.current.lastSuccessAt).toBe(Date.now());

    await act(async () => {
      pending[1].resolve("stale read");
      await Promise.resolve();
    });
    expect(result.current.data).toBe("scored");

    // Reads started after the mutation are adopted as usual.
    await advance(1000);
    await act(async () => {
      pending[2].resolve("after");
      await Promise.resolve();
    });
    expect(result.current.data).toBe("after");
  });

  test("disabled polling keeps its data and does not fetch", async () => {
    const fetcher = vi.fn(() => Promise.resolve("ok"));
    const { result, rerender } = renderHook(({ enabled }) => usePolling(fetcher, 1000, enabled), {
      initialProps: { enabled: true },
    });
    await advance(0);
    rerender({ enabled: false });
    await advance(5000);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(result.current.data).toBe("ok");
    expect(result.current.loading).toBe(false);
  });
});

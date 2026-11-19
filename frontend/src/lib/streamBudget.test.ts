import { afterEach, describe, expect, test, vi } from "vitest";
import {
  activeStreamCount,
  MAX_MJPEG_STREAMS,
  promoteStream,
  releaseStream,
  requestStream,
  resetStreamBudget,
  streamSlotState,
  subscribeStreams,
} from "./streamBudget";

afterEach(() => {
  resetStreamBudget();
});

describe("streamBudget (F13)", () => {
  test("grants slots in request order and queues the rest", () => {
    expect(MAX_MJPEG_STREAMS).toBe(2);
    expect(streamSlotState("a")).toBe("none");
    for (const id of ["a", "b", "c"]) requestStream(id);
    requestStream("a");
    expect(["a", "b", "c"].map(streamSlotState)).toEqual(["granted", "granted", "waiting"]);
    expect(activeStreamCount()).toBe(2);
  });

  test("a released slot passes to the next waiting tile", () => {
    for (const id of ["a", "b", "c"]) requestStream(id);
    releaseStream("a");
    releaseStream("missing");
    expect(["a", "b", "c"].map(streamSlotState)).toEqual(["none", "granted", "granted"]);
  });

  test("promotion takes the slot of the newest holder and notifies subscribers", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeStreams(listener);
    for (const id of ["a", "b", "c"]) requestStream(id);
    listener.mockClear();
    promoteStream("c");
    expect(listener).toHaveBeenCalledTimes(1);
    expect(["a", "b", "c"].map(streamSlotState)).toEqual(["granted", "waiting", "granted"]);
    unsubscribe();
    requestStream("d");
    expect(listener).toHaveBeenCalledTimes(1);
  });
});

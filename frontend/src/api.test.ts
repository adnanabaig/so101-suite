import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  api,
  approveAgentProposal,
  cancelAgentProposal,
  continueEval,
  disarmControl,
  errorDetail,
  EVAL_START_TIMEOUT_MS,
  executeAgentProposal,
  getSimulationFrame,
  heartbeatOperator,
  HEARTBEAT_TIMEOUT_MS,
  holdAgentProposal,
  isMissingRoute,
  pushDataset,
  resetEstop,
  resetFault,
  scoreEval,
  snapshotUrl,
  startEval,
  startRollout,
  videoUrl,
} from "./api";
import { getOperatorId } from "./lib/operatorSession";

function stubFetch(response: () => Response | Promise<Response>) {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response());
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function lastCall(fetchMock: ReturnType<typeof stubFetch>) {
  return fetchMock.mock.calls[fetchMock.mock.calls.length - 1];
}

function lastBody(fetchMock: ReturnType<typeof stubFetch>): unknown {
  const init = lastCall(fetchMock)?.[1];
  return typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("errorDetail (F16)", () => {
  it("returns a string detail as is", () => {
    expect(errorDetail({ detail: "lease required" })).toBe("lease required");
  });

  it("joins Pydantic's array detail with field paths", () => {
    expect(
      errorDetail({
        detail: [
          { type: "missing", loc: ["body", "source"], msg: "Field required", input: {} },
          { type: "less_than_equal", loc: ["body", "constraints", "fps"], msg: "Input should be <= 240" },
          { loc: ["body"], msg: "Invalid JSON" },
        ],
      }),
    ).toBe("source: Field required; constraints.fps: Input should be <= 240; Invalid JSON");
  });

  it("returns null for bodies without a usable detail", () => {
    expect(errorDetail(null)).toBeNull();
    expect(errorDetail({ detail: 42 })).toBeNull();
    expect(errorDetail({ detail: [{ nope: true }] })).toBeNull();
  });
});

describe("request", () => {
  it("surfaces a 422 array detail as the error message (F16)", async () => {
    stubFetch(() =>
      json({ detail: [{ type: "missing", loc: ["body", "source"], msg: "Field required" }] }, 422),
    );
    const error = await api.post("/api/control/lease", {}).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(422);
    expect((error as ApiError).message).toBe("source: Field required");
  });

  it("falls back to the raw text, then to the status line", async () => {
    stubFetch(() => new Response("upstream exploded", { status: 502 }));
    await expect(api.get("/api/x")).rejects.toThrow("upstream exploded");
    stubFetch(() => new Response("", { status: 503 }));
    await expect(api.get("/api/x")).rejects.toThrow("GET /api/x failed: 503");
  });

  it("throws on a non-JSON 2xx instead of returning undefined (F25)", async () => {
    stubFetch(
      () => new Response("<!doctype html><title>SO-101</title>", { status: 200, headers: { "content-type": "text/html; charset=utf-8" } }),
    );
    const error = await api.get("/api/nonexistent").catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toMatch(/returned text\/html instead of JSON/);
    expect(isMissingRoute(error)).toBe(true);
  });

  it("every request goes to a path on the page's own origin, never a fixed host or port", async () => {
    const fetchMock = stubFetch(() => json({ ok: true, detail: "" }));
    await api.get("/api/robot/status");
    await scoreEval(true);
    await cancelAgentProposal("p");
    for (const [input] of fetchMock.mock.calls) {
      expect(typeof input).toBe("string");
      expect(String(input)).toMatch(/^\/api\//);
    }
  });

  it("returns undefined for 204", async () => {
    stubFetch(() => new Response(null, { status: 204 }));
    await expect(api.del("/api/control/lease/1")).resolves.toBeUndefined();
  });

  it("times out while the body is still arriving", async () => {
    vi.useFakeTimers();
    // Like a real fetch, the body stream errors when the request is aborted.
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const stream = new ReadableStream({
          start(controller) {
            init?.signal?.addEventListener("abort", () =>
              controller.error(new DOMException("The operation was aborted.", "AbortError")),
            );
          },
        });
        return new Response(stream, { status: 200, headers: { "content-type": "application/json" } });
      }),
    );
    const pending = api.post("/api/agent/proposals/p/hold", {}, 1000).catch((caught: unknown) => caught);
    await vi.advanceTimersByTimeAsync(1001);
    const error = await pending;
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(408);
    expect((error as ApiError).message).toBe("POST /api/agent/proposals/p/hold timed out");
  });
});

describe("timeouts", () => {
  function stubHangingFetch() {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_input: RequestInfo | URL, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () =>
              reject(new DOMException("The operation was aborted.", "AbortError")),
            );
          }),
      ),
    );
  }

  it("a hung operator heartbeat gives up after 1.5 s so the next pulse can go (C3: 2 s window)", async () => {
    vi.useFakeTimers();
    stubHangingFetch();
    expect(HEARTBEAT_TIMEOUT_MS).toBeLessThan(2000);
    const pending = heartbeatOperator().catch((caught: unknown) => caught);
    await vi.advanceTimersByTimeAsync(HEARTBEAT_TIMEOUT_MS + 1);
    expect(await pending).toMatchObject({ status: 408 });
  });

  it("eval start waits for the server's policy load instead of the default 8 s", async () => {
    vi.useFakeTimers();
    stubHangingFetch();
    let settled: unknown = "pending";
    const pending = startEval({ model_path: "m", n_episodes: 1, max_seconds: 5 }).catch((caught: unknown) => {
      settled = caught;
    });
    await vi.advanceTimersByTimeAsync(30_000);
    expect(settled).toBe("pending");
    await vi.advanceTimersByTimeAsync(EVAL_START_TIMEOUT_MS);
    await pending;
    expect(settled).toMatchObject({ status: 408, message: "POST /api/eval/start timed out" });
  });
});

describe("getSimulationFrame", () => {
  it("asks for a small JPEG of the front camera and reads the instant from the headers", async () => {
    const fetchMock = stubFetch(
      () =>
        new Response(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), {
          headers: {
            "content-type": "image/jpeg",
            "x-so101-sim-camera": "front",
            "x-so101-sim-step": "42",
            "x-so101-sim-generation": "3",
            "x-so101-sim-time-s": "0.700000",
          },
        }),
    );
    const frame = await getSimulationFrame();
    expect(lastCall(fetchMock)?.[0]).toBe(
      "/api/simulation/render?camera=front&width=320&height=240&format=jpeg&quality=80",
    );
    expect(lastCall(fetchMock)?.[1]?.method).toBe("GET");
    expect(frame).toMatchObject({ camera: "front", step: 42, generation: 3, simTimeS: 0.7 });
    expect(frame.blob.size).toBe(4);
  });

  it("accepts the PNG a server without a JPEG encoder sends; missing headers read as null", async () => {
    const fetchMock = stubFetch(() => new Response(new Uint8Array([1]), { headers: { "content-type": "image/png" } }));
    const frame = await getSimulationFrame({ camera: "wrist cam", width: 640, height: 480, format: "png" });
    expect(lastCall(fetchMock)?.[0]).toBe(
      "/api/simulation/render?camera=wrist+cam&width=640&height=480&format=png&quality=80",
    );
    expect(frame).toMatchObject({ camera: null, step: null, generation: null, simTimeS: null });
  });

  it("a 503 surfaces the server's reason", async () => {
    stubFetch(() => json({ detail: "rendering needs the MuJoCo engine; the kinematic engine is active" }, 503));
    const error = await getSimulationFrame().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(503);
    expect((error as ApiError).message).toBe("rendering needs the MuJoCo engine; the kinematic engine is active");
    expect(isMissingRoute(error)).toBe(false);
  });

  it("a page instead of an image is a missing route, not a broken picture", async () => {
    stubFetch(() => json({ ok: true }));
    const error = await getSimulationFrame().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toMatch(/returned application\/json instead of an image/);
    expect(isMissingRoute(error)).toBe(true);
  });
});

describe("isMissingRoute", () => {
  it("recognises FastAPI's default 404 and a 405, not a domain 404", () => {
    expect(isMissingRoute(new ApiError(404, "Not Found", ""))).toBe(true);
    expect(isMissingRoute(new ApiError(405, "Method Not Allowed", ""))).toBe(true);
    expect(isMissingRoute(new ApiError(404, "proposal not found", ""))).toBe(false);
    expect(isMissingRoute(new ApiError(409, "conflict", ""))).toBe(false);
    expect(isMissingRoute(new Error("Not Found"))).toBe(false);
  });
});

describe("camera URLs", () => {
  it("percent-encode the stream key, so a depth key's # does not end the query", () => {
    const depth = "front=realsense:123456789012+depth#depth";
    expect(videoUrl(depth)).toBe("/api/video?cam=front%3Drealsense%3A123456789012%2Bdepth%23depth");
    expect(snapshotUrl(depth)).toBe("/api/video/snapshot?cam=front%3Drealsense%3A123456789012%2Bdepth%23depth");
    expect(new URL(snapshotUrl(depth), "http://robot.local").searchParams.get("cam")).toBe(depth);
    expect(new URL(videoUrl("realsense:front"), "http://robot.local").searchParams.get("cam")).toBe("realsense:front");
  });
});

describe("contract bodies", () => {
  it("C2 E-stop reset sends the operator and the JSON literal true; C5 disarm sends only the lease", async () => {
    const fetchMock = stubFetch(() => json({ ok: true, detail: "" }));
    await resetEstop();
    expect(lastCall(fetchMock)?.[0]).toBe("/api/robot/estop/reset");
    expect(lastBody(fetchMock)).toStrictEqual({ operator_id: getOperatorId(), confirm: true });
    await disarmControl("lease-7");
    expect(lastCall(fetchMock)?.[0]).toBe("/api/control/disarm");
    expect(lastBody(fetchMock)).toStrictEqual({ lease_id: "lease-7" });
  });

  it("C2 fault reset sends the power-cut acknowledgement only when asked", async () => {
    const fetchMock = stubFetch(() => json({ ok: true, detail: "" }));
    await resetFault({ powerCutAcknowledged: true });
    expect(lastBody(fetchMock)).toEqual({
      operator_id: getOperatorId(),
      confirm: true,
      power_cut_acknowledged: true,
    });
    await resetFault();
    expect(lastBody(fetchMock)).toEqual({ operator_id: getOperatorId(), confirm: true });
  });

  it("C3 score and continue send JSON bodies", async () => {
    const fetchMock = stubFetch(() => json({ active: true }));
    await scoreEval(true);
    expect(lastCall(fetchMock)?.[0]).toBe("/api/eval/score");
    expect(lastBody(fetchMock)).toEqual({ operator_id: getOperatorId(), success: true });
    await continueEval();
    expect(lastCall(fetchMock)?.[0]).toBe("/api/eval/continue");
    expect(lastBody(fetchMock)).toEqual({ operator_id: getOperatorId() });
  });

  it("C8 approve, execute and hold carry the lease, operator, digest and token", async () => {
    const fetchMock = stubFetch(() => json({ approval_token: "t", expires_at: "", proposal: { id: "p 1" } }));
    await approveAgentProposal("p 1", "lease-1", "ab".repeat(32));
    expect(lastCall(fetchMock)?.[0]).toBe("/api/agent/proposals/p%201/approve");
    expect(lastBody(fetchMock)).toEqual({
      lease_id: "lease-1",
      operator_id: getOperatorId(),
      target_digest: "ab".repeat(32),
    });
    await executeAgentProposal("p 1", "lease-1", "token-1");
    expect(lastBody(fetchMock)).toEqual({ lease_id: "lease-1", approval_token: "token-1" });
    await holdAgentProposal("p 1", "lease-1");
    expect(lastCall(fetchMock)?.[0]).toBe("/api/agent/proposals/p%201/hold");
    expect(lastBody(fetchMock)).toEqual({ lease_id: "lease-1" });
    await cancelAgentProposal("p 1");
    expect(lastCall(fetchMock)?.[0]).toBe("/api/agent/proposals/p%201/cancel");
    expect(lastCall(fetchMock)?.[1]?.method).toBe("POST");
    expect(lastCall(fetchMock)?.[1]?.body).toBeUndefined();
  });

  it("data audit: rollout and eval accept optional task and fps; push takes a JSON body", async () => {
    const fetchMock = stubFetch(() => json({ ok: true, detail: "" }));
    await startRollout({ model_path: "m", task: "pick the cube", fps: 15 });
    expect(lastBody(fetchMock)).toMatchObject({ model_path: "m", task: "pick the cube", fps: 15 });
    await startEval({
      model_path: "m",
      n_episodes: 2,
      max_seconds: 30,
      task: "place the cube",
      fps: 30,
    });
    expect(lastBody(fetchMock)).toMatchObject({
      model_path: "m",
      task: "place the cube",
      fps: 30,
    });
    await pushDataset("user/repo", { private: true });
    expect(lastCall(fetchMock)?.[0]).toBe("/api/datasets/user%2Frepo/push");
    expect(lastBody(fetchMock)).toEqual({ operator_id: getOperatorId(), private: true });
  });
});

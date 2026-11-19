import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentProposal } from "../api";
import { targetDigest } from "../lib/canonicalJson";
import {
  advance,
  deferred,
  fakeShellTimers,
  installFakeBackend,
  jsonResponse,
  type FakeBackend,
} from "../test/harness";
import {
  APPROVAL_MARGIN_MS,
  DIGEST_MISMATCH_MESSAGE,
  MAX_HOLDS_IN_FLIGHT,
  PRE_C8_MESSAGE,
  PROPOSAL_MAX_AGE_MS,
  PROPOSAL_TOO_OLD_MESSAGE,
  approvalDeadline,
  isApprovalUsable,
  proposalAgeDeadline,
  useAgentExecution,
  type AgentApprovalGrant,
  type UseAgentExecutionOptions,
} from "./useAgentExecution";

const TARGETS = { gripper: 80, wrist_roll: 12.5 };
const DIGEST = targetDigest(TARGETS);
const BASE = "/api/agent/proposals/p1";

function proposal(overrides: Partial<AgentProposal> = {}): AgentProposal {
  return {
    id: "p1",
    title: "Open the gripper",
    status: "proposed",
    targets: TARGETS,
    target_digest: DIGEST,
    constraints: { max_duration_seconds: 10, require_deadman: false, allow_motion: true },
    ...overrides,
  };
}

function held(overrides: Partial<AgentProposal> = {}): AgentProposal {
  return proposal({
    constraints: { max_duration_seconds: 10, require_deadman: true, allow_motion: true },
    ...overrides,
  });
}

// gate's agent_executor._result for a run that reached its targets.
const RESULT = {
  reached: true,
  final_error_deg: { gripper: 0.4, wrist_roll: 0.2 },
  max_abs_error_deg: 0.4,
  duration_s: 1.2,
  steps: 12,
  end_state: "deenergized",
  status: "executed",
  stop_reason: "target_reached",
  error: "",
  targets: TARGETS,
  tolerance: { arm_deg: 2, gripper: 3 },
  started_at: "2026-10-02T12:00:00.000000+00:00",
  finished_at: "2026-10-02T12:00:01.200000+00:00",
};

let backend: FakeBackend;
let server: AgentProposal;

/** A C8 server for proposal p1, whose state the tests move by hand. */
function installAgentRoutes(initial: AgentProposal) {
  server = initial;
  backend.route("GET", BASE, () => jsonResponse(server));
  backend.route("GET", "/api/agent/proposals", () => jsonResponse({ proposals: [server] }));
  backend.route("POST", `${BASE}/approve`, () =>
    jsonResponse({
      approval_token: "tok-1",
      expires_at: new Date(Date.now() + 30_000).toISOString(),
      proposal: { ...server, status: "approved" },
    }),
  );
  backend.route("POST", `${BASE}/execute`, () => {
    server = { ...server, status: "executing" };
    return jsonResponse(server, 202);
  });
  backend.route("POST", `${BASE}/hold`, () => jsonResponse({ ok: true, detail: "" }));
  backend.route("POST", `${BASE}/cancel`, () => {
    server = { ...server, status: "cancelled", result: { ...RESULT, reached: false } };
    return jsonResponse(server);
  });
}

function setup(overrides: Partial<UseAgentExecutionOptions> = {}) {
  const onChanged = vi.fn();
  const initialProps: UseAgentExecutionOptions = {
    leaseId: "lease-1",
    timestampFor: (clientTimeMs = Date.now()) => new Date(clientTimeMs).toISOString(),
    tripped: false,
    onChanged,
    ...overrides,
  };
  const hook = renderHook((props: UseAgentExecutionOptions) => useAgentExecution(props), {
    initialProps,
  });
  const rerender = (next: Partial<UseAgentExecutionOptions>) =>
    hook.rerender({ ...initialProps, ...next });
  return { ...hook, onChanged, rerender };
}

async function call<T>(fn: () => Promise<T>): Promise<T> {
  let out: T | undefined;
  await act(async () => {
    out = await fn();
  });
  return out as T;
}

let visibility: DocumentVisibilityState = "visible";

beforeEach(() => {
  fakeShellTimers();
  backend = installFakeBackend();
  visibility = "visible";
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => visibility });
});

afterEach(() => {
  vi.useRealTimers();
  Reflect.deleteProperty(document, "visibilityState");
});

describe("approvalDeadline", () => {
  const synced = (clientTimeMs = 0) => new Date(clientTimeMs).toISOString();

  it("uses the server's expiry on the server clock, minus a margin", () => {
    // Server clock is 5 s ahead; the token expires 10 s after it was issued.
    const ahead = (clientTimeMs = 0) => new Date(clientTimeMs + 5_000).toISOString();
    const expires = new Date(1_000 + 5_000 + 10_000).toISOString();
    expect(approvalDeadline(expires, 900, 1_000, ahead)).toBe(1_000 + 10_000 - APPROVAL_MARGIN_MS);
  });

  it("never runs past 30 s from the request, whatever the server says", () => {
    const farFuture = new Date(10_000_000).toISOString();
    expect(approvalDeadline(farFuture, 900, 1_000, synced)).toBe(900 + 30_000 - APPROVAL_MARGIN_MS);
  });

  it("falls back to the 30 s cap when the expiry cannot be read", () => {
    expect(approvalDeadline("soon", 900, 1_000, synced)).toBe(900 + 30_000 - APPROVAL_MARGIN_MS);
  });

  it("never runs past the proposal's own 120 s age limit", () => {
    const expires = new Date(1_000 + 30_000).toISOString();
    expect(approvalDeadline(expires, 900, 1_000, synced, 12_000)).toBe(12_000 - APPROVAL_MARGIN_MS);
    expect(approvalDeadline(expires, 900, 1_000, synced, null)).toBe(900 + 30_000 - APPROVAL_MARGIN_MS);
  });
});

describe("proposalAgeDeadline", () => {
  it("is 120 s after created_at on the server clock, in client time", () => {
    // Server clock 5 s ahead; the proposal was created 100 s ago (server time).
    const ahead = (clientTimeMs = 0) => new Date(clientTimeMs + 5_000).toISOString();
    const now = 1_000_000;
    const createdAt = new Date(now + 5_000 - 100_000).toISOString();
    expect(proposalAgeDeadline(createdAt, now, ahead)).toBe(now + PROPOSAL_MAX_AGE_MS - 100_000);
  });

  it("reads gate's microsecond ISO timestamps", () => {
    const synced = (clientTimeMs = 0) => new Date(clientTimeMs).toISOString();
    const now = Date.parse("2026-10-02T12:01:00.000Z");
    expect(proposalAgeDeadline("2026-10-02T12:00:00.123456+00:00", now, synced)).toBe(
      Date.parse("2026-10-02T12:02:00.123Z"),
    );
  });

  it("is unknown without a readable creation time", () => {
    const synced = (clientTimeMs = 0) => new Date(clientTimeMs).toISOString();
    expect(proposalAgeDeadline(undefined, 0, synced)).toBeNull();
    expect(proposalAgeDeadline("yesterday", 0, synced)).toBeNull();
  });
});

describe("isApprovalUsable", () => {
  const grant: AgentApprovalGrant = {
    proposalId: "p1",
    token: "tok",
    digest: DIGEST,
    leaseId: "lease-1",
    deadline: 10_000,
  };

  it("accepts the approved proposal under the same lease before the deadline", () => {
    expect(isApprovalUsable(grant, proposal(), "lease-1", 9_999)).toBe(true);
    expect(isApprovalUsable(grant, proposal({ status: "approved" }), "lease-1", 0)).toBe(true);
    expect(isApprovalUsable(grant, proposal({ target_digest: DIGEST.toUpperCase() }), "lease-1", 0)).toBe(true);
  });

  it("refuses another lease, a late use, a moved-on proposal or changed targets", () => {
    expect(isApprovalUsable(null, proposal(), "lease-1", 0)).toBe(false);
    expect(isApprovalUsable(grant, proposal({ id: "p2" }), "lease-1", 0)).toBe(false);
    expect(isApprovalUsable(grant, proposal(), "lease-2", 0)).toBe(false);
    expect(isApprovalUsable(grant, proposal(), null, 0)).toBe(false);
    expect(isApprovalUsable(grant, proposal(), "lease-1", 10_000)).toBe(false);
    expect(isApprovalUsable(grant, proposal({ status: "executing" }), "lease-1", 0)).toBe(false);
    expect(isApprovalUsable(grant, proposal({ target_digest: "f".repeat(64) }), "lease-1", 0)).toBe(false);
  });
});

describe("useAgentExecution: approve and execute (C8)", () => {
  it("approves by digest, executes with the token once, polls to the result", async () => {
    installAgentRoutes(proposal());
    const { result, onChanged } = setup();

    expect(await call(() => result.current.approve(server))).toBe(true);
    expect(backend.callsTo(`${BASE}/approve`)[0].body).toEqual({
      lease_id: "lease-1",
      operator_id: expect.any(String),
      target_digest: DIGEST,
    });
    expect(result.current.approval).toMatchObject({ proposalId: "p1", token: "tok-1", leaseId: "lease-1" });

    expect(await call(() => result.current.execute(server))).toBe(true);
    expect(backend.callsTo(`${BASE}/execute`)[0].body).toEqual({ lease_id: "lease-1", approval_token: "tok-1" });
    expect(result.current.approval).toBeNull();
    expect(result.current.run).toMatchObject({ proposalId: "p1", phase: "running" });
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(0);

    await advance(250);
    expect(result.current.latest?.status).toBe("executing");
    server = { ...server, status: "executed", result: RESULT };
    await advance(250);
    expect(result.current.latest).toMatchObject({ status: "executed", result: RESULT });
    expect(result.current.run?.phase).toBe("ended");
    expect(onChanged).toHaveBeenCalled();

    // Polling stopped, and the spent token cannot run it again.
    const polls = backend.callsTo(BASE, "GET").length;
    await advance(1000);
    expect(backend.callsTo(BASE, "GET")).toHaveLength(polls);
    expect(await call(() => result.current.execute(proposal()))).toBe(false);
    expect(result.current.error).toMatch(/^Approve this proposal first/);
    expect(backend.callsTo(`${BASE}/execute`)).toHaveLength(1);
  });

  it("refuses to approve targets whose digest does not match the server's", async () => {
    installAgentRoutes(proposal({ target_digest: "0".repeat(64) }));
    const { result } = setup();
    expect(await call(() => result.current.approve(server))).toBe(false);
    expect(result.current.error).toBe(DIGEST_MISMATCH_MESSAGE);
    expect(backend.callsTo(`${BASE}/approve`)).toHaveLength(0);
  });

  it("refuses an approval for a different target set than the one shown", async () => {
    installAgentRoutes(proposal());
    backend.route("POST", `${BASE}/approve`, () =>
      jsonResponse({
        approval_token: "tok-1",
        expires_at: new Date(Date.now() + 30_000).toISOString(),
        proposal: { ...server, target_digest: "1".repeat(64) },
      }),
    );
    const { result } = setup();
    expect(await call(() => result.current.approve(server))).toBe(false);
    expect(result.current.error).toMatch(/approved a different target set/);
    expect(result.current.approval).toBeNull();
  });

  it("explains a server without the C8 routes, and passes refusals through", async () => {
    installAgentRoutes(proposal());
    backend.route("POST", `${BASE}/approve`, () => jsonResponse({ detail: "Not Found" }, 404));
    const { result } = setup();
    await call(() => result.current.approve(server));
    expect(result.current.error).toBe(PRE_C8_MESSAGE);

    backend.route("POST", `${BASE}/approve`, () => jsonResponse({ detail: "lease is not held by this operator" }, 409));
    await call(() => result.current.approve(server));
    expect(result.current.error).toBe("Approval was refused: lease is not held by this operator");
  });

  it("shows gate's expiry refusal verbatim and refreshes, so the proposal reads expired", async () => {
    installAgentRoutes(proposal());
    backend.route("POST", `${BASE}/approve`, () => {
      server = { ...server, status: "expired", result: { age_seconds: 131.2 } };
      return jsonResponse({ detail: "proposal expired; create and review a fresh proposal" }, 409);
    });
    const { result, onChanged } = setup();
    expect(await call(() => result.current.approve(server))).toBe(false);
    expect(result.current.error).toBe(
      "Approval was refused: proposal expired; create and review a fresh proposal",
    );
    expect(onChanged).toHaveBeenCalled();
    expect(result.current.approval).toBeNull();
  });

  it("ends the approval at the proposal's 120 s age limit when that comes first", async () => {
    // Created 100 s ago: the server would refuse to execute it 20 s from now,
    // although the token itself lives 30 s.
    installAgentRoutes(proposal({ created_at: new Date(Date.now() - 100_000).toISOString() }));
    const { result } = setup();
    expect(await call(() => result.current.approve(server))).toBe(true);
    await advance(20_000 - APPROVAL_MARGIN_MS - 100);
    expect(result.current.approval).not.toBeNull();
    await advance(200);
    expect(result.current.approval).toBeNull();
  });

  it("says the proposal is too old when its age limit has passed by this tab's clock", async () => {
    installAgentRoutes(proposal({ created_at: new Date(Date.now() - 125_000).toISOString() }));
    const { result } = setup();
    expect(await call(() => result.current.approve(server))).toBe(false);
    expect(result.current.error).toBe(PROPOSAL_TOO_OLD_MESSAGE);
    expect(result.current.approval).toBeNull();
  });

  it("needs the lease to approve, and drops the approval when the lease changes", async () => {
    installAgentRoutes(proposal());
    const { result, rerender } = setup({ leaseId: null });
    await call(() => result.current.approve(server));
    expect(result.current.error).toBe("Acquire the Agent Lab lease first.");
    expect(backend.callsTo(`${BASE}/approve`)).toHaveLength(0);

    rerender({ leaseId: "lease-1" });
    await call(() => result.current.approve(server));
    rerender({ leaseId: "lease-2" });
    expect(await call(() => result.current.execute(server))).toBe(false);
    expect(backend.callsTo(`${BASE}/execute`)).toHaveLength(0);
  });

  it("expires the approval a second before the server would", async () => {
    installAgentRoutes(proposal());
    const { result } = setup();
    await call(() => result.current.approve(server));
    expect(result.current.approval).not.toBeNull();
    await advance(30_000 - APPROVAL_MARGIN_MS - 100);
    expect(result.current.approval).not.toBeNull();
    await advance(200);
    expect(result.current.approval).toBeNull();
  });

  it("re-checks the digest at execute time and spends the token either way", async () => {
    installAgentRoutes(proposal());
    const { result } = setup();
    await call(() => result.current.approve(server));
    // The targets on screen changed under the approval (same proposal id).
    const changed = proposal({ targets: { gripper: 10 }, target_digest: undefined });
    expect(await call(() => result.current.execute(changed))).toBe(false);
    expect(result.current.error).toBe(DIGEST_MISMATCH_MESSAGE);
    expect(result.current.approval).toBeNull();
    expect(backend.callsTo(`${BASE}/execute`)).toHaveLength(0);
  });

  it("reports a refused execute and clears the run", async () => {
    installAgentRoutes(proposal());
    backend.route("POST", `${BASE}/execute`, () => jsonResponse({ detail: "approval token expired" }, 409));
    const { result } = setup();
    await call(() => result.current.approve(server));
    expect(await call(() => result.current.execute(server))).toBe(false);
    expect(result.current.error).toBe("Execution was refused: approval token expired");
    expect(result.current.run).toBeNull();
  });

  it("stops polling with a warning when no terminal status arrives in time", async () => {
    installAgentRoutes(proposal());
    const { result } = setup();
    await call(() => result.current.approve(server));
    await call(() => result.current.execute(server));
    await advance(10_000 + 30_000 + 500);
    expect(result.current.error).toMatch(/has not reported a terminal status/);
    expect(result.current.run?.phase).toBe("ended");
  });

  it("falls back to the proposal list on a server without GET by id", async () => {
    installAgentRoutes(proposal());
    backend.route("GET", BASE, () => jsonResponse({ detail: "Not Found" }, 404));
    const { result } = setup();
    await call(() => result.current.approve(server));
    await call(() => result.current.execute(server));
    server = { ...server, status: "executed", result: RESULT };
    await advance(500);
    expect(result.current.latest?.status).toBe("executed");
    expect(backend.callsTo(BASE, "GET")).toHaveLength(1);
  });
});

describe("useAgentExecution: hold-to-run (require_deadman)", () => {
  async function startHeld(result: ReturnType<typeof setup>["result"]) {
    await call(() => result.current.approve(server));
    await act(async () => {
      result.current.pressHold(server);
      await vi.advanceTimersByTimeAsync(0);
    });
  }

  it("pings hold every 100 ms while held and cancels on release", async () => {
    installAgentRoutes(held());
    const { result } = setup();
    await startHeld(result);
    expect(result.current.holding).toBe(true);
    expect(result.current.run?.phase).toBe("running");
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(1);
    expect(backend.callsTo(`${BASE}/hold`)[0].body).toEqual({ lease_id: "lease-1" });
    await advance(300);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(4);

    act(() => result.current.releaseHold("deadman released"));
    expect(result.current.holding).toBe(false);
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(result.current.notice).toBe("Run cancelled (deadman released). The result records what moved.");
    expect(result.current.latest?.status).toBe("cancelled");
    expect(result.current.run?.phase).toBe("ended");
    await advance(500);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(4);
  });

  it("cancels on arrival when released before the 202", async () => {
    installAgentRoutes(held());
    const gate = deferred<Response>();
    backend.route("POST", `${BASE}/execute`, () => gate.promise);
    const { result } = setup();
    await startHeld(result);
    expect(result.current.run?.phase).toBe("starting");
    act(() => result.current.releaseHold("deadman released"));
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(0);

    server = { ...server, status: "executing" };
    await act(async () => {
      gate.resolve(jsonResponse(server, 202));
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(0);
    expect(result.current.notice).toMatch(/released before the run started/);
  });

  it("cannot resume a released run by pressing again", async () => {
    installAgentRoutes(held());
    const gate = deferred<Response>();
    const { result } = setup();
    await startHeld(result);
    backend.route("POST", `${BASE}/cancel`, () => gate.promise);
    act(() => result.current.releaseHold("deadman released"));
    expect(result.current.run?.phase).toBe("cancelling");
    act(() => result.current.pressHold(server));
    expect(result.current.holding).toBe(false);
    await advance(300);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(1);
    expect(backend.callsTo(`${BASE}/execute`)).toHaveLength(1);
    gate.resolve(jsonResponse({ ...server, status: "cancelled" }));
  });

  it("cancels when a hold ping fails while the run is live", async () => {
    installAgentRoutes(held());
    const { result } = setup();
    await startHeld(result);
    backend.route("POST", `${BASE}/hold`, () => jsonResponse({ detail: "hold rejected" }, 500));
    await advance(100);
    await advance(0);
    expect(result.current.error).toBe("The hold-to-run ping failed: hold rejected. The run is being cancelled.");
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
  });

  it("sends the first hold before the poll and the refreshes", async () => {
    installAgentRoutes(held());
    const order: string[] = [];
    backend.route("POST", `${BASE}/hold`, () => {
      order.push("hold");
      return jsonResponse({ ok: true, detail: "holding" });
    });
    backend.route("GET", BASE, () => {
      order.push("poll");
      return jsonResponse(server);
    });
    const { result } = setup({ onChanged: () => order.push("refresh") });
    await call(() => result.current.approve(server));
    order.length = 0;
    await act(async () => {
      result.current.pressHold(server);
      await vi.advanceTimersByTimeAsync(0);
    });
    await advance(250);
    expect(order[0]).toBe("hold");
    expect(order.indexOf("refresh")).toBeGreaterThan(0);
    expect(order.indexOf("poll")).toBeGreaterThan(order.indexOf("refresh"));
  });

  it("keeps pinging every 100 ms while one hold is slow, up to three in flight", async () => {
    installAgentRoutes(held());
    const pending: ReturnType<typeof deferred<Response>>[] = [];
    backend.route("POST", `${BASE}/hold`, () => {
      const next = deferred<Response>();
      pending.push(next);
      return next.promise;
    });
    const { result } = setup();
    await startHeld(result);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(1);
    await advance(100);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(2);
    await advance(100);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(MAX_HOLDS_IN_FLIGHT);
    await advance(300);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(MAX_HOLDS_IN_FLIGHT);

    // One answer frees one slot; the next tick uses it.
    await act(async () => {
      pending[0].resolve(jsonResponse({ ok: true, detail: "holding" }));
      await vi.advanceTimersByTimeAsync(0);
    });
    await advance(100);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(MAX_HOLDS_IN_FLIGHT + 1);
    expect(result.current.error).toBeNull();
    for (const entry of pending) entry.resolve(jsonResponse({ ok: true, detail: "holding" }));
  });

  it("acts on the first failed hold only, with one cancel", async () => {
    installAgentRoutes(held());
    const pending: ReturnType<typeof deferred<Response>>[] = [];
    backend.route("POST", `${BASE}/hold`, () => {
      const next = deferred<Response>();
      pending.push(next);
      return next.promise;
    });
    const { result } = setup();
    await startHeld(result);
    await advance(200);
    expect(pending).toHaveLength(3);
    await act(async () => {
      for (const entry of pending) {
        entry.resolve(jsonResponse({ detail: "deadman lapsed; the skill is stopping" }, 409));
      }
      await vi.advanceTimersByTimeAsync(0);
    });
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(result.current.error).toBe(
      "The hold-to-run ping failed: deadman lapsed; the skill is stopping. The run is being cancelled.",
    );
  });

  it("does not treat a refused hold after the run ended as a failure", async () => {
    installAgentRoutes(held());
    const { result } = setup();
    await startHeld(result);
    server = { ...server, status: "executed", result: RESULT };
    backend.route("POST", `${BASE}/hold`, () => jsonResponse({ detail: "proposal is not executing" }, 409));
    await advance(100);
    await advance(0);
    expect(result.current.error).toBeNull();
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(0);
    expect(result.current.latest?.status).toBe("executed");
    expect(result.current.run?.phase).toBe("ended");
  });

  it("stops holding on an E-stop, does not cancel on release, and keeps polling", async () => {
    installAgentRoutes(held());
    const { result, rerender } = setup();
    await startHeld(result);
    rerender({ tripped: true });
    const holds = backend.callsTo(`${BASE}/hold`).length;
    await advance(300);
    expect(backend.callsTo(`${BASE}/hold`)).toHaveLength(holds);

    act(() => result.current.releaseHold("deadman released"));
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(0);

    server = { ...server, status: "failed", result: { ...RESULT, reached: false, end_state: "estopped" } };
    await advance(250);
    expect(result.current.latest?.result?.end_state).toBe("estopped");
    expect(result.current.run?.phase).toBe("ended");
  });

  it("explains a refused cancel unless the run had already ended", async () => {
    installAgentRoutes(held());
    const { result } = setup();
    await startHeld(result);
    backend.route("POST", `${BASE}/cancel`, () => jsonResponse({ detail: "cancel rejected" }, 500));
    act(() => result.current.releaseHold("deadman released"));
    await advance(0);
    expect(result.current.error).toBe("Cancel failed: cancel rejected. Use Halt or E-stop if the arm is moving.");
  });

  it("finishes quietly when the cancel lost the race to the result", async () => {
    installAgentRoutes(held());
    const { result } = setup();
    await startHeld(result);
    server = { ...server, status: "executed", result: RESULT };
    backend.route("POST", `${BASE}/cancel`, () => jsonResponse({ detail: "proposal already executed" }, 409));
    act(() => result.current.releaseHold("deadman released"));
    await advance(0);
    expect(result.current.error).toBeNull();
    expect(result.current.latest?.status).toBe("executed");
  });
});

describe("useAgentExecution: leaving cancels", () => {
  async function running(overrides: Partial<UseAgentExecutionOptions> = {}) {
    installAgentRoutes(proposal());
    const hook = setup(overrides);
    await call(() => hook.result.current.approve(server));
    await call(() => hook.result.current.execute(server));
    return hook;
  }

  it("cancels when the lease is lost", async () => {
    const { result, rerender } = await running();
    rerender({ leaseId: null });
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(result.current.notice).toMatch(/the lease was lost/);
  });

  it("cancels when the tab is hidden", async () => {
    const { result } = await running();
    visibility = "hidden";
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(result.current.notice).toMatch(/the tab was hidden/);
  });

  it("cancels on unmount, once", async () => {
    const { unmount } = await running();
    unmount();
    await advance(0);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
  });

  it("cancels a proposal that is not running here without touching a run", async () => {
    installAgentRoutes(proposal());
    const { result } = setup();
    await call(() => result.current.cancelProposal(server));
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(result.current.notice).toBe("Proposal cancelled.");
    expect(result.current.run).toBeNull();
  });
});

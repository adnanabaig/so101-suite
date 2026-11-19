import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { AgentProposal } from "../api";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import { PRE_C8_MESSAGE } from "../hooks/useAgentExecution";
import { TelemetryProvider } from "../hooks/useTelemetry";
import { targetDigest } from "../lib/canonicalJson";
import {
  FakeWebSocket,
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  telemetryFrame,
  type FakeBackend,
} from "../test/harness";
import AgentLab, { OLD_SERVER_DEADMAN_HINT } from "./AgentLab";

const TARGETS = { gripper: 80, wrist_roll: 12.5 };
const BASE = "/api/agent/proposals/p1";
const primary = { isPrimary: true, button: 0, pointerId: 1 };

function proposal(overrides: Partial<AgentProposal> = {}): AgentProposal {
  return {
    id: "p1",
    title: "Open the gripper",
    status: "proposed",
    risk: "low",
    skill: "open_gripper",
    targets: TARGETS,
    target_digest: targetDigest(TARGETS),
    constraints: { max_duration_seconds: 10, require_deadman: true, allow_motion: true },
    steps: ["Open the gripper to 80%"],
    ...overrides,
  };
}

let backend: FakeBackend;
let server: AgentProposal;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  server = proposal();
  backend.route("POST", "/api/control/lease", () =>
    jsonResponse({ id: "lease-1", source: "agent-lab", server_time: new Date().toISOString() }),
  );
  backend.route("POST", "/api/control/lease/lease-1/heartbeat", () =>
    jsonResponse({ id: "lease-1", source: "agent-lab", server_time: new Date().toISOString() }),
  );
  backend.route("DELETE", "/api/control/lease/lease-1", () => jsonResponse({ ok: true, detail: "" }));
  backend.route("GET", "/api/agent/proposals", () => jsonResponse({ proposals: [server] }));
  backend.route("GET", BASE, () => jsonResponse(server));
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
    server = { ...server, status: "cancelled", result: { reached: false, end_state: "halted", steps: 3 } };
    return jsonResponse(server);
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderLab() {
  const view = render(
    <TelemetryProvider>
      <SafetyProvider>
        <AgentLab />
      </SafetyProvider>
    </TelemetryProvider>,
  );
  await advance(10);
  act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
  await advance(10);
  return view;
}

async function acquireLease() {
  fireEvent.click(screen.getByRole("button", { name: "Acquire the Agent Lab lease" }));
  await advance(10);
}

function card() {
  return screen.getByRole("article", { name: "Open the gripper" });
}

function robotContextTile() {
  const label = screen.getByText("Robot context");
  return label.closest("div.rounded-2xl") as HTMLElement;
}

async function approve() {
  fireEvent.click(within(card()).getByRole("checkbox"));
  fireEvent.click(within(card()).getByRole("button", { name: "Approve these targets" }));
  await advance(10);
}

describe("Agent Lab (C8)", () => {
  test("approve, hold to run, and show the result with an unknown end state verbatim", async () => {
    await renderLab();
    await acquireLease();
    expect(within(robotContextTile()).getByText("Ready")).toBeInTheDocument();
    expect(within(card()).getByText(/Targets identified by digest/)).toBeInTheDocument();

    await approve();
    expect(backend.callsTo(`${BASE}/approve`)[0].body).toMatchObject({
      lease_id: "lease-1",
      target_digest: targetDigest(TARGETS),
    });
    expect(within(card()).getByRole("status")).toHaveTextContent(/Approved for this tab's lease/);

    const deadman = within(card()).getByRole("button", { name: /Hold to run/ });
    fireEvent.pointerDown(deadman, primary);
    await advance(10);
    expect(backend.callsTo(`${BASE}/execute`)[0].body).toEqual({ lease_id: "lease-1", approval_token: "tok-1" });
    expect(deadman).toHaveAttribute("aria-pressed", "true");
    expect(deadman).toHaveTextContent("Running · keep holding");
    expect(within(robotContextTile()).getByText("Running")).toBeInTheDocument();
    await advance(200);
    expect(backend.callsTo(`${BASE}/hold`).length).toBeGreaterThanOrEqual(3);

    server = {
      ...server,
      status: "executed",
      result: {
        reached: true,
        final_error_deg: { gripper: 0.4, wrist_roll: 0.25 },
        max_abs_error_deg: 0.4,
        duration_s: 1.2,
        steps: 12,
        end_state: "parked_on_rest",
      },
    };
    await advance(250);
    const result = within(card()).getByRole("region", { name: "Run result" });
    expect(result).toHaveTextContent("Reached target");
    expect(result).toHaveTextContent("parked_on_rest");
    expect(result).toHaveTextContent("1.2 s");
    expect(result).toHaveTextContent("0.40 gripper units");
    expect(result).toHaveTextContent("0.25°");
    // The run ended on its own: letting go afterwards cancels nothing.
    fireEvent.pointerUp(window);
    await advance(10);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(0);
  });

  test("letting go of hold-to-run cancels the run and reports the halt", async () => {
    await renderLab();
    await acquireLease();
    await approve();
    fireEvent.pointerDown(within(card()).getByRole("button", { name: /Hold to run/ }), primary);
    await advance(10);
    fireEvent.pointerUp(window);
    await advance(10);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
    expect(screen.getByText(/Run cancelled \(deadman released\)/)).toBeInTheDocument();
    const result = within(card()).getByRole("region", { name: "Run result" });
    expect(result).toHaveTextContent("Halted · holding position, motors on");
    expect(result).toHaveTextContent(/holding position with its motors on/);
  });

  test("a non-hold proposal executes on a click and can be cancelled", async () => {
    server = proposal({ constraints: { max_duration_seconds: 10, require_deadman: false, allow_motion: true } });
    await renderLab();
    await acquireLease();
    await approve();
    fireEvent.click(within(card()).getByRole("button", { name: "Execute approved targets" }));
    await advance(10);
    expect(backend.callsTo(`${BASE}/execute`)).toHaveLength(1);
    fireEvent.click(within(card()).getByRole("button", { name: "Cancel run" }));
    await advance(10);
    expect(backend.callsTo(`${BASE}/cancel`)).toHaveLength(1);
  });

  test("a digest mismatch is shown and approval stays blocked", async () => {
    server = proposal({ target_digest: "0".repeat(64) });
    await renderLab();
    await acquireLease();
    expect(within(card()).getByRole("alert")).toHaveTextContent(/Target digest mismatch/);
    fireEvent.click(within(card()).getByRole("checkbox"));
    expect(within(card()).getByRole("button", { name: "Approve these targets" })).toBeDisabled();
  });

  test("a server without C8 approval explains itself", async () => {
    backend.route("POST", `${BASE}/approve`, () => jsonResponse({ detail: "Not Found" }, 404));
    await renderLab();
    await acquireLease();
    await approve();
    expect(screen.getByText(PRE_C8_MESSAGE)).toBeInTheDocument();
    expect(within(card()).queryByRole("button", { name: /Hold to run/ })).toBeNull();
  });

  test("the Robot context tile and the approval gate give the same reason (F9)", async () => {
    await renderLab();
    expect(within(robotContextTile()).getByText("Blocked")).toBeInTheDocument();
    expect(within(robotContextTile()).getByText("Acquire the Agent Lab lease first.")).toBeInTheDocument();
    expect(within(card()).getByText(/Approval is blocked/).parentElement).toHaveTextContent(
      "Acquire the Agent Lab lease first.",
    );
    fireEvent.click(within(card()).getByRole("checkbox"));
    expect(within(card()).getByRole("button", { name: "Approve these targets" })).toBeDisabled();

    await acquireLease();
    expect(within(robotContextTile()).getByText("Approval and execution are allowed")).toBeInTheDocument();
    expect(within(card()).getByRole("button", { name: "Approve these targets" })).toBeEnabled();

    // A latched E-stop leads both lists.
    backend.estopped = true;
    act(() => FakeWebSocket.broadcast(telemetryFrame(true)));
    await advance(1600);
    expect(within(robotContextTile()).getByText(/^E-stop is latched/)).toBeInTheDocument();
    expect(within(card()).getByRole("button", { name: "Approve these targets" })).toBeDisabled();
  });

  test("targets read in degrees, the gripper in gripper units", async () => {
    await renderLab();
    expect(within(card()).getByText("80.00 gripper units")).toBeInTheDocument();
    expect(within(card()).getByText("12.50°")).toBeInTheDocument();
  });

  test("shows the review window left before the server's 120 s proposal limit", async () => {
    server = proposal({ created_at: new Date(Date.now() - 100_000).toISOString() });
    await renderLab();
    await acquireLease();
    expect(within(card()).getByText(/^Review window: about 20 s left/)).toBeInTheDocument();
    await advance(21_000);
    expect(within(card()).getByText(/older than 2 minutes, so the server will refuse it/)).toBeInTheDocument();
  });

  test("an executing proposal shows that it runs, not a half-empty result", async () => {
    server = proposal({ status: "executing", result: { started_at: "2026-10-02T12:00:00.000000+00:00" } });
    await renderLab();
    expect(within(card()).queryByRole("region", { name: "Run result" })).toBeNull();
    const running = within(card()).getByText(/Executing on the server/);
    expect(running.querySelector("time")).toHaveAttribute("dateTime", "2026-10-02T12:00:00.000Z");
  });

  test("leaving the page cancels an active run", async () => {
    const view = await renderLab();
    await acquireLease();
    await approve();
    fireEvent.pointerDown(within(card()).getByRole("button", { name: /Hold to run/ }), primary);
    await advance(10);
    view.unmount();
    await advance(10);
    expect(backend.callsTo(`${BASE}/cancel`).length).toBeGreaterThanOrEqual(1);
  });
});

describe("Agent Lab results as gate reports them (C8)", () => {
  function resultRegion() {
    return within(card()).getByRole("region", { name: "Run result" });
  }

  test("a timeout: the reason, the server's words, errors against tolerance, a limited target", async () => {
    server = proposal({
      status: "timeout",
      result: {
        reached: false,
        final_error_deg: { gripper: 4.2, wrist_roll: 0.5 },
        max_abs_error_deg: 4.2,
        duration_s: 10.02,
        steps: 300,
        end_state: "deenergized",
        status: "timeout",
        stop_reason: "max_duration",
        error: "targets not reached within 10 s",
        targets: { gripper: 75, wrist_roll: 12.5 },
        tolerance: { arm_deg: 2, gripper: 3 },
        started_at: "2026-10-02T12:00:00.000000+00:00",
        finished_at: "2026-10-02T12:00:10.020000+00:00",
      },
    });
    await renderLab();
    const result = resultRegion();
    expect(result).toHaveTextContent("Result · timeout");
    expect(result).toHaveTextContent("Did not reach target");
    expect(result).toHaveTextContent("Why it ended: Ran out of time before reaching the target.");
    expect(result).toHaveTextContent("Server: targets not reached within 10 s");
    expect(result).toHaveTextContent("tolerance ±2° per arm joint, ±3 gripper units");
    expect(result).toHaveTextContent("Largest final error4.20 gripper units (gripper)");
    const gripper = within(result).getByText("gripper").parentElement as HTMLElement;
    expect(gripper).toHaveTextContent("4.20 gripper units (outside tolerance)");
    const wrist = within(result).getByText("wrist_roll").parentElement as HTMLElement;
    expect(wrist).toHaveTextContent("0.50°");
    expect(wrist).not.toHaveTextContent("outside tolerance");
    expect(result).toHaveTextContent(
      "The server limited this target to the arm's calibrated range before running: gripper 80.00 gripper units → 75.00 gripper units.",
    );
    const times = result.querySelectorAll("time");
    expect(times[0]).toHaveAttribute("dateTime", "2026-10-02T12:00:00.000Z");
    expect(times[1]).toHaveAttribute("dateTime", "2026-10-02T12:00:10.020Z");
    expect(result).toHaveTextContent("Motors off");
    expect(result).toHaveTextContent("The motors are off, so the arm is not held in place.");
  });

  test("an unknown end state warns to check the arm and keeps the server's words", async () => {
    server = proposal({
      status: "cancelled",
      result: {
        reached: false,
        final_error_deg: {},
        max_abs_error_deg: null,
        duration_s: null,
        steps: null,
        end_state: "unknown",
        status: "cancelled",
        stop_reason: "cancelled",
        error: "cancelled with no live run",
      },
    });
    await renderLab();
    const result = resultRegion();
    expect(result).toHaveTextContent("Motor state unknown");
    expect(result).toHaveTextContent("Server: cancelled with no live run");
    expect(result).toHaveTextContent(/could not confirm whether the motors are on/);
  });

  test("a restart-interrupted run reads as such", async () => {
    server = proposal({
      status: "failed",
      result: {
        reached: false,
        end_state: "unknown",
        stop_reason: "restart",
        error: "interrupted by restart",
        started_at: "2026-10-02T12:00:00+00:00",
        finished_at: "2026-10-02T12:05:00+00:00",
      },
    });
    await renderLab();
    expect(resultRegion()).toHaveTextContent("Why it ended: Server restarted mid-run.");
    expect(resultRegion()).toHaveTextContent("Server: interrupted by restart");
  });

  test("an expired proposal says nothing moved and how old it was", async () => {
    server = proposal({ status: "expired", result: { age_seconds: 131.2 } });
    await renderLab();
    expect(resultRegion()).toHaveTextContent(
      "Expired before it ran; nothing moved. The server approves and runs a proposal only within 2 minutes of its creation (this one was 131 s old).",
    );
    expect(within(card()).queryByRole("button", { name: "Approve these targets" })).toBeNull();
  });

  test("a proposal from before a server restart says why it expired", async () => {
    // Gate: approve marks it expired with age_seconds null (age unknown).
    server = proposal({ status: "expired", result: { age_seconds: null } });
    await renderLab();
    expect(resultRegion()).toHaveTextContent(
      "Expired before it ran; nothing moved. The server approves and runs a proposal only within 2 minutes of its creation. This one was created before the server restarted, and a restart forgets proposal ages, so it cannot run. Create a fresh proposal to try again.",
    );
    expect(resultRegion()).not.toHaveTextContent("s old");
  });

  test("a proposal cancelled before it ran says nothing moved", async () => {
    server = proposal({ status: "cancelled", result: { cancelled: true, stop_reason: "cancelled" } });
    await renderLab();
    expect(resultRegion()).toHaveTextContent("Cancelled before it ran. Nothing moved.");
    expect(resultRegion()).not.toHaveTextContent("Did not reach target");
  });
});

describe("Agent Lab proposal form", () => {
  test("motion proposals ask for hold-to-run by default", async () => {
    backend.route("POST", "/api/agent/proposals", (call) => jsonResponse({ ...proposal({ id: "p2" }), ...(call.body as object) }));
    await renderLab();
    fireEvent.click(screen.getByRole("button", { name: /Open gripper/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Allow the planner/ }));
    fireEvent.change(screen.getByLabelText("Objective"), { target: { value: "Open up" } });
    fireEvent.click(screen.getByRole("button", { name: "Create proposal" }));
    await advance(10);
    expect(backend.callsTo("/api/agent/proposals")[0].body).toEqual({
      objective: "Open up",
      skill: "open_gripper",
      planner: "deterministic",
      constraints: { max_duration_seconds: 30, require_deadman: true, allow_motion: true },
    });
  });

  test("an older server's deadman refusal gets an explanation", async () => {
    backend.route("POST", "/api/agent/proposals", () =>
      jsonResponse({ detail: "continuous agent deadman is not implemented" }, 422),
    );
    await renderLab();
    fireEvent.click(screen.getByRole("button", { name: /Open gripper/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Allow the planner/ }));
    fireEvent.change(screen.getByLabelText("Objective"), { target: { value: "Open up" } });
    fireEvent.click(screen.getByRole("button", { name: "Create proposal" }));
    await advance(10);
    expect(screen.getByText(new RegExp(OLD_SERVER_DEADMAN_HINT.slice(0, 40)))).toBeInTheDocument();
  });

  test("offers gate's home skill", async () => {
    backend.route("POST", "/api/agent/proposals", (call) => jsonResponse({ ...proposal({ id: "p4" }), ...(call.body as object) }));
    await renderLab();
    fireEvent.click(screen.getByRole("button", { name: /Home pose/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Allow the planner/ }));
    fireEvent.change(screen.getByLabelText("Objective"), { target: { value: "Go home" } });
    fireEvent.click(screen.getByRole("button", { name: "Create proposal" }));
    await advance(10);
    expect(backend.callsTo("/api/agent/proposals")[0].body).toMatchObject({ skill: "home" });
  });

  test("the objective is capped at gate's 500 characters", async () => {
    await renderLab();
    const objective = screen.getByLabelText("Objective");
    expect(objective).toHaveAttribute("maxLength", "500");
    fireEvent.change(objective, { target: { value: "Open up" } });
    expect(screen.getByText("7/500 characters")).toBeInTheDocument();
  });

  test("an unavailable model planner says why, verbatim", async () => {
    backend.route("GET", "/api/agent/proposals/planner/status", () =>
      jsonResponse({
        default_provider: "deterministic",
        configured_provider_available: false,
        configured_provider: "none",
        unavailable_reason: "SO101_OPENAI_API_KEY is not configured",
        model: "",
        bounded_skills: ["inspect_scene", "home", "open_gripper", "close_gripper", "wrist_wave"],
        motion_authority: "none; proposals require separate human approval",
      }),
    );
    await renderLab();
    expect(screen.getByRole("option", { name: "Model planner unavailable" })).toBeDisabled();
    expect(
      screen.getByText(
        "Model planner unavailable: SO101_OPENAI_API_KEY is not configured. The deterministic router still works.",
      ),
    ).toBeInTheDocument();
  });

  test.each([
    [502, "planner returned an unknown skill 'fly'", "The planner answered with something unusable: planner returned an unknown skill 'fly'"],
    [503, "OpenAI planner request failed: timed out", "Planner unavailable: OpenAI planner request failed: timed out"],
  ])("a model planner %i is explained with the server's detail", async (status, detail, shown) => {
    backend.route("GET", "/api/agent/proposals/planner/status", () =>
      jsonResponse({
        default_provider: "openai_responses",
        configured_provider_available: true,
        configured_provider: "openai_responses",
        unavailable_reason: null,
        model: "gpt-test",
        bounded_skills: [],
        motion_authority: "none",
      }),
    );
    backend.route("POST", "/api/agent/proposals", () => jsonResponse({ detail }, status));
    await renderLab();
    fireEvent.change(screen.getByLabelText("Planner"), { target: { value: "configured" } });
    fireEvent.change(screen.getByLabelText("Objective"), { target: { value: "Tidy up" } });
    fireEvent.click(screen.getByRole("button", { name: "Create proposal" }));
    await advance(10);
    expect(backend.callsTo("/api/agent/proposals")[0].body).toMatchObject({ planner: "configured" });
    expect(screen.getByText("Proposal was not created").parentElement).toHaveTextContent(shown);
  });

  test("an observation-only skill never sends a motion grant", async () => {
    backend.route("POST", "/api/agent/proposals", () => jsonResponse(proposal({ id: "p3", targets: {} })));
    await renderLab();
    fireEvent.change(screen.getByLabelText("Objective"), { target: { value: "Look around" } });
    fireEvent.click(screen.getByRole("button", { name: "Create proposal" }));
    await advance(10);
    expect(backend.callsTo("/api/agent/proposals")[0].body).toMatchObject({
      skill: "inspect_scene",
      constraints: { require_deadman: false, allow_motion: false },
    });
  });
});

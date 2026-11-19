import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { EvalState, ModelSummary } from "../api";
import { SafetyProvider } from "../components/safety/SafetyProvider";
import { TelemetryProvider } from "../hooks/useTelemetry";
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
import Eval, {
  EVAL_MOVED_ON_HINT,
  EVAL_OWNER_HINT,
  EVAL_START_TIMEOUT_HINT,
  evalRefusalHint,
} from "./Eval";

const SMOL: ModelSummary = {
  name: "smol",
  path: "/models/smol",
  policy_type: "smolvla",
  is_mock: false,
  checkpoint: "last",
  needs_task: true,
  task: "Pick up the cube",
  fps: 30,
};

const IDLE: EvalState = {
  active: false,
  model_path: "",
  episode: 0,
  n_episodes: 0,
  phase: "idle",
  successes: 0,
  success_rate: 0,
};

let backend: FakeBackend;
let evalState: EvalState;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  evalState = IDLE;
  backend.route("GET", "/api/models", () => jsonResponse([SMOL]));
  backend.route("GET", "/api/eval/state", () => jsonResponse(evalState));
  backend.route("POST", "/api/eval/start", () => jsonResponse({ ok: true, detail: "started" }));
  backend.route("POST", "/api/eval/stop", () => jsonResponse({ ok: true, detail: "stopped" }));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderEval() {
  render(
    <TelemetryProvider>
      <SafetyProvider>
        <Eval />
      </SafetyProvider>
    </TelemetryProvider>,
  );
  await advance(10);
  act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
  await advance(10);
}

async function startFromDialog() {
  fireEvent.click(screen.getByRole("button", { name: "Start eval" }));
  await advance(10);
  const dialog = screen.getByRole("alertdialog", { name: "Start autonomous evaluation?" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Start evaluation" }));
  await advance(10);
}

function blockerText(): string {
  return screen.getByText(/Start is blocked/).parentElement?.textContent ?? "";
}

describe("Eval start", () => {
  test("prefills checkpoint fps and task and sends an edited task override", async () => {
    await renderEval();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: SMOL.path } });
    expect(screen.getByLabelText("FPS (optional)")).toHaveValue(30);
    expect(screen.getByLabelText("Task (optional)")).toHaveValue("Pick up the cube");
    fireEvent.change(screen.getByLabelText("Task (optional)"), {
      target: { value: "Pick up the red cube" },
    });

    await startFromDialog();
    expect(backend.callsTo("/api/eval/start")[0].body).toEqual({
      model_path: SMOL.path,
      n_episodes: 10,
      max_seconds: 30,
      fps: 30,
      task: "Pick up the red cube",
      operator_id: expect.any(String),
    });
  });

  test("shows a refused start's detail exactly as the server sent it", async () => {
    const detail = "could not load policy from '/models/smol': missing config.json";
    backend.route("POST", "/api/eval/start", () => jsonResponse({ detail }, 422));
    await renderEval();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: SMOL.path } });
    await startFromDialog();
    expect(screen.getByText("Evaluation request failed").parentElement).toHaveTextContent(detail);

    backend.route("POST", "/api/eval/start", () =>
      jsonResponse({ detail: "a fresh owning operator heartbeat is required" }, 409),
    );
    await startFromDialog();
    const notice = screen.getByText("Evaluation request failed").parentElement;
    expect(notice).toHaveTextContent("a fresh owning operator heartbeat is required");
    // The owner hint is for score and continue only.
    expect(notice).not.toHaveTextContent(EVAL_OWNER_HINT);
  });

  test("a 422 validation list shows each field and message, not a generic failure", async () => {
    backend.route("POST", "/api/eval/start", () =>
      jsonResponse(
        {
          detail: [
            {
              type: "less_than_equal",
              loc: ["body", "max_seconds"],
              msg: "Input should be less than or equal to 3600",
              input: 3601,
              ctx: { le: 3600 },
            },
            { type: "missing", loc: ["body", "operator_id"], msg: "Field required" },
          ],
        },
        422,
      ),
    );
    await renderEval();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: SMOL.path } });
    await startFromDialog();
    const notice = screen.getByText("Evaluation request failed").parentElement;
    expect(notice).toHaveTextContent(
      "max_seconds: Input should be less than or equal to 3600; operator_id: Field required",
    );
    expect(notice).not.toHaveTextContent("failed: 422");
  });

  test("a start that outlives the client timeout says the eval may still begin", async () => {
    backend.route("POST", "/api/eval/start", () =>
      jsonResponse({ detail: "POST /api/eval/start timed out" }, 408),
    );
    await renderEval();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: SMOL.path } });
    await startFromDialog();
    expect(screen.getByText("Evaluation request failed").parentElement).toHaveTextContent(
      `POST /api/eval/start timed out. ${EVAL_START_TIMEOUT_HINT}`,
    );
  });

  test("bounds Max seconds the way StartEvalRequest does: above 0, at most 3600, finite", async () => {
    await renderEval();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: SMOL.path } });
    const seconds = screen.getByLabelText("Max seconds / episode");
    expect(seconds).toHaveAttribute("max", "3600");
    expect(seconds).toHaveAttribute("min", "0");

    for (const refused of ["3600.5", "3601", "0", "-1", ""]) {
      fireEvent.change(seconds, { target: { value: refused } });
      expect(seconds).toHaveAttribute("aria-invalid", "true");
      expect(blockerText()).toContain("Seconds per episode must be above 0 and at most 3600.");
      expect(screen.getByRole("button", { name: "Start eval" })).toBeDisabled();
    }

    fireEvent.change(seconds, { target: { value: "0.5" } });
    expect(seconds).not.toHaveAttribute("aria-invalid");
    fireEvent.change(seconds, { target: { value: "3600" } });
    expect(seconds).not.toHaveAttribute("aria-invalid");
    await startFromDialog();
    expect(backend.callsTo("/api/eval/start")).toHaveLength(1);
    expect(backend.callsTo("/api/eval/start")[0].body).toMatchObject({ max_seconds: 3600 });
  });

  test("lists every blocker, safety first, and validates the fields", async () => {
    await renderEval();
    expect(screen.getByRole("button", { name: "Start eval" })).toBeDisabled();
    expect(blockerText()).toContain("Choose a model.");

    fireEvent.change(screen.getByLabelText("Model"), { target: { value: SMOL.path } });
    fireEvent.change(screen.getByLabelText("FPS (optional)"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("Episodes"), { target: { value: "2.5" } });
    expect(blockerText()).toMatch(/Episodes must be a whole number/);
    expect(blockerText()).toMatch(/FPS must be a whole number from 1 to 240/);
    expect(screen.getByLabelText("FPS (optional)")).toHaveAttribute("aria-invalid", "true");

    backend.estopped = true;
    act(() => FakeWebSocket.broadcast(telemetryFrame(true)));
    await advance(1600);
    const reasons = blockerText();
    expect(reasons).toMatch(/^Start is blocked: E-stop is latched/);
    expect(reasons.indexOf("E-stop is latched")).toBeLessThan(reasons.indexOf("Episodes must"));
  });
});

describe("Eval scoring (C3)", () => {
  const AWAITING: EvalState = {
    active: true,
    model_path: SMOL.path,
    episode: 1,
    n_episodes: 3,
    phase: "awaiting_score",
    successes: 0,
    success_rate: 0,
  };

  test("posts the score as JSON and shows the state the server answered with", async () => {
    evalState = AWAITING;
    backend.route("POST", "/api/eval/score", () =>
      jsonResponse({ ...AWAITING, phase: "awaiting_reset", successes: 1, success_rate: 1 }),
    );
    await renderEval();
    fireEvent.click(screen.getByRole("button", { name: "Success" }));
    await advance(10);
    expect(backend.callsTo("/api/eval/score")[0].body).toEqual({
      operator_id: expect.any(String),
      success: true,
    });
    // Adopted from the response, before the next state poll.
    expect(screen.getByText("Reset the physical workspace before episode 2")).toBeInTheDocument();

    backend.route("POST", "/api/eval/continue", () =>
      jsonResponse({ ...AWAITING, phase: "rollout", episode: 2, successes: 1, success_rate: 1 }),
    );
    fireEvent.click(screen.getByRole("button", { name: /Workspace reset/ }));
    await advance(10);
    expect(backend.callsTo("/api/eval/continue")[0].body).toEqual({ operator_id: expect.any(String) });
  });

  test("a refused continue shows the server's detail first, then who may continue", async () => {
    evalState = { ...AWAITING, phase: "awaiting_reset" };
    const detail = "only the operator who started this eval may score or continue it";
    backend.route("POST", "/api/eval/continue", () => jsonResponse({ detail }, 409));
    await renderEval();
    fireEvent.click(screen.getByRole("button", { name: /Workspace reset/ }));
    await advance(10);
    expect(screen.getByText("Evaluation request failed").parentElement).toHaveTextContent(
      `${detail}. ${EVAL_OWNER_HINT}`,
    );
  });

  test("a score after the trial was recorded interrupted says the session moved on, not who may score", async () => {
    evalState = AWAITING;
    backend.route("POST", "/api/eval/score", () => {
      // The session thread decided the trial unscored a moment earlier.
      evalState = { ...AWAITING, active: false, phase: "aborted", error: "trial was not scored" };
      return jsonResponse({ detail: "not awaiting a score" }, 409);
    });
    await renderEval();
    const polls = backend.callsTo("/api/eval/state", "GET").length;
    fireEvent.click(screen.getByRole("button", { name: "Success" }));
    await advance(10);
    const notice = screen.getByText("Evaluation request failed").parentElement;
    expect(notice).toHaveTextContent(`not awaiting a score. ${EVAL_MOVED_ON_HINT}`);
    expect(notice).not.toHaveTextContent(EVAL_OWNER_HINT);
    // Re-polled at once, so the score buttons go away with the stale phase.
    expect(backend.callsTo("/api/eval/state", "GET").length).toBeGreaterThan(polls);
    expect(screen.queryByRole("button", { name: "Success" })).toBeNull();
  });

  test("explains who may score when the server refuses with 409", async () => {
    evalState = AWAITING;
    backend.route("POST", "/api/eval/score", () =>
      jsonResponse({ detail: "only the session operator may score" }, 409),
    );
    await renderEval();
    fireEvent.click(screen.getByRole("button", { name: "Fail" }));
    await advance(10);
    expect(screen.getByText(new RegExp(EVAL_OWNER_HINT.slice(0, 30)))).toHaveTextContent(
      "only the session operator may score",
    );
  });
});

describe("evalRefusalHint", () => {
  test("picks the hint from the server's 409 detail", () => {
    expect(evalRefusalHint("only the operator who started this eval may score or continue it")).toBe(
      EVAL_OWNER_HINT,
    );
    expect(evalRefusalHint("a fresh owning operator heartbeat is required")).toBe(EVAL_OWNER_HINT);
    expect(evalRefusalHint("not awaiting a score")).toBe(EVAL_MOVED_ON_HINT);
    expect(evalRefusalHint("not awaiting a workspace reset")).toBe(EVAL_MOVED_ON_HINT);
    expect(evalRefusalHint("episode was already scored")).toBeNull();
    expect(evalRefusalHint("cannot continue: E-stop is engaged")).toBeNull();
  });
});

describe("Eval terminal summary (C3)", () => {
  const ENDED: EvalState = {
    active: false,
    model_path: SMOL.path,
    episode: 3,
    n_episodes: 5,
    phase: "aborted",
    successes: 1,
    success_rate: 0.5,
    report_path: "/outputs/eval_1700000000000_abc123def456.json",
    scored: 2,
    interrupted: 1,
    task: null,
    fps: 30,
    error: "E-stop engaged",
  };

  test("an aborted session shows its cause verbatim, the scored and interrupted counts, and the report", async () => {
    evalState = ENDED;
    await renderEval();
    const summary = screen.getByTestId("eval-summary");
    expect(screen.getByText("Evaluation aborted")).toBeInTheDocument();
    expect(summary).toHaveTextContent("Reason: E-stop engaged");
    expect(summary).toHaveTextContent("2 of 5 requested episodes scored · 1 interrupted");
    expect(summary).toHaveTextContent("1 succeeded: 50.0% of the scored episodes");
    expect(summary).toHaveTextContent("Ran at 30 fps.");
    expect(summary).toHaveTextContent(`Report saved to ${ENDED.report_path}`);
    expect(summary).toHaveTextContent(/does not resume an evaluation/);
    expect(screen.getByText("aborted")).toBeInTheDocument();
  });

  test("a failed session names the server's error; a finished one has no abort note", async () => {
    evalState = { ...ENDED, phase: "failed", error: "rollout thread crashed: bus timeout" };
    await renderEval();
    expect(screen.getByText("Evaluation failed")).toBeInTheDocument();
    expect(screen.getByTestId("eval-summary")).toHaveTextContent("Reason: rollout thread crashed: bus timeout");

    evalState = { ...ENDED, phase: "done", error: "", scored: 5, interrupted: 0, successes: 4, success_rate: 0.8 };
    await advance(1100);
    expect(screen.getByText("Evaluation finished")).toBeInTheDocument();
    const summary = screen.getByTestId("eval-summary");
    expect(summary).not.toHaveTextContent("Reason:");
    expect(summary).toHaveTextContent("5 of 5 requested episodes scored · 0 interrupted");
    expect(summary).not.toHaveTextContent(/does not resume/);
  });

  test.each(["report not saved: disk full", ""])(
    "a done session with report error %j is rendered as danger",
    async (error) => {
      evalState = { ...ENDED, phase: "done", error, report_path: "" };
      await renderEval();

      const title = screen.getByText("Evaluation finished with errors");
      expect(title.parentElement?.parentElement).toHaveClass("border-rose-400/20");
      expect(screen.getByTestId("eval-summary")).toHaveTextContent("Report was not saved.");
      expect(screen.queryByText("Evaluation finished")).toBeNull();
    },
  );

  test("an idle or running session shows no summary", async () => {
    evalState = { ...ENDED, active: true, phase: "rollout", error: "" };
    await renderEval();
    expect(screen.queryByTestId("eval-summary")).toBeNull();
  });
});

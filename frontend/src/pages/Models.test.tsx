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
import Models, { isActiveModel } from "./Models";

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

const ACT: ModelSummary = {
  name: "act-reach",
  path: "/models/act-reach",
  policy_type: "act",
  is_mock: true,
  checkpoint: "020000",
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
  backend.route("GET", "/api/models", () => jsonResponse([SMOL, ACT]));
  backend.route("GET", "/api/eval/state", () => jsonResponse(evalState));
  backend.route("POST", "/api/models/rollout/start", () => jsonResponse({ ok: true, detail: "started" }));
  backend.route("POST", "/api/models/rollout/stop", () => jsonResponse({ ok: true, detail: "stopped" }));
  backend.route("POST", "/api/eval/stop", () => jsonResponse({ ok: true, detail: "stopped" }));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderModels(frame: Record<string, unknown> | null = telemetryFrame(false)) {
  render(
    <TelemetryProvider>
      <SafetyProvider>
        <Models />
      </SafetyProvider>
    </TelemetryProvider>,
  );
  await advance(10);
  if (frame) act(() => FakeWebSocket.broadcast(frame));
  await advance(10);
}

function row(name: string): HTMLElement {
  return screen.getByText(name).closest("tr") as HTMLElement;
}

describe("isActiveModel (C7)", () => {
  test("matches the path, the name, or a path-segment suffix only", () => {
    expect(isActiveModel(SMOL, null)).toBe(false);
    expect(isActiveModel(SMOL, "")).toBe(false);
    expect(isActiveModel(SMOL, "/models/smol")).toBe(true);
    expect(isActiveModel(SMOL, "/models/smol/")).toBe(true);
    expect(isActiveModel(SMOL, "smol")).toBe(true);
    expect(isActiveModel({ ...SMOL, path: "models/smol" }, "/data/models/smol")).toBe(true);
    expect(isActiveModel({ ...SMOL, path: "./models/smol" }, "/data/models/smol")).toBe(true);
    expect(isActiveModel(SMOL, "/models/smol-v2")).toBe(false);
    expect(isActiveModel(SMOL, "/models/notsmol")).toBe(false);
  });
});

describe("Models rollout", () => {
  test("the start dialog is prefilled from the checkpoint and sends the overrides", async () => {
    await renderModels();
    fireEvent.click(screen.getByRole("button", { name: "Start rollout of smol" }));
    const dialog = screen.getByRole("alertdialog", { name: "Start autonomous rollout?" });
    expect(within(dialog).getByLabelText("FPS (optional)")).toHaveValue(30);
    expect(within(dialog).getByLabelText("Task (optional)")).toHaveValue("Pick up the cube");
    fireEvent.change(within(dialog).getByLabelText("FPS (optional)"), { target: { value: "15" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Start rollout" }));
    await advance(10);
    expect(backend.callsTo("/api/models/rollout/start")[0].body).toEqual({
      model_path: SMOL.path,
      fps: 15,
      task: "Pick up the cube",
      operator_id: expect.any(String),
    });
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  test("a non-language policy gets no Task field and an invalid FPS blocks the start", async () => {
    await renderModels();
    fireEvent.click(screen.getByRole("button", { name: "Start rollout of act-reach" }));
    const dialog = screen.getByRole("alertdialog", { name: "Start autonomous rollout?" });
    expect(within(dialog).queryByLabelText("Task (optional)")).toBeNull();
    fireEvent.change(within(dialog).getByLabelText("FPS (optional)"), { target: { value: "500" } });
    expect(within(dialog).getByRole("button", { name: "Start rollout" })).toBeDisabled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(backend.callsTo("/api/models/rollout/start")).toHaveLength(0);
  });

  test("a latched E-stop blocks starting a rollout", async () => {
    backend.estopped = true;
    await renderModels(telemetryFrame(true));
    await advance(1600);
    expect(screen.getByText(/Start is blocked/).parentElement?.textContent).toMatch(
      /^Start is blocked: E-stop is latched/,
    );
    expect(screen.getByRole("button", { name: "Start rollout of smol" })).toBeDisabled();
  });

  test("the running row comes from the server and Stop calls the rollout stop", async () => {
    backend.mode = "rollout";
    backend.statusExtra = { active_model: "/models/smol" };
    await renderModels(telemetryFrame(false, { mode: "rollout" }));
    await advance(1600);
    expect(row("smol")).toHaveAttribute("aria-current", "true");
    expect(within(row("smol")).getByText("running")).toBeInTheDocument();
    expect(row("act-reach")).not.toHaveAttribute("aria-current");
    expect(screen.getByText("Live rollout")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Stop rollout" }));
    await advance(10);
    expect(backend.callsTo("/api/models/rollout/stop")).toHaveLength(1);
    expect(backend.callsTo("/api/eval/stop")).toHaveLength(0);
  });

  test("while an evaluation drives the arm, Stop ends the evaluation", async () => {
    evalState = { ...IDLE, active: true, model_path: "/models/act-reach", episode: 2, n_episodes: 5, phase: "rollout" };
    backend.mode = "rollout";
    await renderModels(telemetryFrame(false, { mode: "rollout" }));
    expect(screen.getByText("An evaluation is driving the arm")).toBeInTheDocument();
    expect(row("act-reach")).toHaveAttribute("aria-current", "true");
    expect(within(row("act-reach")).getByText("evaluating")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Stop evaluation" }));
    await advance(10);
    expect(backend.callsTo("/api/eval/stop")).toHaveLength(1);
    expect(backend.callsTo("/api/models/rollout/stop")).toHaveLength(0);
  });

  test("Stop stays available while the mode is unknown, and hides when idle", async () => {
    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: "down" }, 503));
    await renderModels(null);
    expect(screen.getByRole("button", { name: "Stop rollout" })).toHaveAttribute(
      "title",
      "The robot mode is unknown; stopping is always allowed.",
    );

    act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
    await advance(10);
    expect(screen.queryByRole("button", { name: "Stop rollout" })).toBeNull();
  });

  test("an empty model list says where checkpoints come from", async () => {
    backend.route("GET", "/api/models", () => jsonResponse([]));
    await renderModels();
    expect(
      screen.getByText("No models found. Checkpoints appear here when a training job finishes."),
    ).toBeInTheDocument();
  });
});

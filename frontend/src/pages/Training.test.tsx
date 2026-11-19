import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { DatasetSummary, JobLogs, JobSummary, PolicyPresetOut } from "../api";
import { STATUS_TONE_CLASSES } from "../components/ui/StatusPill";
import {
  advance,
  deferred,
  fakeShellTimers,
  installFakeBackend,
  jsonResponse,
  type FakeBackend,
} from "../test/harness";
import Training, { MAX_LOSS_POINTS, appendLossPoint, jobTone, positiveIntegerError, type LossPoint } from "./Training";

const PRESETS: PolicyPresetOut[] = [
  {
    name: "bc_baseline",
    policy_type: "numpy_bc_mlp",
    description: "Torch-free behaviour cloning baseline",
    default_steps: 2000,
    default_batch_size: 32,
    notes: "",
    trainer: "numpy_in_process",
    requires_torch: false,
  },
  {
    name: "act",
    policy_type: "act",
    description: "Action chunking transformer",
    default_steps: 100000,
    default_batch_size: 8,
    notes: "",
    trainer: "lerobot",
    requires_torch: true,
  },
];

const DATASET: DatasetSummary = {
  repo_id: "lab/pick",
  path: "/data/lab/pick",
  format: "lerobot",
  num_episodes: 3,
  num_frames: 900,
  fps: 30,
  task: "",
  camera_keys: [],
};

function job(id: string, overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id,
    repo_id: "lab/pick",
    policy: "act",
    status: "running",
    step: 10,
    steps: 100,
    loss: 0.5,
    output_dir: `/outputs/${id}`,
    error: null,
    ...overrides,
  };
}

const logs = (id: string, log_tail: string[]): JobLogs => ({ id, status: "running", log_tail });

let backend: FakeBackend;
let jobs: JobSummary[];

beforeEach(() => {
  fakeShellTimers();
  backend = installFakeBackend();
  jobs = [];
  backend.route("GET", "/api/training/presets", () => jsonResponse(PRESETS));
  backend.route("GET", "/api/datasets", () => jsonResponse([DATASET]));
  backend.route("GET", "/api/training/jobs", () => jsonResponse(jobs));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderTraining() {
  render(<Training />);
  await advance(10);
}

const startButton = () => screen.getByRole("button", { name: "Start job" });
const stepsInput = () => screen.getByLabelText(/^Steps/);
const batchInput = () => screen.getByLabelText(/^Batch size/);

describe("Training helpers", () => {
  test("jobTone maps the server's 'succeeded' (not 'completed') to ok (F17)", () => {
    expect(jobTone("succeeded")).toBe("ok");
    expect(jobTone("completed")).toBe("neutral");
    expect(jobTone("running")).toBe("info");
    expect(jobTone("failed")).toBe("danger");
    expect(jobTone("cancelled")).toBe("warn");
    expect(jobTone("queued")).toBe("neutral");
  });

  test("jobTone: an interrupted job warns and a corrupt record is danger, not neutral", () => {
    expect(jobTone("interrupted")).toBe("warn");
    expect(jobTone("corrupt")).toBe("danger");
    expect(jobTone("pending")).toBe("neutral");
  });

  test("positiveIntegerError rejects blank, zero, fractional, signed and huge values (F20)", () => {
    expect(positiveIntegerError("", "Steps")).toBe("Steps is required.");
    expect(positiveIntegerError("   ", "Steps")).toBe("Steps is required.");
    expect(positiveIntegerError("0", "Steps")).toBe("Steps must be at least 1.");
    expect(positiveIntegerError("1.5", "Steps")).toBe("Steps must be a whole number.");
    expect(positiveIntegerError("-3", "Steps")).toBe("Steps must be a whole number.");
    expect(positiveIntegerError("1e3", "Steps")).toBe("Steps must be a whole number.");
    expect(positiveIntegerError("99999999999999999999", "Steps")).toBe("Steps is too large.");
    expect(positiveIntegerError(" 64 ", "Batch size")).toBeNull();
  });

  test("appendLossPoint keeps at most MAX_LOSS_POINTS, thins old history and always keeps the newest (F20)", () => {
    let points: LossPoint[] = [];
    for (let step = 0; step < 5 * MAX_LOSS_POINTS; step += 1) points = appendLossPoint(points, { step, loss: 1 / (step + 1) });
    expect(points.length).toBeLessThanOrEqual(MAX_LOSS_POINTS);
    expect(points.length).toBeGreaterThan(MAX_LOSS_POINTS / 2);
    expect(points[0].step).toBe(0); // the start of the run survives thinning
    expect(points[points.length - 1].step).toBe(5 * MAX_LOSS_POINTS - 1);
    const steps = points.map((point) => point.step);
    expect([...steps].sort((a, b) => a - b)).toEqual(steps);

    const small = appendLossPoint(
      [
        { step: 1, loss: 1 },
        { step: 2, loss: 1 },
        { step: 3, loss: 1 },
      ],
      { step: 4, loss: 1 },
      3,
    );
    expect(small.map((point) => point.step)).toEqual([1, 3, 4]);
  });
});

describe("Training form (F20)", () => {
  test("blank or zero budgets block Start with inline reasons instead of sending 0", async () => {
    await renderTraining();
    fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "lab/pick" } });
    fireEvent.change(screen.getByLabelText(/^Policy/), { target: { value: "act" } });
    expect(startButton()).toBeEnabled();

    fireEvent.change(stepsInput(), { target: { value: "" } });
    fireEvent.change(batchInput(), { target: { value: "0" } });
    expect(startButton()).toBeDisabled();
    expect(screen.getAllByText("Steps is required.").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Batch size must be at least 1.").length).toBeGreaterThan(0);
    expect(stepsInput()).toHaveAttribute("aria-invalid", "true");

    fireEvent.click(startButton());
    await advance(10);
    expect(backend.callsTo("/api/training/jobs")).toHaveLength(0);
  });

  test("choosing a policy loads its preset budget, and Start posts whole numbers", async () => {
    backend.route("POST", "/api/training/jobs", (call) => {
      const created = job("job-new", { status: "queued", step: 0, loss: null, policy: "numpy_bc_mlp" });
      jobs = [created];
      expect(call.body).toBeDefined();
      return jsonResponse(created);
    });
    await renderTraining();
    expect(screen.getByText("no PyTorch needed")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "lab/pick" } });
    fireEvent.change(screen.getByLabelText(/^Policy/), { target: { value: "numpy_bc_mlp" } });
    expect(stepsInput()).toHaveValue("2000");
    expect(batchInput()).toHaveValue("32");
    expect(screen.getByText("Preset default 2000")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /bc_baseline/ })).toHaveAttribute("aria-pressed", "true");

    fireEvent.change(stepsInput(), { target: { value: " 1500 " } });
    fireEvent.click(startButton());
    await advance(10);
    const [call] = backend.callsTo("/api/training/jobs");
    expect(call.body).toEqual({ repo_id: "lab/pick", policy: "numpy_bc_mlp", steps: 1500, batch_size: 32 });
    await advance(10);
    expect(screen.getByText("Loss — job-new")).toBeInTheDocument(); // the new job is selected
    expect(screen.getByText("Waiting for loss data…")).toBeInTheDocument();
  });

  test("a server rejection is shown inline as an alert with the server's words", async () => {
    backend.route("POST", "/api/training/jobs", () =>
      jsonResponse({ detail: [{ loc: ["body", "steps"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" }] }, 422),
    );
    await renderTraining();
    fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "lab/pick" } });
    fireEvent.change(screen.getByLabelText(/^Policy/), { target: { value: "act" } });
    fireEvent.click(startButton());
    await advance(10);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("The server rejected the job");
    expect(alert).toHaveTextContent("Input should be greater than or equal to 1");
  });
});

describe("Training jobs", () => {
  test("a succeeded job reads as succeeded with the ok tone and cannot be cancelled (F17)", async () => {
    jobs = [job("job-done", { status: "succeeded", step: 100 })];
    await renderTraining();
    const row = screen.getByRole("button", { name: "Show job job-done" }).closest("tr") as HTMLElement;
    expect(within(row).getByText("succeeded")).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: "Cancel job job-done" })).toBeDisabled();
    expect(within(row).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
  });

  test("interrupted and corrupt jobs show the warn and danger tones in the job list", async () => {
    jobs = [
      job("job-cut", { status: "interrupted", error: "the server stopped before the job finished" }),
      job("job-bad", { status: "corrupt", error: "the job record could not be read" }),
    ];
    await renderTraining();
    for (const [id, status, tone] of [
      ["job-cut", "interrupted", STATUS_TONE_CLASSES.warn],
      ["job-bad", "corrupt", STATUS_TONE_CLASSES.danger],
    ] as const) {
      const row = screen.getByRole("button", { name: `Show job ${id}` }).closest("tr") as HTMLElement;
      const pill = within(row).getByText(status).closest("span.rounded-full") as HTMLElement;
      expect(pill.className).toContain(tone);
    }
  });

  test("cancel asks first; 'Keep running' sends nothing and confirming posts the cancel", async () => {
    jobs = [job("job-1")];
    backend.route("POST", "/api/training/jobs/job-1/cancel", () => {
      jobs = [job("job-1", { status: "cancelled" })];
      return jsonResponse({ ok: true, detail: "cancelled" });
    });
    await renderTraining();

    fireEvent.click(screen.getByRole("button", { name: "Cancel job job-1" }));
    await advance(10);
    let dialog = screen.getByRole("alertdialog", { name: "Cancel training job job-1?" });
    expect(dialog).toHaveTextContent("/outputs/job-1");
    fireEvent.click(within(dialog).getByRole("button", { name: "Keep running" }));
    await advance(10);
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(backend.callsTo("/api/training/jobs/job-1/cancel")).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Cancel job job-1" }));
    await advance(10);
    dialog = screen.getByRole("alertdialog", { name: "Cancel training job job-1?" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel job" }));
    await advance(10);
    expect(backend.callsTo("/api/training/jobs/job-1/cancel")).toHaveLength(1);
    expect(screen.getByText("cancelled")).toBeInTheDocument();
  });

  test("a failed cancel is reported inline, never through window.alert", async () => {
    const alertSpy = vi.fn();
    vi.stubGlobal("alert", alertSpy);
    jobs = [job("job-1")];
    backend.route("POST", "/api/training/jobs/job-1/cancel", () => jsonResponse({ detail: "job already finished" }, 409));
    await renderTraining();
    fireEvent.click(screen.getByRole("button", { name: "Cancel job job-1" }));
    await advance(10);
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Cancel job" }));
    await advance(10);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Cancel failed");
    expect(alert).toHaveTextContent("Could not cancel job job-1: job already finished");
    expect(alertSpy).not.toHaveBeenCalled();
  });

  test("switching jobs never shows the previous job's logs or loss curve (F18)", async () => {
    jobs = [job("job-a", { loss: 0.9, step: 40 }), job("job-b", { loss: null, step: 0, status: "queued" })];
    const logsB = deferred<Response>();
    backend.route("GET", "/api/training/jobs/job-a/logs", () => jsonResponse(logs("job-a", ["a: step 40 loss 0.9"])));
    backend.route("GET", "/api/training/jobs/job-b/logs", () => logsB.promise);
    await renderTraining();

    fireEvent.click(screen.getByRole("button", { name: "Show job job-a" }));
    await advance(10);
    expect(screen.getByTestId("job-log-tail")).toHaveTextContent("a: step 40 loss 0.9");
    expect(screen.getByRole("button", { name: "Show job job-a" })).toHaveAttribute("aria-current", "true");

    fireEvent.click(screen.getByRole("button", { name: "Show job job-b" }));
    await advance(10);
    expect(screen.getByText("Loss — job-b")).toBeInTheDocument();
    expect(screen.getByTestId("job-log-tail")).toHaveTextContent("Loading logs…");
    expect(screen.queryByText(/a: step 40/)).toBeNull();
    expect(screen.getByText("Waiting for loss data…")).toBeInTheDocument(); // not job-a's curve

    logsB.resolve(jsonResponse(logs("job-b", [])));
    await advance(10);
    expect(screen.getByTestId("job-log-tail")).toHaveTextContent("No log output.");
  });

  test("the trainer's progress line is shown when the server reports one", async () => {
    jobs = [job("job-1", { progress: "train: 10%|#         | 10/100 [00:01<00:09]" })];
    backend.route("GET", "/api/training/jobs/job-1/logs", () => jsonResponse(logs("job-1", [])));
    await renderTraining();
    fireEvent.click(screen.getByRole("button", { name: "Show job job-1" }));
    await advance(10);
    expect(screen.getByText(/10\/100 \[00:01<00:09\]/)).toBeInTheDocument();
  });

  test("a failing job list says it is not updating and keeps the last rows", async () => {
    jobs = [job("job-1")];
    await renderTraining();
    backend.route("GET", "/api/training/jobs", () => jsonResponse({ detail: "boom" }, 500));
    await advance(2100);
    expect(screen.getByText("Job list is not updating")).toBeInTheDocument();
    expect(screen.getByText(/rows below are from the last successful poll/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show job job-1" })).toBeInTheDocument();
  });
});

import { expect, test, type Page } from "@playwright/test";
import { cleanupProblems, readCleanupSnapshot, runBestEffortCleanup } from "./cleanup";
import { E2E_DATASET } from "./contract";
import { E2EAggregateError } from "./errors";
import {
  installBrowserMutationGate,
  type BrowserMutationGate,
} from "./mutation-gate";

const TERMINAL_JOB_TIMEOUT_MS = 60_000;

async function openWorkspace(page: Page, name: string): Promise<void> {
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name }).click();
}

async function collapseHaltToRest(page: Page): Promise<void> {
  const halted = page.getByTestId("halted-banner");
  await expect(halted).toBeVisible();
  await halted.getByRole("button", { name: "Collapse now" }).click();
  await expect(page.getByTestId("safety-chip")).toContainText("disarmed", { timeout: 20_000 });
  await expect(halted).toHaveCount(0);
}

let mutationGate: BrowserMutationGate | null = null;

test.beforeEach(async ({ page, request }) => {
  mutationGate = await installBrowserMutationGate(page, request);
});

test("installed mock UI completes record, train, rollout, and evaluation", async ({ page }) => {
  test.slow();
  await page.goto("/");

  const backend = page.getByTestId("backend-badge");
  await expect(backend).toBeVisible();
  await expect(backend).toHaveAttribute("data-backend", "mock");
  await expect(backend).toContainText("MOCK");
  await expect(page.getByTestId("connection-indicator")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Good to see you, operator." })).toBeVisible();
  await expect(page.locator('[data-pill-label=""]', { hasText: "mode" }).first().locator("..")).toContainText("idle");
  await expect(page.getByText("Mock hardware adapter")).toBeVisible();

  await openWorkspace(page, "Teleop & capture");
  await expect(page.getByRole("heading", { name: "Teleop & Record" })).toBeVisible();
  await page.getByLabel("Dataset repo_id").fill(E2E_DATASET);
  await page.getByLabel("Task").fill("Installed UI end-to-end hold-still check");
  await page.getByLabel("Recording FPS").fill("15");

  const startRecording = page.getByRole("button", { name: "Start recording" });
  await expect(startRecording).toBeEnabled();
  await startRecording.click();
  const startEpisode = page.getByRole("button", { name: "Start episode" });
  await expect(startEpisode).toBeEnabled();
  await startEpisode.click();

  const framesPill = page.locator('[data-pill-label=""]', { hasText: "frames" }).locator("..");
  await expect
    .poll(async () => Number((await framesPill.textContent())?.match(/\d+/)?.[0] ?? 0), {
      message: "recording should collect at least three telemetry frames",
    })
    .toBeGreaterThanOrEqual(3);

  await page.getByRole("button", { name: "Save episode" }).click();
  await expect(page.getByRole("status")).toContainText(/Saved episode #0 with \d+ frames/);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await collapseHaltToRest(page);
  await expect(startRecording).toBeEnabled();

  await openWorkspace(page, "Datasets");
  await expect(page.getByRole("heading", { name: "Dataset workspace" })).toBeVisible();
  const datasetRow = page.getByRole("row").filter({ hasText: E2E_DATASET });
  await expect(datasetRow).toContainText("1");
  await datasetRow.click();
  const outcome = page.getByRole("combobox", { name: "Episode 0 outcome" });
  await expect(outcome).toHaveValue("unlabeled");
  await outcome.selectOption("success");
  await expect(page.getByText("Episode 0 marked success.")).toBeVisible();
  await expect(outcome).toHaveValue("success");

  await openWorkspace(page, "Training");
  await expect(page.getByRole("heading", { name: "Training" })).toBeVisible();
  await page.getByLabel("Dataset").selectOption(E2E_DATASET);
  await page.getByLabel(/^Policy/).selectOption("numpy_bc_mlp");
  await page.getByLabel("Steps").fill("3");
  await page.getByLabel("Batch size").fill("2");
  const startJob = page.getByRole("button", { name: "Start job" });
  await expect(startJob).toBeEnabled();
  await startJob.click();

  const jobRow = page
    .getByRole("row")
    .filter({ hasText: E2E_DATASET })
    .filter({ hasText: "bc_baseline" });
  await expect(jobRow).toContainText("succeeded", { timeout: TERMINAL_JOB_TIMEOUT_MS });
  const jobButton = jobRow.getByRole("button", { name: /^Show job / });
  const jobLabel = await jobButton.getAttribute("aria-label");
  const jobId = jobLabel?.replace(/^Show job /, "");
  expect(jobId, "training row must expose its job id").toMatch(/^[a-f0-9]{12}$/);

  await openWorkspace(page, "Models");
  await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  const modelRow = page.getByRole("row").filter({ hasText: jobId as string });
  await expect(modelRow).toContainText("numpy_bc_mlp");
  const startRollout = modelRow.getByRole("button", { name: /^Start rollout of / });
  await expect(startRollout).toBeEnabled();
  await startRollout.click();
  const rolloutDialog = page.getByRole("alertdialog", { name: "Start autonomous rollout?" });
  await expect(rolloutDialog).toBeVisible();
  const confirmRollout = rolloutDialog.getByRole("button", { name: "Start rollout" });
  await expect(confirmRollout).toBeEnabled();
  await confirmRollout.click();
  const stopRollout = page.getByRole("button", { name: "Stop rollout" });
  await expect(stopRollout).toBeVisible();
  await expect(modelRow).toContainText("running");
  await stopRollout.click();
  await collapseHaltToRest(page);
  await expect(startRollout).toBeEnabled();

  await openWorkspace(page, "Evaluation");
  await expect(page.getByRole("heading", { name: "Evaluation" })).toBeVisible();
  const modelSelect = page.getByLabel("Model");
  const trainedOption = modelSelect.locator("option").filter({ hasText: jobId as string });
  const modelPath = await trainedOption.getAttribute("value");
  expect(modelPath, "trained model must be offered for evaluation").toBeTruthy();
  await modelSelect.selectOption(modelPath as string);
  await page.getByLabel("Episodes").fill("1");
  await page.getByLabel("Max seconds / episode").fill("0.2");
  const startEval = page.getByRole("button", { name: "Start eval" });
  await expect(startEval).toBeEnabled();
  await startEval.click();
  const evalDialog = page.getByRole("alertdialog", { name: "Start autonomous evaluation?" });
  const confirmEval = evalDialog.getByRole("button", { name: "Start evaluation" });
  await expect(confirmEval).toBeEnabled();
  await confirmEval.click();
  const success = page.getByRole("button", { name: "Success" });
  await expect(success).toBeVisible({ timeout: 30_000 });
  await success.click();

  const summary = page.getByTestId("eval-summary");
  await expect(summary).toContainText("1 of 1 requested episode scored");
  await expect(summary).toContainText("Report saved to");
  await expect(summary).not.toContainText("Report was not saved");
  await expect(backend).toHaveAttribute("data-backend", "mock");
});

test.afterEach(async ({ request }) => {
  const errors = await runBestEffortCleanup(request);
  try {
    await expect
      .poll(async () => cleanupProblems(await readCleanupSnapshot(request)), {
        message: "test-created authority, loops, jobs, and latches must reach terminal rest",
        timeout: 20_000,
      })
      .toEqual([]);
  } catch (reason) {
    errors.push(reason instanceof Error ? reason : new Error("cleanup state could not be verified"));
  }
  try {
    mutationGate?.assertNoBlockedMutation();
  } catch (reason) {
    errors.push(reason instanceof Error ? reason : new Error("a browser mutation was blocked"));
  } finally {
    mutationGate = null;
  }
  if (errors.length) {
    throw new E2EAggregateError("installed UI E2E cleanup did not complete cleanly", errors);
  }
});

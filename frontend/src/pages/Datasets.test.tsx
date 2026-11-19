import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type {
  DatasetDetail,
  DatasetQualityReport,
  DatasetSummary,
  EpisodeManifest,
  EpisodeTimeline,
} from "../api";
import {
  advance,
  deferred,
  fakeShellTimers,
  installFakeBackend,
  jsonResponse,
  type FakeBackend,
} from "../test/harness";
import Datasets from "./Datasets";

function summary(repo_id: string): DatasetSummary {
  return {
    repo_id,
    path: `/data/${repo_id}`,
    format: "lerobot",
    num_episodes: 0,
    num_frames: 0,
    fps: 30,
    task: "",
    camera_keys: [],
  };
}

function detail(repo_id: string): DatasetDetail {
  return {
    ...summary(repo_id),
    duration_seconds: 0,
    episodes: [],
    incomplete_captures: [],
    outcome_counts: {},
    capabilities: [],
  };
}

function episode(n: number): EpisodeManifest {
  return {
    episode: n,
    name: `episode_${n}`,
    frames: 90,
    duration_seconds: 3,
    started_at: null,
    ended_at: null,
    outcome: "unlabeled",
    notes: "",
    complete: true,
    camera_chunks: {},
    camera_frames: {},
    issues: [],
  };
}

function timeline(repo_id: string, n: number): EpisodeTimeline {
  return {
    repo_id,
    episode: n,
    fps: 30,
    total_frames: 90,
    offset: 0,
    stride: 1,
    returned_frames: 1,
    has_more: false,
    joint_names: ["shoulder_pan"],
    actuates_hardware: false,
    capability: { id: "replay_preview", available: true, level: "full", detail: "" },
    points: [
      {
        frame_index: 0,
        t: 0,
        relative_time_seconds: 0,
        observation: [1],
        action: [1],
        desired_action: [1],
      },
    ],
  };
}

function report(repo_id: string): DatasetQualityReport {
  return { repo_id, generated_at: 0, overall: "fail", checked_frames: 1234, checks: [], capabilities: [] };
}

const A = "lab/pick";
const B = "lab/place";
const encoded = (repo: string) => `/api/datasets/${encodeURIComponent(repo)}`;

let backend: FakeBackend;

beforeEach(() => {
  fakeShellTimers();
  backend = installFakeBackend();
  backend.route("GET", "/api/datasets", () => jsonResponse([summary(A), summary(B)]));
  backend.route("GET", "/api/datasets/trash", () => jsonResponse([]));
  backend.route("GET", encoded(A), () => jsonResponse(detail(A)));
  backend.route("GET", encoded(B), () => jsonResponse(detail(B)));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderDatasets() {
  render(<Datasets />);
  await advance(10);
}

function repoRow(repo: string): HTMLElement {
  return screen.getAllByText(repo)[0].closest("tr") as HTMLElement;
}

describe("Datasets accessibility", () => {
  test("each icon-only trash button names the dataset it moves", async () => {
    await renderDatasets();
    for (const repo of [A, B]) {
      expect(within(repoRow(repo)).getByRole("button", { name: `Move ${repo} to recoverable trash` })).toBeInTheDocument();
    }
  });
});

describe("Datasets push (data audit)", () => {
  test("confirms, then sends an attributed private push as a JSON body", async () => {
    backend.route("POST", `${encoded(A)}/push`, () => jsonResponse({ ok: true, detail: "pushed lab/pick" }));
    await renderDatasets();
    fireEvent.click(within(repoRow(A)).getByRole("button", { name: "Push" }));
    await advance(10);
    const dialog = screen.getByRole("alertdialog", { name: `Push "${A}" to Hugging Face?` });
    fireEvent.click(within(dialog).getByRole("button", { name: "Push privately" }));
    await advance(10);
    const [call] = backend.callsTo(`${encoded(A)}/push`);
    expect(call.url).toBe(`${encoded(A)}/push`); // no ?private= query any more
    expect(call.body).toEqual({ operator_id: expect.any(String), private: true });
    expect(screen.getByText("pushed lab/pick")).toBeInTheDocument();
  });

  test("cancelling the confirmation sends nothing", async () => {
    await renderDatasets();
    fireEvent.click(within(repoRow(A)).getByRole("button", { name: "Push" }));
    await advance(10);
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Cancel" }));
    await advance(10);
    expect(backend.callsTo(`${encoded(A)}/push`)).toHaveLength(0);
  });
});

describe("Datasets inspector races (F23)", () => {
  test("a quality report for the previous repo never shows under the next one", async () => {
    const slow = deferred<Response>();
    backend.route("GET", `${encoded(A)}/qc`, () => slow.promise);
    await renderDatasets();
    // The first repo is inspected by default.
    expect(screen.getByRole("heading", { name: A })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Quality" }));
    fireEvent.click(screen.getByRole("button", { name: "Run quality control" }));
    await advance(10);
    expect(screen.getByRole("button", { name: "Running checks…" })).toBeDisabled();

    fireEvent.click(repoRow(B));
    await advance(10);
    expect(screen.getByRole("heading", { name: B })).toBeInTheDocument();
    // The open tab is a view choice and stays open for the next repo.
    expect(screen.getByRole("tab", { name: "Quality" })).toHaveAttribute("aria-selected", "true");
    slow.resolve(jsonResponse(report(A)));
    await advance(10);
    expect(screen.queryByText("1234")).toBeNull();
    expect(screen.getByRole("button", { name: "Run quality control" })).toBeEnabled();
  });
});

describe("Datasets inspector tabs", () => {
  test("manifest, replay and quality are tabs; Inspect opens the episode's replay", async () => {
    backend.route("GET", encoded(A), () =>
      jsonResponse({ ...detail(A), num_episodes: 2, episodes: [episode(2), episode(3)] }),
    );
    backend.route("GET", `${encoded(A)}/episodes/2/replay-preview`, () => jsonResponse(timeline(A, 2)));
    backend.route("GET", `${encoded(A)}/episodes/3/replay-preview`, () => jsonResponse(timeline(A, 3)));
    await renderDatasets();
    const list = screen.getByRole("tablist", { name: "Dataset inspector" });
    expect(within(list).getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Episodes (2)",
      "Replay #002",
      "Quality",
    ]);
    // Only the open panel is rendered.
    const manifest = screen.getByRole("tabpanel", { name: "Episodes (2)" });
    expect(within(manifest).getByText("#003")).toBeInTheDocument();
    expect(screen.queryByText("Read-only action replay")).toBeNull();

    // A row click only selects; Inspect also opens the replay.
    fireEvent.click(within(manifest).getByText("#003"));
    await advance(10);
    expect(screen.getByRole("tab", { name: "Replay #003" })).toHaveAttribute("aria-selected", "false");
    const row = within(manifest).getByText("#002").closest("tr") as HTMLElement;
    fireEvent.click(within(row).getByRole("button", { name: "Inspect" }));
    await advance(10);
    const replayTab = screen.getByRole("tab", { name: "Replay #002" });
    expect(replayTab).toHaveAttribute("aria-selected", "true");
    const replay = screen.getByRole("tabpanel", { name: "Replay #002" });
    expect(within(replay).getByText("Read-only action replay")).toBeInTheDocument();
    expect(backend.callsTo(`${encoded(A)}/episodes/2/replay-preview`, "GET").length).toBeGreaterThan(0);

    fireEvent.keyDown(replayTab, { key: "ArrowRight" });
    expect(screen.getByRole("tabpanel", { name: "Quality" })).toHaveTextContent("Dataset quality control");
  });
});

describe("Datasets outcome labels", () => {
  function withEpisodes(format: string): DatasetDetail {
    return { ...detail(A), format, num_episodes: 2, episodes: [episode(0), episode(1)] };
  }

  test.each(["lerobot", "so101-fallback-v2"])("%s episodes have an outcome selector that saves the label", async (format) => {
    backend.route("GET", encoded(A), () => jsonResponse(withEpisodes(format)));
    const labeled = withEpisodes(format);
    labeled.episodes = [episode(0), { ...episode(1), outcome: "failure" }];
    backend.route("POST", `${encoded(A)}/episodes/1/outcome`, () => jsonResponse(labeled));
    await renderDatasets();
    const select = screen.getByRole("combobox", { name: "Episode 1 outcome" });
    fireEvent.change(select, { target: { value: "failure" } });
    await advance(10);
    const [call] = backend.callsTo(`${encoded(A)}/episodes/1/outcome`);
    expect(call.body).toEqual({ outcome: "failure", notes: "" });
    expect(screen.getByRole("combobox", { name: "Episode 1 outcome" })).toHaveValue("failure");
  });

  test("an unknown format shows the outcome read-only", async () => {
    backend.route("GET", encoded(A), () => jsonResponse(withEpisodes("custom-v9")));
    await renderDatasets();
    expect(screen.queryByRole("combobox", { name: "Episode 1 outcome" })).toBeNull();
  });
});

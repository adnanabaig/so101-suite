import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import {
  ArchiveRestore,
  Camera,
  Database,
  HardDrive,
  PlayCircle,
  RefreshCw,
  ShieldCheck,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import {
  deleteDataset,
  getDataset,
  getDatasetQuality,
  getDatasets,
  getDatasetTrash,
  getReplayPreview,
  pushDataset,
  restoreDataset,
  setEpisodeOutcome,
  type DatasetDetail,
  type DatasetQualityReport,
  type EpisodeOutcome,
  type EpisodeTimeline,
} from "../api";
import Button from "../components/Button";
import Card from "../components/Card";
import DatasetQualityPanel from "../components/DatasetQualityPanel";
import DatasetReplayPanel from "../components/DatasetReplayPanel";
import { StatusPill } from "../components/ui/StatusPill";
import { useConfirm } from "../components/ui/Dialog";
import { Tabs } from "../components/ui/Tabs";
import { usePolling } from "../hooks/usePolling";
import { errorMessage } from "../lib/errors";

const OUTCOMES: EpisodeOutcome[] = [
  "unlabeled",
  "success",
  "failure",
  "partial",
  "discarded",
];

function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${(seconds - minutes * 60).toFixed(0)}s`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

type InspectorTab = "episodes" | "replay" | "quality";

/** "#007"-style episode numbers. */
function episodeNumber(episode: number): string {
  return episode.toString().padStart(3, "0");
}

/** Formats whose episodes the server can label: the fallback layout keeps labels
 * in each episode.json, LeRobot datasets in the suite's so101_outcomes.json. */
function outcomeLabelsEditable(format: string): boolean {
  return format === "lerobot" || format.startsWith("so101-fallback-");
}

function outcomeTone(outcome: string) {
  if (outcome === "success") return "ok" as const;
  if (outcome === "failure" || outcome === "discarded") return "danger" as const;
  if (outcome === "partial") return "warn" as const;
  return "neutral" as const;
}

export default function Datasets() {
  const [confirm, confirmDialog] = useConfirm();
  const {
    data,
    error,
    loading,
    refresh: refreshDatasets,
  } = usePolling(getDatasets, 0);
  const {
    data: trash,
    error: trashError,
    refresh: refreshTrash,
  } = usePolling(getDatasetTrash, 0);

  // Repos trashed from this page are hidden at once instead of lingering until
  // the list refetch lands (which would otherwise let the inspector reopen a
  // repo that no longer exists). Any newly fetched list supersedes the hint.
  const [pendingRemoval, setPendingRemoval] = useState<ReadonlySet<string>>(() => new Set());
  const [listSnapshot, setListSnapshot] = useState(data);
  if (listSnapshot !== data) {
    setListSnapshot(data);
    setPendingRemoval(new Set());
  }
  const datasets = useMemo(
    () => (data ?? []).filter((item) => !pendingRemoval.has(item.repo_id)),
    [data, pendingRemoval],
  );

  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  // Which inspector tab is open is a view choice, not a per-repo result, so
  // it lives here and survives moving from repo to repo.
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("episodes");
  const [refreshToken, setRefreshToken] = useState(0);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  // The inspected repo is derived rather than synced by an effect: an explicit
  // choice wins while it is still listed, otherwise the first dataset is shown.
  const activeRepo =
    selectedRepo !== null && datasets.some((item) => item.repo_id === selectedRepo)
      ? selectedRepo
      : (datasets[0]?.repo_id ?? null);

  const totals = useMemo(
    () => ({
      episodes: datasets.reduce((sum, item) => sum + item.num_episodes, 0),
      frames: datasets.reduce((sum, item) => sum + item.num_frames, 0),
      cameras: new Set(datasets.flatMap((item) => item.camera_keys)).size,
    }),
    [datasets],
  );

  const announce = useCallback((message: string | null, failure: string | null = null) => {
    setActionMessage(message);
    setActionError(failure);
  }, []);

  const selectRepo = (repoId: string) => {
    if (repoId === activeRepo) return;
    setSelectedRepo(repoId);
    setActionError(null);
  };

  const refreshAll = () => {
    refreshDatasets();
    refreshTrash();
    setRefreshToken((value) => value + 1);
  };

  const remove = async (event: MouseEvent, repoId: string) => {
    event.stopPropagation();
    const confirmed = await confirm({
      title: `Move "${repoId}" to trash?`,
      body: "The dataset leaves the list but stays recoverable from local trash.",
      confirmLabel: "Move to trash",
    });
    if (!confirmed) return;
    setBusyKey(`delete:${repoId}`);
    announce(null);
    try {
      await deleteDataset(repoId);
      setPendingRemoval((current) => new Set(current).add(repoId));
      announce(`${repoId} moved to recoverable trash.`);
      refreshDatasets();
      refreshTrash();
    } catch (reason) {
      announce(null, errorMessage(reason));
    } finally {
      setBusyKey(null);
    }
  };

  const publish = async (event: MouseEvent, repoId: string) => {
    event.stopPropagation();
    const confirmed = await confirm({
      title: `Push "${repoId}" to Hugging Face?`,
      body: "It is uploaded as a private dataset under your Hugging Face account.",
      confirmLabel: "Push privately",
    });
    if (!confirmed) return;
    setBusyKey(`push:${repoId}`);
    announce(null);
    try {
      const result = await pushDataset(repoId, { private: true });
      announce(result.detail);
    } catch (reason) {
      announce(null, errorMessage(reason));
    } finally {
      setBusyKey(null);
    }
  };

  const restore = async (trashId: string, repoId: string) => {
    setBusyKey(`restore:${trashId}`);
    announce(null);
    try {
      await restoreDataset(trashId);
      setPendingRemoval((current) => {
        if (!current.has(repoId)) return current;
        const next = new Set(current);
        next.delete(repoId);
        return next;
      });
      announce(`${repoId} restored.`);
      refreshDatasets();
      refreshTrash();
      setSelectedRepo(repoId);
    } catch (reason) {
      announce(null, errorMessage(reason));
    } finally {
      setBusyKey(null);
    }
  };

  return (
    <div className="space-y-6">
      {confirmDialog}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dataset workspace</h1>
          <p className="mt-1 text-xs text-slate-400">
            Inspect captures, preview action traces, label outcomes, and validate data before training.
          </p>
        </div>
        <Button variant="secondary" onClick={refreshAll}>
          <RefreshCw aria-hidden="true" size={14} /> Refresh
        </Button>
      </div>

      {(error || trashError || actionError) && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {actionError ?? error?.message ?? trashError?.message}
        </div>
      )}
      {actionMessage && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
          {actionMessage}
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric icon={Database} label="Datasets" value={datasets.length} />
        <Metric icon={HardDrive} label="Episodes" value={totals.episodes} />
        <Metric icon={PlayCircle} label="Frames" value={totals.frames.toLocaleString()} />
        <Metric icon={Camera} label="Camera streams" value={totals.cameras} />
      </div>

      <Card title="Local datasets" subtitle="Select a row to open the episode inspector" bodyClassName="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-[10px] uppercase tracking-widest text-muted">
                <th className="px-5 py-3 font-medium">Repository</th>
                <th className="px-4 py-3 font-medium">Format</th>
                <th className="px-4 py-3 text-right font-medium">Episodes</th>
                <th className="px-4 py-3 text-right font-medium">Frames</th>
                <th className="px-4 py-3 text-right font-medium">FPS</th>
                <th className="px-4 py-3 font-medium">Task</th>
                <th className="px-5 py-3 text-right font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={7} className="px-5 py-8 text-center text-muted">
                    Loading datasets…
                  </td>
                </tr>
              )}
              {!loading && datasets.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-5 py-10 text-center text-muted">
                    No datasets yet. Record an episode to begin.
                  </td>
                </tr>
              )}
              {datasets.map((item) => (
                <tr
                  key={item.repo_id}
                  onClick={() => selectRepo(item.repo_id)}
                  className={`cursor-pointer border-b border-slate-800/60 transition-colors ${
                    activeRepo === item.repo_id
                      ? "bg-cyan-500/[0.08]"
                      : "hover:bg-slate-800/30"
                  }`}
                >
                  <td className="px-5 py-3.5 font-mono text-xs font-semibold text-cyan-300">
                    {item.repo_id}
                  </td>
                  <td className="px-4 py-3.5">
                    <StatusPill
                      label=""
                      value={item.format}
                      tone={item.format === "lerobot" ? "info" : "neutral"}
                    />
                  </td>
                  <td className="px-4 py-3.5 text-right font-mono text-slate-300">
                    {item.num_episodes}
                  </td>
                  <td className="px-4 py-3.5 text-right font-mono text-slate-300">
                    {item.num_frames.toLocaleString()}
                  </td>
                  <td className="px-4 py-3.5 text-right font-mono text-slate-400">{item.fps}</td>
                  <td className="max-w-56 truncate px-4 py-3.5 text-xs text-slate-400">
                    {item.task || "—"}
                  </td>
                  <td className="px-5 py-3.5">
                    <div className="flex justify-end gap-1.5">
                      {item.format === "lerobot" && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(event) => void publish(event, item.repo_id)}
                          disabled={busyKey !== null}
                        >
                          {busyKey === `push:${item.repo_id}` ? "Pushing…" : "Push"}
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-300"
                        onClick={(event) => void remove(event, item.repo_id)}
                        disabled={busyKey !== null}
                        title="Move to recoverable trash"
                        aria-label={`Move ${item.repo_id} to recoverable trash`}
                      >
                        <Trash2 aria-hidden="true" size={13} />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {activeRepo && (
        // Keyed by repo: every per-repo result (manifest, replay, quality
        // report, outcome edits) lives in this instance, so a late response
        // for one repo can never render under another (F23).
        <RepoInspector
          key={activeRepo}
          repoId={activeRepo}
          refreshToken={refreshToken}
          onAnnounce={announce}
          tab={inspectorTab}
          onTab={setInspectorTab}
        />
      )}

      <Card
        title="Recoverable trash"
        subtitle="Dataset deletion is a local atomic move; restore is available until storage is cleared"
        actions={<StatusPill label="items" value={trash?.length ?? 0} />}
        bodyClassName="p-0"
      >
        {!trash?.length ? (
          <div className="px-5 py-8 text-center text-sm text-muted">Trash is empty.</div>
        ) : (
          <div className="divide-y divide-slate-800/70">
            {trash.map((item) => (
              <div
                key={item.id}
                className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5"
              >
                <div className="min-w-0">
                  <p className="truncate font-mono text-xs text-slate-300">{item.repo_id}</p>
                  <p className="mt-1 text-[10px] text-muted">
                    {item.num_episodes} episodes · {item.num_frames.toLocaleString()} frames ·{" "}
                    {formatBytes(item.size_bytes)} · deleted {new Date(item.deleted_at * 1000).toLocaleString()}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => void restore(item.id, item.repo_id)}
                  disabled={busyKey !== null}
                >
                  <ArchiveRestore aria-hidden="true" size={13} />
                  {busyKey === `restore:${item.id}` ? "Restoring…" : "Restore"}
                </Button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/55 p-4">
      <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-widest text-muted">
        <Icon aria-hidden="true" size={14} className="text-cyan-400" /> {label}
      </div>
      <p className="mt-2 font-mono text-2xl font-bold text-slate-100">{value}</p>
    </div>
  );
}

function RepoInspector({
  repoId,
  refreshToken,
  onAnnounce,
  tab,
  onTab,
}: {
  repoId: string;
  refreshToken: number;
  onAnnounce: (message: string | null, failure?: string | null) => void;
  tab: InspectorTab;
  onTab: (tab: InspectorTab) => void;
}) {
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [loadedToken, setLoadedToken] = useState<number | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<number | null>(null);
  const [preview, setPreview] = useState<{
    episode: number;
    timeline: EpisodeTimeline | null;
  } | null>(null);
  const [quality, setQuality] = useState<DatasetQualityReport | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [busyKey, setBusyKey] = useState<string | null>(null);

  // Results that arrive after this inspector unmounted (the operator picked
  // another repo) are dropped instead of being reported against the new one.
  const aliveRef = useRef(false);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);
  const announceRef = useRef(onAnnounce);
  useEffect(() => {
    announceRef.current = onAnnounce;
  });

  useEffect(() => {
    let active = true;
    const token = refreshToken;
    void getDataset(repoId)
      .then((result) => {
        if (!active) return;
        setDetail(result);
        setSelectedEpisode((current) =>
          current !== null && result.episodes.some((item) => item.episode === current)
            ? current
            : (result.episodes[0]?.episode ?? null),
        );
      })
      .catch((reason: unknown) => {
        if (active) announceRef.current(null, errorMessage(reason));
      })
      .finally(() => {
        if (active) setLoadedToken(token);
      });
    return () => {
      active = false;
    };
  }, [repoId, refreshToken]);
  const detailLoading = loadedToken !== refreshToken;

  useEffect(() => {
    if (selectedEpisode === null) return;
    let active = true;
    const episode = selectedEpisode;
    void getReplayPreview(repoId, episode)
      .then((result) => {
        if (active) setPreview({ episode, timeline: result });
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setPreview({ episode, timeline: null });
        announceRef.current(null, errorMessage(reason));
      });
    return () => {
      active = false;
    };
  }, [repoId, selectedEpisode]);
  const timeline = preview?.episode === selectedEpisode ? (preview?.timeline ?? null) : null;
  const previewLoading = selectedEpisode !== null && preview?.episode !== selectedEpisode;

  const runQuality = async () => {
    setQualityLoading(true);
    announceRef.current(null);
    try {
      const report = await getDatasetQuality(repoId);
      if (aliveRef.current) setQuality(report);
    } catch (reason) {
      if (aliveRef.current) announceRef.current(null, errorMessage(reason));
    } finally {
      if (aliveRef.current) setQualityLoading(false);
    }
  };

  const labelOutcome = async (episode: number, outcome: EpisodeOutcome) => {
    setBusyKey(`outcome:${episode}`);
    announceRef.current(null);
    try {
      const updated = await setEpisodeOutcome(repoId, episode, outcome);
      if (!aliveRef.current) return;
      setDetail(updated);
      announceRef.current(`Episode ${episode} marked ${outcome}.`);
    } catch (reason) {
      if (aliveRef.current) announceRef.current(null, errorMessage(reason));
    } finally {
      if (aliveRef.current) setBusyKey(null);
    }
  };

  return (
    <DatasetInspector
      detail={detail}
      loading={detailLoading}
      selectedEpisode={selectedEpisode}
      onSelectEpisode={setSelectedEpisode}
      onOutcome={(episode, outcome) => void labelOutcome(episode, outcome)}
      busyKey={busyKey}
      timeline={timeline}
      previewLoading={previewLoading}
      quality={quality}
      qualityLoading={qualityLoading}
      onRunQuality={() => void runQuality()}
      tab={tab}
      onTab={onTab}
    />
  );
}

function DatasetInspector({
  detail,
  loading,
  selectedEpisode,
  onSelectEpisode,
  onOutcome,
  busyKey,
  timeline,
  previewLoading,
  quality,
  qualityLoading,
  onRunQuality,
  tab,
  onTab,
}: {
  detail: DatasetDetail | null;
  loading: boolean;
  selectedEpisode: number | null;
  onSelectEpisode: (episode: number) => void;
  onOutcome: (episode: number, outcome: EpisodeOutcome) => void;
  busyKey: string | null;
  timeline: EpisodeTimeline | null;
  previewLoading: boolean;
  quality: DatasetQualityReport | null;
  qualityLoading: boolean;
  onRunQuality: () => void;
  tab: InspectorTab;
  onTab: (tab: InspectorTab) => void;
}) {
  // The manifest, the replay and the quality scan were three stacked cards;
  // tabs keep one in view. "Inspect" on an episode opens its replay.
  if (loading && !detail) {
    return (
      <Card>
        <div className="py-16 text-center text-sm text-muted">Loading manifest…</div>
      </Card>
    );
  }
  if (!detail) return null;

  return (
    <div className="space-y-4">
      <Card
        title={detail.repo_id}
        subtitle={detail.path}
        actions={
          <StatusPill
            label="format"
            value={detail.format}
            tone={detail.format === "lerobot" ? "info" : "neutral"}
          />
        }
      >
        <div className="grid gap-4 md:grid-cols-[1fr_auto]">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MiniMetric label="Episodes" value={detail.num_episodes} />
            <MiniMetric label="Frames" value={detail.num_frames.toLocaleString()} />
            <MiniMetric label="Duration" value={formatDuration(detail.duration_seconds)} />
            <MiniMetric label="Capture rate" value={`${detail.fps} fps`} />
          </div>
          <div className="flex flex-wrap items-start gap-2 md:max-w-sm md:justify-end">
            {detail.capabilities.map((capability) => (
              <span
                key={capability.id}
                title={capability.detail}
                className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold ${
                  capability.available
                    ? capability.level === "full"
                      ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-300"
                      : "border-amber-500/25 bg-amber-500/10 text-amber-300"
                    : "border-slate-700 bg-slate-800 text-muted"
                }`}
              >
                {capability.id.replace(/_/g, " ")} · {capability.level}
              </span>
            ))}
          </div>
        </div>
      </Card>

      {detail.incomplete_captures.length > 0 && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/[0.07] p-4">
          <div className="flex items-start gap-3">
            <TriangleAlert aria-hidden="true" className="mt-0.5 shrink-0 text-amber-300" size={18} />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-amber-200">
                {detail.incomplete_captures.length} interrupted capture(s) preserved
              </p>
              <div className="mt-2 grid gap-2 md:grid-cols-2">
                {detail.incomplete_captures.map((capture) => (
                  <div key={capture.id} className="rounded-lg bg-slate-950/40 px-3 py-2 text-xs">
                    <p className="font-mono text-slate-300">{capture.id}</p>
                    <p className="mt-1 text-slate-400">
                      {capture.frames} frames · {formatDuration(capture.duration_seconds)} ·{" "}
                      {capture.reason}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <Tabs
        label="Dataset inspector"
        selected={tab}
        onSelect={(id) => onTab(id as InspectorTab)}
        panelClassName="pt-3"
        tabs={[
          {
            id: "episodes",
            label: `Episodes (${detail.episodes.length})`,
            panel: (
              <Card
                title="Episode manifest"
                subtitle="Frame counts, elapsed capture time, outcome labels, and camera chunk coverage"
                bodyClassName="p-0"
              >
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-left text-[10px] uppercase tracking-widest text-muted">
                        <th className="px-5 py-3 font-medium">Episode</th>
                        <th className="px-4 py-3 text-right font-medium">Frames</th>
                        <th className="px-4 py-3 text-right font-medium">Duration</th>
                        <th className="px-4 py-3 font-medium">Camera chunks</th>
                        <th className="px-4 py-3 font-medium">Outcome</th>
                        <th className="px-5 py-3 text-right font-medium">Preview</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.episodes.length === 0 && (
                        <tr>
                          <td colSpan={6} className="px-5 py-8 text-center text-muted">
                            No finalized episodes.
                          </td>
                        </tr>
                      )}
                      {detail.episodes.map((episode) => (
                        <tr
                          key={episode.episode}
                          onClick={() => onSelectEpisode(episode.episode)}
                          className={`cursor-pointer border-b border-slate-800/60 ${
                            selectedEpisode === episode.episode
                              ? "bg-cyan-500/[0.07]"
                              : "hover:bg-slate-800/30"
                          }`}
                        >
                          <td className="px-5 py-3 font-mono font-semibold text-cyan-300">
                            #{episodeNumber(episode.episode)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-slate-300">
                            {episode.frames ?? "—"}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-slate-400">
                            {formatDuration(episode.duration_seconds)}
                          </td>
                          <td className="px-4 py-3 text-slate-400">
                            {Object.keys(episode.camera_chunks).length
                              ? Object.entries(episode.camera_chunks)
                                  .map(([camera, chunks]) => `${camera}: ${chunks}`)
                                  .join(" · ")
                              : "none"}
                          </td>
                          <td className="px-4 py-3" onClick={(event) => event.stopPropagation()}>
                            {outcomeLabelsEditable(detail.format) ? (
                              <select
                                aria-label={`Episode ${episode.episode} outcome`}
                                value={episode.outcome}
                                onChange={(event) =>
                                  onOutcome(episode.episode, event.target.value as EpisodeOutcome)
                                }
                                disabled={busyKey === `outcome:${episode.episode}`}
                                className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-[11px] text-slate-200 outline-hidden focus-visible:border-cyan-400 focus-visible:ring-2 focus-visible:ring-cyan-400/40"
                              >
                                {OUTCOMES.map((outcome) => (
                                  <option key={outcome} value={outcome}>
                                    {outcome}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <StatusPill
                                label=""
                                value={episode.outcome}
                                tone={outcomeTone(episode.outcome)}
                              />
                            )}
                          </td>
                          <td className="px-5 py-3 text-right">
                            <Button
                              size="sm"
                              variant={selectedEpisode === episode.episode ? "primary" : "secondary"}
                              onClick={(event) => {
                                event.stopPropagation();
                                onSelectEpisode(episode.episode);
                                onTab("replay");
                              }}
                            >
                              <PlayCircle aria-hidden="true" size={13} /> Inspect
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            ),
          },
          {
            id: "replay",
            label: selectedEpisode === null ? "Replay" : `Replay #${episodeNumber(selectedEpisode)}`,
            panel: (
              <Card
                title="Read-only action replay"
                subtitle="Scrub or play the recorded command trace in the browser; this path cannot actuate an arm"
                actions={
                  <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-300">
                    <ShieldCheck aria-hidden="true" size={13} /> SAFE PREVIEW
                  </span>
                }
              >
                <DatasetReplayPanel timeline={timeline} loading={previewLoading} />
              </Card>
            ),
          },
          {
            id: "quality",
            label: "Quality",
            panel: (
              <Card
                title="Dataset quality control"
                subtitle="A bounded local scan; no frames or credentials leave this workstation"
              >
                <DatasetQualityPanel
                  report={quality}
                  loading={qualityLoading}
                  onRun={onRunQuality}
                />
              </Card>
            ),
          },
        ]}
      />
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-3">
      <p className="text-[9px] font-semibold uppercase tracking-widest text-muted">{label}</p>
      <p className="mt-1 font-mono text-base font-bold text-slate-100">{value}</p>
    </div>
  );
}

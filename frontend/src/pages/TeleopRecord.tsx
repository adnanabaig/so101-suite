// Leader-arm teleoperation and dataset recording. Both are internally
// supervised, single-owner loops. F21: each Start lists every unmet
// precondition, the fields are checked the way the server checks them, and the
// mode and episode state shown are the server's current word, or "unknown"
// while the link is down.
import { useState, type ReactNode } from "react";
import Card from "../components/Card";
import Button from "../components/Button";
import { StatusPill } from "../components/ui/StatusPill";
import JointBars from "../components/JointBars";
import CameraView from "../components/CameraView";
import ManualControlPanel from "../components/ManualControlPanel";
import McapWorkflow from "../components/McapWorkflow";
import BlockerList from "../components/BlockerList";
import { useSafety } from "../components/safety/SafetyProvider";
import { Field, INPUT_CLASS } from "../components/ui/Field";
import {
  startTeleop,
  stopTeleop,
  startRecording,
  startEpisode,
  saveEpisode,
  stopRecording,
  type EpisodeResult,
  type StartRecordingRequest,
  getLatestPreflight,
} from "../api";
import { useTelemetry } from "../hooks/useTelemetry";
import { useOperatorSupervision } from "../hooks/useOperatorSupervision";
import { errorMessage } from "../lib/errors";
import { policyLoopBlockers } from "../lib/policyRun";
import { parseLoopFps, parseRepoId } from "../lib/recording";
import { currentMode } from "../lib/safetyView";
import { usePolling } from "../hooks/usePolling";
import { Dialog } from "../components/ui/Dialog";

/** One request at a time per card; errors are kept until the next attempt. */
function useAction(onSettled: () => void) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
    } catch (caught) {
      setErr(errorMessage(caught));
    } finally {
      setBusy(false);
      onSettled();
    }
  };
  return { busy, err, run };
}

interface EpisodeState {
  active: boolean;
  saving: boolean;
  frames: number;
}

export default function TeleopRecord({ tools, readinessInitiallyOpen = false }: { tools?: ReactNode; readinessInitiallyOpen?: boolean }) {
  const [manual, setManual] = useState(false);
  const [nativeRecording, setNativeRecording] = useState(false);
  const { status } = useSafety();
  return <div className="lg:h-full lg:min-h-0">
    <h1 className="sr-only">Operate</h1>
    <McapWorkflow readinessInitiallyOpen={readinessInitiallyOpen} extraControls={<section className="space-y-3">
      <h2 className="text-sm font-medium">Tools</h2>
      <Button variant="secondary" onClick={() => setManual(true)}>Open manual controls</Button>
      {status?.backend !== "real" && <Button variant="secondary" onClick={() => setNativeRecording(true)}>Advanced dataset tools</Button>}
      {tools}
      <a href="/calibration" className="block text-xs text-cyan-300 hover:underline">Calibration &amp; pose checks →</a>
    </section>} />
    <Dialog open={manual} title="Keyboard & thumbstick control" size="wide"
      description="Stop leader teleop before taking manual control."
      actions={<Button variant="secondary" onClick={() => setManual(false)}>Close manual controls</Button>}>
      {manual && <ManualControlPanel />}
    </Dialog>
    <Dialog open={nativeRecording} title="Advanced dataset tools" size="wide"
      actions={<Button variant="secondary" onClick={() => setNativeRecording(false)}>Close dataset tools</Button>}>
      {nativeRecording && <NativeRecordingWorkspace embedded />}
    </Dialog>
  </div>;
}

export function NativeRecordingWorkspace({ embedded = false }: { embedded?: boolean }) {
  const supervised = useOperatorSupervision();
  const { telemetry, transportStale } = useTelemetry();
  // F28: status and control come from the shell's single poll.
  const { view, status, control, refresh: refreshSafety } = useSafety();
  const cameras = status?.cameras ?? [];
  const preflight = usePolling(getLatestPreflight, 5000, { enabled: status?.backend === "real" });

  const [fpsText, setFpsText] = useState("30");
  const teleopStart = useAction(refreshSafety);
  // Stopping is never blocked by a start in flight.
  const teleopStop = useAction(refreshSafety);

  const [repoIdText, setRepoIdText] = useState("");
  const [task, setTask] = useState("");
  const [recFpsText, setRecFpsText] = useState("30");
  const recStart = useAction(refreshSafety);
  // Starting and saving an episode share one slot so they cannot overlap.
  const recEpisode = useAction(refreshSafety);
  const recStop = useAction(refreshSafety);
  const [lastEpisode, setLastEpisode] = useState<EpisodeResult | null>(null);

  const linkLive = !transportStale && telemetry !== null;
  const mode = currentMode(telemetry, linkLive, view, status);
  const modeKnown = mode !== undefined;
  const teleopOn = mode === "teleop";
  const recordingMode = mode === "recording";
  // Episode state only travels on telemetry (RobotStatus has none), so it is
  // known only while the link is live.
  const episode: EpisodeState | null = linkLive
    ? { active: telemetry.recording, saving: telemetry.recording_saving, frames: telemetry.episode_frames }
    : null;
  // Save may act on the last frame seen: the server refuses when no episode is
  // open, and refusing a save the operator asked for could lose the episode.
  const lastKnownEpisodeOpen = telemetry?.recording === true && telemetry.recording_saving !== true;

  const loopBlockers = [
    ...policyLoopBlockers({ linkLive, supervised, view, control, mode }),
    ...(status?.backend === "real" && (!preflight.data?.ready_for_hardware || preflight.stale)
      ? ["Complete the required preflight checks before starting motion."] : []),
  ];
  const teleopFps = parseLoopFps(fpsText);
  const recFps = parseLoopFps(recFpsText);
  const repo = parseRepoId(repoIdText);
  const teleopBlockers = teleopFps.ok ? loopBlockers : [...loopBlockers, "The FPS is not valid."];
  const recordBlockers = [
    ...loopBlockers,
    ...(repo.ok ? [] : [repo.blank ? repo.error : "The dataset repo_id is not valid."]),
    ...(recFps.ok ? [] : ["The recording FPS is not valid."]),
  ];

  const onStartTeleop = () => {
    // The button is disabled in these cases; this also narrows the types.
    if (!teleopFps.ok || teleopBlockers.length > 0) return;
    const fps = teleopFps.fps;
    void teleopStart.run(() => startTeleop({ fps }));
  };

  const onStartRecording = () => {
    if (!repo.ok || !recFps.ok || recordBlockers.length > 0) return;
    const request: StartRecordingRequest = { repo_id: repo.repoId, fps: recFps.fps };
    const trimmedTask = task.trim();
    // Blank keeps the server's default task.
    if (trimmedTask) request.task = trimmedTask;
    void recStart.run(() => startRecording(request));
  };

  const onSaveEpisode = () => {
    void recEpisode.run(async () => {
      setLastEpisode(await saveEpisode());
    });
  };

  const episodeIdle = !recEpisode.busy && !recStop.busy;
  const canStartEpisode =
    episodeIdle && recordingMode && episode !== null && !episode.active && !episode.saving;
  const canSaveEpisode = episodeIdle && lastKnownEpisodeOpen;

  let recordingValue: string;
  let recordingTone: "neutral" | "warn" | "danger";
  if (episode) {
    recordingValue = episode.saving
      ? "finalizing"
      : episode.active
        ? "episode active"
        : recordingMode
          ? "ready"
          : "no";
    recordingTone = episode.active ? "danger" : episode.saving || recordingMode ? "warn" : "neutral";
  } else if (recordingMode) {
    recordingValue = "episode unknown";
    recordingTone = "warn";
  } else {
    recordingValue = modeKnown ? "no" : "unknown";
    recordingTone = modeKnown ? "neutral" : "warn";
  }

  let episodeNote: string | null = null;
  if (!linkLive) {
    episodeNote = "No live telemetry link, so the episode state is unknown.";
    if (telemetry?.recording_saving) {
      episodeNote += ` Last known: finalizing, ${telemetry.episode_frames} frames.`;
    } else if (telemetry?.recording) {
      episodeNote += ` Last known: episode active, ${telemetry.episode_frames} frames. Save episode still asks the server, which refuses if no episode is open.`;
    }
  } else if (episode?.active) {
    episodeNote = "Save the episode to keep it. Stop recording discards an unsaved episode.";
  } else if (recordingMode && episode?.saving) {
    episodeNote = "The previous episode is still finalizing. Start the next one when it is saved.";
  } else if (!recordingMode) {
    episodeNote = "Start recording to begin episodes.";
  }

  return (
    <div className="space-y-6">
      {embedded ? <h2 className="text-2xl font-bold tracking-tight">Teleop &amp; Record</h2> : <h1 className="text-2xl font-bold tracking-tight">Teleop &amp; Record</h1>}

      <div className="rounded-xl border border-cyan-400/15 bg-cyan-400/[0.04] px-4 py-3 text-xs leading-relaxed text-slate-400">
        Leader-arm teleop and recording are internally supervised, single-owner loops.
        They do not acquire the browser command lease. Starting a loop owns and connects
        the leader and follower devices. Anything that blocks a start is listed under
        its button.
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Teleoperation" className="lg:col-span-1">
          <div className="space-y-4">
            <Field label="FPS" error={teleopFps.ok ? null : teleopFps.error} className="w-40">
              {(field) => (
                <input
                  {...field}
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={240}
                  step={1}
                  value={fpsText}
                  onChange={(e) => setFpsText(e.target.value)}
                  className={INPUT_CLASS}
                />
              )}
            </Field>
            <div className="flex gap-2">
              <Button
                onClick={onStartTeleop}
                disabled={teleopStart.busy || teleopStop.busy || teleopOn || teleopBlockers.length > 0}
              >
                Start teleop
              </Button>
              <Button
                variant="secondary"
                onClick={() => void teleopStop.run(() => stopTeleop())}
                disabled={teleopStop.busy}
              >
                Stop
              </Button>
            </div>
            {!teleopOn && <BlockerList reasons={teleopBlockers} title="Start teleop is blocked" />}
            <div className="flex flex-wrap gap-2">
              <StatusPill
                label="teleop"
                value={modeKnown ? (teleopOn ? "running" : "stopped") : "unknown"}
                tone={teleopOn ? "ok" : modeKnown ? "neutral" : "warn"}
              />
              <StatusPill label="mode" value={mode ?? "unknown"} tone={modeKnown ? "info" : "warn"} />
            </div>
            {[teleopStart.err, teleopStop.err].map(
              (message, index) =>
                message && (
                  <p key={index} role="alert" className="text-sm text-red-300">
                    {message}
                  </p>
                ),
            )}
          </div>
        </Card>

        <Card
          title="Live joints"
          subtitle={linkLive ? "Observed / commanded" : "Last known values · no live telemetry link"}
          className="lg:col-span-2"
        >
          <JointBars observation={telemetry?.observation} action={telemetry?.action} />
        </Card>
      </div>

      {(status?.backend !== "real" || (status.live_cameras?.length ?? 0) > 0) && <Card title="Cameras" subtitle={`${cameras.length} stream(s)`}>
        {cameras.length === 0 ? (
          <p className="text-sm text-slate-400">No cameras reported.</p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {cameras.map((cam) => (
              <CameraView key={cam} cam={cam} active={(status?.live_cameras ?? []).includes(cam)} />
            ))}
          </div>
        )}
      </Card>}

      <Card
        title="Recording"
        actions={
          <>
            <StatusPill label="recording" value={recordingValue} tone={recordingTone} />
            <StatusPill label="frames" value={episode ? episode.frames : "—"} tone={episode ? "info" : "neutral"} />
          </>
        }
      >
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-3">
            <Field
              label="Dataset repo_id"
              error={!repo.ok && !repo.blank ? repo.error : null}
            >
              {(field) => (
                <input
                  {...field}
                  value={repoIdText}
                  onChange={(e) => setRepoIdText(e.target.value)}
                  placeholder="user/so101_pick_place"
                  autoComplete="off"
                  spellCheck={false}
                  className={INPUT_CLASS}
                />
              )}
            </Field>
            <Field label="Task" hint="Blank uses the server's default task.">
              {(field) => (
                <input
                  {...field}
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="Pick up the cube"
                  className={INPUT_CLASS}
                />
              )}
            </Field>
            <Field label="Recording FPS" error={recFps.ok ? null : recFps.error}>
              {(field) => (
                <input
                  {...field}
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={240}
                  step={1}
                  value={recFpsText}
                  onChange={(e) => setRecFpsText(e.target.value)}
                  className={INPUT_CLASS}
                />
              )}
            </Field>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              onClick={onStartRecording}
              disabled={recStart.busy || recStop.busy || recordingMode || recordBlockers.length > 0}
            >
              Start recording
            </Button>
            <Button
              variant="secondary"
              onClick={() => void recEpisode.run(() => startEpisode())}
              disabled={!canStartEpisode}
            >
              Start episode
            </Button>
            <Button variant="success" onClick={onSaveEpisode} disabled={!canSaveEpisode}>
              Save episode
            </Button>
            <Button
              variant="danger"
              onClick={() => void recStop.run(() => stopRecording())}
              disabled={recStop.busy}
            >
              Stop recording
            </Button>
          </div>

          {!recordingMode && (
            <BlockerList reasons={recordBlockers} title="Start recording is blocked" />
          )}
          {episodeNote && <p className="text-xs leading-relaxed text-slate-400">{episodeNote}</p>}

          {lastEpisode && (
            <div
              role="status"
              className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-300"
            >
              Saved episode #{lastEpisode.episode} with {lastEpisode.frames} frames.
            </div>
          )}
          {[recStart.err, recEpisode.err, recStop.err].map(
            (message, index) =>
              message && (
                <p key={index} role="alert" className="text-sm text-red-300">
                  {message}
                </p>
              ),
          )}
        </div>
      </Card>
    </div>
  );
}

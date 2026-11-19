import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { api, getCachedCalibrationPose, getCapabilities, getLatestPreflight, getMcapStatus, readCalibrationPose, saveMcapEpisode, startMcapEpisode, startTeleop, stopTeleop } from "../api";
import { usePolling } from "../hooks/usePolling";
import { useNow } from "../hooks/useNow";
import { useDocumentVisible } from "../hooks/useVisibility";
import { useOperatorSupervision } from "../hooks/useOperatorSupervision";
import { errorMessage } from "../lib/errors";
import { useSafety } from "./safety/SafetyProvider";
import Button from "./Button";
import RobotTwin from "./RobotTwin";
import JointBars from "./JointBars";
import MotorVisionPanel from "./MotorVisionPanel";
import { useTelemetry } from "../hooks/useTelemetry";
import SetupChecklist from "./onboarding/SetupChecklist";
import { Dialog } from "./ui/Dialog";
import { SafetyDetails } from "./safety/SafetyBar";

const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
const RELAY_PORTS: Record<string, number> = { obsbot: 8765, camera_1: 8768, camera_2: 8769, realsense: 8770 };
const InspectionWorkspace = lazy(() => import("./InspectionWorkspace"));

export function cameraPreviewUrl(name: string, origin = window.location.origin) {
  const location = new URL(origin);
  const local = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
  if (local && location.protocol === "http:" && RELAY_PORTS[name]) {
    location.port = String(RELAY_PORTS[name]);
    return `${location.origin}/stream.mjpg`;
  }
  return `/api/mcap/cameras/${name}/stream.mjpg`;
}

function CameraTile({ name, live, fps, paused = false }: { name: string; live: boolean; fps?: number; paused?: boolean }) {
  const [failed, setFailed] = useState(false);
  const [retry, setRetry] = useState(0);
  const [proxy, setProxy] = useState(() => !cameraPreviewUrl(name).startsWith("http"));
  const [frame, setFrame] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestedAt = useRef(0);
  const visible = useDocumentVisible();
  useEffect(() => () => { if (timer.current !== null) clearTimeout(timer.current); }, [visible, paused]);
  useEffect(() => {
    if (!failed || !live || !visible || paused) return;
    const reconnect = window.setTimeout(() => {
      setProxy(!cameraPreviewUrl(name).startsWith("http")); setFailed(false); setRetry(value => value + 1);
    }, 2000);
    return () => window.clearTimeout(reconnect);
  }, [failed, live, visible, paused, name]);
  const slow = live && fps !== undefined && fps < 27;
  return <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-950" aria-label={`${name} camera feed`}>
    <div className="flex shrink-0 flex-wrap items-center justify-between gap-x-2 gap-y-1 px-2 py-1.5 text-[10px]">
      <span className="font-medium text-slate-200">{name === "obsbot" ? "OBSBOT · workspace" : name === "realsense" ? "D435i · color" : name.replace("camera_", "Camera ")}</span>
      <span className={live && !failed && !slow ? "text-emerald-300" : "text-amber-200"}>{failed ? "Preview unavailable" : live ? "Live" : "Offline"}{fps !== undefined ? ` · ${fps.toFixed(1)} fps` : ""}{slow ? " · below target" : ""}</span>
    </div>
    {live && visible && !paused && !failed ? <img key={`${retry}-${proxy}`} src={proxy ? `/api/mcap/cameras/${name}/snapshot.jpg?frame=${retry}-${frame}` : cameraPreviewUrl(name)} alt={`${name} arm camera`}
      onLoad={() => {
        if (!proxy) return;
        const interval = 1000 / Math.min(30, fps && fps > 0 ? fps : 15);
        if (!requestedAt.current) requestedAt.current = performance.now();
        timer.current = setTimeout(() => { requestedAt.current = performance.now(); setFrame(value => value + 1); }, Math.max(0, interval - (performance.now() - requestedAt.current)));
      }}
      onError={() => { if (!proxy && cameraPreviewUrl(name).startsWith("http")) setProxy(true); else setFailed(true); }} className="aspect-video min-h-0 w-full flex-1 object-contain lg:aspect-auto" />
      : <div className="flex aspect-video min-h-0 flex-1 items-center justify-center p-4 text-center text-xs text-slate-400 lg:aspect-auto">
        {failed ? <button onClick={() => { setFailed(false); setRetry(retry + 1); }} className="text-cyan-300 underline">Retry camera preview</button>
          : paused ? "Preview paused during inspection." : visible ? "Waiting for fresh camera frames." : "Preview paused while this page is hidden."}
      </div>}
  </section>;
}

export function encoderSampleFresh(capturedAt: number, receivedAt: number | null, clock: number): boolean {
  const age = Math.max(clock, receivedAt ?? clock) - capturedAt * 1000;
  return age >= 0 && age <= 500;
}

function ReadinessChecklist() {
  const capabilities = usePolling(getCapabilities, 15000);
  return <SetupChecklist capabilities={capabilities} />;
}

export default function McapWorkflow({ extraControls, readinessInitiallyOpen = false }: { extraControls?: ReactNode; readinessInitiallyOpen?: boolean }) {
  const { status, statusStale, control, view, refresh } = useSafety();
  const supervised = useOperatorSupervision();
  const { telemetry, transportStale, hasLiveObservation } = useTelemetry();
  const recording = usePolling(getMcapStatus, 1000);
  const preflight = usePolling(getLatestPreflight, 5000, { enabled: status?.backend === "real" });
  const [watching, setWatching] = useState(false);
  const [supported, setSupported] = useState(false);
  const readable = status?.backend === "real" && !status.mock && !statusStale && status.mode === "idle"
    && status.torque_state === "off" && !control?.armed && !control?.lease && supported;
  if (watching && !readable) setWatching(false);
  const sampled = usePolling(() => readCalibrationPose("follower"), 100,
    { enabled: watching && readable, staleAfterFailures: 1 });
  const cached = usePolling(() => getCachedCalibrationPose("follower"), 250,
    { enabled: status?.backend === "real" && !watching });
  const pose = sampled.data && (!cached.data || sampled.data.captured_at >= cached.data.captured_at) ? sampled.data : cached.data;
  const poseReceivedAt = pose === sampled.data ? sampled.lastSuccessAt : cached.lastSuccessAt;
  const now = useNow(100);
  const [task, setTask] = useState("");
  const [busy, setBusy] = useState(false);
  const starting = useRef(false);
  const cancelled = useRef(false);
  const episodeId = useRef<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showVision, setShowVision] = useState(false);
  const [showJoints, setShowJoints] = useState(false);
  const [showSetup, setShowSetup] = useState(false);
  const [inspection, setInspection] = useState<"live" | "recordings" | null>(null);
  const data = recording.data;
  const teleop = !statusStale && status?.mode === "teleop";
  const recorderLive = data?.recorder.recording === true && !recording.stale;
  const recorderReady = (recorderLive || data?.recorder.service_ready === true) && !recording.stale;
  const blockers = [
    ...(!view.known || statusStale ? ["Waiting for the robot connection."] : []),
    ...(transportStale ? ["Waiting for the live telemetry connection."] : []),
    ...(view.tripped ? ["Clear the reported safety fault first."] : []),
    ...(!supervised ? ["Keep this page visible and connected."] : []),
    ...(control?.armed || control?.lease ? ["Release manual control before leader teleop."] : []),
    ...(status?.mode !== "idle" && !teleop ? ["Stop the current operation first."] : []),
    ...(status?.backend === "real" && (!preflight.data?.ready_for_hardware || preflight.stale)
      ? ["Complete the required preflight checks before starting motion."] : []),
    ...(!recorderReady ? ["Start the local MCAP recording service."] : []),
    ...(data?.active ? ["Finish the current episode before starting another."] : []),
  ];
  const act = async (action: () => Promise<unknown>) => {
    setBusy(true); setError(null);
    try { await action(); }
    catch (caught) { setError(errorMessage(caught)); }
    finally { setBusy(false); recording.refresh(); refresh(); }
  };
  const begin = () => {
    if (starting.current) return;
    starting.current = true;
    cancelled.current = false;
    void act(async () => {
      try {
        await startTeleop({ fps: 30 });
        if (cancelled.current) { await stopTeleop(); return; }
        const episode = await startMcapEpisode(task);
        episodeId.current = episode.id;
        if (cancelled.current) { episodeId.current = null; await stopTeleop(); }
      } catch (caught) {
        try { await stopTeleop(); }
        catch (stopError) { throw Object.assign(new Error(`${errorMessage(caught)}; Halt failed: ${errorMessage(stopError)}. Use E-stop.`), { cause: stopError }); }
        throw caught;
      } finally { starting.current = false; }
    });
  };
  const end = () => {
    cancelled.current = true;
    const id = episodeId.current ?? data?.active?.id;
    void act(async () => {
      let saveError: unknown;
      try {
        if (id && !starting.current) {
          const episode = await saveMcapEpisode(id, 1000);
          if (episode.status !== "saved") throw new Error("Episode was interrupted; review the capture before using it.");
          episodeId.current = null;
        }
      } catch (caught) { saveError = caught; }
      try { await stopTeleop(); }
      catch (caught) { throw Object.assign(new Error(`${saveError ? `${errorMessage(saveError)}; ` : ""}Halt failed: ${errorMessage(caught)}. Use E-stop.`), { cause: caught }); }
      if (saveError) throw saveError;
    });
  };
  const complete = telemetry && JOINTS.every((joint) => Number.isFinite(telemetry.observation[`${joint}.pos`]));
  const physical = status?.backend === "real" && !status.mock;
  const cachedPose = physical && pose?.arm === "follower" && pose.backend === "real" && pose.complete
    && JOINTS.every((joint) => Number.isFinite(pose.observation[`${joint}.pos`])) ? pose : null;
  const useControl = Boolean(complete && (!cachedPose || (telemetry?.t ?? 0) >= cachedPose.captured_at));
  const observation = useControl ? telemetry?.observation : cachedPose?.observation;
  const poseLive = useControl
    ? hasLiveObservation && !transportStale && !statusStale
    : Boolean(cachedPose && encoderSampleFresh(cachedPose.captured_at, poseReceivedAt, now) && !(watching ? sampled.stale || sampled.error : cached.stale) && !statusStale);
  const saved = data?.episodes.filter((episode) => episode.status === "saved").length ?? 0;
  const samplingPaused = Boolean(observation && !poseLive && !teleop && status?.mode === "idle" && !statusStale && !transportStale && !watching);
  const cameras = Object.entries(data?.cameras ?? {}).sort(([first], [second]) => (first === "obsbot" ? -1 : second === "obsbot" ? 1 : first.localeCompare(second)));

  return <div data-testid="operate-workflow" className="grid gap-3 lg:h-full lg:min-h-0 lg:grid-cols-[minmax(0,3fr)_minmax(280px,1fr)] lg:overflow-hidden">
    <section aria-label="Live cameras and arm" data-testid="operate-visuals" className="grid min-h-0 gap-3 lg:grid-rows-[minmax(0,1fr)_minmax(0,1fr)] lg:overflow-hidden">
      <div aria-label="Cameras" className="grid min-h-0 grid-cols-2 gap-2 lg:auto-rows-fr">
        {cameras.map(([name, camera]) => <CameraTile key={name} name={name} paused={inspection !== null} live={camera.live && !recording.stale} fps={camera.actual_preview_fps} />)}
        {!cameras.length && <p className="col-span-2 p-5 text-sm text-slate-400">Waiting for camera status.</p>}
      </div>
      <section aria-label="Follower arm" className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60">
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2 text-xs">
          <h2 className="font-semibold">Follower arm · encoder view</h2>
          <span className={poseLive ? "text-emerald-300" : samplingPaused ? "text-slate-300" : "text-amber-200"}>{poseLive ? "Fresh encoder sample" : samplingPaused ? "Sampling paused · last read" : observation ? "Last known pose" : "No encoder pose yet"}</span>
        </div>
        <RobotTwin fitArm observation={observation} action={useControl ? telemetry?.action : undefined}
          className="h-[300px] min-h-0 flex-1 lg:h-full" sourceLabel={physical ? "Follower encoders" : "Rehearsal telemetry"}
          stale={!poseLive && !samplingPaused && !!observation} idle={samplingPaused} mode={teleop ? "teleop" : "idle"} />
      </section>
    </section>

    <section aria-label="Teleop controls and readiness" data-testid="operate-controls" tabIndex={0}
      className="min-w-0 space-y-4 rounded-xl border border-slate-800 bg-slate-900/60 p-3 pt-16 lg:min-h-0 lg:overflow-y-auto lg:overscroll-contain">
      <section className="space-y-3">
        <div className="space-y-1 text-xs text-slate-400" role="status">
          <p className={recorderReady ? "text-emerald-300" : "text-amber-200"}>MCAP {!data && !recording.error ? "checking service…" : recorderLive ? "recording" : recorderReady ? "ready · waiting for activity" : "unavailable"}</p>
          <p>{teleop ? "Leader teleop running" : "Leader teleop stopped"} · {data?.active ? "● Episode recording" : "No episode open"}</p>
          {data && <p className="font-mono text-slate-200">{saved} saved recordings · review required</p>}
        </div>
        <label className="block text-xs text-slate-400">Episode task
          <input className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white focus:border-cyan-400 focus-visible:ring-2 focus-visible:ring-cyan-400/40"
            value={task} onChange={(event) => setTask(event.target.value)} maxLength={1000}
            placeholder="Pick up the cube and place it in the tray" disabled={!!data?.active} />
        </label>
        <div className="grid gap-2">
          <Button disabled={busy || teleop || blockers.length > 0} onClick={begin}>Start Teleop</Button>
          <Button variant="danger" onClick={end}>Stop Teleop</Button>
        </div>
        <p className="text-[11px] text-muted">Start opens an MCAP episode automatically. Stop saves it and Halts; the hold-to-Collapse timer applies. Halt and E-stop above act immediately.</p>
        {!teleop && blockers.length > 0 && <details className="rounded-lg border border-amber-400/20 bg-amber-400/5 p-2">
          <summary className="cursor-pointer text-xs font-medium text-amber-200">Motion blocked · view required checks</summary>
          <ul className="mt-2 space-y-1 text-xs text-amber-100/80">{blockers.map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </details>}
        {(error || recording.error) && <p role="alert" className="text-sm text-rose-300">{error ?? recording.error?.message}</p>}
      </section>

      <details key={readinessInitiallyOpen ? "requested" : "default"} open={readinessInitiallyOpen} className="border-t border-slate-800 pt-3">
        <summary className="cursor-pointer text-xs font-semibold text-cyan-300">Preflight · {preflight.data?.ready_for_hardware && !preflight.stale ? "passed" : "review checks"}</summary>
        <div className="mt-3 space-y-3 text-xs text-slate-400">
          {preflight.data?.created_at && <p>Report: {new Date(preflight.data.created_at).toLocaleString()}</p>}
          {preflight.error && <p role="status">{preflight.error.message}</p>}
          {!preflight.data && <p>No current preflight report. Run so101 preflight in a terminal with the workstation idle.</p>}
          {preflight.data?.checks.map((check) => <section key={check.key} className="rounded-lg border border-slate-800 p-2">
            <h3 className={check.status === "pass" ? "text-emerald-300" : check.blocking ? "text-amber-200" : "text-slate-300"}>{check.title} · {check.status}</h3>
            <p className="mt-1">{check.message}</p>
            {check.remediation && <p className="mt-1 text-slate-300">{check.remediation}</p>}
          </section>)}
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={preflight.refresh}>Refresh report</Button>
            <Button variant="secondary" onClick={() => setShowSetup(true)}>Setup checklist</Button>
          </div>
          <p>Refreshing reads saved results; it never runs hardware checks automatically.</p>
        </div>
      </details>

      <section className="space-y-3 border-t border-slate-800 pt-3">
        <h2 className="text-sm font-semibold">Calibration &amp; pose checks</h2>
        <SafetyDetails />
        {physical && status?.mode === "idle" && <details className="text-xs text-slate-400">
          <summary className="cursor-pointer text-cyan-300">Read the supported arm here</summary>
          <label className="my-3 flex items-center gap-2"><input type="checkbox" checked={supported} disabled={watching} onChange={(event) => setSupported(event.target.checked)} />Arm supported, torque off, workspace clear</label>
          <Button variant="secondary" disabled={!watching && (!readable || busy)}
            onClick={() => { setWatching(!watching); setShowJoints(true); }}>{watching ? "Stop live readings" : "Read current pose"}</Button>
          <p className="mt-2">10 Hz target while visible and torque-off. Stop readings before stopping the motor service. No torque, limits or positions are changed.</p>
          {watching && sampled.error && <p role="alert" className="mt-2 text-amber-200">Live read unavailable: {sampled.error.message}</p>}
        </details>}
        <button className="text-xs text-slate-300 underline" onClick={() => setShowJoints(!showJoints)}>{showJoints ? "Hide joint readings" : "Show joint readings"}</button>
        {showJoints && <div className="space-y-3"><JointBars observation={observation} action={useControl ? telemetry?.action : undefined} />
          <p className="text-xs text-slate-200">Motor 2 · shoulder lift: <strong className="font-mono">{Number.isFinite(observation?.["shoulder_lift.pos"]) ? `${observation!["shoulder_lift.pos"].toFixed(2)}°` : "No reading"}</strong></p>
        </div>}
      </section>

      <section className="space-y-3 border-t border-slate-800 pt-3">
        <h2 className="text-sm font-semibold">Episode log</h2>
        {!data ? <p className="text-xs text-slate-400">Loading recordings…</p> : data.episodes.length === 0 ? <p className="text-xs text-slate-400">Your first saved demonstration will appear here.</p> :
          <ul className="space-y-2 text-xs">{data?.episodes.slice().reverse().map((episode) => <li key={episode.id} className="rounded-lg bg-slate-950 p-2">
            <p className="flex justify-between gap-2"><span className="font-mono">{episode.id.slice(0, 8)}</span><span>{episode.status}</span></p>
            <p className="mt-1 text-slate-400">{episode.task || "Untitled"}</p>
          </li>)}</ul>}
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => setInspection("live")}>Inspect live data</Button>
          <Button variant="secondary" onClick={() => setInspection("recordings")}>Review recordings</Button>
          <Button variant="secondary" disabled={busy || status?.mode !== "idle" || statusStale || !saved}
            onClick={() => act(async () => {
              const latest = data?.episodes.slice().reverse().find((episode) => episode.status === "saved");
              if (latest) await api.post(`/api/mcap/reviews/${latest.id}/open-foxglove`, {}, 60000);
            })}>Open in Foxglove</Button>
        </div>
        <button className="text-xs text-slate-400 underline" onClick={() => setShowVision(!showVision)}>{showVision ? "Close computer vision tools" : "Computer vision diagnostics"}</button>
        {showVision && <MotorVisionPanel />}
      </section>
      {extraControls && <div className="border-t border-slate-800 pt-3">{extraControls}</div>}
    </section>
    <Dialog open={showSetup} title="Workstation setup" size="wide" actions={<Button variant="secondary" onClick={() => setShowSetup(false)}>Close setup</Button>}>
      {showSetup && <ReadinessChecklist />}
    </Dialog>
    <Dialog open={inspection !== null} title="Data inspection" size="workspace" actions={<Button variant="secondary" onClick={() => setInspection(null)}>Close inspection</Button>}>
      {inspection && <Suspense fallback={<p role="status">Loading inspection workspace…</p>}><InspectionWorkspace initialTab={inspection} /></Suspense>}
    </Dialog>
  </div>;
}

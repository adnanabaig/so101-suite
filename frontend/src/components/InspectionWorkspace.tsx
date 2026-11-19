import { memo, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, getControlTicks, getMcapReviews, getMcapSnapshot, getMcapTimeline, startMcapPlayback, type ControlTick, type McapTimeline } from "../api";
import { usePolling } from "../hooks/usePolling";
import { useNow } from "../hooks/useNow";
import { useSafety } from "./safety/SafetyProvider";
import Button from "./Button";
import RobotTwin from "./RobotTwin";
import { Tabs } from "./ui/Tabs";
import CameraTuning from "./CameraTuning";
import CalibrationPoseView from "./CalibrationPoseView";

const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
const LIVE_TOPICS: Record<string, string> = {
  "Motor health": "/api/robot/health", "Robot status": "/api/robot/status",
  "Stops and ownership": "/api/control/state", "Recorder and episodes": "/api/mcap/status",
  "Preflight": "/api/preflight/latest", "Calibration and rest pose": "/api/mcap/evidence",
};

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function number(value: unknown, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function HealthTable({ health }: { health: unknown }) {
  const arms = object(object(health).arms);
  return <div className="space-y-3">
    <p className="text-xs text-amber-200">Servo current conversion and status-bit interpretation may be unverified. Read the recorded reasons; current is not a calibrated supply-current measurement.</p>
    {Object.entries(arms).map(([arm, armValue]) => <section key={arm}>
      <h4 className="mb-1 text-xs font-semibold capitalize">{arm}</h4>
      <p className="mb-2 text-xs text-slate-400">{String(object(armValue).active === false ? "Sampling paused" : "Hardware sample")} · age {number(object(armValue).sample_age_s)} s · {String(object(armValue).inactive_reason ?? object(armValue).error ?? "")}</p>
      <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr>{["Motor", "State", "°C", "V", "mA", "Raw status"].map((title) => <th className="p-1" key={title}>{title}</th>)}</tr></thead>
        <tbody>{Object.entries(object(object(armValue).motors)).map(([joint, value]) => {
          const motor = object(value), sample = object(motor.sample);
          return <tr key={joint} className={motor.level === "ok" ? "text-slate-300" : "text-amber-200"}>
            <th className="p-1 font-normal">{String(sample.motor_id ?? JOINTS.indexOf(joint) + 1)} · {joint}</th><td title={JSON.stringify(motor.reasons)}>{String(motor.level ?? "unknown")}</td>
            <td>{number(sample.temperature_c)}</td><td>{number(sample.voltage_v)}</td><td>{number(sample.current_ma)}</td><td>{String(sample.status_raw ?? "—")}</td>
          </tr>;
        })}</tbody></table></div>
    </section>)}
    {!Object.keys(arms).length && <p className="text-xs text-slate-400">No motor health sample at this time.</p>}
  </div>;
}

function JointTable({ tick, profile }: { tick?: ControlTick; profile?: ControlTick["control_profile"] }) {
  return <div className="overflow-x-auto"><table className="w-full text-left text-xs">
    <thead><tr>{["Motor", "Leader", "Accepted", "Measured", "Operating range", "Bound"].map((title) => <th key={title} className="p-1">{title}</th>)}</tr></thead>
    <tbody>{JOINTS.map((joint, index) => {
      const key = `${joint}.pos`, desired = tick?.desired_action?.[key], accepted = tick?.action?.[key];
      const limits = profile?.limits?.[joint];
      const outside = limits && desired !== undefined && (desired < limits[0] || desired > limits[1]);
      const limited = desired !== undefined && accepted !== undefined && Math.abs(desired - accepted) > 0.1;
      return <tr key={joint} className={outside ? "text-amber-200" : "text-slate-300"}>
        <th className="p-1 font-normal">{index + 1} · {joint}{joint === "gripper" ? " (%)" : " (°)"}</th>
        <td>{number(desired)}</td><td>{number(accepted)}</td><td>{number(tick?.observation?.[key])}</td>
        <td>{limits ? `${number(limits[0])}…${number(limits[1])}` : "—"}</td>
        <td>{outside ? "Position limit" : limited ? "Command/rate limit" : "—"}</td>
      </tr>;
    })}</tbody>
  </table><p className="mt-2 text-xs text-slate-400">Operating limits are software bounds, not measured physical end stops. A leader outside them can create an end-range dead zone.</p></div>;
}

const TraceChart = memo(function TraceChart({ ticks }: { ticks: ControlTick[] }) {
  const [joint, setJoint] = useState("wrist_roll");
  const [timing, setTiming] = useState(false);
  const data = useMemo(() => {
    const first = ticks[0]?.timestamp_ns ?? 0;
    return ticks.map((tick) => ({
      seconds: tick.seconds ?? (tick.timestamp_ns - first) / 1e9,
      leader: tick.desired_action?.[`${joint}.pos`], accepted: tick.action?.[`${joint}.pos`], measured: tick.observation?.[`${joint}.pos`],
      motor: tick.timing?.motor_step_seconds === undefined ? null : tick.timing.motor_step_seconds * 1000,
      lock: tick.timing?.io_lock_wait_seconds === undefined ? null : tick.timing.io_lock_wait_seconds * 1000,
      cycle: tick.timing?.cycle_interval_seconds == null ? null : tick.timing.cycle_interval_seconds * 1000,
      previousTail: tick.timing?.previous_tail_seconds == null ? null : tick.timing.previous_tail_seconds * 1000,
      previousLateness: tick.timing?.previous_pacing?.deadline_lateness_seconds === undefined ? null : tick.timing.previous_pacing.deadline_lateness_seconds * 1000,
    }));
  }, [ticks, joint]);
  const paths = timing ? [["cycle", "Cycle interval", "#94a3b8"], ["motor", "Motor/control step", "#22d3ee"], ["lock", "Bus lock wait", "#fbbf24"], ["previousTail", "Previous publish tail", "#34d399"], ["previousLateness", "Previous deadline lateness", "#fb7185"]] : [["leader", "Leader intent", "#fbbf24"], ["accepted", "Accepted command", "#34d399"], ["measured", "Follower measured", "#22d3ee"]];
  return <section className="space-y-2">
    <div className="flex flex-wrap items-center gap-3 text-xs">
      <label>Plot <select aria-label="Plot type" value={String(timing)} onChange={(event) => setTiming(event.target.value === "true")} className="ml-2 rounded bg-slate-800 p-1"><option value="false">Joint tracking</option><option value="true">Control timing (ms)</option></select></label>
      {!timing && <label>Joint <select aria-label="Plot joint" value={joint} onChange={(event) => setJoint(event.target.value)} className="ml-2 rounded bg-slate-800 p-1">{JOINTS.map((name, index) => <option key={name} value={name}>{index + 1} · {name}</option>)}</select></label>}
    </div>
    <div className="h-56 w-full min-w-0" role="img" aria-label={timing ? "Control phase timings in milliseconds" : `${joint} recorded leader, accepted command and measured positions`}>
      <ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ left: 5, right: 10 }}><CartesianGrid stroke="#334155" /><XAxis dataKey="seconds" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(value: number) => value.toFixed(1)} /><YAxis width={45} /><Tooltip /><Legend />{paths.map(([key, label, color]) => <Line key={key} name={label} dataKey={key} stroke={color} dot={false} isAnimationActive={false} connectNulls={false} />)}</LineChart></ResponsiveContainer>
    </div>
    {timing && <p className="text-xs text-slate-400">Tail and deadline lateness belong to the previous cycle. Older recordings may not contain phase timing.</p>}
  </section>;
});

function LiveInspection() {
  const now = useNow(1000);
  const { status, statusStale } = useSafety();
  const cursor = useRef<{ stream: string; seq: number; ticks: ControlTick[]; missing: number }>({ stream: "", seq: 0, ticks: [], missing: 0 });
  const trace = usePolling(async () => {
    let batch = await getControlTicks(cursor.current.seq, !cursor.current.stream);
    if (!cursor.current.stream && batch.last_seq) batch = await getControlTicks(Math.max(0, batch.last_seq - 1));
    const prior = cursor.current.stream === batch.stream_id ? cursor.current.ticks : [];
    cursor.current = { stream: batch.stream_id, seq: batch.last_seq, ticks: [...prior, ...batch.ticks].slice(-600), missing: batch.missing_before_buffer };
    return cursor.current;
  }, 100);
  const health = usePolling(() => api.get<unknown>("/api/robot/health"), 1000);
  const [topic, setTopic] = useState("Motor health");
  const raw = usePolling(() => api.get<unknown>(LIVE_TOPICS[topic]), 1000, { key: topic });
  const latest = trace.data?.ticks[trace.data.ticks.length - 1];
  const sampleAge = latest ? Math.max(0, now - latest.timestamp_ns / 1e6) : null;
  return <div className="space-y-5">
    <p className="text-xs text-slate-400">Passive live inspection · last 600 control samples · {trace.stale ? "trace connection stale" : "trace connection available"}. No encoder reads or motion are started here.</p>
    <p className="text-xs text-amber-200">{status?.mode === "idle" && !statusStale ? "Control sampling paused · last recorded control tick" : sampleAge !== null && sampleAge > 500 ? "Control sample stale" : "Latest control sample"} · age {sampleAge === null ? "unavailable" : `${(sampleAge / 1000).toFixed(1)} s`}</p>
    {trace.error && <p role="alert" className="text-amber-200">{trace.error.message}</p>}
    {Boolean(trace.data?.missing) && <p role="status" className="text-amber-200">Control samples were overwritten before inspection; this chart is incomplete.</p>}
    {!latest && <p className="text-xs text-slate-300">No control ticks yet. An idle arm has no new control-loop samples.</p>}
    <JointTable tick={latest} profile={latest?.control_profile} />
    <TraceChart ticks={trace.data?.ticks ?? []} />
    {(health.error || health.stale) && <p role="alert" className="text-amber-200">Health connection stale · showing the last available sample. {health.error?.message}</p>}
    <HealthTable health={health.data} />
    <details><summary className="cursor-pointer text-sm text-cyan-300">Full fields and evidence</summary>
      <label className="mt-3 block text-xs">Source <select value={topic} onChange={(event) => setTopic(event.target.value)} className="ml-2 rounded bg-slate-800 p-1">{Object.keys(LIVE_TOPICS).map((name) => <option key={name}>{name}</option>)}</select></label>
      {raw.error && <p role="alert">{raw.error.message}</p>}
      <pre className="mt-2 max-h-72 overflow-auto rounded bg-slate-950 p-3 text-[11px]">{JSON.stringify(raw.data, null, 2)}</pre>
      <p className="mt-2 text-xs">Authoritative control profile and raw tick:</p><pre className="max-h-72 overflow-auto rounded bg-slate-950 p-3 text-[11px]">{JSON.stringify(latest ?? null, null, 2)}</pre>
    </details>
  </div>;
}

function RecordedCamera({ camera, source }: { camera: string; source: string }) {
  const [failed, setFailed] = useState(false);
  return <figure className="min-w-0 rounded border border-slate-700 p-1"><figcaption className="text-[10px] text-slate-300">Recorded {camera}</figcaption>
    {failed ? <p className="aspect-video p-3 text-xs text-amber-200">No recorded frame available at this cursor, or playback stream unavailable.</p> : <img alt={`Recorded ${camera}`} className="aspect-video w-full object-contain" src={source} onError={() => setFailed(true)} />}
  </figure>;
}

function QualityReport({ review }: { review: unknown }) {
  const report = object(review), cameras = object(report.cameras);
  return <section className="space-y-3 text-xs">
    <p>Historical OBSBOT/control integrity checks: {report.data_checks_passed === undefined ? "not reviewed" : report.data_checks_passed ? "passed" : "review required"}. Human task acceptance remains separate.</p>
    <table className="w-full text-left"><thead><tr>{["Camera", "Measured fps", "Largest gap", "30 fps target"].map(label => <th key={label} className="p-1">{label}</th>)}</tr></thead><tbody>{["obsbot", "camera_1", "camera_2", "realsense"].map(name => {
      const camera = object(cameras[name]);
      const passed = typeof camera.measured_fps === "number" && camera.measured_fps >= 27 && typeof camera.max_gap_seconds === "number" && camera.max_gap_seconds <= 0.15;
      return <tr key={name}><th className="p-1 font-normal">{name}</th><td>{number(camera.measured_fps, 2)}</td><td>{typeof camera.max_gap_seconds === "number" ? `${(camera.max_gap_seconds * 1000).toFixed(1)} ms` : "—"}</td><td className={passed ? "text-emerald-300" : "text-amber-200"}>{Object.keys(camera).length ? passed ? "Rate/gap checks pass" : "Below target / gaps" : "Not reviewed"}</td></tr>;
    })}</tbody></table>
    <p>These rate/gap rows do not prove capture-clock provenance, endpoint coverage, motor safety, geometric alignment or task success.</p>
    <details><summary className="text-cyan-300 cursor-pointer">Full quality and health report</summary><pre className="mt-2 max-h-80 overflow-auto rounded bg-slate-950 p-3 text-xs">{JSON.stringify(review ?? { status: "Not reviewed yet" }, null, 2)}</pre></details>
  </section>;
}

function RecordedTopic({ timeline, seconds }: { timeline: McapTimeline; seconds: number }) {
  const topics = Object.keys(timeline.topics).filter((topic) => topic !== "/robot/control_tick");
  const [chosen, setChosen] = useState("/robot/health");
  const topic = topics.includes(chosen) ? chosen : topics[0] ?? "";
  const position = Math.floor(seconds * 4) / 4;
  const data = usePolling(() => getMcapSnapshot(timeline.episode.id, topic, Math.min(position, timeline.duration_seconds)), 0, { key: `${timeline.episode.id}:${topic}:${position}`, enabled: Boolean(topic) });
  return <div className="space-y-3">
    <label className="block text-xs">Recorded source <select className="ml-2 max-w-full rounded bg-slate-800 p-1" aria-label="Recorded source" value={topic} onChange={(event) => setChosen(event.target.value)}>{topics.map((name) => <option key={name}>{name}</option>)}</select></label>
    {data.error && <p role="alert" className="text-amber-200">{data.error.message}</p>}
    {topic === "/robot/health" && <HealthTable health={data.data?.record?.data} />}
    <p className="text-xs text-slate-400">Nearest preceding retained message{data.data?.record ? ` at ${data.data.record.seconds.toFixed(3)} s` : " unavailable at this cursor"}. Original messages remain in MCAP.</p>
    <pre className="max-h-80 overflow-auto rounded bg-slate-950 p-3 text-[11px]">{JSON.stringify(data.data?.record?.data ?? null, null, 2)}</pre>
  </div>;
}

function RecordedPlayer({ timeline, review }: { timeline: McapTimeline; review: unknown }) {
  const [seconds, setSeconds] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [session, setSession] = useState<{ id: string; started: number; seconds: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    if (!session) return;
    const timer = window.setInterval(() => {
      const next = session.seconds + Math.max(0, performance.now() - session.started) / 1000 * speed;
      if (next >= timeline.duration_seconds) setSession(null);
      setSeconds(Math.min(next, timeline.duration_seconds));
    }, 33);
    return () => window.clearInterval(timer);
  }, [session, speed, timeline.duration_seconds]);
  async function play() {
    setStarting(true); setError(null);
    try {
      const began = performance.now();
      const result = await startMcapPlayback(timeline.episode.id, seconds, speed);
      setSession({ id: result.playback_id, started: performance.now() + Math.max(0, result.delay_seconds * 1000 - (performance.now() - began) / 2), seconds });
    } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
    finally { setStarting(false); }
  }
  const index = useMemo(() => {
    let low = 0, high = timeline.ticks.length;
    while (low < high) { const middle = (low + high) >>> 1; if ((timeline.ticks[middle].seconds ?? 0) <= seconds) low = middle + 1; else high = middle; }
    return Math.max(0, low - 1);
  }, [timeline.ticks, seconds]);
  const point = timeline.ticks[index];
  return <div className="space-y-4">
    <p className="text-xs text-cyan-200">RECORDED PLAYBACK · {timeline.episode.id} · {timeline.episode.task || "Untitled tuning recording"} · sends no robot commands</p>
    {timeline.samples_without_episode_identity > 0 && <p role="alert" className="text-xs text-amber-200">Legacy recording: {timeline.samples_without_episode_identity} control samples lack an embedded episode ID. The saved interval selects this playback; per-tick episode provenance is unverified.</p>}
    <div className="flex flex-wrap items-center gap-3">
      <Button disabled={starting} onClick={() => session ? setSession(null) : void play()}>{session ? "Pause playback" : "Play recording"}</Button>
      <label className="text-xs">Speed <select aria-label="Playback speed" value={speed} disabled={Boolean(session)} onChange={(event) => setSpeed(Number(event.target.value))} className="rounded bg-slate-800 p-1">{[0.25, 0.5, 1, 2].map((value) => <option key={value} value={value}>{value}×</option>)}</select></label>
      <span className="font-mono text-xs">{seconds.toFixed(2)} / {timeline.duration_seconds.toFixed(2)} s</span>
      <Button variant="secondary" disabled={starting} onClick={() => {
        setStarting(true); setError(null); setNotice(null);
        void api.post(`/api/mcap/reviews/${timeline.episode.id}/open-foxglove`, {}, 60000)
          .then(() => setNotice(`Foxglove launch requested for ${timeline.episode.id.slice(0, 8)}.`))
          .catch(caught => setError(caught instanceof Error ? caught.message : String(caught)))
          .finally(() => setStarting(false));
      }}>Open this episode in Foxglove</Button>
    </div>
    <input aria-label="Recorded timeline" type="range" min={0} max={timeline.duration_seconds} step={0.033} value={seconds} onChange={(event) => { setSession(null); setSeconds(Number(event.target.value)); }} className="w-full" />
    {error && <p role="alert" className="text-amber-200">{error}</p>}
    {notice && <p role="status" className="text-cyan-200">{notice}</p>}
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="grid grid-cols-2 gap-2">{timeline.cameras.map((camera) => {
        const source = session ? `/api/mcap/reviews/${timeline.episode.id}/streams/${camera}?playback_id=${encodeURIComponent(session.id)}` : `/api/mcap/reviews/${timeline.episode.id}/frames/${camera}?seconds=${seconds}`;
        return <RecordedCamera key={source} camera={camera} source={source} />;
      })}</div>
      <RobotTwin fitArm observation={point?.observation} action={point?.action} mode="recorded playback" sourceLabel="Recorded follower encoders" className="h-72" />
    </div>
    <p className="text-xs text-slate-400">{timeline.sample_count} raw control samples · {timeline.measured_control_hz.toFixed(2)} Hz · max gap {(timeline.max_control_gap_seconds * 1000).toFixed(1)} ms. Charts show {timeline.display_sample_count} samples. Paused camera previews choose the nearest original frame within 500 ms; playing streams retain original frame timing.</p>
    <JointTable tick={point} profile={timeline.profile} />
    <TraceChart ticks={timeline.ticks} />
    <Tabs label="Recorded evidence" tabs={[
      { id: "fields", label: "Health, metadata & raw fields", panel: <RecordedTopic timeline={timeline} seconds={seconds} /> },
      { id: "review", label: "Quality report", panel: <QualityReport review={review} /> },
      { id: "profile", label: "Motor profile", panel: <pre className="max-h-80 overflow-auto rounded bg-slate-950 p-3 text-xs">{JSON.stringify(timeline.profile, null, 2)}</pre> },
    ]} />
  </div>;
}

function RecordedInspection() {
  const { status, statusStale } = useSafety();
  const idle = Boolean(status && status.mode === "idle" && !statusStale);
  const catalog = usePolling(getMcapReviews, 5000);
  const [selected, setSelected] = useState("");
  const recording = catalog.data?.recordings.find((item) => item.episode.id === selected);
  const timeline = usePolling(() => getMcapTimeline(selected), 0, { enabled: idle && Boolean(selected), key: selected });
  return <div className="space-y-4">
    <label className="block text-sm">Recording <select aria-label="Recording" value={selected} onChange={(event) => setSelected(event.target.value)} className="ml-2 max-w-full rounded bg-slate-800 p-2"><option value="">Choose a saved recording</option>{catalog.data?.recordings.map((item) => <option key={item.episode.id} value={item.episode.id} disabled={!item.recording_available}>{item.episode.id.slice(0, 8)} · {item.episode.status} · {item.episode.task || "Untitled"}{!item.recording_available ? " · no finalized MCAP" : ""}</option>)}</select></label>
    {(catalog.error || timeline.error) && <p role="alert" className="text-amber-200">{catalog.error?.message ?? timeline.error?.message}</p>}
    {!idle && <p className="text-amber-200">Stop motion before loading or playing a recording. Live inspection remains available.</p>}
    {idle && selected && timeline.loading && <p role="status">Reading the finalized MCAP index and control samples…</p>}
    {idle && timeline.data && <RecordedPlayer key={selected} timeline={timeline.data} review={recording?.review} />}
  </div>;
}

export default function InspectionWorkspace({ initialTab = "live" }: { initialTab?: "live" | "recordings" }) {
  const navigate = useNavigate();
  const { status } = useSafety();
  const [referenceArm, setReferenceArm] = useState<"follower" | "leader">("follower");
  return <Tabs label="Inspection workspace" defaultSelected={initialTab} tabs={[
    { id: "live", label: "Live data", panel: <LiveInspection /> },
    { id: "recordings", label: "Recorded episodes", panel: <RecordedInspection /> },
    { id: "cameras", label: "Camera settings", panel: <CameraTuning /> },
    { id: "reference", label: "Home & reference checks", panel: <div className="space-y-4"><p className="text-sm text-slate-300">Check the installed encoder reference with the arm supported and torque off. The check reads every joint and compares installed offsets and ranges; it never searches for mechanical end stops or rewrites calibration. Confirm the physical pose and direction before powered motion.</p><label>Arm <select aria-label="Reference arm" value={referenceArm} onChange={event => setReferenceArm(event.target.value as "follower" | "leader")} className="rounded bg-slate-800 p-2"><option value="follower">Follower</option><option value="leader">Leader</option></select></label><CalibrationPoseView key={referenceArm} arm={referenceArm} initiallyOpen inProgress={status?.mode !== "idle"} controllerActive={status?.mode !== "idle"} onJoint={joint => navigate(`/calibration?arm=${referenceArm}&joint=${joint}`)} /><p className="text-sm text-slate-300">Collapse in the operator controls returns to the saved rest/home pose with the existing speed limit and stop supervision, then verifies torque off. A home pose is a reference inside the operating range, not proof that a joint can safely search its limits. Failed joint checks require a supported manual sweep and reviewed calibration before reuse.</p></div> },
  ]} />;
}

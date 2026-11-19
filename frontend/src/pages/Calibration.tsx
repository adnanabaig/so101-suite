import { useEffect, useRef, useState } from "react";
import Card from "../components/Card";
import Button from "../components/Button";
import CalibrationHistory from "../components/CalibrationHistory";
import CalibrationLive from "../components/CalibrationLive";
import CalibrationPoseView from "../components/CalibrationPoseView";
import { StatusPill } from "../components/ui/StatusPill";
import {
  startCalibration,
  getCalibrationStatus,
  type CalibrateRequest,
} from "../api";
import { usePolling } from "../hooks/usePolling";

type Arm = "follower" | "leader";

const JOINTS = [
  ["gripper", 6, "Open / close fingers"],
  ["wrist_roll", 5, "Rotate gripper"],
  ["wrist_flex", 4, "Bend wrist"],
  ["elbow_flex", 3, "Bend elbow"],
  ["shoulder_lift", 2, "Raise / lower arm"],
  ["shoulder_pan", 1, "Rotate base"],
] as const;

const PREFLIGHT = [
  "The arm is on a stable surface, its workspace is clear, and the physical power stop is reachable.",
  "The selected device ID, bus, voltage, motor IDs, and mechanical assembly have been verified.",
  "The arm is at the middle of its usable range; calibration will disable torque for backdriving.",
  "I will sweep every joint through its full safe range for the entire timed capture.",
];

export default function Calibration() {
  const [arm, setArm] = useState<Arm>(() => new URLSearchParams(window.location.search).get("arm") === "leader" ? "leader" : "follower");
  const [joint, setJoint] = useState<string>(() => {
    const requested = new URLSearchParams(window.location.search).get("joint");
    return JOINTS.find(([name]) => name === requested)?.[0] ?? "gripper";
  });
  const [scope, setScope] = useState<"joint" | "arm">("joint");
  const [sampleSeconds, setSampleSeconds] = useState(12);
  const [starting, setStarting] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [confirmed, setConfirmed] = useState<boolean[]>(() => PREFLIGHT.map(() => false));
  const [err, setErr] = useState<string | null>(null);
  const generationRef = useRef(0);

  useEffect(() => () => {
    generationRef.current += 1;
  }, []);

  // Switching arms invalidates every preflight confirmation and any pending
  // countdown. Done in the change handler rather than an effect so the reset
  // lands in the same render as the new arm.
  const changeArm = (next: Arm) => {
    if (next === arm) return;
    setArm(next);
    setConfirmed(PREFLIGHT.map(() => false));
    setCountdown(null);
    generationRef.current += 1;
  };

  const { data: status, error: statusError, refresh } = usePolling(
    () => getCalibrationStatus(arm),
    250,
    { key: arm },
  );

  const resetPreflight = () => {
    setConfirmed(PREFLIGHT.map(() => false));
    setCountdown(null);
    setErr(null);
    generationRef.current += 1;
  };
  const preflight = scope === "joint" ? [
    "The arm is clamped and physically supported; all motors will be torque-off. The workspace is clear and the physical power stop is reachable.",
    PREFLIGHT[1],
    `The selected ${joint} is at the middle of its safe travel; I will hold it still for midpoint capture.`,
    joint === "wrist_roll"
      ? "I will rotate only wrist roll gently within safe cable travel. I will not force a stop or wind cables."
      : `I will move only ${joint} through both safe end positions without forcing stops during the timed capture.`,
  ] : PREFLIGHT;

  const start = async () => {
    const generation = ++generationRef.current;
    setStarting(true);
    setErr(null);
    try {
      for (let remaining = 3; remaining > 0; remaining -= 1) {
        if (generationRef.current !== generation) return;
        setCountdown(remaining);
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
      }
      if (generationRef.current !== generation) return;
      setCountdown(null);
      const req: CalibrateRequest = {
        arm, sample_seconds: sampleSeconds, ...(scope === "joint" ? { joint } : {}),
      };
      await startCalibration(req);
      refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      if (generationRef.current === generation) {
        setStarting(false);
        setCountdown(null);
      }
    }
  };

  const fraction = status?.fraction ?? 0;
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  const sweeping = Boolean(status?.in_progress && (
    status.live ? status.live.phase === "sampling"
      : status.message.includes("now sweep") || status.message === "sweep every joint through its range"
  ));
  const finalizing = Boolean(status?.in_progress && status.live &&
    ["validating", "restoring", "saving", "complete", "failed"].includes(status.live.phase));

  const activeJoint = status?.live?.joint;
  const baselines = status?.joint_baselines ?? {};

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Calibration</h1>

      <CalibrationPoseView key={arm} arm={arm} inProgress={Boolean(status?.in_progress)}
        attempt={status?.live?.attempt_id} onJoint={(name) => {
          setJoint(name); setScope("joint"); resetPreflight();
          document.getElementById("calibration-preflight")?.scrollIntoView({ behavior: "smooth" });
        }} />

      <Card title="Calibration workflow" subtitle="Manual joint baselines → manual arm calibration → automatic joints (planned) → automatic arm (planned)">
        <p className="text-sm text-slate-300">
          Start at the gripper and work toward the base. A joint baseline saves measured encoder ranges and homing evidence,
          then restores prior calibration registers. It does not qualify the arm for motion or replace full arm calibration.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <label className="text-xs text-muted">Calibration scope
            <select aria-label="Calibration scope" value={scope} disabled={starting || status?.in_progress}
              onChange={(event) => { setScope(event.target.value as "joint" | "arm"); resetPreflight(); }}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 p-2 text-sm text-slate-100">
              <option value="joint">Manual joint baseline</option>
              <option value="arm">Manual arm calibration</option>
            </select>
          </label>
          {scope === "joint" && <label className="text-xs text-muted">Joint · expected motor ID
            <select aria-label="Calibration joint" value={joint} disabled={starting || status?.in_progress}
              onChange={(event) => { setJoint(event.target.value); resetPreflight(); }}
              className="mt-1 block w-full rounded-lg border border-slate-700 bg-slate-800 p-2 text-sm text-slate-100">
              {JOINTS.map(([name, id, movement]) => <option key={name} value={name}>{name} · ID {id} · {movement}</option>)}
            </select>
          </label>}
        </div>
      </Card>

      <div id="calibration-preflight">
      <Card title="Calibration preflight" subtitle={scope === "joint" ? `Temporary homing of ${joint}; manual sweep; save baseline; restore prior calibration` : "Writes homing/range data to the selected motors"}>
        <div className="grid gap-5 lg:grid-cols-[18rem_1fr]">
          <div className="flex flex-wrap content-start items-end gap-4">
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Arm
            <select
              value={arm}
              disabled={starting || status?.in_progress}
              onChange={(e) => changeArm(e.target.value as Arm)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100"
            >
              <option value="follower">follower</option>
              <option value="leader">leader</option>
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Sample seconds
            <input
              type="number"
              min={8}
              max={60}
              disabled={starting || status?.in_progress}
              value={sampleSeconds}
              onChange={(e) => setSampleSeconds(Number(e.target.value))}
              className="w-28 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100"
            />
          </label>

            <Button
              onClick={() => void start()}
              disabled={
                starting ||
                status?.in_progress ||
                !confirmed.every(Boolean) ||
                sampleSeconds < 8 ||
                sampleSeconds > 60
              }
            >
              {status?.in_progress
                ? "In progress…"
                : countdown
                  ? `Starting in ${countdown}…`
                  : starting
                    ? "Starting…"
                    : scope === "joint" ? `Capture ${joint} baseline` : "Begin calibrated sweep"}
            </Button>
          </div>

          <div className="space-y-2">
            {preflight.map((item, index) => (
              <label key={item} className="flex items-start gap-3 rounded-lg border border-white/5 bg-slate-950/35 p-3">
                <input
                  type="checkbox"
                  checked={confirmed[index]}
                  disabled={starting || status?.in_progress}
                  onChange={(event) =>
                    setConfirmed((current) =>
                      current.map((value, position) =>
                        position === index ? event.target.checked : value,
                      ),
                    )
                  }
                  className="mt-0.5 h-4 w-4 accent-cyan-400"
                />
                <span className="text-xs leading-relaxed text-slate-400">{item}</span>
              </label>
            ))}
          </div>
        </div>

        {err && (
          <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
            {err}
          </div>
        )}
      </Card>
      </div>

      {status?.in_progress && (
        <div className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-4 text-center">
          <div className="text-sm font-black uppercase tracking-wider text-amber-100">
            {sweeping ? activeJoint ? `Torque is off — sweep only ${activeJoint} now` : "Torque is off — sweep every joint now" : finalizing ? "Capture finished — checking and saving calibration" : "Preparing calibration — hold the midpoint pose still"}
          </div>
          <p className="mt-2 text-xs text-amber-100/70">
            {sweeping
              ? activeJoint ? activeJoint === "wrist_roll"
                ? "Rotate wrist roll gently within safe cable travel. Do not force stops or wind cables. Keep the rest of the arm supported."
                : `Move only ${activeJoint} through both safe end positions; do not force stops. Keep the rest of the arm supported.`
              : "Move shoulder pan/lift, elbow, wrist flex, and gripper through their full safe ranges. Wrist roll uses its official full encoder range. Keep clear of pinch points."
              : finalizing ? "Stop sweeping and wait for the final result."
              : "Wait for midpoint capture and homing read-back verification before moving any joint."}
          </p>
        </div>
      )}

      <Card
        title={`Status — ${arm}`}
        actions={
          <>
            <StatusPill
              label="state"
              value={status?.in_progress ? "running" : "idle"}
              tone={status?.in_progress ? "info" : "neutral"}
            />
            <StatusPill
              label="saved"
              value={status?.has_saved ? "yes" : "no"}
              tone={status?.has_saved ? "ok" : "neutral"}
            />
          </>
        }
      >
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">{status?.message ?? "—"}</span>
            <span className="font-mono text-slate-300">{pct}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-linear-to-r/srgb from-cyan-500 to-emerald-400 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
          {status?.has_saved && Object.keys(status.ranges).length > 0 && (
            <div className="pt-3">
              <div className="mb-2 text-xs text-muted">
                Saved device {status.device_id || "legacy/unknown"} · verify every measured range before motion
              </div>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(status.ranges).map(([joint, range]) => (
                  <div key={joint} className="rounded-lg border border-white/5 bg-slate-950/40 p-2.5">
                    <div className="font-mono text-[10px] text-muted">{joint}</div>
                    <div className="mt-1 font-mono text-xs text-slate-200">
                      {range[0].toFixed(1)} → {range[1].toFixed(1)} {joint === "gripper" ? "%" : "deg"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>
      <CalibrationLive live={status?.live} error={statusError?.message} />
      <Card title="Saved joint baselines" subtitle="Independent measurements; full arm calibration is a separate step">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {JOINTS.map(([name, id]) => {
            const baseline = baselines[name];
            const diagnostics = baseline?.diagnostics;
            const min = diagnostics?.range_mins?.[name];
            const max = diagnostics?.range_maxes?.[name];
            return <div key={name} className="rounded-lg border border-white/5 bg-slate-950/40 p-3">
              <div className="font-mono text-xs">{name} · ID {id}</div>
              <div className="mt-2 text-xs text-muted">{baseline?.error ?? (baseline
                ? baseline.backend === "mock" ? "Mock defaults · no encoder measurement"
                  : `Captured · ${min ?? "—"} → ${max ?? "—"} ticks · offset ${diagnostics?.homing_offsets_requested?.[name] ?? "—"}`
                : "No baseline saved")}</div>
              {baseline?.created_at && <time className="mt-1 block text-xs text-muted" dateTime={new Date(baseline.created_at * 1000).toISOString()}>
                {new Date(baseline.created_at * 1000).toLocaleString()}
              </time>}
            </div>;
          })}
        </div>
      </Card>
      <CalibrationHistory />
    </div>
  );
}

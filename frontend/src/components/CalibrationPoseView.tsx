import { useEffect, useState } from "react";
import { getCachedCalibrationPose, readCalibrationPose } from "../api";
import { usePolling } from "../hooks/usePolling";
import Button from "./Button";
import Card from "./Card";
import RobotTwin from "./RobotTwin";
import { TWIN_JOINTS, TWIN_JOINT_LABELS } from "./twin/types";

export interface ControlPose {
  observation: Record<string, number>;
  capturedAt: number;
  live: boolean;
  active: boolean;
}

export default function CalibrationPoseView({ arm, inProgress, attempt, onJoint, initiallyOpen = false, controlPose, controllerActive = false }: {
  arm: "follower" | "leader";
  inProgress: boolean;
  attempt?: string | null;
  onJoint: (joint: string) => void;
  initiallyOpen?: boolean;
  controlPose?: ControlPose;
  controllerActive?: boolean;
}) {
  const [open, setOpen] = useState(initiallyOpen);
  const [supported, setSupported] = useState(false);
  const [watching, setWatching] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!watching) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [watching]);
  const [matches, setMatches] = useState<{ signature: string; joints: string[] }>({
    signature: "", joints: [],
  });
  const { data: sampled, error, lastSuccessAt } = usePolling(
    () => readCalibrationPose(arm), 100,
    { key: arm, enabled: open && supported && watching && !inProgress && !controllerActive },
  );
  const { data: cached } = usePolling(() => getCachedCalibrationPose(arm), 250,
    { key: arm, enabled: open && !watching });
  const data = sampled ?? cached;
  const usingControl = arm === "follower" && controlPose && (!data || controlPose.capturedAt >= data.captured_at);
  const observation = usingControl ? controlPose.observation : data?.observation;
  const signature = `${arm}:${data?.calibration_sha256 ?? ""}:${attempt ?? ""}`;
  const fresh = usingControl ? controlPose.live : Boolean(watching && !inProgress && !error && lastSuccessAt && now - lastSuccessAt < 2000);
  const checked = matches.signature === signature ? matches.joints : [];
  const real = Boolean(usingControl || data?.backend === "real");
  const complete = Boolean((usingControl || data?.complete) && TWIN_JOINTS.every((joint) => Number.isFinite(observation?.[`${joint}.pos`])));
  const canConfirm = real && fresh && !usingControl;
  const count = canConfirm ? checked.filter((joint) => !data?.joints[joint]?.issues.length).length : 0;

  return <Card title="Real arm → 3D mapping" subtitle="Read the installed encoders and compare the perceived pose with the physical arm">
    <Button onClick={() => setOpen((current) => !current)}>
      {open ? "Hide perceived pose" : "Recalibrate"}
    </Button>
    {open && <div className="mt-4 space-y-4">
      <p className="text-sm text-slate-300">
        The model uses the SO-101 STL parts. Motors 1–6 are base, shoulder, elbow, wrist bend, wrist roll and gripper.
        Compare the silhouette and each movement with your supported arm.
      </p>
      {controllerActive && <p role="status" className="text-sm text-cyan-200">
        {arm === "follower" ? "Following the follower control stream. " : "The control stream contains follower readings only. "}
        Manual encoder reads are paused while the controller owns the arms.
      </p>}
      <label className="flex items-start gap-3 text-sm text-slate-300">
        <input type="checkbox" checked={supported} disabled={watching}
          onChange={(event) => setSupported(event.target.checked)} />
        Both arms are supported, the selected arm is already torque-off, the workspace is clear and the physical cutoff is reachable.
      </label>
      <div className="flex flex-wrap items-center gap-3">
        <Button disabled={!supported || inProgress || controllerActive} onClick={() => setWatching((current) => !current)}>
          {watching ? "Stop reading pose" : "Read current pose"}
        </Button>
        <span role="status" className="text-xs text-slate-400">
          {usingControl ? controlPose.live ? "Live follower encoders · shared control stream" : "Last control pose · live samples unavailable"
            : inProgress ? "Pose reads paused during calibration" : error ? "Pose unavailable — last reading only"
            : !data ? "No physical pose read yet" : !real ? "Preview backend — not the physical arm"
              : fresh ? "Reading encoders · torque unchanged" : "Last read · arm may have moved"}
        </span>
      </div>
      {error && !usingControl && <p role="alert" className="text-sm text-red-300">{error.message}</p>}
      <div className="grid gap-3 lg:grid-cols-2">
        <RobotTwin observation={complete ? observation : undefined} fitArm
          mode="perceived pose" sourceLabel={`${arm} · ${usingControl ? "control encoders" : real ? "motor encoders" : "preview"}`}
          idle={!fresh} stale={usingControl ? !controlPose.live : Boolean(data && error)} showTarget={false} className="h-[24rem] rounded-xl" />
        <div className="overflow-hidden rounded-xl border border-slate-700 bg-slate-950">
          <p className="px-3 py-2 text-xs text-slate-300">OBSBOT · physical arm · compare pose and direction</p>
          <img src="/api/mcap/cameras/obsbot/stream.mjpg" alt="Live OBSBOT view of the physical arms"
            className="h-[21rem] w-full object-contain" />
        </div>
      </div>
      {(data || usingControl) && <p className="text-xs text-slate-400">
        Read at {new Date((usingControl ? controlPose.capturedAt : data!.captured_at) * 1000).toLocaleTimeString()} · {count}/6 visual checks confirmed
        {!complete && " · Incomplete or inconsistent reference: the grey arm is a placeholder"}
      </p>}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {TWIN_JOINTS.map((joint, index) => {
          const reading = data?.joints[joint];
          const issues = reading?.issues ?? [];
          const matchesPose = checked.includes(joint) && canConfirm && !issues.length;
          const value = observation?.[`${joint}.pos`];
          return <div key={joint} className={`rounded-xl border p-3 ${matchesPose ? "border-emerald-400/50 bg-emerald-400/10" : issues.length && real ? "border-red-400/40" : "border-slate-700"}`}>
            <p className="font-semibold">{index + 1} · {TWIN_JOINT_LABELS[joint]}</p>
            <p className="mt-1 text-xs text-slate-400">
              {typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)} ${joint === "gripper" ? "%" : "°"}` : "No valid angle"}
              {!usingControl && reading?.ticks != null && ` · ${reading.ticks} ticks`}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              {usingControl ? "Controller sample · use torque-off reference check to confirm alignment" : matchesPose ? "Visual match confirmed" : issues.length ? issues.join("; ") : reading ? "Installed range read-back checked · compare pose" : "Waiting for encoder read"}
            </p>
            <label className="mt-3 flex items-start gap-2 text-xs">
              <input type="checkbox" aria-label={`Motor ${index + 1} pose and direction match`}
                disabled={!canConfirm || !complete || !reading || issues.length > 0}
                checked={Boolean(matchesPose)} onChange={(event) => setMatches({ signature,
                  joints: event.target.checked ? [...checked, joint] : checked.filter((name) => name !== joint),
                })} />
              Its physical pose and movement direction match the 3D model
            </label>
            <button type="button" disabled={inProgress} onClick={() => onJoint(joint)}
              className="mt-3 text-xs font-semibold text-cyan-300 underline disabled:opacity-40">
              Check motor {index + 1} endpoints
            </button>
          </div>;
        })}
      </div>
      <p className="text-xs text-slate-400">
        Green records your visual check for this reference and session. A sweep checks encoder travel;
        it cannot automatically establish physical alignment. Endpoint capture uses the separate confirmations below,
        and a saved joint baseline does not install new arm calibration. This view sends no motion or torque commands.
      </p>
    </div>}
  </Card>;
}

interface JointBarsProps {
  observation?: Record<string, number>;
  action?: Record<string, number>;
  /** Expected absolute range of joint values for scaling the bars. Default 100. */
  range?: number;
}

const JOINTS = [
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
  "gripper",
] as const;

function pickValue(
  dict: Record<string, number> | undefined,
  joint: string,
): number | undefined {
  if (!dict) return undefined;
  // Backend keys are "<joint>.pos".
  const key = `${joint}.pos`;
  if (key in dict) return dict[key];
  if (joint in dict) return dict[joint];
  return undefined;
}

function pct(value: number | undefined, range: number): number {
  if (value === undefined || !isFinite(value)) return 50;
  // Map [-range, range] -> [0, 100], clamped.
  const clamped = Math.max(-range, Math.min(range, value));
  return ((clamped + range) / (2 * range)) * 100;
}

function fmt(value: number | undefined): string {
  if (value === undefined || !isFinite(value)) return "—";
  return value.toFixed(1);
}

export default function JointBars({
  observation,
  action,
  range = 100,
}: JointBarsProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-3 rounded-xs bg-cyan-400" /> observed
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-3 rounded-xs bg-emerald-400" /> commanded
        </span>
      </div>

      {JOINTS.map((joint) => {
        const obs = pickValue(observation, joint);
        const act = pickValue(action, joint);
        return (
          <div key={joint} className="grid grid-cols-[8rem_1fr_5rem] items-center gap-3">
            <div className="font-mono text-xs text-slate-300">{joint}</div>
            <div className="relative h-5 overflow-hidden rounded-sm bg-slate-800">
              {/* center tick */}
              <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600/70" />
              {/* observed bar */}
              <div
                className="absolute inset-y-0.5 left-0.5 rounded-xs bg-cyan-400/70"
                style={{ width: `calc(${pct(obs, range)}% - 4px)` }}
              />
              {/* commanded marker */}
              <div
                className="absolute inset-y-0 w-0.5 bg-emerald-300 shadow-sm shadow-emerald-400/50"
                style={{ left: `${pct(act, range)}%` }}
              />
            </div>
            <div className="text-right font-mono text-xs">
              <span className="text-cyan-300">{fmt(obs)}</span>
              <span className="mx-1 text-muted">/</span>
              <span className="text-emerald-300">{fmt(act)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

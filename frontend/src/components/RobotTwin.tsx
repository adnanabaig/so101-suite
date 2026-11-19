// The digital twin's light shell. It sits in page chunks, so it must not
// import three.js: the WebGL scene (./twin/TwinScene) loads lazily (F15).
//
// The shell owns everything the operator reads as text: whether the pose is
// live, last known or absent (F14), the command target, and the fallbacks
// when the 3D view cannot load.
import {
  Component,
  Fragment,
  lazy,
  Suspense,
  useId,
  useRef,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";
import { RefreshCw } from "lucide-react";
import { isChunkLoadError, reloadPage } from "./ErrorBoundary";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { useDocumentVisible, useOnScreen } from "../hooks/useVisibility";
import { twinPresence, type TwinPresence } from "./twin/kinematics";
import {
  formatJointValue,
  hasJointData,
  jointValue,
  TWIN_JOINT_LABELS,
  TWIN_JOINTS,
  type RobotTwinScenario,
  type RobotTwinTrajectoryPoint,
} from "./twin/types";

export type {
  RobotTwinScenario,
  RobotTwinScenarioObject,
  RobotTwinTrajectoryPoint,
} from "./twin/types";

const TwinScene = lazy(() => import("./twin/TwinScene"));

interface RobotTwinProps {
  fitArm?: boolean;
  observation?: Readonly<Record<string, number>>;
  action?: Readonly<Record<string, number>>;
  mode?: string;
  className?: string;
  showTarget?: boolean;
  scenario?: RobotTwinScenario;
  trajectory?: readonly RobotTwinTrajectoryPoint[];
  sourceLabel?: string;
  /** The link is down or the sample is overdue: draw the pose as last known. */
  stale?: boolean;
  /**
   * Nothing samples the joints (C6 idle): the pose is the last one read. Shown
   * neutrally, never as a warning (F8), and never as live.
   */
  idle?: boolean;
}

class TwinErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error("Digital twin failed", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const chunk = isChunkLoadError(error);
    return (
      <div className="flex h-full min-h-72 items-center justify-center p-8 text-center">
        <div>
          <div className="text-sm font-semibold text-slate-200">
            {chunk ? "The 3D view could not be downloaded" : "3D rendering unavailable"}
          </div>
          <p className="mt-2 max-w-sm text-xs leading-relaxed text-slate-400">
            {chunk
              ? "The operator UI was probably updated on the server. Reload when the robot is in a safe state."
              : "This browser could not create a WebGL scene."}{" "}
            Joint telemetry and the safety controls stay available.
          </p>
          {chunk && (
            <button
              type="button"
              onClick={reloadPage}
              className="mt-4 inline-flex h-9 items-center gap-2 rounded-lg border border-cyan-400/40 bg-cyan-500 px-3 text-xs font-bold text-slate-950 hover:bg-cyan-400"
            >
              <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
              Reload the UI
            </button>
          )}
        </div>
      </div>
    );
  }
}

function Placeholder({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full min-h-72 items-center justify-center p-8 text-center text-xs text-slate-400">
      {children}
    </div>
  );
}

function chipDot(presence: TwinPresence, mode: string): string {
  if (presence === "none") return "border border-slate-400 bg-transparent";
  if (presence === "stale") return "bg-amber-400";
  if (presence === "idle" || mode === "idle" || mode === "paused") return "bg-slate-300";
  if (mode === "playing") return "bg-emerald-400";
  return "bg-cyan-400";
}

const PRESENCE_CHIP: Record<TwinPresence, (mode: string) => string> = {
  live: (mode) => mode,
  idle: () => "last read",
  stale: () => "last known",
  none: () => "no sample",
};

const SCENE_LABEL: Record<TwinPresence, (mode: string) => string> = {
  live: (mode) => mode === "recorded playback" ? "3D view of the arm: recorded encoder joint state. No physical commands." : `3D view of the arm: live joint state, ${mode}.`,
  idle: () => "3D view of the arm: the last pose read. Nothing is sampling the joints.",
  stale: () => "3D view of the arm: last known joint state. Telemetry is stale.",
  none: () => "3D view of the arm: no live joint sample. The arm shown is a placeholder.",
};

export default function RobotTwin({
  observation,
  action,
  mode = "idle",
  className = "",
  showTarget = true,
  scenario,
  trajectory,
  sourceLabel = "Kinematic twin",
  stale = false,
  idle = false,
  fitArm = false,
}: RobotTwinProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const targetTitleId = useId();
  const onScreen = useOnScreen(rootRef);
  const visible = useDocumentVisible();
  const reducedMotion = usePrefersReducedMotion();
  const showing = onScreen && visible;

  // The WebGL chunk loads the first time the twin can be seen and then stays
  // mounted, so scrolling away and back does not rebuild the scene. While it
  // cannot be seen it stops rendering altogether.
  const [activated, setActivated] = useState(showing);
  if (showing && !activated) setActivated(true);

  const presence = twinPresence(observation, { stale, idle });
  const showCommand = showTarget && hasJointData(action);

  return (
    <div ref={rootRef} className={`relative isolate overflow-hidden bg-[#070b14] ${className}`}>
      <TwinErrorBoundary>
        {activated ? (
          <Suspense fallback={<Placeholder>Loading the 3D view…</Placeholder>}>
            <div role="img" aria-label={SCENE_LABEL[presence](mode)} className="h-full w-full">
              <TwinScene
                observation={observation}
                mode={mode}
                presence={presence}
                scenario={scenario}
                trajectory={trajectory}
                frameloop={showing ? "demand" : "never"}
                reducedMotion={reducedMotion}
                fitArm={fitArm}
              />
            </div>
          </Suspense>
        ) : (
          <Placeholder>The 3D view starts when it is visible.</Placeholder>
        )}
      </TwinErrorBoundary>

      <div className="pointer-events-none absolute left-4 top-4 max-w-[calc(100%-2rem)] space-y-2">
        <div className="inline-flex items-center gap-2 rounded-lg border border-white/5 bg-slate-950/80 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-300 backdrop-blur-sm">
          <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${chipDot(presence, mode)}`} />
          {sourceLabel} · {PRESENCE_CHIP[presence](mode)}
        </div>
        {presence === "none" && (
          <div className="max-w-xs rounded-lg border border-slate-500/30 bg-slate-950/85 px-3 py-2 text-xs backdrop-blur-sm">
            <div className="font-semibold text-slate-100">No live joint sample</div>
            <p className="mt-0.5 leading-relaxed text-slate-400">
              The grey arm is a placeholder. It does not show where the real arm is.
            </p>
          </div>
        )}
        {presence === "stale" && (
          <div className="max-w-xs rounded-lg border border-amber-400/30 bg-slate-950/85 px-3 py-2 text-xs backdrop-blur-sm">
            <div className="font-semibold text-amber-200">Showing the last known pose</div>
            <p className="mt-0.5 leading-relaxed text-slate-400">
              Telemetry is stale, so the real arm may have moved.
            </p>
          </div>
        )}
        {presence === "idle" && (
          <div className="max-w-xs rounded-lg border border-slate-500/30 bg-slate-950/85 px-3 py-2 text-xs backdrop-blur-sm">
            <div className="font-semibold text-slate-100">Joint sampling paused while idle</div>
            <p className="mt-0.5 leading-relaxed text-slate-400">
              This is the last pose read. If the arm moved by hand, use Read current pose.
            </p>
          </div>
        )}
      </div>

      {showCommand && (
        <section
          aria-labelledby={targetTitleId}
          className="pointer-events-none absolute right-4 top-4 rounded-lg border border-emerald-300/20 bg-slate-950/85 px-3 py-2 backdrop-blur-sm"
        >
          <p id={targetTitleId} className="text-[10px] font-bold uppercase tracking-wider text-emerald-200">
            {stale || idle ? "Last command target" : "Command target"}
          </p>
          <dl className="mt-1 grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 font-mono text-[10px]">
            {TWIN_JOINTS.map((joint) => (
              <Fragment key={joint}>
                <dt className="text-slate-400">{TWIN_JOINT_LABELS[joint]}</dt>
                <dd className="text-right text-slate-100">
                  {formatJointValue(joint, jointValue(action, joint))}
                </dd>
              </Fragment>
            ))}
          </dl>
        </section>
      )}

      <div className="pointer-events-none absolute bottom-4 left-4 text-[10px] text-slate-400">
        Drag to orbit · wheel to zoom
      </div>
    </div>
  );
}

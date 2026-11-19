import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  Boxes,
  CircleDotDashed,
  Gauge,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  ScanLine,
  StepForward,
  Target,
  Timer,
} from "lucide-react";
import {
  getSimulationScenarios,
  getSimulationState,
  pauseSimulation,
  playSimulation,
  previewSimulationTrajectory,
  resetSimulation,
  setSimulationTargets,
  stepSimulation,
  type SimulationResetResponse,
  type SimulationScenario,
  type SimulationState,
  type SimulationTrajectoryPreview,
} from "../api";
import Button from "../components/Button";
import Card from "../components/Card";
import JointBars from "../components/JointBars";
import RobotTwin from "../components/RobotTwin";
import CalibrationPoseView from "../components/CalibrationPoseView";
import { useSafety } from "../components/safety/SafetyProvider";
import PhysicsReadout from "../components/simulation/PhysicsReadout";
import SimCameraTile from "../components/simulation/SimCameraTile";
import { useSimulationCapabilities } from "../components/simulation/useSimulationCapabilities";
import {
  describeEngine,
  formatTolerance,
  previewSemantics,
  reachSummary,
  type EngineKind,
} from "../components/simulation/engine";
import {
  EmptyState,
  MetricTile,
  Notice,
  PageHeader,
  SectionLabel,
} from "../components/Workstation";
import { usePolling } from "../hooks/usePolling";
import { useTelemetry } from "../hooks/useTelemetry";

type TwinSource = "simulation" | "robot";

const FALLBACK_SCENARIOS: SimulationScenario[] = [
  {
    id: "tabletop_pick",
    name: "Tabletop pick",
    description: "Single-object pick and place workspace.",
    available: false,
    backend: "kinematic",
    physics: false,
  },
  {
    id: "bin_sort",
    name: "Two-bin sort",
    description: "Two manipulands and fixed target zones.",
    available: false,
    backend: "kinematic",
    physics: false,
  },
  {
    id: "calibration_grid",
    name: "Calibration grid",
    description: "Uncluttered joint and tracking checks.",
    available: false,
    backend: "kinematic",
    physics: false,
  },
];

const JOINT_LABELS: Record<string, string> = {
  shoulder_pan: "Shoulder pan",
  shoulder_lift: "Shoulder lift",
  elbow_flex: "Elbow flex",
  wrist_flex: "Wrist flex",
  wrist_roll: "Wrist roll",
  gripper: "Gripper",
};

const fieldClass =
  "rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-2 text-sm text-slate-100 outline-hidden transition focus:border-cyan-400/50 focus-visible:ring-2 focus-visible:ring-cyan-400/40";

function scenarioList(
  value: Awaited<ReturnType<typeof getSimulationScenarios>> | null,
): SimulationScenario[] {
  if (!value) return [];
  return Array.isArray(value) ? value : value.scenarios;
}

const RUNTIME_TONE: Record<EngineKind, "cyan" | "green" | "amber" | "red"> = {
  physics: "green",
  kinematic: "cyan",
  none: "red",
  unknown: "amber",
};

/** Live rigid-body facts from physics_state, appended to an object's line. */
function objectMotion(body: { speed_m_s?: number; grasped?: boolean } | undefined): string {
  if (!body) return "";
  const speed =
    typeof body.speed_m_s === "number" && Number.isFinite(body.speed_m_s)
      ? ` · ${(body.speed_m_s * 100).toFixed(1)} cm/s`
      : "";
  return `${speed}${body.grasped ? " · grasped" : ""}`;
}

function maxTrajectoryError(preview: SimulationTrajectoryPreview | null): number {
  if (!preview) return 0;
  return preview.points.reduce(
    (maximum, point) =>
      Math.max(maximum, ...Object.values(point.error).map((value) => Math.abs(value))),
    0,
  );
}

function trajectoryTravel(preview: SimulationTrajectoryPreview | null): number {
  if (!preview || preview.points.length < 2) return 0;
  let distance = 0;
  for (let index = 1; index < preview.points.length; index += 1) {
    const previous = preview.points[index - 1].end_effector.position_m;
    const current = preview.points[index].end_effector.position_m;
    distance += Math.hypot(
      (current[0] ?? 0) - (previous[0] ?? 0),
      (current[1] ?? 0) - (previous[1] ?? 0),
      (current[2] ?? 0) - (previous[2] ?? 0),
    );
  }
  return distance;
}

export default function Simulation({ embedded = false }: { embedded?: boolean }) {
  const [isolated, setIsolated] = useState(false);
  const [arm, setArm] = useState<"follower" | "leader">("follower");
  const { telemetry, hasLiveObservation } = useTelemetry();
  const { status, statusStale } = useSafety();
  const physicalBackend = status?.backend === "real" && !status.mock && !statusStale;
  const [lastControlPose, setLastControlPose] = useState<{ observation: Record<string, number>; capturedAt: number } | null>(null);
  const complete = telemetry && ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    .every((joint) => Number.isFinite(telemetry.observation[`${joint}.pos`]));
  if (physicalBackend && hasLiveObservation && complete && telemetry.t !== lastControlPose?.capturedAt) {
    setLastControlPose({ observation: telemetry.observation, capturedAt: telemetry.t });
  }
  const controlPose = arm === "follower" && physicalBackend && lastControlPose ? {
    ...lastControlPose,
    live: Boolean(hasLiveObservation && complete),
    active: Boolean(telemetry && telemetry.mode !== "idle"),
  } : undefined;
  return <div className="space-y-6">
    {embedded ? <h2 className="text-2xl font-bold">Digital twin</h2> : <h1 className="text-2xl font-bold">Digital twin</h1>}
    <p className="text-sm text-slate-400">The physical view follows measured encoders. The isolated simulator is a separate workspace for proposed motion.</p>
    <div className="flex flex-wrap items-center gap-3">
      <Button variant={isolated ? "secondary" : "primary"} onClick={() => setIsolated(false)}>Physical arm</Button>
      <Button variant={isolated ? "primary" : "secondary"} onClick={() => setIsolated(true)}>Isolated simulator</Button>
      {!isolated && <label className="text-sm">Arm
        <select aria-label="Visualized arm" value={arm} onChange={(event) => setArm(event.target.value as "follower" | "leader")}
          className="ml-2 rounded-lg border border-slate-700 bg-slate-900 p-2">
          <option value="follower">follower</option><option value="leader">leader</option>
        </select>
      </label>}
    </div>
    {isolated ? <SimulationWorkspace embedded={embedded} /> : <CalibrationPoseView key={arm} arm={arm} inProgress={false} initiallyOpen
      controlPose={controlPose}
      controllerActive={Boolean(physicalBackend && telemetry && telemetry.mode !== "idle")}
      onJoint={(joint) => { window.location.assign(`/calibration?arm=${arm}&joint=${joint}`); }} />}
  </div>;
}

export function SimulationWorkspace({ embedded = false }: { embedded?: boolean }) {
  const {
    telemetry: robotTelemetry,
    state: robotTransport,
    stale: robotStale,
    sampleState: robotSampleState,
  } = useTelemetry();
  const {
    data: scenarioResponse,
    error: scenarioError,
    refresh: refreshScenarios,
  } = usePolling(getSimulationScenarios, 0);
  const {
    data: simulationState,
    error: stateError,
    stale: stateStale,
    refresh: refreshState,
  } = usePolling(getSimulationState, 250);
  const {
    data: simulationCapabilities,
    error: capabilityError,
    refresh: refreshCapabilities,
  } = useSimulationCapabilities();

  const backendScenarios = useMemo(
    () => scenarioList(scenarioResponse),
    [scenarioResponse],
  );
  const scenarios = backendScenarios.length
    ? backendScenarios
    : FALLBACK_SCENARIOS;
  // F28: usePolling keeps the last state through failures, so "we have a
  // state" is not "the backend answers". Controls follow the live poll.
  const backendAvailable = simulationState !== null && !stateStale;
  const engine = describeEngine(simulationCapabilities, simulationState);

  const [source, setSource] = useState<TwinSource>("simulation");
  const [selected, setSelected] = useState("");
  const [seed, setSeed] = useState(101);
  const [randomize, setRandomize] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [manualSteps, setManualSteps] = useState(1);
  // Operator edits only; while not dirty the draft mirrors the backend target.
  const [draftEdits, setDraftEdits] = useState<Record<string, number>>({});
  const [targetDirty, setTargetDirty] = useState(false);
  const [previewDuration, setPreviewDuration] = useState(1);
  const [previewSampleHz, setPreviewSampleHz] = useState(20);
  const [chartJoint, setChartJoint] = useState("shoulder_pan");
  // A preview is only meaningful for the scene generation it was computed
  // from, so it is stored with that generation and hidden once it changes.
  const [trajectoryResult, setTrajectoryResult] = useState<{
    generation: number | undefined;
    preview: SimulationTrajectoryPreview;
  } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const activeScenario =
    scenarios.find((scenario) => scenario.id === selected) ??
    scenarios.find((scenario) => scenario.id === simulationState?.scene.scenario_id) ??
    scenarios[0];

  // Adopt the backend's scene once, the first time it reports one (adjusting
  // state during render instead of in an effect; converges after one pass).
  if (!selected && simulationState?.scene.scenario_id) {
    setSelected(simulationState.scene.scenario_id);
    setSeed(simulationState.scene.seed);
    setRandomize(simulationState.scene.randomized);
  }

  const backendTarget = simulationState?.telemetry.target;
  const targetDraft = useMemo<Record<string, number>>(
    () => (targetDirty ? draftEdits : { ...(backendTarget ?? {}) }),
    [targetDirty, draftEdits, backendTarget],
  );
  const editTarget = (joint: string, value: number) => {
    const base = targetDraft;
    setDraftEdits((current) => ({ ...(targetDirty ? current : base), [joint]: value }));
    setTargetDirty(true);
  };

  const trajectory =
    trajectoryResult && trajectoryResult.generation === simulationState?.generation
      ? trajectoryResult.preview
      : null;

  const runStateAction = async (
    key: string,
    operation: () => Promise<SimulationState | SimulationResetResponse>,
    success: (result: SimulationState | SimulationResetResponse) => string,
  ) => {
    setBusy(key);
    setActionError(null);
    setMessage(null);
    try {
      const result = await operation();
      // The draft re-derives from the refreshed backend target.
      setTargetDirty(false);
      setMessage(success(result));
      refreshState();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(null);
    }
  };

  const handleReset = () => {
    if (!activeScenario) return;
    void runStateAction(
      "reset",
      () =>
        resetSimulation({
          scenario_id: activeScenario.id,
          seed,
          randomize,
        }),
      (result) =>
        "state" in result
          ? result.detail
          : `${activeScenario.name ?? activeScenario.label ?? activeScenario.id} reset.`,
    );
  };

  const handlePlay = () => {
    void runStateAction(
      "play",
      async () => {
        if (targetDirty) await setSimulationTargets(targetDraft);
        return playSimulation(playbackRate);
      },
      () => `Simulation playing at ${playbackRate.toFixed(2)}×.`,
    );
  };

  const handlePreview = async () => {
    if (!Object.keys(targetDraft).length) return;
    setBusy("preview");
    setActionError(null);
    setMessage(null);
    const generation = simulationState?.generation;
    try {
      const result = await previewSimulationTrajectory({
        target: targetDraft,
        duration_s: previewDuration,
        sample_hz: previewSampleHz,
      });
      setTrajectoryResult({ generation, preview: result });
      setMessage(
        `Preview generated from simulation step ${result.source_step_count}; ${
          result.mutates_state
            ? "the backend reports that it changed the runtime state."
            : "runtime state was not changed."
        }`,
      );
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(null);
    }
  };

  const twinObservation =
    source === "simulation"
      ? simulationState?.telemetry.measured
      : robotTelemetry?.observation;
  const twinAction =
    source === "simulation"
      ? simulationState?.telemetry.target
      : robotTelemetry?.action;
  const twinMode =
    source === "simulation"
      ? simulationState?.status ?? "offline"
      : robotTelemetry?.mode ?? "offline";
  const twinStale = source === "simulation" ? stateStale : robotStale;
  const jointCount =
    source === "simulation"
      ? simulationState?.joint_order.length ?? 0
      : Object.keys(robotTelemetry?.observation ?? {}).length;
  const validSeed =
    Number.isInteger(seed) && seed >= 0 && seed <= 2 ** 32 - 1;
  const validSteps =
    Number.isInteger(manualSteps) && manualSteps >= 1 && manualSteps <= 10000;

  const previewChart = useMemo(
    () =>
      (trajectory?.points ?? []).map((point) => ({
        time: point.time_s,
        measured: point.measured[chartJoint] ?? 0,
        target: point.target[chartJoint] ?? 0,
        error: Math.abs(point.error[chartJoint] ?? 0),
      })),
    [trajectory, chartJoint],
  );
  const previewPath = useMemo(
    () =>
      (trajectory?.points ?? []).map((point) => ({
        position_m: point.end_effector.position_m,
      })),
    [trajectory],
  );
  const maximumPreviewError = maxTrajectoryError(trajectory);
  const endEffectorTravel = trajectoryTravel(trajectory);

  return (
    <div className="space-y-6">
      {/* First, not last: v4's space-y spaces every child but the last, so an
          out-of-flow last child would leave a gap under the page. */}
      <div className="sr-only" aria-live="polite">
        {jointCount} joint channels. Simulation {simulationState?.status ?? "offline"}.
      </div>
      <PageHeader
        headingLevel={embedded ? 2 : 1}
        eyebrow="Digital twin / isolated runtime"
        title="SO-101 simulation workspace"
        description="Drive the hardware-isolated simulator, inspect seeded scenes, and preview trajectories without entering a hardware command path."
        icon={Boxes}
        actions={
          <>
            <Button
              variant="secondary"
              onClick={() => {
                refreshScenarios();
                refreshState();
                refreshCapabilities();
              }}
            >
              <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" /> Refresh
            </Button>
            <Button
              onClick={handleReset}
              disabled={!backendAvailable || !activeScenario || !validSeed || busy !== null}
            >
              <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
              {busy === "reset" ? "Resetting…" : "Reset scene"}
            </Button>
          </>
        }
      />

      <Notice tone={engine.tone} title={engine.title}>
        {engine.summary}
        {engine.why && (
          <span className="mt-1 block">
            Why: <span className="font-mono text-slate-300">{engine.why}</span>
          </span>
        )}
        {engine.renderProblem && (
          <span className="mt-1 block">
            Camera frames are unavailable:{" "}
            <span className="font-mono text-slate-300">{engine.renderProblem}</span>
          </span>
        )}
      </Notice>

      {(scenarioError || stateError || capabilityError) && (
        <Notice tone="danger" title="Simulation API is partially unavailable">
          {[scenarioError, stateError, capabilityError]
            .filter((error): error is Error => error !== null)
            .map((error) => error.message)
            .join(" · ")}
          {robotTelemetry
            ? " Real robot telemetry remains available as an alternate twin source."
            : ""}
        </Notice>
      )}
      {actionError && (
        <Notice tone="danger" title="Simulation action failed">
          {actionError}
        </Notice>
      )}
      {message && (
        <Notice tone="success" title="Simulation updated">
          {message}
        </Notice>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Twin source"
          value={source === "simulation" ? "Simulation" : "Real robot"}
          detail={
            source === "simulation"
              ? stateError
                ? "State polling interrupted"
                : "Stateful API · 250 ms refresh"
              : `${robotTransport} WebSocket${robotStale ? " · stale" : ""}`
          }
          tone={twinStale ? "amber" : source === "simulation" ? "cyan" : "green"}
          icon={CircleDotDashed}
        />
        <MetricTile
          label="Runtime"
          value={`${simulationState?.fixed_hz ?? "—"} Hz`}
          detail={engine.runtimeDetail}
          tone={RUNTIME_TONE[engine.kind]}
          icon={Gauge}
        />
        <MetricTile
          label="Simulation clock"
          value={`${(simulationState?.sim_time_s ?? 0).toFixed(2)} s`}
          detail={`${simulationState?.step_count ?? 0} steps · generation ${simulationState?.generation ?? "—"}`}
          tone={simulationState?.status === "playing" ? "green" : "neutral"}
          icon={Timer}
        />
        <MetricTile
          label="Hardware boundary"
          value={simulationState?.hardware_isolated ? "Isolated" : "Unknown"}
          detail={
            simulationState?.writes_motor_commands === false
              ? "Motor command writes: disabled"
              : "Awaiting backend truth"
          }
          tone={simulationState?.hardware_isolated ? "green" : "amber"}
          icon={ScanLine}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.55fr)_minmax(21rem,0.8fr)]">
        <Card
          title="SO-101 digital twin"
          subtitle={
            source === "simulation"
              ? "Simulated measured pose, target overlay, scene objects, and preview path"
              : "Real robot observation and action telemetry; no simulated scene claim"
          }
          actions={
            <div className="flex rounded-lg border border-slate-700 bg-slate-950/60 p-0.5">
              <button
                type="button"
                aria-pressed={source === "simulation"}
                onClick={() => setSource("simulation")}
                className={`rounded-md px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide transition ${
                  source === "simulation"
                    ? "bg-cyan-400/15 text-cyan-200"
                    : "text-muted hover:text-slate-300"
                }`}
              >
                Simulation
              </button>
              <button
                type="button"
                aria-pressed={source === "robot"}
                onClick={() => setSource("robot")}
                className={`rounded-md px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide transition ${
                  source === "robot"
                    ? "bg-emerald-400/15 text-emerald-200"
                    : "text-muted hover:text-slate-300"
                }`}
              >
                Real robot
              </button>
            </div>
          }
          bodyClassName="p-0"
          className="overflow-hidden"
        >
          <RobotTwin
            observation={twinObservation}
            action={twinAction}
            mode={twinMode}
            scenario={source === "simulation" ? simulationState?.scene : undefined}
            trajectory={source === "simulation" ? previewPath : undefined}
            sourceLabel={source === "simulation" ? "Isolated simulation" : "Real robot telemetry"}
            stale={twinStale}
            idle={source === "robot" && robotSampleState === "idle"}
            className="h-[38rem] min-h-[30rem] w-full"
          />
        </Card>

        <div className="space-y-6">
          {engine.kind === "physics" && (
            <Card title="Simulated camera" subtitle="Rendered by MuJoCo from the live scene">
              {engine.canRender ? (
                <SimCameraTile
                  cameras={engine.cameras}
                  sceneVersion={`${simulationState?.generation ?? "-"}:${simulationState?.step_count ?? "-"}`}
                />
              ) : engine.renderChecking ? (
                <EmptyState
                  title="Checking camera rendering"
                  detail="The server is still probing whether it can render camera frames. This view asks again every second."
                />
              ) : (
                <EmptyState
                  title="Camera frames unavailable"
                  detail={engine.renderProblem ?? "The server cannot render camera frames."}
                />
              )}
            </Card>
          )}
          <Card
            title="Runtime controls"
            subtitle="Request-synchronized fixed-step scheduler"
          >
            <div className="grid grid-cols-3 gap-2">
              <Button
                variant="success"
                onClick={handlePlay}
                disabled={!backendAvailable || busy !== null}
              >
                <Play aria-hidden="true" className="h-3.5 w-3.5" /> Play
              </Button>
              <Button
                variant="secondary"
                onClick={() =>
                  void runStateAction(
                    "pause",
                    pauseSimulation,
                    () => "Simulation paused at the current fixed step.",
                  )
                }
                disabled={!backendAvailable || busy !== null}
              >
                <Pause aria-hidden="true" className="h-3.5 w-3.5" /> Pause
              </Button>
              <Button
                variant="secondary"
                onClick={() =>
                  void runStateAction(
                    "step",
                    () =>
                      stepSimulation({
                        steps: manualSteps,
                        ...(targetDirty ? { targets: targetDraft } : {}),
                      }),
                    () => `Advanced ${manualSteps} fixed step${manualSteps === 1 ? "" : "s"}.`,
                  )
                }
                disabled={
                  !backendAvailable ||
                  simulationState?.status === "playing" ||
                  !validSteps ||
                  busy !== null
                }
              >
                <StepForward aria-hidden="true" className="h-3.5 w-3.5" /> Step
              </Button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <label className="flex flex-col gap-1.5 text-xs text-muted">
                Playback rate
                <select
                  value={playbackRate}
                  onChange={(event) => setPlaybackRate(Number(event.target.value))}
                  className={fieldClass}
                >
                  {[0.1, 0.25, 0.5, 1, 2, 4].map((rate) => (
                    <option key={rate} value={rate}>
                      {rate.toFixed(rate < 1 ? 2 : 1)}×
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1.5 text-xs text-muted">
                Manual steps
                <input
                  type="number"
                  min={1}
                  max={10000}
                  value={manualSteps}
                  onChange={(event) => setManualSteps(Number(event.target.value))}
                  className={fieldClass}
                />
              </label>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-lg bg-slate-950/50 p-3">
                <div className="text-muted">Status</div>
                <div className="mt-1 font-semibold capitalize text-slate-200">
                  {simulationState?.status ?? "offline"}
                </div>
              </div>
              <div className="rounded-lg bg-slate-950/50 p-3">
                <div className="text-muted">Target error</div>
                <div className="mt-1 font-mono font-semibold text-cyan-200">
                  {simulationState
                    ? Math.max(
                        ...Object.values(simulationState.telemetry.error).map(Math.abs),
                      ).toFixed(2)
                    : "—"}
                  {simulationState ? "°" : ""}
                </div>
              </div>
            </div>
          </Card>

          <Card
            title="Scenario and seed"
            subtitle="Reset is deterministic for the same manifest and seed"
          >
            <div className="space-y-2">
              {scenarios.map((scenario) => {
                const isSelected = activeScenario?.id === scenario.id;
                const isRunning = simulationState?.scene.scenario_id === scenario.id;
                return (
                  <button
                    key={scenario.id}
                    type="button"
                    onClick={() => setSelected(scenario.id)}
                    className={`w-full rounded-xl border p-3 text-left transition ${
                      isSelected
                        ? "border-cyan-400/30 bg-cyan-400/10"
                        : "border-white/5 bg-slate-950/30 hover:border-slate-700 hover:bg-slate-800/50"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-slate-200">
                        {scenario.name ?? scenario.label ?? scenario.id}
                      </span>
                      <span className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wide text-muted">
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            isRunning ? "bg-emerald-400" : "bg-slate-600"
                          }`}
                        />
                        {isRunning ? "loaded" : "ready"}
                      </span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      {scenario.description ?? scenario.detail ?? "Simulation scenario"}
                    </p>
                  </button>
                );
              })}
            </div>

            <div className="mt-4 grid grid-cols-[1fr_auto] items-end gap-3">
              <label className="flex flex-col gap-1.5 text-xs text-muted">
                Unsigned 32-bit seed
                <input
                  type="number"
                  min={0}
                  max={2 ** 32 - 1}
                  step={1}
                  value={seed}
                  onChange={(event) => setSeed(Number(event.target.value))}
                  className={fieldClass}
                />
              </label>
              <label className="mb-2 inline-flex items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={randomize}
                  onChange={(event) => setRandomize(event.target.checked)}
                  className="h-4 w-4 rounded-sm border-slate-600 bg-slate-900 text-cyan-400"
                />
                Randomize
              </label>
            </div>
            {!validSeed && (
              <p className="mt-2 text-xs text-rose-300">
                Seed must be an integer between 0 and 4,294,967,295.
              </p>
            )}
            <Button
              className="mt-4 w-full"
              onClick={handleReset}
              disabled={!backendAvailable || !validSeed || busy !== null}
            >
              <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" /> Load and reset scenario
            </Button>
          </Card>
        </div>
      </div>

      <Card
        title="Joint targets"
        subtitle="Degrees at the API boundary; values are rejected, never silently clamped"
        actions={
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                if (!simulationState) return;
                setTargetDirty(false);
              }}
              disabled={!simulationState || busy !== null}
            >
              Revert
            </Button>
            <Button
              size="sm"
              onClick={() =>
                void runStateAction(
                  "targets",
                  () => setSimulationTargets(targetDraft),
                  () => "Simulation target updated; physical hardware was not addressed.",
                )
              }
              disabled={
                !simulationState || !targetDirty || !Object.keys(targetDraft).length || busy !== null
              }
            >
              <Target aria-hidden="true" className="h-3.5 w-3.5" /> Apply target
            </Button>
          </div>
        }
      >
        {!simulationState ? (
          <EmptyState
            title="Simulation state unavailable"
            detail="Target controls appear after the isolated runtime responds."
          />
        ) : (
          <div className="grid gap-x-6 gap-y-5 md:grid-cols-2 xl:grid-cols-3">
            {simulationState.joint_order.map((joint) => {
              const limits = simulationState.joint_limits_deg[joint] ?? [-180, 180];
              const low = limits[0] ?? -180;
              const high = limits[1] ?? 180;
              const value = targetDraft[joint] ?? simulationState.telemetry.target[joint] ?? 0;
              return (
                <div key={joint} className="rounded-xl border border-white/5 bg-slate-950/30 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-xs font-semibold text-slate-200">
                        {JOINT_LABELS[joint] ?? joint}
                      </div>
                      <div className="mt-0.5 font-mono text-[9px] text-muted">{joint}</div>
                    </div>
                    <input
                      aria-label={`${JOINT_LABELS[joint] ?? joint} target in degrees`}
                      type="number"
                      min={low}
                      max={high}
                      step={0.1}
                      value={value}
                      onChange={(event) => editTarget(joint, Number(event.target.value))}
                      className="w-20 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-right font-mono text-xs text-cyan-200 outline-hidden focus:border-cyan-400/50 focus-visible:ring-2 focus-visible:ring-cyan-400/40"
                    />
                  </div>
                  <input
                    aria-label={`${JOINT_LABELS[joint] ?? joint} target slider`}
                    type="range"
                    min={low}
                    max={high}
                    step={0.1}
                    value={value}
                    onChange={(event) => editTarget(joint, Number(event.target.value))}
                    className="mt-4 w-full accent-cyan-400"
                  />
                  <div className="mt-2 flex justify-between font-mono text-[9px] text-muted">
                    <span>{low.toFixed(0)}°</span>
                    <span>
                      measured {(simulationState.telemetry.measured[joint] ?? 0).toFixed(1)}°
                    </span>
                    <span>{high.toFixed(0)}°</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <div className="grid gap-6 xl:grid-cols-[minmax(20rem,0.7fr)_minmax(0,1.3fr)]">
        <Card
          title="Trajectory preview"
          subtitle={
            engine.kind === "physics"
              ? "Side-effect-free physics rollout on a copy of the live world"
              : "Side-effect-free rate-limited rollout from the current measured pose"
          }
        >
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1.5 text-xs text-muted">
              Duration (seconds)
              <input
                type="number"
                min={0.01}
                max={30}
                step={0.1}
                value={previewDuration}
                onChange={(event) => setPreviewDuration(Number(event.target.value))}
                className={fieldClass}
              />
            </label>
            <label className="flex flex-col gap-1.5 text-xs text-muted">
              Samples / second
              <input
                type="number"
                min={1}
                max={simulationState?.fixed_hz ?? 60}
                step={1}
                value={previewSampleHz}
                onChange={(event) => setPreviewSampleHz(Number(event.target.value))}
                className={fieldClass}
              />
            </label>
          </div>
          <Button
            className="mt-4 w-full"
            variant="secondary"
            onClick={() => void handlePreview()}
            disabled={
              !simulationState ||
              !Object.keys(targetDraft).length ||
              previewDuration < 0.01 ||
              previewDuration > 30 ||
              previewSampleHz < 1 ||
              previewSampleHz > (simulationState?.fixed_hz ?? 60) ||
              busy !== null
            }
          >
            <Activity aria-hidden="true" className="h-3.5 w-3.5" />
            {busy === "preview" ? "Computing…" : "Preview current target"}
          </Button>

          <div className="mt-4 rounded-xl border border-violet-400/15 bg-violet-400/[0.04] p-3 text-xs leading-relaxed text-slate-400">
            Preview points are rendered as a violet end-effector path in the 3D
            view. The operation never mutates runtime state or commands hardware.
          </div>

          {trajectory && (
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-slate-950/50 p-3">
                <div className="text-[9px] uppercase tracking-wide text-muted">Samples</div>
                <div className="mt-1 font-mono text-sm text-slate-200">{trajectory.points.length}</div>
              </div>
              <div className="rounded-lg bg-slate-950/50 p-3">
                <div className="text-[9px] uppercase tracking-wide text-muted">Effective rate</div>
                <div className="mt-1 font-mono text-sm text-slate-200">{trajectory.sample_hz.toFixed(1)} Hz</div>
              </div>
              <div className="rounded-lg bg-slate-950/50 p-3">
                <div className="text-[9px] uppercase tracking-wide text-muted">Max error</div>
                <div className="mt-1 font-mono text-sm text-slate-200">{maximumPreviewError.toFixed(2)}°</div>
              </div>
              <div className="rounded-lg bg-slate-950/50 p-3">
                <div className="text-[9px] uppercase tracking-wide text-muted">Tool travel</div>
                <div className="mt-1 font-mono text-sm text-slate-200">{(endEffectorTravel * 100).toFixed(1)} cm</div>
              </div>
              {typeof trajectory.reached_tolerance === "number" &&
                Number.isFinite(trajectory.reached_tolerance) && (
                  <div className="col-span-2 rounded-lg bg-slate-950/50 p-3">
                    <div className="text-[9px] uppercase tracking-wide text-muted">Reach tolerance</div>
                    <div className="mt-1 font-mono text-sm text-slate-200">
                      ±{formatTolerance(trajectory.reached_tolerance)}
                    </div>
                    <div className="mt-0.5 text-[10px] text-slate-400">
                      Degrees for the arm joints, 0-100 units for the gripper
                    </div>
                  </div>
                )}
            </div>
          )}
          {trajectory && previewSemantics(trajectory.semantics) && (
            <p className="mt-3 text-xs leading-relaxed text-slate-400">
              {previewSemantics(trajectory.semantics)}
            </p>
          )}
        </Card>

        <Card
          title="Preview trace"
          subtitle={
            trajectory
              ? `${trajectory.quantized_duration_s.toFixed(3)} s quantized duration · ${reachSummary(trajectory)}`
              : "Generate a preview to compare desired and rate-limited motion"
          }
          actions={
            trajectory ? (
              <select
                aria-label="Trajectory chart joint"
                value={chartJoint}
                onChange={(event) => setChartJoint(event.target.value)}
                className={`${fieldClass} py-1 text-xs`}
              >
                {(simulationState?.joint_order ?? Object.keys(trajectory.starting_target)).map(
                  (joint) => (
                    <option key={joint} value={joint}>
                      {JOINT_LABELS[joint] ?? joint}
                    </option>
                  ),
                )}
              </select>
            ) : undefined
          }
        >
          {!trajectory ? (
            <EmptyState
              title="No trajectory preview"
              detail="Choose a joint target and generate a side-effect-free preview."
              icon={Activity}
            />
          ) : (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={previewChart}
                  margin={{ top: 8, right: 12, bottom: 4, left: -8 }}
                >
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="time"
                    type="number"
                    domain={[0, "dataMax"]}
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    tickFormatter={(value: number) => `${value.toFixed(1)}s`}
                  />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} unit="°" />
                  <Tooltip
                    contentStyle={{
                      background: "#020617",
                      border: "1px solid #334155",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    labelFormatter={(value) => `${Number(value).toFixed(3)} s`}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    type="monotone"
                    dataKey="measured"
                    name="Measured"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="target"
                    name="Desired"
                    stroke="#34d399"
                    strokeWidth={1.5}
                    strokeDasharray="5 4"
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="error"
                    name="Absolute error"
                    stroke="#a78bfa"
                    strokeWidth={1.2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(20rem,0.65fr)]">
        <Card
          title="Joint binding"
          subtitle={
            source === "simulation"
              ? "Simulated measured position / simulated target"
              : "Real observed position / real commanded position"
          }
        >
          <JointBars observation={twinObservation} action={twinAction} />
        </Card>

        <Card title="Scene inspector" subtitle="Instantiated manifest state">
          {!simulationState ? (
            <EmptyState
              title="No scene state"
              detail="The inspector populates from the isolated simulation API."
            />
          ) : (
            <div className="space-y-4">
              <div>
                <SectionLabel>Loaded scene</SectionLabel>
                <div className="mt-2 flex items-center justify-between gap-3">
                  <span className="font-mono text-sm text-cyan-300">
                    {simulationState.scene.scenario_id}
                  </span>
                  <span className="rounded-full border border-slate-700 px-2 py-0.5 font-mono text-[10px] text-slate-400">
                    seed {simulationState.scene.seed}
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg bg-slate-950/50 p-3">
                  <div className="text-muted">Objects</div>
                  <div className="mt-1 font-semibold text-slate-300">
                    {simulationState.scene.objects.length}
                  </div>
                </div>
                <div className="rounded-lg bg-slate-950/50 p-3">
                  <div className="text-muted">Randomization</div>
                  <div className="mt-1 font-semibold text-slate-300">
                    {simulationState.scene.randomized ? "Seeded" : "Nominal"}
                  </div>
                </div>
              </div>
              <div className="max-h-48 space-y-2 overflow-y-auto pr-1 scroll-thin">
                {simulationState.scene.objects.length ? (
                  simulationState.scene.objects.map((object) => {
                    const sample = simulationState.scene.randomization.samples[object.id];
                    return (
                      <div
                        key={object.id}
                        className="rounded-lg border border-white/5 bg-slate-950/30 p-2.5"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="font-mono text-xs text-slate-300">{object.id}</span>
                          <span className="text-[9px] uppercase tracking-wide text-muted">
                            {object.shape} · {object.role ?? "scene"}
                          </span>
                        </div>
                        <div className="mt-1 font-mono text-[9px] text-muted">
                          xyz {object.pose.slice(0, 3).map((value) => value.toFixed(3)).join(" / ")} m
                          {sample?.applied ? ` · scale ${sample.scale.toFixed(3)}` : " · fixed"}
                          {objectMotion(simulationState.physics_state?.objects?.[object.id])}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-xs text-muted">This scene contains no task objects.</p>
                )}
              </div>
              {simulationState.physics_state && (
                <PhysicsReadout
                  physics={simulationState.physics_state}
                  jointOrder={simulationState.joint_order}
                />
              )}
              <p className="text-xs leading-relaxed text-muted">
                {simulationState.truth_model}
              </p>
            </div>
          )}
        </Card>
      </div>

    </div>
  );
}

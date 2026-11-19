// What the Simulation page says about the engine. Physics versus kinematic
// preview comes from the backend's own flags (GET /api/simulation/capabilities,
// then the state's `physics`), never from labels or engine names.
import type {
  SimulationCapabilities,
  SimulationMujocoAdapter,
  SimulationState,
  SimulationTrajectoryPreview,
} from "../../api";

export type EngineKind = "physics" | "kinematic" | "none" | "unknown";

export interface EngineView {
  kind: EngineKind;
  tone: "info" | "warn" | "danger";
  title: string;
  /** What this engine models. */
  summary: string;
  /** Why this engine runs, verbatim from the backend. */
  why: string | null;
  /** Physics only: why camera frames cannot be rendered, verbatim. */
  renderProblem: string | null;
  /** Camera frames are worth asking for. */
  canRender: boolean;
  /** Physics only: the server is still probing whether it can render; ask again soon. */
  renderChecking: boolean;
  cameras: string[];
  /** One line for the Runtime tile. */
  runtimeDetail: string;
}

export const KINEMATIC_SUMMARY =
  "This runtime models bounded joint motion and approximate forward kinematics. It does not model contacts, collision, torque, gravity, self-collision, or motor dynamics, and it never writes motor commands.";

const DEFAULT_CAMERAS = ["front"];

function formatMs(seconds: number): string {
  const ms = seconds * 1000;
  return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

const UNAVAILABLE_PREFIX = "unavailable:";

/**
 * The server answers capabilities without waiting for its GL probe; while the
 * probe runs, render_available is null and render_status is "probing". That
 * is "checking", never "unavailable". An older server that sends neither field
 * is not probing: the camera tile asks the render endpoint and reports it.
 */
function isProbing(mujoco: SimulationMujocoAdapter | undefined): boolean {
  return mujoco?.render_status === "probing" || mujoco?.render_available === null;
}

/** Physics runs and its render probe has not answered yet: poll capabilities again soon. */
export function renderProbing(capabilities: SimulationCapabilities | null): boolean {
  return !!capabilities?.physics_active && isProbing(capabilities.adapters.mujoco);
}

function physicsView(mujoco: SimulationMujocoAdapter | undefined, why: string | null): EngineView {
  const substeps = mujoco?.substeps_per_tick;
  const timestep = mujoco?.timestep_s;
  const hz = mujoco?.fixed_hz;
  const clock =
    finite(substeps) && finite(timestep) && finite(hz)
      ? `: ${substeps} substeps of ${formatMs(timestep)} per ${Number(hz.toFixed(1))} Hz tick`
      : "";
  const checking = isProbing(mujoco);
  const status = mujoco?.render_status ?? "";
  const renderBroken =
    !checking && (mujoco?.render_available === false || status.startsWith(UNAVAILABLE_PREFIX));
  return {
    kind: "physics",
    tone: "info",
    title: "MuJoCo physics",
    summary: `Gravity, contacts, friction and the STS3215 servo profile are simulated${clock}. The simulator is isolated from the hardware and never writes motor commands.`,
    why,
    renderProblem: renderBroken
      ? mujoco?.render_error ||
        status.slice(UNAVAILABLE_PREFIX.length).trim() ||
        "The server reports that it cannot render camera frames."
      : null,
    canRender: !renderBroken && !checking,
    renderChecking: checking,
    cameras: mujoco?.cameras?.length ? [...mujoco.cameras] : DEFAULT_CAMERAS,
    runtimeDetail:
      finite(substeps) && finite(timestep)
        ? `MuJoCo physics · ${substeps} × ${formatMs(timestep)} substeps`
        : "MuJoCo physics",
  };
}

function kinematicView(why: string | null): EngineView {
  return {
    kind: "kinematic",
    tone: "warn",
    title: "Kinematic preview: no physics",
    summary: KINEMATIC_SUMMARY,
    why,
    renderProblem: null,
    canRender: false,
    renderChecking: false,
    cameras: [],
    runtimeDetail: "Kinematic fixed-step · no physics",
  };
}

function joinWhy(...parts: Array<string | null | undefined>): string | null {
  const text = parts.filter((part): part is string => !!part && part.trim() !== "");
  return text.length ? text.join(" ") : null;
}

export function describeEngine(
  capabilities: SimulationCapabilities | null,
  state: SimulationState | null,
): EngineView {
  if (capabilities) {
    const mujoco = capabilities.adapters.mujoco;
    if (capabilities.physics_active) return physicsView(mujoco, mujoco?.selection_reason ?? null);
    const why = joinWhy(mujoco?.selection_reason, mujoco?.render_error);
    if (capabilities.backend_kind === "kinematic" || capabilities.active_backend === "kinematic") {
      return kinematicView(why);
    }
    return {
      kind: "none",
      tone: "danger",
      title: "No simulation engine can run",
      summary:
        "The simulation API answers, but neither engine is serving it. Scene, target and preview requests fail until the configuration is fixed.",
      why: why ?? mujoco?.error ?? null,
      renderProblem: null,
      canRender: false,
      renderChecking: false,
      cameras: [],
      runtimeDetail: "No simulation engine",
    };
  }
  // Capabilities not reported (yet): the state's own flag is the next best
  // backend truth. Render details are unknown, so the camera tile reports
  // whatever the render endpoint says.
  if (state) return state.physics ? physicsView(undefined, null) : kinematicView(null);
  return {
    kind: "unknown",
    tone: "warn",
    title: "Simulation engine not reported",
    summary:
      "The backend has not said which engine runs. Until it does, treat the scene as a kinematic preview: no contacts, forces or motor dynamics.",
    why: null,
    renderProblem: null,
    canRender: false,
    renderChecking: false,
    cameras: [],
    runtimeDetail: "Engine not reported",
  };
}

const SEMANTICS_TEXT: Record<string, string> = {
  step_targets_like_play:
    "Targets are applied tick by tick, exactly as Apply target followed by Step or Play would apply them.",
};

/** The preview's `semantics` in words; unknown values are shown as sent. */
export function previewSemantics(semantics: string | null | undefined): string | null {
  if (!semantics) return null;
  return SEMANTICS_TEXT[semantics] ?? `Preview semantics: ${semantics}`;
}

export function formatTolerance(tolerance: number): string {
  return String(Number(tolerance.toPrecision(2)));
}

/** "target reached within ±0.5", or why it was not. */
export function reachSummary(preview: SimulationTrajectoryPreview): string {
  const tolerance = preview.reached_tolerance;
  const within =
    typeof tolerance === "number" && Number.isFinite(tolerance)
      ? ` within ±${formatTolerance(tolerance)}`
      : "";
  if (preview.final_reached) return `target reached${within}`;
  return preview.physics
    ? `target not reached${within} by the end of the preview`
    : "rate-limited before target";
}

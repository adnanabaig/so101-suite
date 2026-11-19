import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type {
  SimulationCapabilities,
  SimulationMujocoAdapter,
  SimulationPhysicsState,
  SimulationState,
  SimulationTrajectoryPreview,
} from "../api";
import { KINEMATIC_SUMMARY } from "../components/simulation/engine";
import { RENDER_PROBE_POLL_MS } from "../components/simulation/useSimulationCapabilities";
import { TelemetryProvider } from "../hooks/useTelemetry";
import {
  FakeWebSocket,
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  telemetryFrame,
  type FakeBackend,
} from "../test/harness";
import { SimulationWorkspace as Simulation } from "./Simulation";

// The WebGL twin has its own tests. Here the stub records what the page
// tells it about freshness and source.
vi.mock("../components/RobotTwin", () => ({
  default: (props: { stale?: boolean; idle?: boolean; sourceLabel?: string }) => (
    <div
      data-testid="robot-twin"
      data-stale={String(Boolean(props.stale))}
      data-idle={String(Boolean(props.idle))}
      data-source={props.sourceLabel ?? ""}
    />
  ),
}));

const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
const zeros = () => Object.fromEntries(JOINTS.map((joint) => [joint, 0]));

function simState(overrides: Partial<SimulationState> = {}): SimulationState {
  return {
    backend: "kinematic",
    backend_kind: "kinematic",
    physics: false,
    truth_model: "Kinematic truth model: no contacts.",
    hardware_isolated: true,
    writes_motor_commands: false,
    status: "paused",
    scheduler: "request-synchronized",
    playback_rate: 1,
    fixed_dt_s: 1 / 60,
    fixed_hz: 60,
    step_count: 42,
    sim_time_s: 0.7,
    generation: 3,
    updated_at: "2026-10-02T00:00:00Z",
    joint_order: JOINTS,
    joint_limits_deg: Object.fromEntries(JOINTS.map((joint) => [joint, [-110, 110]])),
    telemetry: {
      source: "simulation",
      position_units: "degrees",
      velocity_units: "degrees/s",
      measured: zeros(),
      target: zeros(),
      error: zeros(),
      velocity: zeros(),
    },
    kinematics: {
      model: "so101_new_calib",
      position_m: [0.2, 0, 0.1],
      orientation_rpy_deg: [0, 0, 0],
      gripper_aperture_m: 0.02,
      frames: [],
    },
    scene: {
      scenario_id: "tabletop_pick",
      label: "Tabletop pick",
      description: "Single-object pick and place workspace.",
      seed: 101,
      randomized: true,
      surface: { size_m: [0.6, 0.6], height_m: 0 },
      objects: [
        { id: "cube", shape: "box", size_m: [0.03, 0.03, 0.03], pose: [0.25, 0, 0.015, 0, 0, 0], role: "manipuland" },
      ],
      randomization: {
        applied: true,
        seed: 101,
        settings: {},
        samples: { cube: { position_offset_m: [0, 0, 0], yaw_offset_deg: 0, scale: 1, applied: true } },
      },
    },
    physics_state: null,
    ...overrides,
  };
}

const KINEMATIC_REASON =
  "auto: MuJoCo is unavailable (ModuleNotFoundError: No module named 'mujoco'), so the kinematic engine runs";

function mujocoAdapter(overrides: Partial<SimulationMujocoAdapter> = {}): SimulationMujocoAdapter {
  return {
    package_installed: true,
    package_version: "3.3.7",
    adapter_implemented: true,
    available: true,
    active: true,
    kind: "physics",
    engine_setting: "mujoco",
    selection_reason: "SO101_SIM_ENGINE=mujoco",
    error: null,
    render_available: true,
    render_error: null,
    fixed_hz: 60,
    timestep_s: 1 / 240,
    substeps_per_tick: 4,
    servo_mode: "profile",
    cameras: ["front", "wrist_cam"],
    detail: "MuJoCo physics of the vendored Menagerie SO-101.",
    ...overrides,
  };
}

function capabilities(
  kind: "physics" | "kinematic" | "none",
  mujoco: Partial<SimulationMujocoAdapter> = {},
): SimulationCapabilities {
  const physics = kind === "physics";
  return {
    active_backend: physics ? "mujoco" : kind,
    backend_kind: kind,
    physics_active: physics,
    fixed_step: true,
    hardware_isolated: true,
    writes_motor_commands: false,
    adapters: {
      kinematic: {
        available: true,
        active: kind === "kinematic",
        kind: "kinematic",
        model: "so101_new_calib",
        detail: "Deterministic rate-limited joint-space motion.",
      },
      mujoco: mujocoAdapter({
        active: physics,
        available: physics,
        selection_reason: physics ? "SO101_SIM_ENGINE=mujoco" : KINEMATIC_REASON,
        render_available: physics ? true : null,
        fixed_hz: physics ? 60 : null,
        timestep_s: physics ? 1 / 240 : null,
        substeps_per_tick: physics ? 4 : null,
        cameras: physics ? ["front", "wrist_cam"] : null,
        ...mujoco,
      }),
    },
  };
}

function physicsState(overrides: Partial<SimulationPhysicsState> = {}): SimulationPhysicsState {
  return {
    engine: "mujoco",
    mujoco_version: "3.3.7",
    model: "so101_new_calib",
    timestep_s: 1 / 240,
    substeps_per_tick: 4,
    physics_steps: 168,
    physics_time_s: 0.7,
    dropped_ticks: 0,
    torque_enabled: true,
    servo: {
      mode: "profile",
      acceleration_deg_s2: 2000,
      max_speed: {},
      speed_units: "deg/s",
      reference: {},
    },
    effort_pct: { shoulder_pan: 12.34, shoulder_lift: -40, elbow_flex: 3, wrist_flex: 0, wrist_roll: 0, gripper: 55.55 },
    contacts: { total: 3, arm_self: 0, arm_table: 1, max_penetration_m: 0.0004 },
    objects: {
      cube: {
        position_m: [0.25, 0, 0.015],
        quat_wxyz: [1, 0, 0, 0],
        speed_m_s: 0.051,
        grasped: true,
        contacts: { table: 1 },
      },
    },
    markers: {},
    ...overrides,
  };
}

function preview(overrides: Partial<SimulationTrajectoryPreview> = {}): SimulationTrajectoryPreview {
  const point = (step: number, time: number) => ({
    step,
    time_s: time,
    measured: zeros(),
    target: zeros(),
    error: zeros(),
    end_effector: {
      model: "so101_new_calib",
      position_m: [0.2, 0, 0.1 + time / 100],
      orientation_rpy_deg: [0, 0, 0],
      gripper_aperture_m: 0.02,
      frames: [],
    },
  });
  return {
    backend: "mujoco",
    physics: true,
    mutates_state: false,
    source_step_count: 42,
    fixed_dt_s: 1 / 60,
    requested_duration_s: 1,
    quantized_duration_s: 1,
    requested_sample_hz: 20,
    sample_hz: 20,
    points: [point(42, 0), point(52, 0.5), point(62, 1)],
    final_reached: false,
    joint_limits_deg: {},
    starting_target: zeros(),
    semantics: "step_targets_like_play",
    reached_tolerance: 0.5,
    ...overrides,
  };
}

let backend: FakeBackend;
let state: SimulationState;
let created = 0;
const originalCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");

function frame(): Response {
  return new Response(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), {
    headers: { "content-type": "image/jpeg", "x-so101-sim-step": String(state.step_count) },
  });
}

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  state = simState();
  created = 0;
  backend.route("GET", "/api/simulation/scenarios", () =>
    jsonResponse([{ id: "tabletop_pick", name: "Tabletop pick", description: "Single-object pick." }]),
  );
  backend.route("GET", "/api/simulation/state", () => jsonResponse(state));
  backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("kinematic")));
  backend.route("GET", "/api/simulation/render", () => frame());
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    writable: true,
    value: () => `blob:frame-${++created}`,
  });
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, writable: true, value: () => undefined });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  for (const [name, descriptor] of [
    ["createObjectURL", originalCreate],
    ["revokeObjectURL", originalRevoke],
  ] as const) {
    if (descriptor) Object.defineProperty(URL, name, descriptor);
    else Reflect.deleteProperty(URL, name);
  }
});

async function renderSimulation() {
  render(
    <MemoryRouter>
      <TelemetryProvider>
        <Simulation />
      </TelemetryProvider>
    </MemoryRouter>,
  );
  await advance(10);
}

/**
 * Advance in small slices. Each advance() is one act() scope, and React only
 * commits at the end of a scope, so one long advance would hide the renders
 * (and the effects) that happen between timers in a real browser.
 */
async function advanceRendering(ms: number, slice = 50) {
  for (let elapsed = 0; elapsed < ms; elapsed += slice) await advance(slice);
}

const renderCalls = () => backend.callsTo("/api/simulation/render", "GET");
const button = (name: string | RegExp) => screen.getByRole("button", { name });
const twin = () => screen.getByTestId("robot-twin");

describe("Simulation engine: physics versus preview comes from the backend's flags", () => {
  test("the kinematic engine says it is not physics, and why, verbatim", async () => {
    await renderSimulation();
    expect(screen.getByText("Kinematic preview: no physics")).toBeInTheDocument();
    expect(screen.getByText(KINEMATIC_SUMMARY, { exact: false })).toBeInTheDocument();
    expect(screen.getByText(KINEMATIC_REASON)).toBeInTheDocument();
    expect(screen.getByText("Kinematic fixed-step · no physics")).toBeInTheDocument();
    expect(screen.getByText("Side-effect-free rate-limited rollout from the current measured pose")).toBeInTheDocument();
    expect(screen.queryByText("Simulated camera")).toBeNull();
    await advance(3000);
    expect(renderCalls()).toHaveLength(0);
  });

  test("MuJoCo physics says so, with its clock, and shows the camera", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true });
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("physics")));
    await renderSimulation();
    expect(screen.getByText("MuJoCo physics")).toBeInTheDocument();
    expect(
      screen.getByText(/Gravity, contacts, friction and the STS3215 servo profile are simulated: 4 substeps of 4\.17 ms per 60 Hz tick\./),
    ).toBeInTheDocument();
    expect(screen.getByText("SO101_SIM_ENGINE=mujoco")).toBeInTheDocument();
    expect(screen.getByText("MuJoCo physics · 4 × 4.17 ms substeps")).toBeInTheDocument();
    expect(screen.getByText("Side-effect-free physics rollout on a copy of the live world")).toBeInTheDocument();
    expect(screen.queryByText(/no physics/i)).toBeNull();

    expect(screen.getByText("Simulated camera")).toBeInTheDocument();
    expect(renderCalls()).toHaveLength(1);
    expect(screen.getByRole("img")).toHaveAccessibleName("Simulated front camera at step 42");
    expect(within(screen.getByLabelText("Camera")).getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Front",
      "Wrist",
    ]);
  });

  test("a scene that moves gets new frames; a paused one does not", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true });
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("physics")));
    await renderSimulation();
    await advance(2000);
    expect(renderCalls()).toHaveLength(1);

    state = { ...state, step_count: 90, status: "playing" };
    await advanceRendering(1000);
    expect(renderCalls()).toHaveLength(2);
    expect(screen.getByRole("img")).toHaveAccessibleName("Simulated front camera at step 90");
  });

  test("a server that cannot render says why and is never asked for frames", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true });
    backend.route("GET", "/api/simulation/capabilities", () =>
      jsonResponse(
        capabilities("physics", { render_available: false, render_error: "GLError: no OpenGL context is available" }),
      ),
    );
    await renderSimulation();
    expect(screen.getByText("MuJoCo physics")).toBeInTheDocument();
    expect(screen.getAllByText("GLError: no OpenGL context is available").length).toBeGreaterThan(0);
    expect(screen.getByText(/Camera frames are unavailable:/)).toBeInTheDocument();
    expect(screen.getByText("Camera frames unavailable")).toBeInTheDocument();
    await advance(3000);
    expect(renderCalls()).toHaveLength(0);
  });

  test("while the server probes GL the camera says it is checking, and capabilities are asked again every second", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true });
    let probing = true;
    backend.route("GET", "/api/simulation/capabilities", () =>
      jsonResponse(
        capabilities(
          "physics",
          probing ? { render_available: null, render_status: "probing" } : { render_available: true, render_status: "available" },
        ),
      ),
    );
    const capabilityCalls = () => backend.callsTo("/api/simulation/capabilities", "GET").length;
    await renderSimulation();
    expect(screen.getByText("Checking camera rendering")).toBeInTheDocument();
    expect(screen.queryByText("Camera frames unavailable")).toBeNull();
    expect(renderCalls()).toHaveLength(0);

    const before = capabilityCalls();
    await advanceRendering(RENDER_PROBE_POLL_MS * 3);
    expect(capabilityCalls()).toBeGreaterThanOrEqual(before + 3);
    expect(renderCalls()).toHaveLength(0);

    probing = false;
    await advanceRendering(RENDER_PROBE_POLL_MS + 100);
    expect(screen.queryByText("Checking camera rendering")).toBeNull();
    expect(renderCalls()).toHaveLength(1);
    expect(screen.getByRole("img")).toHaveAccessibleName("Simulated front camera at step 42");

    // Back to the slow poll once the probe has answered.
    const settled = capabilityCalls();
    await advanceRendering(5000, 500);
    expect(capabilityCalls()).toBeLessThanOrEqual(settled + 1);
  });

  test("no engine: the reason is shown and nothing can be driven", async () => {
    const reason = "SO101_SIM_ENGINE=mujoco, but MuJoCo cannot run here: ModuleNotFoundError: No module named 'mujoco'";
    backend.route("GET", "/api/simulation/capabilities", () =>
      jsonResponse(capabilities("none", { selection_reason: reason, available: false })),
    );
    backend.route("GET", "/api/simulation/state", () => jsonResponse({ detail: reason }, 503));
    backend.route("GET", "/api/simulation/scenarios", () => jsonResponse({ detail: reason }, 503));
    await renderSimulation();
    expect(screen.getByText("No simulation engine can run")).toBeInTheDocument();
    expect(screen.getAllByText(reason, { exact: false }).length).toBeGreaterThan(0);
    expect(screen.getByText("No simulation engine")).toBeInTheDocument();
    for (const name of [/Play/, /Pause/, /Step/, /Reset scene/, /Load and reset scenario/]) {
      expect(button(name)).toBeDisabled();
    }
  });

  test("an older server without capabilities: the state's own physics flag decides", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true });
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse({ detail: "Not Found" }, 404));
    await renderSimulation();
    // The notice title and the Runtime tile; the clock is not known.
    expect(screen.getAllByText("MuJoCo physics")).toHaveLength(2);
    expect(screen.getByText(/servo profile are simulated\. The simulator is isolated/)).toBeInTheDocument();
    expect(screen.queryByText(/^Why:/)).toBeNull();
    // Render details are unknown, so the tile asks and reports what it gets.
    expect(renderCalls()).toHaveLength(1);
  });

  test("an older kinematic server without capabilities still reads as a preview", async () => {
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse({ detail: "Not Found" }, 404));
    await renderSimulation();
    expect(screen.getByText("Kinematic preview: no physics")).toBeInTheDocument();
    expect(screen.queryByText("Simulated camera")).toBeNull();
  });

  test("nothing reported: the page says so instead of guessing", async () => {
    backend.route("GET", "/api/simulation/capabilities", () => new Response("", { status: 502 }));
    backend.route("GET", "/api/simulation/state", () => new Response("", { status: 502 }));
    await renderSimulation();
    expect(screen.getByText("Simulation engine not reported")).toBeInTheDocument();
    expect(screen.getByText("Engine not reported")).toBeInTheDocument();
    expect(button(/Play/)).toBeDisabled();
  });
});

describe("Simulation controls follow the live state poll (F28)", () => {
  test("two failed polls disable the controls and stale the twin; recovery restores both", async () => {
    await renderSimulation();
    expect(button(/Play/)).toBeEnabled();
    expect(twin()).toHaveAttribute("data-stale", "false");
    expect(twin()).toHaveAttribute("data-source", "Isolated simulation");

    backend.route("GET", "/api/simulation/state", () => jsonResponse({ detail: "backend restarting" }, 500));
    await advance(260);
    // One failure is a blip: the last state stays usable.
    expect(button(/Play/)).toBeEnabled();
    await advance(260);
    expect(button(/Play/)).toBeDisabled();
    expect(button(/Pause/)).toBeDisabled();
    expect(button(/Step/)).toBeDisabled();
    expect(button(/Reset scene/)).toBeDisabled();
    expect(twin()).toHaveAttribute("data-stale", "true");
    expect(screen.getByText("State polling interrupted")).toBeInTheDocument();

    backend.route("GET", "/api/simulation/state", () => jsonResponse(state));
    await advance(260);
    expect(button(/Play/)).toBeEnabled();
    expect(twin()).toHaveAttribute("data-stale", "false");
  });

  test("the real robot source tells the twin when the pose is only the last one read", async () => {
    await renderSimulation();
    fireEvent.click(screen.getByRole("button", { name: "Real robot" }));
    act(() =>
      FakeWebSocket.broadcast(
        telemetryFrame(false, { observation: { "shoulder_pan.pos": 3 }, observation_live: false, sample_age_ms: 40 }),
      ),
    );
    await advance(10);
    expect(twin()).toHaveAttribute("data-source", "Real robot telemetry");
    expect(twin()).toHaveAttribute("data-idle", "true");

    act(() => FakeWebSocket.broadcast(telemetryFrame(false, { observation: { "shoulder_pan.pos": 4 } })));
    await advance(10);
    expect(twin()).toHaveAttribute("data-idle", "false");
  });
});

describe("Simulation physics readout", () => {
  test("contacts, servo load, grasp and skipped ticks come from physics_state", async () => {
    state = simState({
      backend: "mujoco",
      backend_kind: "physics",
      physics: true,
      physics_state: physicsState({ dropped_ticks: 7 }),
    });
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("physics")));
    await renderSimulation();
    const readout = screen.getByRole("region", { name: "Physics state" });
    const term = (name: string) => within(readout).getByText(name).nextElementSibling?.textContent;
    expect(term("Contacts")).toBe("3");
    expect(term("Arm on table")).toBe("1");
    expect(term("Arm on itself")).toBe("0");
    expect(term("Shoulder pan")).toBe("12.3 %");
    expect(term("Shoulder lift")).toBe("-40.0 %");
    expect(term("Gripper")).toBe("55.5 %");
    expect(within(readout).getByText(/Servo profile mode · torque on · deepest penetration 0\.40 mm/)).toBeInTheDocument();
    expect(within(readout).getByRole("note")).toHaveTextContent(
      "7 ticks were skipped because nothing polled the simulator for more than 2 s during play. The clock counts only simulated ticks.",
    );
    expect(screen.getByText(/xyz 0\.250 \/ 0\.000 \/ 0\.015 m · scale 1\.000 · 5\.1 cm\/s · grasped/)).toBeInTheDocument();
  });

  test("no skipped ticks, no note; the kinematic engine shows no readout", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true, physics_state: physicsState() });
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("physics")));
    await renderSimulation();
    expect(within(screen.getByRole("region", { name: "Physics state" })).queryByRole("note")).toBeNull();

    state = simState();
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("kinematic")));
    await advance(260);
    expect(screen.queryByRole("region", { name: "Physics state" })).toBeNull();
  });
});

describe("Simulation trajectory preview", () => {
  test("shows how targets were applied, the reach tolerance, and the outcome", async () => {
    state = simState({ backend: "mujoco", backend_kind: "physics", physics: true });
    backend.route("GET", "/api/simulation/capabilities", () => jsonResponse(capabilities("physics")));
    backend.route("POST", "/api/simulation/trajectory/preview", () => jsonResponse(preview()));
    await renderSimulation();
    fireEvent.click(button("Preview current target"));
    await advance(10);
    expect(backend.callsTo("/api/simulation/trajectory/preview")[0]?.body).toEqual({
      target: zeros(),
      duration_s: 1,
      sample_hz: 20,
    });
    expect(
      screen.getByText("Targets are applied tick by tick, exactly as Apply target followed by Step or Play would apply them."),
    ).toBeInTheDocument();
    expect(screen.getByText("Reach tolerance")).toBeInTheDocument();
    expect(screen.getByText("±0.5")).toBeInTheDocument();
    expect(screen.getByText("1.000 s quantized duration · target not reached within ±0.5 by the end of the preview")).toBeInTheDocument();
    expect(screen.getByText(/runtime state was not changed\./)).toBeInTheDocument();
  });

  test("an older server without the new fields keeps the old wording", async () => {
    backend.route("POST", "/api/simulation/trajectory/preview", () =>
      jsonResponse(
        preview({ backend: "kinematic", physics: false, semantics: undefined, reached_tolerance: undefined, final_reached: false }),
      ),
    );
    await renderSimulation();
    fireEvent.click(button("Preview current target"));
    await advance(10);
    expect(screen.queryByText("Reach tolerance")).toBeNull();
    expect(screen.getByText("1.000 s quantized duration · rate-limited before target")).toBeInTheDocument();
  });

  test("a preview that says it changed the runtime is reported as such", async () => {
    backend.route("POST", "/api/simulation/trajectory/preview", () =>
      jsonResponse(preview({ mutates_state: true, final_reached: true })),
    );
    await renderSimulation();
    fireEvent.click(button("Preview current target"));
    await advance(10);
    expect(screen.getByText(/the backend reports that it changed the runtime state\./)).toBeInTheDocument();
    expect(screen.getByText("1.000 s quantized duration · target reached within ±0.5")).toBeInTheDocument();
  });
});

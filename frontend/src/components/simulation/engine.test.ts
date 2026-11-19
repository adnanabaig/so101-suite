import { describe, expect, test } from "vitest";
import type { SimulationCapabilities, SimulationState, SimulationTrajectoryPreview } from "../../api";
import {
  KINEMATIC_SUMMARY,
  describeEngine,
  formatTolerance,
  previewSemantics,
  reachSummary,
  renderProbing,
} from "./engine";

function caps(overrides: Partial<SimulationCapabilities>): SimulationCapabilities {
  return {
    active_backend: "kinematic",
    backend_kind: "kinematic",
    physics_active: false,
    fixed_step: true,
    hardware_isolated: true,
    writes_motor_commands: false,
    adapters: {},
    ...overrides,
  };
}

const stateWith = (physics: boolean) => ({ physics }) as SimulationState;

describe("describeEngine", () => {
  test("physics_active decides, whatever the engine is called", () => {
    const view = describeEngine(
      caps({
        active_backend: "some-future-engine",
        backend_kind: "kinematic",
        physics_active: true,
        adapters: { mujoco: { active: true, kind: "physics", detail: "", selection_reason: "SO101_SIM_ENGINE=auto" } },
      }),
      stateWith(false),
    );
    expect(view.kind).toBe("physics");
    expect(view.why).toBe("SO101_SIM_ENGINE=auto");
    expect(view.canRender).toBe(true);
    expect(view.cameras).toEqual(["front"]);
    expect(view.runtimeDetail).toBe("MuJoCo physics");
    expect(view.summary).not.toMatch(/substeps/);
  });

  test("a kinematic engine is never called physics, even if a label says MuJoCo", () => {
    const view = describeEngine(
      caps({
        active_backend: "kinematic",
        physics_active: false,
        adapters: {
          mujoco: {
            active: false,
            kind: "physics",
            detail: "MuJoCo physics of the vendored Menagerie SO-101",
            selection_reason: "SO101_SIM_ENGINE=kinematic",
            render_error: "not probed",
          },
        },
      }),
      stateWith(true),
    );
    expect(view.kind).toBe("kinematic");
    expect(view.summary).toBe(KINEMATIC_SUMMARY);
    // The reason and the render error, both verbatim.
    expect(view.why).toBe("SO101_SIM_ENGINE=kinematic not probed");
    expect(view.canRender).toBe(false);
  });

  test("render_available false names the problem and turns frames off", () => {
    const view = describeEngine(
      caps({
        physics_active: true,
        backend_kind: "physics",
        active_backend: "mujoco",
        adapters: { mujoco: { active: true, kind: "physics", detail: "", render_available: false, render_error: null } },
      }),
      null,
    );
    expect(view.canRender).toBe(false);
    expect(view.renderProblem).toBe("The server reports that it cannot render camera frames.");
  });

  test("render_available null means the GL probe is still running: checking, not unavailable", () => {
    const view = describeEngine(
      caps({
        physics_active: true,
        adapters: {
          mujoco: { active: true, kind: "physics", detail: "", render_available: null, render_status: "probing", cameras: [] },
        },
      }),
      null,
    );
    expect(view.renderChecking).toBe(true);
    expect(view.canRender).toBe(false);
    expect(view.renderProblem).toBeNull();
    expect(view.cameras).toEqual(["front"]);
  });

  test("render_status 'unavailable: <reason>' names the reason even without render_error", () => {
    const view = describeEngine(
      caps({
        physics_active: true,
        adapters: {
          mujoco: {
            active: true,
            kind: "physics",
            detail: "",
            render_available: false,
            render_error: null,
            render_status: "unavailable: the GL probe did not finish within 10 s",
          },
        },
      }),
      null,
    );
    expect(view.renderChecking).toBe(false);
    expect(view.canRender).toBe(false);
    expect(view.renderProblem).toBe("the GL probe did not finish within 10 s");
  });

  test("an older server without render fields lets the tile ask the render endpoint", () => {
    const view = describeEngine(
      caps({ physics_active: true, adapters: { mujoco: { active: true, kind: "physics", detail: "" } } }),
      null,
    );
    expect(view.renderChecking).toBe(false);
    expect(view.canRender).toBe(true);
  });

  test("an incomplete clock is left out rather than shown as NaN", () => {
    const view = describeEngine(
      caps({
        physics_active: true,
        adapters: {
          mujoco: { active: true, kind: "physics", detail: "", substeps_per_tick: 4, timestep_s: null, fixed_hz: 60 },
        },
      }),
      null,
    );
    expect(view.summary).toMatch(/servo profile are simulated\. The simulator/);
    expect(view.runtimeDetail).toBe("MuJoCo physics");
  });

  test("no engine falls back to the adapter error when there is no reason", () => {
    const view = describeEngine(
      caps({
        active_backend: "none",
        backend_kind: "none",
        adapters: { mujoco: { active: false, kind: "physics", detail: "", error: "MuJoCo failed to load" } },
      }),
      null,
    );
    expect(view.kind).toBe("none");
    expect(view.tone).toBe("danger");
    expect(view.why).toBe("MuJoCo failed to load");
  });

  test("without capabilities the state's flag decides; without either, nothing is claimed", () => {
    expect(describeEngine(null, stateWith(true)).kind).toBe("physics");
    expect(describeEngine(null, stateWith(false)).kind).toBe("kinematic");
    const unknown = describeEngine(null, null);
    expect(unknown.kind).toBe("unknown");
    expect(unknown.canRender).toBe(false);
  });
});

describe("preview wording", () => {
  test("known semantics in words; unknown ones as sent; none says nothing", () => {
    expect(previewSemantics("step_targets_like_play")).toMatch(/^Targets are applied tick by tick/);
    expect(previewSemantics("teleport")).toBe("Preview semantics: teleport");
    expect(previewSemantics("")).toBeNull();
    expect(previewSemantics(null)).toBeNull();
    expect(previewSemantics(undefined)).toBeNull();
  });

  test("tolerances keep two significant digits and never switch to exponent form", () => {
    expect(formatTolerance(0.5)).toBe("0.5");
    expect(formatTolerance(2)).toBe("2");
    expect(formatTolerance(0.123456)).toBe("0.12");
    expect(formatTolerance(1e-6)).toBe("0.000001");
  });

  test("reachSummary says reached, not reached, or rate-limited", () => {
    const base = { final_reached: false, physics: false } as SimulationTrajectoryPreview;
    expect(reachSummary(base)).toBe("rate-limited before target");
    expect(reachSummary({ ...base, physics: true, reached_tolerance: 0.5 })).toBe(
      "target not reached within ±0.5 by the end of the preview",
    );
    expect(reachSummary({ ...base, final_reached: true, reached_tolerance: 0.5 })).toBe("target reached within ±0.5");
    expect(reachSummary({ ...base, final_reached: true, reached_tolerance: null })).toBe("target reached");
    expect(reachSummary({ ...base, final_reached: true, reached_tolerance: Number.NaN })).toBe("target reached");
  });
});

describe("renderProbing", () => {
  const physics = (mujoco: Record<string, unknown>) =>
    caps({ physics_active: true, adapters: { mujoco: { active: true, kind: "physics", detail: "", ...mujoco } } });

  test("is true only while a physics server is still probing GL", () => {
    expect(renderProbing(physics({ render_available: null, render_status: "probing" }))).toBe(true);
    expect(renderProbing(physics({ render_available: null }))).toBe(true);
    expect(renderProbing(physics({ render_available: true, render_status: "available" }))).toBe(false);
    expect(renderProbing(physics({ render_available: false, render_status: "unavailable: no EGL" }))).toBe(false);
    expect(renderProbing(physics({}))).toBe(false);
    expect(renderProbing(caps({ adapters: { mujoco: { active: false, kind: "physics", detail: "", render_available: null } } }))).toBe(false);
    expect(renderProbing(null)).toBe(false);
  });
});

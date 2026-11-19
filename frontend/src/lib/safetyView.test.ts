import { describe, expect, test } from "vitest";
import type { ControlState, RobotStatus, Telemetry } from "../api";
import {
  collapsePercent,
  currentMode,
  deriveSafetyView,
  haltRemainingMs,
  STATUS_FRESH_MS,
  stopKindsOf,
  unsupervisedRemainingMs,
  unsupervisedText,
  type SafetySources,
} from "./safetyView";

const NOW = 1_000_000;

function status(overrides: Partial<RobotStatus> = {}): RobotStatus {
  return {
    mode: "idle",
    follower_connected: true,
    leader_connected: true,
    estopped: false,
    mock: true,
    cameras: [],
    live_cameras: [],
    camera_frame_age_ms: {},
    operation_owner: null,
    fps: 0,
    ...overrides,
  };
}

function frame(overrides: Partial<Telemetry> = {}): Telemetry {
  return {
    t: NOW / 1000,
    mode: "idle",
    estopped: false,
    fps: 0,
    observation: {},
    action: {},
    recording: false,
    recording_saving: false,
    episode_frames: 0,
    sample_age_ms: 0,
    sample_seq: 0,
    observation_live: false,
    ...overrides,
  };
}

function control(overrides: Partial<ControlState> = {}): ControlState {
  return {
    armed: false,
    safety_state: "disarmed",
    lease: null,
    watchdog_ms: 500,
    last_command_at: null,
    ...overrides,
  };
}

function sources(overrides: Partial<SafetySources> = {}): SafetySources {
  return {
    telemetry: null,
    transportStale: true,
    telemetryAt: null,
    status: status(),
    statusAt: NOW - 100,
    statusStale: false,
    control: control(),
    controlStale: false,
    now: NOW,
    ...overrides,
  };
}

describe("F4: E-stop precedence", () => {
  test("fresh telemetry decides the latch", () => {
    const view = deriveSafetyView(
      sources({ telemetry: frame({ estopped: true }), transportStale: false, telemetryAt: NOW }),
    );
    expect(view.tripped).toBe(true);
    expect(view.source).toBe("telemetry");
  });

  test("stale telemetry never outvotes fresh HTTP status", () => {
    const view = deriveSafetyView(
      sources({
        telemetry: frame({ estopped: true }),
        transportStale: true,
        telemetryAt: NOW - 10_000,
        status: status({ estopped: false }),
      }),
    );
    expect(view.tripped).toBe(false);
    expect(view.source).toBe("status");
  });

  test("status alone reports the latch while the socket is down", () => {
    const view = deriveSafetyView(sources({ status: status({ estopped: true }) }));
    expect(view.tripped).toBe(true);
    expect(view.chipLabel).toBe("E-stopped");
    expect(view.tone).toBe("danger");
  });

  test("no source at all is unknown, not safe", () => {
    const view = deriveSafetyView(sources({ status: null, statusAt: null, control: null }));
    expect(view.tripped).toBe(false);
    expect(view.source).toBe("none");
    expect(view.known).toBe(false);
    expect(view.chipLabel).toBe("state unknown");
    expect(view.tone).toBe("unknown");
  });

  test("status older than the freshness window is no longer known", () => {
    const view = deriveSafetyView(sources({ statusAt: NOW - STATUS_FRESH_MS - 1 }));
    expect(view.known).toBe(false);
    expect(view.chipLabel).toBe("state unknown");
  });

  test("a status poller that keeps failing is not trusted", () => {
    const view = deriveSafetyView(sources({ statusStale: true }));
    expect(view.known).toBe(false);
  });
});

describe("chip label precedence", () => {
  test("E-stop beats torque-unconfirmed beats halted", () => {
    const tripped = deriveSafetyView(
      sources({
        status: status({ estopped: true, torque_unconfirmed: true, stop_state: "halted" }),
      }),
    );
    expect(tripped.chipLabel).toBe("E-stopped");

    const torque = deriveSafetyView(
      sources({ status: status({ torque_unconfirmed: true, stop_state: "halted" }) }),
    );
    expect(torque.chipLabel).toBe("torque unconfirmed");
    expect(torque.torqueUnconfirmed).toBe(true);
    expect(torque.tone).toBe("danger");
  });

  test("torque_state 'unconfirmed' alone also raises the C2 state", () => {
    const view = deriveSafetyView(sources({ status: status({ torque_state: "unconfirmed" }) }));
    expect(view.torqueUnconfirmed).toBe(true);
    expect(view.torqueState).toBe("unconfirmed");
  });

  test("collapsing and halted are warnings", () => {
    expect(deriveSafetyView(sources({ status: status({ stop_state: "collapsing" }) })).chipLabel).toBe(
      "collapsing",
    );
    const halted = deriveSafetyView(sources({ status: status({ stop_state: "halted" }) }));
    expect(halted.chipLabel).toBe("halted");
    expect(halted.tone).toBe("warn");
  });

  test("unsafe control states are danger; armed is a warning; disarmed is neutral", () => {
    expect(deriveSafetyView(sources({ control: control({ safety_state: "watchdog_timeout" }) }))).toMatchObject({
      chipLabel: "watchdog timeout",
      tone: "danger",
    });
    expect(deriveSafetyView(sources({ control: control({ armed: true }) }))).toMatchObject({
      chipLabel: "armed",
      tone: "warn",
      armed: true,
    });
    expect(deriveSafetyView(sources())).toMatchObject({ chipLabel: "disarmed", tone: "neutral" });
  });

  test("stale control state is shown as unknown, never as disarmed", () => {
    const view = deriveSafetyView(sources({ control: control({ armed: true }), controlStale: true }));
    expect(view.chipLabel).toBe("control unknown");
    expect(view.armed).toBe(false);
    expect(view.safetyState).toBe("unknown");
  });
});

describe("C11 stop kinds", () => {
  test("absent stop_kinds means E-stop only", () => {
    expect(stopKindsOf(status())).toEqual(["estop"]);
    expect(stopKindsOf(null)).toEqual(["estop"]);
    const view = deriveSafetyView(sources());
    expect(view.canHalt).toBe(false);
    expect(view.canCollapse).toBe(false);
  });

  test("advertised kinds are kept in a fixed order and always include estop", () => {
    expect(stopKindsOf(status({ stop_kinds: ["collapse", "halt", "estop"] }))).toEqual([
      "estop",
      "halt",
      "collapse",
    ]);
    expect(stopKindsOf(status({ stop_kinds: ["halt"] }))).toEqual(["estop", "halt"]);
    expect(stopKindsOf(status({ stop_kinds: ["warp-drive", "halt"] }))).toEqual(["estop", "halt"]);
  });

  test("telemetry carries the stop fields when present, else status does", () => {
    const halt = {
      since: "2030-01-01T00:00:00Z",
      reason: "deadman released",
      hold_expires_at: "2030-01-01T00:01:00Z",
      server_time: "2030-01-01T00:00:00Z",
    };
    const fromStatus = deriveSafetyView(
      sources({
        telemetry: frame(),
        transportStale: false,
        telemetryAt: NOW - 50,
        status: status({ stop_state: "halted", halt }),
        statusAt: NOW - 400,
      }),
    );
    expect(fromStatus.stopState).toBe("halted");
    expect(fromStatus.halt).toEqual({ info: halt, receivedAt: NOW - 400 });

    const fromTelemetry = deriveSafetyView(
      sources({
        telemetry: frame({ stop_state: "halted", halt }),
        transportStale: false,
        telemetryAt: NOW - 50,
        status: status({ stop_state: "none", halt: null }),
      }),
    );
    expect(fromTelemetry.stopState).toBe("halted");
    expect(fromTelemetry.halt?.receivedAt).toBe(NOW - 50);
  });

  test("unknown stop_state strings are kept verbatim", () => {
    expect(deriveSafetyView(sources({ status: status({ stop_state: "parking" }) })).stopState).toBe(
      "parking",
    );
  });
});

describe("unsupervised halt countdown", () => {
  // Halted at 12:00:00 with a 60 s hold timer; the status arrived at client
  // time 5 s, saying the server clock read 12:00:10.
  const halt = {
    info: {
      since: "2030-05-01T12:00:00Z",
      reason: "deadman released",
      hold_expires_at: "2030-05-01T12:01:00Z",
      server_time: "2030-05-01T12:00:10Z",
    },
    receivedAt: 5_000,
  };
  const lost = (lost_at: string | null, grace_s = 5) => ({ supervised: false, lost_at, grace_s });

  test("counts the grace period from lost_at on the server clock", () => {
    // Lost at 12:00:08 with 5 s grace: collapses at 12:00:13, 3 s after receipt.
    expect(unsupervisedRemainingMs(halt, lost("2030-05-01T12:00:08Z"), 5_000)).toBe(3_000);
    expect(unsupervisedRemainingMs(halt, lost("2030-05-01T12:00:08Z"), 6_500)).toBe(1_500);
    expect(unsupervisedRemainingMs(halt, lost("2030-05-01T12:00:08Z"), 60_000)).toBe(0);
    // A client clock that jumps backwards never adds time.
    expect(unsupervisedRemainingMs(halt, lost("2030-05-01T12:00:08Z"), 1_000)).toBe(3_000);
  });

  test("counts from the halt when supervision was lost before it, or never existed", () => {
    // Grace from 12:00:00 (the halt), 20 s: collapses at 12:00:20.
    expect(unsupervisedRemainingMs(halt, lost("2030-05-01T11:00:00Z", 20), 5_000)).toBe(10_000);
    expect(unsupervisedRemainingMs(halt, lost(null, 20), 5_000)).toBe(10_000);
  });

  test("never runs past the hold timer", () => {
    expect(unsupervisedRemainingMs(halt, lost("2030-05-01T12:00:08Z", 600), 5_000)).toBe(50_000);
  });

  test("supervised, or a server that does not report supervision: no unsupervised countdown", () => {
    expect(unsupervisedRemainingMs(halt, { supervised: true, lost_at: null, grace_s: 5 }, 5_000)).toBeNull();
    expect(unsupervisedRemainingMs(halt, null, 5_000)).toBeNull();
    expect(unsupervisedRemainingMs(halt, undefined, 5_000)).toBeNull();
  });

  test("an unreadable server time falls back to the hold countdown", () => {
    const broken = { ...halt, info: { ...halt.info, server_time: "garbage" } };
    expect(unsupervisedRemainingMs(broken, lost("2030-05-01T12:00:08Z"), 5_000)).toBeNull();
  });

  test("the text rounds up, so it reaches 0 only at the end", () => {
    expect(unsupervisedText(3_000)).toBe("Unsupervised: collapses in 3 s");
    expect(unsupervisedText(2_100)).toBe("Unsupervised: collapses in 3 s");
    expect(unsupervisedText(0)).toBe("Unsupervised: collapses in 0 s");
  });

  test("the view carries supervision from status, and null from an older server", () => {
    const base: SafetySources = {
      telemetry: null,
      transportStale: true,
      telemetryAt: null,
      status: null,
      statusAt: 1_000,
      statusStale: false,
      control: null,
      controlStale: false,
      now: 1_000,
    };
    const status = { estopped: false, stop_kinds: ["estop", "halt"] } as RobotStatus;
    expect(deriveSafetyView({ ...base, status }).supervision).toBeNull();
    const supervision = lost("2030-05-01T12:00:08Z");
    expect(deriveSafetyView({ ...base, status: { ...status, supervision } }).supervision).toEqual(supervision);
  });
});

describe("halt countdown", () => {
  test("uses the server clock, whatever the client clock says", () => {
    // Server time is in 2030 while the client clock is near the epoch: only
    // the server-side difference and client-side elapsed time matter.
    const halt = {
      info: {
        since: "2030-05-01T12:00:00Z",
        reason: "deadman released",
        hold_expires_at: "2030-05-01T12:00:42Z",
        server_time: "2030-05-01T12:00:00Z",
      },
      receivedAt: 5_000,
    };
    expect(haltRemainingMs(halt, 5_000)).toBe(42_000);
    expect(haltRemainingMs(halt, 15_000)).toBe(32_000);
    expect(haltRemainingMs(halt, 60_000)).toBe(0);
    // A client clock that jumps backwards never adds time.
    expect(haltRemainingMs(halt, 1_000)).toBe(42_000);
  });

  test("unparseable times give an unknown countdown", () => {
    expect(
      haltRemainingMs(
        {
          info: { since: "", reason: "", hold_expires_at: "soon", server_time: "now" },
          receivedAt: 0,
        },
        0,
      ),
    ).toBeNull();
  });
});

describe("collapse progress", () => {
  const base = { since: "", reason: "", eta_s: 3 };
  test("0..1 fractions become percentages", () => {
    expect(collapsePercent({ ...base, progress: 0.25 })).toBe(25);
    expect(collapsePercent({ ...base, progress: 1 })).toBe(100);
  });
  test("values above 1 are read as percentages and clamped", () => {
    expect(collapsePercent({ ...base, progress: 40 })).toBe(40);
    expect(collapsePercent({ ...base, progress: 140 })).toBe(100);
  });
  test("missing progress is unknown", () => {
    expect(collapsePercent({ ...base, progress: Number.NaN })).toBeNull();
  });
});

describe("F21: current mode", () => {
  test("a live frame decides, else fresh status, else unknown", () => {
    const live = sources({ telemetry: frame({ mode: "teleop" }), transportStale: false, telemetryAt: NOW });
    expect(currentMode(live.telemetry, true, deriveSafetyView(live), status({ mode: "idle" }))).toBe("teleop");

    // Link down, status fresh: status speaks.
    const fresh = sources({ telemetry: frame({ mode: "teleop" }), status: status({ mode: "recording" }) });
    expect(currentMode(fresh.telemetry, false, deriveSafetyView(fresh), fresh.status)).toBe("recording");

    // Link down and the status poll failing or old: nobody knows.
    const failing = sources({ telemetry: frame({ mode: "teleop" }), statusStale: true });
    expect(currentMode(failing.telemetry, false, deriveSafetyView(failing), failing.status)).toBeUndefined();
    const old = sources({ statusAt: NOW - STATUS_FRESH_MS - 1 });
    expect(currentMode(null, false, deriveSafetyView(old), old.status)).toBeUndefined();
    expect(currentMode(null, false, deriveSafetyView(sources({ status: null })), null)).toBeUndefined();
  });
});

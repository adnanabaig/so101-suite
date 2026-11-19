import { describe, expect, test } from "vitest";
import {
  ApiError,
  type CalibrationStatus,
  type CapabilitiesResponse,
  type PreflightCheckResult,
  type PreflightReport,
} from "../api";
import {
  checkLines,
  isComplete,
  MAX_CHECK_LINES,
  setupItems,
  type Fetched,
  type SetupInputs,
  type SetupItem,
  type SetupItemId,
} from "./setupChecklist";

// The 404 detail GET /api/preflight/latest sends when nothing is saved.
const NO_REPORT_DETAIL = "no saved preflight report for the real backend; run POST /api/preflight or `so101 preflight`";

function capabilities(overrides: Partial<CapabilitiesResponse> = {}): CapabilitiesResponse {
  return {
    system_version: "0.2.0",
    mock: false,
    capabilities: [
      { id: "control", label: "Safe manual command API", available: true, maturity: "stable", detail: "API contract only." },
    ],
    ...overrides,
  };
}

function calibration(overrides: Partial<CalibrationStatus> = {}): CalibrationStatus {
  return {
    arm: "follower",
    in_progress: false,
    fraction: 0,
    message: "",
    has_saved: false,
    device_id: "",
    created_at: null,
    ranges: {},
    ...overrides,
  };
}

function check(overrides: Partial<PreflightCheckResult> = {}): PreflightCheckResult {
  return {
    id: "motors",
    key: "motors.follower",
    arm: "follower",
    title: "Motors",
    status: "pass",
    message: "6 of 6 motors answered",
    remediation: "",
    measured: {},
    blocking: true,
    rehearsal: false,
    source: "real",
    references: [],
    duration_ms: 12,
    ...overrides,
  };
}

function report(overrides: Partial<PreflightReport> = {}): PreflightReport {
  return {
    kind: "so101.preflight",
    schema_version: 1,
    created_at: "2026-10-01T10:00:00+00:00",
    so101_version: "0.2.0",
    origin: "cli",
    backend: "real",
    rehearsal: false,
    arms: ["follower", "leader"],
    bus_access: "opened",
    bus_access_detail: "",
    overall: "pass",
    ready_for_hardware: true,
    exit_code: 0,
    counts: { pass: 1, warn: 0, fail: 0, skip: 0 },
    checks: [check()],
    notes: [],
    ...overrides,
  };
}

const ok = <T>(data: T): Fetched<T> => ({ data, error: null });
const failed = <T>(error: Error): Fetched<T> => ({ data: null, error });
const pending = <T>(): Fetched<T> => ({ data: null, error: null });
const missingRoute = () => new ApiError(404, "Not Found", '{"detail":"Not Found"}');

// A server before v0.2: mock=false read as a real arm, not reported.
const INFERRED_REAL = { kind: "real", reported: false } as const;
const MOCK = { kind: "mock", reported: true } as const;
const SIM = { kind: "sim", reported: true } as const;
const REAL = { kind: "real", reported: true } as const;

function inputs(overrides: Partial<SetupInputs> = {}): SetupInputs {
  return {
    capabilities: ok(capabilities()),
    backend: INFERRED_REAL,
    follower: ok(calibration({ has_saved: true, device_id: "so101_follower", created_at: 1_790_000_000 })),
    leader: ok(calibration({ arm: "leader", has_saved: true, device_id: "so101_leader", created_at: 1_790_000_000 })),
    preflight: ok(report()),
    stopsAcknowledged: true,
    canHalt: true,
    canCollapse: true,
    ...overrides,
  };
}

function item(overrides: Partial<SetupInputs>, id: SetupItemId): SetupItem {
  const found = setupItems(inputs(overrides)).find((candidate) => candidate.id === id);
  if (!found) throw new Error(`no ${id} item`);
  return found;
}

describe("setupItems", () => {
  test("a ready real workstation: every item complete, in working order", () => {
    const items = setupItems(inputs());
    expect(items.map((entry) => entry.id)).toEqual([
      "backend",
      "calibration.follower",
      "calibration.leader",
      "preflight",
      "stops",
    ]);
    expect(items.every((entry) => isComplete(entry, INFERRED_REAL))).toBe(true);
    expect(items.filter((entry) => entry.required).map((entry) => entry.id)).toEqual([
      "backend",
      "calibration.follower",
      "preflight",
      "stops",
    ]);
  });

  describe("robot backend", () => {
    test("mock is a rehearsal backend and says so", () => {
      const backend = item({ capabilities: ok(capabilities({ mock: true })), backend: MOCK }, "backend");
      expect(backend.state).toBe("done");
      expect(backend.summary).toMatch(/^Mock rehearsal: no real motors move/);
    });

    test("sim says the arm is simulated; real says the motors move", () => {
      const sim = item({ capabilities: ok(capabilities({ backend: "sim" })), backend: SIM }, "backend");
      expect(sim.state).toBe("done");
      expect(sim.summary).toMatch(/^MuJoCo simulation: commands move a simulated arm, not the physical one\./);
      const real = item({ capabilities: ok(capabilities({ backend: "real" })), backend: REAL }, "backend");
      expect(real.summary).toBe("A real arm: commands move the physical motors.");
    });

    test("an older server without backend: not mock is real or simulated, and it cannot say which", () => {
      const noReport = item({ preflight: failed(new ApiError(404, NO_REPORT_DETAIL, "")) }, "backend");
      expect(noReport.state).toBe("done");
      expect(noReport.summary).toMatch(/real or simulated arm; the server does not say which\. A preflight report names/);
      const sim = item({ preflight: ok(report({ backend: "sim", rehearsal: true, ready_for_hardware: false })) }, "backend");
      expect(sim.summary).toMatch(/The last preflight ran on the sim backend\.$/);
    });

    test("without the robot software it is a step to do, with the server's note verbatim", () => {
      const detail = "Control protocol is online; install LeRobot before real arming.";
      const backend = item(
        {
          capabilities: ok(
            capabilities({
              capabilities: [{ id: "control", label: "Safe manual command API", available: false, maturity: "stable", detail }],
            }),
          ),
        },
        "backend",
      );
      expect(backend.state).toBe("todo");
      expect(backend.details).toEqual([detail]);
    });

    test("checking, unreadable, and a server without a control capability", () => {
      expect(item({ capabilities: pending() }, "backend").state).toBe("checking");
      const unreadable = item({ capabilities: failed(new ApiError(500, "boom", "")) }, "backend");
      expect(unreadable.state).toBe("unknown");
      expect(unreadable.details).toEqual(["boom"]);
      expect(item({ capabilities: ok(capabilities({ capabilities: [] })) }, "backend").state).toBe("unknown");
    });
  });

  describe("calibration", () => {
    test("a saved calibration names the device and when it was made", () => {
      const follower = item({}, "calibration.follower");
      expect(follower.state).toBe("done");
      expect(follower.summary).toBe("Saved for so101_follower.");
      expect(follower.when).toEqual({ label: "Calibrated", iso: new Date(1_790_000_000_000).toISOString() });
      const legacy = item({ follower: ok(calibration({ has_saved: true })) }, "calibration.follower");
      expect(legacy.summary).toBe("Saved.");
      expect(legacy.when).toBeUndefined();
    });

    test("no saved file: the follower is a step to do and the leader is optional", () => {
      const follower = item({ follower: ok(calibration()) }, "calibration.follower");
      expect(follower.state).toBe("todo");
      expect(follower.required).toBe(true);
      expect(follower.link).toEqual({ to: "/calibration", label: "Open Calibration" });
      expect(isComplete(follower, INFERRED_REAL)).toBe(false);
      const leader = item({ leader: ok(calibration({ arm: "leader" })) }, "calibration.leader");
      expect(leader.state).toBe("optional");
      expect(leader.required).toBe(false);
      expect(leader.summary).toMatch(/Needed only to teleoperate with the leader arm/);
    });

    test("on mock, defaults stand in; until the backend is known, it is still checking", () => {
      const onMock = item({ backend: MOCK, follower: ok(calibration()) }, "calibration.follower");
      expect(onMock.state).toBe("done");
      expect(onMock.summary).toMatch(/mock backend uses default ranges/);
      expect(item({ backend: null, follower: ok(calibration()) }, "calibration.follower").state).toBe("checking");
    });

    test("on a real arm the server's reason for no calibration is shown, e.g. a mock run's file", () => {
      const message =
        "the saved calibration was made on the mock backend, not on this arm, so it counts as missing; recalibrate on the real arm";
      const follower = item({ backend: REAL, follower: ok(calibration({ message })) }, "calibration.follower");
      expect(follower.state).toBe("todo");
      expect(follower.details).toEqual([message]);
      const leader = item({ backend: REAL, leader: ok(calibration({ arm: "leader", message })) }, "calibration.leader");
      expect(leader.state).toBe("optional");
      expect(leader.details).toEqual([message]);
    });

    test("in sim there is nothing to calibrate, as the server says; a reported real arm still needs it", () => {
      const onSim = item({ backend: SIM, follower: ok(calibration()) }, "calibration.follower");
      expect(onSim.state).toBe("done");
      expect(onSim.summary).toMatch(/^Not needed: the simulated arm has no encoders to calibrate\./);
      expect(item({ backend: REAL, follower: ok(calibration()) }, "calibration.follower").state).toBe("todo");
    });

    test("an unreadable saved file is a problem, even on mock, with the server's message verbatim", () => {
      const message = "invalid saved calibration: Expecting value: line 1 column 1 (char 0)";
      for (const backend of [INFERRED_REAL, MOCK, SIM]) {
        const follower = item({ backend, follower: ok(calibration({ message })) }, "calibration.follower");
        expect(follower.state).toBe("problem");
        expect(follower.details).toEqual([message]);
        expect(follower.link?.to).toBe("/calibration");
      }
    });

    test("a calibration in progress shows its progress", () => {
      const follower = item(
        { follower: ok(calibration({ in_progress: true, fraction: 0.426, message: "Move every joint through its range" })) },
        "calibration.follower",
      );
      expect(follower.state).toBe("running");
      expect(follower.summary).toBe("Calibrating now: 43%.");
      expect(follower.details).toEqual(["Move every joint through its range"]);
    });

    test("checking, unreadable, and a server without the status route", () => {
      expect(item({ follower: pending() }, "calibration.follower").state).toBe("checking");
      const unreadable = item({ follower: failed(new ApiError(408, "GET /api/calibration/status/follower timed out", "")) }, "calibration.follower");
      expect(unreadable.state).toBe("unknown");
      expect(unreadable.details).toEqual(["GET /api/calibration/status/follower timed out"]);
      expect(item({ leader: failed(missingRoute()) }, "calibration.leader").state).toBe("unavailable");
    });
  });

  describe("preflight (the wave-2 slot)", () => {
    test("no saved report: a step to do, with the server's 404 detail exactly as sent", () => {
      const preflight = item({ preflight: failed(new ApiError(404, NO_REPORT_DETAIL, "")) }, "preflight");
      expect(preflight.state).toBe("todo");
      expect(preflight.summary).toBe("No preflight report yet. Run `so101 preflight` in a terminal on the workstation.");
      expect(preflight.details).toEqual([NO_REPORT_DETAIL]);
    });

    test("a server without the preflight API, a failed read, and the first read", () => {
      const absent = item({ preflight: failed(missingRoute()) }, "preflight");
      expect(absent.state).toBe("unavailable");
      expect(absent.summary).toMatch(/no preflight API/);
      expect(isComplete(absent, INFERRED_REAL)).toBe(false);
      const broken = item({ preflight: failed(new ApiError(500, "Internal Server Error", "")) }, "preflight");
      expect(broken.state).toBe("unknown");
      expect(broken.details).toEqual(["Internal Server Error"]);
      expect(item({ preflight: pending() }, "preflight").state).toBe("checking");
    });

    test("a real pass is done; with warnings it lists them", () => {
      const pass = item({}, "preflight");
      expect(pass.state).toBe("done");
      expect(pass.summary).toBe("Passed on the real backend.");
      expect(pass.when).toEqual({ label: "Ran with so101 preflight", iso: "2026-10-01T10:00:00+00:00" });
      const warned = item(
        {
          preflight: ok(
            report({
              origin: "api",
              overall: "warn",
              checks: [check(), check({ id: "storage", key: "storage", arm: null, title: "Storage", status: "warn", message: "1.2 GiB free", remediation: "Free space before recording" })],
            }),
          ),
        },
        "preflight",
      );
      expect(warned.state).toBe("done");
      expect(warned.summary).toBe("Passed with warnings on the real backend.");
      expect(warned.details).toEqual(["Storage: 1.2 GiB free. Free space before recording"]);
      expect(warned.when?.label).toBe("Ran on the server");
    });

    test("on a mock or sim server a rehearsal pass is complete, and never called hardware evidence", () => {
      for (const [kind, server] of [
        ["mock", MOCK],
        ["sim", SIM],
      ] as const) {
        const rehearsal = item(
          {
            backend: server,
            preflight: ok(report({ backend: kind, rehearsal: true, ready_for_hardware: false, bus_access: "synthetic" })),
          },
          "preflight",
        );
        expect(rehearsal.state).toBe("rehearsal");
        expect(isComplete(rehearsal, server)).toBe(true);
        expect(rehearsal.summary).toBe(`Rehearsal passed on the ${kind} backend. A rehearsal is not evidence about a real arm.`);
      }
    });

    test("on a real server a rehearsal is a step still to do, never complete", () => {
      for (const server of [REAL, INFERRED_REAL]) {
        for (const kind of ["mock", "sim"] as const) {
          const preflight = item(
            {
              backend: server,
              preflight: ok(report({ backend: kind, rehearsal: true, ready_for_hardware: false, bus_access: "synthetic" })),
            },
            "preflight",
          );
          expect(preflight.state).toBe("todo");
          expect(isComplete(preflight, server)).toBe(false);
          expect(preflight.summary).toBe(
            `The last preflight was a ${kind} rehearsal, which is not evidence about this arm. Run \`so101 preflight\` on the real backend.`,
          );
        }
      }
      const unknown = item({ backend: null, preflight: ok(report({ backend: "sim", rehearsal: true, ready_for_hardware: false })) }, "preflight");
      expect(unknown.state).toBe("checking");
    });

    test("isComplete counts a rehearsal only on a mock or sim server", () => {
      const rehearsal: SetupItem = { id: "preflight", title: "Preflight", state: "rehearsal", summary: "", details: [], required: true };
      expect(isComplete(rehearsal, MOCK)).toBe(true);
      expect(isComplete(rehearsal, SIM)).toBe(true);
      expect(isComplete(rehearsal, REAL)).toBe(false);
      expect(isComplete(rehearsal, INFERRED_REAL)).toBe(false);
      expect(isComplete(rehearsal, null)).toBe(false);
    });

    test("a failure is a problem that lists the failed checks with their remediation", () => {
      const preflight = item(
        {
          preflight: ok(
            report({
              overall: "fail",
              ready_for_hardware: false,
              exit_code: 1,
              checks: [
                check(),
                check({ status: "fail", message: "motor 3 did not answer", remediation: "Check the daisy chain" }),
                check({ id: "power", key: "power", arm: null, title: "Supply voltage", status: "fail", message: "6.1 V is below 6.5 V." }),
              ],
            }),
          ),
        },
        "preflight",
      );
      expect(preflight.state).toBe("problem");
      expect(preflight.summary).toBe("Preflight failed.");
      expect(preflight.details).toEqual([
        "Motors (follower): motor 3 did not answer. Check the daisy chain",
        "Supply voltage: 6.1 V is below 6.5 V.",
      ]);
      const rehearsalFail = item(
        { backend: MOCK, preflight: ok(report({ backend: "mock", rehearsal: true, overall: "fail", ready_for_hardware: false })) },
        "preflight",
      );
      expect(rehearsalFail.summary).toBe("The mock rehearsal failed.");
    });

    test("incomplete lists only the blocking checks that were skipped", () => {
      const preflight = item(
        {
          preflight: ok(
            report({
              origin: "api",
              overall: "incomplete",
              ready_for_hardware: false,
              exit_code: 3,
              bus_access: "deferred",
              checks: [
                check({ status: "skip", message: "the server holds the bus" }),
                check({ id: "cameras", key: "cameras", arm: null, title: "Cameras", status: "skip", blocking: false, message: "none configured" }),
              ],
            }),
          ),
        },
        "preflight",
      );
      expect(preflight.state).toBe("todo");
      expect(preflight.summary).toBe("Preflight is incomplete: a required check was skipped.");
      expect(preflight.details).toEqual(["Motors (follower): the server holds the bus"]);
    });
  });

  test("check lines are capped, and the rest are counted", () => {
    const checks = Array.from({ length: MAX_CHECK_LINES + 2 }, (_, index) =>
      check({ key: `c${index}`, arm: null, title: `Check ${index}`, message: "failed" }),
    );
    const lines = checkLines(checks);
    expect(lines).toHaveLength(MAX_CHECK_LINES + 1);
    expect(lines[0]).toBe("Check 0: failed");
    expect(lines[lines.length - 1]).toBe("and 2 more.");
    expect(checkLines(checks.slice(0, MAX_CHECK_LINES))).toHaveLength(MAX_CHECK_LINES);
  });

  describe("know your stops", () => {
    test("is the one thing the operator confirms, and describes this server's stops", () => {
      const unconfirmed = item({ stopsAcknowledged: false }, "stops");
      expect(unconfirmed.state).toBe("todo");
      expect(unconfirmed.summary).toBe(
        "Escape is E-stop on every page, even in a field or a dialog: motor power is cut. " +
          "H is Halt: the arm freezes and the motors hold. " +
          "Collapse has no key: its button lowers the arm to rest, then turns the motors off.",
      );
      expect(item({}, "stops").state).toBe("done");
      const older = item({ canHalt: false, canCollapse: false }, "stops");
      expect(older.summary).toBe(
        "Escape is E-stop on every page, even in a field or a dialog: motor power is cut. This server has no Halt, so H does nothing.",
      );
    });
  });
});

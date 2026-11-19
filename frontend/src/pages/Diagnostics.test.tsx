import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { AuditEvent, DiagnosticsResponse } from "../api";
import { SafetyProvider } from "../components/safety/SafetyProvider";
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
import Diagnostics, { buildDiagnosticExport, mergeEvents } from "./Diagnostics";

describe("diagnostic export redaction", () => {
  test("uses an allowlist and removes URLs, credentials, paths, and free-form event text", () => {
    const diagnostics: DiagnosticsResponse = {
      generated_at: "2026-10-08T12:00:00Z",
      overall: "warn",
      checks: [
        {
          id: "plain-check-id-secret",
          label:
            "plain-label-secret rtsp://operator:path-pass@camera.local/private/path-token?token=query-token#fragment-token",
          status: "warn",
          detail: "Authorization=detail-secret",
          remediation: "open https://user:remediation-pass@example.test/private/remediation-token?key=query-secret",
        },
      ],
      ports: [
        {
          path: "/dev/serial/by-id/operator-private-id",
          detail: "password=port-secret",
          token: "extension-secret",
        },
      ],
      events: [],
    };
    const events: AuditEvent[] = [
      {
        timestamp: "2026-10-08T12:01:00Z",
        level: "warn",
        type: "plain-type-secret-with-no-delimiter",
        source: "plain-source-secret-with-no-delimiter",
        message: "Bearer message-secret",
        detail: { api_key: "detail-api-secret" },
        credential: "event-extension-secret",
      },
    ];

    const exported = buildDiagnosticExport(
      diagnostics,
      events,
      {
        state: "open",
        transport_stale: false,
        sample_state: "live",
        last_message_at: 123,
      },
      true,
    );
    const serialized = JSON.stringify(exported);

    for (const secret of [
      "path-pass",
      "plain-check-id-secret",
      "plain-label-secret",
      "path-token",
      "query-token",
      "fragment-token",
      "detail-secret",
      "remediation-pass",
      "remediation-token",
      "query-secret",
      "operator-private-id",
      "port-secret",
      "extension-secret",
      "event-pass",
      "event-token",
      "plain-type-secret-with-no-delimiter",
      "plain-source-secret-with-no-delimiter",
      "message-secret",
      "detail-api-secret",
    ]) {
      expect(serialized).not.toContain(secret);
    }
    expect(exported.page).toBe("diagnostics");
    expect(exported.redaction_policy).toBe("support-allowlist-v1");
    expect(exported.diagnostics.port_count).toBe(1);
    expect(exported.diagnostics.check_status_counts).toEqual({
      ok: 0,
      warn: 1,
      danger: 0,
      neutral: 0,
    });
    expect(exported.events[0]).toEqual({
      timestamp: "2026-10-08T12:01:00.000Z",
    });
  });
});

describe("mergeEvents (F24)", () => {
  test("drops duplicates across sources and sorts newest first", () => {
    const a: AuditEvent = { id: 1, timestamp: "2026-10-01T10:00:00Z", message: "armed" };
    const b: AuditEvent = { id: 2, timestamp: "2026-10-01T10:05:00Z", message: "estop" };
    const c: AuditEvent = { timestamp: "2026-10-01T09:00:00Z", source: "camera", message: "wrist lost" };
    const merged = mergeEvents([a, c], [b, a, { ...c }]);
    expect(merged.map((event) => event.message)).toEqual(["estop", "armed", "wrist lost"]);
  });

  test("events without a parseable time go last, in arrival order", () => {
    const merged = mergeEvents([
      { id: "x", message: "no time" },
      { id: "y", timestamp: "garbage", message: "bad time" },
      { id: "z", created_at: "2026-10-01T08:00:00Z", message: "created" },
    ]);
    expect(merged.map((event) => event.message)).toEqual(["created", "no time", "bad time"]);
  });
});

let backend: FakeBackend;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
  backend.route("GET", "/api/diagnostics", () =>
    jsonResponse({ overall: "pass", generated_at: new Date().toISOString(), checks: [], ports: [], events: [] }),
  );
  backend.route("GET", "/api/events", () => jsonResponse({ events: [] }));
  backend.route("POST", "/api/robot/fault/reset", () => jsonResponse({ ok: true, detail: "cleared" }));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderDiagnostics() {
  render(
    <TelemetryProvider>
      <SafetyProvider>
        <Diagnostics />
      </SafetyProvider>
    </TelemetryProvider>,
  );
  await advance(10);
  act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
  await advance(10);
}

/** The server's 409 when its torque-off retry still cannot confirm the motors are off. */
const RETRY_UNCONFIRMED_DETAIL =
  "torque-off is still unconfirmed after a retry; cut physical power to the arm, then reset the fault with power_cut_acknowledged: true";

describe("Diagnostics fault reset (C2)", () => {
  test("with torque-off unconfirmed, the first reset asks the server to retry, without the acknowledgement", async () => {
    backend.torqueUnconfirmed = true;
    await renderDiagnostics();
    expect(screen.getByText("Torque-off is unconfirmed")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Clear remediated fault/ }));
    await advance(10);
    const dialog = screen.getByRole("alertdialog", { name: "Retry torque-off and clear the fault?" });
    // Nobody ticks "I cut power" before the server has even retried.
    expect(within(dialog).queryByRole("checkbox")).toBeNull();
    fireEvent.click(within(dialog).getByRole("button", { name: "Retry and clear" }));
    await advance(10);
    expect(backend.callsTo("/api/robot/fault/reset").map((call) => call.body)).toEqual([
      { operator_id: expect.any(String), confirm: true },
    ]);
  });

  test("a 409 after the retry shows the server's detail and offers the acknowledge path", async () => {
    backend.torqueUnconfirmed = true;
    backend.route("POST", "/api/robot/fault/reset", (call) =>
      (call.body as { power_cut_acknowledged?: boolean }).power_cut_acknowledged === true
        ? jsonResponse({ ok: true, detail: "runtime fault cleared" })
        : jsonResponse({ detail: RETRY_UNCONFIRMED_DETAIL }, 409),
    );
    await renderDiagnostics();
    fireEvent.click(screen.getByRole("button", { name: /Clear remediated fault/ }));
    await advance(10);
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Retry and clear" }));
    await advance(10);

    expect(screen.getByText("Fault reset rejected")).toBeInTheDocument();
    expect(screen.getByText(RETRY_UNCONFIRMED_DETAIL)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "I have cut physical power…" }));
    await advance(10);
    const dialog = screen.getByRole("alertdialog", { name: "Clear the fault with torque-off unconfirmed?" });
    const clear = within(dialog).getByRole("button", { name: "Clear fault" });
    expect(clear).toBeDisabled();
    fireEvent.click(within(dialog).getByRole("checkbox", { name: "I have cut physical power to the arm" }));
    fireEvent.click(clear);
    await advance(10);

    expect(backend.callsTo("/api/robot/fault/reset").map((call) => call.body)).toEqual([
      { operator_id: expect.any(String), confirm: true },
      { operator_id: expect.any(String), confirm: true, power_cut_acknowledged: true },
    ]);
    expect(screen.queryByText("Fault reset rejected")).toBeNull();
    expect(screen.queryByRole("button", { name: "I have cut physical power…" })).toBeNull();
  });

  test("a 409 for another reason shows the detail, with no acknowledge path", async () => {
    backend.safetyState = "fault";
    backend.route("POST", "/api/robot/fault/reset", () =>
      jsonResponse({ detail: "cannot clear a fault while a control loop is active" }, 409),
    );
    await renderDiagnostics();
    fireEvent.click(screen.getByRole("button", { name: /Clear remediated fault/ }));
    await advance(10);
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Clear fault" }));
    await advance(10);
    expect(screen.getByText("Fault reset rejected")).toBeInTheDocument();
    expect(screen.getByText("cannot clear a fault while a control loop is active")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "I have cut physical power…" })).toBeNull();
  });

  test("a plain latched fault resets without the acknowledgement field", async () => {
    backend.safetyState = "fault";
    await renderDiagnostics();
    fireEvent.click(screen.getByRole("button", { name: /Clear remediated fault/ }));
    await advance(10);
    const dialog = screen.getByRole("alertdialog", { name: "Clear the remediated fault?" });
    expect(within(dialog).queryByRole("checkbox")).toBeNull();
    fireEvent.click(within(dialog).getByRole("button", { name: "Clear fault" }));
    await advance(10);
    expect(backend.callsTo("/api/robot/fault/reset")[0].body).toEqual({
      operator_id: expect.any(String),
      confirm: true,
    });
  });

  test("no fault, no reset button", async () => {
    await renderDiagnostics();
    expect(screen.queryByRole("button", { name: /Clear remediated fault/ })).toBeNull();
  });
});

describe("Diagnostics timeline", () => {
  test("shows merged events newest first and says so honestly when there are none", async () => {
    backend.route("GET", "/api/diagnostics", () =>
      jsonResponse({
        overall: "warn",
        checks: [],
        ports: [],
        events: [{ id: 1, timestamp: "2026-10-01T10:00:00Z", message: "older from snapshot" }],
      }),
    );
    backend.route("GET", "/api/events", () =>
      jsonResponse({ events: [{ id: 2, timestamp: "2026-10-01T11:00:00Z", message: "newer from stream" }] }),
    );
    await renderDiagnostics();
    const newer = screen.getByText("newer from stream");
    const older = screen.getByText("older from snapshot");
    expect(newer.compareDocumentPosition(older) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  test("an empty timeline does not fabricate history", async () => {
    await renderDiagnostics();
    expect(screen.getByText("No audit events reported")).toBeInTheDocument();
  });
});

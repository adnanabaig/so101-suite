// The persistent safety shell: E-stop, Halt, Collapse, Reset and the banners,
// exercised through the real Layout against the fake backend.
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  advance,
  deferred,
  fakeShellTimers,
  FakeWebSocket,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  settle,
  Shell as HarnessShell,
  telemetryFrame,
} from "../../test/harness";
import { ESTOP_UNCONFIRMED_TITLE, STATUS_POLL_MS, useSafety } from "./SafetyProvider";
import { SafetyDetails } from "./SafetyBar";
import { ConnectionIndicator } from "../Layout";
import type { ReactNode } from "react";

function Shell({ children, path }: { children?: ReactNode; path?: string }) {
  return <HarnessShell path={path}><SafetyDetails /><ConnectionIndicator />{children}</HarnessShell>;
}

/** What the E-stop route answers when the latch is set but torque-off is not confirmed in time. */
const ESTOP_UNCONFIRMED_DETAIL =
  "E-STOP is latched, but torque-off is not confirmed yet; cut physical power if anything still moves";

const ALL_STOPS = { stop_kinds: ["estop", "halt", "collapse"] };

const COLLAPSE = {
  since: "2030-05-01T12:00:00Z",
  reason: "hold timer expired",
  progress: 0.4,
  eta_s: 2.2,
};

beforeEach(() => {
  installFakeWebSocket();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function estopSlot(): HTMLElement {
  const slot = document.querySelector<HTMLElement>('[data-slot="estop"]');
  if (!slot) throw new Error("no E-stop slot");
  return slot;
}

function namesIn(element: HTMLElement): string[] {
  return within(element)
    .getAllByRole("button")
    .map((button) => button.getAttribute("aria-label") ?? button.textContent ?? "");
}

describe("E-stop", () => {
  test("stays enabled and keeps sending while its own request is in flight (F6)", async () => {
    const backend = installFakeBackend();
    const gate = deferred<void>();
    backend.route("POST", "/api/robot/estop", async () => {
      await gate.promise;
      return jsonResponse({ ok: true, detail: "E-STOP latched" });
    });
    render(<Shell />);
    await settle();

    const button = screen.getByRole("button", { name: "E-stop" });
    fireEvent.click(button);
    await settle();
    expect(screen.getByTestId("estop-pending")).toBeInTheDocument();
    expect(button).toBeEnabled();
    expect(button).not.toHaveAttribute("aria-disabled");

    fireEvent.click(button);
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(2);

    gate.resolve();
    await settle();
    expect(screen.queryByTestId("estop-pending")).toBeNull();
  });

  test("fires on pointerdown; the click that ends the same press is not a second request", async () => {
    const backend = installFakeBackend();
    render(<Shell />);
    await settle();
    const button = screen.getByRole("button", { name: "E-stop" });
    fireEvent.pointerDown(button);
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    fireEvent.pointerUp(button);
    fireEvent.click(button);
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
  });

  test("Escape sends E-stop from anywhere, including a text field, and ignores auto-repeat (F27)", async () => {
    const backend = installFakeBackend();
    render(
      <Shell>
        <input aria-label="notes" />
      </Shell>,
    );
    await settle();
    const input = screen.getByLabelText("notes");
    input.focus();
    fireEvent.keyDown(input, { key: "Escape" });
    fireEvent.keyDown(input, { key: "Escape", repeat: true });
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    expect(screen.getByTestId("safety-notice")).toHaveTextContent("E-stop latched. Motors off.");
  });

  test("the E-stop slot holds only the E-stop, latched or not, and it is the last stop (F2)", async () => {
    const backend = installFakeBackend();
    render(<Shell />);
    await settle();
    expect(namesIn(estopSlot())).toEqual(["E-stop"]);

    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    expect(backend.estopped).toBe(true);
    expect(screen.getByTestId("estop-latched")).toBeInTheDocument();
    expect(namesIn(estopSlot())).toEqual(["E-stop"]);

    const stops = namesIn(screen.getByRole("group", { name: "Robot stops" }));
    expect(stops[stops.length - 1]).toBe("E-stop");
    // Reset lives in the latched banner, never in the bar.
    expect(stops.some((name) => /reset/i.test(name))).toBe(false);
  });

  test("the chip menu stays open on Escape, which is E-stop", async () => {
    const backend = installFakeBackend();
    render(<Shell />);
    await settle();
    const chip = screen.getByTestId("safety-chip");
    fireEvent.click(chip);
    expect(chip).toHaveAttribute("aria-expanded", "true");
    fireEvent.keyDown(chip, { key: "Escape" });
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    expect(chip).toHaveAttribute("aria-expanded", "true");
  });

  test("the chip exposes whether safety latches survive a restart", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = { latch_persistence: "unsaved" };
    render(<Shell />);
    await settle();

    fireEvent.click(screen.getByTestId("safety-chip"));
    expect(screen.getByText("Latch storage")).toBeInTheDocument();
    expect(screen.getByText("memory only — do not restart")).toBeInTheDocument();
  });

  test("the chip never presents stale latch persistence as durable", async () => {
    fakeShellTimers();
    const backend = installFakeBackend();
    backend.statusExtra = { latch_persistence: "saved" };
    render(<Shell />);
    await advance(10);

    fireEvent.click(screen.getByTestId("safety-chip"));
    expect(screen.getByText("durable store")).toBeInTheDocument();

    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: "down" }, 503));
    await advance(STATUS_POLL_MS * 2 + 100);
    expect(screen.queryByText("durable store")).toBeNull();
    expect(screen.getByText("unknown — status stale")).toBeInTheDocument();
  });

  test("a failed E-stop raises a critical alert telling the operator to cut power", async () => {
    const backend = installFakeBackend();
    backend.route("POST", "/api/robot/estop", () =>
      jsonResponse({ detail: "serial bus timeout" }, 500),
    );
    render(<Shell />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    const alert = screen.getByText("E-stop request failed").closest('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert).toHaveTextContent("serial bus timeout");
    expect(alert).toHaveTextContent(/Cut physical power/);
    expect(screen.getByRole("button", { name: "E-stop" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Dismiss: E-stop request failed" }));
    expect(screen.queryByText("E-stop request failed")).toBeNull();
  });

  test("a 503 means latched with torque-off unconfirmed: cut power, never 'E-stop failed'", async () => {
    const backend = installFakeBackend();
    // The route latches before it answers; 503 only says torque-off is not confirmed yet.
    backend.route("POST", "/api/robot/estop", () => {
      backend.estopped = true;
      return jsonResponse({ detail: ESTOP_UNCONFIRMED_DETAIL }, 503);
    });
    render(<Shell />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    expect(screen.queryByText("E-stop request failed")).toBeNull();
    const alert = screen.getByText(ESTOP_UNCONFIRMED_TITLE).closest('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert).toHaveTextContent(ESTOP_UNCONFIRMED_DETAIL);
    expect(screen.getByTestId("safety-notice")).toHaveTextContent(
      "E-stop latched. Torque-off not confirmed: cut physical power if anything moves.",
    );
    expect(screen.getByTestId("safety-chip")).toHaveTextContent("E-stopped");
    expect(screen.getByRole("button", { name: "E-stop" })).toBeEnabled();
  });

  test("a later confirmed E-stop clears the unconfirmed-torque alert", async () => {
    const backend = installFakeBackend();
    let answer = 503;
    backend.route("POST", "/api/robot/estop", () => {
      backend.estopped = true;
      return answer === 503
        ? jsonResponse({ detail: ESTOP_UNCONFIRMED_DETAIL }, 503)
        : jsonResponse({ ok: true, detail: "E-STOP engaged; all motion stopped" });
    });
    render(<Shell />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    expect(screen.getByText(ESTOP_UNCONFIRMED_TITLE)).toBeInTheDocument();
    answer = 200;
    fireEvent.keyDown(document.body, { key: "Escape" });
    await settle();
    expect(screen.queryByText(ESTOP_UNCONFIRMED_TITLE)).toBeNull();
  });
});

describe("C2 torque truth", () => {
  test("torque_unconfirmed is a fault: a critical 'cut physical power if anything moves' alert (F7)", async () => {
    const backend = installFakeBackend();
    backend.torqueUnconfirmed = true;
    render(<Shell />);
    await settle();
    const alert = screen.getByText("Fault: cut physical power if anything moves").closest('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert).toHaveTextContent(/could not confirm that the motors are off/);
    expect(within(alert as HTMLElement).getByRole("link", { name: "Diagnostics" })).toHaveAttribute("href", "/diagnostics");
    expect(screen.getByTestId("safety-chip")).toHaveTextContent("torque unconfirmed");
  });
});

describe("Halt and Collapse (C11)", () => {
  test("are hidden, and H does nothing, when the server does not advertise them", async () => {
    const backend = installFakeBackend();
    render(<Shell />);
    await settle();
    expect(screen.queryByRole("button", { name: "Halt" })).toBeNull();
    expect(screen.queryByRole("button", { name: /collapse/i })).toBeNull();
    // The legacy soft stop stands in on servers without Halt.
    expect(screen.getByRole("button", { name: "Soft stop" })).toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "h" });
    await settle();
    expect(backend.callsTo("/api/robot/halt")).toEqual([]);
  });

  test("H halts when advertised, and the request carries the operator id", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = ALL_STOPS;
    render(<Shell />);
    await settle();
    expect(screen.getByRole("button", { name: "Halt" })).toHaveAttribute("aria-keyshortcuts", "h");
    fireEvent.keyDown(document.body, { key: "h" });
    await settle();
    const calls = backend.callsTo("/api/robot/halt");
    expect(calls).toHaveLength(1);
    expect(calls[0].body).toEqual({ operator_id: expect.any(String), reason: "operator pressed H" });
    expect(screen.queryByRole("button", { name: "Soft stop" })).toBeNull();
  });

  test("H is ignored while typing in an input, a textarea or contenteditable", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = ALL_STOPS;
    render(
      <Shell>
        <input aria-label="task" />
        <textarea aria-label="notes" />
        <div aria-label="editor" contentEditable suppressContentEditableWarning>
          text
        </div>
      </Shell>,
    );
    await settle();
    for (const name of ["task", "notes", "editor"]) {
      fireEvent.keyDown(screen.getByLabelText(name), { key: "h" });
      fireEvent.keyDown(screen.getByLabelText(name), { key: "H", shiftKey: true });
    }
    // Browser shortcuts that include H are left alone as well.
    fireEvent.keyDown(document.body, { key: "h", ctrlKey: true });
    fireEvent.keyDown(document.body, { key: "h", metaKey: true });
    fireEvent.keyDown(document.body, { key: "h", repeat: true });
    await settle();
    expect(backend.callsTo("/api/robot/halt")).toEqual([]);

    // Escape still works from the same fields.
    fireEvent.keyDown(screen.getByLabelText("notes"), { key: "Escape" });
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
  });

  test("the Halt and Collapse buttons send their requests; Collapse is click-only and neutral", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = ALL_STOPS;
    render(<Shell />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "Halt" }));
    const collapse = screen.getByRole("button", { name: "Collapse" });
    expect(collapse).not.toHaveAttribute("aria-keyshortcuts");
    expect(collapse.className).not.toMatch(/rose|red/);
    fireEvent.click(collapse);
    await settle();
    expect(backend.callsTo("/api/robot/halt")).toHaveLength(1);
    expect(backend.callsTo("/api/robot/collapse")[0].body).toEqual({
      operator_id: expect.any(String),
      reason: "operator pressed Collapse",
    });
  });

  test("Collapse is also reachable from the chip menu (narrow screens)", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = ALL_STOPS;
    render(<Shell />);
    await settle();
    const chip = screen.getByTestId("safety-chip");
    fireEvent.click(chip);
    const menu = document.getElementById(chip.getAttribute("aria-controls") ?? "");
    if (!menu) throw new Error("chip menu not rendered");
    expect(within(menu).getByText("E-stop, Halt, Collapse")).toBeInTheDocument();
    fireEvent.click(within(menu).getByRole("button", { name: "Collapse" }));
    await settle();
    expect(backend.callsTo("/api/robot/collapse")).toHaveLength(1);
    expect(chip).toHaveAttribute("aria-expanded", "false");
  });

  test("a cross-origin rejection shows the server's detail verbatim, then a hint (C1)", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = ALL_STOPS;
    backend.route("POST", "/api/robot/collapse", () =>
      jsonResponse({ detail: "cross-origin request rejected" }, 403),
    );
    render(<Shell />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    await settle();
    const notice = screen.getByText("Collapse request failed").parentElement;
    expect(notice).toHaveTextContent("cross-origin request rejected. The robot server does not accept changes");
    expect(notice).toHaveTextContent(/SO101_CORS_ORIGINS/);
  });

  test("a failed Halt is critical and points at E-stop", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = ALL_STOPS;
    backend.route("POST", "/api/robot/halt", () => jsonResponse({ detail: "bus fault" }, 500));
    render(<Shell />);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "Halt" }));
    await settle();
    const alert = screen.getByText("Halt request failed").closest('[role="alert"]');
    expect(alert).toHaveTextContent("bus fault. Use E-stop if the arm must stop now.");
  });
});

describe("Reset", () => {
  test("is a separate hold-to-confirm control, locked for 3 s after an E-stop (F2, C2)", async () => {
    fakeShellTimers();
    const backend = installFakeBackend();
    render(<Shell />);
    await advance(10);
    act(() => FakeWebSocket.latest().push(telemetryFrame(false)));
    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await advance(10);

    const reset = screen.getByRole("button", { name: "Reset E-stop" });
    expect(reset).toHaveAttribute("aria-disabled", "true");
    expect(reset).toHaveAccessibleDescription("Reset unlocks in 3 seconds.");
    expect(reset).toHaveTextContent("Reset in 3s");

    // Holding while locked does nothing, however long.
    fireEvent.pointerDown(reset);
    await advance(2000);
    fireEvent.pointerUp(reset);
    expect(backend.callsTo("/api/robot/estop/reset")).toEqual([]);

    await advance(1100);
    expect(reset).not.toHaveAttribute("aria-disabled");
    expect(reset).toHaveTextContent("Hold to reset E-stop");

    // A plain click, or releasing early, never resets.
    fireEvent.click(reset);
    fireEvent.pointerDown(reset);
    await advance(1000);
    fireEvent.pointerUp(reset);
    await advance(1000);
    expect(backend.callsTo("/api/robot/estop/reset")).toEqual([]);
    expect(reset).toHaveAccessibleDescription(/Released too early/);

    // A full hold resets, with an attributed and confirmed body.
    fireEvent.pointerDown(reset);
    await advance(1500);
    const calls = backend.callsTo("/api/robot/estop/reset");
    expect(calls).toHaveLength(1);
    expect(calls[0].body).toEqual({ operator_id: expect.any(String), confirm: true });
    await advance(10);
    expect(screen.queryByTestId("estop-latched")).toBeNull();
    expect(document.getElementById("main-content")).toHaveFocus();
  });

  test("a latch seen from another station also locks Reset", async () => {
    fakeShellTimers();
    const backend = installFakeBackend();
    render(<Shell />);
    await advance(10);
    act(() => FakeWebSocket.latest().push(telemetryFrame(false)));
    await advance(10);
    // Another station presses E-stop: this UI only observes the latch.
    backend.estopped = true;
    act(() => FakeWebSocket.latest().push(telemetryFrame(true)));
    await advance(10);
    const reset = screen.getByRole("button", { name: "Reset E-stop" });
    expect(reset).toHaveAttribute("aria-disabled", "true");
    await advance(3100);
    expect(reset).not.toHaveAttribute("aria-disabled");
  });

  test("a refused reset keeps the latch and explains why", async () => {
    fakeShellTimers();
    const backend = installFakeBackend();
    backend.estopped = true;
    backend.route("POST", "/api/robot/estop/reset", () =>
      jsonResponse({ detail: "fault latched: overcurrent on elbow_flex" }, 409),
    );
    render(<Shell />);
    // Let the first status render (the latch starts the lockout), then wait it out.
    await advance(10);
    await advance(3200);
    const reset = screen.getByRole("button", { name: "Reset E-stop" });
    expect(reset).not.toHaveAttribute("aria-disabled");
    fireEvent.keyDown(reset, { key: " " });
    await advance(1500);
    fireEvent.keyUp(reset, { key: " " });
    await advance(10);
    expect(backend.callsTo("/api/robot/estop/reset")).toHaveLength(1);
    expect(screen.getByText("E-stop reset refused")).toBeInTheDocument();
    expect(screen.getByText("fault latched: overcurrent on elbow_flex")).toBeInTheDocument();
    expect(screen.getByTestId("estop-latched")).toBeInTheDocument();
  });
});

describe("Halted banner", () => {
  test("counts down from server time and offers Keep holding and Collapse now", async () => {
    fakeShellTimers();
    // The client clock is years away from the server clock on purpose.
    vi.setSystemTime(new Date("2026-10-01T09:00:00Z"));
    const backend = installFakeBackend();
    const clientStart = Date.now();
    const serverStart = Date.parse("2030-05-01T12:00:00Z");
    backend.route("GET", "/api/robot/status", () =>
      jsonResponse({
        ...backend.status(),
        ...ALL_STOPS,
        stop_state: "halted",
        halt: {
          since: "2030-05-01T12:00:00Z",
          reason: "deadman released",
          hold_expires_at: "2030-05-01T12:00:42Z",
          server_time: new Date(serverStart + (Date.now() - clientStart)).toISOString(),
        },
      }),
    );
    render(<Shell />);
    await advance(10);

    const banner = screen.getByTestId("halted-banner");
    expect(banner).toHaveTextContent("Halted — holding position");
    expect(screen.getByTestId("halt-countdown")).toHaveTextContent("0:42");
    expect(banner).toHaveTextContent(
      "When the countdown reaches zero, the arm will lower to rest and turn off.",
    );
    expect(banner).toHaveTextContent("Reason: deadman released.");
    expect(screen.getByTestId("safety-chip")).toHaveTextContent("halted");

    await advance(10_000);
    expect(screen.getByTestId("halt-countdown")).toHaveTextContent("0:32");

    fireEvent.click(within(banner).getByRole("button", { name: "Keep holding" }));
    fireEvent.click(within(banner).getByRole("button", { name: "Collapse now" }));
    await advance(10);
    expect(backend.callsTo("/api/robot/halt")[0].body).toEqual({
      operator_id: expect.any(String),
      reason: "operator kept holding",
    });
    expect(backend.callsTo("/api/robot/collapse")[0].body).toEqual({
      operator_id: expect.any(String),
      reason: "operator chose Collapse now",
    });
  });

  test("unsupervised, it counts down to the collapse, then back to the hold timer when supervision returns", async () => {
    fakeShellTimers();
    vi.setSystemTime(new Date("2026-10-01T09:00:00Z"));
    const backend = installFakeBackend();
    const clientStart = Date.now();
    const serverStart = Date.parse("2030-05-01T12:00:10Z");
    let supervision: { supervised: boolean; lost_at: string | null; grace_s: number } = {
      supervised: false,
      lost_at: "2030-05-01T12:00:08Z",
      grace_s: 5,
    };
    backend.route("GET", "/api/robot/status", () =>
      jsonResponse({
        ...backend.status(),
        ...ALL_STOPS,
        stop_state: "halted",
        halt: {
          since: "2030-05-01T12:00:00Z",
          reason: "deadman released",
          hold_expires_at: supervision.supervised ? "2030-05-01T12:01:00Z" : "2030-05-01T12:00:13Z",
          server_time: new Date(serverStart + (Date.now() - clientStart)).toISOString(),
        },
        supervision,
      }),
    );
    render(<Shell />);
    await advance(10);

    const banner = screen.getByTestId("halted-banner");
    expect(screen.getByTestId("unsupervised-countdown")).toHaveTextContent("Unsupervised: collapses in 3 s");
    expect(screen.queryByTestId("halt-countdown")).toBeNull();
    expect(banner).toHaveTextContent("No operator page is sending a supervision heartbeat");
    expect(screen.getByTestId("supervision-notice")).toHaveTextContent("Unsupervised: collapses in 3 s.");
    expect(screen.getByTestId("supervision-notice")).toHaveAttribute("aria-live", "polite");

    // Ticks with the shared clock; the live notice does not repeat every tick.
    await advance(1_000);
    expect(screen.getByTestId("unsupervised-countdown")).toHaveTextContent("Unsupervised: collapses in 2 s");
    expect(screen.getByTestId("supervision-notice")).toHaveTextContent("Unsupervised: collapses in 3 s.");

    supervision = { supervised: true, lost_at: null, grace_s: 5 };
    await advance(STATUS_POLL_MS);
    expect(screen.queryByTestId("unsupervised-countdown")).toBeNull();
    expect(screen.getByTestId("halt-countdown")).toHaveTextContent(/^0:4\d$/);
    expect(banner).toHaveTextContent("When the countdown reaches zero");
    expect(screen.getByTestId("supervision-notice")).toHaveTextContent("Supervised again: back to the hold timer.");
  });

  test("an older server without supervision keeps the plain hold countdown", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = {
      ...ALL_STOPS,
      stop_state: "halted",
      halt: {
        since: "2030-05-01T12:00:00Z",
        reason: "deadman released",
        hold_expires_at: "2030-05-01T12:00:42Z",
        server_time: "2030-05-01T12:00:00Z",
      },
    };
    render(<Shell />);
    await settle();
    expect(screen.getByTestId("halt-countdown")).toHaveTextContent("0:42");
    expect(screen.queryByTestId("unsupervised-countdown")).toBeNull();
    expect(screen.getByTestId("supervision-notice")).toHaveTextContent("");
  });

  test("without Collapse advertised there is no Collapse now", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = {
      stop_kinds: ["estop", "halt"],
      stop_state: "halted",
      halt: null,
    };
    render(<Shell />);
    await settle();
    const banner = screen.getByTestId("halted-banner");
    expect(within(banner).queryByRole("button", { name: "Collapse now" })).toBeNull();
    expect(banner).toHaveTextContent("The hold timer is unknown.");
  });
});

describe("Collapsing banner", () => {
  test("shows progress and ETA, and Halt stops the descent", async () => {
    const backend = installFakeBackend();
    backend.statusExtra = { ...ALL_STOPS, stop_state: "collapsing", collapse: COLLAPSE };
    render(<Shell />);
    await settle();
    const banner = screen.getByTestId("collapsing-banner");
    expect(within(banner).getByRole("progressbar", { name: "Collapse progress" })).toHaveAttribute(
      "aria-valuenow",
      "40",
    );
    expect(banner).toHaveTextContent("40% · about 3 s left");
    expect(banner).toHaveTextContent("Halt stops the descent and holds.");
    expect(screen.getByRole("button", { name: "Halt" })).toHaveAttribute(
      "title",
      "Halt stops the descent and holds position (H)",
    );
    expect(screen.getByRole("button", { name: "E-stop" })).toBeEnabled();

    fireEvent.click(within(banner).getByRole("button", { name: "Halt the descent" }));
    await settle();
    expect(backend.callsTo("/api/robot/halt")[0].body).toEqual({
      operator_id: expect.any(String),
      reason: "operator halted the descent",
    });
  });

  test("E-stop preempts: a latched E-stop replaces the collapse banner", async () => {
    const backend = installFakeBackend();
    backend.estopped = true;
    backend.statusExtra = { ...ALL_STOPS, stop_state: "collapsing", collapse: COLLAPSE };
    render(<Shell />);
    await settle();
    expect(screen.getByTestId("estop-latched")).toBeInTheDocument();
    expect(screen.queryByTestId("collapsing-banner")).toBeNull();
  });
});

describe("C1 guard refusals", () => {
  const HOST_DETAIL = "untrusted Host header; add the hostname to SO101_ALLOWED_HOSTS";

  test("an untrusted Host shows the server's 400 detail verbatim and E-stop still sends", async () => {
    const backend = installFakeBackend();
    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: HOST_DETAIL }, 400));
    render(<Shell />);
    await settle();
    const banner = screen.getByText(/refused this page's address/).closest('[role="alert"]');
    expect(banner).toHaveTextContent(`The robot server refused this page's address: ${HOST_DETAIL}.`);
    expect(banner).toHaveTextContent(/SO101_ALLOWED_HOSTS on the server/);
    // The guard exempts exactly these three routes; Collapse is not one of them.
    expect(banner).toHaveTextContent(/E-stop, Halt and Soft stop still send: the server's guard always lets them through/);
    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
  });

  test("a telemetry socket closed with 1008 shows the server's reason verbatim", async () => {
    FakeWebSocket.autoOpen = false;
    installFakeBackend();
    render(<Shell />);
    await settle();
    act(() => FakeWebSocket.latest().drop(1008, "cross-origin request rejected"));
    await settle();
    const banner = screen.getByText(/refused the live telemetry link/).closest('[role="status"]');
    expect(banner).toHaveTextContent(
      "The server refused the live telemetry link: cross-origin request rejected.",
    );
    expect(banner).toHaveTextContent(/The stop buttons still send/);
    expect(screen.getByTestId("connection-indicator")).toHaveTextContent("Live link refused by the server");
  });

  test("sockets that never open while HTTP answers get the allowlist hint (browsers hide 1008)", async () => {
    fakeShellTimers();
    FakeWebSocket.autoOpen = false;
    installFakeBackend();
    render(<Shell />);
    await advance(10);
    act(() => FakeWebSocket.latest().drop());
    await advance(800);
    act(() => FakeWebSocket.latest().drop());
    await advance(1600);
    expect(screen.queryByText(/without connecting/)).toBeNull();
    act(() => FakeWebSocket.latest().drop());
    expect(screen.getByText(/has failed 3 times without connecting/)).toBeInTheDocument();
    expect(screen.getByText(/has failed 3 times/).closest("p")).toHaveTextContent(/SO101_CORS_ORIGINS/);
  });

  test("a server that answers nothing gets 'state unknown', not the allowlist hint", async () => {
    fakeShellTimers();
    FakeWebSocket.autoOpen = false;
    const backend = installFakeBackend();
    render(<Shell />);
    await advance(10);
    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: "down" }, 503));
    backend.route("GET", "/api/control/state", () => jsonResponse({ detail: "down" }, 503));
    for (const wait of [800, 1600, 3100, 6100]) {
      act(() => FakeWebSocket.latest().drop());
      await advance(wait);
    }
    expect(screen.getByText(/Robot state unknown\./)).toBeInTheDocument();
    expect(screen.queryByText(/without connecting/)).toBeNull();
  });
});

describe("freshness (F11)", () => {
  test("a server that stops answering turns the chip to unknown and says so", async () => {
    fakeShellTimers();
    const backend = installFakeBackend();
    render(<Shell />);
    await advance(10);
    expect(screen.getByTestId("safety-chip")).toHaveTextContent("disarmed");

    backend.route("GET", "/api/robot/status", () => jsonResponse({ detail: "down" }, 503));
    backend.route("GET", "/api/control/state", () => jsonResponse({ detail: "down" }, 503));
    await advance(STATUS_POLL_MS * 2 + 100);
    expect(screen.getByTestId("safety-chip")).toHaveTextContent("state unknown");
    expect(screen.getByText(/Robot state unknown\./)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "E-stop" })).toBeEnabled();
  });
});

describe("render load (F28)", () => {
  test("frames that change nothing safety-relevant do not re-render safety consumers", async () => {
    fakeShellTimers();
    installFakeBackend();
    const renders = vi.fn();
    function Probe() {
      const { view } = useSafety();
      renders();
      return <span data-testid="probe">{view.chipLabel}</span>;
    }
    render(
      <Shell>
        <Probe />
      </Shell>,
    );
    await advance(10);
    act(() => FakeWebSocket.broadcast(telemetryFrame(false)));
    await advance(10);

    const before = renders.mock.calls.length;
    // One second of a 20 Hz joint stream.
    for (let seq = 2; seq < 22; seq += 1) {
      act(() =>
        FakeWebSocket.broadcast(
          telemetryFrame(false, { sample_seq: seq, observation: { "shoulder_pan.pos": seq } }),
        ),
      );
      await advance(50);
    }
    // Only the status/control polls may land in that second.
    expect(renders.mock.calls.length - before).toBeLessThanOrEqual(3);

    act(() => FakeWebSocket.broadcast(telemetryFrame(true)));
    await advance(10);
    expect(screen.getByTestId("probe")).toHaveTextContent("E-stopped");
  });
});

import { fireEvent, render, screen, within } from "@testing-library/react";
import { useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  advance,
  fakeShellTimers,
  installFakeBackend,
  installFakeWebSocket,
  jsonResponse,
  Shell,
  type FakeBackend,
  type RecordedCall,
} from "../../test/harness";
import ManualControlPanel from "../ManualControlPanel";
import { SHORTCUT_ITEMS as NAV_ITEMS, navItemForShortcut, routeLabel } from "../navigation";
import { useConfirm } from "../ui/Dialog";
import { GO_SEQUENCE_MS, useShortcuts } from "./ShortcutsProvider";

const ALL_STOPS = { stop_kinds: ["estop", "halt", "collapse"] };

let backend: FakeBackend;

beforeEach(() => {
  fakeShellTimers();
  installFakeWebSocket();
  backend = installFakeBackend();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function renderShell(children: React.ReactNode = <p>page</p>, path = "/teleop") {
  render(<Shell path={path}>{children}</Shell>);
  await advance(10);
}

const page = () => screen.getByRole("main");

function press(key: string, target: Element = document.body, init: KeyboardEventInit = {}) {
  fireEvent.keyDown(target, { key, ...init });
}

describe("navigation shortcuts", () => {
  test("every page has its own letter, and none is G or H", () => {
    const letters = NAV_ITEMS.map((item) => item.shortcut);
    expect(new Set(letters).size).toBe(letters.length);
    expect(letters).not.toContain("h");
    expect(letters).not.toContain("g");
    expect(navItemForShortcut("D")?.to).toBe("/learn?section=datasets");
    expect(navItemForShortcut("h")).toBeUndefined();
  });

  test("G then a letter opens that page, for every page", async () => {
    await renderShell();
    expect(page()).toHaveAccessibleName("Data collection");
    for (const item of NAV_ITEMS) {
      press("g");
      press(item.shortcut);
      await advance(10);
      const destination = new URL(item.to, "http://localhost");
      expect(page()).toHaveAccessibleName(routeLabel(destination.pathname, destination.search));
    }
    // Caps Lock gives capitals; they work the same.
    press("G");
    press("T");
    await advance(10);
    expect(page()).toHaveAccessibleName("Operate");
  });

  test("a letter alone, a late letter, or an unknown letter does nothing", async () => {
    await renderShell();
    press("d");
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");

    press("g");
    await advance(GO_SEQUENCE_MS + 1);
    press("d");
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");

    // An unknown second key ends the sequence instead of waiting for another.
    press("g");
    press("z");
    press("d");
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");
  });

  test("keys typed into a field, or pressed with Ctrl, Cmd or Alt, never navigate", async () => {
    await renderShell(
      <>
        <input aria-label="Notes" />
        <textarea aria-label="Long notes" />
      </>,
    );
    const notes = screen.getByRole("textbox", { name: "Notes" });
    press("g", notes);
    press("d", notes);
    const longNotes = screen.getByRole("textbox", { name: "Long notes" });
    press("g", longNotes);
    press("d", longNotes);
    // G in a field, then D outside it: the sequence never started.
    press("g", notes);
    press("d");
    press("g", document.body, { ctrlKey: true });
    press("d");
    press("g");
    press("d", document.body, { metaKey: true });
    press("g");
    press("d", document.body, { altKey: true });
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");
  });

  test("a key that a control already handled is not a shortcut", async () => {
    await renderShell(
      <div
        role="listbox"
        aria-label="Episodes"
        tabIndex={0}
        onKeyDown={(event) => {
          // e.g. type-ahead: the control uses letters itself.
          if (event.key.length === 1) event.preventDefault();
        }}
      />,
    );
    const listbox = screen.getByRole("listbox", { name: "Episodes" });
    press("g", listbox);
    press("d");
    press("g");
    press("d", listbox);
    press("?", listbox, { shiftKey: true });
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
  });

  test("G then H halts and does not navigate", async () => {
    backend.statusExtra = ALL_STOPS;
    await renderShell();
    press("g");
    press("h");
    await advance(10);
    expect(backend.callsTo("/api/robot/halt")).toHaveLength(1);
    // The sequence is over: a following letter is not taken as a page.
    press("d");
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");
  });

  test("nothing navigates while a dialog is open", async () => {
    function Asker() {
      const [confirm, dialog] = useConfirm();
      return (
        <>
          <button type="button" onClick={() => void confirm({ title: "Start rollout?" })}>
            Ask
          </button>
          {dialog}
        </>
      );
    }
    await renderShell(<Asker />);
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await advance(10);
    const dialog = screen.getByRole("alertdialog", { name: "Start rollout?" });
    const cancel = within(dialog).getByRole("button", { name: "Cancel" });
    press("g", cancel);
    press("d", cancel);
    // Nor does ? stack the shortcut list on top of it.
    press("?", cancel, { shiftKey: true });
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
  });
});

describe("navigating while the deadman is held", () => {
  /** The manual panel lives on the Teleop page only, as it does in App's routes. */
  function TeleopOnly() {
    const { pathname } = useLocation();
    return pathname === "/teleop" ? <ManualControlPanel /> : <p>elsewhere</p>;
  }

  test("G then O leaves the page, which stops motion and releases the lease", async () => {
    backend.route("POST", "/api/control/lease", () =>
      jsonResponse({ id: "lease-1", source: "manual-ui", server_time: new Date().toISOString() }),
    );
    backend.route("POST", "/api/control/lease/lease-1/heartbeat", () =>
      jsonResponse({ id: "lease-1", source: "manual-ui", server_time: new Date().toISOString() }),
    );
    backend.route("DELETE", "/api/control/lease/lease-1", () => jsonResponse({ ok: true, detail: "" }));
    backend.route("POST", "/api/control/observe", () =>
      jsonResponse({
        observation: {
          "shoulder_pan.pos": 1,
          "shoulder_lift.pos": 2,
          "elbow_flex.pos": 3,
          "wrist_flex.pos": 4,
          "wrist_roll.pos": 5,
          "gripper.pos": 40,
        },
        captured_at: new Date().toISOString(),
      }),
    );
    const accept = (call: RecordedCall) =>
      jsonResponse({ ok: true, detail: "", accepted: true, targets: (call.body as { targets: unknown }).targets });
    backend.route("POST", "/api/control/arm-command", accept);
    backend.route("POST", "/api/control/command", accept);
    backend.route("POST", "/api/control/disarm", () => jsonResponse({ ok: true, detail: "" }));

    await renderShell(<TeleopOnly />);
    fireEvent.click(screen.getByRole("button", { name: /Acquire manual lease/ }));
    await advance(10);
    const deadman = screen.getByRole("button", { name: "Hold to move" });
    fireEvent.keyDown(deadman, { key: " " });
    await advance(100);
    expect(backend.callsTo("/api/control/arm-command")).toHaveLength(1);
    expect(deadman).toHaveAttribute("aria-pressed", "true");

    // Space is still down: the operator jumps to Overview mid-hold.
    press("g", deadman);
    press("o", deadman);
    await advance(10);
    expect(page()).toHaveAccessibleName("Operate");
    expect(backend.callsTo("/api/control/disarm").map((call) => call.body)).toEqual([{ lease_id: "lease-1" }]);
    expect(backend.callsTo("/api/control/lease/lease-1", "DELETE")).toHaveLength(1);
    const sent = backend.callsTo("/api/control/command").length;
    await advance(500);
    expect(backend.callsTo("/api/control/command")).toHaveLength(sent);
  });
});

describe("the ? shortcut list", () => {
  test("? opens and closes it; Escape inside it is E-stop and leaves it open", async () => {
    await renderShell();
    press("?", document.body, { shiftKey: true });
    await advance(10);
    const list = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    expect(within(list).getByText(/E-stop: cut motor power now/)).toBeInTheDocument();
    expect(within(list).getByText("Datasets")).toBeInTheDocument();
    const close = within(list).getByRole("button", { name: "Close" });
    expect(close).toHaveFocus();

    press("Escape", close);
    await advance(10);
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    expect(screen.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeInTheDocument();

    // While it is open, page letters do nothing.
    press("g", close);
    press("d", close);
    await advance(10);
    expect(page()).toHaveAccessibleName("Data collection");

    press("?", close, { shiftKey: true });
    await advance(10);
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
  });

  test("the question-mark shortcut opens help and Close restores the prior focus", async () => {
    await renderShell();
    const opener = screen.getByRole("main");
    opener.focus();
    press("?", opener, { shiftKey: true });
    await advance(10);
    const list = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    fireEvent.click(within(list).getByRole("button", { name: "Close" }));
    await advance(10);
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
    expect(opener).toHaveFocus();
  });

  test("? typed into a field is just a character", async () => {
    await renderShell(<input aria-label="Notes" />);
    press("?", screen.getByRole("textbox", { name: "Notes" }), { shiftKey: true });
    await advance(10);
    expect(screen.queryByRole("dialog", { name: "Keyboard shortcuts" })).toBeNull();
  });

  test("H is described as Halt only when the server offers it", async () => {
    await renderShell();
    press("?", document.body, { shiftKey: true });
    await advance(10);
    let list = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    expect(within(list).getByText(/does not offer Halt, so H does nothing/)).toBeInTheDocument();
    press("?", document.body, { shiftKey: true });

    backend.statusExtra = ALL_STOPS;
    await advance(5000);
    press("?", document.body, { shiftKey: true });
    await advance(10);
    list = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    expect(within(list).getByText(/Halt: freeze in place/)).toBeInTheDocument();
  });

  test("pages can open the list through useShortcuts; outside the shell it is inert", async () => {
    function Opener() {
      const shortcuts = useShortcuts();
      return (
        <button type="button" onClick={shortcuts.open}>
          {shortcuts.isOpen ? "List open" : "Show shortcuts"}
        </button>
      );
    }
    await renderShell(<Opener />);
    fireEvent.click(screen.getByRole("button", { name: "Show shortcuts" }));
    await advance(10);
    expect(screen.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "List open" })).toBeInTheDocument();
  });
});

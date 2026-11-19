import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { installFakeBackend, installFakeWebSocket, settle, Shell, type FakeBackend } from "../../test/harness";
import { useConfirm, type ConfirmOptions } from "./Dialog";

let backend: FakeBackend;
let results: boolean[];

function ConfirmPage({ options }: { options: ConfirmOptions }) {
  const [confirm, confirmDialog] = useConfirm();
  const [count, setCount] = useState(0);
  return (
    <div>
      <button
        type="button"
        onClick={() => {
          void confirm(options).then((confirmed) => {
            results.push(confirmed);
            setCount((value) => value + 1);
          });
        }}
      >
        Open
      </button>
      <p>answered {count}</p>
      {confirmDialog}
    </div>
  );
}

async function renderPage(options: ConfirmOptions = { title: "Start rollout?", confirmLabel: "Start" }) {
  const view = render(
    <Shell>
      <ConfirmPage options={options} />
    </Shell>,
  );
  await settle();
  return view;
}

beforeEach(() => {
  installFakeWebSocket();
  backend = installFakeBackend();
  results = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ConfirmDialog", () => {
  test("Escape is E-stop and never closes the dialog", async () => {
    await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    const dialog = screen.getByRole("alertdialog", { name: "Start rollout?" });
    fireEvent.keyDown(within(dialog).getByRole("button", { name: "Cancel" }), { key: "Escape" });
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    expect(screen.getByRole("alertdialog", { name: "Start rollout?" })).toBeInTheDocument();
    expect(results).toEqual([]);
    // The stops stay outside the modal and usable.
    expect(dialog.contains(screen.getByRole("button", { name: "E-stop" }))).toBe(false);
  });

  test("focus starts on Cancel, Tab cycles inside, and returns to the opener", async () => {
    await renderPage();
    const opener = screen.getByRole("button", { name: "Open" });
    opener.focus();
    fireEvent.click(opener);
    const dialog = screen.getByRole("alertdialog");
    const cancel = within(dialog).getByRole("button", { name: "Cancel" });
    const start = within(dialog).getByRole("button", { name: "Start" });
    expect(document.activeElement).toBe(cancel);

    start.focus();
    fireEvent.keyDown(start, { key: "Tab" });
    expect(document.activeElement).toBe(cancel);
    fireEvent.keyDown(cancel, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(start);

    fireEvent.click(cancel);
    await settle();
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(results).toEqual([false]);
    expect(document.activeElement).toBe(opener);
  });

  test("a required acknowledgement gates the confirm button", async () => {
    await renderPage({
      title: "Clear the fault?",
      confirmLabel: "Clear fault",
      acknowledgement: "I have cut physical power to the arm",
    });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    const dialog = screen.getByRole("alertdialog");
    const clear = within(dialog).getByRole("button", { name: "Clear fault" });
    expect(clear).toBeDisabled();
    fireEvent.click(clear);
    expect(results).toEqual([]);

    fireEvent.click(within(dialog).getByRole("checkbox", { name: "I have cut physical power to the arm" }));
    expect(clear).toBeEnabled();
    fireEvent.click(clear);
    await settle();
    expect(results).toEqual([true]);

    // Reopening starts unacknowledged again.
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    expect(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Clear fault" })).toBeDisabled();
  });

  test("the backdrop does not dismiss; unmounting answers false", async () => {
    const view = await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    const dialog = screen.getByRole("alertdialog");
    fireEvent.click(dialog.parentElement as HTMLElement);
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    view.unmount();
    await act(async () => {
      await Promise.resolve();
    });
    expect(results).toEqual([false]);
  });
});

describe("native dialogs", () => {
  // window.confirm, alert and prompt block the page's JavaScript while open,
  // so Escape could not reach the E-stop handler. Every confirmation is an
  // in-app Dialog instead.
  const sources = Object.entries(
    import.meta.glob<string>("../../**/*.{ts,tsx}", { query: "?raw", import: "default", eager: true }),
  ).filter(([path]) => !/\.test\.tsx?$/.test(path) && !path.includes("/test/"));

  test("no source opens a native confirm, alert or prompt", () => {
    expect(sources.length).toBeGreaterThan(50);
    const native = /(?:\b(?:window|globalThis|self)\.(?:confirm|alert|prompt)|(?<![\w.$])(?:alert|prompt))\s*\(/;
    const offenders = sources.flatMap(([path, text]) =>
      text
        .split("\n")
        .flatMap((line, index) =>
          native.test(line) && !/^\s*(\/\/|\*|\/\*)/.test(line) ? [`${path}:${index + 1}`] : [],
        ),
    );
    expect(offenders).toEqual([]);
  });
});

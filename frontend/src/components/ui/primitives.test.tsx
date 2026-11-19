import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { BoolPill, StatusPill } from "./StatusPill";
import { Tabs, type TabItem } from "./Tabs";
import { MAX_TOASTS, TOAST_MS, ToastProvider, useToast, type ToastTone } from "./Toast";

afterEach(() => {
  vi.useRealTimers();
});

describe("StatusPill", () => {
  test("shows label and value, and reads as 'label: value'", () => {
    render(<StatusPill label="lease" value="owned" tone="ok" />);
    const label = screen.getByText("lease");
    expect(label).toHaveAttribute("data-pill-label");
    expect(label.nextElementSibling).toHaveTextContent("owned");
    // The colon is for screen readers only.
    expect(label.querySelector(".sr-only")).toHaveTextContent(":");
    expect(label.parentElement).toHaveTextContent("lease:owned");
    expect(label.parentElement?.className).toContain("emerald");
  });

  test("an empty label gets no separator, and a label alone is the whole status", () => {
    const { container, rerender } = render(<StatusPill label="" value="front" />);
    expect(container.querySelector(".sr-only")).toBeNull();
    expect(container.querySelector("[data-pill-label]")).toBeNull();
    expect(container).toHaveTextContent(/^front$/);
    rerender(<StatusPill label="idle" />);
    expect(container.querySelector(".sr-only")).toBeNull();
    // A lone word is not dimmed like a label: it keeps full contrast.
    expect(container.querySelector("[data-pill-label]")).toBeNull();
    expect(container.querySelector(".font-semibold")).toHaveTextContent(/^idle$/);
  });

  test("BoolPill says yes or no in words and inverts the tone on request", () => {
    const { container, rerender } = render(<BoolPill label="follower" value />);
    expect(container).toHaveTextContent("yes");
    expect(container.firstElementChild?.className).toContain("emerald");
    rerender(<BoolPill label="estopped" value invert />);
    expect(container).toHaveTextContent("yes");
    expect(container.firstElementChild?.className).toContain("red");
  });
});

const TABS: TabItem[] = [
  { id: "one", label: "One", panel: <p>Panel one</p> },
  { id: "two", label: "Two", panel: <p>Panel two</p> },
  { id: "three", label: "Three", panel: <button type="button">Inside three</button> },
];

describe("Tabs", () => {
  test("wires tablist, tabs and the one rendered panel", () => {
    render(<Tabs label="Inspector" tabs={TABS} />);
    const list = screen.getByRole("tablist", { name: "Inspector" });
    const tabs = within(list).getAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual(["One", "Two", "Three"]);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(tabs[0]).toHaveAttribute("tabindex", "0");
    expect(tabs[1]).toHaveAttribute("aria-selected", "false");
    expect(tabs[1]).toHaveAttribute("tabindex", "-1");
    expect(tabs[1]).not.toHaveAttribute("aria-controls");
    const panel = screen.getByRole("tabpanel", { name: "One" });
    expect(tabs[0]).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveTextContent("Panel one");
    expect(screen.queryByText("Panel two")).toBeNull();
  });

  test("arrow keys wrap, Home and End jump, and focus follows the selection", () => {
    render(<Tabs label="Inspector" tabs={TABS} />);
    const [one, two, three] = screen.getAllByRole("tab");
    one.focus();
    fireEvent.keyDown(one, { key: "ArrowLeft" });
    expect(three).toHaveAttribute("aria-selected", "true");
    expect(three).toHaveFocus();
    expect(screen.getByRole("tabpanel")).toHaveTextContent("Inside three");
    fireEvent.keyDown(three, { key: "ArrowRight" });
    expect(one).toHaveFocus();
    fireEvent.keyDown(one, { key: "End" });
    expect(three).toHaveFocus();
    fireEvent.keyDown(three, { key: "Home" });
    expect(one).toHaveFocus();
    fireEvent.keyDown(one, { key: "ArrowRight" });
    expect(two).toHaveFocus();
    expect(two).toHaveAttribute("tabindex", "0");
    expect(one).toHaveAttribute("tabindex", "-1");
    // Other keys are left alone.
    const event = new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true });
    two.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(two).toHaveAttribute("aria-selected", "true");
  });

  test("a click selects; a controlled selection follows its owner", () => {
    function Controlled() {
      const [selected, setSelected] = useState("two");
      return (
        <>
          <button type="button" onClick={() => setSelected("three")}>
            Jump to three
          </button>
          <Tabs label="Controlled" tabs={TABS} selected={selected} onSelect={setSelected} />
        </>
      );
    }
    render(<Controlled />);
    expect(screen.getByRole("tab", { name: "Two" })).toHaveAttribute("aria-selected", "true");
    fireEvent.click(screen.getByRole("button", { name: "Jump to three" }));
    expect(screen.getByRole("tabpanel", { name: "Three" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "One" }));
    expect(screen.getByRole("tabpanel", { name: "One" })).toHaveTextContent("Panel one");
  });

  test("an unknown selection falls back to the first tab", () => {
    render(<Tabs label="Fallback" tabs={TABS} selected="missing" />);
    expect(screen.getByRole("tab", { name: "One" })).toHaveAttribute("aria-selected", "true");
  });
});

function ToastButtons() {
  const toast = useToast();
  const fire = (message: string, tone?: ToastTone) => () => toast.show(message, { tone });
  return (
    <>
      <button type="button" onClick={fire("Saved", "success")}>
        success
      </button>
      <button type="button" onClick={fire("Could not refresh", "error")}>
        error
      </button>
      <button type="button" onClick={fire("Note")}>
        info
      </button>
      {["a", "b", "c", "d"].map((name) => (
        <button key={name} type="button" onClick={fire(`Toast ${name}`)}>
          {`toast ${name}`}
        </button>
      ))}
    </>
  );
}

describe("Toast", () => {
  test("the live region exists before any toast, and info fades after TOAST_MS", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider>
        <ToastButtons />
      </ToastProvider>,
    );
    const region = screen.getByTestId("toast-region");
    expect(region).toHaveAttribute("aria-live", "polite");
    expect(region).toBeEmptyDOMElement();
    fireEvent.click(screen.getByRole("button", { name: "info" }));
    expect(within(region).getByText("Note")).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(TOAST_MS - 1);
    });
    expect(within(region).getByText("Note")).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(within(region).queryByText("Note")).toBeNull();
  });

  test("errors stay until dismissed; the same message replaces itself", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider>
        <ToastButtons />
      </ToastProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "error" }));
    fireEvent.click(screen.getByRole("button", { name: "error" }));
    act(() => {
      vi.advanceTimersByTime(TOAST_MS * 5);
    });
    const region = screen.getByTestId("toast-region");
    expect(within(region).getAllByText("Could not refresh")).toHaveLength(1);
    expect(region.querySelector('[data-tone="error"]')).not.toBeNull();
    fireEvent.click(within(region).getByRole("button", { name: "Dismiss notification" }));
    expect(region).toBeEmptyDOMElement();
  });

  test(`at most ${MAX_TOASTS} at once, oldest first out`, () => {
    render(
      <ToastProvider>
        <ToastButtons />
      </ToastProvider>,
    );
    for (const name of ["a", "b", "c", "d"]) fireEvent.click(screen.getByRole("button", { name: `toast ${name}` }));
    const region = screen.getByTestId("toast-region");
    expect(within(region).queryByText("Toast a")).toBeNull();
    expect(within(region).getByText("Toast d")).toBeInTheDocument();
    expect(region.children).toHaveLength(MAX_TOASTS);
  });

  test("outside a provider, useToast is inert instead of throwing", () => {
    render(<ToastButtons />);
    expect(() => fireEvent.click(screen.getByRole("button", { name: "success" }))).not.toThrow();
    expect(screen.queryByText("Saved")).toBeNull();
  });
});

import { useState } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import DeadmanButton from "./DeadmanButton";

function setup(props: { disabled?: boolean } = {}) {
  const onHoldStart = vi.fn();
  const onHoldEnd = vi.fn();
  const view = render(
    <DeadmanButton onHoldStart={onHoldStart} onHoldEnd={onHoldEnd} active={false} {...props}>
      Hold to run
    </DeadmanButton>,
  );
  return { onHoldStart, onHoldEnd, button: screen.getByRole("button", { name: "Hold to run" }), ...view };
}

const primary = { isPrimary: true, button: 0, pointerId: 1 };

let visibility: DocumentVisibilityState = "visible";
function setVisibility(next: DocumentVisibilityState) {
  visibility = next;
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => visibility });
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
}

afterEach(() => {
  visibility = "visible";
  Reflect.deleteProperty(document, "visibilityState");
});

describe("DeadmanButton", () => {
  it("starts on a primary press and ends once on release, wherever it happens", () => {
    const { onHoldStart, onHoldEnd, button } = setup();
    fireEvent.pointerDown(button, primary);
    expect(onHoldStart).toHaveBeenCalledTimes(1);
    fireEvent.pointerUp(window);
    fireEvent.pointerUp(button);
    expect(onHoldEnd).toHaveBeenCalledTimes(1);
    expect(onHoldEnd).toHaveBeenCalledWith("deadman released");
  });

  it("ignores secondary buttons and does not start twice while held", () => {
    const { onHoldStart, button } = setup();
    fireEvent.pointerDown(button, { ...primary, button: 2 });
    expect(onHoldStart).not.toHaveBeenCalled();
    fireEvent.pointerDown(button, primary);
    fireEvent.pointerDown(button, primary);
    expect(onHoldStart).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["pointercancel", (button: HTMLElement) => fireEvent.pointerCancel(button), "deadman released"],
    ["lost pointer capture", (button: HTMLElement) => fireEvent.lostPointerCapture(button), "deadman released"],
    ["focus loss", (button: HTMLElement) => fireEvent.blur(button), "deadman lost focus"],
    ["window blur", () => fireEvent.blur(window), "window lost focus"],
  ])("ends on %s", (_name, trigger, reason) => {
    const { onHoldEnd, button } = setup();
    fireEvent.pointerDown(button, primary);
    trigger(button);
    expect(onHoldEnd).toHaveBeenCalledWith(reason);
  });

  it("ends when the tab is hidden", () => {
    const { onHoldEnd, button } = setup();
    fireEvent.pointerDown(button, primary);
    setVisibility("hidden");
    expect(onHoldEnd).toHaveBeenCalledWith("tab hidden");
  });

  it("holds with Space or Enter, ignoring key repeat", () => {
    const { onHoldStart, onHoldEnd, button } = setup();
    fireEvent.keyDown(button, { key: " " });
    fireEvent.keyDown(button, { key: " ", repeat: true });
    expect(onHoldStart).toHaveBeenCalledTimes(1);
    fireEvent.keyUp(button, { key: " " });
    expect(onHoldEnd).toHaveBeenCalledWith("deadman released");
    fireEvent.keyDown(button, { key: "Enter" });
    expect(onHoldStart).toHaveBeenCalledTimes(2);
    fireEvent.keyDown(button, { key: "a" });
    expect(onHoldStart).toHaveBeenCalledTimes(2);
  });

  it("ends a hold when it becomes disabled or unmounts, and never starts while disabled", () => {
    function Toggle({ onHoldEnd }: { onHoldEnd: (reason: string) => void }) {
      const [disabled, setDisabled] = useState(false);
      return (
        <>
          <DeadmanButton onHoldStart={() => undefined} onHoldEnd={onHoldEnd} active={false} disabled={disabled}>
            Hold
          </DeadmanButton>
          <button type="button" onClick={() => setDisabled(true)}>
            disable
          </button>
        </>
      );
    }
    const onHoldEnd = vi.fn();
    render(<Toggle onHoldEnd={onHoldEnd} />);
    fireEvent.pointerDown(screen.getByRole("button", { name: "Hold" }), primary);
    fireEvent.click(screen.getByRole("button", { name: "disable" }));
    expect(onHoldEnd).toHaveBeenCalledWith("control disabled");

    const second = setup();
    fireEvent.pointerDown(second.button, primary);
    second.unmount();
    expect(second.onHoldEnd).toHaveBeenCalledWith("control closed");

    const disabled = setup({ disabled: true });
    fireEvent.pointerDown(disabled.button, primary);
    fireEvent.keyDown(disabled.button, { key: " " });
    expect(disabled.onHoldStart).not.toHaveBeenCalled();
  });

  it("keeps touch scrolling and the context menu off the control", () => {
    const { button } = setup();
    expect(button.className).toContain("touch-none");
    expect(fireEvent.contextMenu(button)).toBe(false);
  });
});

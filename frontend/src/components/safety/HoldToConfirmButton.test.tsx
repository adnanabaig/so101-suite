import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { advance, fakeShellTimers } from "../../test/harness";
import HoldToConfirmButton from "./HoldToConfirmButton";

beforeEach(() => {
  fakeShellTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

function setup(props: { locked?: boolean; busy?: boolean } = {}) {
  const onConfirm = vi.fn();
  render(<HoldToConfirmButton label="Reset fault" onConfirm={onConfirm} holdMs={1000} {...props} />);
  return { onConfirm, button: screen.getByRole("button", { name: "Reset fault" }) };
}

describe("HoldToConfirmButton", () => {
  test("a full pointer hold confirms exactly once", async () => {
    const { onConfirm, button } = setup();
    expect(button).toHaveAccessibleDescription("Press and hold for 1 second to confirm.");
    fireEvent.pointerDown(button);
    expect(button).toHaveAttribute("data-holding", "true");
    await advance(999);
    expect(onConfirm).not.toHaveBeenCalled();
    await advance(1);
    expect(onConfirm).toHaveBeenCalledTimes(1);
    fireEvent.pointerUp(button);
    await advance(2000);
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  test("the fill is the time left to hold, so it keeps moving under reduced motion", () => {
    const { button } = setup();
    const fill = button.querySelector<HTMLElement>("[data-essential-motion]");
    expect(fill).not.toBeNull();
    expect(fill).toHaveAttribute("aria-hidden", "true");
    expect(fill?.style.transform).toBe("scaleX(0)");
    fireEvent.pointerDown(button);
    expect(fill?.style.transform).toBe("scaleX(1)");
    expect(fill?.style.transition).toBe("transform 1000ms linear");
  });

  test("a click never confirms", async () => {
    const { onConfirm, button } = setup();
    fireEvent.click(button);
    await advance(2000);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  test("Space and Enter hold it from the keyboard; auto-repeat does not restart the hold", async () => {
    const { onConfirm, button } = setup();
    fireEvent.keyDown(button, { key: "Enter" });
    await advance(600);
    fireEvent.keyDown(button, { key: "Enter", repeat: true });
    await advance(400);
    expect(onConfirm).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(button, { key: " " });
    await advance(500);
    fireEvent.keyUp(button, { key: " " });
    await advance(1000);
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(button).toHaveAccessibleDescription("Released too early. Press and hold for 1 second.");
  });

  test("losing focus, cancelling or losing pointer capture aborts the hold", async () => {
    const { onConfirm, button } = setup();
    for (const abort of [
      () => fireEvent.blur(button),
      () => fireEvent.pointerCancel(button),
      () => fireEvent.lostPointerCapture(button),
    ]) {
      fireEvent.pointerDown(button);
      await advance(500);
      abort();
      await advance(1000);
    }
    expect(onConfirm).not.toHaveBeenCalled();
  });

  test("a secondary mouse button does not start a hold", async () => {
    const { onConfirm, button } = setup();
    fireEvent.pointerDown(button, { pointerType: "mouse", button: 2 });
    await advance(1500);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  test("locked or busy refuses to start and says why", async () => {
    const { onConfirm, button } = setup({ locked: true });
    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(button).toHaveAccessibleDescription("Unavailable.");
    fireEvent.pointerDown(button);
    await advance(1500);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  test("a lock that arrives mid-hold aborts it", async () => {
    const onConfirm = vi.fn();
    function Host() {
      const [locked, setLocked] = useState(false);
      return (
        <>
          <HoldToConfirmButton label="Reset" onConfirm={onConfirm} holdMs={1000} locked={locked} />
          <button type="button" onClick={() => setLocked(true)}>
            lock
          </button>
        </>
      );
    }
    render(<Host />);
    const button = screen.getByRole("button", { name: "Reset" });
    fireEvent.pointerDown(button);
    await advance(500);
    fireEvent.click(screen.getByRole("button", { name: "lock" }));
    await advance(1000);
    expect(onConfirm).not.toHaveBeenCalled();
    expect(button).not.toHaveAttribute("data-holding");
  });
});

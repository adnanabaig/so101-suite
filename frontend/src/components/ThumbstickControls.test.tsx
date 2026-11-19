import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import ThumbstickControls from "./ThumbstickControls";
import { CENTERED_STICKS } from "../lib/thumbsticks";

test("inactive sticks cannot produce input; active keys center on release and blur", () => {
  const changed = vi.fn();
  const view = render(<ThumbstickControls enabled={false} onAxesChange={changed} />);
  fireEvent.keyDown(window, { key: "w" });
  expect(changed).toHaveBeenLastCalledWith(CENTERED_STICKS);
  view.rerender(<ThumbstickControls enabled onAxesChange={changed} />);
  fireEvent.keyDown(window, { key: "w" });
  fireEvent.keyDown(window, { key: "ArrowRight" });
  expect(changed).toHaveBeenLastCalledWith({ ...CENTERED_STICKS, leftY: 1, rightX: 1 });
  fireEvent.keyUp(window, { key: "w" });
  expect(changed).toHaveBeenLastCalledWith({ ...CENTERED_STICKS, rightX: 1 });
  fireEvent.blur(window);
  expect(changed).toHaveBeenLastCalledWith(CENTERED_STICKS);
});

test("typing in a form and modified shortcuts do not move the sticks", () => {
  const changed = vi.fn();
  render(<><input aria-label="task" /><ThumbstickControls enabled onAxesChange={changed} /></>);
  fireEvent.keyDown(screen.getByRole("textbox"), { key: "w" });
  fireEvent.keyDown(window, { key: "w", ctrlKey: true });
  expect(changed).toHaveBeenLastCalledWith(CENTERED_STICKS);
});

import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { CHECKLIST_DISMISSED_KEY, STOPS_ACKNOWLEDGED_KEY, usePersistentFlag } from "./onboarding";

function Flag({ name, storageKey }: { name: string; storageKey: string }) {
  const [value, setValue] = usePersistentFlag(storageKey);
  return (
    <button type="button" onClick={() => setValue(!value)}>
      {name}: {value ? "yes" : "no"}
    </button>
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("usePersistentFlag", () => {
  test("keys are versioned and distinct", () => {
    expect(CHECKLIST_DISMISSED_KEY).toBe("so101.onboarding.dismissed.v1");
    expect(STOPS_ACKNOWLEDGED_KEY).toBe("so101.onboarding.stops.v1");
  });

  test("an answer persists across remounts and every reader sees it at once", () => {
    const { unmount } = render(
      <>
        <Flag name="a" storageKey={CHECKLIST_DISMISSED_KEY} />
        <Flag name="b" storageKey={CHECKLIST_DISMISSED_KEY} />
        <Flag name="stops" storageKey={STOPS_ACKNOWLEDGED_KEY} />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "a: no" }));
    expect(screen.getByRole("button", { name: "b: yes" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "stops: no" })).toBeInTheDocument();
    expect(window.localStorage.getItem(CHECKLIST_DISMISSED_KEY)).toBe("1");
    unmount();

    render(<Flag name="again" storageKey={CHECKLIST_DISMISSED_KEY} />);
    expect(screen.getByRole("button", { name: "again: yes" })).toBeInTheDocument();
    // Turning it off removes the key instead of storing "0".
    fireEvent.click(screen.getByRole("button", { name: "again: yes" }));
    expect(window.localStorage.getItem(CHECKLIST_DISMISSED_KEY)).toBeNull();
  });

  test("a change made in another tab shows up here", () => {
    render(<Flag name="a" storageKey={STOPS_ACKNOWLEDGED_KEY} />);
    act(() => {
      window.localStorage.setItem(STOPS_ACKNOWLEDGED_KEY, "1");
      window.dispatchEvent(new StorageEvent("storage", { key: STOPS_ACKNOWLEDGED_KEY }));
    });
    expect(screen.getByRole("button", { name: "a: yes" })).toBeInTheDocument();
  });

  test("with storage blocked, the answer still holds for this page load", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    render(<Flag name="a" storageKey={STOPS_ACKNOWLEDGED_KEY} />);
    fireEvent.click(screen.getByRole("button", { name: "a: no" }));
    expect(screen.getByRole("button", { name: "a: yes" })).toBeInTheDocument();
  });
});

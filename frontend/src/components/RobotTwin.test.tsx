import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { TwinSceneProps } from "./twin/TwinScene";

// The WebGL scene cannot run in jsdom. The stub records what the shell asks
// of it: presence, frame loop and reduced motion.
const scene = vi.hoisted(() => ({ error: null as Error | null, renders: 0 }));
const reload = vi.hoisted(() => ({ count: 0 }));

vi.mock("./twin/TwinScene", () => ({
  default: (props: TwinSceneProps) => {
    if (scene.error) throw scene.error;
    scene.renders += 1;
    return (
      <div
        data-testid="twin-scene"
        data-presence={props.presence}
        data-frameloop={props.frameloop}
        data-reduced-motion={String(props.reducedMotion)}
      />
    );
  },
}));

vi.mock("./ErrorBoundary", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./ErrorBoundary")>()),
  reloadPage: () => {
    reload.count += 1;
  },
}));

import RobotTwin from "./RobotTwin";

const FULL_SAMPLE = {
  "shoulder_pan.pos": 12.5,
  "shoulder_lift.pos": -30,
  "elbow_flex.pos": 45,
  "wrist_flex.pos": 10,
  "wrist_roll.pos": -5,
  "gripper.pos": 80,
};

function setHidden(hidden: boolean) {
  if (hidden) {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
  } else {
    Reflect.deleteProperty(document, "visibilityState");
  }
  act(() => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
}

beforeEach(() => {
  scene.error = null;
  scene.renders = 0;
  reload.count = 0;
});

afterEach(() => {
  Reflect.deleteProperty(document, "visibilityState");
  vi.unstubAllGlobals();
});

describe("RobotTwin never shows missing data as a pose (F14)", () => {
  test("without a sample the arm is a labelled placeholder", async () => {
    const view = render(<RobotTwin mode="idle" />);
    expect(view.container.firstElementChild).toHaveClass("isolate");
    const twin = await screen.findByTestId("twin-scene");
    expect(twin).toHaveAttribute("data-presence", "none");
    expect(screen.getByText("No live joint sample")).toBeInTheDocument();
    expect(screen.getByText(/does not show where the real arm is/)).toBeInTheDocument();
    expect(screen.getByText("Kinematic twin · no sample")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAccessibleName(
      "3D view of the arm: no live joint sample. The arm shown is a placeholder.",
    );
  });

  test("an idle frame with empty observation and action shows no pose and no command target", async () => {
    render(<RobotTwin observation={{}} action={{}} mode="idle" />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-presence", "none");
    expect(screen.queryByText(/command target/i)).toBeNull();
    expect(screen.queryByText(/0\.0/)).toBeNull();
  });

  test("non-finite joint values count as missing", async () => {
    render(<RobotTwin observation={{ "shoulder_pan.pos": Number.NaN }} mode="teleop" />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-presence", "none");
  });

  test("a fresh sample is drawn as live and says nothing extra", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} mode="teleop" />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-presence", "live");
    expect(screen.getByText("Kinematic twin · teleop")).toBeInTheDocument();
    expect(screen.queryByText("No live joint sample")).toBeNull();
    expect(screen.queryByText("Showing the last known pose")).toBeNull();
  });

  test("a stale sample is the last known pose, never live", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} action={FULL_SAMPLE} mode="teleop" stale />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-presence", "stale");
    expect(screen.getByText("Showing the last known pose")).toBeInTheDocument();
    expect(screen.getByText("Kinematic twin · last known")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Last command target" })).toBeInTheDocument();
  });

  test("an idle robot's last read is neutral: not live, not a warning (F8)", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} mode="idle" idle />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-presence", "idle");
    expect(screen.getByText("Kinematic twin · last read")).toBeInTheDocument();
    expect(screen.queryByText("Showing the last known pose")).toBeNull();
    expect(screen.queryByText("No live joint sample")).toBeNull();
  });

  test("stale outranks idle", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} mode="idle" idle stale />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-presence", "stale");
  });
});

describe("RobotTwin command target", () => {
  test("lists every joint in its units and dashes the missing ones", async () => {
    render(
      <RobotTwin
        observation={FULL_SAMPLE}
        action={{ "shoulder_pan.pos": 12.5, gripper: 80 }}
        mode="teleop"
      />,
    );
    await screen.findByTestId("twin-scene");
    const panel = screen.getByRole("region", { name: "Command target" });
    const rows = within(panel).getAllByRole("definition").map((cell) => cell.textContent);
    expect(within(panel).getAllByRole("term").map((cell) => cell.textContent)).toEqual([
      "Shoulder pan",
      "Shoulder lift",
      "Elbow flex",
      "Wrist flex",
      "Wrist roll",
      "Gripper (0-100)",
    ]);
    expect(rows).toEqual(["12.5°", "—", "—", "—", "—", "80.0"]);
  });

  test("showTarget={false} hides it", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} action={FULL_SAMPLE} showTarget={false} />);
    await screen.findByTestId("twin-scene");
    expect(screen.queryByRole("region", { name: /command target/i })).toBeNull();
  });
});

describe("RobotTwin rendering cost (F15)", () => {
  test("renders on demand while visible, and not at all in a hidden tab", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} />);
    const twin = await screen.findByTestId("twin-scene");
    expect(twin).toHaveAttribute("data-frameloop", "demand");

    setHidden(true);
    expect(screen.getByTestId("twin-scene")).toHaveAttribute("data-frameloop", "never");

    setHidden(false);
    expect(screen.getByTestId("twin-scene")).toHaveAttribute("data-frameloop", "demand");
  });

  test("a twin that mounts in a hidden tab loads nothing until the tab is shown", async () => {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
    render(<RobotTwin observation={FULL_SAMPLE} />);
    expect(screen.getByText("The 3D view starts when it is visible.")).toBeInTheDocument();
    await act(async () => {});
    expect(screen.queryByTestId("twin-scene")).toBeNull();
    expect(scene.renders).toBe(0);

    setHidden(false);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-frameloop", "demand");
  });

  test("scrolled off screen it stops rendering but keeps the scene", async () => {
    const observers: { callback: IntersectionObserverCallback }[] = [];
    class FakeIntersectionObserver {
      constructor(public callback: IntersectionObserverCallback) {
        observers.push(this);
      }
      observe() {}
      unobserve() {}
      disconnect() {}
      takeRecords() {
        return [];
      }
    }
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    render(<RobotTwin observation={FULL_SAMPLE} />);
    await screen.findByTestId("twin-scene");
    const observer = observers[observers.length - 1];
    const report = (isIntersecting: boolean) =>
      act(() => {
        observer?.callback(
          [{ isIntersecting } as IntersectionObserverEntry],
          observer as unknown as IntersectionObserver,
        );
      });

    report(false);
    expect(screen.getByTestId("twin-scene")).toHaveAttribute("data-frameloop", "never");
    report(true);
    expect(screen.getByTestId("twin-scene")).toHaveAttribute("data-frameloop", "demand");
  });

  test("honours prefers-reduced-motion", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query === "(prefers-reduced-motion: reduce)",
      media: query,
      onchange: null,
      addEventListener() {},
      removeEventListener() {},
      addListener() {},
      removeListener() {},
      dispatchEvent: () => false,
    }));
    render(<RobotTwin observation={FULL_SAMPLE} />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-reduced-motion", "true");
  });

  test("motion is allowed by default", async () => {
    render(<RobotTwin observation={FULL_SAMPLE} />);
    expect(await screen.findByTestId("twin-scene")).toHaveAttribute("data-reduced-motion", "false");
  });
});

describe("RobotTwin fallbacks", () => {
  // React reports each caught render error to the console and jsdom reports
  // it as an uncaught window error; both are expected here.
  const swallowWindowError = (event: ErrorEvent) => event.preventDefault();
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    window.addEventListener("error", swallowWindowError);
  });
  afterEach(() => {
    window.removeEventListener("error", swallowWindowError);
  });

  test("no WebGL: says so, and the overlays still work", async () => {
    scene.error = new Error("Error creating WebGL context.");
    render(<RobotTwin observation={FULL_SAMPLE} action={FULL_SAMPLE} mode="teleop" />);
    expect(await screen.findByText("3D rendering unavailable")).toBeInTheDocument();
    expect(screen.getByText(/could not create a WebGL scene/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reload the UI" })).toBeNull();
    expect(screen.getByRole("region", { name: "Command target" })).toBeInTheDocument();
  });

  test("a chunk that cannot be downloaded offers a reload", async () => {
    scene.error = new TypeError(
      "Failed to fetch dynamically imported module: http://robot.local/assets/TwinScene-abc.js",
    );
    render(<RobotTwin observation={FULL_SAMPLE} />);
    expect(await screen.findByText("The 3D view could not be downloaded")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reload the UI" }));
    expect(reload.count).toBe(1);
  });
});

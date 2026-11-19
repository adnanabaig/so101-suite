import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { installFakeBackend, installFakeWebSocket, settle, Shell, type FakeBackend } from "../test/harness";
import { isChunkLoadError, RootErrorBoundary } from "./ErrorBoundary";

let backend: FakeBackend;
let failure: Error | null;

function Page() {
  if (failure) throw failure;
  return <p>page ok</p>;
}

// React reports each caught render error to the console and jsdom reports it
// as an uncaught window error; both are expected here, so keep output readable.
const swallowWindowError = (event: ErrorEvent) => event.preventDefault();

beforeEach(() => {
  installFakeWebSocket();
  backend = installFakeBackend();
  failure = null;
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  window.addEventListener("error", swallowWindowError);
});

afterEach(() => {
  window.removeEventListener("error", swallowWindowError);
  vi.unstubAllGlobals();
});

describe("isChunkLoadError", () => {
  test("recognises the chunk-load messages of each browser and Vite", () => {
    const messages = [
      "Failed to fetch dynamically imported module: http://robot.local/assets/Models-1a2b.js",
      "Importing a module script failed.",
      "error loading dynamically imported module: http://robot.local/assets/Eval.js",
      "Unable to preload CSS for /assets/Simulation-9f.css",
      "Loading chunk 42 failed.",
    ];
    for (const message of messages) expect(isChunkLoadError(new Error(message))).toBe(true);
    const named = new Error("whatever");
    named.name = "ChunkLoadError";
    expect(isChunkLoadError(named)).toBe(true);
  });

  test("ordinary errors and non-errors are not chunk errors", () => {
    expect(isChunkLoadError(new TypeError("Cannot read properties of undefined"))).toBe(false);
    expect(isChunkLoadError("Failed to fetch dynamically imported module")).toBe(false);
    expect(isChunkLoadError(null)).toBe(false);
  });
});

describe("route error boundary (F5)", () => {
  test("a crashed page is replaced by a recovery panel while E-stop keeps working", async () => {
    failure = new Error("page exploded");
    render(
      <Shell>
        <Page />
      </Shell>,
    );
    await settle();
    const panel = screen.getByText("This page stopped working").closest("[role=alert]");
    expect(panel).not.toBeNull();
    expect(panel).toHaveTextContent("page exploded");

    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    expect(backend.posts()).toContain("/api/robot/estop");

    failure = null;
    fireEvent.click(within(panel as HTMLElement).getByRole("button", { name: "Try again" }));
    expect(screen.getByText("page ok")).toBeInTheDocument();
  });

  test("an unloadable chunk asks for a reload instead of a retry", async () => {
    failure = new Error("Failed to fetch dynamically imported module: /assets/Models-1a2b.js");
    render(
      <Shell>
        <Page />
      </Shell>,
    );
    await settle();
    const panel = screen.getByText("This page could not be loaded").closest("[role=alert]") as HTMLElement;
    expect(within(panel).getByRole("button", { name: "Reload the UI" })).toBeInTheDocument();
    expect(within(panel).queryByRole("button", { name: "Try again" })).toBeNull();
    expect(panel).toHaveTextContent(/outdated copy/);
  });

  test("navigating to another page clears the error", async () => {
    failure = new Error("page exploded");
    render(
      <Shell path="/teleop">
        <Page />
      </Shell>,
    );
    await settle();
    expect(screen.getByText("This page stopped working")).toBeInTheDocument();
    failure = null;
    fireEvent.click(screen.getByRole("link", { name: "Learn" }));
    await settle();
    expect(screen.getByText("page ok")).toBeInTheDocument();
  });

  test("a vite:preloadError shows the update prompt without reloading", async () => {
    render(
      <Shell>
        <Page />
      </Shell>,
    );
    await settle();
    expect(screen.queryByRole("button", { name: "Reload to update" })).toBeNull();
    act(() => {
      window.dispatchEvent(new Event("vite:preloadError"));
    });
    expect(screen.getByRole("button", { name: "Reload to update" })).toBeInTheDocument();
    expect(screen.getByText("page ok")).toBeInTheDocument();
  });

  test("switching to another view in the same workspace clears the error", async () => {
    failure = new Error("diagnostics exploded");
    render(<Shell path="/system?section=diagnostics"><Page /></Shell>);
    await settle();
    expect(screen.getByText("This page stopped working")).toBeInTheDocument();
    const stop = screen.getByRole("button", { name: "E-stop" });
    failure = null;
    fireEvent.keyDown(document.body, { key: "g" });
    fireEvent.keyDown(document.body, { key: "c" });
    await settle();
    expect(screen.getByText("page ok")).toBeInTheDocument();
    expect(document.title).toBe("System · Calibration · SO-101");
    expect(screen.getByRole("button", { name: "E-stop" })).toBe(stop);
  });
});

describe("root error boundary", () => {
  test("the crash screen still sends E-stop by button and by Escape", async () => {
    failure = new Error("provider exploded");
    render(
      <RootErrorBoundary>
        <Page />
      </RootErrorBoundary>,
    );
    expect(screen.getByRole("heading", { name: "The operator UI crashed" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "E-stop" }));
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    expect(screen.getByText(/E-stop sent/)).toBeInTheDocument();

    fireEvent.keyDown(document.body, { key: "Escape" });
    fireEvent.keyDown(document.body, { key: "Escape", repeat: true });
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(2);
  });
});

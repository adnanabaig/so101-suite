import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { installFakeBackend, installFakeWebSocket, settle, Shell, type FakeBackend } from "../test/harness";
import { routeLabel } from "./navigation";
import { useToast } from "./ui/Toast";
import BackendBadge from "./safety/BackendBadge";

let backend: FakeBackend;

beforeEach(() => {
  installFakeWebSocket();
  backend = installFakeBackend();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function renderShell(path = "/teleop") {
  render(
    <Shell path={path}>
      <p>page</p>
    </Shell>,
  );
  await settle();
}

describe("Layout navigation drawer (F22, F27)", () => {
  test("three workspace links replace categories and the redundant footer", async () => {
    backend.statusExtra = { stop_kinds: ["estop", "halt", "collapse"] };
    await renderShell("/operate");
    expect(screen.getByRole("main")).toHaveAccessibleName("Operate");
    const navigation = screen.getByRole("navigation", { name: "Main" });
    const links = within(navigation).getAllByRole("link");
    expect(links.map((link) => link.textContent)).toEqual(["Operate", "Learn", "System"]);
    expect(links[0]).toHaveAttribute("href", "/operate");
    expect(links[0]).toHaveAttribute("aria-current", "page");
    expect(navigation.parentElement).toHaveClass("w-20");
    expect(screen.getByRole("main")).toHaveClass("lg:overflow-hidden", "lg:min-h-0");
    for (const link of links) {
      expect(link).toHaveAttribute("title", link.textContent);
      expect(link.querySelector("span")).toHaveClass("sr-only");
    }
    expect(within(navigation).queryByRole("link", { name: "Teleop & capture" })).not.toBeInTheDocument();
    expect(screen.queryByText("Safety-first control plane")).not.toBeInTheDocument();
    expect(within(navigation).getByText("SO101")).toBeInTheDocument();
    expect(screen.queryByText("Robotics workstation")).not.toBeInTheDocument();
    expect(screen.queryByText("Live workspace")).not.toBeInTheDocument();
    expect(screen.queryByTestId("backend-badge")).not.toBeInTheDocument();
    expect(screen.queryByTestId("connection-indicator")).not.toBeInTheDocument();
    const stops = screen.getByRole("banner", { name: "Global robot stops" });
    expect(stops).toHaveClass("fixed", "right-3", "top-3");
    expect(within(stops).getAllByRole("button").map((button) => button.getAttribute("aria-label"))).toEqual(["Halt", "E-stop"]);
  });

  test("opening the drawer makes the page inert but never the stops", async () => {
    await renderShell();
    const menu = screen.getByRole("button", { name: "Open navigation" });
    fireEvent.click(menu);
    const drawer = screen.getByTestId("nav-drawer");
    const main = document.getElementById("main-content");
    expect(main).toHaveAttribute("inert");

    const estop = screen.getByRole("button", { name: "E-stop" });
    expect(estop.closest("[inert]")).toBeNull();
    expect(drawer.contains(estop)).toBe(false);

    // Focus moves into the drawer, and back to the menu button on close.
    expect(document.activeElement).toBe(within(drawer).getAllByRole("link")[0]);
    fireEvent.click(screen.getByRole("button", { name: "Close navigation" }));
    expect(screen.queryByTestId("nav-drawer")).toBeNull();
    expect(main).not.toHaveAttribute("inert");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Open navigation" }));
  });

  test("Escape inside the drawer is E-stop and does not close the drawer", async () => {
    await renderShell();
    fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
    const drawer = screen.getByTestId("nav-drawer");
    fireEvent.keyDown(within(drawer).getAllByRole("link")[0], { key: "Escape" });
    await settle();
    expect(backend.callsTo("/api/robot/estop")).toHaveLength(1);
    expect(screen.getByTestId("nav-drawer")).toBeInTheDocument();
  });

  test("choosing a page closes the drawer, retitles the tab and focuses the page", async () => {
    await renderShell();
    expect(document.title).toBe("Data collection · SO-101");
    fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
    fireEvent.click(within(screen.getByTestId("nav-drawer")).getByRole("link", { name: "Learn" }));
    await settle();
    expect(screen.queryByTestId("nav-drawer")).toBeNull();
    expect(document.title).toBe("Learn · Datasets · SO-101");
    const main = document.getElementById("main-content");
    expect(document.activeElement).toBe(main);
    expect(main).toHaveAccessibleName("Learn · Datasets");
  });

  test("the skip link moves focus to the page", async () => {
    await renderShell();
    fireEvent.click(screen.getByRole("link", { name: "Skip to content" }));
    expect(document.activeElement).toBe(document.getElementById("main-content"));
  });
});

describe("backend badge when explicitly mounted in a page", () => {
  async function renderBadge() {
    render(<Shell><BackendBadge /></Shell>);
    await settle();
  }
  test.each([
    ["real", "REAL", "Backend: Real arm: commands move the physical motors."],
    ["sim", "SIM", "Backend: Simulation (MuJoCo): commands move a simulated arm, not the physical one."],
    ["mock", "MOCK", "Backend: Mock backend: no motors move."],
  ])("backend %s shows %s", async (kind, text, spoken) => {
    backend.statusExtra = { backend: kind, mock: kind === "mock" };
    await renderBadge();
    const badge = within(screen.getByRole("main")).getByTestId("backend-badge");
    expect(badge).toHaveAttribute("data-backend", kind);
    expect(badge).toHaveTextContent(text);
    expect(badge).toHaveTextContent(spoken);
    // Only the real arm gets the normal, solid style.
    expect(badge.className.includes("border-dashed")).toBe(kind !== "real");
  });

  test("an older server without backend: mock reads MOCK, otherwise a real arm is assumed", async () => {
    backend.statusExtra = { mock: false };
    await renderBadge();
    const badge = screen.getByTestId("backend-badge");
    expect(badge).toHaveAttribute("data-backend", "real");
    expect(badge).toHaveTextContent("REAL");
    expect(badge).toHaveAttribute("title", "Real arm assumed: this server does not say whether it is simulated");
  });

  test("an older mock server reads MOCK", async () => {
    await renderBadge();
    expect(screen.getByTestId("backend-badge")).toHaveAttribute("data-backend", "mock");
  });
});

describe("Layout toasts", () => {
  test("pages can raise a toast; it lands in the shell's polite region, outside the page", async () => {
    function Saver() {
      const toast = useToast();
      return (
        <button type="button" onClick={() => toast.show("Saved the thing", { tone: "success" })}>
          Save
        </button>
      );
    }
    render(
      <Shell path="/teleop">
        <Saver />
      </Shell>,
    );
    await settle();
    const region = screen.getByTestId("toast-region");
    expect(region).toHaveAttribute("aria-live", "polite");
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(within(region).getByText("Saved the thing")).toBeInTheDocument();
    expect(document.getElementById("main-content")?.contains(region)).toBe(false);
  });
});

describe("routeLabel", () => {
  test("names known routes and falls back for others", () => {
    expect(routeLabel("/")).toBe("Overview");
    expect(routeLabel("/teleop")).toBe("Data collection");
    expect(routeLabel("/agent")).toBe("Agent lab");
    expect(routeLabel("/nope")).toBe("Page not found");
  });
});

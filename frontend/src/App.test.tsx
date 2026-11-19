import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App";
import { WORKSPACES } from "./components/navigation";
import { TelemetryProvider } from "./hooks/useTelemetry";
import { installFakeBackend, installFakeWebSocket, settle } from "./test/harness";

vi.mock("./pages/TeleopRecord", () => ({ default: ({ tools }: { tools?: ReactNode }) => <><h1>Collection view</h1>{tools}</> }));
vi.mock("./pages/Dashboard", () => ({ default: () => <h1>Overview view</h1> }));
vi.mock("./pages/Simulation", () => ({ default: () => <h2>Simulation view</h2> }));
vi.mock("./pages/Connect", () => ({ default: () => <h2>Connect view</h2> }));
vi.mock("./pages/Datasets", () => ({ default: () => <h1>Datasets view</h1> }));
vi.mock("./pages/Training", () => ({ default: () => <h1>Training view</h1> }));
vi.mock("./pages/Models", () => ({ default: () => <h1>Models view</h1> }));
vi.mock("./pages/Eval", () => ({ default: () => <h1>Evaluation view</h1> }));
vi.mock("./pages/Diagnostics", () => ({ default: () => <h1>Diagnostics view</h1> }));
vi.mock("./pages/Calibration", () => ({ default: () => <h1>Calibration view</h1> }));
vi.mock("./pages/Settings", () => ({ default: () => <h1>Settings view</h1> }));
vi.mock("./pages/AgentLab", () => ({ default: () => <h1>Agent view</h1> }));

beforeEach(() => {
  installFakeWebSocket();
  installFakeBackend();
});
afterEach(() => vi.unstubAllGlobals());

function Location() {
  const location = useLocation();
  return <output data-testid="address">{location.pathname + location.search + location.hash}</output>;
}

async function open(path: string) {
  render(<MemoryRouter initialEntries={[path]}><TelemetryProvider><App /><Location /></TelemetryProvider></MemoryRouter>);
  await screen.findByRole("heading", { level: 1 });
  await settle();
}

test.each(WORKSPACES.flatMap((workspace) => workspace.sections.map((section) => ({ workspace, section }))))(
  "legacy $section.legacy redirects to $workspace.path while preserving arm/joint/query and anchor",
  async ({ workspace, section }) => {
    await open(`${section.legacy}?arm=leader&joint=2&section=wrong#details`);
    expect(screen.getByTestId("address")).toHaveTextContent(`${workspace.path}?arm=leader&joint=2&section=${section.id}#details`);
    const links = within(screen.getByRole("navigation", { name: "Main" })).getAllByRole("link");
    expect(links).toHaveLength(3);
    expect(links.find((link) => link.getAttribute("aria-current") === "page")).toHaveTextContent(workspace.label);
    expect(screen.getByRole("main")).toHaveAccessibleName(workspace.path === "/operate" ? "Operate" : `${workspace.label} · ${section.label}`);
    expect(screen.getByRole("button", { name: "E-stop" })).toBeInTheDocument();
  },
);

test("Operate keeps collection mounted, has no tabs, and opens specialist tools only on demand", async () => {
  await open("/operate");
  const stop = screen.getByRole("button", { name: "E-stop" });
  expect(screen.getByRole("heading", { name: "Collection view" })).toBeInTheDocument();
  expect(screen.queryByRole("navigation", { name: "Operate views" })).not.toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Simulation view" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Simulation tools" }));
  await screen.findByRole("heading", { name: "Simulation view" });
  expect(screen.getByRole("heading", { name: "Collection view" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "E-stop" })).toBe(stop);
  expect(document.title).toBe("Operate · SO-101");
  fireEvent.click(screen.getByRole("button", { name: "Close tools" }));
  await settle();
  expect(screen.queryByRole("heading", { name: "Simulation view" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("link", { name: "Learn" }));
  await screen.findByRole("heading", { name: "Datasets view" });
  expect(screen.queryByRole("heading", { name: "Simulation view" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "E-stop" })).toBe(stop);
});

test("an unknown section falls back to its workspace default without mounting extra tools", async () => {
  await open("/system?section=not-a-tool");
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  expect(screen.getByRole("heading", { name: "Diagnostics view" })).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Calibration view" })).not.toBeInTheDocument();
});

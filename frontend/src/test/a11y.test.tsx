import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import App from "../App";
import { WORKSPACES, NOT_FOUND_LABEL, routeLabel } from "../components/navigation";
import { TelemetryProvider } from "../hooks/useTelemetry";
import { installFakeBackend, installFakeWebSocket, jsonResponse, settle } from "./harness";

// Structural accessibility checks on every page, as each first renders against
// a server that answers only the shell's routes. jsdom has no layout or CSS,
// so contrast and focus styling are checked in a11yStyles.test.ts instead.

// WebGL and MediaPipe have their own tests; jsdom has neither.
vi.mock("../components/RobotTwin", () => ({ default: () => <div data-testid="robot-twin" /> }));
vi.mock("../lib/handTracking", () => ({
  startBrowserHandTracking: () => Promise.resolve({ stop: () => undefined }),
}));

/** Roles whose accessible name is required: a control nobody can name is a control nobody can find. */
const NAMED_ROLES = [
  "button",
  "link",
  "textbox",
  "searchbox",
  "combobox",
  "listbox",
  "spinbutton",
  "checkbox",
  "radio",
  "slider",
  "switch",
  "tab",
  "menuitem",
  "progressbar",
  "dialog",
  "alertdialog",
] as const;

const ID_REFERENCES = ["aria-labelledby", "aria-describedby", "aria-controls", "aria-owns", "aria-activedescendant"];

/** Elements whose text is inside a landmark or announces itself. */
const CONTAINERS = [
  "header",
  "nav",
  "main",
  "aside",
  "footer",
  "section[aria-label]",
  "section[aria-labelledby]",
  ...["banner", "navigation", "main", "complementary", "contentinfo", "region", "search", "form"].map(
    (role) => `[role=${role}]`,
  ),
  ...["alert", "status", "log", "dialog", "alertdialog"].map((role) => `[role=${role}]`),
  '[aria-live="polite"]',
  '[aria-live="assertive"]',
].join(", ");

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';


const ROUTES = [...WORKSPACES.flatMap((workspace) => workspace.sections.map((section) => `${workspace.path}?section=${section.id}`)), "/no-such-page"];

beforeEach(() => {
  installFakeWebSocket();
  const backend = installFakeBackend();
  backend.route("GET", "/api/capabilities", () =>
    jsonResponse({
      system_version: "0.2.0",
      mock: true,
      capabilities: [
        { id: "control", label: "Safe manual command API", available: true, maturity: "stable", detail: "API contract only." },
      ],
    }),
  );
  backend.route("GET", "/api/events", () => jsonResponse({ events: [] }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function renderRoute(path: string): Promise<HTMLElement> {
  render(
    <MemoryRouter initialEntries={[path]}>
      <TelemetryProvider>
        <App />
      </TelemetryProvider>
    </MemoryRouter>,
  );
  // Pages are lazy; the first import of each can take a while under Vitest.
  await screen.findByRole("heading", { level: 1 }, { timeout: 15_000 });
  // Let the page's first requests answer and render.
  await settle(10);
  return screen.getByRole("main");
}

function headingLevel(element: Element): number {
  const level = element.getAttribute("aria-level");
  if (level !== null) return Number(level);
  const tag = /^H([1-6])$/.exec(element.tagName);
  return tag ? Number(tag[1]) : 2;
}

/** Visible text that no landmark or live region contains. The skip link is the one exception. */
function textOutsideLandmarks(): string[] {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const outside: string[] = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent?.trim();
    const parent = node.parentElement;
    if (!text || !parent) continue;
    if (parent.closest(CONTAINERS) || parent.closest('[aria-hidden="true"]') || parent.closest('a[href="#main-content"]')) {
      continue;
    }
    outside.push(text);
  }
  return outside;
}

function duplicates(values: string[]): string[] {
  return [...new Set(values.filter((value, index) => values.indexOf(value) !== index))];
}

describe.each(ROUTES)("accessibility of %s", (path) => {
  test(
    "landmarks, headings, names and references",
    async () => {
      const main = await renderRoute(path);

      // One of each landmark; the whole sidebar is the "Main" navigation.
      expect(screen.getAllByRole("main")).toHaveLength(1);
      expect(screen.getAllByRole("banner")).toHaveLength(1);
      expect(screen.getAllByRole("navigation", { name: "Main" })).toHaveLength(1);
      expect(screen.queryAllByRole("complementary")).toEqual([]);
      const destination = new URL(path, "http://localhost");
      expect(main).toHaveAccessibleName(routeLabel(destination.pathname, destination.search));
      expect(textOutsideLandmarks()).toEqual([]);

      // One h1, in main, and no level skipped below it.
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
      expect(within(main).getAllByRole("heading", { level: 1 })).toHaveLength(1);
      const levels = [...main.querySelectorAll("h1, h2, h3, h4, h5, h6, [role=heading]")].map(headingLevel);
      expect(levels[0]).toBe(1);
      const skips = levels.flatMap((level, index) =>
        index > 0 && level > levels[index - 1] + 1 ? [`h${levels[index - 1]} -> h${level}`] : [],
      );
      expect(skips).toEqual([]);

      for (const role of NAMED_ROLES) {
        const unnamed = screen
          .queryAllByRole(role, { name: (name) => name.trim() === "" })
          .map((element) => element.outerHTML.slice(0, 160));
        expect(unnamed, `${role} without an accessible name`).toEqual([]);
      }

      const positiveTabIndex = [...document.querySelectorAll("[tabindex]")].filter(
        (element) => Number(element.getAttribute("tabindex")) > 0,
      );
      expect(positiveTabIndex).toEqual([]);
      expect(document.querySelectorAll("img:not([alt])")).toHaveLength(0);

      // An icon is decoration (hidden) or says something (named); never read as an unnamed graphic.
      const loudIcons = [...document.querySelectorAll("svg.lucide")].filter(
        (icon) => !icon.closest('[aria-hidden="true"]') && !icon.hasAttribute("aria-label") && !icon.hasAttribute("aria-labelledby"),
      );
      expect(loudIcons.map((icon) => icon.getAttribute("class"))).toEqual([]);

      // Nothing focusable hides from assistive technology.
      const hiddenFocusable = [...document.querySelectorAll('[aria-hidden="true"]')].flatMap((element) => [
        ...(element.matches(FOCUSABLE) ? [element] : []),
        ...element.querySelectorAll(FOCUSABLE),
      ]);
      expect(hiddenFocusable.map((element) => element.outerHTML.slice(0, 160))).toEqual([]);

      const ids = [...document.querySelectorAll("[id]")].map((element) => element.id);
      expect(duplicates(ids)).toEqual([]);

      // Every id reference points at something. A collapsed control may name
      // a popup that is not rendered yet.
      const broken: string[] = [];
      for (const attribute of ID_REFERENCES) {
        for (const element of document.querySelectorAll(`[${attribute}]`)) {
          if (attribute === "aria-controls" && element.getAttribute("aria-expanded") === "false") continue;
          for (const id of (element.getAttribute(attribute) ?? "").split(/\s+/).filter(Boolean)) {
            if (!document.getElementById(id)) broken.push(`${attribute}="${id}"`);
          }
        }
      }
      for (const label of document.querySelectorAll("label[for]")) {
        const target = label.getAttribute("for") ?? "";
        if (!document.getElementById(target)) broken.push(`for="${target}"`);
      }
      expect(broken).toEqual([]);
    },
    30_000,
  );
});

test("an unknown address is a page with a heading and a way back", async () => {
  const main = await renderRoute("/no-such-page");
  expect(within(main).getByRole("heading", { level: 1, name: NOT_FOUND_LABEL })).toBeInTheDocument();
  expect(within(main).getByRole("link", { name: "Go to Operate" })).toHaveAttribute("href", "/operate");
  expect(document.title).toContain(NOT_FOUND_LABEL);
}, 30_000);

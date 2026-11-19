import {
  Boxes,
  BrainCircuit,
  Camera,
  ChartNoAxesCombined,
  CircleGauge,
  Database,
  FlaskConical,
  Radio,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Wrench,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
  badge?: string;
  /**
   * Second key of the "G then key" navigation shortcut. Never "h": H is
   * Halt, and the safety handler takes it before navigation sees it.
   */
  shortcut: string;
}

const LEGACY_GROUPS: ReadonlyArray<{ label: string; items: readonly NavItem[] }> = [
  {
    label: "Operate",
    items: [
      { to: "/teleop", label: "Data collection", icon: Camera, shortcut: "t" },
      { to: "/", label: "Overview", icon: CircleGauge, end: true, shortcut: "o" },
      { to: "/simulation", label: "Digital twin", icon: Boxes, badge: "3D", shortcut: "w" },
      { to: "/connect", label: "Quest & hands", icon: Radio, badge: "XR", shortcut: "q" },
    ],
  },
  {
    label: "Learn",
    items: [
      { to: "/datasets", label: "Datasets", icon: Database, shortcut: "d" },
      { to: "/training", label: "Training", icon: ChartNoAxesCombined, shortcut: "r" },
      { to: "/models", label: "Models", icon: BrainCircuit, shortcut: "m" },
      { to: "/eval", label: "Evaluation", icon: FlaskConical, shortcut: "e" },
    ],
  },
  {
    label: "System",
    items: [
      { to: "/calibration", label: "Calibration", icon: SlidersHorizontal, shortcut: "c" },
      { to: "/diagnostics", label: "Diagnostics", icon: Wrench, shortcut: "x" },
      { to: "/agent", label: "Agent lab", icon: Sparkles, badge: "LAB", shortcut: "a" },
      { to: "/settings", label: "Settings", icon: Settings, shortcut: "s" },
    ],
  },
];

export const NAV_ITEMS: readonly NavItem[] = [
  { to: "/operate", label: "Operate", icon: Camera, shortcut: "t" },
  { to: "/learn", label: "Learn", icon: Database, shortcut: "d" },
  { to: "/system", label: "System", icon: Wrench, shortcut: "s" },
];

export const WORKSPACES = [
  { path: "/operate", label: "Operate", sections: [
    { id: "collection", label: "Data collection", legacy: "/teleop" },
    { id: "overview", label: "Overview", legacy: "/" },
    { id: "simulation", label: "Digital twin", legacy: "/simulation" },
    { id: "connect", label: "Quest & hands", legacy: "/connect" },
  ] },
  { path: "/learn", label: "Learn", sections: [
    { id: "datasets", label: "Datasets", legacy: "/datasets" },
    { id: "training", label: "Training", legacy: "/training" },
    { id: "models", label: "Models", legacy: "/models" },
    { id: "evaluation", label: "Evaluation", legacy: "/eval" },
  ] },
  { path: "/system", label: "System", sections: [
    { id: "diagnostics", label: "Diagnostics", legacy: "/diagnostics" },
    { id: "calibration", label: "Calibration", legacy: "/calibration" },
    { id: "settings", label: "Settings", legacy: "/settings" },
    { id: "agent", label: "Agent lab", legacy: "/agent" },
  ] },
] as const;

export function workspaceDestination(pathname: string, search = "", hash = ""): string | undefined {
  for (const workspace of WORKSPACES) {
    const section = workspace.sections.find((candidate) => candidate.legacy === pathname);
    if (section) {
      const params = new URLSearchParams(search);
      params.set("section", section.id);
      return `${workspace.path}?${params.toString()}${hash}`;
    }
  }
  return undefined;
}

export const SHORTCUT_ITEMS: readonly NavItem[] = LEGACY_GROUPS.flatMap((group) =>
  group.items.map((item) => ({ ...item, to: workspaceDestination(item.to)! })),
);

/** The name of any address that is not a page. */
export const NOT_FOUND_LABEL = "Page not found";

/** The page name for a path, as the header, title and main landmark show it. */
export function routeLabel(pathname: string, search = ""): string {
  const workspace = WORKSPACES.find((candidate) => candidate.path === pathname);
  if (workspace) {
    if (workspace.path === "/operate") return "Operate";
    const section = workspace.sections.find((candidate) => candidate.id === new URLSearchParams(search).get("section")) ?? workspace.sections[0];
    return `${workspace.label} · ${section.label}`;
  }
  return LEGACY_GROUPS.flatMap((group) => group.items).find((item) => item.to === pathname)?.label ?? NOT_FOUND_LABEL;
}

/** The page a "G then key" sequence opens, if any. */
export function navItemForShortcut(key: string): NavItem | undefined {
  const wanted = key.toLowerCase();
  return SHORTCUT_ITEMS.find((item) => item.shortcut === wanted);
}

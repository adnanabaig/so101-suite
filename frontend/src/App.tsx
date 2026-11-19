import { lazy, Suspense } from "react";
import { Link, Navigate, Route, Routes, useLocation, useSearchParams } from "react-router-dom";
import Layout from "./components/Layout";
import { NOT_FOUND_LABEL, WORKSPACES, workspaceDestination } from "./components/navigation";
import { PageHeader } from "./components/Workstation";
import { Dialog } from "./components/ui/Dialog";
import Button from "./components/Button";

const Calibration = lazy(() => import("./pages/Calibration"));
const TeleopRecord = lazy(() => import("./pages/TeleopRecord"));
const Datasets = lazy(() => import("./pages/Datasets"));
const Training = lazy(() => import("./pages/Training"));
const Models = lazy(() => import("./pages/Models"));
const Eval = lazy(() => import("./pages/Eval"));
const Settings = lazy(() => import("./pages/Settings"));
const Simulation = lazy(() => import("./pages/Simulation"));
const Connect = lazy(() => import("./pages/Connect"));
const Diagnostics = lazy(() => import("./pages/Diagnostics"));
const AgentLab = lazy(() => import("./pages/AgentLab"));

const SECTION_PAGES = {
  collection: TeleopRecord,
  overview: TeleopRecord,
  simulation: Simulation,
  connect: Connect,
  datasets: Datasets,
  training: Training,
  models: Models,
  evaluation: Eval,
  diagnostics: Diagnostics,
  calibration: Calibration,
  settings: Settings,
  agent: AgentLab,
};

export function Workspace({ workspace }: { workspace: (typeof WORKSPACES)[number] }) {
  return workspace.path === "/operate" ? <OperateWorkspace /> : <SectionWorkspace workspace={workspace} />;
}

function OperateWorkspace() {
  const [params, setParams] = useSearchParams();
  const requested = params.get("section");
  const specialist = requested === "simulation" || requested === "connect" ? requested : null;
  const select = (section: string) => { const next = new URLSearchParams(params); next.set("section", section); setParams(next); };
  return <div className="lg:h-full lg:min-h-0">
    <TeleopRecord readinessInitiallyOpen={requested === "overview"} tools={<div className="flex flex-wrap gap-2">
      <Button variant="secondary" onClick={() => select("simulation")}>Simulation tools</Button>
      <Button variant="secondary" onClick={() => select("connect")}>Quest &amp; hands</Button>
    </div>} />
    <Dialog open={specialist !== null} title={specialist === "simulation" ? "Simulation tools" : "Quest & hands"} size="wide"
      actions={<Button variant="secondary" onClick={() => select("collection")}>Close tools</Button>}>
      <Suspense fallback={<RouteFallback />}>
        {specialist === "simulation" && <Simulation embedded />}
        {specialist === "connect" && <Connect embedded />}
      </Suspense>
    </Dialog>
  </div>;
}

function SectionWorkspace({ workspace }: { workspace: (typeof WORKSPACES)[number] }) {
  const [params] = useSearchParams();
  const selected = workspace.sections.find((section) => section.id === params.get("section")) ?? workspace.sections[0];
  const Page = SECTION_PAGES[selected.id];
  return (
    <div className="space-y-5">
      <nav aria-label={`${workspace.label} views`} className="flex flex-wrap gap-1 rounded-xl border border-white/5 bg-slate-950/70 p-1.5">
        {workspace.sections.map((section) => {
          const next = new URLSearchParams(params);
          next.set("section", section.id);
          return (
            <Link
              key={section.id}
              to={`${workspace.path}?${next.toString()}`}
              aria-current={selected.id === section.id ? "page" : undefined}
              className={`rounded-lg px-3 py-2 text-sm font-medium focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-400 ${selected.id === section.id ? "bg-cyan-400/10 text-cyan-200" : "text-slate-400 hover:bg-white/5 hover:text-white"}`}
            >
              {section.label}
            </Link>
          );
        })}
      </nav>
      <Suspense fallback={<RouteFallback />}>
        <Page key={selected.id} />
      </Suspense>
    </div>
  );
}

function LegacyRedirect() {
  const location = useLocation();
  return <Navigate replace to={workspaceDestination(location.pathname, location.search, location.hash) ?? "/operate"} />;
}

function RouteFallback() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <div role="status" className="flex flex-col items-center gap-3 text-muted">
        <span
          aria-hidden="true"
          className="h-7 w-7 motion-safe:animate-spin rounded-full border-2 border-slate-800 border-t-cyan-400"
        />
        <span className="text-xs font-semibold uppercase tracking-[0.18em]">
          Loading workspace
        </span>
      </div>
    </div>
  );
}

export function NotFound() {
  return (
    <div className="space-y-4">
      <PageHeader title={NOT_FOUND_LABEL} description="Nothing in the operator UI lives at this address." />
      <Link to="/operate" className="inline-flex text-sm font-semibold text-cyan-300 hover:text-cyan-200">
        Go to Operate
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          {WORKSPACES.map((workspace) => <Route key={workspace.path} path={workspace.path} element={<Workspace workspace={workspace} />} />)}
          {WORKSPACES.flatMap((workspace) => workspace.sections.map((section) => <Route key={section.legacy} path={section.legacy} element={<LegacyRedirect />} />))}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
